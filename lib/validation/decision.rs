use crate::state::Error;
use crate::state::decisions::DecisionId;
use crate::types::{Address, ClaimDecisionPayload, FilledTransaction};
use sneed::RoTxn;

use std::collections::BTreeMap;

use super::{
    DecisionValidationInterface, MarketValidator, add_listing_fee,
    validate_listing_outputs,
};

pub struct DecisionValidator;

impl DecisionValidator {
    pub fn parse_decision_id_from_hex(
        decision_id_hex: &str,
    ) -> Result<DecisionId, Error> {
        Ok(DecisionId::from_hex(decision_id_hex)?)
    }

    /// The non-standard claims of one transaction fit, together with those
    /// already claimed, in each period's non-standard cap
    pub fn validate_nonstandard_cap<T>(
        decisions_db: &T,
        rotxn: &RoTxn,
        decision_ids: impl IntoIterator<Item = DecisionId>,
    ) -> Result<(), Error>
    where
        T: DecisionValidationInterface,
    {
        use crate::math::decisions::NONSTANDARD_SLOTS_PER_PERIOD_MAX;

        let mut claims_per_period = std::collections::BTreeMap::new();
        for decision_id in decision_ids {
            if !decision_id.is_standard() {
                *claims_per_period
                    .entry(decision_id.period_index())
                    .or_insert(0u64) += 1;
            }
        }
        for (period, claims) in claims_per_period {
            let claimed = decisions_db
                .get_nonstandard_claimed_count_in_period(rotxn, period)?;
            if claimed.saturating_add(claims) > NONSTANDARD_SLOTS_PER_PERIOD_MAX
            {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "{claims} non-standard claims would exceed the \
                         {NONSTANDARD_SLOTS_PER_PERIOD_MAX} cap of period \
                         {period}, which holds {claimed}"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Validate a single claim payload's type constraints and slot
    /// availability. Returns the listing fees that this payload owes, by
    /// decision period.
    pub fn validate_claim_payload<T>(
        decisions_db: &T,
        rotxn: &RoTxn,
        payload: &ClaimDecisionPayload,
        market_maker_address: Address,
        override_height: Option<u32>,
    ) -> Result<BTreeMap<u32, u64>, Error>
    where
        T: DecisionValidationInterface,
    {
        use crate::state::decisions::DecisionType;

        match &payload.decision_type {
            DecisionType::Binary | DecisionType::Scaled { .. } => {
                if payload.decisions.len() != 1 {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "{:?} claim must have exactly 1 \
                             decision, got {}",
                            payload.decision_type,
                            payload.decisions.len()
                        ),
                    });
                }
                if payload.decisions[0].option_labels.is_some() {
                    return Err(Error::InvalidTransaction {
                        reason: "option_labels must not be set \
                             for binary/scaled decisions"
                            .to_string(),
                    });
                }
            }
            DecisionType::Category { options } => {
                if payload.decisions.len() != 1 {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "Category claim must have exactly \
                             1 decision entry, got {}",
                            payload.decisions.len()
                        ),
                    });
                }
                if options.len() < 2 {
                    return Err(Error::InvalidTransaction {
                        reason: "Category claim requires at \
                             least 2 options"
                            .to_string(),
                    });
                }
                let entry = &payload.decisions[0];
                let entry_labels = entry.option_labels.as_ref().ok_or(
                    Error::InvalidTransaction {
                        reason: "Category claim entry must \
                             have option_labels"
                            .to_string(),
                    },
                )?;
                if entry_labels.len() < 2 {
                    return Err(Error::InvalidTransaction {
                        reason: "Category claim entry must \
                             have at least 2 option_labels"
                            .to_string(),
                    });
                }
                if entry_labels != options {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "option_labels ({}) must match \
                             decision_type options ({})",
                            entry_labels.len(),
                            options.len()
                        ),
                    });
                }
            }
        }

        let current_ts = decisions_db
            .try_get_mainchain_timestamp(rotxn)?
            .unwrap_or(0);

        let current_height = override_height
            .or_else(|| decisions_db.try_get_height(rotxn).ok().flatten());

        let genesis_ts =
            decisions_db.try_get_genesis_timestamp(rotxn)?.unwrap_or(0);

        for entry in &payload.decisions {
            let decision_id = DecisionId::from_bytes(entry.decision_id_bytes)?;
            let entry_type = payload.decision_type.clone();

            let decision = crate::state::decisions::Decision::new(
                market_maker_address.0,
                entry_type,
                entry.header.clone(),
                entry.description.clone(),
                entry.option_0_label.clone(),
                entry.option_1_label.clone(),
                entry.tags.clone().unwrap_or_default(),
            )?;

            decisions_db
                .validate_decision_claim(
                    rotxn,
                    decision_id,
                    &decision,
                    current_ts,
                    current_height,
                    genesis_ts,
                )
                .map_err(|e| match e {
                    Error::DecisionNotAvailable {
                        decision_id: _,
                        reason,
                    } => Error::InvalidDecisionId { reason },
                    Error::DecisionAlreadyClaimed { decision_id: _ } => {
                        Error::InvalidDecisionId {
                            reason: format!(
                                "Decision {} already claimed",
                                const_hex::encode(entry.decision_id_bytes)
                            ),
                        }
                    }
                    other => other,
                })?;
        }

        let mut listing_fees = BTreeMap::new();
        for entry in &payload.decisions {
            let id = DecisionId::from_bytes(entry.decision_id_bytes)?;
            let fee = decisions_db.fee_for_decision_id(rotxn, id)?;
            add_listing_fee(&mut listing_fees, id.period_index(), fee)?;
        }

        Ok(listing_fees)
    }

    pub fn validate_complete_decision_claim<T>(
        decisions_db: &T,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error>
    where
        T: DecisionValidationInterface,
    {
        let payload =
            tx.claim_decision()
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: "Not a decision claim transaction".to_string(),
                })?;

        let market_maker_address = MarketValidator::validate_maker_auth(tx)?;

        let listing_fees = Self::validate_claim_payload(
            decisions_db,
            rotxn,
            payload,
            market_maker_address,
            override_height,
        )?;
        Self::validate_nonstandard_cap(
            decisions_db,
            rotxn,
            payload.decisions.iter().filter_map(|entry| {
                DecisionId::from_bytes(entry.decision_id_bytes).ok()
            }),
        )?;
        validate_listing_outputs(tx, &listing_fees)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Output, OutputContent};

    const PERIOD: u32 = 2;

    fn pool_output(period: u32, sats: u64) -> Output {
        Output {
            address:
                crate::state::markets::payouts::generate_listing_pool_address(
                    period,
                ),
            content: OutputContent::ListingFees {
                period,
                amount: bitcoin::Amount::from_sat(sats),
            },
        }
    }

    /// A non-standard claim in `PERIOD` with `outputs`, funded so that it
    /// pays `miner_sats` to the miner
    fn claim_tx(outputs: Vec<Output>, miner_sats: u64) -> FilledTransaction {
        use crate::types::{
            ClaimDecisionPayload, DecisionClaimEntry, GetValue as _, OutPoint,
            Transaction, TransactionData, Txid,
        };

        let paid: u64 = outputs
            .iter()
            .map(|output| output.get_value().to_sat())
            .sum();
        FilledTransaction {
            transaction: Transaction {
                inputs: vec![(
                    OutPoint::Regular {
                        txid: Txid([1; 32]),
                        vout: 0,
                    },
                    [0; 32],
                )]
                .into(),
                outputs: outputs.into(),
                data: Some(TransactionData::ClaimDecision(
                    ClaimDecisionPayload {
                        decision_type:
                            crate::state::decisions::DecisionType::Binary,
                        decisions: vec![DecisionClaimEntry {
                            decision_id_bytes: DecisionId::new(
                                false, PERIOD, 0,
                            )
                            .unwrap()
                            .as_bytes(),
                            header: "decision".to_string(),
                            description: String::new(),
                            option_0_label: None,
                            option_1_label: None,
                            option_labels: None,
                            tags: None,
                        }],
                    },
                )),
                ..Default::default()
            },
            spent_utxos: vec![Output {
                address: Address([9; 20]),
                content: OutputContent::Value(bitcoin::Amount::from_sat(
                    paid + miner_sats,
                )),
            }],
            actor_address: None,
        }
    }

    /// A state with minted genesis periods, and the listing fee of the
    /// non-standard claim in `PERIOD`
    fn claim_state() -> (tempfile::TempDir, sneed::Env, crate::state::State, u64)
    {
        let dir = tempfile::tempdir().unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024)
            .max_dbs(crate::state::State::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, dir.path()) }.unwrap();
        let state = crate::state::State::new(&env, None, true).unwrap();
        let mut rwtxn = env.write_txn().unwrap();
        state
            .put_timestamps_for_test(&mut rwtxn, 1000, 1000)
            .unwrap();
        state
            .decisions()
            .mint_genesis(&mut rwtxn, 1000, 0, 1000)
            .unwrap();
        rwtxn.commit().unwrap();
        let rotxn = env.read_txn().unwrap();
        let fee = state
            .decisions()
            .fee_for_decision_id(
                &rotxn,
                DecisionId::new(false, PERIOD, 0).unwrap(),
            )
            .unwrap();
        drop(rotxn);
        (dir, env, state, fee)
    }

    fn validate(
        env: &sneed::Env,
        state: &crate::state::State,
        tx: &FilledTransaction,
    ) -> Result<(), Error> {
        let rotxn = env.read_txn().unwrap();
        DecisionValidator::validate_complete_decision_claim(
            state.decisions(),
            &rotxn,
            tx,
            Some(1),
        )
    }

    /// The minimum miner fee of a claim with `outputs`
    fn floor(outputs: Vec<Output>) -> u64 {
        crate::validation::min_tx_fee(&claim_tx(outputs, 0).transaction, false)
            .unwrap()
            .to_sat()
    }

    /// Non-standard claims pay the tier-zero price, floored at the minimum
    #[test]
    fn nonstandard_claim_fee() {
        use crate::math::decisions::{
            NONSTANDARD_MIN_LISTING_FEE_SATS, nonstandard_listing_fee,
            slot_price,
        };

        assert_eq!(
            nonstandard_listing_fee(0),
            NONSTANDARD_MIN_LISTING_FEE_SATS
        );
        assert_eq!(nonstandard_listing_fee(100_000), slot_price(100_000, 0));
        let (_dir, _env, _state, fee) = claim_state();
        assert!(fee >= NONSTANDARD_MIN_LISTING_FEE_SATS);
    }

    /// A claim paying its listing fee to its period's pool and the minimum
    /// miner fee validates
    #[test]
    fn claim_pays_pool() {
        let (_dir, env, state, fee) = claim_state();
        let outputs = vec![pool_output(PERIOD, fee)];
        validate(&env, &state, &claim_tx(outputs.clone(), floor(outputs)))
            .unwrap();
    }

    /// A claim paying its listing fee only as miner fee is rejected
    #[test]
    fn claim_fee_only() {
        let (_dir, env, state, fee) = claim_state();
        let tx = claim_tx(Vec::new(), fee + floor(Vec::new()));
        assert!(validate(&env, &state, &tx).is_err());
    }

    /// A pool output one sat short of the listing fee is rejected
    #[test]
    fn claim_underpays() {
        let (_dir, env, state, fee) = claim_state();
        let outputs = vec![pool_output(PERIOD, fee - 1)];
        let tx = claim_tx(outputs.clone(), floor(outputs));
        assert!(validate(&env, &state, &tx).is_err());
    }

    /// A claim quoted before a price drop overpays its pool and validates
    #[test]
    fn claim_overpays() {
        let (_dir, env, state, fee) = claim_state();
        let outputs = vec![pool_output(PERIOD, fee + 1)];
        validate(&env, &state, &claim_tx(outputs.clone(), floor(outputs)))
            .unwrap();
    }

    /// Listing fees paid to another period's pool are rejected
    #[test]
    fn claim_wrong_period() {
        let (_dir, env, state, fee) = claim_state();
        let outputs = vec![pool_output(PERIOD + 1, fee)];
        let tx = claim_tx(outputs.clone(), floor(outputs));
        assert!(validate(&env, &state, &tx).is_err());
    }

    /// A listing fee output that does not pay the pool address is rejected
    #[test]
    fn claim_wrong_address() {
        let (_dir, env, state, fee) = claim_state();
        let mut output = pool_output(PERIOD, fee);
        output.address = Address([9; 20]);
        let outputs = vec![output];
        let tx = claim_tx(outputs.clone(), floor(outputs));
        assert!(validate(&env, &state, &tx).is_err());
    }

    /// The listing fee split across two outputs of one period is rejected
    #[test]
    fn claim_split_output() {
        let (_dir, env, state, fee) = claim_state();
        let outputs =
            vec![pool_output(PERIOD, fee - 1), pool_output(PERIOD, 1)];
        let tx = claim_tx(outputs.clone(), floor(outputs));
        assert!(validate(&env, &state, &tx).is_err());
    }

    #[test]
    fn parse_valid_hex_decision_id() {
        let hex = "00010001";
        let result = DecisionValidator::parse_decision_id_from_hex(hex);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_invalid_hex_decision_id() {
        assert!(
            DecisionValidator::parse_decision_id_from_hex("zzzzzzzz").is_err()
        );
    }

    #[test]
    fn parse_wrong_length_decision_id() {
        assert!(DecisionValidator::parse_decision_id_from_hex("01").is_err());
        assert!(
            DecisionValidator::parse_decision_id_from_hex("0100010001")
                .is_err()
        );
    }
}
