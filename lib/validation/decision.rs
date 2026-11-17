use crate::state::Error;
use crate::state::decisions::DecisionId;
use crate::types::{Address, ClaimDecisionPayload, FilledTransaction};
use sneed::RoTxn;

use super::{DecisionValidationInterface, MarketValidator};

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
    /// availability. Returns the total listing fee that this payload owes.
    /// Caller is responsible for verifying the enclosing tx pays at least
    /// the summed listing fees.
    pub fn validate_claim_payload<T>(
        decisions_db: &T,
        rotxn: &RoTxn,
        payload: &ClaimDecisionPayload,
        market_maker_address: Address,
        override_height: Option<u32>,
    ) -> Result<u64, Error>
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

        let mut total_listing_fee: u64 = 0;
        for entry in &payload.decisions {
            let id = DecisionId::from_bytes(entry.decision_id_bytes)?;
            let fee = decisions_db.fee_for_decision_id(rotxn, id)?;
            total_listing_fee =
                total_listing_fee.checked_add(fee).ok_or_else(|| {
                    Error::InvalidTransaction {
                        reason: "Listing fee overflow".to_string(),
                    }
                })?;
        }

        Ok(total_listing_fee)
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

        let total_listing_fee = Self::validate_claim_payload(
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

        if total_listing_fee > 0 {
            let tx_fee = crate::validation::tx_fee(tx)?;
            if tx_fee.to_sat() < total_listing_fee {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Insufficient listing fee: tx fee is \
                         {} sats but listing fee requires \
                         {} sats",
                        tx_fee.to_sat(),
                        total_listing_fee
                    ),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nonstandard_claim_tx(fee_sats: u64) -> FilledTransaction {
        use crate::types::{
            Address, ClaimDecisionPayload, DecisionClaimEntry, OutPoint,
            Output, OutputContent, Transaction, TransactionData, Txid,
        };

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
                data: Some(TransactionData::ClaimDecision(
                    ClaimDecisionPayload {
                        decision_type:
                            crate::state::decisions::DecisionType::Binary,
                        decisions: vec![DecisionClaimEntry {
                            decision_id_bytes: DecisionId::new(false, 2, 0)
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
                    fee_sats,
                )),
            }],
            actor_address: None,
        }
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
            .fee_for_decision_id(&rotxn, DecisionId::new(false, 2, 0).unwrap())
            .unwrap();
        assert!(fee >= NONSTANDARD_MIN_LISTING_FEE_SATS);

        let validate = |fee_sats| {
            DecisionValidator::validate_complete_decision_claim(
                state.decisions(),
                &rotxn,
                &nonstandard_claim_tx(fee_sats),
                Some(1),
            )
        };
        assert!(validate(fee - 1).is_err());
        validate(fee).unwrap();
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
