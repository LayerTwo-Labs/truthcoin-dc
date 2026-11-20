use std::collections::{BTreeMap, HashMap, HashSet};

use crate::state::Error;
use crate::state::decisions::{DecisionId, DecisionType};
use crate::state::markets::{DimensionSpec, MarketState};
use crate::types::{Address, DecisionClaimEntry, FilledTransaction};
use sneed::RoTxn;

use super::DecisionValidator;

pub struct MarketValidator;

impl MarketValidator {
    pub fn validate_maker_auth(
        tx: &FilledTransaction,
    ) -> Result<Address, Error> {
        if tx.transaction.inputs.is_empty() {
            return Err(Error::InvalidTransaction {
                reason: "Transaction must have at least one input".to_string(),
            });
        }

        if tx.spent_utxos.is_empty() {
            return Err(Error::InvalidTransaction {
                reason: "No spent UTXOs found".to_string(),
            });
        }

        let first_utxo = &tx.spent_utxos[0];
        let market_maker_address = first_utxo.address;

        Ok(market_maker_address)
    }

    pub fn validate_market_creation(
        state: &crate::state::State,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error> {
        let view = tx.as_market_creation().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a market creation transaction".to_string(),
            }
        })?;

        let market_maker_address = Self::validate_maker_auth(tx)?;

        let mut pending_claims: HashMap<DecisionId, &DecisionClaimEntry> =
            HashMap::new();
        let mut pending_types: HashMap<DecisionId, DecisionType> =
            HashMap::new();
        let mut listing_fees = BTreeMap::new();

        for payload in view.new_claims {
            for (period, fee) in DecisionValidator::validate_claim_payload(
                state,
                rotxn,
                payload,
                market_maker_address,
                override_height,
            )? {
                super::add_listing_fee(&mut listing_fees, period, fee)?;
            }

            for entry in &payload.decisions {
                let decision_id =
                    DecisionId::from_bytes(entry.decision_id_bytes)?;

                if pending_claims.contains_key(&decision_id) {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "Duplicate new claim for decision {}",
                            const_hex::encode(entry.decision_id_bytes)
                        ),
                    });
                }

                if state
                    .decisions()
                    .get_decision_entry(rotxn, decision_id)?
                    .is_some_and(|e| e.decision.is_some())
                {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "New claim for decision {} collides with \
                             already-claimed decision",
                            const_hex::encode(entry.decision_id_bytes)
                        ),
                    });
                }

                pending_claims.insert(decision_id, entry);
                pending_types
                    .insert(decision_id, payload.decision_type.clone());
            }
        }
        DecisionValidator::validate_nonstandard_cap(
            state,
            rotxn,
            pending_claims.keys().copied(),
        )?;

        let dimension_specs = view.dimension_specs;

        if dimension_specs.is_empty() {
            return Err(Error::InvalidTransaction {
                reason: "Market must have at least one dimension".to_string(),
            });
        }

        let mut seen_decisions = HashSet::new();
        let mut outcome_counts = Vec::with_capacity(dimension_specs.len());
        for spec in dimension_specs {
            let decision_id = match spec {
                DimensionSpec::Single(id) | DimensionSpec::Categorical(id) => {
                    id
                }
            };

            if !seen_decisions.insert(decision_id) {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Duplicate dimension for decision {decision_id:?}"
                    ),
                });
            }

            if let Some(ty) = pending_types.get(decision_id) {
                let mut outcome_count = 2;
                if let DimensionSpec::Categorical(_) = spec {
                    match ty {
                        DecisionType::Category { options } => {
                            outcome_count = options.len();
                            if options.len() < 2 {
                                return Err(Error::InvalidTransaction {
                                    reason: "Categorical dimensions \
                                         require at least 2 options"
                                        .to_string(),
                                });
                            }
                        }
                        _ => {
                            return Err(Error::InvalidTransaction {
                                reason: format!(
                                    "Categorical dimension references new \
                                     decision {decision_id:?} which is not a \
                                     Category type"
                                ),
                            });
                        }
                    }
                }
                outcome_counts.push(outcome_count);
                continue;
            }

            let entry = state
                .decisions()
                .get_decision_entry(rotxn, *decision_id)?
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: format!(
                        "Referenced decision {decision_id:?} does not exist"
                    ),
                })?;

            let decision =
                entry.decision.ok_or_else(|| Error::InvalidTransaction {
                    reason: format!(
                        "Referenced decision {decision_id:?} was never claimed"
                    ),
                })?;

            let mut outcome_count = 2;
            if let DimensionSpec::Categorical(_) = spec {
                if !decision.is_categorical() {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "Categorical dimension references decision {decision_id:?} which is not a Category type"
                        ),
                    });
                }
                let option_count = decision.option_count().unwrap_or(0);
                outcome_count = option_count;
                if option_count < 2 {
                    return Err(Error::InvalidTransaction {
                        reason:
                            "Categorical dimensions require at least 2 options"
                                .to_string(),
                    });
                }
            }
            outcome_counts.push(outcome_count);
        }

        let outcome_count =
            crate::math::markets::outcome_space_size(&outcome_counts)
                .filter(|count| *count <= crate::math::lmsr::MAX_OUTCOMES)
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: "Market outcome count exceeds the limit"
                        .to_string(),
                })?;

        if let Some(fee) = view.trading_fee
            && (!(0.0..=1.0).contains(&fee))
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Trading fee must be between 0.0 and 1.0, got {fee}"
                ),
            });
        }

        if let Some(difficulty) = view.tx_pow_difficulty
            && difficulty > 0
        {
            let config = crate::types::tx_pow::TxPowConfig {
                hash_selector: view.tx_pow_hash_selector.unwrap_or(0),
                ordering: view.tx_pow_ordering.unwrap_or(0),
                difficulty,
            };
            if !config.validate() {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "TX-PoW config invalid: difficulty must be \
                         0..={} and hash_selector must be non-zero \
                         when difficulty > 0",
                        crate::types::tx_pow::MAX_POW_DIFFICULTY
                    ),
                });
            }
        }

        use crate::state::markets::{
            compute_market_id, generate_market_treasury_address,
        };
        let expected_market_id = compute_market_id(
            view.title,
            view.description,
            &market_maker_address,
            dimension_specs,
        );
        let expected_market_id_bytes = *expected_market_id.as_bytes();
        let treasury_address =
            generate_market_treasury_address(&expected_market_id);

        let treasury_amount_sats = tx
            .transaction
            .outputs
            .iter()
            .find_map(|output| match &output.content {
                crate::types::OutputContent::MarketFunds {
                    market_id,
                    amount,
                    is_fee: false,
                } if market_id == &expected_market_id_bytes
                    && output.address == treasury_address =>
                {
                    Some(amount.to_sat())
                }
                _ => None,
            })
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!(
                    "CreateMarket tx must have MarketFunds (treasury) output with market_id {} to {treasury_address}",
                    const_hex::encode(expected_market_id_bytes)
                ),
            })?;

        if treasury_amount_sats == 0 {
            return Err(Error::InvalidTransaction {
                reason: "CreateMarket treasury output must be positive"
                    .to_string(),
            });
        }

        let storage_fee =
            crate::math::markets::market_storage_fee(outcome_count);
        super::validate_listing_outputs(tx, &listing_fees)?;
        super::validate_fee_floor(tx, bitcoin::Amount::from_sat(storage_fee))?;

        Ok(())
    }

    pub fn validate_trade(
        state: &crate::state::State,
        archive: &crate::archive::Archive,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::state::markets::MarketState;
        use crate::types::tx_pow::POW_BOUND_WINDOW_BLOCKS;

        let trade = tx.trade().ok_or_else(|| Error::InvalidTransaction {
            reason: "Not a trade transaction".to_string(),
        })?;

        // Market must exist
        let market = state
            .markets()
            .get_market(rotxn, &trade.market_id)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!("Market {:?} does not exist", trade.market_id),
            })?;

        // Market must be in Trading state
        if market.state() != MarketState::Trading {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Market not in trading state (current: {:?})",
                    market.state()
                ),
            });
        }

        for decision_id in &market.decision_ids {
            if state
                .decisions()
                .get_decision_current_state(rotxn, *decision_id)?
                .allows_reveal()
            {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Trading halted: decision {} is revealing votes",
                        decision_id.to_hex()
                    ),
                });
            }
        }

        let tip = state.try_get_tip(rotxn)?.ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "No chain tip when validating trade".to_string(),
            }
        })?;
        let tip_height = state.try_get_height(rotxn)?.ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "No chain height when validating trade".to_string(),
            }
        })?;
        let prev_block_height = archive
            .try_get_height(rotxn, trade.prev_block_hash)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!(
                    "Trade prev_block_hash {} does not reference a \
                     known block",
                    const_hex::encode(trade.prev_block_hash.0)
                ),
            })?;
        if trade.prev_block_hash != tip
            && !archive.is_descendant(rotxn, trade.prev_block_hash, tip)?
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Trade prev_block_hash {} is not on the active chain",
                    const_hex::encode(trade.prev_block_hash.0)
                ),
            });
        }
        let depth = tip_height.saturating_sub(prev_block_height);
        if depth > POW_BOUND_WINDOW_BLOCKS {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Trade prev_block_hash is {depth} blocks behind tip, \
                     exceeds window of {POW_BOUND_WINDOW_BLOCKS}"
                ),
            });
        }

        let tx_pow_config = market.tx_pow_config();
        if tx_pow_config.is_enabled() {
            let nonce = trade.tx_pow_nonce.ok_or_else(|| {
                Error::InvalidTransaction {
                    reason: "Trade requires TX-PoW nonce for this market"
                        .to_string(),
                }
            })?;

            let pow_data = crate::types::tx_pow::serialize_trade_for_pow(
                market.id.as_bytes(),
                &trade.deltas,
                &trade.trader,
                trade.max_cost,
                &trade.prev_block_hash,
            );

            if !tx_pow_config.verify(&pow_data, nonce) {
                return Err(Error::InvalidTransaction {
                    reason: "TX-PoW verification failed: insufficient \
                             proof-of-work"
                        .to_string(),
                });
            }
        }

        // Each leg must be a tradeable outcome. Abstain/invalid states
        // (voter-only) are not part of the tradeable outcome space and fall
        // outside `market.shares().len()`.
        crate::math::trading::check_deltas(
            &trade.deltas,
            market.shares().len(),
        )
        .map_err(|e| Error::InvalidTransaction {
            reason: format!("Invalid trade: {e}"),
        })?;

        if !tx.transaction.outputs.is_empty() {
            return Err(Error::InvalidTransaction {
                reason: "Trade transactions must have no outputs".to_string(),
            });
        }

        Self::validate_maker_auth(tx)?;

        let input_value = tx
            .get_value_in()
            .map_err(|_| Error::InvalidTransaction {
                reason: "Failed to compute input value".to_string(),
            })?
            .to_sat();

        let miner_fee =
            super::trade_miner_fee(&tx.transaction, tx.actor_address.is_some())
                .map_err(Error::BincodeSerialize)?
                .to_sat();
        let required_input = u64::try_from(trade.max_cost.max(0))
            .ok()
            .and_then(|max_cost| max_cost.checked_add(miner_fee))
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Trade max cost overflows".to_string(),
            })?;
        if input_value < required_input {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Input value {input_value} sats insufficient for max cost \
                     {} sats plus miner fee {miner_fee} sats",
                    trade.max_cost.max(0)
                ),
            });
        }

        if trade.sells() {
            let has_trader_input = tx
                .spent_utxos
                .iter()
                .any(|utxo| utxo.address == trade.trader);
            let has_actor_proof = tx.actor_address == Some(trade.trader);
            if !has_trader_input && !has_actor_proof {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Sell transaction has no proof of \
                         trader {} identity",
                        trade.trader
                    ),
                });
            }

            let account = state
                .markets()
                .get_user_share_account(rotxn, &trade.trader)?;
            for delta in trade.deltas.iter().filter(|delta| delta.shares < 0) {
                let owned_shares = account
                    .as_ref()
                    .and_then(|acc| {
                        acc.positions
                            .get(&(trade.market_id, delta.outcome_index))
                            .copied()
                    })
                    .unwrap_or(0);
                if owned_shares < -delta.shares {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "Insufficient shares: trying to sell {} but only \
                             own {owned_shares} for outcome {}",
                            -delta.shares, delta.outcome_index
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    pub fn validate_amplify_beta(
        state: &crate::state::State,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        let amplify =
            tx.amplify_beta().ok_or_else(|| Error::InvalidTransaction {
                reason: "Not an amplify_beta transaction".to_string(),
            })?;

        if amplify.amount == 0 {
            return Err(Error::InvalidTransaction {
                reason: "AmplifyBeta amount must be positive".to_string(),
            });
        }

        let market = state
            .markets()
            .get_market(rotxn, &amplify.market_id)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!(
                    "Market {:?} does not exist",
                    amplify.market_id
                ),
            })?;

        if market.state() != MarketState::Trading {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Market not in trading state (current: {:?})",
                    market.state()
                ),
            });
        }

        if market.creator_address != amplify.market_author {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "AmplifyBeta author {} does not match market creator {}",
                    amplify.market_author, market.creator_address
                ),
            });
        }

        let has_author_input = tx
            .spent_utxos
            .iter()
            .any(|utxo| utxo.address == amplify.market_author);
        let has_actor_proof = tx.actor_address == Some(amplify.market_author);
        if !has_author_input && !has_actor_proof {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "AmplifyBeta has no proof of market author {} identity",
                    amplify.market_author
                ),
            });
        }

        if !tx.transaction.outputs.is_empty() {
            return Err(Error::InvalidTransaction {
                reason: "AmplifyBeta transactions must have no outputs"
                    .to_string(),
            });
        }

        let input_value = tx
            .get_value_in()
            .map_err(|_| Error::InvalidTransaction {
                reason: "Failed to compute input value".to_string(),
            })?
            .to_sat();

        let miner_fee =
            super::trade_miner_fee(&tx.transaction, tx.actor_address.is_some())
                .map_err(Error::BincodeSerialize)?
                .to_sat();
        let required =
            amplify.amount.checked_add(miner_fee).ok_or_else(|| {
                Error::InvalidTransaction {
                    reason: "AmplifyBeta amount overflow".to_string(),
                }
            })?;

        if input_value < required {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Input value {input_value} sats insufficient for \
                     amplify amount {} + miner fee {miner_fee} sats",
                    amplify.amount
                ),
            });
        }

        Ok(())
    }

    pub fn validate_market_shares(
        shares: &ndarray::Array1<i64>,
    ) -> Result<(), Error> {
        for (idx, &share_qty) in shares.iter().enumerate() {
            if share_qty < 0 {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Share quantity at index {idx} must be non-negative, got {share_qty}"
                    ),
                });
            }
        }

        if shares.len() < 2 {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Market must have at least 2 outcomes, got {}",
                    shares.len()
                ),
            });
        }

        Ok(())
    }
}

pub struct MarketStateValidator;

impl MarketStateValidator {
    pub fn validate_market_state_transition(
        from_state: MarketState,
        to_state: MarketState,
    ) -> Result<(), Error> {
        if from_state.can_transition_to(to_state) {
            Ok(())
        } else {
            Err(Error::InvalidTransaction {
                reason: format!(
                    "Invalid market state transition \
                     from {from_state:?} to {to_state:?}"
                ),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        archive::Archive,
        state::{
            State,
            decisions::Decision,
            markets::{compute_market_id, generate_market_treasury_address},
        },
        types::{
            Hash, OutPoint, Output, OutputContent, Transaction,
            TransactionData, Txid,
        },
    };
    use ndarray::Array1;

    #[test]
    fn market_valid_shares() {
        let shares = Array1::from_vec(vec![100, 100]);
        assert!(MarketValidator::validate_market_shares(&shares).is_ok());
    }

    #[test]
    fn market_negative_shares() {
        let shares = Array1::from_vec(vec![100, -1]);
        assert!(MarketValidator::validate_market_shares(&shares).is_err());
    }

    #[test]
    fn market_too_few_outcomes() {
        let shares = Array1::from_vec(vec![100]);
        assert!(MarketValidator::validate_market_shares(&shares).is_err());
    }

    #[test]
    fn market_duplicate_dimension() {
        use crate::state::decisions::Decision;
        use crate::state::markets::compute_market_id;
        use crate::types::{
            OutPoint, Output, OutputContent, Transaction, TransactionData, Txid,
        };

        let dir = tempfile::tempdir().unwrap();
        let mut options = heed::EnvOpenOptions::new().read_txn_without_tls();
        options
            .map_size(16 * 1024 * 1024)
            .max_dbs(crate::state::State::NUM_DBS);
        let env = unsafe { sneed::Env::open(&options, dir.path()) }.unwrap();
        let state = crate::state::State::new(&env, None, true).unwrap();

        let maker = Address([1; 20]);
        let decision_id = DecisionId::new(false, 1, 0).unwrap();
        let decision = Decision::new(
            maker.0,
            DecisionType::Binary,
            "h".to_string(),
            String::new(),
            None,
            None,
            Vec::new(),
        )
        .unwrap();
        let mut rwtxn = env.write_txn().unwrap();
        state
            .decisions()
            .claim_decision(
                &mut rwtxn,
                decision_id,
                decision,
                Txid::default(),
                None,
            )
            .unwrap();
        rwtxn.commit().unwrap();

        let dimension_specs = vec![
            DimensionSpec::Single(decision_id),
            DimensionSpec::Single(decision_id),
        ];
        let market_id = compute_market_id("t", "", &maker, &dimension_specs);
        let tx = FilledTransaction {
            transaction: Transaction {
                inputs: vec![(
                    OutPoint::Regular {
                        txid: Txid::default(),
                        vout: 0,
                    },
                    [0; 32],
                )]
                .into(),
                proof: Default::default(),
                outputs: vec![Output {
                    address: maker,
                    content: OutputContent::MarketFunds {
                        market_id: *market_id.as_bytes(),
                        amount: bitcoin::Amount::from_sat(1000),
                        is_fee: false,
                    },
                }]
                .into(),
                data: Some(TransactionData::CreateMarket {
                    title: "t".to_string(),
                    description: String::new(),
                    dimension_specs,
                    new_claims: Vec::new(),
                    trading_fee: None,
                    tx_pow_hash_selector: None,
                    tx_pow_ordering: None,
                    tx_pow_difficulty: None,
                }),
            },
            spent_utxos: vec![Output {
                address: maker,
                content: OutputContent::Value(bitcoin::Amount::from_sat(2000)),
            }],
            actor_address: None,
        };

        let rotxn = env.read_txn().unwrap();
        assert!(
            MarketValidator::validate_market_creation(
                &state, &rotxn, &tx, None
            )
            .is_err()
        );
    }

    #[test]
    fn state_transition_valid() {
        assert!(
            MarketStateValidator::validate_market_state_transition(
                MarketState::Trading,
                MarketState::Settled
            )
            .is_ok()
        );
        assert!(
            MarketStateValidator::validate_market_state_transition(
                MarketState::Trading,
                MarketState::Cancelled
            )
            .is_ok()
        );
        assert!(
            MarketStateValidator::validate_market_state_transition(
                MarketState::Trading,
                MarketState::Invalid
            )
            .is_ok()
        );
        assert!(
            MarketStateValidator::validate_market_state_transition(
                MarketState::Trading,
                MarketState::Trading
            )
            .is_ok()
        );
    }

    #[test]
    fn state_transition_invalid() {
        assert!(
            MarketStateValidator::validate_market_state_transition(
                MarketState::Settled,
                MarketState::Trading
            )
            .is_err()
        );
        assert!(
            MarketStateValidator::validate_market_state_transition(
                MarketState::Cancelled,
                MarketState::Trading
            )
            .is_err()
        );
    }

    fn create_market_tx(
        maker: Address,
        dimension_specs: Vec<DimensionSpec>,
        treasury_address: Address,
    ) -> FilledTransaction {
        let market_id =
            compute_market_id("title", "", &maker, &dimension_specs);
        FilledTransaction {
            transaction: Transaction {
                inputs: vec![(
                    OutPoint::Regular {
                        txid: Txid(Hash::from([1; 32])),
                        vout: 0,
                    },
                    [0; 32],
                )]
                .into(),
                proof: Default::default(),
                outputs: vec![Output {
                    address: treasury_address,
                    content: OutputContent::MarketFunds {
                        market_id: *market_id.as_bytes(),
                        amount: bitcoin::Amount::from_sat(10_000),
                        is_fee: false,
                    },
                }]
                .into(),
                data: Some(TransactionData::CreateMarket {
                    title: "title".to_string(),
                    description: String::new(),
                    dimension_specs,
                    new_claims: vec![],
                    trading_fee: None,
                    tx_pow_hash_selector: None,
                    tx_pow_ordering: None,
                    tx_pow_difficulty: None,
                }),
            },
            spent_utxos: vec![Output {
                address: maker,
                content: OutputContent::Value(bitcoin::Amount::from_sat(
                    20_000,
                )),
            }],
            actor_address: None,
        }
    }

    /// The treasury output must pay the market's treasury address
    #[test]
    fn treasury_output_address() {
        let dir = tempfile::tempdir().unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024)
            .max_dbs(State::NUM_DBS + Archive::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, dir.path()) }.unwrap();
        let state = State::new(&env, None, true).unwrap();

        let maker = Address([7; 20]);
        let decision_id = DecisionId::new(false, 1, 0).unwrap();
        let mut rwtxn = env.write_txn().unwrap();
        state
            .decisions()
            .claim_decision(
                &mut rwtxn,
                decision_id,
                Decision {
                    market_maker_pubkey_hash: maker.0,
                    header: "decision".to_string(),
                    description: String::new(),
                    decision_type: DecisionType::Binary,
                    option_0_label: None,
                    option_1_label: None,
                    tags: Vec::new(),
                },
                Txid(Hash::from([2; 32])),
                None,
            )
            .unwrap();
        rwtxn.commit().unwrap();
        let rotxn = env.read_txn().unwrap();

        let specs = vec![DimensionSpec::Single(decision_id)];
        let treasury = generate_market_treasury_address(&compute_market_id(
            "title", "", &maker, &specs,
        ));
        let canonical = create_market_tx(maker, specs.clone(), treasury);
        MarketValidator::validate_market_creation(
            &state, &rotxn, &canonical, None,
        )
        .unwrap();

        let maker_owned = create_market_tx(maker, specs, maker);
        assert!(
            MarketValidator::validate_market_creation(
                &state,
                &rotxn,
                &maker_owned,
                None
            )
            .is_err()
        );
    }
    fn check_market_fee(decision_type: DecisionType, new_claim: bool) {
        use crate::types::{ClaimDecisionPayload, DecisionClaimEntry};

        let dir = tempfile::tempdir().unwrap();
        let mut options = heed::EnvOpenOptions::new().read_txn_without_tls();
        options.map_size(16 * 1024 * 1024).max_dbs(State::NUM_DBS);
        let env = unsafe { sneed::Env::open(&options, dir.path()) }.unwrap();
        let state = State::new(&env, None, true).unwrap();
        let maker = Address([7; 20]);
        let decision_id = DecisionId::new(true, 1, 0).unwrap();
        let labels = match &decision_type {
            DecisionType::Category { options } => Some(options.clone()),
            _ => None,
        };
        let spec = if labels.is_some() {
            DimensionSpec::Categorical(decision_id)
        } else {
            DimensionSpec::Single(decision_id)
        };
        let outcome_count = labels.as_ref().map_or(2, Vec::len);
        let mut rwtxn = env.write_txn().unwrap();
        state.decisions().mint_genesis(&mut rwtxn, 0, 0, 0).unwrap();
        let claim_fee = if new_claim {
            state
                .decisions()
                .fee_for_decision_id(&rwtxn, decision_id)
                .unwrap()
        } else {
            state
                .decisions()
                .claim_decision(
                    &mut rwtxn,
                    decision_id,
                    Decision::new(
                        maker.0,
                        decision_type.clone(),
                        "decision".into(),
                        String::new(),
                        None,
                        None,
                        Vec::new(),
                    )
                    .unwrap(),
                    Txid::default(),
                    Some(0),
                )
                .unwrap();
            0
        };
        rwtxn.commit().unwrap();
        let specs = vec![spec];
        let treasury = generate_market_treasury_address(&compute_market_id(
            "title", "", &maker, &specs,
        ));
        let mut tx = create_market_tx(maker, specs, treasury);
        if new_claim {
            let Some(TransactionData::CreateMarket { new_claims, .. }) =
                &mut tx.transaction.data
            else {
                unreachable!()
            };
            new_claims.push(ClaimDecisionPayload {
                decision_type,
                decisions: vec![DecisionClaimEntry {
                    decision_id_bytes: decision_id.as_bytes(),
                    header: "decision".into(),
                    description: String::new(),
                    option_0_label: None,
                    option_1_label: None,
                    option_labels: labels,
                    tags: None,
                }],
            });
        }
        let rotxn = env.read_txn().unwrap();
        pay_listing_fees(&state, &rotxn, &mut tx);
        let storage_fee =
            crate::math::markets::market_storage_fee(outcome_count);
        let fee = storage_fee
            + crate::validation::min_tx_fee(&tx.transaction, false)
                .unwrap()
                .to_sat();
        for paid in [fee, fee - 1] {
            tx.spent_utxos[0].content = OutputContent::Value(
                bitcoin::Amount::from_sat(10_000 + claim_fee + paid),
            );
            let result = MarketValidator::validate_market_creation(
                &state,
                &rotxn,
                &tx,
                Some(0),
            );
            assert_eq!(result.is_ok(), paid == fee, "paid {paid}: {result:?}");
        }
    }

    /// Add outputs paying the listing fees that a market creation's new
    /// claims owe into their pools, and fund them
    fn pay_listing_fees(
        state: &State,
        rotxn: &RoTxn,
        tx: &mut FilledTransaction,
    ) {
        use crate::{
            state::markets::payouts::generate_listing_pool_address,
            types::GetValue as _,
        };

        let Some(TransactionData::CreateMarket { new_claims, .. }) =
            &tx.transaction.data
        else {
            return;
        };
        let mut owed: BTreeMap<u32, u64> = BTreeMap::new();
        for entry in new_claims.iter().flat_map(|claim| &claim.decisions) {
            let id = DecisionId::from_bytes(entry.decision_id_bytes).unwrap();
            *owed.entry(id.period_index()).or_default() +=
                state.decisions().fee_for_decision_id(rotxn, id).unwrap();
        }
        tx.transaction
            .outputs
            .0
            .extend(owed.iter().map(|(&period, &sats)| Output {
                address: generate_listing_pool_address(period),
                content: OutputContent::ListingFees {
                    period,
                    amount: bitcoin::Amount::from_sat(sats),
                },
            }));
        let funded =
            tx.spent_utxos[0].get_value().to_sat() + owed.values().sum::<u64>();
        tx.spent_utxos[0].content =
            OutputContent::Value(bitcoin::Amount::from_sat(funded));
    }

    /// A binary market validates only when paying its storage fee and the
    /// minimum fee rate
    #[test]
    fn binary_storage_fee() {
        check_market_fee(DecisionType::Binary, false);
    }

    /// A categorical market's storage fee scales with its option count
    #[test]
    fn category_storage_fee() {
        check_market_fee(
            DecisionType::Category {
                options: vec!["a".into(), "b".into(), "c".into()],
            },
            false,
        );
    }

    /// The market fee includes the fee for a newly claimed decision
    #[test]
    fn market_claim_fee() {
        check_market_fee(DecisionType::Binary, true);
    }

    /// A newly claimed categorical decision's fee uses its option count
    #[test]
    fn category_claim_fee() {
        check_market_fee(
            DecisionType::Category {
                options: vec!["a".into(), "b".into(), "c".into()],
            },
            true,
        );
    }

    fn create_market_with_new_claim_tx(
        maker: Address,
        decision_id: DecisionId,
    ) -> FilledTransaction {
        use crate::types::ClaimDecisionPayload;

        let specs = vec![DimensionSpec::Single(decision_id)];
        let market_id = compute_market_id("title", "", &maker, &specs);
        let mut tx = create_market_tx(
            maker,
            specs,
            generate_market_treasury_address(&market_id),
        );
        if let Some(TransactionData::CreateMarket { new_claims, .. }) =
            &mut tx.transaction.data
        {
            new_claims.push(ClaimDecisionPayload {
                decision_type: DecisionType::Binary,
                decisions: vec![DecisionClaimEntry {
                    decision_id_bytes: decision_id.as_bytes(),
                    header: "decision".to_string(),
                    description: String::new(),
                    option_0_label: None,
                    option_1_label: None,
                    option_labels: None,
                    tags: None,
                }],
            });
        }
        tx
    }

    /// One market creation cannot claim more non-standard decisions than a
    /// period's cap holds, even with claims no dimension uses
    #[test]
    fn new_claims_nonstandard_cap() {
        use crate::{
            math::decisions::NONSTANDARD_SLOTS_PER_PERIOD_MAX,
            types::ClaimDecisionPayload,
        };

        let dir = tempfile::tempdir().unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024)
            .max_dbs(State::NUM_DBS + Archive::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, dir.path()) }.unwrap();
        let state = State::new(&env, None, true).unwrap();
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

        let maker = Address([9; 20]);
        let market_with_claims = |claims: u64| {
            let mut tx = create_market_with_new_claim_tx(
                maker,
                DecisionId::new(false, 2, 0).unwrap(),
            );
            tx.spent_utxos[0].content =
                OutputContent::Value(bitcoin::Amount::from_sat(100_000_000));
            if let Some(TransactionData::CreateMarket { new_claims, .. }) =
                &mut tx.transaction.data
            {
                new_claims.extend((1..claims).map(|index| {
                    ClaimDecisionPayload {
                        decision_type: DecisionType::Binary,
                        decisions: vec![DecisionClaimEntry {
                            decision_id_bytes: DecisionId::new(
                                false,
                                2,
                                index as u32,
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
                    }
                }));
            }
            pay_listing_fees(&state, &rotxn, &mut tx);
            MarketValidator::validate_market_creation(&state, &rotxn, &tx, None)
        };

        market_with_claims(NONSTANDARD_SLOTS_PER_PERIOD_MAX).unwrap();
        let over_cap = market_with_claims(NONSTANDARD_SLOTS_PER_PERIOD_MAX + 1)
            .unwrap_err()
            .to_string();
        assert!(over_cap.contains("cap"), "{over_cap}");
    }

    /// A market creation claiming in two periods pays each period's pool in
    /// its own output, neither merged nor missing
    #[test]
    fn claims_two_periods() {
        use crate::types::{ClaimDecisionPayload, GetValue as _};

        let dir = tempfile::tempdir().unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024)
            .max_dbs(State::NUM_DBS + Archive::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, dir.path()) }.unwrap();
        let state = State::new(&env, None, true).unwrap();
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

        let mut tx = create_market_with_new_claim_tx(
            Address([9; 20]),
            DecisionId::new(false, 2, 0).unwrap(),
        );
        if let Some(TransactionData::CreateMarket { new_claims, .. }) =
            &mut tx.transaction.data
        {
            new_claims.push(ClaimDecisionPayload {
                decision_type: DecisionType::Binary,
                decisions: vec![DecisionClaimEntry {
                    decision_id_bytes: DecisionId::new(false, 3, 0)
                        .unwrap()
                        .as_bytes(),
                    header: "decision".to_string(),
                    description: String::new(),
                    option_0_label: None,
                    option_1_label: None,
                    option_labels: None,
                    tags: None,
                }],
            });
        }
        pay_listing_fees(&state, &rotxn, &mut tx);
        let validate = |tx: &FilledTransaction| {
            MarketValidator::validate_market_creation(&state, &rotxn, tx, None)
        };
        validate(&tx).unwrap();

        let mut merged = tx.clone();
        let second = merged.transaction.outputs.0.pop().unwrap();
        let first = merged.transaction.outputs.0.last_mut().unwrap();
        let OutputContent::ListingFees { amount, .. } = &mut first.content
        else {
            unreachable!()
        };
        *amount += second.get_value();
        assert!(validate(&merged).is_err());

        let mut missing = tx;
        missing.transaction.outputs.0.pop();
        assert!(validate(&missing).is_err());
    }

    /// New decision claims must fall in the current period's window
    #[test]
    fn new_claim_window() {
        let dir = tempfile::tempdir().unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024)
            .max_dbs(State::NUM_DBS + Archive::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, dir.path()) }.unwrap();
        let state = State::new(&env, None, true).unwrap();

        use crate::state::voting::period_calculator::{
            utc_quarter, utc_quarter_start,
        };

        let genesis_ts = 1_700_000_000;
        let tenth_period_ts =
            utc_quarter_start(utc_quarter(genesis_ts) + 9) + 60;
        let mut rwtxn = env.write_txn().unwrap();
        state
            .put_timestamps_for_test(&mut rwtxn, genesis_ts, tenth_period_ts)
            .unwrap();
        state
            .decisions()
            .mint_genesis(&mut rwtxn, tenth_period_ts, 0, genesis_ts)
            .unwrap();
        rwtxn.commit().unwrap();
        let rotxn = env.read_txn().unwrap();

        let maker = Address([9; 20]);
        let mut in_window = create_market_with_new_claim_tx(
            maker,
            DecisionId::new(false, 25, 0).unwrap(),
        );
        pay_listing_fees(&state, &rotxn, &mut in_window);
        MarketValidator::validate_market_creation(
            &state, &rotxn, &in_window, None,
        )
        .unwrap();

        let past_window = create_market_with_new_claim_tx(
            maker,
            DecisionId::new(false, 30, 0).unwrap(),
        );
        assert!(
            MarketValidator::validate_market_creation(
                &state,
                &rotxn,
                &past_window,
                None
            )
            .is_err()
        );
    }
}
