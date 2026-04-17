use crate::state::Error;
use crate::state::markets::{DimensionSpec, MarketState};
use crate::types::{Address, FilledTransaction};
use sneed::RoTxn;

pub struct MarketValidator;

impl MarketValidator {
    pub fn validate_maker_auth(
        tx: &FilledTransaction,
    ) -> Result<Address, Error> {
        if tx.inputs().is_empty() {
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
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        let market_data =
            tx.create_market()
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: "Not a market creation transaction".to_string(),
                })?;

        let dimension_specs = &market_data.dimension_specs;

        if dimension_specs.is_empty() {
            return Err(Error::InvalidTransaction {
                reason: "Market must have at least one dimension".to_string(),
            });
        }

        for spec in dimension_specs {
            let decision_id = match spec {
                DimensionSpec::Single(id) => id,
                DimensionSpec::Categorical(id) => id,
            };

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

            if let DimensionSpec::Categorical(_) = spec {
                if !decision.is_categorical() {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "Categorical dimension references decision {decision_id:?} which is not a Category type"
                        ),
                    });
                }
                let option_count = decision.option_count().unwrap_or(0);
                if option_count < 2 {
                    return Err(Error::InvalidTransaction {
                        reason:
                            "Categorical dimensions require at least 2 options"
                                .to_string(),
                    });
                }
            }
        }

        if let Some(fee) = market_data.trading_fee
            && (!(0.0..=1.0).contains(&fee))
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Trading fee must be between 0.0 and 1.0, got {fee}"
                ),
            });
        }

        if let Some(difficulty) = market_data.tx_pow_difficulty
            && difficulty > 0
        {
            let config = crate::types::tx_pow::TxPowConfig {
                hash_selector: market_data.tx_pow_hash_selector.unwrap_or(0),
                ordering: market_data.tx_pow_ordering.unwrap_or(0),
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

        let market_maker_address = Self::validate_maker_auth(tx)?;

        // Compute expected market_id and validate treasury output
        use crate::state::markets::compute_market_id;
        let expected_market_id = compute_market_id(
            &market_data.title,
            &market_data.description,
            &market_maker_address,
            dimension_specs,
        );
        let expected_market_id_bytes = *expected_market_id.as_bytes();

        // Check for MarketFunds (treasury) output with matching market_id
        // and a strictly positive amount. Beta is derived from the treasury
        // as `beta = treasury / ln(num_outcomes)`, so a zero-sat treasury
        // would produce a degenerate beta and break LMSR pricing.
        let treasury_amount_sats = tx
            .outputs()
            .iter()
            .find_map(|output| match &output.content {
                crate::types::OutputContent::MarketFunds {
                    market_id,
                    amount,
                    is_fee: false,
                } if market_id == &expected_market_id_bytes => {
                    Some(amount.0.to_sat())
                }
                _ => None,
            })
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!(
                    "CreateMarket tx must have MarketFunds (treasury) output with market_id {}",
                    hex::encode(expected_market_id_bytes)
                ),
            })?;

        if treasury_amount_sats == 0 {
            return Err(Error::InvalidTransaction {
                reason: "CreateMarket treasury output must be positive"
                    .to_string(),
            });
        }

        Ok(())
    }

    pub fn validate_trade(
        state: &crate::state::State,
        archive: &crate::archive::Archive,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::math::trading::TRADE_MINER_FEE_SATS;
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
                    hex::encode(trade.prev_block_hash.0)
                ),
            })?;
        if trade.prev_block_hash != tip
            && !archive.is_descendant(rotxn, trade.prev_block_hash, tip)?
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Trade prev_block_hash {} is not on the active chain",
                    hex::encode(trade.prev_block_hash.0)
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
                trade.outcome_index,
                trade.shares,
                &trade.trader,
                trade.limit_sats,
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

        // Outcome index must be a valid tradeable outcome. Abstain/invalid
        // states (voter-only) are not part of the tradeable outcome space
        // and will naturally fall outside `market.shares().len()`.
        if trade.outcome_index as usize >= market.shares().len() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Outcome index {} out of range (market has {} tradeable \
                     outcomes)",
                    trade.outcome_index,
                    market.shares().len()
                ),
            });
        }

        // Shares must be non-zero
        if trade.shares == 0 {
            return Err(Error::InvalidTransaction {
                reason: "Shares to trade must be non-zero".to_string(),
            });
        }

        if !tx.outputs().is_empty() {
            return Err(Error::InvalidTransaction {
                reason: "Trade transactions must have no outputs".to_string(),
            });
        }

        Self::validate_maker_auth(tx)?;

        let input_value = tx
            .spent_bitcoin_value()
            .map_err(|_| Error::InvalidTransaction {
                reason: "Failed to compute input value".to_string(),
            })?
            .to_sat();

        if trade.is_buy() {
            if trade.limit_sats == 0 {
                return Err(Error::InvalidTransaction {
                    reason: "Buy limit (max_cost) must be positive".to_string(),
                });
            }

            if input_value < trade.limit_sats {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Input value {} sats insufficient for limit_sats {} sats",
                        input_value, trade.limit_sats
                    ),
                });
            }
        } else {
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

            let owned_shares = state
                .markets()
                .get_user_share_account(rotxn, &trade.trader)?
                .and_then(|acc| {
                    acc.positions
                        .get(&(trade.market_id, trade.outcome_index))
                        .copied()
                })
                .unwrap_or(0);

            if owned_shares < 0 || (owned_shares as u64) < trade.shares_abs() {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Insufficient shares: trying to sell {} but only own {} for outcome {}",
                        trade.shares_abs(),
                        owned_shares,
                        trade.outcome_index
                    ),
                });
            }

            if input_value < TRADE_MINER_FEE_SATS {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Input value {input_value} sats insufficient for miner fee {TRADE_MINER_FEE_SATS} sats"
                    ),
                });
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
        use crate::math::trading::TRADE_MINER_FEE_SATS;

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

        if !tx.outputs().is_empty() {
            return Err(Error::InvalidTransaction {
                reason: "AmplifyBeta transactions must have no outputs"
                    .to_string(),
            });
        }

        let input_value = tx
            .spent_bitcoin_value()
            .map_err(|_| Error::InvalidTransaction {
                reason: "Failed to compute input value".to_string(),
            })?
            .to_sat();

        let required = amplify
            .amount
            .checked_add(TRADE_MINER_FEE_SATS)
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "AmplifyBeta amount overflow".to_string(),
            })?;

        if input_value < required {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Input value {input_value} sats insufficient for \
                     amplify amount {} + miner fee {TRADE_MINER_FEE_SATS} sats",
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
    use ndarray::Array1;

    #[test]
    fn market_valid_shares() {
        let shares = Array1::from_vec(vec![100, 100]);
        assert!(MarketValidator::validate_market_shares(&shares).is_ok());
    }

    #[test]
    fn market_negative_shares_rejected() {
        let shares = Array1::from_vec(vec![100, -1]);
        assert!(MarketValidator::validate_market_shares(&shares).is_err());
    }

    #[test]
    fn market_fewer_than_two_outcomes_rejected() {
        let shares = Array1::from_vec(vec![100]);
        assert!(MarketValidator::validate_market_shares(&shares).is_err());
    }

    #[test]
    fn state_transition_valid() {
        assert!(
            MarketStateValidator::validate_market_state_transition(
                MarketState::Trading,
                MarketState::Ossified
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
                MarketState::Ossified,
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
}
