use crate::math::trading::MIN_TRADING_FEE_SATS;
use crate::state::Error;
use crate::state::markets::MarketState::*;
use crate::state::markets::{
    DFunction, DimensionSpec, MarketError, MarketState,
};
use crate::state::slots::{Decision, SlotId};
use crate::types::{Address, FilledTransaction};
use sneed::RoTxn;
use std::collections::HashSet;

pub trait SlotValidationInterface {
    fn validate_slot_claim(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<(), Error>;

    fn try_get_height(&self, rotxn: &RoTxn) -> Result<Option<u32>, Error>;

    fn try_get_genesis_timestamp(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error>;

    fn try_get_mainchain_timestamp(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error>;
}

pub struct SlotValidator;

impl SlotValidator {
    pub fn parse_slot_id_from_hex(slot_id_hex: &str) -> Result<SlotId, Error> {
        SlotId::from_hex(slot_id_hex)
    }

    pub fn validate_slot_id_consistency(
        slot_id: &SlotId,
        slot_id_bytes: [u8; 3],
    ) -> Result<(), Error> {
        if slot_id.as_bytes() != slot_id_bytes {
            return Err(Error::InvalidSlotId {
                reason: "Slot ID bytes don't match computed slot ID"
                    .to_string(),
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_decision_structure(
        market_maker_address_bytes: [u8; 20],
        slot_id_bytes: [u8; 3],
        is_standard: bool,
        is_scaled: bool,
        question: &str,
        min: Option<i64>,
        max: Option<i64>,
        option_0_label: Option<String>,
        option_1_label: Option<String>,
    ) -> Result<Decision, Error> {
        Decision::new(
            market_maker_address_bytes,
            slot_id_bytes,
            is_standard,
            is_scaled,
            question.to_string(),
            min,
            max,
            option_0_label,
            option_1_label,
        )
    }

    pub fn validate_complete_decision_slot_claim<T>(
        slots_db: &T,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error>
    where
        T: SlotValidationInterface,
    {
        let claim = tx.claim_decision_slot().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a decision slot claim transaction".to_string(),
            }
        })?;

        let slot_id = SlotId::from_bytes(claim.slot_id_bytes)?;
        Self::validate_slot_id_consistency(&slot_id, claim.slot_id_bytes)?;

        let market_maker_address =
            MarketValidator::validate_market_maker_authorization(tx)?;
        let decision = Self::validate_decision_structure(
            market_maker_address.0,
            claim.slot_id_bytes,
            claim.is_standard,
            claim.is_scaled,
            &claim.question,
            claim.min,
            claim.max,
            claim.option_0_label.clone(),
            claim.option_1_label.clone(),
        )?;

        let current_ts =
            slots_db.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);

        let current_height = override_height
            .or_else(|| slots_db.try_get_height(rotxn).ok().flatten());

        let genesis_ts =
            slots_db.try_get_genesis_timestamp(rotxn)?.unwrap_or(0);

        slots_db
            .validate_slot_claim(
                rotxn,
                slot_id,
                &decision,
                current_ts,
                current_height,
                genesis_ts,
            )
            .map_err(|e| match e {
                Error::SlotNotAvailable { slot_id: _, reason } => {
                    Error::InvalidSlotId { reason }
                }
                Error::SlotAlreadyClaimed { slot_id: _ } => {
                    Error::InvalidSlotId {
                        reason: "Slot already claimed".to_string(),
                    }
                }
                other => other,
            })
    }

    pub fn validate_complete_category_slots_claim<T>(
        slots_db: &T,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error>
    where
        T: SlotValidationInterface,
    {
        let category_claim = tx.claim_category_slots().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a category slots claim transaction".to_string(),
            }
        })?;

        if category_claim.slots.len() < 2 {
            return Err(Error::InvalidTransaction {
                reason: "Category claim requires at least 2 slots".to_string(),
            });
        }

        let mut seen_ids = HashSet::new();
        for (slot_id_bytes, _) in &category_claim.slots {
            if !seen_ids.insert(*slot_id_bytes) {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Duplicate slot ID {} in category claim",
                        hex::encode(slot_id_bytes)
                    ),
                });
            }
        }

        for (slot_id_bytes, question) in &category_claim.slots {
            if question.len() > 1000 {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Question for slot {} exceeds 1000 bytes",
                        hex::encode(slot_id_bytes)
                    ),
                });
            }

            let _slot_id = SlotId::from_bytes(*slot_id_bytes)?;
        }

        let market_maker_address =
            MarketValidator::validate_market_maker_authorization(tx)?;

        let current_ts =
            slots_db.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);

        let current_height = override_height
            .or_else(|| slots_db.try_get_height(rotxn).ok().flatten());

        let genesis_ts =
            slots_db.try_get_genesis_timestamp(rotxn)?.unwrap_or(0);

        let mut first_period: Option<u32> = None;

        for (slot_id_bytes, question) in &category_claim.slots {
            let slot_id = SlotId::from_bytes(*slot_id_bytes)?;

            let slot_period = slot_id.period_index();
            match first_period {
                None => first_period = Some(slot_period),
                Some(expected_period) if expected_period != slot_period => {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "All category slots must be in the same period. Slot {} is in period {} but expected period {}",
                            hex::encode(slot_id_bytes),
                            slot_period,
                            expected_period
                        ),
                    });
                }
                _ => {}
            }

            let decision = Decision::new(
                market_maker_address.0,
                *slot_id_bytes,
                category_claim.is_standard,
                false, // is_scaled = false for category slots
                question.clone(),
                None, // min = None for binary
                None, // max = None for binary
                None,
                None,
            )?;

            slots_db
                .validate_slot_claim(
                    rotxn,
                    slot_id,
                    &decision,
                    current_ts,
                    current_height,
                    genesis_ts,
                )
                .map_err(|e| match e {
                    Error::SlotNotAvailable { slot_id: _, reason } => {
                        Error::InvalidSlotId { reason }
                    }
                    Error::SlotAlreadyClaimed { slot_id: _ } => {
                        Error::InvalidSlotId {
                            reason: format!(
                                "Slot {} already claimed",
                                hex::encode(slot_id_bytes)
                            ),
                        }
                    }
                    other => other,
                })?;
        }

        Ok(())
    }
}

pub struct MarketValidator;

impl MarketValidator {
    pub fn validate_market_maker_authorization(
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
            let slot_ids = match spec {
                DimensionSpec::Single(slot_id) => vec![*slot_id],
                DimensionSpec::Categorical(slot_ids) => slot_ids.clone(),
            };

            if let DimensionSpec::Categorical(ids) = spec
                && ids.len() < 2
            {
                return Err(Error::InvalidTransaction {
                    reason: "Categorical dimensions require at least 2 options"
                        .to_string(),
                });
            }

            for slot_id in slot_ids {
                let slot = state
                    .slots()
                    .get_slot(rotxn, slot_id)?
                    .ok_or_else(|| Error::InvalidTransaction {
                        reason: format!(
                            "Referenced decision slot {slot_id:?} does not exist"
                        ),
                    })?;

                let decision =
                    slot.decision.ok_or_else(|| Error::InvalidTransaction {
                        reason: format!(
                            "Referenced slot {slot_id:?} has no decision claimed"
                        ),
                    })?;

                if let DimensionSpec::Categorical(_) = spec
                    && decision.is_scaled
                {
                    return Err(Error::InvalidTransaction {
                        reason:
                            "Categorical dimensions can only use binary decisions"
                                .to_string(),
                    });
                }
            }
        }

        let categorical_dimensions: Vec<_> = dimension_specs
            .iter()
            .filter_map(|spec| match spec {
                DimensionSpec::Categorical(slots) => Some(slots.clone()),
                _ => None,
            })
            .collect();

        if !categorical_dimensions.is_empty() {
            let category_txids = market_data.category_txids.as_ref().ok_or_else(|| {
                Error::InvalidTransaction {
                    reason: "category_txids required when using categorical dimensions [[...]]"
                        .to_string(),
                }
            })?;

            if category_txids.len() != categorical_dimensions.len() {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Expected {} category_txids for {} categorical dimensions, got {}",
                        categorical_dimensions.len(),
                        categorical_dimensions.len(),
                        category_txids.len()
                    ),
                });
            }

            for (dim_index, (dim_slots, category_txid_bytes)) in
                categorical_dimensions
                    .iter()
                    .zip(category_txids.iter())
                    .enumerate()
            {
                let category_slots: HashSet<_> = dim_slots
                    .iter()
                    .map(|slot_id| {
                        let slot = state
                            .slots()
                            .get_slot(rotxn, *slot_id)?
                            .ok_or_else(|| Error::InvalidTransaction {
                                reason: format!("Slot {slot_id:?} not found"),
                            })?;

                        if slot.claiming_txid.0 != *category_txid_bytes {
                            return Err(Error::InvalidTransaction {
                                reason: format!(
                                    "Slot {} in categorical dimension {} was not claimed by the specified category (expected txid {}, slot has txid {})",
                                    hex::encode(slot_id.as_bytes()),
                                    dim_index,
                                    hex::encode(category_txid_bytes),
                                    hex::encode(slot.claiming_txid.0)
                                ),
                            });
                        }
                        Ok(*slot_id)
                    })
                    .collect::<Result<HashSet<_>, Error>>()?;

                if category_slots.len() != dim_slots.len() {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "Duplicate slots found in categorical dimension {dim_index}"
                        ),
                    });
                }
            }
        }

        let beta = market_data.b;
        if beta <= 0.0 {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "LMSR beta parameter must be positive, got {beta}"
                ),
            });
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

        let market_maker_address =
            Self::validate_market_maker_authorization(tx)?;

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
        let treasury_output = tx.outputs().iter().find(|output| {
            matches!(
                &output.content,
                crate::types::OutputContent::MarketFunds { market_id, is_fee: false, .. }
                    if market_id == &expected_market_id_bytes
            )
        });

        if treasury_output.is_none() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "CreateMarket tx must have MarketFunds (treasury) output with market_id {}",
                    hex::encode(expected_market_id_bytes)
                ),
            });
        }

        Ok(())
    }

    pub fn validate_trade(
        state: &crate::state::State,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::math::trading::TRADE_MINER_FEE_SATS;
        use crate::state::markets::MarketState;

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

        // Outcome index must be valid
        if trade.outcome_index as usize >= market.shares().len() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Outcome index {} out of range (market has {} outcomes)",
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

        let _tx_sender = Self::validate_market_maker_authorization(tx)?;

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
            if !has_trader_input {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Sell transaction has no input from trader {} to prove ownership",
                        trade.trader
                    ),
                });
            }

            let owned_shares = state
                .markets()
                .get_user_share_account(rotxn, &trade.trader)?
                .and_then(|acc| {
                    acc.positions
                        .get(&(trade.market_id.clone(), trade.outcome_index))
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

    pub fn validate_batched_trades(
        state: &crate::state::State,
        rotxn: &RoTxn,
        batched_trades: &[crate::state::markets::BatchedMarketTrade],
    ) -> Result<Vec<f64>, Error> {
        use crate::state::markets::{MarketError, MarketState};

        let mut trade_costs = Vec::with_capacity(batched_trades.len());

        for (trade_index, trade) in batched_trades.iter().enumerate() {
            let market = state
                .markets()
                .get_market(rotxn, &trade.market_id)
                .map_err(|e| {
                    Error::DatabaseError(format!("Market access failed: {e}"))
                })?
                .ok_or_else(|| {
                    Error::Market(MarketError::MarketNotFound {
                        id: trade.market_id.clone(),
                    })
                })?;

            if market.state() != MarketState::Trading {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Batch trade {}: Market {:?} not in trading state (current: {:?})",
                        trade_index,
                        trade.market_id,
                        market.state()
                    ),
                });
            }

            if trade.outcome_index as usize >= market.shares().len() {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Batch trade {}: Outcome index {} out of range (market has {} outcomes)",
                        trade_index,
                        trade.outcome_index,
                        market.shares().len()
                    ),
                });
            }

            // Use calculate_trade_cost_sats() which enforces MIN_TRADING_FEE_SATS
            let buy_cost = trade.calculate_trade_cost_sats().map_err(|e| {
                Error::InvalidTransaction {
                    reason: format!(
                        "Batch trade {trade_index}: Trade cost calculation failed: {e}"
                    ),
                }
            })?;

            if buy_cost.trading_fee_sats < MIN_TRADING_FEE_SATS {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Batch trade {}: Trading fee {} sats below minimum {} sats",
                        trade_index,
                        buy_cost.trading_fee_sats,
                        MIN_TRADING_FEE_SATS
                    ),
                });
            }

            if buy_cost.total_cost_sats > trade.max_cost {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Batch trade {}: Trade cost {} sats exceeds max cost {} sats",
                        trade_index, buy_cost.total_cost_sats, trade.max_cost
                    ),
                });
            }

            trade_costs.push(buy_cost.total_cost_sats as f64);
        }

        Ok(trade_costs)
    }

    pub fn validate_lmsr_parameters(
        beta: f64,
        shares: &ndarray::Array1<i64>,
    ) -> Result<(), Error> {
        if beta <= 0.0 || !beta.is_finite() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "LMSR beta must be positive and finite, got {beta}"
                ),
            });
        }

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

/// D-function validation utilities for market constraints.
pub struct DFunctionValidator;

impl DFunctionValidator {
    /// Validate D-function constraints against market dimensions.
    ///
    /// Ensures that D-function references are within bounds and structurally valid.
    ///
    /// # Arguments
    /// * `d_function` - The D-function to validate
    /// * `max_decision_index` - Maximum valid decision index
    ///
    /// # Returns
    /// * `Ok(())` - D-function is valid
    /// * `Err(MarketError)` - Invalid D-function with detailed reason
    pub fn validate_constraint(
        d_function: &DFunction,
        max_decision_index: usize,
    ) -> Result<(), MarketError> {
        match d_function {
            DFunction::Decision(idx) => {
                if *idx >= max_decision_index {
                    return Err(MarketError::InvalidDimensions);
                }
                Ok(())
            }
            DFunction::Equals(func, value) => {
                Self::validate_constraint(func, max_decision_index)?;
                if let DFunction::Decision(_) = func.as_ref()
                    && *value > 2
                {
                    return Err(MarketError::InvalidOutcomeCombination);
                }
                Ok(())
            }
            DFunction::And(left, right) => {
                Self::validate_constraint(left, max_decision_index)?;
                Self::validate_constraint(right, max_decision_index)?;
                Ok(())
            }
            DFunction::Or(left, right) => {
                Self::validate_constraint(left, max_decision_index)?;
                Self::validate_constraint(right, max_decision_index)?;
                Ok(())
            }
            DFunction::Not(func) => {
                Self::validate_constraint(func, max_decision_index)?;
                Ok(())
            }
            DFunction::True => Ok(()),
        }
    }

    /// Check if this D-function creates valid categorical constraints.
    ///
    /// For categorical dimensions, exactly one option should be true, with all others false.
    /// This validates that the D-function properly enforces mutual exclusivity.
    ///
    /// # Arguments
    /// * `d_function` - The D-function to validate
    /// * `categorical_slots` - Slot indices that form a categorical dimension
    /// * `combo` - The outcome combination to validate
    /// * `decision_slots` - Available decision slots
    ///
    /// # Returns
    /// * `Ok(true)` - Valid categorical constraint (exactly one true)
    /// * `Ok(false)` - Invalid categorical constraint (zero or multiple true)
    /// * `Err(MarketError)` - Evaluation error
    pub fn validate_categorical_constraint(
        _d_function: &DFunction,
        categorical_slots: &[usize],
        combo: &[usize],
        _decision_slots: &[SlotId],
    ) -> Result<bool, MarketError> {
        let mut true_count = 0;

        for &slot_idx in categorical_slots {
            if slot_idx >= combo.len() {
                return Err(MarketError::InvalidDimensions);
            }

            if combo[slot_idx] == 1 {
                true_count += 1;
            }
        }

        Ok(true_count <= 1)
    }

    /// Validate dimensional consistency across all D-functions.
    ///
    /// Ensures that D-functions properly represent the market's dimensional structure
    /// and that all outcome combinations are valid according to whitepaper specifications.
    ///
    /// # Arguments
    /// * `d_functions` - All D-functions for the market
    /// * `decision_slots_len` - Number of decision slots
    /// * `all_combos` - All possible outcome combinations
    ///
    /// # Returns
    /// * `Ok(())` - All constraints are dimensionally consistent
    /// * `Err(MarketError)` - Inconsistent dimensional constraints
    pub fn validate_dimensional_consistency(
        d_functions: &[DFunction],
        decision_slots_len: usize,
        all_combos: &[Vec<usize>],
    ) -> Result<(), MarketError> {
        if d_functions.len() != all_combos.len() {
            return Err(MarketError::InvalidDimensions);
        }

        // Validate each D-function references valid decision indices
        for df in d_functions {
            Self::validate_constraint(df, decision_slots_len)?;
        }

        // Key invariant: exactly one D-function satisfies each combo
        for combo in all_combos {
            let satisfied_count = d_functions
                .iter()
                .filter(|df| df.evaluate(combo).unwrap_or(false))
                .count();
            if satisfied_count != 1 {
                return Err(MarketError::InvalidOutcomeCombination);
            }
        }

        Ok(())
    }
}

/// Market state transition validation.
pub struct MarketStateValidator;

impl MarketStateValidator {
    /// Validate market state transition according to Bitcoin Hivemind specification.
    ///
    /// Ensures state transitions follow valid paths per whitepaper requirements:
    /// - Trading -> Ossified (automatic payout when voting completes)
    /// - Trading -> Cancelled (if no trades occurred)
    /// - Trading -> Invalid (governance action)
    ///
    /// # Arguments
    /// * `from_state` - Current market state
    /// * `to_state` - Proposed new state
    ///
    /// # Returns
    /// * `Ok(())` - Valid state transition
    /// * `Err(Error)` - Invalid transition with detailed reason
    pub fn validate_market_state_transition(
        from_state: MarketState,
        to_state: MarketState,
    ) -> Result<(), Error> {
        let valid_transition = match (from_state, to_state) {
            (a, b) if a == b => true,
            (Trading, Ossified) => true,
            (Trading, Cancelled) => true,
            (Trading, Invalid) => true,
            (Invalid, Ossified) => true,
            _ => false,
        };

        if !valid_transition {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Invalid market state transition from {from_state:?} to {to_state:?}"
                ),
            });
        }

        Ok(())
    }
}

/// Vote submission validation utilities.
///
/// This validator ensures vote submissions comply with Bitcoin Hivemind
/// specifications for the consensus mechanism.
pub struct VoteValidator;

impl VoteValidator {
    /// Convert f64 vote value to VoteValue enum.
    ///
    /// Per Bitcoin Hivemind whitepaper section 3.3, vote values are represented as:
    /// - NaN: Abstain (voter chooses not to vote)
    /// - 0.0 or 1.0: Binary (false/true for binary decisions)
    /// - Other values in [0.0, 1.0]: Scalar (continuous values for scalar decisions)
    ///
    /// This serves as the single source of truth for vote value conversion,
    /// eliminating duplication between vote submission and batch processing.
    ///
    /// # Arguments
    /// * `vote_value` - Raw f64 vote value from transaction
    ///
    /// # Returns
    /// * `VoteValue` - Typed vote value for storage
    pub fn convert_vote_value(
        vote_value: f64,
    ) -> crate::state::voting::types::VoteValue {
        use crate::state::voting::types::VoteValue;

        if vote_value.is_nan() {
            VoteValue::Abstain
        } else if vote_value == 0.0 || vote_value == 1.0 {
            VoteValue::Binary(vote_value == 1.0)
        } else {
            VoteValue::Scalar(vote_value)
        }
    }

    /// Validate voter eligibility and Votecoin balance.
    ///
    /// Ensures the voter has voting rights according to Bitcoin Hivemind whitepaper
    /// section 3.2: voters must hold Votecoin to participate.
    ///
    /// # Arguments
    /// * `state` - Blockchain state for balance queries
    /// * `rotxn` - Read-only transaction
    /// * `voter_address` - Address to validate
    ///
    /// # Returns
    /// * `Ok(u32)` - Votecoin balance if voter is eligible
    /// * `Err(Error)` - Voter has no voting rights
    fn validate_voter_eligibility(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voter_address: &crate::types::Address,
    ) -> Result<u32, Error> {
        let votecoin_balance =
            state.get_votecoin_balance(rotxn, voter_address)?;
        if votecoin_balance == 0 {
            return Err(Error::InvalidTransaction {
                reason: "Voter has no Votecoin balance (voting rights)"
                    .to_string(),
            });
        }
        Ok(votecoin_balance)
    }

    /// Validate decision slot exists and has a decision.
    ///
    /// Per Bitcoin Hivemind whitepaper section 3.1, decisions must be claimed
    /// before they can receive votes.
    ///
    /// # Arguments
    /// * `state` - Blockchain state for slot queries
    /// * `rotxn` - Read-only transaction
    /// * `decision_id` - Slot ID to validate
    ///
    /// # Returns
    /// * `Ok(Decision)` - Valid decision for the slot
    /// * `Err(Error)` - Slot doesn't exist or has no decision
    fn validate_decision_slot(
        state: &crate::state::State,
        rotxn: &RoTxn,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<crate::state::slots::Decision, Error> {
        let slot =
            state.slots().get_slot(rotxn, decision_id)?.ok_or_else(|| {
                Error::InvalidSlotId {
                    reason: format!(
                        "Decision slot {decision_id:?} does not exist"
                    ),
                }
            })?;

        let decision = slot.decision.ok_or_else(|| Error::InvalidSlotId {
            reason: format!("Slot {decision_id:?} has no decision"),
        })?;

        Ok(decision)
    }

    /// Validate vote value is in valid normalized range.
    ///
    /// Vote values are stored internally as normalized 0-1 values:
    /// - Binary decisions: 0.0 (No), 1.0 (Yes), 0.5 (Inconclusive), or any value in [0,1]
    /// - Scaled decisions: 0.0-1.0 (normalized from user-facing min/max range)
    /// - NaN represents abstention for both types
    ///
    /// The RPC layer normalizes user-facing values (e.g., $152,500 for BTC price
    /// in range [10000, 200000] becomes 0.75) to internal 0-1 representation
    /// before storage.
    ///
    /// # Arguments
    /// * `decision` - Decision to validate against
    /// * `vote_value` - Normalized vote value to validate (0-1 range)
    ///
    /// # Returns
    /// * `Ok(())` - Vote value is valid for decision type
    /// * `Err(Error)` - Invalid vote value
    fn validate_vote_value(
        _decision: &crate::state::slots::Decision,
        vote_value: f64,
    ) -> Result<(), Error> {
        // All vote values (binary and scaled) are stored normalized to 0-1 range
        // - Binary: 0.0 (No), 1.0 (Yes), 0.5 (Inconclusive), or any value in between
        // - Scaled: normalized from user-facing [min, max] to [0, 1]
        // NaN is valid for abstaining
        if !vote_value.is_nan() && !(0.0..=1.0).contains(&vote_value) {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Vote value {vote_value} outside valid normalized range [0.0, 1.0]"
                ),
            });
        }

        Ok(())
    }

    /// Validate slot is in voting period.
    ///
    /// Per Bitcoin Hivemind whitepaper section 3.2, votes can only be submitted
    /// during active voting periods.
    ///
    /// # Arguments
    /// * `state` - Blockchain state
    /// * `decision_id` - Slot to check
    /// * `current_ts` - Current timestamp (cached for efficiency)
    /// * `current_height` - Current height (cached for efficiency)
    ///
    /// # Returns
    /// * `Ok(())` - Slot is in voting period
    /// * `Err(Error)` - Slot is not accepting votes
    fn validate_voting_period(
        state: &crate::state::State,
        rotxn: &sneed::RoTxn,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<(), Error> {
        if !state.slots().is_slot_in_voting(rotxn, decision_id)? {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Decision slot {decision_id:?} is not in voting period"
                ),
            });
        }
        Ok(())
    }

    /// Validate no duplicate votes exist.
    ///
    /// Per Bitcoin Hivemind whitepaper section 3.2, each voter can submit
    /// exactly one vote per decision per voting period.
    ///
    /// # Arguments
    /// * `state` - Blockchain state
    /// * `rotxn` - Read-only transaction
    /// * `voter_id` - Voter to check
    /// * `period_id` - Voting period
    /// * `decision_id` - Decision slot
    ///
    /// # Returns
    /// * `Ok(())` - No duplicate vote exists
    /// * `Err(Error)` - Duplicate vote detected
    fn validate_no_duplicate_vote(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voter_address: crate::types::Address,
        period_id: crate::state::voting::types::VotingPeriodId,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<(), Error> {
        if state
            .voting()
            .databases()
            .get_vote(rotxn, period_id, voter_address, decision_id)?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: "Duplicate vote: voter already voted on this decision in this period"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// Validate complete vote submission transaction
    ///
    /// Ensures all Bitcoin Hivemind requirements are met:
    /// 1. Voter has Votecoin balance > 0 (voting rights)
    /// 2. Voting period exists and is active
    /// 3. Decision slot exists and is in voting period
    /// 4. Vote value is valid for decision type
    /// 5. No duplicate votes (one per voter per decision per period)
    ///
    /// # Arguments
    /// * `state` - Blockchain state for validation queries
    /// * `rotxn` - Read-only transaction
    /// * `filled_tx` - Filled transaction to validate
    /// * `override_height` - Optional height override for validation context
    ///
    /// # Returns
    /// * `Ok(())` - Valid vote submission
    /// * `Err(Error)` - Invalid vote with detailed reason
    pub fn validate_vote_submission(
        state: &crate::state::State,
        rotxn: &RoTxn,
        filled_tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::state::{slots::SlotId, voting::types::VotingPeriodId};

        let vote_data = filled_tx.submit_vote().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a vote submission transaction".to_string(),
            }
        })?;

        // Extract voter address
        let voter_address = filled_tx
            .spent_utxos
            .first()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Vote transaction must have inputs".to_string(),
            })?
            .address;

        let _votecoin_balance =
            Self::validate_voter_eligibility(state, rotxn, &voter_address)?;

        let decision_id = SlotId::from_bytes(vote_data.slot_id_bytes)?;
        let decision = Self::validate_decision_slot(state, rotxn, decision_id)?;

        Self::validate_vote_value(&decision, vote_data.vote_value)?;

        Self::validate_voting_period(state, rotxn, decision_id)?;

        // Voting period is deterministically derived from slot: voting_period = period_index + 1
        let period_id = VotingPeriodId::new(decision_id.voting_period());
        Self::validate_no_duplicate_vote(
            state,
            rotxn,
            voter_address,
            period_id,
            decision_id,
        )?;

        Ok(())
    }

    /// Validate batch vote submission transaction
    ///
    /// Validates all votes in a batch submission according to Bitcoin Hivemind
    /// specifications.
    ///
    /// # Arguments
    /// * `state` - Blockchain state for validation queries
    /// * `rotxn` - Read-only transaction
    /// * `filled_tx` - Filled transaction containing batch votes
    /// * `override_height` - Optional height override for validation context
    ///
    /// # Returns
    /// * `Ok(())` - All votes in batch are valid
    /// * `Err(Error)` - Invalid batch with detailed reason
    pub fn validate_vote_batch(
        state: &crate::state::State,
        rotxn: &RoTxn,
        filled_tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::state::{slots::SlotId, voting::types::VotingPeriodId};

        let batch_data = filled_tx.submit_vote_batch().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a vote batch submission transaction".to_string(),
            }
        })?;

        let voter_address = filled_tx
            .spent_utxos
            .first()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Vote batch transaction must have inputs".to_string(),
            })?
            .address;

        let _votecoin_balance =
            Self::validate_voter_eligibility(state, rotxn, &voter_address)?;

        let mut seen_votes =
            std::collections::HashSet::<(VotingPeriodId, SlotId)>::new();
        for (idx, vote_item) in batch_data.votes.iter().enumerate() {
            let decision_id = SlotId::from_bytes(vote_item.slot_id_bytes)?;

            // Voting period is deterministically derived from slot: voting_period = period_index + 1
            let period_id = VotingPeriodId::new(decision_id.voting_period());

            if !seen_votes.insert((period_id, decision_id)) {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Vote batch item {idx}: duplicate vote for slot {decision_id:?} in period {period_id:?}"
                    ),
                });
            }

            let decision =
                Self::validate_decision_slot(state, rotxn, decision_id)
                    .map_err(|e| match e {
                        Error::InvalidSlotId { reason } => {
                            Error::InvalidSlotId {
                                reason: format!(
                                    "Vote batch item {idx}: {reason}"
                                ),
                            }
                        }
                        other => other,
                    })?;

            Self::validate_vote_value(&decision, vote_item.vote_value)
                .map_err(|e| match e {
                    Error::InvalidTransaction { reason } => {
                        Error::InvalidTransaction {
                            reason: format!("Vote batch item {idx}: {reason}"),
                        }
                    }
                    other => other,
                })?;

            Self::validate_voting_period(state, rotxn, decision_id).map_err(
                |e| match e {
                    Error::InvalidTransaction { reason } => {
                        Error::InvalidTransaction {
                            reason: format!("Vote batch item {idx}: {reason}"),
                        }
                    }
                    other => other,
                },
            )?;

            Self::validate_no_duplicate_vote(
                state,
                rotxn,
                voter_address,
                period_id,
                decision_id,
            )
            .map_err(|e| match e {
                Error::InvalidTransaction { reason } => {
                    Error::InvalidTransaction {
                        reason: format!("Vote batch item {idx}: {reason}"),
                    }
                }
                other => other,
            })?;
        }

        Ok(())
    }
}

/// Voter registration and reputation validation
pub struct VoterValidator;

impl VoterValidator {
    pub fn validate_voter_not_registered(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voter_address: crate::types::Address,
    ) -> Result<(), Error> {
        if state
            .voting()
            .databases()
            .get_voter_reputation(rotxn, voter_address)?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: "Voter already registered".to_string(),
            });
        }
        Ok(())
    }
}

/// Voting period lifecycle validation
pub struct PeriodValidator;

impl PeriodValidator {
    pub fn validate_decision_in_period(
        period: &crate::state::voting::types::VotingPeriod,
        decision_id: SlotId,
    ) -> Result<(), Error> {
        if !period.decision_slots.contains(&decision_id) {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Decision {:?} not available in period {:?}",
                    decision_id, period.id
                ),
            });
        }
        Ok(())
    }
}
