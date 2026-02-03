use crate::math::trading::MIN_TRADING_FEE_SATS;
use crate::state::Error;
use crate::state::markets::MarketState::*;
use crate::state::markets::{
    DFunction, DimensionSpec, MarketError, MarketState,
    generate_market_author_fee_address, generate_market_treasury_address,
};
use crate::state::slots::{Decision, SlotId};
use crate::types::{Address, FilledOutputContent, FilledTransaction};
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
    ) -> Result<(), Error>;

    fn try_get_height(&self, rotxn: &RoTxn) -> Result<Option<u32>, Error>;
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

    pub fn validate_decision_structure(
        market_maker_address_bytes: [u8; 20],
        slot_id_bytes: [u8; 3],
        is_standard: bool,
        is_scaled: bool,
        question: &str,
        min: Option<i64>,
        max: Option<i64>,
    ) -> Result<Decision, Error> {
        Decision::new(
            market_maker_address_bytes,
            slot_id_bytes,
            is_standard,
            is_scaled,
            question.to_string(),
            min,
            max,
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
        )?;

        let current_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();

        let current_height = override_height
            .or_else(|| slots_db.try_get_height(rotxn).ok().flatten());

        slots_db
            .validate_slot_claim(
                rotxn,
                slot_id,
                &decision,
                current_ts,
                current_height,
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

    /// Validate a complete category slots claim transaction.
    ///
    /// Category claims allow atomic claiming of multiple slots under a single txid
    /// that serves as the category identifier. All slots in a category are binary.
    ///
    /// # Validation Rules
    /// 1. At least 2 slots in the category (otherwise use single claim)
    /// 2. No duplicate slot IDs within the claim
    /// 3. All questions under 1000 bytes
    /// 4. All slot IDs valid format
    /// 5. All slots pass individual validation (not claimed, not ossified, not voting)
    /// 6. All slots in same period (since they form a category)
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

        let current_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();

        let current_height = override_height
            .or_else(|| slots_db.try_get_height(rotxn).ok().flatten());

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
            )?;

            slots_db
                .validate_slot_claim(
                    rotxn,
                    slot_id,
                    &decision,
                    current_ts,
                    current_height,
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

/// Market validation utilities for Bitcoin Hivemind prediction markets.
///
/// Provides centralized validation for all market operations following the
/// single-source-of-truth pattern established for slots and voting validation.
pub struct MarketValidator;

impl MarketValidator {
    /// Validate market maker/trader authorization from transaction inputs.
    ///
    /// Extracts and validates the address of the market participant (maker or trader)
    /// from the first UTXO in the transaction, following Bitcoin's standard pattern
    /// of using the first input to identify the transaction originator.
    ///
    /// # Arguments
    /// * `tx` - Filled transaction containing spent UTXOs
    ///
    /// # Returns
    /// * `Ok(Address)` - Validated market participant address
    /// * `Err(Error)` - Invalid transaction structure
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper section on market participant identification
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

    /// Validate complete market creation transaction.
    ///
    /// Ensures all Bitcoin Hivemind requirements are met for market creation:
    /// 1. Valid market type (independent or categorical)
    /// 2. At least one decision slot referenced
    /// 3. All decision slots exist and have decisions
    /// 4. Categorical markets use only binary decisions
    /// 5. LMSR parameters are valid (beta > 0, 0 <= fee <= 1)
    /// 6. Market maker is properly authorized
    ///
    /// # Arguments
    /// * `state` - Blockchain state for validation queries
    /// * `rotxn` - Read-only transaction
    /// * `tx` - Filled transaction to validate
    /// * `_override_height` - Optional height override for validation context
    ///
    /// # Returns
    /// * `Ok(())` - Valid market creation
    /// * `Err(Error)` - Invalid market with detailed reason
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper sections on market creation and LMSR parameters
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
        use crate::state::markets::MarketState;

        let trade = tx.trade().ok_or_else(|| Error::InvalidTransaction {
            reason: "Not a trade transaction".to_string(),
        })?;

        let is_buy = trade.is_buy();
        let shares_abs = trade.shares_abs();

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
        if trade.shares == 0.0 {
            return Err(Error::InvalidTransaction {
                reason: "Shares to trade must be non-zero".to_string(),
            });
        }

        if is_buy && trade.limit_sats == 0 {
            return Err(Error::InvalidTransaction {
                reason: "Buy limit (max_cost) must be positive".to_string(),
            });
        }

        // Trading fee must meet minimum (prevents 0-value fee UTXOs)
        if trade.fee_sats < MIN_TRADING_FEE_SATS {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Trading fee {} sats below minimum {} sats",
                    trade.fee_sats, MIN_TRADING_FEE_SATS
                ),
            });
        }

        // Verify trader authorization (validates tx has inputs and spent_utxos)
        let _tx_sender = Self::validate_market_maker_authorization(tx)?;

        if is_buy {
            // Buy-specific validation

            // Slippage protection: embedded total cost must not exceed limit (max_cost)
            let embedded_total = trade.base_sats + trade.fee_sats;
            if embedded_total > trade.limit_sats {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Embedded cost {} sats (base: {}, fee: {}) exceeds max cost {} sats",
                        embedded_total,
                        trade.base_sats,
                        trade.fee_sats,
                        trade.limit_sats
                    ),
                });
            }

            // Verify outputs match embedded costs (treasury + fee outputs required for buy)
            let treasury_address =
                generate_market_treasury_address(&trade.market_id);
            let fee_address =
                generate_market_author_fee_address(&trade.market_id);
            let market_id_bytes = *trade.market_id.as_bytes();

            let (treasury_output_amount, fee_output_amount) = tx
                .filled_outputs()
                .map(|outputs| {
                    let treasury_amt = outputs
                        .iter()
                        .filter_map(|o| {
                            if o.address == treasury_address {
                                match &o.content {
                                    FilledOutputContent::MarketFunds {
                                        market_id,
                                        amount,
                                        is_fee: false,
                                    } if *market_id == market_id_bytes => {
                                        Some(amount.0.to_sat())
                                    }
                                    _ => None,
                                }
                            } else {
                                None
                            }
                        })
                        .sum::<u64>();

                    let fee_amt = outputs
                        .iter()
                        .filter_map(|o| {
                            if o.address == fee_address {
                                match &o.content {
                                    FilledOutputContent::MarketFunds {
                                        market_id,
                                        amount,
                                        is_fee: true,
                                    } if *market_id == market_id_bytes => {
                                        Some(amount.0.to_sat())
                                    }
                                    _ => None,
                                }
                            } else {
                                None
                            }
                        })
                        .sum::<u64>();

                    (treasury_amt, fee_amt)
                })
                .unwrap_or((0, 0));

            if treasury_output_amount < trade.base_sats {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Trade tx missing treasury output: expected {} sats to {}, found {} sats",
                        trade.base_sats,
                        treasury_address,
                        treasury_output_amount
                    ),
                });
            }

            if fee_output_amount < trade.fee_sats {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Trade tx missing fee output: expected {} sats to {}, found {} sats",
                        trade.fee_sats, fee_address, fee_output_amount
                    ),
                });
            }
        } else {
            // Sell-specific validation

            // Verify at least one input is from trader address (proves ownership).
            // The wallet may include inputs from other addresses to pay fees,
            // but must include at least one from the trader who owns the shares.
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
                .unwrap_or(0.0);

            if owned_shares < shares_abs {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Insufficient shares: trying to sell {} but only own {} for outcome {}",
                        shares_abs, owned_shares, trade.outcome_index
                    ),
                });
            }

            // Slippage protection: embedded net proceeds must meet limit (min_proceeds)
            // Skip if limit_sats == 0 (no minimum proceeds requirement)
            if trade.limit_sats > 0 {
                let embedded_net_proceeds =
                    trade.base_sats.saturating_sub(trade.fee_sats);
                if embedded_net_proceeds < trade.limit_sats {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "Embedded net proceeds {} sats below minimum {} sats (slippage protection)",
                            embedded_net_proceeds, trade.limit_sats
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate batched market trades for atomic processing.
    ///
    /// # Arguments
    /// * `state` - Blockchain state for validation queries
    /// * `rotxn` - Read-only transaction
    /// * `batched_trades` - Vector of batched market trades to validate
    ///
    /// # Returns
    /// * `Ok(Vec<f64>)` - Vector of validated trade costs
    /// * `Err(Error)` - Invalid batch with detailed reason
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

            let cost = buy_cost.total_cost_sats as f64;
            if cost > trade.max_cost as f64 {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Batch trade {}: Trade cost {:.4} exceeds max cost {}",
                        trade_index, cost, trade.max_cost
                    ),
                });
            }

            trade_costs.push(cost);
        }

        Ok(trade_costs)
    }

    /// Validate LMSR parameters for market integrity.
    ///
    /// Ensures LMSR parameters (beta and share quantities) are valid and within
    /// acceptable ranges to prevent numerical instability or overflow.
    ///
    /// # Arguments
    /// * `beta` - LMSR beta parameter (liquidity sensitivity)
    /// * `shares` - Current share quantities for all outcomes
    ///
    /// # Returns
    /// * `Ok(())` - Valid LMSR parameters
    /// * `Err(Error)` - Invalid parameters with detailed reason
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper section on LMSR market maker algorithm
    pub fn validate_lmsr_parameters(
        beta: f64,
        shares: &ndarray::Array1<f64>,
    ) -> Result<(), Error> {
        if beta <= 0.0 || !beta.is_finite() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "LMSR beta must be positive and finite, got {beta}"
                ),
            });
        }

        for (idx, &share_qty) in shares.iter().enumerate() {
            if share_qty < 0.0 || !share_qty.is_finite() {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Share quantity at index {idx} must be non-negative and finite, got {share_qty}"
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

    /// Check if market has entered voting period based on slot states.
    ///
    /// A market enters voting when any of its decision slots enter voting.
    ///
    /// # Arguments
    /// * `market_slots` - Set of slot IDs used by this market
    /// * `slots_in_voting` - Set of slot IDs currently in voting
    ///
    /// # Returns
    /// * `true` if market should transition to voting
    /// * `false` otherwise
    pub fn should_enter_voting(
        market_slots: &HashSet<SlotId>,
        slots_in_voting: &HashSet<SlotId>,
    ) -> bool {
        market_slots
            .iter()
            .any(|slot_id| slots_in_voting.contains(slot_id))
    }

    /// Check if all decision slots are ossified.
    ///
    /// A market is ready for resolution when all its decision slots are ossified.
    ///
    /// # Arguments
    /// * `market_slots` - Set of slot IDs used by this market
    /// * `slot_states` - Map of slot IDs to their ossification status
    ///
    /// # Returns
    /// * `true` if all slots are ossified
    /// * `false` otherwise
    pub fn all_slots_ossified(
        market_slots: &HashSet<SlotId>,
        slot_states: &std::collections::HashMap<SlotId, bool>,
    ) -> bool {
        market_slots
            .iter()
            .all(|slot_id| slot_states.get(slot_id).copied().unwrap_or(false))
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

        for (idx, vote_item) in batch_data.votes.iter().enumerate() {
            let decision_id = SlotId::from_bytes(vote_item.slot_id_bytes)?;

            // Voting period is deterministically derived from slot: voting_period = period_index + 1
            let period_id = VotingPeriodId::new(decision_id.voting_period());

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

    pub fn validate_voter_exists(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voter_address: crate::types::Address,
    ) -> Result<(), Error> {
        if state
            .voting()
            .databases()
            .get_voter_reputation(rotxn, voter_address)?
            .is_none()
        {
            return Err(Error::InvalidTransaction {
                reason: "Voter not found".to_string(),
            });
        }
        Ok(())
    }

    pub fn validate_reputation_update(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voter_address: crate::types::Address,
        period_id: crate::state::voting::types::VotingPeriodId,
    ) -> Result<(), Error> {
        Self::validate_voter_exists(state, rotxn, voter_address)?;

        let consensus_outcomes = state
            .voting()
            .databases()
            .get_consensus_outcomes_for_period(rotxn, period_id)?;

        if consensus_outcomes.is_empty() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "No consensus outcomes found for period {period_id:?}"
                ),
            });
        }

        let voter_votes = state
            .voting()
            .databases()
            .get_votes_by_voter(rotxn, voter_address)?;

        let has_votes_in_period = voter_votes
            .iter()
            .any(|(key, _)| key.period_id == period_id);

        if !has_votes_in_period {
            return Err(Error::InvalidTransaction {
                reason: "Voter has no votes in this period".to_string(),
            });
        }

        Ok(())
    }
}

/// Voting period lifecycle validation
pub struct PeriodValidator;

impl PeriodValidator {
    pub fn validate_period_can_close(
        period: &crate::state::voting::types::VotingPeriod,
        current_timestamp: u64,
    ) -> Result<(), Error> {
        use crate::state::voting::types::VotingPeriodStatus;

        if current_timestamp < period.end_timestamp {
            return Err(Error::InvalidTransaction {
                reason: "Cannot close period before end time".to_string(),
            });
        }

        if period.status != VotingPeriodStatus::Active {
            return Err(Error::InvalidTransaction {
                reason: format!("Period {:?} is not active", period.id),
            });
        }

        Ok(())
    }

    pub fn validate_period_is_active(
        period: &crate::state::voting::types::VotingPeriod,
        timestamp: u64,
    ) -> Result<(), Error> {
        use crate::state::voting::types::VotingPeriodStatus;

        if period.status != VotingPeriodStatus::Active {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Period {:?} is not active for voting",
                    period.id
                ),
            });
        }

        if !period.is_active(timestamp) {
            return Err(Error::InvalidTransaction {
                reason: "Timestamp is outside period window".to_string(),
            });
        }

        Ok(())
    }

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
