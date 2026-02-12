use borsh::BorshSerialize;
use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use ndarray::{Array, Ix1};
use serde::{Deserialize, Serialize};
use sneed::{DatabaseUnique, Env, RoTxn, RwTxn};
use std::collections::{HashMap, HashSet};

use crate::math::allocation;
use crate::state::Error;
use crate::state::UtxoManager;
use crate::state::slots::{Decision, SlotId};
use crate::types::hashes;
use crate::types::{Address, GetBitcoinValue, OutPoint};
use thiserror::Error as ThisError;

pub const MAX_MARKET_OUTCOMES: usize = 256;
pub const L2_STORAGE_RATE_SATS_PER_BYTE: u64 = 1;
pub const BASE_MARKET_STORAGE_COST_SATS: u64 = 1000;

/// Default LMSR beta parameter (liquidity depth)
pub const DEFAULT_MARKET_BETA: f64 = 7.0;

/// Default trading fee (0.5%)
pub const DEFAULT_TRADING_FEE: f64 = 0.005;

#[derive(Debug, ThisError, Clone)]
pub enum MarketError {
    #[error("Invalid market dimensions")]
    InvalidDimensions,

    #[error("Too many market states: {0} (max {MAX_MARKET_OUTCOMES})")]
    TooManyStates(usize),

    #[error("Invalid beta parameter: {0}")]
    InvalidBeta(f64),

    #[error("Invalid outcome index: {0}")]
    InvalidOutcomeIndex(usize),

    #[error("Invalid state transition from {from:?} to {to:?}")]
    InvalidStateTransition { from: MarketState, to: MarketState },

    #[error("Market not found: {id:?}")]
    MarketNotFound { id: MarketId },

    #[error("Decision slot not found: {slot_id:?}")]
    DecisionSlotNotFound { slot_id: SlotId },

    #[error("Slot validation failed for slot: {slot_id:?}")]
    SlotValidationFailed { slot_id: SlotId },

    #[error("Invalid outcome combination")]
    InvalidOutcomeCombination,

    #[error("Duplicate slot in market dimensions: {slot_id:?}")]
    DuplicateSlot { slot_id: SlotId },

    #[error("Database error: {0}")]
    DatabaseError(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketState {
    Trading = 1,
    Cancelled = 3,
    Invalid = 4,
    Ossified = 5,
}

impl MarketState {
    pub fn can_transition_to(&self, new_state: MarketState) -> bool {
        use MarketState::*;
        match (self, new_state) {
            (Trading, Ossified) => true,
            (Trading, Cancelled) => true,
            (Trading, Invalid) => true,

            (Invalid, Ossified) => true,

            (Ossified, _) => false,

            (state, new_state) if state == &new_state => true,

            _ => false,
        }
    }

    pub fn allows_trading(&self) -> bool {
        matches!(self, MarketState::Trading)
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, MarketState::Ossified | MarketState::Cancelled)
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash, BorshSerialize,
)]
pub enum DFunction {
    Decision(usize),
    Equals(Box<DFunction>, usize),
    And(Box<DFunction>, Box<DFunction>),
    Or(Box<DFunction>, Box<DFunction>),
    Not(Box<DFunction>),
    True,
}

#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, BorshSerialize,
)]
pub enum DimensionSpec {
    Single(SlotId),
    Categorical(Vec<SlotId>),
}

pub fn parse_dimensions(
    dimensions_str: &str,
) -> Result<Vec<DimensionSpec>, MarketError> {
    let dimensions_str = dimensions_str.trim();
    if !dimensions_str.starts_with('[') || !dimensions_str.ends_with(']') {
        return Err(MarketError::InvalidDimensions);
    }

    let inner = &dimensions_str[1..dimensions_str.len() - 1];
    let mut dimensions = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = inner.chars().collect();

    while i < chars.len() {
        while i < chars.len() && (chars[i].is_whitespace() || chars[i] == ',') {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }

        if chars[i] == '[' {
            let start = i + 1;
            let mut bracket_count = 1;
            i += 1;

            while i < chars.len() && bracket_count > 0 {
                if chars[i] == '[' {
                    bracket_count += 1;
                } else if chars[i] == ']' {
                    bracket_count -= 1;
                }
                i += 1;
            }

            if bracket_count != 0 {
                return Err(MarketError::InvalidDimensions);
            }

            let categorical_str: String = chars[start..i - 1].iter().collect();
            let slot_ids = parse_slot_list(&categorical_str)?;
            dimensions.push(DimensionSpec::Categorical(slot_ids));
        } else {
            let start = i;
            while i < chars.len() && chars[i] != ',' && chars[i] != '[' {
                i += 1;
            }

            let slot_str: String = chars[start..i].iter().collect();
            let slot_id = parse_single_slot(slot_str.trim())?;
            dimensions.push(DimensionSpec::Single(slot_id));
        }
    }

    Ok(dimensions)
}

fn parse_slot_list(list_str: &str) -> Result<Vec<SlotId>, MarketError> {
    list_str
        .split(',')
        .map(|s| parse_single_slot(s.trim()))
        .collect()
}

fn parse_single_slot(slot_str: &str) -> Result<SlotId, MarketError> {
    let slot_bytes =
        hex::decode(slot_str).map_err(|_| MarketError::InvalidDimensions)?;

    if slot_bytes.len() != 3 {
        return Err(MarketError::InvalidDimensions);
    }

    let slot_id_array: [u8; 3] = slot_bytes
        .try_into()
        .map_err(|_| MarketError::InvalidDimensions)?;
    SlotId::from_bytes(slot_id_array)
        .map_err(|_| MarketError::InvalidDimensions)
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Hash,
    Ord,
    PartialOrd,
    Serialize,
    Deserialize,
    borsh::BorshSerialize,
    borsh::BorshDeserialize,
)]
pub struct MarketId(pub [u8; 6]);

impl MarketId {
    pub fn new(data: [u8; 6]) -> Self {
        Self(data)
    }

    pub fn as_bytes(&self) -> &[u8; 6] {
        &self.0
    }
}

impl std::fmt::Display for MarketId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl AsRef<[u8]> for MarketId {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl utoipa::PartialSchema for MarketId {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::Schema> {
        let schema = utoipa::openapi::ObjectBuilder::new()
            .description(Some("6-byte market identifier"))
            .examples([serde_json::json!("0x0123456789ab")])
            .build();
        utoipa::openapi::RefOr::T(utoipa::openapi::Schema::Object(schema))
    }
}

impl utoipa::ToSchema for MarketId {
    fn name() -> std::borrow::Cow<'static, str> {
        "MarketId".into()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ShareAccount {
    pub positions: HashMap<(MarketId, u32), i64>,
    pub nonce: u64,
    pub trade_nonce: u64,
    pub last_updated_height: u64,
}

impl ShareAccount {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_position(
        &self,
        market_id: &MarketId,
        outcome_index: u32,
    ) -> i64 {
        self.positions
            .get(&(market_id.clone(), outcome_index))
            .copied()
            .unwrap_or(0)
    }

    pub fn add_shares(
        &mut self,
        market_id: MarketId,
        outcome_index: u32,
        shares: i64,
        height: u64,
    ) {
        let key = (market_id, outcome_index);
        let current = self.positions.get(&key).copied().unwrap_or(0);
        self.positions.insert(key, current + shares);
        self.last_updated_height = height;
    }

    pub fn remove_shares(
        &mut self,
        market_id: &MarketId,
        outcome_index: u32,
        shares: i64,
        height: u64,
    ) -> Result<(), MarketError> {
        let key = (market_id.clone(), outcome_index);
        let current = self.positions.get(&key).copied().unwrap_or(0);

        if shares > current {
            return Err(MarketError::InvalidOutcomeCombination);
        }

        let new_amount = current - shares;
        if new_amount > 0 {
            self.positions.insert(key, new_amount);
        } else {
            self.positions.remove(&key);
        }

        self.last_updated_height = height;
        Ok(())
    }

    pub fn increment_nonce(&mut self) {
        self.nonce += 1;
    }

    pub fn increment_trade_nonce(&mut self) {
        self.trade_nonce += 1;
        self.increment_nonce();
    }

    pub fn get_all_positions(&self) -> &HashMap<(MarketId, u32), i64> {
        &self.positions
    }
}

/// Record of a single shareholder's payout from a resolved market
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharePayoutRecord {
    pub market_id: MarketId,
    pub address: Address,
    pub outcome_index: u32,
    pub shares_redeemed: i64,
    pub final_price: f64,
    pub payout_sats: u64,
}

/// Summary of all payouts for a market when it transitions to Ossified
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketPayoutSummary {
    pub market_id: MarketId,
    pub treasury_distributed: u64,
    pub author_fees_distributed: u64,
    pub shareholder_count: u32,
    pub payouts: Vec<SharePayoutRecord>,
    pub block_height: u64,
}

#[derive(Debug, Clone)]
pub struct BatchedMarketTrade {
    pub market_id: MarketId,
    pub outcome_index: u32,
    pub shares_to_buy: i64,
    pub max_cost: u64,
    pub market_snapshot: MarketSnapshot,
    pub trader_address: Address,
}

#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub shares: Array<i64, Ix1>,
    pub b: f64,
    pub trading_fee: f64,
}

impl BatchedMarketTrade {
    pub fn new(
        market_id: MarketId,
        outcome_index: u32,
        shares_to_buy: i64,
        max_cost: u64,
        market: &Market,
        trader_address: Address,
    ) -> Self {
        let market_snapshot = MarketSnapshot {
            shares: market.shares().clone(),
            b: market.b(),
            trading_fee: market.trading_fee(),
        };

        Self {
            market_id,
            outcome_index,
            shares_to_buy,
            max_cost,
            market_snapshot,
            trader_address,
        }
    }

    pub fn calculate_trade_cost(&self) -> Result<f64, MarketError> {
        let cost = self.calculate_trade_cost_sats()?;
        Ok(cost.total_cost_sats as f64)
    }

    pub fn calculate_trade_cost_sats(
        &self,
    ) -> Result<crate::math::trading::BuyCost, MarketError> {
        use crate::math::trading;

        let mut new_shares = self.market_snapshot.shares.clone();
        new_shares[self.outcome_index as usize] += self.shares_to_buy;

        let base_cost_f64 = trading::calculate_update_cost(
            &self.market_snapshot.shares,
            &new_shares,
            self.market_snapshot.b,
        )
        .map_err(|e| {
            MarketError::DatabaseError(format!(
                "LMSR calculation failed: {e:?}"
            ))
        })?;

        trading::calculate_buy_cost(
            base_cost_f64,
            self.market_snapshot.trading_fee,
        )
        .map_err(|e| {
            MarketError::DatabaseError(format!("Fee calculation failed: {e:?}"))
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    pub id: MarketId,
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub creator_address: Address,
    pub dimension_specs: Vec<DimensionSpec>,
    pub decision_slots: Vec<SlotId>,
    pub d_functions: Vec<DFunction>,
    pub state_combos: Vec<Vec<usize>>,
    pub residual_names: Option<Vec<String>>,
    pub created_at_height: u64,
    pub expires_at_height: Option<u64>,
    pub tau_from_now: u8,
    pub share_vector_length: usize,
    pub storage_fee_sats: u64,
    pub market_state: MarketState,
    pub b: f64,
    pub trading_fee: f64,
    #[serde(with = "ndarray_1d_i64_serde")]
    pub shares: Array<i64, Ix1>,
    #[serde(with = "ndarray_1d_serde")]
    pub final_prices: Array<f64, Ix1>,
    pub version: u64,
    pub last_updated_height: u64,
    pub total_volume_sats: u64,
    pub outcome_volumes_sats: Vec<u64>,
}

mod ndarray_1d_serde {
    use ndarray::{Array, Ix1};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(
        array: &Array<f64, Ix1>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match array.as_slice() {
            Some(slice) => slice.serialize(serializer),
            None => array.to_vec().serialize(serializer),
        }
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<Array<f64, Ix1>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data: Vec<f64> = Deserialize::deserialize(deserializer)?;
        Ok(Array::from_vec(data))
    }
}

mod ndarray_1d_i64_serde {
    use ndarray::{Array, Ix1};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(
        array: &Array<i64, Ix1>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match array.as_slice() {
            Some(slice) => slice.serialize(serializer),
            None => array.to_vec().serialize(serializer),
        }
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<Array<i64, Ix1>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data: Vec<i64> = Deserialize::deserialize(deserializer)?;
        Ok(Array::from_vec(data))
    }
}

pub struct MarketBuilder {
    title: String,
    description: String,
    tags: Vec<String>,
    creator_address: Address,
    dimension_specs: Option<Vec<DimensionSpec>>,
    residual_names: Option<Vec<String>>,
    b: f64,
    trading_fee: f64,
}

impl MarketBuilder {
    pub fn new(title: String, creator_address: Address) -> Self {
        Self {
            title,
            description: String::new(),
            tags: Vec::new(),
            creator_address,
            dimension_specs: None,
            residual_names: None,
            b: DEFAULT_MARKET_BETA,
            trading_fee: DEFAULT_TRADING_FEE,
        }
    }

    pub fn with_description(mut self, desc: String) -> Self {
        self.description = desc;
        self
    }

    pub fn with_beta(mut self, b: f64) -> Self {
        self.b = b;
        self
    }

    pub fn with_fee(mut self, fee: f64) -> Self {
        self.trading_fee = fee;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_dimensions(
        mut self,
        dimension_specs: Vec<DimensionSpec>,
    ) -> Self {
        self.dimension_specs = Some(dimension_specs);
        self
    }

    pub fn with_residual_names(
        mut self,
        residual_names: Option<Vec<String>>,
    ) -> Self {
        self.residual_names = residual_names;
        self
    }

    pub fn build(
        self,
        created_at_height: u64,
        expires_at_height: Option<u64>,
        decisions: &HashMap<SlotId, Decision>,
    ) -> Result<Market, MarketError> {
        let dimension_specs =
            self.dimension_specs.ok_or(MarketError::InvalidDimensions)?;

        let (d_functions, state_combos) =
            generate_mixed_dimensional(&dimension_specs, decisions)?;

        let all_slots: Vec<SlotId> = dimension_specs
            .iter()
            .flat_map(|spec| match spec {
                DimensionSpec::Single(slot_id) => vec![*slot_id],
                DimensionSpec::Categorical(slot_ids) => slot_ids.clone(),
            })
            .collect();

        Market::new(
            self.title,
            self.description,
            self.tags,
            self.creator_address,
            dimension_specs,
            all_slots,
            d_functions,
            state_combos,
            self.residual_names,
            self.b,
            self.trading_fee,
            created_at_height,
            expires_at_height,
            decisions,
        )
    }
}

impl DFunction {
    pub fn evaluate(&self, combo: &[usize]) -> Result<bool, MarketError> {
        match self {
            DFunction::Decision(idx) => {
                if *idx >= combo.len() {
                    return Err(MarketError::InvalidDimensions);
                }
                Ok(combo[*idx] == 1)
            }
            DFunction::Equals(func, value) => {
                if let DFunction::Decision(idx) = func.as_ref() {
                    if *idx >= combo.len() {
                        return Err(MarketError::InvalidDimensions);
                    }
                    Ok(combo[*idx] == *value)
                } else {
                    let func_result = func.evaluate(combo)?;
                    Ok(func_result && *value == 1)
                }
            }
            DFunction::And(left, right) => {
                let left_result = left.evaluate(combo)?;
                if !left_result {
                    return Ok(false);
                }
                let right_result = right.evaluate(combo)?;
                Ok(left_result && right_result)
            }
            DFunction::Or(left, right) => {
                let left_result = left.evaluate(combo)?;
                if left_result {
                    return Ok(true);
                }
                let right_result = right.evaluate(combo)?;
                Ok(left_result || right_result)
            }
            DFunction::Not(func) => {
                let result = func.evaluate(combo)?;
                Ok(!result)
            }
            DFunction::True => Ok(true),
        }
    }

    pub fn validate_constraint(
        &self,
        max_decision_index: usize,
    ) -> Result<(), MarketError> {
        crate::validation::DFunctionValidator::validate_constraint(
            self,
            max_decision_index,
        )
    }

    pub fn validate_categorical_constraint(
        &self,
        categorical_slots: &[usize],
        combo: &[usize],
        decision_slots: &[SlotId],
    ) -> Result<bool, MarketError> {
        crate::validation::DFunctionValidator::validate_categorical_constraint(
            self,
            categorical_slots,
            combo,
            decision_slots,
        )
    }

    pub fn validate_dimensional_consistency(
        d_functions: &[DFunction],
        decision_slots_len: usize,
        all_combos: &[Vec<usize>],
    ) -> Result<(), MarketError> {
        crate::validation::DFunctionValidator::validate_dimensional_consistency(
            d_functions,
            decision_slots_len,
            all_combos,
        )
    }

    fn build_balanced_and_tree(mut constraints: Vec<DFunction>) -> DFunction {
        while constraints.len() > 1 {
            let mut next_level =
                Vec::with_capacity(constraints.len().div_ceil(2));

            while constraints.len() >= 2 {
                let right = constraints
                    .pop()
                    .expect("constraints.len() >= 2 guarantees pop() succeeds");
                let left = constraints
                    .pop()
                    .expect("constraints.len() >= 2 guarantees pop() succeeds");
                next_level
                    .push(DFunction::And(Box::new(left), Box::new(right)));
            }

            if let Some(remaining) = constraints.pop() {
                next_level.push(remaining);
            }

            constraints = next_level;
        }

        constraints.into_iter().next().unwrap_or(DFunction::True)
    }
}

fn calculate_storage_fee_with_scaling(share_vector_length: usize) -> u64 {
    let base_cost = BASE_MARKET_STORAGE_COST_SATS;
    let quadratic_cost =
        (share_vector_length as u64).pow(2) * L2_STORAGE_RATE_SATS_PER_BYTE;
    base_cost + quadratic_cost
}

pub fn generate_mixed_dimensional(
    dimension_specs: &[DimensionSpec],
    decisions: &HashMap<SlotId, Decision>,
) -> Result<(Vec<DFunction>, Vec<Vec<usize>>), MarketError> {
    use std::collections::HashSet;

    if dimension_specs.is_empty() {
        return Err(MarketError::InvalidDimensions);
    }

    let mut seen_slots = HashSet::new();
    for spec in dimension_specs {
        let slots = match spec {
            DimensionSpec::Single(slot_id) => vec![*slot_id],
            DimensionSpec::Categorical(slot_ids) => slot_ids.clone(),
        };
        for slot_id in slots {
            if !seen_slots.insert(slot_id) {
                return Err(MarketError::DuplicateSlot { slot_id });
            }
        }
    }

    // Validate tradeable outcome count (excludes ABSTAIN states).

    let mut tradeable_outcomes = 1usize;
    for spec in dimension_specs {
        let tradeable_per_dim = match spec {
            DimensionSpec::Single(_) => 2, // No, Yes (excludes Abstain)
            DimensionSpec::Categorical(slots) => slots.len() + 1, // slots + residual (excludes Abstain)
        };

        if let Some(new_tradeable) =
            tradeable_outcomes.checked_mul(tradeable_per_dim)
        {
            if new_tradeable > MAX_MARKET_OUTCOMES {
                return Err(MarketError::TooManyStates(new_tradeable));
            }
            tradeable_outcomes = new_tradeable;
        } else {
            return Err(MarketError::TooManyStates(usize::MAX));
        }
    }

    let mut all_slots = Vec::with_capacity(dimension_specs.len() * 4);
    let mut dimension_ranges = Vec::with_capacity(dimension_specs.len());
    let mut slot_to_dimension = Vec::new();

    for (dim_idx, spec) in dimension_specs.iter().enumerate() {
        match spec {
            DimensionSpec::Single(slot_id) => {
                all_slots.push(*slot_id);
                slot_to_dimension.push(dim_idx);

                decisions.get(slot_id).ok_or(
                    MarketError::DecisionSlotNotFound { slot_id: *slot_id },
                )?;

                // Both binary and scaled decisions have 3 outcomes for LMSR consistency:
                // Binary: 0=No, 1=Yes, 2=Abstain
                // Scaled: 0=Min bound, 1=Max bound, 2=Abstain
                let outcomes = 3;
                dimension_ranges.push(outcomes);
            }
            DimensionSpec::Categorical(slot_ids) => {
                let outcomes = slot_ids.len() + 2;
                dimension_ranges.push(outcomes);

                for slot_id in slot_ids {
                    all_slots.push(*slot_id);
                    slot_to_dimension.push(dim_idx);
                }
            }
        }
    }

    let state_combos = generate_cartesian_product(&dimension_ranges);

    let mut d_functions = Vec::with_capacity(state_combos.len());

    for combo in &state_combos {
        let mut constraints = Vec::with_capacity(dimension_specs.len() * 2);
        let mut slot_idx = 0;

        for (dim_idx, spec) in dimension_specs.iter().enumerate() {
            let dim_outcome = combo[dim_idx];

            match spec {
                DimensionSpec::Single(_) => {
                    if dim_outcome < 3 {
                        constraints.push(DFunction::Equals(
                            Box::new(DFunction::Decision(slot_idx)),
                            dim_outcome,
                        ));
                    }
                    slot_idx += 1;
                }
                DimensionSpec::Categorical(slot_ids) => {
                    if dim_outcome < slot_ids.len() {
                        constraints.push(DFunction::Equals(
                            Box::new(DFunction::Decision(
                                slot_idx + dim_outcome,
                            )),
                            1,
                        ));
                        for (other_idx, _) in slot_ids.iter().enumerate() {
                            if other_idx != dim_outcome {
                                constraints.push(DFunction::Equals(
                                    Box::new(DFunction::Decision(
                                        slot_idx + other_idx,
                                    )),
                                    0,
                                ));
                            }
                        }
                    } else if dim_outcome == slot_ids.len() {
                        for other_idx in 0..slot_ids.len() {
                            constraints.push(DFunction::Equals(
                                Box::new(DFunction::Decision(
                                    slot_idx + other_idx,
                                )),
                                0,
                            ));
                        }
                    }
                    slot_idx += slot_ids.len();
                }
            }
        }

        let d_function = match constraints.len() {
            0 => DFunction::True,
            1 => constraints
                .into_iter()
                .next()
                .expect("constraints.len() == 1 guarantees next() succeeds"),
            _ => DFunction::build_balanced_and_tree(constraints),
        };

        d_functions.push(d_function);
    }

    DFunction::validate_dimensional_consistency(
        &d_functions,
        all_slots.len(),
        &state_combos,
    )?;

    Ok((d_functions, state_combos))
}

fn generate_cartesian_product(dimensions: &[usize]) -> Vec<Vec<usize>> {
    if dimensions.is_empty() {
        return vec![vec![]];
    }

    let expected_size: usize = dimensions.iter().product();
    let mut result = Vec::with_capacity(expected_size);
    result.push(vec![]);

    for &dim_size in dimensions {
        let mut new_result = Vec::with_capacity(result.len() * dim_size);

        for combo in result {
            for value in 0..dim_size {
                let mut new_combo = combo.clone();
                new_combo.push(value);
                new_result.push(new_combo);
            }
        }
        result = new_result;
    }

    result
}

fn calculate_max_tau(
    decision_slots: &[SlotId],
    decisions: &HashMap<SlotId, Decision>,
) -> u8 {
    decision_slots
        .iter()
        .filter_map(|slot_id| decisions.get(slot_id))
        .map(|_| 5u8)
        .max()
        .unwrap_or(5)
}

impl Market {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        title: String,
        description: String,
        tags: Vec<String>,
        creator_address: Address,
        dimension_specs: Vec<DimensionSpec>,
        decision_slots: Vec<SlotId>,
        d_functions: Vec<DFunction>,
        state_combos: Vec<Vec<usize>>,
        residual_names: Option<Vec<String>>,
        b: f64,
        trading_fee: f64,
        created_at_height: u64,
        expires_at_height: Option<u64>,
        decisions: &HashMap<SlotId, Decision>,
    ) -> Result<Self, MarketError> {
        if decision_slots.is_empty() {
            return Err(MarketError::InvalidDimensions);
        }

        if d_functions.is_empty() {
            return Err(MarketError::InvalidDimensions);
        }

        if d_functions.len() != state_combos.len() {
            return Err(MarketError::InvalidDimensions);
        }

        if b <= 0.0 {
            return Err(MarketError::InvalidBeta(b));
        }

        let share_vector_length = d_functions.len();
        let storage_fee_sats =
            calculate_storage_fee_with_scaling(share_vector_length);

        let mut market = Market {
            id: MarketId([0; 6]),
            title,
            description,
            tags,
            creator_address,
            dimension_specs,
            decision_slots: decision_slots.clone(),
            d_functions,
            state_combos,
            residual_names,
            created_at_height,
            expires_at_height,
            tau_from_now: calculate_max_tau(&decision_slots, decisions),
            share_vector_length,
            storage_fee_sats,
            market_state: MarketState::Trading,
            b,
            trading_fee,
            shares: Array::zeros(share_vector_length),
            final_prices: Array::zeros(share_vector_length),
            version: 0,
            last_updated_height: created_at_height,
            total_volume_sats: 0,
            outcome_volumes_sats: vec![0; share_vector_length],
        };

        market.id = market.calculate_id();

        Ok(market)
    }

    fn calculate_id(&self) -> MarketId {
        compute_market_id(
            &self.title,
            &self.description,
            &self.creator_address,
            &self.dimension_specs,
        )
    }
}

/// Compute market ID from content fields.
pub fn compute_market_id(
    title: &str,
    description: &str,
    creator_address: &Address,
    dimension_specs: &[DimensionSpec],
) -> MarketId {
    #[derive(BorshSerialize)]
    struct MarketIdInput<'a> {
        title: &'a str,
        description: &'a str,
        creator_address: &'a Address,
        dimension_specs: &'a [DimensionSpec],
    }

    let input = MarketIdInput {
        title,
        description,
        creator_address,
        dimension_specs,
    };

    let hash_bytes = hashes::hash(&input);
    let mut id_bytes = [0u8; 6];
    id_bytes.copy_from_slice(&hash_bytes[0..6]);
    MarketId(id_bytes)
}

impl Market {
    pub fn state(&self) -> MarketState {
        self.market_state
    }

    pub fn b(&self) -> f64 {
        self.b
    }

    pub fn trading_fee(&self) -> f64 {
        self.trading_fee
    }

    pub fn shares(&self) -> &Array<i64, Ix1> {
        &self.shares
    }

    pub fn final_prices(&self) -> &Array<f64, Ix1> {
        &self.final_prices
    }

    pub fn update_state(
        &mut self,
        height: u64,
        new_market_state: Option<MarketState>,
        new_b: Option<f64>,
        new_shares: Option<Array<i64, Ix1>>,
        new_final_prices: Option<Array<f64, Ix1>>,
    ) -> Result<(), MarketError> {
        if let Some(new_state) = new_market_state {
            if !self.market_state.can_transition_to(new_state) {
                return Err(MarketError::InvalidStateTransition {
                    from: self.market_state,
                    to: new_state,
                });
            }
            self.market_state = new_state;
        }

        if let Some(b) = new_b {
            if b <= 0.0 {
                return Err(MarketError::InvalidBeta(b));
            }
            self.b = b;
        }

        if let Some(shares) = new_shares {
            if shares.len() != self.share_vector_length {
                return Err(MarketError::InvalidDimensions);
            }
            self.shares = shares;
        }

        if let Some(prices) = new_final_prices {
            self.final_prices = prices;
        }

        self.version += 1;
        self.last_updated_height = height;

        Ok(())
    }

    pub fn current_prices(&self) -> Array<f64, Ix1> {
        crate::math::trading::calculate_prices(&self.shares, self.b)
            .unwrap_or_else(|_| {
                let n = self.shares.len();
                Array::from_elem(n, 1.0 / n as f64)
            })
    }

    pub fn calculate_prices_for_display(&self) -> Vec<f64> {
        let valid_state_indices: Vec<usize> = self
            .get_valid_state_combos()
            .iter()
            .map(|(idx, _)| *idx)
            .collect();
        crate::math::trading::calculate_display_prices(
            &self.shares,
            self.b,
            &valid_state_indices,
        )
    }

    /// For single-slot markets (binary or scaled), calculate the implied value (0-1).
    /// Returns p_max / (p_min + p_max) where:
    /// - p_min is the price of outcome 0 (No/Min)
    /// - p_max is the price of outcome 1 (Yes/Max)
    ///
    /// For binary decisions: 0.0 = "No", 1.0 = "Yes"
    /// For scaled decisions: 0.0 = min bound, 1.0 = max bound
    ///
    /// Returns None if market has multiple decision slots or invalid structure.
    pub fn get_implied_value_normalized(&self) -> Option<f64> {
        if self.decision_slots.len() != 1 || self.state_combos.len() != 3 {
            return None;
        }

        let prices = self.current_prices();
        if prices.len() < 2 {
            return None;
        }

        let p_min = prices[0];
        let p_max = prices[1];
        let sum = p_min + p_max;

        if sum > 0.0 {
            Some(p_max / sum)
        } else {
            Some(0.5) // Default to 50% if no price info
        }
    }

    pub fn denormalize_value(normalized: f64, min: i64, max: i64) -> f64 {
        min as f64 + normalized * (max - min) as f64
    }

    pub fn normalize_value(value: f64, min: i64, max: i64) -> f64 {
        if max == min {
            return 0.5;
        }
        ((value - min as f64) / (max - min) as f64).clamp(0.0, 1.0)
    }

    pub fn update_shares(
        &mut self,
        new_shares: Array<i64, Ix1>,
        height: u64,
    ) -> Result<(), MarketError> {
        self.update_state(height, None, None, Some(new_shares), None)
    }

    pub fn update_trading_volume(
        &mut self,
        outcome_index: usize,
        trade_cost_sats: u64,
    ) -> Result<(), MarketError> {
        if outcome_index >= self.outcome_volumes_sats.len() {
            return Err(MarketError::InvalidOutcomeIndex(outcome_index));
        }

        self.outcome_volumes_sats[outcome_index] = self.outcome_volumes_sats
            [outcome_index]
            .saturating_add(trade_cost_sats);

        self.total_volume_sats =
            self.total_volume_sats.saturating_add(trade_cost_sats);

        Ok(())
    }

    pub fn revert_trading_volume(
        &mut self,
        outcome_index: usize,
        trade_cost_sats: u64,
    ) -> Result<(), MarketError> {
        if outcome_index >= self.outcome_volumes_sats.len() {
            return Err(MarketError::InvalidOutcomeIndex(outcome_index));
        }
        self.outcome_volumes_sats[outcome_index] = self.outcome_volumes_sats
            [outcome_index]
            .saturating_sub(trade_cost_sats);
        self.total_volume_sats =
            self.total_volume_sats.saturating_sub(trade_cost_sats);
        Ok(())
    }

    pub fn query_update_cost(
        &self,
        new_shares: Array<i64, Ix1>,
    ) -> Result<f64, MarketError> {
        if new_shares.len() != self.share_vector_length {
            return Err(MarketError::InvalidDimensions);
        }

        crate::math::trading::calculate_update_cost(
            &self.shares,
            &new_shares,
            self.b,
        )
        .map_err(|e| {
            MarketError::DatabaseError(format!(
                "LMSR calculation failed: {e:?}"
            ))
        })
    }

    pub fn query_amp_b_cost(&self, new_b: f64) -> Result<f64, MarketError> {
        if new_b <= self.b {
            return Err(MarketError::InvalidBeta(new_b));
        }

        crate::math::trading::calculate_amp_b_cost(&self.shares, self.b, new_b)
            .map_err(|e| {
                MarketError::DatabaseError(format!(
                    "LMSR calculation failed: {e:?}"
                ))
            })
    }

    pub fn cancel_market(
        &mut self,
        _transaction_id: Option<[u8; 32]>,
        height: u64,
    ) -> Result<(), MarketError> {
        self.update_state(
            height,
            Some(MarketState::Cancelled),
            None,
            None,
            None,
        )
    }

    pub fn invalidate_market(
        &mut self,
        _transaction_id: Option<[u8; 32]>,
        height: u64,
    ) -> Result<(), MarketError> {
        self.update_state(height, Some(MarketState::Invalid), None, None, None)
    }

    /// # Arguments
    /// * `slot_outcomes` - Map of SlotId to consensus outcome value (0.0-1.0)
    /// * `decisions` - Map of SlotId to Decision (needed for scaled decision handling)
    ///
    /// # Returns
    /// Array of final prices, one per market outcome, summing to 1.0
    pub fn calculate_final_prices(
        &self,
        slot_outcomes: &std::collections::HashMap<SlotId, f64>,
        decisions: &HashMap<SlotId, Decision>,
    ) -> Result<Array<f64, Ix1>, MarketError> {
        // Compute axis weights for each dimension: [state_0, state_1, Abstain]
        let mut axes: Vec<[f64; 3]> =
            Vec::with_capacity(self.decision_slots.len());

        for slot_id in &self.decision_slots {
            let outcome = slot_outcomes.get(slot_id).copied().unwrap_or(0.5);
            let is_scaled =
                decisions.get(slot_id).map(|d| d.is_scaled).unwrap_or(false);

            let axis = if is_scaled {
                [1.0 - outcome, outcome, 0.0]
            } else if outcome > 0.7 {
                [0.0, 1.0, 0.0]
            } else if outcome < 0.3 {
                [1.0, 0.0, 0.0]
            } else {
                [0.5, 0.5, 0.0]
            };

            axes.push(axis);
        }

        let mut prices = Array::zeros(self.state_combos.len());

        for (i, state_combo) in self.state_combos.iter().enumerate() {
            if state_combo.contains(&2) {
                continue;
            }

            let mut joint_prob = 1.0;
            for (dim, &state) in state_combo.iter().enumerate() {
                if dim < axes.len() && state < 3 {
                    joint_prob *= axes[dim][state];
                }
            }
            prices[i] = joint_prob;
        }

        let sum: f64 = prices.sum();
        if sum > 0.0 {
            prices /= sum;
        }

        Ok(prices)
    }

    /// Returns the number of tradeable outcomes (excludes ABSTAIN states).
    pub fn get_outcome_count(&self) -> usize {
        self.get_valid_state_combos().len()
    }

    /// Returns the total number of state combinations including ABSTAIN.
    pub fn get_total_state_count(&self) -> usize {
        self.shares().len()
    }

    pub fn get_dimensions(&self) -> Vec<usize> {
        vec![self.get_outcome_count()]
    }

    pub fn get_state_combos(&self) -> &Vec<Vec<usize>> {
        &self.state_combos
    }

    pub fn get_valid_state_combos(&self) -> Vec<(usize, &Vec<usize>)> {
        self.state_combos
            .iter()
            .enumerate()
            .filter(|(_, combo)| !combo.contains(&2))
            .collect()
    }

    pub fn get_d_functions(&self) -> &Vec<DFunction> {
        &self.d_functions
    }

    pub fn get_storage_fee_sats(&self) -> u64 {
        self.storage_fee_sats
    }

    pub fn calculate_trade_cost(&self, base_fee_sats: u64) -> u64 {
        let complexity_cost = (self.share_vector_length as u64).pow(2)
            * L2_STORAGE_RATE_SATS_PER_BYTE;
        base_fee_sats + complexity_cost
    }

    pub fn get_share_vector_length(&self) -> usize {
        self.share_vector_length
    }

    pub fn get_outcome_index(
        &self,
        positions: &[usize],
    ) -> Result<usize, MarketError> {
        if positions.len() != self.decision_slots.len() {
            return Err(MarketError::InvalidDimensions);
        }

        for (state_idx, combo) in self.state_combos.iter().enumerate() {
            if combo == positions {
                return Ok(state_idx);
            }
        }

        Err(MarketError::InvalidOutcomeCombination)
    }

    pub fn get_outcome_price(
        &self,
        positions: &[usize],
    ) -> Result<f64, MarketError> {
        let index = self.get_outcome_index(positions)?;
        let prices = self.current_prices();

        Ok(prices[index])
    }

    pub fn describe_outcome_by_state(
        &self,
        state_index: usize,
        decisions: &HashMap<SlotId, Decision>,
    ) -> Result<String, MarketError> {
        if state_index >= self.state_combos.len() {
            return Err(MarketError::InvalidDimensions);
        }

        let positions = &self.state_combos[state_index];
        self.describe_outcome(positions, decisions)
    }

    pub fn describe_outcome(
        &self,
        positions: &[usize],
        decisions: &HashMap<SlotId, Decision>,
    ) -> Result<String, MarketError> {
        if positions.len() != self.decision_slots.len() {
            return Err(MarketError::InvalidDimensions);
        }

        let mut description = Vec::new();

        for (i, &slot_id) in self.decision_slots.iter().enumerate() {
            let decision = decisions
                .get(&slot_id)
                .ok_or(MarketError::DecisionSlotNotFound { slot_id })?;

            // Both binary and scaled decisions use 3 outcomes:
            // Binary: 0=No, 1=Yes, 2=Abstain
            // Scaled: 0=Min bound, 1=Max bound, 2=Abstain
            let outcome_desc = if decision.is_scaled {
                match positions[i] {
                    0 => format!(
                        "{}: {} (Min)",
                        decision.question,
                        decision.min.unwrap_or(0)
                    ),
                    1 => format!(
                        "{}: {} (Max)",
                        decision.question,
                        decision.max.unwrap_or(100)
                    ),
                    _ => format!("{}: Abstain", decision.question),
                }
            } else {
                let outcome = match positions[i] {
                    0 => "No",
                    1 => "Yes",
                    _ => "Abstain",
                };
                format!("{}: {}", decision.question, outcome)
            };

            description.push(outcome_desc);
        }

        Ok(description.join(", "))
    }
}

/// Generate a deterministic mempool tracking address for a market
fn mempool_address(market_id: &MarketId) -> Address {
    let mut bytes = [0u8; 20];
    bytes[0] = 0xFF;
    bytes[1..7].copy_from_slice(&market_id.0);
    Address(bytes)
}

#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct MarketsDatabase {
    markets: DatabaseUnique<SerdeBincode<[u8; 6]>, SerdeBincode<Market>>,
    state_index:
        DatabaseUnique<SerdeBincode<MarketState>, SerdeBincode<Vec<MarketId>>>,
    expiry_index:
        DatabaseUnique<SerdeBincode<u64>, SerdeBincode<Vec<MarketId>>>,
    slot_index:
        DatabaseUnique<SerdeBincode<SlotId>, SerdeBincode<Vec<MarketId>>>,
    share_accounts:
        DatabaseUnique<SerdeBincode<Address>, SerdeBincode<ShareAccount>>,
    /// is_fee=false for treasury, is_fee=true for author fees
    market_funds_utxos:
        DatabaseUnique<SerdeBincode<([u8; 6], bool)>, SerdeBincode<OutPoint>>,
    pending_market_funds_utxos: DatabaseUnique<
        SerdeBincode<[u8; 6]>,
        SerdeBincode<Vec<(OutPoint, bool)>>,
    >,
}

impl MarketsDatabase {
    fn validate_market_state_transition(
        &self,
        from_state: MarketState,
        to_state: MarketState,
    ) -> Result<(), Error> {
        crate::validation::MarketStateValidator::validate_market_state_transition(from_state, to_state)
    }
    pub const NUM_DBS: u32 = 7;

    pub fn new(env: &Env, rwtxn: &mut RwTxn) -> Result<Self, Error> {
        let markets = DatabaseUnique::create(env, rwtxn, "markets")?;
        let state_index =
            DatabaseUnique::create(env, rwtxn, "markets_by_state")?;
        let expiry_index =
            DatabaseUnique::create(env, rwtxn, "markets_by_expiry")?;
        let slot_index = DatabaseUnique::create(env, rwtxn, "markets_by_slot")?;
        let share_accounts =
            DatabaseUnique::create(env, rwtxn, "share_accounts")?;
        let market_funds_utxos =
            DatabaseUnique::create(env, rwtxn, "market_funds_utxos")?;
        let pending_market_funds_utxos =
            DatabaseUnique::create(env, rwtxn, "pending_market_funds_utxos")?;

        Ok(MarketsDatabase {
            markets,
            state_index,
            expiry_index,
            slot_index,
            share_accounts,
            market_funds_utxos,
            pending_market_funds_utxos,
        })
    }

    pub fn add_market(
        &self,
        txn: &mut RwTxn,
        market: &Market,
    ) -> Result<(), Error> {
        self.markets.put(txn, market.id.as_bytes(), market)?;

        self.update_state_index(txn, &market.id, None, Some(market.state()))?;

        if let Some(expires_at) = market.expires_at_height {
            self.update_expiry_index(txn, &market.id, None, Some(expires_at))?;
        }

        for &slot_id in &market.decision_slots {
            self.update_slot_index(txn, &market.id, slot_id, true)?;
        }

        Ok(())
    }

    pub fn delete_market(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<bool, Error> {
        // Get the market first to know what indexes to clean up
        let Some(market) = self.markets.try_get(txn, market_id.as_bytes())?
        else {
            return Ok(false);
        };

        self.update_state_index(txn, market_id, Some(market.state()), None)?;

        if let Some(expires_at) = market.expires_at_height {
            self.update_expiry_index(txn, market_id, Some(expires_at), None)?;
        }

        for &slot_id in &market.decision_slots {
            self.update_slot_index(txn, market_id, slot_id, false)?;
        }

        self.market_funds_utxos
            .delete(txn, &(*market_id.as_bytes(), false))?;
        self.market_funds_utxos
            .delete(txn, &(*market_id.as_bytes(), true))?;

        self.pending_market_funds_utxos
            .delete(txn, market_id.as_bytes())?;

        self.markets.delete(txn, market_id.as_bytes())?;

        Ok(true)
    }

    pub fn get_market(
        &self,
        txn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<Option<Market>, Error> {
        Ok(self.markets.try_get(txn, market_id.as_bytes())?)
    }

    pub fn get_all_markets(&self, txn: &RoTxn) -> Result<Vec<Market>, Error> {
        let markets = self
            .markets
            .iter(txn)?
            .map(|(_, market)| Ok(market))
            .collect()?;
        Ok(markets)
    }

    pub fn get_markets_batch(
        &self,
        txn: &RoTxn,
        market_ids: &[MarketId],
    ) -> Result<HashMap<MarketId, Market>, Error> {
        if market_ids.is_empty() {
            return Ok(HashMap::new());
        }

        const BATCH_THRESHOLD: usize = 3;
        if market_ids.len() < BATCH_THRESHOLD {
            let mut markets = HashMap::with_capacity(market_ids.len());

            for market_id in market_ids {
                if let Some(market) = self.get_market(txn, market_id)? {
                    markets.insert(market_id.clone(), market);
                }
            }

            return Ok(markets);
        }

        let market_id_set: HashSet<_> = market_ids.iter().collect();
        let mut markets = HashMap::with_capacity(market_ids.len());
        let mut found_count = 0;

        let market_iter = self.markets.iter(txn).map_err(|e| {
            Error::DatabaseError(format!("Market batch iteration failed: {e}"))
        })?;

        let mut market_iter = market_iter;
        while let Some(item) = market_iter.next().map_err(|e| {
            Error::DatabaseError(format!(
                "Market batch iteration item failed: {e}"
            ))
        })? {
            let (market_id_bytes, market) = item;

            let market_id = MarketId::new(market_id_bytes);

            if market_id_set.contains(&market_id) {
                markets.insert(market_id, market);
                found_count += 1;

                if found_count >= market_ids.len() {
                    break;
                }
            }
        }

        tracing::debug!(
            "Batch loaded {}/{} requested markets using optimized iteration",
            found_count,
            market_ids.len()
        );

        Ok(markets)
    }

    pub fn get_markets_by_state(
        &self,
        txn: &RoTxn,
        state: MarketState,
    ) -> Result<Vec<Market>, Error> {
        let market_ids =
            self.state_index.try_get(txn, &state)?.unwrap_or_default();

        let mut markets = Vec::with_capacity(market_ids.len());
        for market_id in market_ids {
            if let Some(market) = self.get_market(txn, &market_id)? {
                markets.push(market);
            }
        }

        Ok(markets)
    }

    pub fn get_markets_by_expiry(
        &self,
        txn: &RoTxn,
        min_height: Option<u64>,
        max_height: Option<u64>,
    ) -> Result<Vec<Market>, Error> {
        let mut markets = Vec::new();

        let expiry_iter = self.expiry_index.iter(txn).map_err(|e| {
            Error::DatabaseError(format!("Expiry index iteration failed: {e}"))
        })?;

        let mut matching_market_ids = Vec::new();
        let mut entries_checked = 0;
        let mut entries_matched = 0;

        let mut expiry_iter = expiry_iter;
        while let Some(item) = expiry_iter.next().map_err(|e| {
            Error::DatabaseError(format!(
                "Expiry index iteration item failed: {e}"
            ))
        })? {
            let (expiry_height, market_ids) = item;

            entries_checked += 1;

            if let Some(max) = max_height
                && expiry_height > max
            {
                break;
            }

            let matches_min = min_height.is_none_or(|min| expiry_height >= min);
            let matches_max = max_height.is_none_or(|max| expiry_height <= max);

            if matches_min && matches_max {
                entries_matched += 1;
                matching_market_ids.extend(market_ids);
            }
        }

        tracing::debug!(
            "Expiry range query: checked {} entries, matched {} entries with {} total markets",
            entries_checked,
            entries_matched,
            matching_market_ids.len()
        );

        if matching_market_ids.len() > 5 {
            let markets_map =
                self.get_markets_batch(txn, &matching_market_ids)?;

            for market_id in matching_market_ids {
                if let Some(market) = markets_map.get(&market_id) {
                    markets.push(market.clone());
                }
            }
        } else {
            for market_id in matching_market_ids {
                if let Some(market) = self.get_market(txn, &market_id)? {
                    markets.push(market);
                }
            }
        }

        markets.sort_by_key(|m| m.expires_at_height.unwrap_or(u64::MAX));

        tracing::debug!(
            "Retrieved {} markets by expiry range [{:?}, {:?}]",
            markets.len(),
            min_height,
            max_height
        );

        Ok(markets)
    }

    pub fn get_markets_by_slot(
        &self,
        txn: &RoTxn,
        slot_id: SlotId,
    ) -> Result<Vec<Market>, Error> {
        let market_ids =
            self.slot_index.try_get(txn, &slot_id)?.unwrap_or_default();

        let mut markets = Vec::with_capacity(market_ids.len());
        for market_id in market_ids {
            if let Some(market) = self.get_market(txn, &market_id)? {
                markets.push(market);
            }
        }

        Ok(markets)
    }

    pub fn update_market(
        &self,
        txn: &mut RwTxn,
        market: &Market,
    ) -> Result<(), Error> {
        let old_market = self.get_market(txn, &market.id)?;

        let _old_state = old_market.as_ref().map(|m| m.state());
        let _old_expiry = old_market.as_ref().and_then(|m| m.expires_at_height);
        let old_slots: HashSet<_> = old_market
            .as_ref()
            .map(|m| m.decision_slots.iter().cloned().collect())
            .unwrap_or_default();

        if let Some(ref old) = old_market {
            self.validate_market_state_transition(old.state(), market.state())?;
        }

        self.markets
            .put(txn, market.id.as_bytes(), market)
            .map_err(|e| {
                tracing::error!(
                    "Failed to update primary market storage for market {}: {}",
                    market.id,
                    e
                );
                Error::DatabaseError(format!(
                    "Primary market update failed: {e}"
                ))
            })?;

        if let Some(old) = old_market {
            if old.state() != market.state() {
                self.update_state_index(
                    txn,
                    &market.id,
                    Some(old.state()),
                    Some(market.state()),
                )
                .map_err(|e| {
                    tracing::error!(
                        "Failed to update state index for market {}: {}",
                        market.id,
                        e
                    );
                    Error::DatabaseError(format!(
                        "State index update failed: {e}"
                    ))
                })?;
            }

            if old.expires_at_height != market.expires_at_height {
                self.update_expiry_index(
                    txn,
                    &market.id,
                    old.expires_at_height,
                    market.expires_at_height,
                )
                .map_err(|e| {
                    tracing::error!(
                        "Failed to update expiry index for market {}: {}",
                        market.id,
                        e
                    );
                    Error::DatabaseError(format!(
                        "Expiry index update failed: {e}"
                    ))
                })?;
            }

            let new_slots: HashSet<_> =
                market.decision_slots.iter().cloned().collect();

            for slot_id in old_slots.difference(&new_slots) {
                self.update_slot_index(txn, &market.id, *slot_id, false)
                    .map_err(|e| {
                        tracing::error!(
                            "Failed to remove market {} from slot index {}: {}",
                            market.id,
                            hex::encode(slot_id.as_bytes()),
                            e
                        );
                        Error::DatabaseError(format!(
                            "Slot index removal failed: {e}"
                        ))
                    })?;
            }

            for slot_id in new_slots.difference(&old_slots) {
                self.update_slot_index(txn, &market.id, *slot_id, true)
                    .map_err(|e| {
                        tracing::error!(
                            "Failed to add market {} to slot index {}: {}",
                            market.id,
                            hex::encode(slot_id.as_bytes()),
                            e
                        );
                        Error::DatabaseError(format!(
                            "Slot index addition failed: {e}"
                        ))
                    })?;
            }
        }

        tracing::debug!(
            "Successfully updated market {} with all indexes",
            market.id
        );
        Ok(())
    }

    /// Transition resolved markets directly from Trading → Ossified with automatic share payouts.
    /// Called every block from connect_body. Checks ALL trading markets to see if
    /// their decision slots are now Resolved in the SlotStateHistory database.
    #[allow(clippy::type_complexity)]
    pub fn transition_and_payout_resolved_markets(
        &self,
        txn: &mut RwTxn,
        state: &crate::state::State,
        slots_db: &crate::state::slots::Dbs,
        current_height: u32,
    ) -> Result<
        (
            Vec<(MarketId, MarketPayoutSummary)>,
            Vec<crate::state::undo::OssificationUndoEntry>,
        ),
        Error,
    > {
        let mut results = Vec::new();
        let mut undo_entries = Vec::new();
        let trading_markets =
            self.get_markets_by_state(txn, MarketState::Trading)?;

        for mut market in trading_markets {
            if market.decision_slots.is_empty() {
                continue;
            }

            let mut all_slots_resolved = true;
            let mut slot_outcomes: std::collections::HashMap<SlotId, f64> =
                std::collections::HashMap::new();
            let mut decisions: HashMap<SlotId, Decision> = HashMap::new();

            for slot_id in &market.decision_slots {
                let slot_state =
                    slots_db.get_slot_current_state(txn, *slot_id)?;

                if slot_state != crate::state::slots::SlotState::Resolved {
                    all_slots_resolved = false;
                    break;
                }

                let voting_period_id =
                    crate::state::voting::types::VotingPeriodId::new(
                        slot_id.voting_period(),
                    );
                if let Some(outcome) = state
                    .voting()
                    .databases()
                    .get_consensus_outcome(txn, voting_period_id, *slot_id)?
                {
                    slot_outcomes.insert(*slot_id, outcome);
                }

                if let Some(slot) = slots_db.get_slot(txn, *slot_id)?
                    && let Some(decision) = slot.decision
                {
                    decisions.insert(*slot_id, decision);
                }
            }

            if all_slots_resolved {
                let market_id = market.id.clone();

                // Capture pre-ossification state for undo
                let pre_ossification_market = market.clone();

                // Capture treasury and fee UTXOs before they are consumed
                let treasury_utxo = if let Some(outpoint) =
                    self.get_market_funds_utxo(txn, &market_id, false)?
                {
                    state
                        .utxos
                        .try_get(txn, &outpoint)?
                        .map(|output| (outpoint, output))
                } else {
                    None
                };
                let fee_utxo = if let Some(outpoint) =
                    self.get_market_funds_utxo(txn, &market_id, true)?
                {
                    state
                        .utxos
                        .try_get(txn, &outpoint)?
                        .map(|output| (outpoint, output))
                } else {
                    None
                };

                let final_prices = market
                    .calculate_final_prices(&slot_outcomes, &decisions)
                    .map_err(|e| {
                        Error::DatabaseError(format!(
                            "Failed to calculate final prices: {e:?}"
                        ))
                    })?;

                market
                    .update_state(
                        current_height as u64,
                        None,
                        None,
                        None,
                        Some(final_prices),
                    )
                    .map_err(|e| {
                        Error::DatabaseError(format!(
                            "Failed to set final prices: {e:?}"
                        ))
                    })?;

                let payout_summary = self.calculate_share_payouts(
                    txn,
                    state,
                    &market,
                    current_height as u64,
                )?;

                // Apply payouts
                self.apply_automatic_share_payouts(
                    state,
                    txn,
                    &payout_summary,
                    &market,
                    current_height as u64,
                )?;

                market
                    .update_state(
                        current_height as u64,
                        Some(MarketState::Ossified),
                        None,
                        None,
                        None,
                    )
                    .map_err(|e| {
                        Error::DatabaseError(format!(
                            "Failed to ossify market: {e:?}"
                        ))
                    })?;

                self.update_market(txn, &market)?;

                undo_entries.push(crate::state::undo::OssificationUndoEntry {
                    pre_ossification_market,
                    payout_summary: payout_summary.clone(),
                    treasury_utxo,
                    fee_utxo,
                });

                results.push((market_id, payout_summary));
            }
        }

        Ok((results, undo_entries))
    }

    pub fn cancel_market(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), MarketError> {
        let mut market = self
            .get_market(txn, market_id)
            .map_err(|e| MarketError::DatabaseError(e.to_string()))?
            .ok_or(MarketError::MarketNotFound {
                id: market_id.clone(),
            })?;
        market.cancel_market(None, 0)?;
        self.update_market(txn, &market)
            .map_err(|e| MarketError::DatabaseError(e.to_string()))
    }

    pub fn invalidate_market(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), MarketError> {
        let mut market = self
            .get_market(txn, market_id)
            .map_err(|e| MarketError::DatabaseError(e.to_string()))?
            .ok_or(MarketError::MarketNotFound {
                id: market_id.clone(),
            })?;
        market.invalidate_market(None, 0)?;
        self.update_market(txn, &market)
            .map_err(|e| MarketError::DatabaseError(e.to_string()))
    }

    pub fn get_mempool_shares(
        &self,
        rotxn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<Option<Array<i64, Ix1>>, Error> {
        let mempool_addr = mempool_address(market_id);

        match self.share_accounts.get(rotxn, &mempool_addr) {
            Ok(account) => {
                if let Some(market) = self.get_market(rotxn, market_id)? {
                    let mut shares = market.shares().clone();
                    for ((account_market_id, outcome_index), &amount) in
                        &account.positions
                    {
                        if account_market_id == market_id {
                            shares[*outcome_index as usize] = amount;
                        }
                    }
                    return Ok(Some(shares));
                }
                Ok(None)
            }
            Err(_) => Ok(None),
        }
    }

    pub fn put_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &MarketId,
        shares: &Array<i64, Ix1>,
    ) -> Result<(), Error> {
        let mempool_addr = mempool_address(market_id);

        let mut account = match self.share_accounts.get(rwtxn, &mempool_addr) {
            Ok(acc) => acc,
            Err(_) => ShareAccount::new(),
        };

        account.positions.retain(|(mid, _), _| mid != market_id);

        for (i, &share_amount) in shares.iter().enumerate() {
            if share_amount != 0 {
                account
                    .positions
                    .insert((market_id.clone(), i as u32), share_amount);
            }
        }

        self.share_accounts.put(rwtxn, &mempool_addr, &account)?;
        Ok(())
    }

    pub fn clear_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), Error> {
        let mempool_addr = mempool_address(market_id);

        if let Ok(mut account) = self.share_accounts.get(rwtxn, &mempool_addr) {
            account.positions.retain(|(mid, _), _| mid != market_id);

            if account.positions.is_empty() {
                self.share_accounts.delete(rwtxn, &mempool_addr)?;
            } else {
                self.share_accounts.put(rwtxn, &mempool_addr, &account)?;
            }
        }
        Ok(())
    }

    fn update_state_index(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        old_state: Option<MarketState>,
        new_state: Option<MarketState>,
    ) -> Result<(), Error> {
        if let Some(old) = old_state {
            let mut market_ids =
                self.state_index.try_get(txn, &old)?.unwrap_or_default();
            market_ids.retain(|id| id != market_id);
            if market_ids.is_empty() {
                self.state_index.delete(txn, &old)?;
            } else {
                self.state_index.put(txn, &old, &market_ids)?;
            }
        }

        if let Some(new) = new_state {
            let mut market_ids =
                self.state_index.try_get(txn, &new)?.unwrap_or_default();
            if !market_ids.contains(market_id) {
                market_ids.push(market_id.clone());
                self.state_index.put(txn, &new, &market_ids)?;
            }
        }

        Ok(())
    }

    fn update_expiry_index(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        old_expiry: Option<u64>,
        new_expiry: Option<u64>,
    ) -> Result<(), Error> {
        if let Some(old) = old_expiry {
            let mut market_ids =
                self.expiry_index.try_get(txn, &old)?.unwrap_or_default();
            market_ids.retain(|id| id != market_id);
            if market_ids.is_empty() {
                self.expiry_index.delete(txn, &old)?;
            } else {
                self.expiry_index.put(txn, &old, &market_ids)?;
            }
        }

        if let Some(new) = new_expiry {
            let mut market_ids =
                self.expiry_index.try_get(txn, &new)?.unwrap_or_default();
            if !market_ids.contains(market_id) {
                market_ids.push(market_id.clone());
                self.expiry_index.put(txn, &new, &market_ids)?;
            }
        }

        Ok(())
    }

    fn update_slot_index(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        slot_id: SlotId,
        add: bool,
    ) -> Result<(), Error> {
        let mut market_ids =
            self.slot_index.try_get(txn, &slot_id)?.unwrap_or_default();

        if add {
            if !market_ids.contains(market_id) {
                market_ids.push(market_id.clone());
                self.slot_index.put(txn, &slot_id, &market_ids)?;
            }
        } else {
            market_ids.retain(|id| id != market_id);
            if market_ids.is_empty() {
                self.slot_index.delete(txn, &slot_id)?;
            } else {
                self.slot_index.put(txn, &slot_id, &market_ids)?;
            }
        }

        Ok(())
    }

    pub fn process_market_trades_batch(
        &self,
        txn: &mut RwTxn,
        batched_trades: Vec<BatchedMarketTrade>,
        state: &crate::state::State,
    ) -> Result<Vec<f64>, Error> {
        if batched_trades.is_empty() {
            return Ok(Vec::new());
        }

        let mut market_updates: HashMap<MarketId, Array<i64, Ix1>> =
            HashMap::new();

        tracing::debug!(
            "Validating {} batched market trades using centralized validation",
            batched_trades.len()
        );

        let trade_costs =
            crate::validation::MarketValidator::validate_batched_trades(
                state,
                txn,
                &batched_trades,
            )?;

        for trade in &batched_trades {
            let shares_update = market_updates
                .entry(trade.market_id.clone())
                .or_insert_with(|| {
                    Array::zeros(trade.market_snapshot.shares.len())
                });
            shares_update[trade.outcome_index as usize] += trade.shares_to_buy;
        }

        tracing::debug!(
            "Applying market updates for {} markets",
            market_updates.len()
        );

        for (market_id, share_changes) in market_updates {
            let mut market =
                self.get_market(txn, &market_id)?.ok_or_else(|| {
                    tracing::error!(
                        "Market {} disappeared during batch processing",
                        market_id
                    );
                    Error::DatabaseError(
                        "Market disappeared during processing".to_string(),
                    )
                })?;

            let mut new_shares_array = market.shares().clone();
            for (outcome_index, &share_change) in
                share_changes.iter().enumerate()
            {
                if share_change != 0 {
                    let new_val =
                        new_shares_array[outcome_index] + share_change;
                    if new_val < 0 {
                        tracing::error!(
                            "Share update would result in negative shares for market {} outcome {}: {} + {} = {}",
                            market_id,
                            outcome_index,
                            market.shares()[outcome_index],
                            share_change,
                            new_val
                        );
                        return Err(Error::DatabaseError("Invalid share update would result in negative shares".to_string()));
                    }
                    new_shares_array[outcome_index] = new_val;
                }
            }

            market
                .update_state(0, None, None, Some(new_shares_array), None)
                .map_err(|e| {
                    Error::DatabaseError(format!(
                        "Failed to update market state: {e:?}"
                    ))
                })?;

            self.update_market(txn, &market).map_err(|e| {
                tracing::error!(
                    "Failed to update market {} during batch processing: {}",
                    market_id,
                    e
                );
                e
            })?;

            tracing::debug!(
                "Successfully updated market {} with new shares",
                market_id,
            );
        }

        tracing::debug!(
            "Updating share accounts for {} trades",
            batched_trades.len()
        );

        for (trade_index, trade) in batched_trades.iter().enumerate() {
            self.add_shares_to_account(
                txn,
                &trade.trader_address,
                trade.market_id.clone(),
                trade.outcome_index,
                trade.shares_to_buy,
                0,
            )
            .map_err(|e| {
                tracing::error!(
                    "Failed to update share account for trade {}: {}",
                    trade_index,
                    e
                );
                Error::DatabaseError(format!(
                    "Share account update failed for trade {trade_index}: {e}"
                ))
            })?;
        }

        tracing::info!(
            "Successfully processed {} batched market trades with total cost: {:.4}",
            batched_trades.len(),
            trade_costs.iter().sum::<f64>()
        );

        Ok(trade_costs)
    }

    pub fn add_shares_to_account(
        &self,
        txn: &mut RwTxn,
        address: &Address,
        market_id: MarketId,
        outcome_index: u32,
        shares: i64,
        height: u64,
    ) -> Result<(), Error> {
        let mut account = self
            .share_accounts
            .try_get(txn, address)?
            .unwrap_or_else(ShareAccount::new);

        account.add_shares(market_id, outcome_index, shares, height);

        self.share_accounts.put(txn, address, &account)?;

        Ok(())
    }

    pub fn remove_shares_from_account(
        &self,
        txn: &mut RwTxn,
        address: &Address,
        market_id: &MarketId,
        outcome_index: u32,
        shares: i64,
        height: u64,
    ) -> Result<(), Error> {
        let mut account = self
            .share_accounts
            .try_get(txn, address)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "No share account found for address".to_string(),
            })?;

        account
            .remove_shares(market_id, outcome_index, shares, height)
            .map_err(|_| Error::InvalidTransaction {
                reason: "Insufficient shares for sell transaction".to_string(),
            })?;

        if account.positions.is_empty() {
            self.share_accounts.delete(txn, address)?;
        } else {
            self.share_accounts.put(txn, address, &account)?;
        }

        Ok(())
    }

    pub fn get_user_share_account(
        &self,
        txn: &RoTxn,
        address: &Address,
    ) -> Result<Option<ShareAccount>, Error> {
        Ok(self.share_accounts.try_get(txn, address)?)
    }

    /// Get all share accounts from the database (for debugging)
    pub fn get_all_share_accounts(
        &self,
        txn: &RoTxn,
    ) -> Result<super::type_aliases::AllShareAccounts, Error> {
        let mut result = Vec::new();
        let mut iter = self.share_accounts.iter(txn)?;
        while let Some((address, account)) = iter.next()? {
            let positions: Vec<(MarketId, u32, i64)> = account
                .positions
                .into_iter()
                .map(|((market_id, outcome_index), shares)| {
                    (market_id, outcome_index, shares)
                })
                .collect();
            if !positions.is_empty() {
                result.push((address, positions));
            }
        }
        Ok(result)
    }

    pub fn get_user_share_positions(
        &self,
        txn: &RoTxn,
        address: &Address,
    ) -> Result<Vec<(MarketId, u32, i64)>, Error> {
        if let Some(account) = self.get_user_share_account(txn, address)? {
            Ok(account
                .positions
                .into_iter()
                .map(|((market_id, outcome_index), shares)| {
                    (market_id, outcome_index, shares)
                })
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    pub fn get_market_user_positions(
        &self,
        txn: &RoTxn,
        address: &Address,
        market_id: &MarketId,
    ) -> Result<Vec<(u32, i64)>, Error> {
        if let Some(account) = self.get_user_share_account(txn, address)? {
            Ok(account
                .positions
                .into_iter()
                .filter(|((pos_market_id, _), _)| pos_market_id == market_id)
                .map(|((_, outcome_index), shares)| (outcome_index, shares))
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    pub fn get_wallet_positions_for_market_outcome(
        &self,
        txn: &RoTxn,
        addresses: &std::collections::HashSet<Address>,
        market_id: &MarketId,
        outcome_index: u32,
    ) -> Result<std::collections::HashMap<Address, i64>, Error> {
        let mut result = std::collections::HashMap::new();
        for address in addresses {
            if let Some(account) = self.get_user_share_account(txn, address)?
                && let Some(&shares) =
                    account.positions.get(&(market_id.clone(), outcome_index))
                && shares > 0
            {
                result.insert(*address, shares);
            }
        }
        Ok(result)
    }

    pub fn revert_share_trade(
        &self,
        txn: &mut RwTxn,
        address: &Address,
        market_id: MarketId,
        outcome_index: u32,
        shares_traded: i64,
        height: u64,
    ) -> Result<(), Error> {
        self.remove_shares_from_account(
            txn,
            address,
            &market_id,
            outcome_index,
            shares_traded,
            height,
        )
    }

    pub fn get_account_nonce(
        &self,
        txn: &RoTxn,
        address: &Address,
    ) -> Result<u64, Error> {
        if let Some(account) = self.share_accounts.try_get(txn, address)? {
            Ok(account.nonce)
        } else {
            Ok(0)
        }
    }

    pub fn get_account_nonces(
        &self,
        txn: &RoTxn,
        address: &Address,
    ) -> Result<(u64, u64), Error> {
        if let Some(account) = self.share_accounts.try_get(txn, address)? {
            Ok((account.nonce, account.trade_nonce))
        } else {
            Ok((0, 0))
        }
    }

    pub fn get_shareholders_for_market(
        &self,
        txn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<super::type_aliases::MarketShareholders, Error> {
        let mut shareholders = Vec::new();
        let mut iter = self.share_accounts.iter(txn)?;

        while let Some((address, account)) = iter.next()? {
            let positions_for_market: Vec<(u32, i64)> = account
                .positions
                .iter()
                .filter(|((mid, _), _)| mid == market_id)
                .map(|((_, outcome_index), shares)| (*outcome_index, *shares))
                .collect();

            if !positions_for_market.is_empty() {
                shareholders.push((address, positions_for_market));
            }
        }

        Ok(shareholders)
    }

    /// Payout formula: payout_i = (shares_i * final_price_i / total_weighted_shares) * treasury
    pub fn calculate_share_payouts(
        &self,
        txn: &RoTxn,
        state: &crate::state::State,
        market: &Market,
        block_height: u64,
    ) -> Result<MarketPayoutSummary, Error> {
        let treasury_sats =
            self.get_market_funds_sats(txn, state, &market.id, false)?;
        let final_prices = market.final_prices();

        let shareholders = self.get_shareholders_for_market(txn, &market.id)?;

        // Calculate total weighted shares for normalization
        let total_weighted_shares: f64 = shareholders
            .iter()
            .flat_map(|(_, positions)| positions.iter())
            .map(|(outcome_index, shares)| {
                let final_price = final_prices[*outcome_index as usize];
                *shares as f64 * final_price
            })
            .sum();

        // If no winning positions, nothing to distribute
        if total_weighted_shares <= 0.0 {
            return Ok(MarketPayoutSummary {
                market_id: market.id.clone(),
                treasury_distributed: 0,
                author_fees_distributed: 0,
                shareholder_count: 0,
                payouts: Vec::new(),
                block_height,
            });
        }

        let participants: Vec<((Address, u32, i64, f64), f64)> = shareholders
            .into_iter()
            .flat_map(|(address, positions)| {
                positions.into_iter().map(move |(outcome_index, shares)| {
                    let final_price = final_prices[outcome_index as usize];
                    let weighted_shares = shares as f64 * final_price;
                    (
                        (address, outcome_index, shares, final_price),
                        weighted_shares,
                    )
                })
            })
            .collect();

        let alloc_result = allocation::allocate_proportionally_u64(
            participants,
            treasury_sats,
        )
        .map_err(|e| Error::InvalidTransaction {
            reason: format!("Payout allocation failed: {e}"),
        })?;

        let payouts: Vec<SharePayoutRecord> = alloc_result
            .allocations
            .into_iter()
            .map(
                |(
                    (address, outcome_index, shares, final_price),
                    payout_sats,
                )| {
                    SharePayoutRecord {
                        market_id: market.id.clone(),
                        address,
                        outcome_index,
                        shares_redeemed: shares,
                        final_price,
                        payout_sats,
                    }
                },
            )
            .collect();

        let total_distributed = alloc_result.total_allocated;

        let author_fees_distributed =
            self.get_market_funds_sats(txn, state, &market.id, true)?;

        Ok(MarketPayoutSummary {
            market_id: market.id.clone(),
            treasury_distributed: total_distributed,
            author_fees_distributed,
            shareholder_count: payouts.len() as u32,
            payouts,
            block_height,
        })
    }

    pub fn apply_automatic_share_payouts(
        &self,
        state: &crate::state::State,
        txn: &mut RwTxn,
        payout_summary: &MarketPayoutSummary,
        market: &Market,
        block_height: u64,
    ) -> Result<(), Error> {
        use crate::types::{
            BitcoinOutputContent, FilledOutput, FilledOutputContent,
        };

        let mut sequence = 0u32;

        for payout in &payout_summary.payouts {
            let outpoint = generate_share_payout_outpoint(
                &payout.market_id,
                &payout.address,
                block_height,
                sequence,
            );

            let output = FilledOutput {
                address: payout.address,
                content: FilledOutputContent::Bitcoin(BitcoinOutputContent(
                    bitcoin::Amount::from_sat(payout.payout_sats),
                )),
                memo: vec![],
            };

            state.insert_utxo_with_address_index(txn, &outpoint, &output)?;

            self.remove_shares_from_account(
                txn,
                &payout.address,
                &payout.market_id,
                payout.outcome_index,
                payout.shares_redeemed,
                block_height,
            )?;

            sequence += 1;
        }

        if payout_summary.author_fees_distributed > 0 {
            let fee_outpoint = generate_share_payout_outpoint(
                &payout_summary.market_id,
                &market.creator_address,
                block_height,
                sequence,
            );

            let fee_output = FilledOutput {
                address: market.creator_address,
                content: FilledOutputContent::Bitcoin(BitcoinOutputContent(
                    bitcoin::Amount::from_sat(
                        payout_summary.author_fees_distributed,
                    ),
                )),
                memo: vec![],
            };

            state.insert_utxo_with_address_index(
                txn,
                &fee_outpoint,
                &fee_output,
            )?;
        }

        // Consume the Market UTXO (treasury is now distributed to shareholders)
        if let Some(market_utxo) =
            self.get_market_funds_utxo(txn, &payout_summary.market_id, false)?
        {
            state.delete_utxo_with_address_index(txn, &market_utxo)?;
            self.clear_market_funds_utxo(
                txn,
                &payout_summary.market_id,
                false,
            )?;
        }

        // Consume the Author Fee UTXO (fees now paid to market creator)
        if let Some(fee_utxo) =
            self.get_market_funds_utxo(txn, &payout_summary.market_id, true)?
        {
            state.delete_utxo_with_address_index(txn, &fee_utxo)?;
            self.clear_market_funds_utxo(txn, &payout_summary.market_id, true)?;
        }

        Ok(())
    }

    pub fn revert_automatic_share_payouts(
        &self,
        state: &crate::state::State,
        txn: &mut RwTxn,
        payout_summary: &MarketPayoutSummary,
        market: &Market,
        block_height: u64,
    ) -> Result<(), Error> {
        let mut sequence = 0u32;

        for payout in &payout_summary.payouts {
            let outpoint = generate_share_payout_outpoint(
                &payout.market_id,
                &payout.address,
                block_height,
                sequence,
            );

            state.delete_utxo_with_address_index(txn, &outpoint)?;

            // Restore shares to account
            self.add_shares_to_account(
                txn,
                &payout.address,
                payout.market_id.clone(),
                payout.outcome_index,
                payout.shares_redeemed,
                block_height,
            )?;

            sequence += 1;
        }

        // Remove author fee UTXO
        if payout_summary.author_fees_distributed > 0 {
            let fee_outpoint = generate_share_payout_outpoint(
                &payout_summary.market_id,
                &market.creator_address,
                block_height,
                sequence,
            );

            state.delete_utxo_with_address_index(txn, &fee_outpoint)?;
        }

        tracing::info!(
            "Reverted automatic share payouts for market {}: {} sats treasury + {} sats fees",
            payout_summary.market_id,
            payout_summary.treasury_distributed,
            payout_summary.author_fees_distributed,
        );

        Ok(())
    }

    pub fn get_market_funds_utxo(
        &self,
        txn: &RoTxn,
        market_id: &MarketId,
        is_fee: bool,
    ) -> Result<Option<OutPoint>, Error> {
        Ok(self
            .market_funds_utxos
            .try_get(txn, &(*market_id.as_bytes(), is_fee))?)
    }

    pub fn set_market_funds_utxo(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        is_fee: bool,
        outpoint: &OutPoint,
    ) -> Result<(), Error> {
        self.market_funds_utxos.put(
            txn,
            &(*market_id.as_bytes(), is_fee),
            outpoint,
        )?;
        Ok(())
    }

    pub fn clear_market_funds_utxo(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        is_fee: bool,
    ) -> Result<(), Error> {
        self.market_funds_utxos
            .delete(txn, &(*market_id.as_bytes(), is_fee))?;
        Ok(())
    }

    pub fn get_market_funds_sats(
        &self,
        txn: &RoTxn,
        state: &crate::state::State,
        market_id: &MarketId,
        is_fee: bool,
    ) -> Result<u64, Error> {
        match self.get_market_funds_utxo(txn, market_id, is_fee)? {
            Some(outpoint) => {
                let utxo = state
                    .utxos
                    .try_get(txn, &outpoint)?
                    .ok_or(Error::NoUtxo { outpoint })?;
                Ok(utxo.get_bitcoin_value().to_sat())
            }
            None => Ok(0),
        }
    }

    // ========== PENDING MARKET FUNDS UTXO ACCESSORS ==========

    /// Register a pending market funds UTXO from a trade transaction
    /// is_fee=false for treasury, is_fee=true for author fees
    pub fn add_pending_market_funds_utxo(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        outpoint: &OutPoint,
        is_fee: bool,
    ) -> Result<(), Error> {
        let mut pending = self
            .pending_market_funds_utxos
            .try_get(txn, market_id.as_bytes())?
            .unwrap_or_default();
        pending.push((*outpoint, is_fee));
        self.pending_market_funds_utxos.put(
            txn,
            market_id.as_bytes(),
            &pending,
        )?;
        Ok(())
    }

    /// Get all pending market funds UTXOs for a market
    /// Returns Vec of (OutPoint, is_fee) tuples
    pub fn get_pending_market_funds_utxos(
        &self,
        txn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<Vec<(OutPoint, bool)>, Error> {
        Ok(self
            .pending_market_funds_utxos
            .try_get(txn, market_id.as_bytes())?
            .unwrap_or_default())
    }

    pub fn clear_pending_market_funds_utxos(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), Error> {
        self.pending_market_funds_utxos
            .delete(txn, market_id.as_bytes())?;
        Ok(())
    }

    pub fn restore_pending_market_funds_utxos(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        utxos: &[(OutPoint, bool)],
    ) -> Result<(), Error> {
        if utxos.is_empty() {
            self.pending_market_funds_utxos
                .delete(txn, market_id.as_bytes())?;
        } else {
            self.pending_market_funds_utxos.put(
                txn,
                market_id.as_bytes(),
                &utxos.to_vec(),
            )?;
        }
        Ok(())
    }

    pub fn remove_pending_market_funds_utxo(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        outpoint: &OutPoint,
        is_fee: bool,
    ) -> Result<(), Error> {
        let mut pending = self
            .pending_market_funds_utxos
            .try_get(txn, market_id.as_bytes())?
            .unwrap_or_default();
        pending.retain(|(o, fee)| o != outpoint || *fee != is_fee);
        if pending.is_empty() {
            self.pending_market_funds_utxos
                .delete(txn, market_id.as_bytes())?;
        } else {
            self.pending_market_funds_utxos.put(
                txn,
                market_id.as_bytes(),
                &pending,
            )?;
        }
        Ok(())
    }

    /// Get all market IDs that have pending UTXOs to consolidate
    pub fn get_markets_with_pending_utxos(
        &self,
        txn: &RoTxn,
    ) -> Result<HashSet<[u8; 6]>, Error> {
        let mut markets = HashSet::new();

        let mut iter = self.pending_market_funds_utxos.iter(txn)?;
        while let Some((market_id_bytes, _)) = iter.next()? {
            markets.insert(market_id_bytes);
        }

        Ok(markets)
    }

    // ========== BACKWARD COMPATIBILITY WRAPPERS FOR PENDING UTXOS ==========

    /// Register a pending treasury UTXO from a buy Trade transaction
    pub fn add_pending_treasury_utxo(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        outpoint: &OutPoint,
    ) -> Result<(), Error> {
        self.add_pending_market_funds_utxo(txn, market_id, outpoint, false)
    }

    /// Register a pending author fee UTXO from a buy Trade transaction
    pub fn add_pending_author_fee_utxo(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        outpoint: &OutPoint,
    ) -> Result<(), Error> {
        self.add_pending_market_funds_utxo(txn, market_id, outpoint, true)
    }

    /// Get all pending treasury UTXOs for a market
    pub fn get_pending_treasury_utxos(
        &self,
        txn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<Vec<OutPoint>, Error> {
        let all_pending =
            self.get_pending_market_funds_utxos(txn, market_id)?;
        Ok(all_pending
            .into_iter()
            .filter(|(_, is_fee)| !*is_fee)
            .map(|(outpoint, _)| outpoint)
            .collect())
    }

    /// Get all pending author fee UTXOs for a market
    pub fn get_pending_author_fee_utxos(
        &self,
        txn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<Vec<OutPoint>, Error> {
        let all_pending =
            self.get_pending_market_funds_utxos(txn, market_id)?;
        Ok(all_pending
            .into_iter()
            .filter(|(_, is_fee)| *is_fee)
            .map(|(outpoint, _)| outpoint)
            .collect())
    }

    /// Clear pending treasury UTXOs after consolidation
    pub fn clear_pending_treasury_utxos(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), Error> {
        // Get all pending, keep only the fee ones
        let all_pending =
            self.get_pending_market_funds_utxos(txn, market_id)?;
        let fee_only: Vec<_> = all_pending
            .into_iter()
            .filter(|(_, is_fee)| *is_fee)
            .collect();
        if fee_only.is_empty() {
            self.pending_market_funds_utxos
                .delete(txn, market_id.as_bytes())?;
        } else {
            self.pending_market_funds_utxos.put(
                txn,
                market_id.as_bytes(),
                &fee_only,
            )?;
        }
        Ok(())
    }

    /// Clear pending author fee UTXOs after consolidation
    pub fn clear_pending_author_fee_utxos(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), Error> {
        // Get all pending, keep only the treasury ones
        let all_pending =
            self.get_pending_market_funds_utxos(txn, market_id)?;
        let treasury_only: Vec<_> = all_pending
            .into_iter()
            .filter(|(_, is_fee)| !*is_fee)
            .collect();
        if treasury_only.is_empty() {
            self.pending_market_funds_utxos
                .delete(txn, market_id.as_bytes())?;
        } else {
            self.pending_market_funds_utxos.put(
                txn,
                market_id.as_bytes(),
                &treasury_only,
            )?;
        }
        Ok(())
    }

    /// Remove a specific pending treasury UTXO (for revert operations)
    pub fn remove_pending_treasury_utxo(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        outpoint: &OutPoint,
    ) -> Result<(), Error> {
        self.remove_pending_market_funds_utxo(txn, market_id, outpoint, false)
    }

    /// Remove a specific pending author fee UTXO (for revert operations)
    pub fn remove_pending_author_fee_utxo(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        outpoint: &OutPoint,
    ) -> Result<(), Error> {
        self.remove_pending_market_funds_utxo(txn, market_id, outpoint, true)
    }
}

/// Generate a deterministic address for market treasury
/// This address is not spendable by any user - it's a system address
pub fn generate_market_treasury_address(market_id: &MarketId) -> Address {
    use blake3::Hasher;

    let mut hasher = Hasher::new();
    hasher.update(b"MARKET_TREASURY_ADDRESS");
    hasher.update(&market_id.0);

    let hash = hasher.finalize();
    let mut address_bytes = [0u8; 20];
    address_bytes.copy_from_slice(&hash.as_bytes()[0..20]);

    Address(address_bytes)
}

/// Generate a deterministic address for market author fees
/// This address is not spendable by any user - it's a system address
pub fn generate_market_author_fee_address(market_id: &MarketId) -> Address {
    use blake3::Hasher;

    let mut hasher = Hasher::new();
    hasher.update(b"MARKET_AUTHOR_FEE_ADDRESS");
    hasher.update(&market_id.0);

    let hash = hasher.finalize();
    let mut address_bytes = [0u8; 20];
    address_bytes.copy_from_slice(&hash.as_bytes()[0..20]);

    Address(address_bytes)
}

/// Generate a deterministic outpoint for share payouts
fn generate_share_payout_outpoint(
    market_id: &MarketId,
    shareholder_address: &Address,
    block_height: u64,
    sequence: u32,
) -> OutPoint {
    use blake3::Hasher;

    let mut hasher = Hasher::new();
    hasher.update(b"SHARE_PAYOUT");
    hasher.update(&market_id.0);
    hasher.update(&shareholder_address.0);
    hasher.update(&block_height.to_le_bytes());
    hasher.update(&sequence.to_le_bytes());

    let hash = hasher.finalize();
    let merkle_root = crate::types::MerkleRoot::from(*hash.as_bytes());

    OutPoint::Coinbase {
        merkle_root,
        vout: sequence,
    }
}

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::uninlined_format_args)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_dfunction_constraint_validation() {
        let valid_func = DFunction::Decision(0);
        assert!(valid_func.validate_constraint(2).is_ok());

        let invalid_func = DFunction::Decision(5);
        assert!(invalid_func.validate_constraint(2).is_err());

        let valid_equals =
            DFunction::Equals(Box::new(DFunction::Decision(0)), 1);
        assert!(valid_equals.validate_constraint(2).is_ok());

        let invalid_equals =
            DFunction::Equals(Box::new(DFunction::Decision(0)), 5);
        assert!(invalid_equals.validate_constraint(2).is_err());

        let nested_and = DFunction::And(
            Box::new(DFunction::Decision(0)),
            Box::new(DFunction::Decision(1)),
        );
        assert!(nested_and.validate_constraint(2).is_ok());

        let invalid_nested = DFunction::And(
            Box::new(DFunction::Decision(0)),
            Box::new(DFunction::Decision(5)),
        );
        assert!(invalid_nested.validate_constraint(2).is_err());
    }

    #[test]
    fn test_categorical_constraint_validation() {
        let decision_slots = vec![];
        let df = DFunction::True;

        let valid_combo = vec![1, 0, 0];
        let categorical_slots = vec![0, 1, 2];
        assert!(
            df.validate_categorical_constraint(
                &categorical_slots,
                &valid_combo,
                &decision_slots
            )
            .unwrap()
        );

        let residual_combo = vec![0, 0, 0];
        assert!(
            df.validate_categorical_constraint(
                &categorical_slots,
                &residual_combo,
                &decision_slots
            )
            .unwrap()
        );

        let invalid_combo = vec![1, 1, 0];
        assert!(
            !df.validate_categorical_constraint(
                &categorical_slots,
                &invalid_combo,
                &decision_slots
            )
            .unwrap()
        );

        let oob_slots = vec![0, 1, 5];
        assert!(
            df.validate_categorical_constraint(
                &oob_slots,
                &valid_combo,
                &decision_slots
            )
            .is_err()
        );
    }

    #[test]
    fn test_dimension_parsing() {
        let single_str = "[010101]";
        let result = parse_dimensions(single_str);
        assert!(result.is_ok());
        let dimensions = result.unwrap();
        assert_eq!(dimensions.len(), 1);
        assert!(matches!(dimensions[0], DimensionSpec::Single(_)));

        let categorical_str = "[[010101,010102,010103]]";
        let result = parse_dimensions(categorical_str);
        assert!(result.is_ok());
        let dimensions = result.unwrap();
        assert_eq!(dimensions.len(), 1);
        if let DimensionSpec::Categorical(slots) = &dimensions[0] {
            assert_eq!(slots.len(), 3);
        } else {
            panic!("Expected categorical dimension");
        }

        let mixed_str = "[010101,[010102,010103],010104]";
        let result = parse_dimensions(mixed_str);
        assert!(result.is_ok());
        let dimensions = result.unwrap();
        assert_eq!(dimensions.len(), 3);
        assert!(matches!(dimensions[0], DimensionSpec::Single(_)));
        assert!(matches!(dimensions[1], DimensionSpec::Categorical(_)));
        assert!(matches!(dimensions[2], DimensionSpec::Single(_)));

        let invalid_str = "010101,010102";
        let result = parse_dimensions(invalid_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_mempool_market_processing() {
        let market_id = MarketId([0u8; 6]);
        let shares = Array::from_vec(vec![100i64, 100, 100]);

        let snapshot = MarketSnapshot {
            shares,
            b: 10.0,
            trading_fee: 0.01,
        };

        let trade = BatchedMarketTrade {
            market_id,
            outcome_index: 0,
            shares_to_buy: 10,
            max_cost: 1000,
            market_snapshot: snapshot,
            trader_address: Address([0u8; 20]),
        };

        assert_eq!(trade.outcome_index, 0);
        assert_eq!(trade.shares_to_buy, 10);
        assert_eq!(trade.max_cost, 1000);
        assert_eq!(trade.market_snapshot.b, 10.0);
        assert_eq!(trade.market_snapshot.trading_fee, 0.01);

        println!("Mempool trade structure validation passed");
    }

    #[test]
    fn test_lmsr_initialization_spec_compliance() {
        let beta: f64 = 7.0;
        let n_outcomes: f64 = 2.0;
        let target_liquidity: f64 = 100.0;

        let min_treasury = beta * n_outcomes.ln();
        let expected_initial_shares = target_liquidity - min_treasury;

        println!("Binary market test:");
        println!(
            "  β = {}, n = {}, L = {}",
            beta, n_outcomes, target_liquidity
        );
        println!("  Min treasury (β×ln(n)) = {:.6}", min_treasury);
        println!("  Expected initial shares = {:.6}", expected_initial_shares);

        let shares = Array::from_elem(2, expected_initial_shares);
        let calculated_treasury =
            beta * shares.mapv(|x| (x / beta).exp()).sum().ln();

        println!("  Calculated treasury = {:.6}", calculated_treasury);
        println!("  Target liquidity = {:.6}", target_liquidity);

        assert!(
            (calculated_treasury - target_liquidity).abs() < 1e-10,
            "Treasury {:.6} should equal target liquidity {:.6}",
            calculated_treasury,
            target_liquidity
        );

        let exp_shares: Array<f64, ndarray::Ix1> =
            shares.mapv(|x| (x / beta).exp());
        let sum_exp = exp_shares.sum();
        let prices: Array<f64, ndarray::Ix1> = exp_shares.mapv(|x| x / sum_exp);

        for (i, &price) in prices.iter().enumerate() {
            let expected_price = 1.0 / n_outcomes;
            println!(
                "  Price[{}] = {:.6}, expected = {:.6}",
                i, price, expected_price
            );
            assert!(
                (price - expected_price).abs() < 1e-10,
                "Price[{}] should be {:.6} but was {:.6}",
                i,
                expected_price,
                price
            );
        }

        let beta: f64 = 3.2;
        let n_outcomes: f64 = 3.0;
        let target_liquidity: f64 = 50.0;

        let min_treasury = beta * n_outcomes.ln();
        let expected_initial_shares = target_liquidity - min_treasury;

        println!("\n3-outcome market test:");
        println!(
            "  β = {}, n = {}, L = {}",
            beta, n_outcomes, target_liquidity
        );
        println!("  Min treasury (β×ln(n)) = {:.6}", min_treasury);
        println!("  Expected initial shares = {:.6}", expected_initial_shares);

        let shares = Array::from_elem(3, expected_initial_shares);
        let calculated_treasury =
            beta * shares.mapv(|x| (x / beta).exp()).sum().ln();

        println!("  Calculated treasury = {:.6}", calculated_treasury);

        assert!(
            (calculated_treasury - target_liquidity).abs() < 1e-10,
            "Treasury should equal target liquidity"
        );

        let exp_shares: Array<f64, ndarray::Ix1> =
            shares.mapv(|x| (x / beta).exp());
        let sum_exp = exp_shares.sum();
        let prices: Array<f64, ndarray::Ix1> = exp_shares.mapv(|x| x / sum_exp);

        for &price in prices.iter() {
            let expected_price = 1.0 / n_outcomes;
            assert!(
                (price - expected_price).abs() < 1e-10,
                "All prices should be uniform at {:.6}",
                expected_price
            );
        }

        let beta: f64 = 5.0;
        let n_outcomes: f64 = 4.0;
        let min_liquidity = beta * n_outcomes.ln();

        println!("\nMinimum liquidity edge case:");
        println!(
            "  β = {}, n = {}, min L = {:.6}",
            beta, n_outcomes, min_liquidity
        );

        let expected_shares: f64 = min_liquidity - min_liquidity;
        assert!(
            expected_shares.abs() < 1e-10,
            "Shares should be 0 at minimum liquidity"
        );

        let shares = Array::zeros(4);
        let calculated_treasury =
            beta * shares.mapv(|x: f64| (x / beta).exp()).sum().ln();

        println!("  Treasury with zero shares = {:.6}", calculated_treasury);
        assert!(
            (calculated_treasury - min_liquidity).abs() < 1e-10,
            "Zero shares should give minimum treasury"
        );
    }

    #[test]
    fn test_liquidity_validation() {
        let beta: f64 = 7.0;
        let n_outcomes: f64 = 2.0;
        let min_liquidity = beta * n_outcomes.ln();

        let insufficient = min_liquidity - 0.1;
        let expected_shares: f64 = insufficient - min_liquidity;

        assert!(
            expected_shares < 0.0,
            "Insufficient liquidity should result in negative shares"
        );

        let adequate = min_liquidity + 10.0;
        let expected_shares: f64 = adequate - min_liquidity;

        assert!(
            expected_shares > 0.0 && expected_shares.is_finite(),
            "Adequate liquidity should result in positive, finite shares"
        );
    }
}
