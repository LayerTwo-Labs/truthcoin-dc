use borsh::BorshSerialize;
use ndarray::{Array, Ix1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::state::decisions::{Decision, DecisionId};
use crate::types::Address;
use crate::types::hashes;

use super::types::{
    DEFAULT_TRADING_FEE, DimensionSpec, MarketError, MarketId, MarketState,
};
use crate::math::lmsr::MAX_OUTCOMES;
use crate::math::markets;

/// Market metadata. The LMSR `beta` parameter is not stored here — it is
/// always derived from the current treasury UTXO (plus any pending mempool
/// amplify_beta deposits) via `beta = treasury / ln(num_outcomes)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    pub id: MarketId,
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub creator_address: Address,
    pub dimension_specs: Vec<DimensionSpec>,
    pub decision_ids: Vec<DecisionId>,
    pub state_combos: Vec<Vec<usize>>,
    pub created_at_height: u32,
    pub expires_at_height: Option<u32>,
    pub tau_from_now: u8,
    pub storage_fee_sats: u64,
    pub market_state: MarketState,
    pub trading_fee: f64,
    #[serde(with = "ndarray_1d_i64_serde")]
    pub shares: Array<i64, Ix1>,
    #[serde(with = "ndarray_1d_serde")]
    pub final_prices: Array<f64, Ix1>,
    pub version: u64,
    pub last_updated_height: u32,
    pub total_volume_sats: u64,
    pub outcome_volumes_sats: Vec<u64>,
    pub tx_pow_hash_selector: u8,
    pub tx_pow_ordering: u8,
    pub tx_pow_difficulty: u8,
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
    trading_fee: f64,
    tx_pow_hash_selector: u8,
    tx_pow_ordering: u8,
    tx_pow_difficulty: u8,
}

impl MarketBuilder {
    pub fn new(title: String, creator_address: Address) -> Self {
        Self {
            title,
            description: String::new(),
            tags: Vec::new(),
            creator_address,
            dimension_specs: None,
            trading_fee: DEFAULT_TRADING_FEE,
            tx_pow_hash_selector: 0,
            tx_pow_ordering: 0,
            tx_pow_difficulty: 0,
        }
    }

    pub fn with_description(mut self, desc: String) -> Self {
        self.description = desc;
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

    pub fn with_tx_pow(
        mut self,
        hash_selector: u8,
        ordering: u8,
        difficulty: u8,
    ) -> Self {
        self.tx_pow_hash_selector = hash_selector;
        self.tx_pow_ordering = ordering;
        self.tx_pow_difficulty = difficulty;
        self
    }

    pub fn build(
        self,
        created_at_height: u32,
        expires_at_height: Option<u32>,
        decisions: &HashMap<DecisionId, Decision>,
    ) -> Result<Market, MarketError> {
        let dimension_specs =
            self.dimension_specs.ok_or(MarketError::InvalidDimensions)?;

        let state_combos = generate_state_combos(&dimension_specs, decisions)?;

        let all_decisions: Vec<DecisionId> = dimension_specs
            .iter()
            .map(|spec| match spec {
                DimensionSpec::Single(id) | DimensionSpec::Categorical(id) => {
                    *id
                }
            })
            .collect();

        Market::new(
            self.title,
            self.description,
            self.tags,
            self.creator_address,
            dimension_specs,
            all_decisions,
            state_combos,
            self.trading_fee,
            created_at_height,
            expires_at_height,
            decisions,
            self.tx_pow_hash_selector,
            self.tx_pow_ordering,
            self.tx_pow_difficulty,
        )
    }
}

pub fn generate_state_combos(
    dimension_specs: &[DimensionSpec],
    decisions: &HashMap<DecisionId, Decision>,
) -> Result<Vec<Vec<usize>>, MarketError> {
    use std::collections::HashSet;

    if dimension_specs.is_empty() {
        return Err(MarketError::InvalidDimensions);
    }

    let mut seen_decisions = HashSet::new();
    for spec in dimension_specs {
        let decision_id = match spec {
            DimensionSpec::Single(id) | DimensionSpec::Categorical(id) => *id,
        };
        if !seen_decisions.insert(decision_id) {
            return Err(MarketError::DuplicateDecision { decision_id });
        }
    }

    let mut dimension_ranges = Vec::with_capacity(dimension_specs.len());
    let mut tradeable_ranges = Vec::with_capacity(dimension_specs.len());

    for spec in dimension_specs {
        match spec {
            DimensionSpec::Single(decision_id) => {
                decisions.get(decision_id).ok_or(
                    MarketError::DecisionNotFound {
                        decision_id: *decision_id,
                    },
                )?;
                dimension_ranges.push(3);
                tradeable_ranges.push(2);
            }
            DimensionSpec::Categorical(decision_id) => {
                let decision = decisions.get(decision_id).ok_or(
                    MarketError::DecisionNotFound {
                        decision_id: *decision_id,
                    },
                )?;
                let n = decision
                    .option_count()
                    .ok_or(MarketError::InvalidDimensions)?;
                if n < 2 {
                    return Err(MarketError::InvalidDimensions);
                }
                dimension_ranges.push(n);
                tradeable_ranges.push(n);
            }
        }
    }

    let tradeable_outcomes = markets::outcome_space_size(&tradeable_ranges)
        .ok_or(MarketError::TooManyStates(usize::MAX))?;

    if tradeable_outcomes > MAX_OUTCOMES {
        return Err(MarketError::TooManyStates(tradeable_outcomes));
    }

    Ok(markets::cartesian_product(&dimension_ranges))
}

fn calculate_max_tau(
    decision_ids: &[DecisionId],
    decisions: &HashMap<DecisionId, Decision>,
) -> u8 {
    decision_ids
        .iter()
        .filter_map(|decision_id| decisions.get(decision_id))
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
        decision_ids: Vec<DecisionId>,
        state_combos: Vec<Vec<usize>>,
        trading_fee: f64,
        created_at_height: u32,
        expires_at_height: Option<u32>,
        decisions: &HashMap<DecisionId, Decision>,
        tx_pow_hash_selector: u8,
        tx_pow_ordering: u8,
        tx_pow_difficulty: u8,
    ) -> Result<Self, MarketError> {
        if decision_ids.is_empty() || state_combos.is_empty() {
            return Err(MarketError::InvalidDimensions);
        }

        let mut market = Market {
            id: MarketId([0; 6]),
            title,
            description,
            tags,
            creator_address,
            dimension_specs,
            decision_ids: decision_ids.clone(),
            state_combos,
            created_at_height,
            expires_at_height,
            tau_from_now: calculate_max_tau(&decision_ids, decisions),
            storage_fee_sats: 0,
            market_state: MarketState::Trading,
            trading_fee,
            shares: Array::zeros(0),
            final_prices: Array::zeros(0),
            version: 0,
            last_updated_height: created_at_height,
            total_volume_sats: 0,
            outcome_volumes_sats: Vec::new(),
            tx_pow_hash_selector,
            tx_pow_ordering,
            tx_pow_difficulty,
        };

        let tradeable_outcomes = market.get_valid_state_combos().len();
        market.shares = Array::zeros(tradeable_outcomes);
        market.final_prices = Array::zeros(tradeable_outcomes);
        market.outcome_volumes_sats = vec![0; tradeable_outcomes];
        market.storage_fee_sats =
            markets::market_storage_fee(tradeable_outcomes);

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
    pub fn tx_pow_config(&self) -> crate::types::tx_pow::TxPowConfig {
        crate::types::tx_pow::TxPowConfig {
            hash_selector: self.tx_pow_hash_selector,
            ordering: self.tx_pow_ordering,
            difficulty: self.tx_pow_difficulty,
        }
    }

    pub fn state(&self) -> MarketState {
        self.market_state
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
        height: u32,
        new_market_state: Option<MarketState>,
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

        if let Some(shares) = new_shares {
            if shares.len() != self.shares.len() {
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

    pub fn current_prices(&self, beta: f64) -> Array<f64, Ix1> {
        crate::math::trading::calculate_prices(&self.shares, beta)
            .unwrap_or_else(|_| {
                let n = self.shares.len();
                Array::from_elem(n, 1.0 / n as f64)
            })
    }

    /// For single-decision markets (binary or scaled), calculate the implied value (0-1).
    /// Returns p_max / (p_min + p_max) where:
    /// - p_min is the price of outcome 0 (No/Min)
    /// - p_max is the price of outcome 1 (Yes/Max)
    ///
    /// For binary decisions: 0.0 = "No", 1.0 = "Yes"
    /// For scaled decisions: 0.0 = min bound, 1.0 = max bound
    ///
    /// Returns None if market has multiple decisions or invalid structure.
    pub fn get_implied_value_normalized(&self, beta: f64) -> Option<f64> {
        if self.decision_ids.len() != 1 || self.state_combos.len() != 3 {
            return None;
        }

        let prices = self.current_prices(beta);
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

    pub fn update_shares(
        &mut self,
        new_shares: Array<i64, Ix1>,
        height: u32,
    ) -> Result<(), MarketError> {
        self.update_state(height, None, Some(new_shares), None)
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

    pub fn cancel_market(
        &mut self,
        _transaction_id: Option<[u8; 32]>,
        height: u32,
    ) -> Result<(), MarketError> {
        self.update_state(height, Some(MarketState::Cancelled), None, None)
    }

    pub fn invalidate_market(
        &mut self,
        _transaction_id: Option<[u8; 32]>,
        height: u32,
    ) -> Result<(), MarketError> {
        self.update_state(height, Some(MarketState::Invalid), None, None)
    }

    pub fn calculate_final_prices(
        &self,
        decision_outcomes: &std::collections::HashMap<DecisionId, f64>,
        decisions: &HashMap<DecisionId, Decision>,
    ) -> Result<Array<f64, Ix1>, MarketError> {
        let mut axes: Vec<Vec<f64>> =
            Vec::with_capacity(self.dimension_specs.len());

        for spec in &self.dimension_specs {
            match spec {
                DimensionSpec::Single(decision_id) => {
                    let outcome = decision_outcomes
                        .get(decision_id)
                        .copied()
                        .unwrap_or(0.5);
                    let is_scaled = decisions
                        .get(decision_id)
                        .map(|d| d.is_scaled())
                        .unwrap_or(false);

                    let axis = if is_scaled {
                        vec![1.0 - outcome, outcome]
                    } else if outcome > 0.7 {
                        vec![0.0, 1.0]
                    } else if outcome < 0.3 {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.5, 0.5]
                    };
                    axes.push(axis);
                }
                DimensionSpec::Categorical(decision_id) => {
                    let outcome_opt =
                        decision_outcomes.get(decision_id).copied();
                    let n = decisions
                        .get(decision_id)
                        .and_then(|d| d.option_count())
                        .unwrap_or(2);

                    let real_winner: Option<usize> =
                        outcome_opt.and_then(|o| {
                            if o == 0.5 || !o.is_finite() {
                                return None;
                            }
                            let rounded = o.round();
                            if (o - rounded).abs() > f64::EPSILON {
                                return None;
                            }
                            let idx = rounded as i64;
                            if idx >= 0 && (idx as usize) < n {
                                Some(idx as usize)
                            } else {
                                None
                            }
                        });

                    let mut axis = Vec::with_capacity(n);
                    if let Some(winner) = real_winner {
                        for k in 0..n {
                            axis.push(if k == winner { 1.0 } else { 0.0 });
                        }
                    } else {
                        let equal = 1.0 / n as f64;
                        for _ in 0..n {
                            axis.push(equal);
                        }
                    }
                    axes.push(axis);
                }
            }
        }

        let valid_combos = self.get_valid_state_combos();
        let mut prices = Array::zeros(valid_combos.len());

        for (i, (_full_idx, state_combo)) in valid_combos.iter().enumerate() {
            let mut joint_prob = 1.0;
            for (dim, &state) in state_combo.iter().enumerate() {
                if dim < axes.len() && state < axes[dim].len() {
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
            .filter(|(_, combo)| {
                combo.iter().zip(self.dimension_specs.iter()).all(
                    |(&state, spec)| match spec {
                        DimensionSpec::Single(_) => state < 2,
                        DimensionSpec::Categorical(_) => true,
                    },
                )
            })
            .collect()
    }

    pub fn get_storage_fee_sats(&self) -> u64 {
        self.storage_fee_sats
    }

    pub fn tradeable_index_for_positions(
        &self,
        positions: &[usize],
    ) -> Result<usize, MarketError> {
        if positions.len() != self.decision_ids.len() {
            return Err(MarketError::InvalidDimensions);
        }

        for (tradeable_idx, (_full_idx, combo)) in
            self.get_valid_state_combos().iter().enumerate()
        {
            if *combo == positions {
                return Ok(tradeable_idx);
            }
        }

        Err(MarketError::InvalidOutcomeCombination)
    }

    pub fn get_outcome_price(
        &self,
        positions: &[usize],
        beta: f64,
    ) -> Result<f64, MarketError> {
        let index = self.tradeable_index_for_positions(positions)?;
        let prices = self.current_prices(beta);

        Ok(prices[index])
    }

    pub fn describe_outcome_by_state(
        &self,
        state_index: usize,
        decisions: &HashMap<DecisionId, Decision>,
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
        decisions: &HashMap<DecisionId, Decision>,
    ) -> Result<String, MarketError> {
        if positions.len() != self.decision_ids.len() {
            return Err(MarketError::InvalidDimensions);
        }

        let mut description = Vec::new();

        for (i, &decision_id) in self.decision_ids.iter().enumerate() {
            let decision = decisions
                .get(&decision_id)
                .ok_or(MarketError::DecisionNotFound { decision_id })?;

            let outcome_desc = if decision.is_categorical() {
                let labels = decision.get_category_labels().unwrap_or(&[]);
                let label = labels
                    .get(positions[i])
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown");
                format!("{}: {}", decision.header, label)
            } else if decision.is_scaled() {
                match positions[i] {
                    0 => format!(
                        "{}: {} (Min)",
                        decision.header,
                        decision.scale_min().unwrap_or(0.0)
                    ),
                    1 => format!(
                        "{}: {} (Max)",
                        decision.header,
                        decision.scale_max().unwrap_or(100.0)
                    ),
                    _ => format!("{}: Abstain", decision.header),
                }
            } else {
                match positions[i] {
                    0 => format!("{}: No", decision.header),
                    1 => format!("{}: Yes", decision.header),
                    _ => format!("{}: Abstain", decision.header),
                }
            };

            description.push(outcome_desc);
        }

        Ok(description.join(", "))
    }
}
