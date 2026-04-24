use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::math::lmsr::MAX_OUTCOMES;
use crate::state::decisions::DecisionId;
use crate::types::Address;
use thiserror::Error as ThisError;

/// Default LMSR beta parameter (liquidity depth)
pub const DEFAULT_MARKET_BETA: f64 = 7.0;

/// Default trading fee (0.5%)
pub const DEFAULT_TRADING_FEE: f64 = 0.005;

#[derive(Debug, ThisError, Clone)]
pub enum MarketError {
    #[error("Invalid market dimensions")]
    InvalidDimensions,

    #[error("Too many market states: {0} (max {MAX_OUTCOMES})")]
    TooManyStates(usize),

    #[error("Invalid outcome index: {0}")]
    InvalidOutcomeIndex(usize),

    #[error("Invalid state transition from {from:?} to {to:?}")]
    InvalidStateTransition { from: MarketState, to: MarketState },

    #[error("Market not found: {id:?}")]
    MarketNotFound { id: MarketId },

    #[error("Decision not found: {decision_id:?}")]
    DecisionNotFound { decision_id: DecisionId },

    #[error("Decision validation failed for decision: {decision_id:?}")]
    DecisionValidationFailed { decision_id: DecisionId },

    #[error("Invalid outcome combination")]
    InvalidOutcomeCombination,

    #[error("Duplicate decision in market dimensions: {decision_id:?}")]
    DuplicateDecision { decision_id: DecisionId },

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
            (Trading, Trading)
            | (Cancelled, Cancelled)
            | (Invalid, Invalid)
            | (Ossified, Ossified) => true,
            (Trading, Cancelled | Invalid | Ossified) => true,
            (Invalid, Ossified) => true,
            (Cancelled, Trading | Invalid | Ossified) => false,
            (Invalid, Trading | Cancelled) => false,
            (Ossified, Trading | Cancelled | Invalid) => false,
        }
    }

    pub fn allows_trading(&self) -> bool {
        matches!(self, MarketState::Trading)
    }
}

#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, BorshSerialize,
)]
pub enum DimensionSpec {
    Single(DecisionId),
    Categorical(DecisionId),
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
            let decision_id = parse_single_decision(categorical_str.trim())?;
            dimensions.push(DimensionSpec::Categorical(decision_id));
        } else {
            let start = i;
            while i < chars.len() && chars[i] != ',' && chars[i] != '[' {
                i += 1;
            }

            let decision_str: String = chars[start..i].iter().collect();
            let decision_id = parse_single_decision(decision_str.trim())?;
            dimensions.push(DimensionSpec::Single(decision_id));
        }
    }

    Ok(dimensions)
}

fn parse_single_decision(
    decision_str: &str,
) -> Result<DecisionId, MarketError> {
    let decision_bytes = hex::decode(decision_str)
        .map_err(|_| MarketError::InvalidDimensions)?;

    if decision_bytes.len() != 3 {
        return Err(MarketError::InvalidDimensions);
    }

    let decision_id_array: [u8; 3] = decision_bytes
        .try_into()
        .map_err(|_| MarketError::InvalidDimensions)?;
    DecisionId::from_bytes(decision_id_array)
        .map_err(|_| MarketError::InvalidDimensions)
}

#[derive(
    Debug,
    Clone,
    Copy,
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
    pub last_updated_height: u32,
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
            .get(&(*market_id, outcome_index))
            .copied()
            .unwrap_or(0)
    }

    pub fn add_shares(
        &mut self,
        market_id: MarketId,
        outcome_index: u32,
        shares: i64,
        height: u32,
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
        height: u32,
    ) -> Result<(), MarketError> {
        let key = (*market_id, outcome_index);
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FeeRole {
    CorrectVoter,
    DecisionAuthor,
    MarketAuthor,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeePayoutRecord {
    pub address: Address,
    pub amount_sats: u64,
    pub fee_role: FeeRole,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreatorRefund {
    pub address: Address,
    pub amount_sats: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketPayoutSummary {
    pub market_id: MarketId,
    pub treasury_distributed: u64,
    pub total_fees_distributed: u64,
    pub shareholder_count: u32,
    pub payouts: Vec<SharePayoutRecord>,
    pub fee_payouts: Vec<FeePayoutRecord>,
    pub creator_refund: Option<CreatorRefund>,
    pub block_height: u32,
}
