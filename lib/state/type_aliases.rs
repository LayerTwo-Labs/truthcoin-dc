//! Type aliases to reduce complexity in function signatures.
//!
//! These aliases provide semantic meaning to complex tuple types used
//! throughout the state management code.

use super::markets::MarketId;
use crate::types::Address;

/// A pair of (period_id, decision_count) used in period summaries.
pub type PeriodDecisionPair = (u32, u64);

/// Period summary data: (active_periods, voting_periods).
pub type PeriodSummary = (Vec<PeriodDecisionPair>, Vec<PeriodDecisionPair>);

/// A position in a market: (market_id, outcome_index, share_balance).
pub type MarketPosition = (MarketId, u32, i64);

/// All share accounts: a list of (address, positions) pairs.
pub type AllShareAccounts = Vec<(Address, Vec<MarketPosition>)>;

/// Shareholders of a specific market: (address, positions) where positions are (outcome_index, share_balance).
pub type MarketShareholders = Vec<(Address, Vec<(u32, i64)>)>;
