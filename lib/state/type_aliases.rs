//! Type aliases to reduce complexity in function signatures.
//!
//! These aliases provide semantic meaning to complex tuple types used
//! throughout the state management code.

use super::markets::MarketId;
use crate::types::Address;
use std::collections::HashMap;

/// A pair of (period_id, slot_count) used in period summaries.
pub type PeriodSlotPair = (u32, u64);

/// Period summary data: (active_periods, voting_periods).
pub type PeriodSummary = (Vec<PeriodSlotPair>, Vec<PeriodSlotPair>);

/// A position in a market: (market_id, outcome_index, share_balance).
pub type MarketPosition = (MarketId, u32, f64);

/// All share accounts: a list of (address, positions) pairs.
pub type AllShareAccounts = Vec<(Address, Vec<MarketPosition>)>;

/// Shareholders of a specific market: (address, positions) where positions are (outcome_index, share_balance).
pub type MarketShareholders = Vec<(Address, Vec<(u32, f64)>)>;

/// A single market delta: (outcome_index, share_delta, volume_sats, fee_sats, transaction_id).
pub type MarketDelta = (usize, f64, Option<u64>, Option<u64>, Option<[u8; 32]>);

/// Aggregated deltas per market for block processing.
pub type AggregatedDeltas = HashMap<MarketId, Vec<MarketDelta>>;
