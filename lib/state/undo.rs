//! Undo data for disconnect_tip chain reorganization support.
//!
//! During block connection, irreversible state changes are captured into
//! undo records keyed by block height. During disconnect_tip, these records
//! are loaded and applied in reverse to fully revert the block's effects.

use crate::state::markets::{Market, MarketId, MarketPayoutSummary};
use crate::state::voting::types::VotingPeriodId;
use crate::types::{Address, FilledOutput, OutPoint};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Undo data for market ossification and automatic payouts (C1).
///
/// Captured before `transition_and_payout_resolved_markets` during connect.
/// Used to restore markets from Ossified back to Trading during disconnect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OssificationUndoData {
    pub entries: Vec<OssificationUndoEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OssificationUndoEntry {
    /// The complete market state before ossification (Trading state, no final prices)
    pub pre_ossification_market: Market,
    /// The payout summary that was applied (needed for revert_automatic_share_payouts)
    pub payout_summary: MarketPayoutSummary,
    /// Treasury UTXO that existed before payouts consumed it
    pub treasury_utxo: Option<(OutPoint, FilledOutput)>,
    /// Fee UTXO that existed before payouts consumed it
    pub fee_utxo: Option<(OutPoint, FilledOutput)>,
}

/// Undo data for consensus voting state commit (C2).
///
/// Captured before `commit_consensus_result` during connect.
/// Used to revert voter reputations, decision outcomes, period stats,
/// and reputation redistribution during disconnect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsensusUndoData {
    pub entries: Vec<ConsensusUndoEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsensusUndoEntry {
    pub period_id: VotingPeriodId,
    pub decision_outcome_ids: Vec<crate::state::decisions::DecisionId>,
    pub had_period_stats: bool,
    pub resolved_decision_ids: Vec<crate::state::decisions::DecisionId>,
    pub pre_consensus_reputation: BTreeMap<Address, f64>,
}

/// Undo data for reputation transfers within a block.
///
/// Reputation transfers happen in the transaction processing phase,
/// after consensus. On disconnect, these are reverted before consensus
/// undo restores the pre-consensus reputation snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReputationTransferUndoData {
    pub entries: Vec<ReputationTransferUndoEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReputationTransferUndoEntry {
    pub sender: Address,
    pub sender_pre_reputation: f64,
    pub receiver: Address,
    pub receiver_pre_reputation: f64,
}

/// Undo data for market treasury UTXO consolidation (C3).
///
/// Captured before `consolidate_market_utxos` during connect.
/// Used to restore pre-consolidation UTXO state during disconnect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsolidationUndoData {
    pub entries: Vec<ConsolidationUndoEntry>,
    /// Sell input change UTXOs created (to delete on revert)
    pub sell_input_change_utxos: Vec<OutPoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsolidationUndoEntry {
    pub market_id: MarketId,
    /// Old treasury UTXOs that were consumed (outpoint + filled output)
    pub old_treasury_utxos: Vec<(OutPoint, FilledOutput)>,
    /// Old fee UTXOs that were consumed
    pub old_fee_utxos: Vec<(OutPoint, FilledOutput)>,
    /// Old market_funds_utxo pointer for treasury (to restore)
    pub old_treasury_pointer: Option<OutPoint>,
    /// Old market_funds_utxo pointer for fees (to restore)
    pub old_fee_pointer: Option<OutPoint>,
    /// New treasury UTXO created (to delete on revert)
    pub new_treasury_utxo: Option<OutPoint>,
    /// New fee UTXO created (to delete on revert)
    pub new_fee_utxo: Option<OutPoint>,
    /// Sell payout UTXOs created (to delete on revert)
    pub sell_payout_utxos: Vec<OutPoint>,
    /// Buy change UTXOs created (to delete on revert)
    pub buy_change_utxos: Vec<OutPoint>,
}
