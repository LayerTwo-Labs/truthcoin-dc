//! Undo data for disconnect_tip chain reorganization support.
//!
//! During block connection, irreversible state changes are captured into
//! undo records keyed by block height. During disconnect_tip, these records
//! are loaded and applied in reverse to fully revert the block's effects.

use crate::state::markets::{Market, MarketId, MarketPayoutSummary};
use crate::state::voting::types::{VoterReputation, VotingPeriodId};
use crate::types::{Address, FilledOutput, OutPoint};
use serde::{Deserialize, Serialize};

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
/// and votecoin redistribution during disconnect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsensusUndoData {
    pub entries: Vec<ConsensusUndoEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsensusUndoEntry {
    pub period_id: VotingPeriodId,
    /// Previous voter reputations (None = voter didn't exist before)
    pub previous_voter_reputations: Vec<(Address, Option<VoterReputation>)>,
    /// Decision outcome slot IDs that were written (to be deleted on revert)
    pub decision_outcome_slot_ids: Vec<crate::state::slots::SlotId>,
    /// Whether period stats existed before (to delete on revert)
    pub had_period_stats: bool,
    /// Slot IDs that were transitioned to Resolved (to revert back to Voting)
    pub resolved_slot_ids: Vec<crate::state::slots::SlotId>,
    /// Votecoin UTXOs created during redistribution (to delete on revert)
    pub created_votecoin_utxos: Vec<OutPoint>,
    /// Votecoin UTXOs consumed during redistribution (to restore on revert)
    pub consumed_votecoin_utxos: Vec<(OutPoint, FilledOutput)>,
    /// Whether a pending redistribution record was written
    pub had_pending_redistribution: bool,
    /// Previous pending redistribution state (to restore on revert)
    pub previous_pending_redistribution:
        Option<crate::state::voting::redistribution::PeriodRedistribution>,
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
    /// Pending market funds UTXOs that were cleared (to restore)
    pub old_pending_utxos: Vec<(OutPoint, bool)>,
    /// New treasury UTXO created (to delete on revert)
    pub new_treasury_utxo: Option<OutPoint>,
    /// New fee UTXO created (to delete on revert)
    pub new_fee_utxo: Option<OutPoint>,
    /// Sell payout UTXOs created (to delete on revert)
    pub sell_payout_utxos: Vec<OutPoint>,
    /// Buy change UTXOs created (to delete on revert)
    pub buy_change_utxos: Vec<OutPoint>,
}
