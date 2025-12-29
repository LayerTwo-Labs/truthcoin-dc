use crate::state::rollback::{RollBack, TxidStamped};
use crate::state::slots::SlotId;
use crate::types::{Address, Txid, hashes};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    PartialOrd,
    Ord,
    Deserialize,
    Serialize,
    utoipa::ToSchema,
)]
pub struct VotingPeriodId(pub u32);

impl VotingPeriodId {
    pub const fn new(period: u32) -> Self {
        Self(period)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }

    pub fn as_bytes(self) -> [u8; 4] {
        self.0.to_be_bytes()
    }

    pub fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(u32::from_be_bytes(bytes))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VotingPeriodStatus {
    Pending,
    Active,
    Closed,
    Resolved,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VotingPeriod {
    pub id: VotingPeriodId,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub status: VotingPeriodStatus,
    pub decision_slots: Vec<SlotId>,
}

impl VotingPeriod {
    pub fn new(
        id: VotingPeriodId,
        start_timestamp: u64,
        end_timestamp: u64,
        decision_slots: Vec<SlotId>,
    ) -> Self {
        Self {
            id,
            start_timestamp,
            end_timestamp,
            status: VotingPeriodStatus::Pending,
            decision_slots,
        }
    }

    pub fn is_active(&self, current_timestamp: u64) -> bool {
        self.status == VotingPeriodStatus::Active
            && current_timestamp >= self.start_timestamp
            && current_timestamp < self.end_timestamp
    }

    pub fn has_ended(&self, current_timestamp: u64) -> bool {
        current_timestamp >= self.end_timestamp
    }

    pub fn duration_seconds(&self) -> u64 {
        self.end_timestamp.saturating_sub(self.start_timestamp)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum VoteValue {
    Binary(bool),
    Scalar(f64),
    Abstain,
}

impl VoteValue {
    /// Convert vote to Option<f64>, returning None for Abstain.
    pub fn to_float_opt(&self) -> Option<f64> {
        match self {
            VoteValue::Binary(false) => Some(0.0),
            VoteValue::Binary(true) => Some(1.0),
            VoteValue::Scalar(value) => Some(*value),
            VoteValue::Abstain => None,
        }
    }

    pub fn is_abstain(&self) -> bool {
        matches!(self, VoteValue::Abstain)
    }

    pub fn binary(value: bool) -> Self {
        VoteValue::Binary(value)
    }

    pub fn scalar(value: f64) -> Self {
        VoteValue::Scalar(value)
    }

    pub fn abstain() -> Self {
        VoteValue::Abstain
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vote {
    pub voter_address: Address,
    pub period_id: VotingPeriodId,
    pub decision_id: SlotId,
    pub value: VoteValue,
    pub timestamp: u64,
    pub block_height: u64,
    pub tx_hash: [u8; 32],
}

impl Vote {
    pub fn new(
        voter_address: Address,
        period_id: VotingPeriodId,
        decision_id: SlotId,
        value: VoteValue,
        timestamp: u64,
        block_height: u64,
        tx_hash: [u8; 32],
    ) -> Self {
        Self {
            voter_address,
            period_id,
            decision_id,
            value,
            timestamp,
            block_height,
            tx_hash,
        }
    }

    pub fn compute_hash(&self) -> [u8; 32] {
        let vote_data = (
            &self.voter_address.0,
            self.period_id.as_bytes(),
            self.decision_id.as_bytes(),
        );
        hashes::hash(&vote_data)
    }
}

/// Voter reputation with Votecoin integration.
/// Final Voting Weight = Base Reputation × Votecoin Holdings Proportion
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoterReputation {
    pub address: Address,
    pub reputation: f64,
    pub votecoin_proportion: f64,
    pub total_decisions: u64,
    pub correct_decisions: u64,
    pub last_updated: u64,
    pub last_period: VotingPeriodId,
    pub votecoin_updated_height: Option<u64>,
    pub reputation_history: RollBack<TxidStamped<f64>>,
}

impl VoterReputation {
    pub fn new(
        address: Address,
        initial_reputation: f64,
        timestamp: u64,
        period_id: VotingPeriodId,
    ) -> Self {
        let reputation = initial_reputation.clamp(
            crate::math::voting::constants::REPUTATION_MIN,
            crate::math::voting::constants::REPUTATION_MAX,
        );

        let initial_txid = Txid([0u8; 32]);
        let reputation_history =
            RollBack::<TxidStamped<f64>>::new(reputation, initial_txid, 0);

        Self {
            address,
            reputation,
            votecoin_proportion: 0.0,
            total_decisions: 0,
            correct_decisions: 0,
            last_updated: timestamp,
            last_period: period_id,
            votecoin_updated_height: None,
            reputation_history,
        }
    }

    pub fn new_default(
        address: Address,
        timestamp: u64,
        period_id: VotingPeriodId,
    ) -> Self {
        Self::new(
            address,
            crate::math::voting::constants::BITCOIN_HIVEMIND_NEUTRAL_VALUE,
            timestamp,
            period_id,
        )
    }

    pub fn update(
        &mut self,
        was_correct: bool,
        timestamp: u64,
        period_id: VotingPeriodId,
        txid: Txid,
        height: u32,
    ) {
        self.reputation_history.push(self.reputation, txid, height);

        self.total_decisions += 1;
        if was_correct {
            self.correct_decisions += 1;
        }

        let accuracy_rate = if self.total_decisions > 0 {
            self.correct_decisions as f64 / self.total_decisions as f64
        } else {
            0.0
        };
        self.reputation = accuracy_rate.clamp(0.0, 1.0);
        self.last_updated = timestamp;
        self.last_period = period_id;
    }

    pub fn rollback_update(&mut self) -> Option<f64> {
        if let Some(previous) = self.reputation_history.pop() {
            let previous_reputation = previous.data;
            self.reputation = previous_reputation;
            self.total_decisions = self.total_decisions.saturating_sub(1);
            Some(previous_reputation)
        } else {
            None
        }
    }

    pub fn update_votecoin_proportion(
        &mut self,
        votecoin_proportion: f64,
        current_height: u64,
    ) {
        self.votecoin_proportion = votecoin_proportion.clamp(0.0, 1.0);
        self.votecoin_updated_height = Some(current_height);
    }

    pub fn get_voting_weight(&self) -> f64 {
        self.reputation * self.votecoin_proportion
    }

    pub fn get_base_reputation(&self) -> f64 {
        self.reputation
    }

    pub fn get_votecoin_proportion(&self) -> f64 {
        self.votecoin_proportion
    }

    pub fn get_accuracy_rate(&self) -> f64 {
        if self.total_decisions > 0 {
            self.correct_decisions as f64 / self.total_decisions as f64
        } else {
            0.0
        }
    }

    pub fn needs_votecoin_refresh(
        &self,
        current_height: u64,
        max_staleness: u64,
    ) -> bool {
        match self.votecoin_updated_height {
            None => true,
            Some(last_height) => {
                current_height.saturating_sub(last_height) > max_staleness
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionResolutionStatus {
    Pending,
    AwaitingResolution,
    Resolved,
    Defaulted,
    Cancelled,
}

impl DecisionResolutionStatus {
    pub fn accepts_votes(&self) -> bool {
        matches!(self, DecisionResolutionStatus::Pending)
    }

    pub fn is_finalized(&self) -> bool {
        matches!(
            self,
            DecisionResolutionStatus::Resolved
                | DecisionResolutionStatus::Defaulted
                | DecisionResolutionStatus::Cancelled
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionResolution {
    pub decision_id: SlotId,
    pub period_id: VotingPeriodId,
    pub status: DecisionResolutionStatus,
    pub status_changed_at: u64,
    pub status_changed_height: u64,
    pub vote_count: u32,
    pub min_votes_required: u32,
    pub voting_deadline: u64,
    pub outcome_ready: bool,
    pub reason: Option<String>,
}

impl DecisionResolution {
    pub fn new(
        decision_id: SlotId,
        period_id: VotingPeriodId,
        voting_deadline: u64,
        min_votes_required: u32,
        current_timestamp: u64,
        current_height: u64,
    ) -> Self {
        Self {
            decision_id,
            period_id,
            status: DecisionResolutionStatus::Pending,
            status_changed_at: current_timestamp,
            status_changed_height: current_height,
            vote_count: 0,
            min_votes_required,
            voting_deadline,
            outcome_ready: false,
            reason: None,
        }
    }

    pub fn update_status(
        &mut self,
        new_status: DecisionResolutionStatus,
        timestamp: u64,
        block_height: u64,
        reason: Option<String>,
    ) {
        self.status = new_status;
        self.status_changed_at = timestamp;
        self.status_changed_height = block_height;
        self.reason = reason;
    }

    pub fn add_vote(&mut self) {
        if self.status.accepts_votes() {
            self.vote_count += 1;
        }
    }

    pub fn is_voting_expired(&self, current_timestamp: u64) -> bool {
        current_timestamp >= self.voting_deadline
    }

    pub fn has_minimum_votes(&self) -> bool {
        self.vote_count >= self.min_votes_required
    }

    pub fn is_ready_for_consensus(&self, current_timestamp: u64) -> bool {
        matches!(self.status, DecisionResolutionStatus::Pending)
            && (self.is_voting_expired(current_timestamp)
                || self.has_minimum_votes())
    }

    pub fn mark_outcome_ready(&mut self) {
        self.outcome_ready = true;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionOutcome {
    pub decision_id: SlotId,
    pub period_id: VotingPeriodId,
    pub outcome_value: f64,
    pub min: f64,
    pub max: f64,
    pub confidence: f64,
    pub total_votes: u64,
    pub total_reputation_weight: f64,
    pub finalized_at: u64,
    pub block_height: u64,
    pub is_consensus: bool,
    pub resolution: DecisionResolution,
}

impl DecisionOutcome {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        decision_id: SlotId,
        period_id: VotingPeriodId,
        outcome_value: f64,
        min: f64,
        max: f64,
        confidence: f64,
        total_votes: u64,
        total_reputation_weight: f64,
        finalized_at: u64,
        block_height: u64,
        is_consensus: bool,
        resolution: DecisionResolution,
    ) -> Self {
        Self {
            decision_id,
            period_id,
            outcome_value,
            min,
            max,
            confidence: confidence.clamp(0.0, 1.0),
            total_votes,
            total_reputation_weight,
            finalized_at,
            block_height,
            is_consensus,
            resolution,
        }
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
)]
pub struct VoteMatrixKey {
    pub period_id: VotingPeriodId,
    pub voter_address: Address,
    pub decision_id: SlotId,
}

impl VoteMatrixKey {
    pub fn new(
        period_id: VotingPeriodId,
        voter_address: Address,
        decision_id: SlotId,
    ) -> Self {
        Self {
            period_id,
            voter_address,
            decision_id,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoteMatrixEntry {
    pub value: VoteValue,
    pub timestamp: u64,
    pub block_height: u64,
}

impl VoteMatrixEntry {
    pub fn new(value: VoteValue, timestamp: u64, block_height: u64) -> Self {
        Self {
            value,
            timestamp,
            block_height,
        }
    }

    pub fn to_float_opt(&self) -> Option<f64> {
        self.value.to_float_opt()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoteBatch {
    pub period_id: VotingPeriodId,
    pub votes: Vec<Vote>,
    pub created_at: u64,
    pub block_height: u64,
}

impl VoteBatch {
    pub fn new(
        period_id: VotingPeriodId,
        created_at: u64,
        block_height: u64,
    ) -> Self {
        Self {
            period_id,
            votes: Vec::new(),
            created_at,
            block_height,
        }
    }

    pub fn add_vote(&mut self, vote: Vote) {
        self.votes.push(vote);
    }

    pub fn len(&self) -> usize {
        self.votes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.votes.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VotingPeriodStats {
    pub period_id: VotingPeriodId,
    pub total_voters: u64,
    pub total_votes: u64,
    pub total_decisions: u64,
    pub avg_participation_rate: f64,
    pub total_reputation_weight: f64,
    pub consensus_decisions: u64,
    pub calculated_at: u64,
    pub first_loading: Option<Vec<f64>>,
    pub certainty: Option<f64>,
    pub reputation_changes: Option<HashMap<Address, (f64, f64)>>,
}

impl VotingPeriodStats {
    /// Returns the absolute slot ids referenced in the reputation_changes field.
    pub fn get_changed_addresses(&self) -> Vec<Address> {
        self.reputation_changes
            .as_ref()
            .map(|changes| changes.keys().copied().collect())
            .unwrap_or_default()
    }

    pub fn new(period_id: VotingPeriodId, calculated_at: u64) -> Self {
        Self {
            period_id,
            total_voters: 0,
            total_votes: 0,
            total_decisions: 0,
            avg_participation_rate: 0.0,
            total_reputation_weight: 0.0,
            consensus_decisions: 0,
            calculated_at,
            first_loading: None,
            certainty: None,
            reputation_changes: None,
        }
    }

    pub fn participation_rate(&self) -> f64 {
        if self.total_decisions > 0 && self.total_voters > 0 {
            self.total_votes as f64
                / (self.total_decisions * self.total_voters) as f64
        } else {
            0.0
        }
    }

    pub fn consensus_rate(&self) -> f64 {
        if self.total_decisions > 0 {
            self.consensus_decisions as f64 / self.total_decisions as f64
        } else {
            0.0
        }
    }
}

#[derive(Clone, Debug)]
pub struct SlotResolution {
    pub slot_id: SlotId,
    pub outcome_value: f64,
}

/// Holds all pending changes from consensus calculation for atomic commit.
#[derive(Clone, Debug)]
pub struct ConsensusResult {
    pub period_id: VotingPeriodId,
    pub voter_reputations: Vec<VoterReputation>,
    pub period_stats: VotingPeriodStats,
    pub decision_outcomes: Vec<DecisionOutcome>,
    pub slot_resolutions: Vec<SlotResolution>,
    pub period_redistribution:
        Option<super::redistribution::PeriodRedistribution>,
    pub redistribution_summary:
        Option<super::redistribution::RedistributionSummary>,
    pub block_height: u64,
    pub timestamp: u64,
}

impl ConsensusResult {
    pub fn new(
        period_id: VotingPeriodId,
        block_height: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            period_id,
            voter_reputations: Vec::new(),
            period_stats: VotingPeriodStats::new(period_id, timestamp),
            decision_outcomes: Vec::new(),
            slot_resolutions: Vec::new(),
            period_redistribution: None,
            redistribution_summary: None,
            block_height,
            timestamp,
        }
    }

    pub fn has_outcomes(&self) -> bool {
        !self.decision_outcomes.is_empty()
    }

    pub fn resolved_slot_count(&self) -> usize {
        self.slot_resolutions.len()
    }

    pub fn abstained_slot_count(&self) -> usize {
        self.decision_outcomes
            .len()
            .saturating_sub(self.slot_resolutions.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Address;

    #[test]
    fn test_voting_period_id() {
        let period_id = VotingPeriodId::new(42);
        assert_eq!(period_id.as_u32(), 42);

        let bytes = period_id.as_bytes();
        let reconstructed = VotingPeriodId::from_bytes(bytes);
        assert_eq!(period_id, reconstructed);
    }

    #[test]
    fn test_vote_value() {
        let binary_true = VoteValue::binary(true);
        assert_eq!(binary_true.to_float_opt(), Some(1.0));
        assert!(!binary_true.is_abstain());

        let binary_false = VoteValue::binary(false);
        assert_eq!(binary_false.to_float_opt(), Some(0.0));

        let scalar = VoteValue::scalar(0.75);
        assert_eq!(scalar.to_float_opt(), Some(0.75));

        let abstain = VoteValue::abstain();
        assert!(abstain.is_abstain());
        assert_eq!(abstain.to_float_opt(), None);
    }

    #[test]
    fn test_voting_period() {
        let start = 1000;
        let end = 2000;
        let period =
            VotingPeriod::new(VotingPeriodId::new(1), start, end, vec![]);

        assert_eq!(period.duration_seconds(), 1000);
        assert!(!period.is_active(500));
        assert!(!period.is_active(2500));
        assert!(period.has_ended(2500));
        assert!(!period.has_ended(500));
    }

    #[test]
    fn test_voter_reputation() {
        let address = Address([1u8; 20]);
        let mut reputation =
            VoterReputation::new(address, 0.5, 1000, VotingPeriodId::new(1));

        assert_eq!(reputation.reputation, 0.5);
        assert_eq!(reputation.total_decisions, 0);

        reputation.update(
            true,
            1100,
            VotingPeriodId::new(2),
            Txid([1u8; 32]),
            1,
        );
        assert_eq!(reputation.total_decisions, 1);
        assert_eq!(reputation.correct_decisions, 1);
        assert_eq!(reputation.get_accuracy_rate(), 1.0);
        assert_eq!(reputation.reputation, 1.0);

        reputation.update(
            false,
            1200,
            VotingPeriodId::new(3),
            Txid([2u8; 32]),
            2,
        );
        assert_eq!(reputation.total_decisions, 2);
        assert_eq!(reputation.correct_decisions, 1);
        assert_eq!(reputation.get_accuracy_rate(), 0.5);
        assert_eq!(reputation.reputation, 0.5);
    }

    #[test]
    fn test_vote_batch() {
        let period_id = VotingPeriodId::new(1);
        let mut batch = VoteBatch::new(period_id, 1000, 100);

        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);

        let vote = Vote::new(
            Address([1u8; 20]),
            period_id,
            SlotId::new(1, 0).unwrap(),
            VoteValue::binary(true),
            1000,
            100,
            [0u8; 32],
        );

        batch.add_vote(vote);
        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_voting_period_stats() {
        let period_id = VotingPeriodId::new(1);
        let mut stats = VotingPeriodStats::new(period_id, 1000);

        stats.total_voters = 10;
        stats.total_votes = 80;
        stats.total_decisions = 10;
        stats.consensus_decisions = 8;

        assert_eq!(stats.participation_rate(), 0.8);
        assert_eq!(stats.consensus_rate(), 0.8);
    }

    #[test]
    fn test_vote_value_to_float_opt() {
        assert_eq!(VoteValue::binary(true).to_float_opt(), Some(1.0));
        assert_eq!(VoteValue::binary(false).to_float_opt(), Some(0.0));
        assert_eq!(VoteValue::scalar(0.75).to_float_opt(), Some(0.75));
        assert_eq!(VoteValue::abstain().to_float_opt(), None);
    }
}
