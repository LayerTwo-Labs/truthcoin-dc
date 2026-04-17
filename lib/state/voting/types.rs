use crate::state::decisions::DecisionId;
use crate::types::Address;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
    pub decision_ids: Vec<DecisionId>,
}

impl VotingPeriod {
    pub fn new(
        id: VotingPeriodId,
        start_timestamp: u64,
        end_timestamp: u64,
        decision_ids: Vec<DecisionId>,
    ) -> Self {
        Self {
            id,
            start_timestamp,
            end_timestamp,
            status: VotingPeriodStatus::Pending,
            decision_ids,
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
    Categorical(u16),
    Abstain,
}

impl VoteValue {
    pub fn to_float_opt(&self) -> Option<f64> {
        match self {
            VoteValue::Binary(false) => Some(0.0),
            VoteValue::Binary(true) => Some(1.0),
            VoteValue::Scalar(value) => Some(*value),
            VoteValue::Categorical(idx) => Some(*idx as f64),
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

    pub fn categorical(index: u16) -> Self {
        VoteValue::Categorical(index)
    }

    pub fn abstain() -> Self {
        VoteValue::Abstain
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vote {
    pub voter_address: Address,
    pub period_id: VotingPeriodId,
    pub decision_id: DecisionId,
    pub value: VoteValue,
    pub timestamp: u64,
    pub block_height: u32,
    pub tx_hash: [u8; 32],
}

impl Vote {
    pub fn new(
        voter_address: Address,
        period_id: VotingPeriodId,
        decision_id: DecisionId,
        value: VoteValue,
        timestamp: u64,
        block_height: u32,
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
        crate::types::hashes::hash(&vote_data)
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
    pub decision_id: DecisionId,
    pub period_id: VotingPeriodId,
    pub status: DecisionResolutionStatus,
    pub status_changed_at: u64,
    pub status_changed_height: u32,
    pub vote_count: u32,
    pub min_votes_required: u32,
    pub voting_deadline: u64,
    pub outcome_ready: bool,
    pub reason: Option<String>,
}

impl DecisionResolution {
    pub fn new(
        decision_id: DecisionId,
        period_id: VotingPeriodId,
        voting_deadline: u64,
        min_votes_required: u32,
        current_timestamp: u64,
        current_height: u32,
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
        block_height: u32,
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
    pub decision_id: DecisionId,
    pub period_id: VotingPeriodId,
    pub outcome_value: f64,
    pub min: f64,
    pub max: f64,
    pub confidence: f64,
    pub total_votes: u64,
    pub total_reputation_weight: f64,
    pub finalized_at: u64,
    pub block_height: u32,
    pub is_consensus: bool,
    pub resolution: DecisionResolution,
    pub categorical_winner: Option<Option<u16>>,
}

impl DecisionOutcome {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        decision_id: DecisionId,
        period_id: VotingPeriodId,
        outcome_value: f64,
        min: f64,
        max: f64,
        confidence: f64,
        total_votes: u64,
        total_reputation_weight: f64,
        finalized_at: u64,
        block_height: u32,
        is_consensus: bool,
        resolution: DecisionResolution,
        categorical_winner: Option<Option<u16>>,
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
            categorical_winner,
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
    pub decision_id: DecisionId,
}

impl VoteMatrixKey {
    pub fn new(
        period_id: VotingPeriodId,
        voter_address: Address,
        decision_id: DecisionId,
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
    pub block_height: u32,
}

impl VoteMatrixEntry {
    pub fn new(value: VoteValue, timestamp: u64, block_height: u32) -> Self {
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
pub struct Ballot {
    pub period_id: VotingPeriodId,
    pub votes: Vec<Vote>,
    pub created_at: u64,
    pub block_height: u32,
}

impl Ballot {
    pub fn new(
        period_id: VotingPeriodId,
        created_at: u64,
        block_height: u32,
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
    pub reputation_changes: Option<BTreeMap<Address, (f64, f64)>>,
}

impl VotingPeriodStats {
    /// Returns the absolute decision ids referenced in the reputation_changes field.
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
pub struct DecisionResolutionEntry {
    pub decision_id: DecisionId,
    pub outcome_value: f64,
}

/// Holds all pending changes from consensus calculation for atomic commit.
#[derive(Clone, Debug)]
pub struct ConsensusResult {
    pub period_id: VotingPeriodId,
    pub period_stats: VotingPeriodStats,
    pub decision_outcomes: Vec<DecisionOutcome>,
    pub decision_resolutions: Vec<DecisionResolutionEntry>,
    pub updated_reputations: BTreeMap<Address, f64>,
    pub block_height: u32,
    pub timestamp: u64,
    pub total_decision_count: usize,
}

impl ConsensusResult {
    pub fn new(
        period_id: VotingPeriodId,
        block_height: u32,
        timestamp: u64,
        total_decision_count: usize,
    ) -> Self {
        Self {
            period_id,
            period_stats: VotingPeriodStats::new(period_id, timestamp),
            decision_outcomes: Vec::new(),
            decision_resolutions: Vec::new(),
            updated_reputations: BTreeMap::new(),
            block_height,
            timestamp,
            total_decision_count,
        }
    }

    pub fn has_outcomes(&self) -> bool {
        !self.decision_outcomes.is_empty()
    }

    pub fn resolved_decision_count(&self) -> usize {
        self.decision_resolutions.len()
    }

    pub fn abstained_decision_count(&self) -> usize {
        self.total_decision_count
            .saturating_sub(self.decision_resolutions.len())
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
    fn test_ballot() {
        let period_id = VotingPeriodId::new(1);
        let mut batch = Ballot::new(period_id, 1000, 100);

        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);

        let vote = Vote::new(
            Address([1u8; 20]),
            period_id,
            DecisionId::new(true, 1, 0).unwrap(),
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
