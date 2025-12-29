pub mod consensus;
pub mod constants;

use crate::state::slots::SlotId;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, thiserror::Error)]
pub enum VotingMathError {
    #[error("Matrix dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    #[error("Empty vote matrix provided")]
    EmptyMatrix,

    #[error("Invalid reputation values: {reason}")]
    InvalidReputation { reason: String },

    #[error(
        "Invalid reputation value: {value} (must be finite and in [0.0, 1.0])"
    )]
    InvalidReputationValue { value: f64 },

    #[error("Numerical computation error: {reason}")]
    NumericalError { reason: String },

    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },
}

#[derive(Debug, Clone)]
pub struct SparseVoteMatrix {
    entries: HashMap<(usize, usize), f64>,
    voter_indices: HashMap<crate::types::Address, usize>,
    decision_indices: HashMap<SlotId, usize>,
    voters: Vec<crate::types::Address>,
    decisions: Vec<SlotId>,
    num_voters: usize,
    num_decisions: usize,
    /// set of decision indices with votes
    row_entries: Vec<HashSet<usize>>,
    /// set of voter indices with votes
    col_entries: Vec<HashSet<usize>>,
}

impl SparseVoteMatrix {
    pub fn new(
        voters: Vec<crate::types::Address>,
        decisions: Vec<SlotId>,
    ) -> Self {
        let num_voters = voters.len();
        let num_decisions = decisions.len();

        let voter_indices: HashMap<_, _> = voters
            .iter()
            .enumerate()
            .map(|(i, &addr)| (addr, i))
            .collect();

        let decision_indices: HashMap<_, _> = decisions
            .iter()
            .enumerate()
            .map(|(j, &slot)| (slot, j))
            .collect();

        Self {
            entries: HashMap::new(),
            voter_indices,
            decision_indices,
            voters,
            decisions,
            num_voters,
            num_decisions,
            row_entries: vec![HashSet::new(); num_voters],
            col_entries: vec![HashSet::new(); num_decisions],
        }
    }

    pub fn set_vote(
        &mut self,
        voter_address: crate::types::Address,
        decision_id: SlotId,
        value: f64,
    ) -> Result<(), VotingMathError> {
        let voter_idx =
            *self.voter_indices.get(&voter_address).ok_or_else(|| {
                VotingMathError::InvalidReputation {
                    reason: format!(
                        "Voter {voter_address:?} not found in matrix"
                    ),
                }
            })?;

        let decision_idx = *self
            .decision_indices
            .get(&decision_id)
            .ok_or_else(|| VotingMathError::InvalidReputation {
                reason: format!("Decision {decision_id:?} not found in matrix"),
            })?;

        self.entries.insert((voter_idx, decision_idx), value);
        self.row_entries[voter_idx].insert(decision_idx);
        self.col_entries[decision_idx].insert(voter_idx);

        Ok(())
    }

    pub fn get_vote(
        &self,
        voter_address: crate::types::Address,
        decision_id: SlotId,
    ) -> Option<f64> {
        let voter_idx = *self.voter_indices.get(&voter_address)?;
        let decision_idx = *self.decision_indices.get(&decision_id)?;
        self.entries.get(&(voter_idx, decision_idx)).copied()
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.num_voters, self.num_decisions)
    }

    pub fn num_votes(&self) -> usize {
        self.entries.len()
    }

    /// Get all votes by a specific voter. O(k) where k = votes by a voter.
    pub fn get_voter_votes(
        &self,
        voter_address: crate::types::Address,
    ) -> HashMap<SlotId, f64> {
        let mut votes = HashMap::new();

        if let Some(&voter_idx) = self.voter_indices.get(&voter_address) {
            for &decision_idx in &self.row_entries[voter_idx] {
                if let Some(&value) =
                    self.entries.get(&(voter_idx, decision_idx))
                {
                    let decision_id = self.decisions[decision_idx];
                    votes.insert(decision_id, value);
                }
            }
        }

        votes
    }

    pub fn get_decision_votes(
        &self,
        decision_id: SlotId,
    ) -> HashMap<crate::types::Address, f64> {
        let mut votes = HashMap::new();

        if let Some(&decision_idx) = self.decision_indices.get(&decision_id) {
            // Use column index for O(k) lookup instead of O(M) full scan
            for &voter_idx in &self.col_entries[decision_idx] {
                if let Some(&value) =
                    self.entries.get(&(voter_idx, decision_idx))
                {
                    // Vec index lookup is O(1) and bounds-checked
                    let voter_id = self.voters[voter_idx];
                    votes.insert(voter_id, value);
                }
            }
        }

        votes
    }

    /// Returns a slice of all voters. Avoids cloning.
    pub fn get_voters(&self) -> &[crate::types::Address] {
        &self.voters
    }

    /// Returns a slice of all decisions. Avoids cloning.
    pub fn get_decisions(&self) -> &[SlotId] {
        &self.decisions
    }
}

#[derive(Debug, Clone)]
pub struct VotingWeightVector {
    reputations: HashMap<crate::types::Address, f64>,
    total_weight: Option<f64>,
}

impl VotingWeightVector {
    pub fn new() -> Self {
        Self {
            reputations: HashMap::new(),
            total_weight: None,
        }
    }

    pub fn set_reputation(
        &mut self,
        voter_address: crate::types::Address,
        voting_weight: f64,
    ) {
        self.reputations
            .insert(voter_address, voting_weight.clamp(0.0, 1.0));
        self.total_weight = None;
    }

    pub fn get_reputation(&self, voter_address: crate::types::Address) -> f64 {
        self.reputations.get(&voter_address).copied().unwrap_or(0.0)
    }

    pub fn total_weight(&mut self) -> f64 {
        if let Some(weight) = self.total_weight {
            return weight;
        }

        let weight: f64 = self.reputations.values().sum();
        self.total_weight = Some(weight);
        weight
    }

    pub fn len(&self) -> usize {
        self.reputations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.reputations.is_empty()
    }

    pub fn from_voter_reputations(
        voter_reputations: &HashMap<
            crate::types::Address,
            crate::state::voting::types::VoterReputation,
        >,
    ) -> Self {
        let mut reputation_vector = Self::new();

        for (voter_id, voter_reputation) in voter_reputations {
            reputation_vector.set_reputation(
                *voter_id,
                voter_reputation.get_voting_weight(),
            );
        }

        reputation_vector
    }
}

impl Default for VotingWeightVector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct DetailedConsensusResult {
    pub outcomes: HashMap<SlotId, Option<f64>>,
    pub first_loading: Vec<f64>,
    pub certainty: f64,
    pub updated_reputations: HashMap<crate::types::Address, f64>,
    pub outliers: Vec<crate::types::Address>,
}

/// Calculate consensus using SVD-based PCA and current reputation weights.
/// Outcomes are then used to update reputation in the next period.
pub fn calculate_consensus(
    vote_matrix: &SparseVoteMatrix,
    reputation_vector: &VotingWeightVector,
) -> Result<DetailedConsensusResult, VotingMathError> {
    consensus::run_consensus(vote_matrix, reputation_vector)
}

#[cfg(test)]
mod tests;
