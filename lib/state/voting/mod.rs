//! Bitcoin Hivemind Voting Module
//!
//! Implements the Bitcoin Hivemind SVD-based consensus mechanism.
//! Voting weight is derived directly from the reputation vector,
//! updated each period by the Sztorc consensus algorithm.

pub mod database;
pub mod period_calculator;
pub mod types;

use crate::state::{Error, decisions::DecisionId};
use database::VotingDatabases;
use sneed::{Env, RoTxn, RwTxn};
use std::collections::{BTreeMap, HashMap, HashSet};
use types::{
    ConsensusResult, DecisionOutcome, DecisionResolution,
    DecisionResolutionEntry, Vote, VoteValue, VotingPeriod, VotingPeriodId,
    VotingPeriodStats, VotingPeriodStatus,
};

fn compute_post_consensus_reputation(
    pre: &BTreeMap<crate::types::Address, f64>,
    updates: &BTreeMap<crate::types::Address, f64>,
) -> BTreeMap<crate::types::Address, f64> {
    let mut post = pre.clone();
    for (&addr, &rep) in updates {
        if rep > 0.0 {
            post.insert(addr, rep);
        } else {
            post.remove(&addr);
        }
    }
    post
}

fn reanchor_to_initial_total(post: &mut BTreeMap<crate::types::Address, f64>) {
    use crate::math::voting::constants::INITIAL_REPUTATION_TOTAL;

    let sum: f64 = post.values().sum();
    let residual = INITIAL_REPUTATION_TOTAL - sum;
    if residual == 0.0 {
        return;
    }
    let Some((&addr, _)) = post.iter().max_by(|a, b| {
        a.1.partial_cmp(b.1)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    }) else {
        return;
    };
    if let Some(rep) = post.get_mut(&addr) {
        *rep += residual;
    }
}

fn verify_reputation_conservation(
    pre: &BTreeMap<crate::types::Address, f64>,
    post: &BTreeMap<crate::types::Address, f64>,
) -> Result<(f64, f64, f64, f64), Error> {
    use crate::math::voting::constants::{
        CONSENSUS_CONSERVATION_EPSILON, INITIAL_REPUTATION_TOTAL,
        REPUTATION_ANCHOR_EPSILON,
    };

    let pre_sum: f64 = pre.values().sum();
    let post_sum: f64 = post.values().sum();
    let block_drift = (post_sum - pre_sum).abs();
    if block_drift > CONSENSUS_CONSERVATION_EPSILON {
        return Err(Error::ConsensusConservationViolation {
            pre_sum,
            post_sum,
            drift: block_drift,
        });
    }

    let anchor_drift = (post_sum - INITIAL_REPUTATION_TOTAL).abs();
    if anchor_drift > REPUTATION_ANCHOR_EPSILON {
        return Err(Error::ReputationAnchorViolation {
            total: post_sum,
            anchor: INITIAL_REPUTATION_TOTAL,
            drift: anchor_drift,
        });
    }

    Ok((pre_sum, post_sum, block_drift, anchor_drift))
}

#[derive(Clone)]
pub struct VotingSystem {
    databases: VotingDatabases,
}

impl VotingSystem {
    pub const NUM_DBS: u32 = VotingDatabases::NUM_DBS;

    pub fn new(env: &Env, rwtxn: &mut RwTxn<'_>) -> Result<Self, Error> {
        let databases = VotingDatabases::new(env, rwtxn)?;
        Ok(Self { databases })
    }

    pub fn databases(&self) -> &VotingDatabases {
        &self.databases
    }

    /// Pure computation phase no db writes
    fn compute_consensus(
        &self,
        rotxn: &RoTxn,
        period: &VotingPeriod,
        state: &crate::state::State,
        current_timestamp: u64,
        current_height: u32,
    ) -> Result<Option<ConsensusResult>, Error> {
        let period_id = period.id;
        use crate::math::voting::{
            SparseVoteMatrix, VotingWeightVector,
            calculate_consensus as math_calculate_consensus,
        };

        let existing_outcomes = self
            .databases
            .get_consensus_outcomes_for_period(rotxn, period_id)?;
        if !existing_outcomes.is_empty() {
            tracing::warn!(
                "compute_consensus: Consensus already calculated for period {} ({} outcomes exist), skipping",
                period_id.0,
                existing_outcomes.len()
            );
            return Ok(None);
        }

        let all_decision_ids = &period.decision_ids;
        let total_decision_count = all_decision_ids.len();

        let categorical_default = |_decision_id: DecisionId| -> f64 { 0.5 };

        let all_votes =
            self.databases.get_votes_for_period(rotxn, period_id)?;

        if all_votes.is_empty() {
            tracing::info!(
                "compute_consensus: No votes for period {}, assigning defaults to all {} decisions",
                period_id.0,
                total_decision_count
            );

            let mut result = ConsensusResult::new(
                period_id,
                current_height,
                current_timestamp,
                total_decision_count,
            );

            for &decision_id in all_decision_ids {
                let default_value = categorical_default(decision_id);

                tracing::info!(
                    "Decision {} no votes — using default {:.4}",
                    hex::encode(decision_id.as_bytes()),
                    default_value
                );

                let mut resolution =
                    DecisionResolution::new(decision_id, period_id, 0, 0, 0, 0);
                resolution.mark_outcome_ready();

                let outcome = DecisionOutcome::new(
                    decision_id,
                    period_id,
                    default_value,
                    0.0,
                    1.0,
                    0.0,
                    0,
                    0.0,
                    0,
                    0,
                    true,
                    resolution,
                    None,
                );

                result.decision_outcomes.push(outcome);
                result.decision_resolutions.push(DecisionResolutionEntry {
                    decision_id,
                    outcome_value: default_value,
                });
            }

            let mut period_stats = VotingPeriodStats::new(period_id, 0);
            period_stats.certainty = Some(0.0);
            result.period_stats = period_stats;

            return Ok(Some(result));
        }

        let reputations = state.reputation().get_all_reputations(rotxn)?;
        let total_weight: f64 = reputations.values().sum();

        if reputations.is_empty() || total_weight <= 0.0 {
            return Ok(None);
        }

        let mut voters: Vec<_> = reputations.keys().copied().collect();
        voters.sort();
        let mut decisions: Vec<_> = all_decision_ids.to_vec();
        decisions.sort();

        let mut scaled_decisions: HashSet<DecisionId> = HashSet::new();
        let mut categorical_decisions: HashMap<DecisionId, u16> =
            HashMap::new();
        for decision_id in &decisions {
            if let Some(entry) =
                state.decisions().get_decision_entry(rotxn, *decision_id)?
                && let Some(decision) = &entry.decision
            {
                if decision.is_scaled() {
                    scaled_decisions.insert(*decision_id);
                } else if let Some(n) = decision.option_count() {
                    categorical_decisions.insert(*decision_id, n as u16);
                }
            }
        }

        let mut vote_matrix = SparseVoteMatrix::new(voters, decisions);

        for (vote_key, vote_entry) in &all_votes {
            if let Some(vote_value) = vote_entry.to_float_opt() {
                vote_matrix
                    .set_vote(
                        vote_key.voter_address,
                        vote_key.decision_id,
                        vote_value,
                    )
                    .map_err(|e| Error::InvalidTransaction {
                        reason: format!("Failed to set vote in matrix: {e:?}"),
                    })?;
            }
        }

        let weight_vector =
            VotingWeightVector::from_reputation_map(&reputations);
        let math_result = math_calculate_consensus(
            &vote_matrix,
            &weight_vector,
            &scaled_decisions,
            &categorical_decisions,
        )
        .map_err(|e| Error::InvalidTransaction {
            reason: format!("Failed to calculate consensus: {e:?}"),
        })?;

        let mut period_stats = self
            .databases
            .get_period_stats(rotxn, period_id)?
            .unwrap_or_else(|| VotingPeriodStats::new(period_id, 0));

        period_stats.first_loading = Some(math_result.first_loading.clone());
        period_stats.certainty = Some(math_result.certainty);

        let mut score_changes: BTreeMap<crate::types::Address, (f64, f64)> =
            BTreeMap::new();
        for (&voter_id, &new_score) in &math_result.updated_reputations {
            let old_rep = reputations.get(&voter_id).copied().unwrap_or(0.0);
            score_changes.insert(voter_id, (old_rep, new_score));
        }
        if !score_changes.is_empty() {
            period_stats.reputation_changes = Some(score_changes);
        }

        let mut decision_outcomes = Vec::new();
        let mut decision_resolutions = Vec::new();
        let mut resolved_by_svd: HashSet<DecisionId> = HashSet::new();

        for (decision_id, outcome_value) in &math_result.outcomes {
            let outcome_f64 = match outcome_value {
                Some(v) => *v,
                None => {
                    let default_value = categorical_default(*decision_id);
                    tracing::info!(
                        "Decision {} has unanimous abstention \
                         — using default {:.4}",
                        hex::encode(decision_id.as_bytes()),
                        default_value
                    );
                    default_value
                }
            };

            resolved_by_svd.insert(*decision_id);

            let mut resolution =
                DecisionResolution::new(*decision_id, period_id, 0, 1, 0, 0);
            resolution.mark_outcome_ready();

            let cat_winner =
                math_result.categorical_winners.get(decision_id).copied();

            let outcome = DecisionOutcome::new(
                *decision_id,
                period_id,
                outcome_f64,
                0.0,
                1.0,
                1.0,
                all_votes.len() as u64,
                total_weight,
                0,
                0,
                true,
                resolution,
                cat_winner,
            );

            decision_outcomes.push(outcome);
            decision_resolutions.push(DecisionResolutionEntry {
                decision_id: *decision_id,
                outcome_value: outcome_f64,
            });
        }

        for &decision_id in all_decision_ids {
            if !resolved_by_svd.contains(&decision_id) {
                let default_value = categorical_default(decision_id);

                tracing::info!(
                    "Decision {} had no votes — using default {:.4}",
                    hex::encode(decision_id.as_bytes()),
                    default_value
                );

                let mut resolution =
                    DecisionResolution::new(decision_id, period_id, 0, 0, 0, 0);
                resolution.mark_outcome_ready();

                let outcome = DecisionOutcome::new(
                    decision_id,
                    period_id,
                    default_value,
                    0.0,
                    1.0,
                    0.0,
                    0,
                    0.0,
                    0,
                    0,
                    true,
                    resolution,
                    None,
                );

                decision_outcomes.push(outcome);
                decision_resolutions.push(DecisionResolutionEntry {
                    decision_id,
                    outcome_value: default_value,
                });
            }
        }

        let mut result = ConsensusResult::new(
            period_id,
            current_height,
            current_timestamp,
            total_decision_count,
        );
        result.period_stats = period_stats;
        result.decision_outcomes = decision_outcomes;
        result.decision_resolutions = decision_resolutions;
        result.updated_reputations = math_result.updated_reputations.clone();

        Ok(Some(result))
    }

    fn validate_consensus_commit(
        &self,
        rwtxn: &RwTxn,
        result: &ConsensusResult,
        decisions_db: &crate::state::decisions::Dbs,
    ) -> Result<(), Error> {
        for decision_resolution in &result.decision_resolutions {
            decisions_db
                .get_decision_state_history(
                    rwtxn,
                    decision_resolution.decision_id,
                )?
                .ok_or(Error::InvalidDecisionId {
                    reason: format!(
                        "Pre-validation failed: DecisionEntry {:?} \
                         has no state history",
                        decision_resolution.decision_id
                    ),
                })?;
        }
        Ok(())
    }

    fn commit_consensus_result(
        &self,
        rwtxn: &mut RwTxn,
        result: ConsensusResult,
        state: &crate::state::State,
        decisions_db: &crate::state::decisions::Dbs,
    ) -> Result<crate::state::undo::ConsensusUndoEntry, Error> {
        tracing::debug!(
            period_id = result.period_id.0,
            block_height = result.block_height,
            outcome_count = result.decision_outcomes.len(),
            "Starting atomic consensus commit"
        );

        self.validate_consensus_commit(rwtxn, &result, decisions_db)?;

        let had_period_stats = self
            .databases
            .get_period_stats(rwtxn, result.period_id)?
            .is_some();

        let decision_outcome_ids: Vec<crate::state::decisions::DecisionId> =
            result
                .decision_outcomes
                .iter()
                .map(|o| o.decision_id)
                .collect();

        let resolved_decision_ids: Vec<crate::state::decisions::DecisionId> =
            result
                .decision_resolutions
                .iter()
                .map(|sr| sr.decision_id)
                .collect();

        let pre_consensus_reputation =
            state.reputation().get_all_reputations(rwtxn)?;

        let mut post_consensus_reputation = compute_post_consensus_reputation(
            &pre_consensus_reputation,
            &result.updated_reputations,
        );

        let (pre_sum, post_sum, block_drift, anchor_drift) =
            verify_reputation_conservation(
                &pre_consensus_reputation,
                &post_consensus_reputation,
            )?;

        reanchor_to_initial_total(&mut post_consensus_reputation);

        self.databases
            .put_period_stats(rwtxn, &result.period_stats)?;

        for outcome in &result.decision_outcomes {
            self.databases.put_decision_outcome(rwtxn, outcome)?;
        }

        state
            .reputation()
            .clear_and_restore(rwtxn, &post_consensus_reputation)?;

        tracing::debug!(
            voter_count = result.updated_reputations.len(),
            pre_sum,
            post_sum,
            block_drift,
            anchor_drift,
            "Applied reputation updates (conservation verified)"
        );

        for decision_resolution in &result.decision_resolutions {
            decisions_db.transition_decision_to_resolved(
                rwtxn,
                decision_resolution.decision_id,
                result.block_height,
                decision_resolution.outcome_value,
            )?;
        }

        let resolved_count = result.resolved_decision_count();
        let abstained_count = result.abstained_decision_count();

        tracing::info!(
            period_id = result.period_id.0,
            block_height = result.block_height,
            resolved = resolved_count,
            abstained = abstained_count,
            "Consensus commit completed successfully"
        );

        Ok(crate::state::undo::ConsensusUndoEntry {
            period_id: result.period_id,
            decision_outcome_ids,
            had_period_stats,
            resolved_decision_ids,
            pre_consensus_reputation,
        })
    }

    pub(crate) fn calculate_and_store_consensus(
        &self,
        rwtxn: &mut RwTxn,
        period: &VotingPeriod,
        state: &crate::state::State,
        current_timestamp: u64,
        current_height: u32,
        decisions_db: &crate::state::decisions::Dbs,
    ) -> Result<Option<crate::state::undo::ConsensusUndoEntry>, Error> {
        tracing::debug!(
            period_id = period.id.0,
            "calculate_and_store_consensus: Starting"
        );

        let result = self.compute_consensus(
            rwtxn,
            period,
            state,
            current_timestamp,
            current_height,
        )?;

        if let Some(consensus_result) = result {
            let undo_entry = self.commit_consensus_result(
                rwtxn,
                consensus_result,
                state,
                decisions_db,
            )?;
            Ok(Some(undo_entry))
        } else {
            Ok(None)
        }
    }

    pub fn get_all_periods(
        &self,
        rotxn: &RoTxn,
        current_timestamp: u64,
        current_height: u32,
        config: &crate::state::decisions::DecisionConfig,
        decisions_db: &crate::state::decisions::Dbs,
        genesis_ts: u64,
    ) -> Result<HashMap<VotingPeriodId, VotingPeriod>, Error> {
        period_calculator::get_all_active_periods(
            rotxn,
            decisions_db,
            config,
            current_timestamp,
            current_height,
            &self.databases,
            genesis_ts,
        )
    }

    pub fn get_active_period(
        &self,
        rotxn: &RoTxn,
        current_timestamp: u64,
        current_height: u32,
        config: &crate::state::decisions::DecisionConfig,
        decisions_db: &crate::state::decisions::Dbs,
        genesis_ts: u64,
    ) -> Result<Option<VotingPeriod>, Error> {
        let all_periods = self.get_all_periods(
            rotxn,
            current_timestamp,
            current_height,
            config,
            decisions_db,
            genesis_ts,
        )?;

        for period in all_periods.values() {
            if period.status == VotingPeriodStatus::Active {
                return Ok(Some(period.clone()));
            }
        }

        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cast_vote(
        &self,
        rwtxn: &mut RwTxn,
        voter_address: crate::types::Address,
        period_id: VotingPeriodId,
        decision_id: DecisionId,
        value: VoteValue,
        timestamp: u64,
        block_height: u32,
        tx_hash: [u8; 32],
        config: &crate::state::decisions::DecisionConfig,
        decisions_db: &crate::state::decisions::Dbs,
        genesis_ts: u64,
    ) -> Result<(), Error> {
        let has_outcomes = self.databases.has_consensus(rwtxn, period_id)?;
        let period = period_calculator::calculate_voting_period(
            rwtxn,
            period_id,
            block_height,
            timestamp,
            config,
            decisions_db,
            has_outcomes,
            genesis_ts,
        )?;

        if !period_calculator::can_accept_votes(&period) {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Period {:?} cannot accept votes (status: {:?}, height: {}, timestamp: {})",
                    period_id, period.status, block_height, timestamp
                ),
            });
        }

        crate::validation::PeriodValidator::validate_decision_in_period(
            &period,
            decision_id,
        )?;

        let vote = Vote::new(
            voter_address,
            period_id,
            decision_id,
            value,
            timestamp,
            block_height,
            tx_hash,
        );

        self.databases.put_vote(rwtxn, &vote)?;

        Ok(())
    }

    pub fn get_votes_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<(crate::types::Address, DecisionId), VoteValue>, Error>
    {
        let vote_entries =
            self.databases.get_votes_for_period(rotxn, period_id)?;
        let mut votes = HashMap::new();

        for (key, entry) in vote_entries {
            votes.insert((key.voter_address, key.decision_id), entry.value);
        }

        Ok(votes)
    }

    pub fn get_vote_matrix(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<(crate::types::Address, DecisionId), f64>, Error> {
        let vote_entries =
            self.databases.get_votes_for_period(rotxn, period_id)?;
        let mut matrix = HashMap::new();

        for (key, entry) in vote_entries {
            if let Some(vote_value) = entry.to_float_opt() {
                matrix.insert((key.voter_address, key.decision_id), vote_value);
            }
        }

        Ok(matrix)
    }

    pub fn get_participation_stats(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
        _config: &crate::state::decisions::DecisionConfig,
        decisions_db: &crate::state::decisions::Dbs,
    ) -> Result<(u64, u64, f64), Error> {
        let votes = self.databases.get_votes_for_period(rotxn, period_id)?;

        let decision_ids = period_calculator::get_decision_ids_for_period(
            rotxn,
            period_id,
            decisions_db,
        )?;

        let total_votes = votes.len() as u64;
        let unique_voters: HashSet<crate::types::Address> =
            votes.keys().map(|k| k.voter_address).collect();
        let total_voters = unique_voters.len() as u64;
        let total_decisions = decision_ids.len() as u64;

        let participation_rate = if total_voters > 0 && total_decisions > 0 {
            total_votes as f64 / (total_voters * total_decisions) as f64
        } else {
            0.0
        };

        Ok((total_voters, total_votes, participation_rate))
    }

    pub fn resolve_period_decisions(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<Vec<DecisionOutcome>, Error> {
        let stored_outcomes =
            self.databases.get_outcomes_for_period(rotxn, period_id)?;

        if stored_outcomes.is_empty() {
            return Err(Error::ConsensusNotYetCalculated(period_id));
        }

        let mut outcomes: Vec<DecisionOutcome> =
            stored_outcomes.into_values().collect();
        outcomes.sort_by_key(|o| o.decision_id);
        Ok(outcomes)
    }

    pub fn get_period_outcomes(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<DecisionId, DecisionOutcome>, Error> {
        self.databases.get_outcomes_for_period(rotxn, period_id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn calculate_period_statistics(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
        current_height: u32,
        current_timestamp: u64,
        config: &crate::state::decisions::DecisionConfig,
        decisions_db: &crate::state::decisions::Dbs,
        state: &crate::state::State,
        genesis_ts: u64,
    ) -> Result<VotingPeriodStats, Error> {
        let (total_voters, total_votes, participation_rate) = self
            .get_participation_stats(rotxn, period_id, config, decisions_db)?;

        let has_outcomes = self.databases.has_consensus(rotxn, period_id)?;
        let period = period_calculator::calculate_voting_period(
            rotxn,
            period_id,
            current_height,
            current_timestamp,
            config,
            decisions_db,
            has_outcomes,
            genesis_ts,
        )?;

        let votes = self.databases.get_votes_for_period(rotxn, period_id)?;
        let voter_addrs: HashSet<crate::types::Address> =
            votes.keys().map(|k| k.voter_address).collect();
        let all_reputations = state.reputation().get_all_reputations(rotxn)?;
        let total_reputation_weight: f64 = voter_addrs
            .iter()
            .filter_map(|addr| all_reputations.get(addr))
            .sum();

        let outcomes =
            self.databases.get_outcomes_for_period(rotxn, period_id)?;
        let consensus_decisions = outcomes
            .values()
            .filter(|outcome| outcome.is_consensus)
            .count() as u64;

        let mut stats = VotingPeriodStats::new(period_id, current_timestamp);
        stats.total_voters = total_voters;
        stats.total_votes = total_votes;
        stats.total_decisions = period.decision_ids.len() as u64;
        stats.avg_participation_rate = participation_rate;
        stats.total_reputation_weight = total_reputation_weight;
        stats.consensus_decisions = consensus_decisions;

        Ok(stats)
    }

    pub fn validate_consistency(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<String>, Error> {
        self.databases.check_consistency(rotxn)
    }

    pub fn get_system_stats(
        &self,
        rotxn: &RoTxn,
        decisions_db: &crate::state::decisions::Dbs,
        state: &crate::state::State,
    ) -> Result<(u64, u64, u64, f64), Error> {
        let total_votes = self.databases.count_total_votes(rotxn)?;
        let all_voters = self.databases.get_all_voters(rotxn)?;
        let total_voters = all_voters.len() as u64;

        let rep_holders = state.reputation().count_holders(rotxn)?;
        let avg_weight = if rep_holders > 0 {
            1.0 / rep_holders as f64
        } else {
            0.0
        };

        let all_decisions = decisions_db.get_all_claimed_decisions(rotxn)?;
        let mut unique_periods = std::collections::HashSet::new();
        for entry in all_decisions {
            let voting_period = entry.decision_id.voting_period();
            unique_periods.insert(voting_period);
        }
        let total_periods = unique_periods.len() as u64;

        Ok((total_periods, total_votes, total_voters, avg_weight))
    }
}

#[cfg(test)]
mod conservation_tests {
    use super::*;
    use crate::types::Address;

    fn addr(b: u8) -> Address {
        Address([b; 20])
    }

    fn pre_state_unit() -> BTreeMap<Address, f64> {
        let mut m = BTreeMap::new();
        m.insert(addr(1), 0.6);
        m.insert(addr(2), 0.4);
        m
    }

    #[test]
    fn compute_post_replaces_and_drops() {
        let pre = pre_state_unit();
        let mut updates = BTreeMap::new();
        updates.insert(addr(1), 0.5);
        updates.insert(addr(2), 0.0);
        updates.insert(addr(3), 0.5);
        let post = compute_post_consensus_reputation(&pre, &updates);
        assert_eq!(post.get(&addr(1)), Some(&0.5));
        assert_eq!(post.get(&addr(2)), None);
        assert_eq!(post.get(&addr(3)), Some(&0.5));
    }

    #[test]
    fn per_block_rejects_mint() {
        let pre = pre_state_unit();
        let mut post = pre.clone();
        post.insert(addr(1), 0.6 + 1e-8);
        let err = verify_reputation_conservation(&pre, &post).unwrap_err();
        assert!(
            matches!(err, Error::ConsensusConservationViolation { .. }),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn per_block_rejects_burn() {
        let pre = pre_state_unit();
        let mut post = pre.clone();
        post.insert(addr(1), 0.6 - 1e-8);
        let err = verify_reputation_conservation(&pre, &post).unwrap_err();
        assert!(
            matches!(err, Error::ConsensusConservationViolation { .. }),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn per_block_accepts_within_epsilon() {
        let pre = pre_state_unit();
        let mut post = pre.clone();
        post.insert(addr(1), 0.6 + 1e-11);
        post.insert(addr(2), 0.4 - 1e-11);
        assert!(verify_reputation_conservation(&pre, &post).is_ok());
    }

    #[test]
    fn anchor_rejects_cumulative_drift() {
        let mut pre = BTreeMap::new();
        pre.insert(addr(1), 0.6 + 1.5e-6);
        pre.insert(addr(2), 0.4);
        let post = pre.clone();
        let err = verify_reputation_conservation(&pre, &post).unwrap_err();
        assert!(
            matches!(err, Error::ReputationAnchorViolation { .. }),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn anchor_accepts_within_epsilon() {
        let mut pre = BTreeMap::new();
        pre.insert(addr(1), 0.6 + 5e-8);
        pre.insert(addr(2), 0.4);
        let post = pre.clone();
        assert!(verify_reputation_conservation(&pre, &post).is_ok());
    }

    #[test]
    fn compute_post_preserves_untouched_voters() {
        let pre = pre_state_unit();
        let mut updates = BTreeMap::new();
        updates.insert(addr(1), 0.7);
        let post = compute_post_consensus_reputation(&pre, &updates);
        assert_eq!(post.get(&addr(2)), Some(&0.4));
    }

    #[test]
    fn reanchor_restores_unit_sum_after_drift() {
        let mut post = BTreeMap::new();
        post.insert(addr(1), 0.5 + 1e-10);
        post.insert(addr(2), 0.5 - 2e-10);
        reanchor_to_initial_total(&mut post);
        let sum: f64 = post.values().sum();
        assert_eq!(sum.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn reanchor_handles_empty_map() {
        let mut post: BTreeMap<Address, f64> = BTreeMap::new();
        reanchor_to_initial_total(&mut post);
        assert!(post.is_empty());
    }

    #[test]
    fn reanchor_targets_largest_voter_with_deterministic_tiebreak() {
        let mut post = BTreeMap::new();
        post.insert(addr(2), 0.5);
        post.insert(addr(1), 0.5 - 3e-10);
        reanchor_to_initial_total(&mut post);
        let sum: f64 = post.values().sum();
        assert_eq!(sum.to_bits(), 1.0_f64.to_bits());
        assert!(*post.get(&addr(2)).unwrap() > 0.5);
    }

    #[test]
    fn reanchor_repeated_drift_never_accumulates() {
        let mut post = BTreeMap::new();
        post.insert(addr(1), 0.6);
        post.insert(addr(2), 0.4);
        for i in 0..10_000 {
            let delta = 1e-12 * (i as f64 % 7.0 - 3.0);
            *post.get_mut(&addr(1)).unwrap() += delta;
            *post.get_mut(&addr(2)).unwrap() -= delta;
            reanchor_to_initial_total(&mut post);
        }
        let sum: f64 = post.values().sum();
        assert!(
            (sum - 1.0).abs() < 1e-15,
            "drift accumulated: sum = {sum:.20}"
        );
    }
}
