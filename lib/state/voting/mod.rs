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

pub(in crate::state) fn reanchor_to_initial_total(
    post: &mut BTreeMap<crate::types::Address, f64>,
) {
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

    pub fn new<Tls>(
        env: &Env<Tls>,
        rwtxn: &mut RwTxn<'_>,
    ) -> Result<Self, Error> {
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
                    const_hex::encode(decision_id.as_bytes()),
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
        let mut decisions: Vec<_> = all_decision_ids
            .iter()
            .copied()
            .filter(|decision_id| decision_id.is_standard())
            .collect();
        decisions.sort();
        let has_standard_decisions = !decisions.is_empty();

        let mut scaled_decisions: HashSet<DecisionId> = HashSet::new();
        let mut categorical_decisions: HashMap<DecisionId, u16> =
            HashMap::new();
        for decision_id in all_decision_ids {
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
        let mut unguaranteed_votes: BTreeMap<
            DecisionId,
            BTreeMap<crate::types::Address, (f64, f64)>,
        > = BTreeMap::new();

        for (vote_key, vote_entry) in &all_votes {
            let Some(&weight) = reputations.get(&vote_key.voter_address) else {
                continue;
            };
            let Some(vote_value) = vote_entry.to_float_opt() else {
                continue;
            };
            if vote_key.decision_id.is_standard() {
                vote_matrix
                    .set_vote(
                        vote_key.voter_address,
                        vote_key.decision_id,
                        vote_value,
                    )
                    .map_err(|e| Error::InvalidTransaction {
                        reason: format!("Failed to set vote in matrix: {e:?}"),
                    })?;
            } else {
                unguaranteed_votes
                    .entry(vote_key.decision_id)
                    .or_default()
                    .insert(vote_key.voter_address, (vote_value, weight));
            }
        }

        let math_result = if has_standard_decisions {
            let weight_vector =
                VotingWeightVector::from_reputation_map(&reputations);
            Some(
                math_calculate_consensus(
                    &vote_matrix,
                    &weight_vector,
                    &scaled_decisions,
                    &categorical_decisions,
                )
                .map_err(|e| Error::InvalidTransaction {
                    reason: format!("Failed to calculate consensus: {e:?}"),
                })?,
            )
        } else {
            None
        };

        let mut period_stats = self
            .databases
            .get_period_stats(rotxn, period_id)?
            .unwrap_or_else(|| VotingPeriodStats::new(period_id, 0));

        let mut resolved: Vec<(DecisionId, Option<f64>, Option<Option<u16>>)> =
            Vec::new();
        if let Some(math_result) = &math_result {
            period_stats.first_loading =
                Some(math_result.first_loading.clone());
            period_stats.certainty = Some(math_result.certainty);

            let mut score_changes: BTreeMap<crate::types::Address, (f64, f64)> =
                BTreeMap::new();
            for (&voter_id, &new_score) in &math_result.updated_reputations {
                let old_rep =
                    reputations.get(&voter_id).copied().unwrap_or(0.0);
                score_changes.insert(voter_id, (old_rep, new_score));
            }
            if !score_changes.is_empty() {
                period_stats.reputation_changes = Some(score_changes);
            }

            resolved.extend(math_result.outcomes.iter().map(
                |(decision_id, outcome_value)| {
                    (
                        *decision_id,
                        *outcome_value,
                        math_result
                            .categorical_winners
                            .get(decision_id)
                            .copied(),
                    )
                },
            ));
        }

        for (decision_id, votes) in &unguaranteed_votes {
            let votes: Vec<(f64, f64)> = votes.values().copied().collect();
            let Some((outcome_value, winner)) =
                crate::math::voting::oracle::resolve_unguaranteed(
                    &votes,
                    scaled_decisions.contains(decision_id),
                    categorical_decisions.get(decision_id).copied(),
                )
                .map_err(|e| Error::InvalidTransaction {
                    reason: format!(
                        "Failed to resolve non-standard decision: {e:?}"
                    ),
                })?
            else {
                continue;
            };
            resolved.push((
                *decision_id,
                Some(outcome_value),
                categorical_decisions
                    .contains_key(decision_id)
                    .then_some(winner),
            ));
        }

        let mut decision_outcomes = Vec::new();
        let mut decision_resolutions = Vec::new();
        let mut resolved_by_svd: HashSet<DecisionId> = HashSet::new();

        for (decision_id, outcome_value, cat_winner) in &resolved {
            let outcome_f64 = match outcome_value {
                Some(v) => *v,
                None => {
                    let default_value = categorical_default(*decision_id);
                    tracing::info!(
                        "Decision {} has unanimous abstention \
                         — using default {:.4}",
                        const_hex::encode(decision_id.as_bytes()),
                        default_value
                    );
                    default_value
                }
            };

            resolved_by_svd.insert(*decision_id);

            let mut resolution =
                DecisionResolution::new(*decision_id, period_id, 0, 1, 0, 0);
            resolution.mark_outcome_ready();

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
                *cat_winner,
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
                    const_hex::encode(decision_id.as_bytes()),
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
        if let Some(math_result) = math_result {
            result.updated_reputations = math_result.updated_reputations;
        }

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

    /// Updates overwrite prior reputation and zeroed voters are dropped
    #[test]
    fn compute_post_replaces() {
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
    fn per_block_accepts_epsilon() {
        let pre = pre_state_unit();
        let mut post = pre.clone();
        post.insert(addr(1), 0.6 + 1e-11);
        post.insert(addr(2), 0.4 - 1e-11);
        assert!(verify_reputation_conservation(&pre, &post).is_ok());
    }

    #[test]
    fn anchor_rejects_drift() {
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
    fn anchor_accepts_epsilon() {
        let mut pre = BTreeMap::new();
        pre.insert(addr(1), 0.6 + 5e-8);
        pre.insert(addr(2), 0.4);
        let post = pre.clone();
        assert!(verify_reputation_conservation(&pre, &post).is_ok());
    }

    #[test]
    fn compute_post_keeps_others() {
        let pre = pre_state_unit();
        let mut updates = BTreeMap::new();
        updates.insert(addr(1), 0.7);
        let post = compute_post_consensus_reputation(&pre, &updates);
        assert_eq!(post.get(&addr(2)), Some(&0.4));
    }

    #[test]
    fn reanchor_restores_sum() {
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

    /// Drift correction goes to the largest voter with a deterministic tiebreak
    #[test]
    fn reanchor_targets_largest() {
        let mut post = BTreeMap::new();
        post.insert(addr(2), 0.5);
        post.insert(addr(1), 0.5 - 3e-10);
        reanchor_to_initial_total(&mut post);
        let sum: f64 = post.values().sum();
        assert_eq!(sum.to_bits(), 1.0_f64.to_bits());
        assert!(*post.get(&addr(2)).unwrap() > 0.5);
    }

    /// Repeated drift and reanchoring never accumulates error in the sum
    #[test]
    fn reanchor_no_accumulation() {
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

    /// A transaction that spends one coin of `payer` worth `value_sats` and
    /// pays all of it as fee
    fn funded_tx(
        data: crate::types::TransactionData,
        payer: Address,
        value_sats: u64,
    ) -> crate::types::FilledTransaction {
        use crate::types::{OutPoint, Output, OutputContent, PointedOutputRef};

        let input = Output {
            address: payer,
            content: OutputContent::Value(bitcoin::Amount::from_sat(
                value_sats,
            )),
        };
        let outpoint = OutPoint::Regular {
            txid: crate::types::Txid([3; 32]),
            vout: 0,
        };
        let input_hash = crate::types::hash(&PointedOutputRef {
            outpoint,
            output: &input,
        });
        crate::types::FilledTransaction {
            transaction: crate::types::Transaction {
                inputs: vec![(outpoint, input_hash)].into(),
                data: Some(data),
                ..Default::default()
            },
            spent_utxos: vec![input],
            actor_address: Some(payer),
        }
    }

    fn sealed_ballot_tx(
        reveal: &crate::types::RevealBallot,
    ) -> crate::types::FilledTransaction {
        funded_tx(
            crate::types::TransactionData::SubmitBallot {
                voter: reveal.voter,
                voting_period: reveal.voting_period,
                commitment: reveal.commitment(),
            },
            reveal.voter,
            10_000,
        )
    }

    fn reveal_ballot_tx(
        reveal: &crate::types::RevealBallot,
    ) -> crate::types::FilledTransaction {
        funded_tx(
            crate::types::TransactionData::RevealBallot {
                voter: reveal.voter,
                voting_period: reveal.voting_period,
                votes: reveal.votes.clone(),
                salt: reveal.salt,
            },
            reveal.voter,
            10_000,
        )
    }

    /// The transaction alone, without the coin it spends, which these
    /// tests never put in the utxo set
    fn without_inputs(
        filled_tx: &crate::types::FilledTransaction,
    ) -> crate::types::Transaction {
        crate::types::Transaction {
            inputs: Default::default(),
            ..filled_tx.transaction.clone()
        }
    }

    fn transfer_tx(
        sender: Address,
        dest: Address,
        amount: f64,
    ) -> crate::types::FilledTransaction {
        let mut transfer = funded_tx(
            crate::types::TransactionData::TransferReputation {
                sender,
                dest,
                amount,
            },
            sender,
            1000,
        );
        transfer.actor_address = None;
        transfer
    }

    fn binary_ballot(
        voter: Address,
        decision_id: crate::state::decisions::DecisionId,
        vote_value: f64,
        salt: u8,
    ) -> crate::types::RevealBallot {
        crate::types::RevealBallot {
            voter,
            voting_period: decision_id.voting_period(),
            votes: vec![crate::types::BallotItem {
                decision_id_bytes: decision_id.as_bytes(),
                vote_value: Some(vote_value),
            }],
            salt: [salt; 32],
        }
    }

    fn connect_height(
        state: &crate::state::State,
        rwtxn: &mut sneed::RwTxn,
        height: u32,
        transactions: &[crate::types::FilledTransaction],
    ) -> anyhow::Result<()> {
        use crate::{
            state::markets::block,
            types::{AccumulatorDiff, Body, Coinbase},
        };

        let timestamp = 1000 + u64::from(height);
        state.mainchain_timestamp.put(rwtxn, &(), &timestamp)?;
        let context = block::connect_begin(
            state,
            rwtxn,
            &Body::new(vec![], Coinbase::default()),
            height,
            timestamp,
        )?;
        let changes = block::connect_txs(
            state,
            rwtxn,
            &context,
            transactions,
            bitcoin::Amount::ZERO,
        )?;
        block::connect_end(
            state,
            rwtxn,
            &mut AccumulatorDiff::default(),
            context,
            changes,
        )?;
        Ok(())
    }

    fn sealed_ballot_state(
        dir: &tempfile::TempDir,
        reputations: &[(Address, f64)],
    ) -> anyhow::Result<(
        sneed::Env,
        crate::state::State,
        crate::state::decisions::DecisionId,
    )> {
        use crate::{
            state::{
                State,
                decisions::{Decision, DecisionId, DecisionType},
            },
            types::Txid,
        };

        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024).max_dbs(State::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, dir.path()) }?;
        let state = State::new(&env, Some(10), true)?;
        let mut rwtxn = env.write_txn()?;
        for (voter, reputation) in reputations {
            state.reputation().set_reputation(
                &mut rwtxn,
                voter,
                *reputation,
            )?;
        }
        state.genesis_timestamp.put(&mut rwtxn, &(), &1000)?;
        state.decisions().mint_genesis(&mut rwtxn, 1000, 0, 1000)?;
        let decision_id = DecisionId::new(true, 1, 0)?;
        let decision = Decision::new(
            reputations[0].0.0,
            DecisionType::Binary,
            "Test decision".to_owned(),
            String::new(),
            None,
            None,
            vec![],
        )?;
        state.decisions().claim_decision(
            &mut rwtxn,
            decision_id,
            decision,
            Txid([0; 32]),
            Some(1),
        )?;
        rwtxn.commit()?;
        Ok((env, state, decision_id))
    }

    /// Reputation transfers are locked from the sealed ballot until resolution
    #[test]
    fn ballot_locks_reputation() -> anyhow::Result<()> {
        use crate::{
            state::decisions::DecisionState, validation::VoteValidator,
        };

        let dir = tempfile::tempdir()?;
        let sender = addr(1);
        let receiver = addr(2);
        let outsider = addr(3);
        let (env, state, decision_id) = sealed_ballot_state(
            &dir,
            &[(sender, 0.4), (receiver, 0.5), (outsider, 0.1)],
        )?;
        let mut rwtxn = env.write_txn()?;
        let ballots = [(sender, 0.0, 1), (receiver, 1.0, 2)].map(
            |(voter, vote_value, salt)| {
                binary_ballot(voter, decision_id, vote_value, salt)
            },
        );
        let seals = ballots.each_ref().map(sealed_ballot_tx);
        let reveals = ballots.each_ref().map(reveal_ballot_tx);
        let locked_transfers = [
            transfer_tx(sender, outsider, 0.01),
            transfer_tx(outsider, receiver, 0.01),
        ];
        let is_locked = |rwtxn: &sneed::RwTxn| -> anyhow::Result<bool> {
            let mut locked = true;
            for transfer in &locked_transfers {
                locked &= VoteValidator::validate_reputation_transfer(
                    &state, rwtxn, transfer, None,
                )
                .is_err();
            }
            Ok(locked)
        };

        connect_height(&state, &mut rwtxn, 10, &[])?;
        assert!(!is_locked(&rwtxn)?);
        connect_height(&state, &mut rwtxn, 11, &seals)?;
        assert!(is_locked(&rwtxn)?);
        VoteValidator::validate_reputation_transfer(
            &state,
            &rwtxn,
            &transfer_tx(outsider, addr(4), 0.01),
            None,
        )?;
        connect_height(&state, &mut rwtxn, 20, &[])?;
        assert_eq!(
            state
                .decisions()
                .get_decision_current_state(&rwtxn, decision_id)?,
            DecisionState::Revealing,
        );
        assert!(is_locked(&rwtxn)?);
        connect_height(&state, &mut rwtxn, 21, &reveals)?;
        assert_eq!(
            state
                .decisions()
                .get_decision_current_state(&rwtxn, decision_id)?,
            DecisionState::Resolved,
        );
        for transfer in &locked_transfers {
            VoteValidator::validate_reputation_transfer(
                &state, &rwtxn, transfer, None,
            )?;
        }
        Ok(())
    }

    /// A claim in the block opening its voting period votes with it
    #[test]
    fn late_claim_votes() -> anyhow::Result<()> {
        use crate::{
            state::{
                decisions::{
                    Decision, DecisionId, DecisionState, DecisionType,
                },
                markets::block,
            },
            types::{AccumulatorDiff, Body, Coinbase, Txid},
        };

        let dir = tempfile::tempdir()?;
        let (env, state, early) = sealed_ballot_state(&dir, &[(addr(1), 1.0)])?;
        let mut rwtxn = env.write_txn()?;
        let late = DecisionId::new(true, 1, 1)?;
        let height = 10;
        let timestamp = 1000 + u64::from(height);
        state.mainchain_timestamp.put(&mut rwtxn, &(), &timestamp)?;
        let context = block::connect_begin(
            &state,
            &mut rwtxn,
            &Body::new(vec![], Coinbase::default()),
            height,
            timestamp,
        )?;
        state.decisions().claim_decision(
            &mut rwtxn,
            late,
            Decision::new(
                addr(1).0,
                DecisionType::Binary,
                "Late decision".to_owned(),
                String::new(),
                None,
                None,
                vec![],
            )?,
            Txid([1; 32]),
            Some(height),
        )?;
        let changes = block::connect_txs(
            &state,
            &mut rwtxn,
            &context,
            &[],
            bitcoin::Amount::ZERO,
        )?;
        block::connect_end(
            &state,
            &mut rwtxn,
            &mut AccumulatorDiff::default(),
            context,
            changes,
        )?;
        for decision_id in [early, late] {
            assert_eq!(
                state
                    .decisions()
                    .get_decision_current_state(&rwtxn, decision_id)?,
                DecisionState::Voting,
            );
        }
        Ok(())
    }

    /// Sealed ballots resolve once the reveal window closes
    #[test]
    fn sealed_ballots_resolve() -> anyhow::Result<()> {
        use crate::{
            state::{decisions::DecisionState, markets::block},
            types::AccumulatorDiff,
            validation::VoteValidator,
        };

        let dir = tempfile::tempdir()?;
        let [late, early, silent] = [addr(1), addr(2), addr(3)];
        let (env, state, decision_id) = sealed_ballot_state(
            &dir,
            &[(late, 0.5), (early, 0.3), (silent, 0.2)],
        )?;
        let mut rwtxn = env.write_txn()?;
        let period_id = VotingPeriodId::new(decision_id.voting_period());
        let late_ballot = binary_ballot(late, decision_id, 1.0, 1);
        let second_ballot = binary_ballot(late, decision_id, 0.0, 2);
        let early_ballot = binary_ballot(early, decision_id, 1.0, 3);
        let silent_ballot = binary_ballot(silent, decision_id, 0.0, 4);
        let seals = [
            sealed_ballot_tx(&late_ballot),
            sealed_ballot_tx(&early_ballot),
            sealed_ballot_tx(&silent_ballot),
        ];
        let reveals = [
            reveal_ballot_tx(&late_ballot),
            reveal_ballot_tx(&early_ballot),
        ];
        let sealed = |rwtxn: &sneed::RwTxn, voter| {
            state
                .voting()
                .databases()
                .get_sealed_ballot(rwtxn, period_id, voter)
        };

        connect_height(&state, &mut rwtxn, 10, &[])?;
        for seal in &seals {
            VoteValidator::validate_ballot(&state, &rwtxn, seal, None)?;
        }
        connect_height(&state, &mut rwtxn, 11, &seals)?;
        assert!(
            VoteValidator::validate_ballot(
                &state,
                &rwtxn,
                &sealed_ballot_tx(&second_ballot),
                None,
            )
            .is_err()
        );

        connect_height(&state, &mut rwtxn, 20, &[])?;
        assert_eq!(
            state
                .decisions()
                .get_decision_current_state(&rwtxn, decision_id)?,
            DecisionState::Revealing,
        );
        assert!(
            !state
                .voting()
                .databases()
                .has_consensus(&rwtxn, period_id)?
        );
        assert_eq!(state.reputation().get_reputation(&rwtxn, &late)?, 0.5);
        assert!(
            VoteValidator::validate_reveal(
                &state,
                &rwtxn,
                &reveal_ballot_tx(&second_ballot),
                None,
            )
            .is_err()
        );
        for reveal in &reveals {
            VoteValidator::validate_reveal(&state, &rwtxn, reveal, None)?;
        }

        connect_height(&state, &mut rwtxn, 21, &reveals)?;
        assert_eq!(
            state
                .decisions()
                .get_decision_current_state(&rwtxn, decision_id)?,
            DecisionState::Resolved,
        );
        let outcomes =
            state.voting().resolve_period_decisions(&rwtxn, period_id)?;
        assert_eq!(outcomes[0].outcome_value, 1.0);
        let votes = state.voting().get_votes_for_period(&rwtxn, period_id)?;
        assert_eq!(votes.len(), 2);
        assert_eq!(
            state
                .voting()
                .databases()
                .get_vote(&rwtxn, period_id, late, decision_id)?
                .map(|entry| entry.value),
            Some(types::VoteValue::Binary(true)),
        );
        assert!(state.reputation().get_reputation(&rwtxn, &silent)? < 0.2);
        assert!(sealed(&rwtxn, early)?.is_some_and(|ballot| ballot.revealed));
        assert!(sealed(&rwtxn, silent)?.is_some_and(|ballot| !ballot.revealed));

        let mut diff = AccumulatorDiff::default();
        block::disconnect_tx(
            &state,
            &mut rwtxn,
            &mut diff,
            &without_inputs(&reveals[1]),
            &mut Vec::new(),
        )?;
        assert_eq!(
            state
                .voting()
                .get_votes_for_period(&rwtxn, period_id)?
                .len(),
            1
        );
        assert_eq!(
            sealed(&rwtxn, early)?,
            Some(types::SealedBallot {
                commitment: early_ballot.commitment(),
                revealed: false,
            }),
        );
        block::disconnect_tx(
            &state,
            &mut rwtxn,
            &mut diff,
            &without_inputs(&seals[0]),
            &mut Vec::new(),
        )?;
        assert_eq!(sealed(&rwtxn, late)?, None);
        Ok(())
    }

    fn claim_nonstandard(
        state: &crate::state::State,
        rwtxn: &mut sneed::RwTxn,
        period: u32,
    ) -> anyhow::Result<crate::state::decisions::DecisionId> {
        use crate::state::decisions::{Decision, DecisionId, DecisionType};

        let decision_id = DecisionId::new(false, period, 0)?;
        let decision = Decision::new(
            [0; 20],
            DecisionType::Binary,
            "Optional decision".to_owned(),
            String::new(),
            None,
            None,
            vec![],
        )?;
        state.decisions().claim_decision(
            rwtxn,
            decision_id,
            decision,
            crate::types::Txid([1; 32]),
            Some(1),
        )?;
        Ok(decision_id)
    }

    fn ballot(
        voter: Address,
        voting_period: u32,
        votes: &[(crate::state::decisions::DecisionId, f64)],
        salt: u8,
    ) -> crate::types::RevealBallot {
        crate::types::RevealBallot {
            voter,
            voting_period,
            votes: votes
                .iter()
                .map(|(decision_id, vote_value)| crate::types::BallotItem {
                    decision_id_bytes: decision_id.as_bytes(),
                    vote_value: Some(*vote_value),
                })
                .collect(),
            salt: [salt; 32],
        }
    }

    fn outcome_of(
        state: &crate::state::State,
        rotxn: &sneed::RoTxn,
        decision_id: crate::state::decisions::DecisionId,
    ) -> anyhow::Result<Option<f64>> {
        let period_id = VotingPeriodId::new(decision_id.voting_period());
        Ok(state
            .voting()
            .resolve_period_decisions(rotxn, period_id)?
            .into_iter()
            .find(|outcome| outcome.decision_id == decision_id)
            .map(|outcome| outcome.outcome_value))
    }

    /// A non-standard decision resolves from the reputation behind its
    /// votes, and dissent on it costs no reputation
    #[test]
    fn nonstandard_skips_svd() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let voters = [(addr(1), 0.5), (addr(2), 0.3), (addr(3), 0.2)];
        let (env, state, standard) = sealed_ballot_state(&dir, &voters)?;
        let mut rwtxn = env.write_txn()?;
        let optional = claim_nonstandard(&state, &mut rwtxn, 1)?;
        let voting_period = standard.voting_period();
        let ballots: Vec<_> = voters
            .iter()
            .zip([0.0, 0.0, 1.0])
            .enumerate()
            .map(|(salt, ((voter, _), optional_vote))| {
                ballot(
                    *voter,
                    voting_period,
                    &[(standard, 1.0), (optional, optional_vote)],
                    salt as u8,
                )
            })
            .collect();

        connect_height(&state, &mut rwtxn, 10, &[])?;
        let seals: Vec<_> = ballots.iter().map(sealed_ballot_tx).collect();
        connect_height(&state, &mut rwtxn, 11, &seals)?;
        connect_height(&state, &mut rwtxn, 20, &[])?;
        let reveals: Vec<_> = ballots.iter().map(reveal_ballot_tx).collect();
        connect_height(&state, &mut rwtxn, 21, &reveals)?;

        assert_eq!(outcome_of(&state, &rwtxn, standard)?, Some(1.0));
        assert_eq!(outcome_of(&state, &rwtxn, optional)?, Some(0.0));
        for (voter, reputation) in voters {
            let after = state.reputation().get_reputation(&rwtxn, &voter)?;
            assert!((after - reputation).abs() < 1e-9);
        }
        Ok(())
    }

    /// A period of only non-standard decisions resolves them and leaves
    /// reputation as it was
    #[test]
    fn nonstandard_only_period() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let voters = [(addr(1), 0.5), (addr(2), 0.3), (addr(3), 0.2)];
        let (env, state, _) = sealed_ballot_state(&dir, &voters)?;
        let mut rwtxn = env.write_txn()?;
        let optional = claim_nonstandard(&state, &mut rwtxn, 2)?;
        let voting_period = optional.voting_period();
        let ballots = [
            ballot(voters[0].0, voting_period, &[(optional, 1.0)], 1),
            ballot(voters[1].0, voting_period, &[(optional, 1.0)], 2),
            ballot(voters[2].0, voting_period, &[], 3),
        ];

        connect_height(&state, &mut rwtxn, 20, &[])?;
        let seals: Vec<_> = ballots.iter().map(sealed_ballot_tx).collect();
        connect_height(&state, &mut rwtxn, 21, &seals)?;
        connect_height(&state, &mut rwtxn, 30, &[])?;
        let reveals: Vec<_> = ballots.iter().map(reveal_ballot_tx).collect();
        connect_height(&state, &mut rwtxn, 31, &reveals)?;

        assert_eq!(outcome_of(&state, &rwtxn, optional)?, Some(1.0));
        let before: BTreeMap<_, _> = voters.into_iter().collect();
        let after = state.reputation().get_all_reputations(&rwtxn)?;
        assert_eq!(after.len(), before.len());
        for (voter, reputation) in &before {
            assert!((after[voter] - reputation).abs() < 1e-9);
        }
        Ok(())
    }
}
