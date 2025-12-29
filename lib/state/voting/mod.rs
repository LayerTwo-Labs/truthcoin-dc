//! Bitcoin Hivemind Voting Module
//!
//! Implements the Bitcoin Hivemind voting mechanism with Votecoin economic stake integration.
//!
//! ## Voting Weight Formula
//! Final Voting Weight = Base Reputation × Votecoin Holdings Proportion

pub mod database;
pub mod period_calculator;
pub mod redistribution;
pub mod types;

use crate::state::{Error, slots::SlotId};
use database::VotingDatabases;
use parking_lot::Mutex;
use sneed::{Env, RoTxn, RwTxn};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use types::{
    ConsensusResult, DecisionOutcome, DecisionResolution, SlotResolution, Vote,
    VoteValue, VoterReputation, VotingPeriod, VotingPeriodId,
    VotingPeriodStats, VotingPeriodStatus,
};

/// Voting system for Bitcoin Hivemind consensus.
#[derive(Clone)]
pub struct VotingSystem {
    databases: VotingDatabases,
    /// Lock to synchronize concurrent consensus calculations.
    /// Prevents race conditions when multiple periods resolve simultaneously.
    consensus_lock: Arc<Mutex<()>>,
}

impl VotingSystem {
    pub const NUM_DBS: u32 = VotingDatabases::NUM_DBS;

    pub fn new(env: &Env, rwtxn: &mut RwTxn<'_>) -> Result<Self, Error> {
        let databases = VotingDatabases::new(env, rwtxn)?;
        Ok(Self {
            databases,
            consensus_lock: Arc::new(Mutex::new(())),
        })
    }

    pub fn databases(&self) -> &VotingDatabases {
        &self.databases
    }

    pub fn snapshot_votecoin_proportions(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        state: &crate::state::State,
        current_height: u64,
    ) -> Result<(), Error> {
        let votes = self.databases.get_votes_for_period(rwtxn, period_id)?;
        let voters: HashSet<crate::types::Address> =
            votes.keys().map(|k| k.voter_address).collect();

        let votecoin_proportions =
            state.get_votecoin_proportions_batch(rwtxn, &voters)?;

        for voter_address in voters {
            if let Some(mut reputation) =
                self.databases.get_voter_reputation(rwtxn, voter_address)?
            {
                let proportion = votecoin_proportions
                    .get(&voter_address)
                    .copied()
                    .unwrap_or(0.0);

                reputation
                    .update_votecoin_proportion(proportion, current_height);
                self.databases.put_voter_reputation(rwtxn, &reputation)?;
            }
        }

        Ok(())
    }

    /// Pure computation phase no db writes
    fn compute_consensus(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        state: &crate::state::State,
        current_timestamp: u64,
        current_height: u64,
    ) -> Result<Option<ConsensusResult>, Error> {
        use crate::math::voting::{
            SparseVoteMatrix, VotingWeightVector,
            calculate_consensus as math_calculate_consensus,
        };

        let existing_outcomes = self
            .databases
            .get_consensus_outcomes_for_period(rwtxn, period_id)?;
        if !existing_outcomes.is_empty() {
            tracing::warn!(
                "compute_consensus: Consensus already calculated for period {} ({} outcomes exist), skipping",
                period_id.0,
                existing_outcomes.len()
            );
            return Ok(None);
        }

        let all_votes =
            self.databases.get_votes_for_period(rwtxn, period_id)?;
        if all_votes.is_empty() {
            tracing::warn!(
                "compute_consensus: No votes found for period {}, returning empty",
                period_id.0
            );
            return Ok(None);
        }

        // Collect unique voters and decisions from votes
        let mut voters_set: HashSet<crate::types::Address> = HashSet::new();
        let mut decisions_set = HashSet::new();
        for vote_key in all_votes.keys() {
            voters_set.insert(vote_key.voter_address);
            decisions_set.insert(vote_key.decision_id);
        }

        // Batch fetch all voter reputations in one query instead of N individual lookups
        let mut voter_reputations = self
            .databases
            .get_voter_reputations_batch(rwtxn, &voters_set)?;
        let mut new_voter_reputations = Vec::new();

        // Create default reputations for new voters not in database
        for &voter_address in &voters_set {
            if let std::collections::hash_map::Entry::Vacant(entry) =
                voter_reputations.entry(voter_address)
            {
                let default_rep =
                    VoterReputation::new_default(voter_address, 0, period_id);
                new_voter_reputations.push(default_rep.clone());
                entry.insert(default_rep);
            }
        }

        if voter_reputations.is_empty() {
            return Ok(None);
        }

        let voters: Vec<_> = voters_set.into_iter().collect();
        let decisions: Vec<_> = decisions_set.into_iter().collect();
        let mut vote_matrix = SparseVoteMatrix::new(voters, decisions);

        for (vote_key, vote_entry) in &all_votes {
            // Skip abstent votes
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

        let reputation_vector =
            VotingWeightVector::from_voter_reputations(&voter_reputations);
        let math_result =
            math_calculate_consensus(&vote_matrix, &reputation_vector)
                .map_err(|e| Error::InvalidTransaction {
                    reason: format!("Failed to calculate consensus: {e:?}"),
                })?;

        let mut period_stats = self
            .databases
            .get_period_stats(rwtxn, period_id)?
            .unwrap_or_else(|| VotingPeriodStats::new(period_id, 0));

        period_stats.first_loading = Some(math_result.first_loading.clone());
        period_stats.certainty = Some(math_result.certainty);

        let mut reputation_changes = HashMap::new();
        let mut updated_voter_reputations = new_voter_reputations;

        for (voter_id, new_reputation) in math_result.updated_reputations.iter()
        {
            let mut voter_rep =
                voter_reputations.get(voter_id).cloned().unwrap_or_else(|| {
                    VoterReputation::new_default(*voter_id, 0, period_id)
                });

            let old_reputation = voter_rep.reputation;
            reputation_changes
                .insert(*voter_id, (old_reputation, *new_reputation));

            let consensus_txid =
                crate::types::Txid::for_consensus_redistribution(
                    period_id,
                    current_height,
                );
            voter_rep.reputation_history.push(
                old_reputation,
                consensus_txid,
                current_height as u32,
            );
            voter_rep.reputation = *new_reputation;
            voter_rep.last_updated = 0;
            voter_rep.last_period = period_id;

            updated_voter_reputations.push(voter_rep);
        }

        if !reputation_changes.is_empty() {
            period_stats.reputation_changes = Some(reputation_changes.clone());
        }

        let mut decision_outcomes = Vec::new();
        let mut slot_resolutions = Vec::new();

        for (slot_id, outcome_value) in &math_result.outcomes {
            let Some(outcome_f64) = outcome_value else {
                tracing::warn!(
                    "Slot {} has unanimous abstention - no consensus outcome stored",
                    hex::encode(slot_id.as_bytes())
                );
                continue;
            };

            let mut resolution =
                DecisionResolution::new(*slot_id, period_id, 0, 1, 0, 0);
            resolution.mark_outcome_ready();

            let outcome = DecisionOutcome::new(
                *slot_id,
                period_id,
                *outcome_f64,
                0.0,
                1.0,
                1.0,
                all_votes.len() as u64,
                voter_reputations.values().map(|r| r.reputation).sum(),
                0,
                0,
                true,
                resolution,
            );

            decision_outcomes.push(outcome);
            slot_resolutions.push(SlotResolution {
                slot_id: *slot_id,
                outcome_value: *outcome_f64,
            });
        }

        let redistribution_summary =
            redistribution::redistribute_votecoin_after_consensus(
                state,
                rwtxn,
                period_id,
                &reputation_changes,
                current_timestamp,
                current_height,
            )?;

        let resolved_slot_ids: Vec<SlotId> =
            slot_resolutions.iter().map(|r| r.slot_id).collect();

        let period_redistribution = redistribution::PeriodRedistribution::new(
            period_id,
            resolved_slot_ids,
            redistribution_summary.clone(),
            current_height,
        );

        let mut result =
            ConsensusResult::new(period_id, current_height, current_timestamp);
        result.voter_reputations = updated_voter_reputations;
        result.period_stats = period_stats;
        result.decision_outcomes = decision_outcomes;
        result.slot_resolutions = slot_resolutions;
        result.period_redistribution = Some(period_redistribution);
        result.redistribution_summary = Some(redistribution_summary);

        Ok(Some(result))
    }

    fn validate_consensus_commit(
        &self,
        rwtxn: &RwTxn,
        result: &ConsensusResult,
        state: &crate::state::State,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<(), Error> {
        // Validate all slots exist and can be transitioned
        for slot_resolution in &result.slot_resolutions {
            let _history = slots_db
                .get_slot_state_history(rwtxn, slot_resolution.slot_id)?
                .ok_or(Error::InvalidSlotId {
                    reason: format!(
                        "Pre-validation failed: Slot {:?} has no state history",
                        slot_resolution.slot_id
                    ),
                })?;
        }

        // Validate redistribution won't fail due to insufficient balances
        if let Some(ref redistribution_summary) = result.redistribution_summary
        {
            if let Some(existing) = self
                .databases
                .get_pending_redistribution(rwtxn, result.period_id)?
                && existing.applied
            {
                return Ok(());
            }

            let mut voter_net_changes: HashMap<crate::types::Address, i64> =
                HashMap::new();
            for transfer in &redistribution_summary.transfers {
                *voter_net_changes.entry(transfer.from_address).or_insert(0) +=
                    transfer.amount;
            }

            // Validate voters with negative balances have enough votecoin
            for (&address, &net_change) in &voter_net_changes {
                if net_change < 0 {
                    let amount_needed = (-net_change) as u32;
                    let balance =
                        state.get_votecoin_balance(rwtxn, &address)?;
                    if balance < amount_needed {
                        return Err(Error::InvalidTransaction {
                            reason: format!(
                                "Pre-validation failed: Address {} has {} VoteCoin but needs {} for redistribution",
                                address.as_base58(),
                                balance,
                                amount_needed
                            ),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    fn commit_consensus_result(
        &self,
        rwtxn: &mut RwTxn,
        result: ConsensusResult,
        state: &crate::state::State,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<(), Error> {
        tracing::debug!(
            period_id = result.period_id.0,
            block_height = result.block_height,
            voter_count = result.voter_reputations.len(),
            outcome_count = result.decision_outcomes.len(),
            "Starting atomic consensus commit"
        );

        self.validate_consensus_commit(rwtxn, &result, state, slots_db)?;

        // Phase 1: Write voter reputations
        for voter_rep in &result.voter_reputations {
            self.databases.put_voter_reputation(rwtxn, voter_rep)?;
        }
        tracing::trace!(
            count = result.voter_reputations.len(),
            "Wrote voter reputations"
        );

        // Phase 2: Write period stats
        self.databases
            .put_period_stats(rwtxn, &result.period_stats)?;
        tracing::trace!(period_id = result.period_id.0, "Wrote period stats");

        // Phase 3: Write decision outcomes
        for outcome in &result.decision_outcomes {
            self.databases.put_decision_outcome(rwtxn, outcome)?;
        }
        tracing::trace!(
            count = result.decision_outcomes.len(),
            "Wrote decision outcomes"
        );

        // Phase 4: Apply votecoin redistribution
        if let Some(ref redistribution_summary) = result.redistribution_summary
        {
            if let Some(existing) = self
                .databases
                .get_pending_redistribution(rwtxn, result.period_id)?
            {
                if existing.applied {
                    tracing::trace!(
                        period_id = result.period_id.0,
                        applied_at = ?existing.applied_at_height,
                        "Redistribution already applied, skipping"
                    );
                } else {
                    redistribution::apply_votecoin_redistribution(
                        state,
                        rwtxn,
                        redistribution_summary,
                        result.block_height,
                    )?;
                    tracing::trace!(
                        transfer_count = redistribution_summary.transfers.len(),
                        "Applied votecoin redistribution"
                    );
                }
            } else {
                redistribution::apply_votecoin_redistribution(
                    state,
                    rwtxn,
                    redistribution_summary,
                    result.block_height,
                )?;
                tracing::trace!(
                    transfer_count = redistribution_summary.transfers.len(),
                    "Applied votecoin redistribution"
                );
            }
        }

        // Phase 5: Transition slots to resolved state
        for slot_resolution in &result.slot_resolutions {
            slots_db.transition_slot_to_resolved(
                rwtxn,
                slot_resolution.slot_id,
                result.block_height,
                result.timestamp,
                slot_resolution.outcome_value,
            )?;
        }
        tracing::trace!(
            count = result.slot_resolutions.len(),
            "Transitioned slots to resolved"
        );

        // Capture counts before moving period_redistribution
        let resolved_count = result.resolved_slot_count();
        let abstained_count = result.abstained_slot_count();

        // Phase 6: Record redistribution as applied
        if let Some(mut period_redistribution) = result.period_redistribution {
            period_redistribution.mark_applied(result.block_height);
            self.databases
                .put_pending_redistribution(rwtxn, &period_redistribution)?;
        }

        tracing::info!(
            period_id = result.period_id.0,
            block_height = result.block_height,
            resolved = resolved_count,
            abstained = abstained_count,
            "Consensus commit completed successfully"
        );

        Ok(())
    }

    pub(crate) fn calculate_and_store_consensus(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        state: &crate::state::State,
        current_timestamp: u64,
        current_height: u64,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<(), Error> {
        // Acquire consensus lock to prevent concurrent consensus calculations.
        // This ensures votecoin balance reads during validation match state at application time.
        let _consensus_guard = self.consensus_lock.lock();

        tracing::debug!(
            period_id = period_id.0,
            "calculate_and_store_consensus: Starting"
        );

        let result = self.compute_consensus(
            rwtxn,
            period_id,
            state,
            current_timestamp,
            current_height,
        )?;

        if let Some(consensus_result) = result {
            self.commit_consensus_result(
                rwtxn,
                consensus_result,
                state,
                slots_db,
            )?;
        }

        Ok(())
    }

    pub fn get_all_periods(
        &self,
        rotxn: &RoTxn,
        current_timestamp: u64,
        current_height: u32,
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<HashMap<VotingPeriodId, VotingPeriod>, Error> {
        period_calculator::get_all_active_periods(
            rotxn,
            slots_db,
            config,
            current_timestamp,
            current_height,
            &self.databases,
        )
    }

    pub fn get_active_period(
        &self,
        rotxn: &RoTxn,
        current_timestamp: u64,
        current_height: u32,
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<Option<VotingPeriod>, Error> {
        let all_periods = self.get_all_periods(
            rotxn,
            current_timestamp,
            current_height,
            config,
            slots_db,
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
        decision_id: SlotId,
        value: VoteValue,
        timestamp: u64,
        block_height: u64,
        tx_hash: [u8; 32],
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<(), Error> {
        let has_outcomes = self.databases.has_consensus(rwtxn, period_id)?;
        let period = period_calculator::calculate_voting_period(
            rwtxn,
            period_id,
            block_height as u32,
            timestamp,
            config,
            slots_db,
            has_outcomes,
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
    ) -> Result<HashMap<(crate::types::Address, SlotId), VoteValue>, Error>
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
    ) -> Result<HashMap<(crate::types::Address, SlotId), f64>, Error> {
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
        _config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<(u64, u64, f64), Error> {
        let votes = self.databases.get_votes_for_period(rotxn, period_id)?;

        let decision_slots = period_calculator::get_decision_slots_for_period(
            rotxn, period_id, slots_db,
        )?;

        let total_votes = votes.len() as u64;
        let unique_voters: HashSet<crate::types::Address> =
            votes.keys().map(|k| k.voter_address).collect();
        let total_voters = unique_voters.len() as u64;
        let total_decisions = decision_slots.len() as u64;

        let participation_rate = if total_voters > 0 && total_decisions > 0 {
            total_votes as f64 / (total_voters * total_decisions) as f64
        } else {
            0.0
        };

        Ok((total_voters, total_votes, participation_rate))
    }

    pub fn initialize_voter_reputation(
        &self,
        rwtxn: &mut RwTxn,
        voter_address: crate::types::Address,
        initial_reputation: f64,
        timestamp: u64,
        period_id: VotingPeriodId,
    ) -> Result<(), Error> {
        if self
            .databases
            .get_voter_reputation(rwtxn, voter_address)?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Voter {voter_address:?} already has reputation"
                ),
            });
        }

        let reputation = VoterReputation::new(
            voter_address,
            initial_reputation,
            timestamp,
            period_id,
        );
        self.databases.put_voter_reputation(rwtxn, &reputation)?;

        Ok(())
    }

    pub fn get_reputation_weights(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<crate::types::Address, f64>, Error> {
        let votes = self.databases.get_votes_for_period(rotxn, period_id)?;
        let voters: HashSet<crate::types::Address> =
            votes.keys().map(|k| k.voter_address).collect();
        let mut weights = HashMap::new();

        for voter_address in voters {
            let reputation = self
                .databases
                .get_voter_reputation(rotxn, voter_address)?
                .unwrap_or_else(|| {
                    VoterReputation::new_default(voter_address, 0, period_id)
                });

            weights.insert(voter_address, reputation.get_voting_weight());
        }

        Ok(weights)
    }

    pub fn get_fresh_reputation_weights(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        state: &crate::state::State,
        current_height: u64,
    ) -> Result<HashMap<crate::types::Address, f64>, Error> {
        let votes = self.databases.get_votes_for_period(rwtxn, period_id)?;
        let voters: HashSet<crate::types::Address> =
            votes.keys().map(|k| k.voter_address).collect();

        let votecoin_proportions =
            state.get_votecoin_proportions_batch(rwtxn, &voters)?;

        let mut weights = HashMap::new();

        for voter_address in voters {
            let mut reputation = self
                .databases
                .get_voter_reputation(rwtxn, voter_address)?
                .unwrap_or_else(|| {
                    VoterReputation::new_default(voter_address, 0, period_id)
                });

            if reputation.needs_votecoin_refresh(
                current_height,
                crate::math::voting::constants::VOTECOIN_STALENESS_BLOCKS,
            ) {
                let proportion = votecoin_proportions
                    .get(&voter_address)
                    .copied()
                    .unwrap_or(0.0);
                reputation
                    .update_votecoin_proportion(proportion, current_height);

                self.databases.put_voter_reputation(rwtxn, &reputation)?;
            }

            weights.insert(voter_address, reputation.get_voting_weight());
        }

        Ok(weights)
    }

    pub fn store_decision_outcome(
        &self,
        rwtxn: &mut RwTxn,
        outcome: DecisionOutcome,
    ) -> Result<(), Error> {
        if self
            .databases
            .get_decision_outcome(rwtxn, outcome.decision_id)?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Outcome already exists for decision {:?} in period {:?}",
                    outcome.decision_id, outcome.period_id
                ),
            });
        }

        self.databases.put_decision_outcome(rwtxn, &outcome)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn resolve_period_decisions(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        current_timestamp: u64,
        block_height: u64,
        state: &crate::state::State,
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<Vec<DecisionOutcome>, Error> {
        let has_outcomes = self.databases.has_consensus(rwtxn, period_id)?;
        let period = period_calculator::calculate_voting_period(
            rwtxn,
            period_id,
            block_height as u32,
            current_timestamp,
            config,
            slots_db,
            has_outcomes,
        )?;

        let effective_time = if config.testing_mode {
            block_height
        } else {
            current_timestamp
        };

        period_calculator::validate_transition(
            &period,
            VotingPeriodStatus::Resolved,
            effective_time,
        )?;

        let consensus_outcomes = self
            .databases
            .get_consensus_outcomes_for_period(rwtxn, period_id)?;

        if consensus_outcomes.is_empty() {
            return Err(Error::ConsensusNotYetCalculated(period_id));
        }

        let reputation_weights = self.get_fresh_reputation_weights(
            rwtxn,
            period_id,
            state,
            block_height,
        )?;

        let votes = self.databases.get_votes_for_period(rwtxn, period_id)?;

        let mut voter_reputations = HashMap::new();
        for vote_key in votes.keys() {
            if !voter_reputations.contains_key(&vote_key.voter_address)
                && let Some(rep) = self
                    .databases
                    .get_voter_reputation(rwtxn, vote_key.voter_address)?
            {
                voter_reputations.insert(vote_key.voter_address, rep);
            }
        }

        let period_stats = self.databases.get_period_stats(rwtxn, period_id)?;
        let certainty = period_stats
            .and_then(|stats| stats.certainty)
            .unwrap_or(0.5);

        let mut outcomes = Vec::new();
        for decision_id in &period.decision_slots {
            let outcome_value =
                consensus_outcomes.get(decision_id).copied().unwrap_or(0.5);

            let decision_votes_count = votes
                .iter()
                .filter(|(key, _)| key.decision_id == *decision_id)
                .count();

            let mut resolution = DecisionResolution::new(
                *decision_id,
                period_id,
                period.end_timestamp,
                1,
                current_timestamp,
                block_height,
            );
            resolution.update_status(
                types::DecisionResolutionStatus::Resolved,
                current_timestamp,
                block_height,
                Some("Consensus reached via SVD".to_string()),
            );
            resolution.mark_outcome_ready();

            let outcome = DecisionOutcome::new(
                *decision_id,
                period_id,
                outcome_value,
                0.0,
                1.0,
                certainty,
                decision_votes_count as u64,
                reputation_weights.values().sum(),
                current_timestamp,
                block_height,
                true,
                resolution,
            );

            self.databases.put_decision_outcome(rwtxn, &outcome)?;
            outcomes.push(outcome);
        }

        Ok(outcomes)
    }

    pub fn get_period_outcomes(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<SlotId, DecisionOutcome>, Error> {
        self.databases.get_outcomes_for_period(rotxn, period_id)
    }

    pub fn calculate_period_statistics(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
        current_height: u32,
        current_timestamp: u64,
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<VotingPeriodStats, Error> {
        let (total_voters, total_votes, participation_rate) =
            self.get_participation_stats(rotxn, period_id, config, slots_db)?;

        let has_outcomes = self.databases.has_consensus(rotxn, period_id)?;
        let period = period_calculator::calculate_voting_period(
            rotxn,
            period_id,
            current_height,
            current_timestamp,
            config,
            slots_db,
            has_outcomes,
        )?;

        let reputation_weights =
            self.get_reputation_weights(rotxn, period_id)?;
        let total_reputation_weight: f64 = reputation_weights.values().sum();

        let outcomes =
            self.databases.get_outcomes_for_period(rotxn, period_id)?;
        let consensus_decisions = outcomes
            .values()
            .filter(|outcome| outcome.is_consensus)
            .count() as u64;

        let mut stats = VotingPeriodStats::new(period_id, current_timestamp);
        stats.total_voters = total_voters;
        stats.total_votes = total_votes;
        stats.total_decisions = period.decision_slots.len() as u64;
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
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<(u64, u64, u64, f64), Error> {
        let total_votes = self.databases.count_total_votes(rotxn)?;
        let all_voters = self.databases.get_all_voters(rotxn)?;
        let total_voters = all_voters.len() as u64;

        let (_, avg_reputation, _, _) =
            self.databases.get_reputation_stats(rotxn)?;

        let all_slots = slots_db.get_all_claimed_slots(rotxn)?;
        let mut unique_periods = std::collections::HashSet::new();
        for slot in all_slots {
            let voting_period = slot.slot_id.voting_period();
            unique_periods.insert(voting_period);
        }
        let total_periods = unique_periods.len() as u64;

        Ok((total_periods, total_votes, total_voters, avg_reputation))
    }
}
