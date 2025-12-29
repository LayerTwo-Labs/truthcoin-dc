//! Bitcoin Hivemind Voting Database Operations
//!
//! The `VoteMatrixKey` structure is ordered as `(period_id, voter_address, decision_id)`.
//! This ordering enables efficient period-based queries using early termination.
//!
//! TODO: Fork sneed to expose heed's `range()` method for O(log n + k) queries.

use crate::math::safe_math::safe_cmp_f64;
use crate::state::{
    Error,
    voting::types::{
        DecisionOutcome, Vote, VoteBatch, VoteMatrixEntry, VoteMatrixKey,
        VoterReputation, VotingPeriodId, VotingPeriodStats,
    },
};
use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use sneed::{DatabaseUnique, Env, RoTxn, RwTxn};
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
pub struct VotingDatabases {
    votes: DatabaseUnique<
        SerdeBincode<VoteMatrixKey>,
        SerdeBincode<VoteMatrixEntry>,
    >,

    vote_batches: DatabaseUnique<
        SerdeBincode<(VotingPeriodId, u32)>,
        SerdeBincode<VoteBatch>,
    >,

    voter_reputation: DatabaseUnique<
        SerdeBincode<crate::types::Address>,
        SerdeBincode<VoterReputation>,
    >,

    decision_outcomes: DatabaseUnique<
        SerdeBincode<crate::state::slots::SlotId>,
        SerdeBincode<DecisionOutcome>,
    >,

    period_stats: DatabaseUnique<
        SerdeBincode<VotingPeriodId>,
        SerdeBincode<VotingPeriodStats>,
    >,

    pending_period_redistributions: DatabaseUnique<
        SerdeBincode<VotingPeriodId>,
        SerdeBincode<
            crate::state::voting::redistribution::PeriodRedistribution,
        >,
    >,
}

impl VotingDatabases {
    pub const NUM_DBS: u32 = 6;

    pub fn new(env: &Env, rwtxn: &mut RwTxn<'_>) -> Result<Self, Error> {
        Ok(Self {
            votes: DatabaseUnique::create(env, rwtxn, "votes")?,
            vote_batches: DatabaseUnique::create(env, rwtxn, "vote_batches")?,
            voter_reputation: DatabaseUnique::create(
                env,
                rwtxn,
                "voter_reputation",
            )?,
            decision_outcomes: DatabaseUnique::create(
                env,
                rwtxn,
                "decision_outcomes",
            )?,
            period_stats: DatabaseUnique::create(env, rwtxn, "period_stats")?,
            pending_period_redistributions: DatabaseUnique::create(
                env,
                rwtxn,
                "pending_period_redistributions",
            )?,
        })
    }

    pub fn put_vote(
        &self,
        rwtxn: &mut RwTxn,
        vote: &Vote,
    ) -> Result<(), Error> {
        let key = VoteMatrixKey::new(
            vote.period_id,
            vote.voter_address,
            vote.decision_id,
        );
        let entry =
            VoteMatrixEntry::new(vote.value, vote.timestamp, vote.block_height);

        tracing::debug!(
            "put_vote: Storing vote for period {}, voter {}, decision {:?}, value {:?}",
            vote.period_id.0,
            vote.voter_address.as_base58(),
            vote.decision_id.to_hex(),
            vote.value
        );

        self.votes.put(rwtxn, &key, &entry)?;
        Ok(())
    }

    pub fn get_vote(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
        voter_address: crate::types::Address,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<Option<VoteMatrixEntry>, Error> {
        let key = VoteMatrixKey::new(period_id, voter_address, decision_id);
        Ok(self.votes.try_get(rotxn, &key)?)
    }

    pub fn get_votes_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<VoteMatrixKey, VoteMatrixEntry>, Error> {
        let mut votes = HashMap::new();
        let mut iter = self.votes.iter(rotxn)?;
        let mut total_votes_scanned = 0;

        while let Some((key, entry)) = iter.next()? {
            total_votes_scanned += 1;

            match key.period_id.cmp(&period_id) {
                std::cmp::Ordering::Less => {
                    continue;
                }
                std::cmp::Ordering::Equal => {
                    votes.insert(key, entry);
                }
                std::cmp::Ordering::Greater => {
                    tracing::debug!(
                        "get_votes_for_period: Early termination at period {} (target: {}), \
                        scanned {} votes, found {} matches",
                        key.period_id.0,
                        period_id.0,
                        total_votes_scanned,
                        votes.len()
                    );
                    break;
                }
            }
        }

        tracing::debug!(
            "get_votes_for_period: Period {}, found {} votes after scanning {} entries",
            period_id.0,
            votes.len(),
            total_votes_scanned
        );

        Ok(votes)
    }

    pub fn get_votes_by_voter(
        &self,
        rotxn: &RoTxn,
        voter_address: crate::types::Address,
    ) -> Result<HashMap<VoteMatrixKey, VoteMatrixEntry>, Error> {
        let mut votes = HashMap::new();
        let mut iter = self.votes.iter(rotxn)?;

        while let Some((key, entry)) = iter.next()? {
            if key.voter_address == voter_address {
                votes.insert(key, entry);
            }
        }

        Ok(votes)
    }

    pub fn get_votes_for_decision(
        &self,
        rotxn: &RoTxn,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<HashMap<VoteMatrixKey, VoteMatrixEntry>, Error> {
        let mut votes = HashMap::new();
        let mut iter = self.votes.iter(rotxn)?;

        while let Some((key, entry)) = iter.next()? {
            if key.decision_id == decision_id {
                votes.insert(key, entry);
            }
        }

        Ok(votes)
    }

    pub fn delete_vote(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        voter_address: crate::types::Address,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<bool, Error> {
        let key = VoteMatrixKey::new(period_id, voter_address, decision_id);
        Ok(self.votes.delete(rwtxn, &key)?)
    }

    pub fn put_vote_batch(
        &self,
        rwtxn: &mut RwTxn,
        batch: &VoteBatch,
        batch_index: u32,
    ) -> Result<(), Error> {
        let key = (batch.period_id, batch_index);
        self.vote_batches.put(rwtxn, &key, batch)?;
        Ok(())
    }

    pub fn get_vote_batch(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
        batch_index: u32,
    ) -> Result<Option<VoteBatch>, Error> {
        let key = (period_id, batch_index);
        Ok(self.vote_batches.try_get(rotxn, &key)?)
    }

    pub fn get_vote_batches_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<Vec<VoteBatch>, Error> {
        let mut batches = Vec::new();
        let mut iter = self.vote_batches.iter(rotxn)?;

        while let Some(((p_id, _batch_index), batch)) = iter.next()? {
            if p_id == period_id {
                batches.push(batch);
            }
        }

        batches.sort_by_key(|batch| batch.created_at);
        Ok(batches)
    }

    pub fn put_voter_reputation(
        &self,
        rwtxn: &mut RwTxn,
        reputation: &VoterReputation,
    ) -> Result<(), Error> {
        self.voter_reputation
            .put(rwtxn, &reputation.address, reputation)?;
        Ok(())
    }

    pub fn get_voter_reputation(
        &self,
        rotxn: &RoTxn,
        address: crate::types::Address,
    ) -> Result<Option<VoterReputation>, Error> {
        Ok(self.voter_reputation.try_get(rotxn, &address)?)
    }

    pub fn get_voter_reputations_batch(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<crate::types::Address>,
    ) -> Result<HashMap<crate::types::Address, VoterReputation>, Error> {
        let mut reputations = HashMap::with_capacity(addresses.len());

        if addresses.is_empty() {
            return Ok(reputations);
        }

        // For small sets, individual lookups are faster due to B-tree efficiency
        if addresses.len() < 10 {
            for &addr in addresses {
                if let Some(rep) =
                    self.voter_reputation.try_get(rotxn, &addr)?
                {
                    reputations.insert(addr, rep);
                }
            }
        } else {
            // For larger sets, single scan of reputation table is more efficient
            let mut iter = self.voter_reputation.iter(rotxn)?;
            while let Some((addr, rep)) = iter.next()? {
                if addresses.contains(&addr) {
                    reputations.insert(addr, rep);
                    if reputations.len() == addresses.len() {
                        break;
                    }
                }
            }
        }

        Ok(reputations)
    }

    pub fn get_reputation_stats(
        &self,
        rotxn: &RoTxn,
    ) -> Result<(u64, f64, f64, f64), Error> {
        let mut reputations = Vec::new();
        let mut iter = self.voter_reputation.iter(rotxn)?;

        while let Some((_voter_address, reputation)) = iter.next()? {
            reputations.push(reputation.reputation);
        }

        if reputations.is_empty() {
            return Ok((0, 0.0, 0.0, 0.0));
        }

        let total_voters = reputations.len() as u64;
        let total_weight: f64 = reputations.iter().sum();
        let avg_reputation = total_weight / total_voters as f64;

        reputations.sort_by(|a, b| safe_cmp_f64(*a, *b));
        let median_reputation = if reputations.len().is_multiple_of(2) {
            let mid = reputations.len() / 2;
            (reputations[mid - 1] + reputations[mid]) / 2.0
        } else {
            reputations[reputations.len() / 2]
        };

        Ok((
            total_voters,
            avg_reputation,
            median_reputation,
            total_weight,
        ))
    }

    pub fn put_decision_outcome(
        &self,
        rwtxn: &mut RwTxn,
        outcome: &DecisionOutcome,
    ) -> Result<(), Error> {
        self.decision_outcomes
            .put(rwtxn, &outcome.decision_id, outcome)?;
        Ok(())
    }

    pub fn get_decision_outcome(
        &self,
        rotxn: &RoTxn,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<Option<DecisionOutcome>, Error> {
        Ok(self.decision_outcomes.try_get(rotxn, &decision_id)?)
    }

    pub fn get_outcomes_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<crate::state::slots::SlotId, DecisionOutcome>, Error>
    {
        let mut outcomes = HashMap::new();
        let mut iter = self.decision_outcomes.iter(rotxn)?;
        let target_period_index = period_id.0.saturating_sub(1);

        while let Some((decision_id, outcome)) = iter.next()? {
            let slot_period_index = decision_id.period_index();

            if slot_period_index < target_period_index {
                continue;
            } else if slot_period_index == target_period_index {
                if outcome.period_id == period_id {
                    outcomes.insert(decision_id, outcome);
                }
            } else {
                break;
            }
        }

        Ok(outcomes)
    }

    pub fn put_period_stats(
        &self,
        rwtxn: &mut RwTxn,
        stats: &VotingPeriodStats,
    ) -> Result<(), Error> {
        self.period_stats.put(rwtxn, &stats.period_id, stats)?;
        Ok(())
    }

    pub fn get_period_stats(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<Option<VotingPeriodStats>, Error> {
        Ok(self.period_stats.try_get(rotxn, &period_id)?)
    }

    pub fn count_total_votes(&self, rotxn: &RoTxn) -> Result<u64, Error> {
        let mut count = 0;
        let mut iter = self.votes.iter(rotxn)?;

        while iter.next()?.is_some() {
            count += 1;
        }

        Ok(count)
    }

    pub fn get_all_voters(
        &self,
        rotxn: &RoTxn,
    ) -> Result<HashSet<crate::types::Address>, Error> {
        let mut voters = HashSet::new();
        let mut iter = self.votes.iter(rotxn)?;

        while let Some((key, _entry)) = iter.next()? {
            voters.insert(key.voter_address);
        }

        Ok(voters)
    }

    pub fn check_consistency(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<String>, Error> {
        let mut issues = Vec::new();

        let mut vote_count = 0;
        let mut vote_iter = self.votes.iter(rotxn)?;
        while let Some((_key, _entry)) = vote_iter.next()? {
            vote_count += 1;
        }

        let mut outcome_count = 0;
        let mut outcome_iter = self.decision_outcomes.iter(rotxn)?;
        while let Some((_decision_id, outcome)) = outcome_iter.next()? {
            outcome_count += 1;

            let expected_period = outcome.decision_id.period_index() + 1;
            if outcome.period_id.as_u32() != expected_period {
                issues.push(format!(
                    "Outcome period mismatch: decision {:?} claims period {:?} but should be {}",
                    outcome.decision_id,
                    outcome.period_id,
                    expected_period
                ));
            }
        }

        if issues.is_empty() {
            issues.push(format!(
                "Database consistent: {vote_count} votes, {outcome_count} outcomes"
            ));
        }

        Ok(issues)
    }

    pub fn get_consensus_outcome(
        &self,
        rotxn: &RoTxn,
        _period_id: VotingPeriodId,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<Option<f64>, Error> {
        if let Some(outcome) =
            self.decision_outcomes.try_get(rotxn, &decision_id)?
        {
            Ok(Some(outcome.outcome_value))
        } else {
            Ok(None)
        }
    }

    pub fn get_consensus_outcomes_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<crate::state::slots::SlotId, f64>, Error> {
        let mut outcomes = HashMap::new();
        let mut iter = self.decision_outcomes.iter(rotxn)?;
        let target_period_index = period_id.0.saturating_sub(1);

        while let Some((decision_id, outcome)) = iter.next()? {
            let slot_period_index = decision_id.period_index();

            if slot_period_index < target_period_index {
                continue;
            } else if slot_period_index == target_period_index {
                if outcome.period_id == period_id {
                    outcomes.insert(decision_id, outcome.outcome_value);
                }
            } else {
                break;
            }
        }

        Ok(outcomes)
    }

    pub fn has_consensus(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<bool, Error> {
        let outcomes =
            self.get_consensus_outcomes_for_period(rotxn, period_id)?;
        Ok(!outcomes.is_empty())
    }

    pub fn put_pending_redistribution(
        &self,
        rwtxn: &mut RwTxn,
        redistribution: &crate::state::voting::redistribution::PeriodRedistribution,
    ) -> Result<(), Error> {
        self.pending_period_redistributions.put(
            rwtxn,
            &redistribution.period_id,
            redistribution,
        )?;
        Ok(())
    }

    pub fn get_pending_redistribution(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<
        Option<crate::state::voting::redistribution::PeriodRedistribution>,
        Error,
    > {
        Ok(self
            .pending_period_redistributions
            .try_get(rotxn, &period_id)?)
    }

    pub fn get_all_pending_redistributions(
        &self,
        rotxn: &RoTxn,
    ) -> Result<
        HashMap<
            VotingPeriodId,
            crate::state::voting::redistribution::PeriodRedistribution,
        >,
        Error,
    > {
        let mut redistributions = HashMap::new();
        let mut iter = self.pending_period_redistributions.iter(rotxn)?;

        let mut total_count = 0;
        let mut pending_count = 0;

        while let Some((period_id, redistribution)) = iter.next()? {
            total_count += 1;
            if redistribution.is_pending() {
                pending_count += 1;
                tracing::debug!(
                    "Found pending redistribution for period {} (applied={})",
                    period_id.0,
                    redistribution.applied
                );
                redistributions.insert(period_id, redistribution);
            } else {
                tracing::debug!(
                    "Skipping already-applied redistribution for period {} (applied={})",
                    period_id.0,
                    redistribution.applied
                );
            }
        }

        tracing::debug!(
            "get_all_pending_redistributions: Found {} pending out of {} total redistributions",
            pending_count,
            total_count
        );

        Ok(redistributions)
    }

    pub fn mark_redistribution_applied(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        height: u64,
    ) -> Result<(), Error> {
        if let Some(mut redistribution) = self
            .pending_period_redistributions
            .try_get(rwtxn, &period_id)?
        {
            redistribution.mark_applied(height);
            self.pending_period_redistributions.put(
                rwtxn,
                &period_id,
                &redistribution,
            )?;
        }
        Ok(())
    }
}
