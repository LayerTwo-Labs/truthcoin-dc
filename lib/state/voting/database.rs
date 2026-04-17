//! Bitcoin Hivemind Voting Database Operations
//!
//! The `VoteMatrixKey` structure is ordered as `(period_id, voter_address, decision_id)`.
//! This ordering enables efficient period-based queries using early termination.
//!
//! TODO: Fork sneed to expose heed's `range()` method for O(log n + k) queries.

use crate::state::{
    Error,
    voting::types::{
        Ballot, DecisionOutcome, Vote, VoteMatrixEntry, VoteMatrixKey,
        VotingPeriodId, VotingPeriodStats,
    },
};
use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use sneed::{DatabaseUnique, Env, RoTxn, RwTxn};
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Clone)]
pub struct VotingDatabases {
    votes: DatabaseUnique<
        SerdeBincode<VoteMatrixKey>,
        SerdeBincode<VoteMatrixEntry>,
    >,

    ballots: DatabaseUnique<
        SerdeBincode<(VotingPeriodId, u32)>,
        SerdeBincode<Ballot>,
    >,

    decision_outcomes: DatabaseUnique<
        SerdeBincode<crate::state::decisions::DecisionId>,
        SerdeBincode<DecisionOutcome>,
    >,

    period_stats: DatabaseUnique<
        SerdeBincode<VotingPeriodId>,
        SerdeBincode<VotingPeriodStats>,
    >,
}

impl VotingDatabases {
    pub const NUM_DBS: u32 = 4;

    pub fn new(env: &Env, rwtxn: &mut RwTxn<'_>) -> Result<Self, Error> {
        Ok(Self {
            votes: DatabaseUnique::create(env, rwtxn, "votes")?,
            ballots: DatabaseUnique::create(env, rwtxn, "ballots")?,
            decision_outcomes: DatabaseUnique::create(
                env,
                rwtxn,
                "decision_outcomes",
            )?,
            period_stats: DatabaseUnique::create(env, rwtxn, "period_stats")?,
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
        decision_id: crate::state::decisions::DecisionId,
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
        decision_id: crate::state::decisions::DecisionId,
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
        decision_id: crate::state::decisions::DecisionId,
    ) -> Result<bool, Error> {
        let key = VoteMatrixKey::new(period_id, voter_address, decision_id);
        Ok(self.votes.delete(rwtxn, &key)?)
    }

    pub fn put_ballot(
        &self,
        rwtxn: &mut RwTxn,
        ballot: &Ballot,
        ballot_index: u32,
    ) -> Result<(), Error> {
        let key = (ballot.period_id, ballot_index);
        self.ballots.put(rwtxn, &key, ballot)?;
        Ok(())
    }

    pub fn get_ballot(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
        ballot_index: u32,
    ) -> Result<Option<Ballot>, Error> {
        let key = (period_id, ballot_index);
        Ok(self.ballots.try_get(rotxn, &key)?)
    }

    pub fn get_ballots_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<Vec<Ballot>, Error> {
        let mut ballots = Vec::new();
        let mut iter = self.ballots.iter(rotxn)?;

        while let Some(((p_id, _ballot_index), ballot)) = iter.next()? {
            if p_id == period_id {
                ballots.push(ballot);
            }
        }

        ballots.sort_by_key(|ballot| ballot.created_at);
        Ok(ballots)
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
        decision_id: crate::state::decisions::DecisionId,
    ) -> Result<Option<DecisionOutcome>, Error> {
        Ok(self.decision_outcomes.try_get(rotxn, &decision_id)?)
    }

    pub fn get_outcomes_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<
        HashMap<crate::state::decisions::DecisionId, DecisionOutcome>,
        Error,
    > {
        let mut outcomes = HashMap::new();
        let mut iter = self.decision_outcomes.iter(rotxn)?;
        let target_period_index = period_id.0.saturating_sub(1);

        while let Some((decision_id, outcome)) = iter.next()? {
            if decision_id.period_index() == target_period_index
                && outcome.period_id == period_id
            {
                outcomes.insert(decision_id, outcome);
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
        decision_id: crate::state::decisions::DecisionId,
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
    ) -> Result<HashMap<crate::state::decisions::DecisionId, f64>, Error> {
        let mut outcomes = HashMap::new();
        let mut iter = self.decision_outcomes.iter(rotxn)?;
        let target_period_index = period_id.0.saturating_sub(1);

        while let Some((decision_id, outcome)) = iter.next()? {
            if decision_id.period_index() == target_period_index
                && outcome.period_id == period_id
            {
                outcomes.insert(decision_id, outcome.outcome_value);
            }
        }

        Ok(outcomes)
    }

    pub fn has_consensus(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<bool, Error> {
        Ok(self.period_stats.try_get(rotxn, &period_id)?.is_some())
    }

    pub fn delete_decision_outcome(
        &self,
        rwtxn: &mut RwTxn,
        decision_id: crate::state::decisions::DecisionId,
    ) -> Result<bool, Error> {
        Ok(self.decision_outcomes.delete(rwtxn, &decision_id)?)
    }

    pub fn delete_period_stats(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
    ) -> Result<bool, Error> {
        Ok(self.period_stats.delete(rwtxn, &period_id)?)
    }
}
