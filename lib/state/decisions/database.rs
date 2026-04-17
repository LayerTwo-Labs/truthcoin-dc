use crate::state::Error;
use crate::types::Txid;
use crate::validation::DecisionValidationInterface;
use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use sneed::{DatabaseUnique, Env, RoTxn, RwTxn};
use std::collections::BTreeSet;

use super::types::{
    Decision, DecisionConfig, DecisionEntry, DecisionId, DecisionState,
    DecisionStateHistory, FUTURE_PERIODS, INITIAL_DECISIONS_PER_PERIOD,
    calculate_listing_fee, period_to_string,
};

const DECISIONS_DECLINING_RATE: u64 = 25;

#[derive(Clone)]
pub struct Dbs {
    period_decisions: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<BTreeSet<DecisionEntry>>,
    >,
    decision_state_histories: DatabaseUnique<
        SerdeBincode<DecisionId>,
        SerdeBincode<DecisionStateHistory>,
    >,
    config: DecisionConfig,
}

impl Dbs {
    pub const NUM_DBS: u32 = 2;

    /// Derive claimed decision IDs from period_decisions (a decision is claimed if decision.is_some())
    fn get_claimed_decision_ids_for_period(
        &self,
        rotxn: &RoTxn,
        period_index: u32,
    ) -> Result<BTreeSet<DecisionId>, Error> {
        let period_decisions = self
            .period_decisions
            .try_get(rotxn, &period_index)?
            .unwrap_or_default();
        Ok(period_decisions
            .iter()
            .filter(|entry| entry.decision.is_some())
            .map(|entry| entry.decision_id)
            .collect())
    }

    pub fn new(env: &Env, rwtxn: &mut RwTxn<'_>) -> Result<Self, Error> {
        Self::new_with_config(env, rwtxn, DecisionConfig::default())
    }

    pub fn new_with_config(
        env: &Env,
        rwtxn: &mut RwTxn<'_>,
        config: DecisionConfig,
    ) -> Result<Self, Error> {
        Ok(Self {
            period_decisions: DatabaseUnique::create(
                env,
                rwtxn,
                "period_decisions",
            )?,
            decision_state_histories: DatabaseUnique::create(
                env,
                rwtxn,
                "decision_state_histories",
            )?,
            config,
        })
    }

    pub fn get_current_period(
        &self,
        ts_secs: u64,
        block_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<u32, Error> {
        crate::state::voting::period_calculator::get_current_period(
            ts_secs,
            block_height,
            genesis_ts,
            &self.config,
        )
    }

    fn get_active_window(
        &self,
        ts_secs: u64,
        block_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<(u32, u32), Error> {
        let current =
            self.get_current_period(ts_secs, block_height, genesis_ts)?;
        Ok((current, current + FUTURE_PERIODS - 1))
    }

    #[inline]
    const fn calculate_available_decisions(
        &self,
        period: u32,
        current_period: u32,
    ) -> u64 {
        if period < current_period {
            return 0;
        }

        let offset = period.saturating_sub(current_period);
        if offset >= FUTURE_PERIODS {
            return 0;
        }

        INITIAL_DECISIONS_PER_PERIOD
            .saturating_sub((offset as u64) * DECISIONS_DECLINING_RATE)
    }

    pub fn mint_genesis(
        &self,
        rwtxn: &mut RwTxn<'_>,
        ts_secs: u64,
        block_height: u32,
        genesis_ts: u64,
    ) -> Result<(), Error> {
        let current_period =
            self.get_current_period(ts_secs, Some(block_height), genesis_ts)?;
        self.mint_periods_up_to(rwtxn, current_period + FUTURE_PERIODS - 1)?;
        Ok(())
    }

    pub fn mint_up_to(
        &self,
        rwtxn: &mut RwTxn<'_>,
        ts_secs: u64,
        block_height: u32,
        genesis_ts: u64,
    ) -> Result<(), Error> {
        let current_period =
            self.get_current_period(ts_secs, Some(block_height), genesis_ts)?;
        let target_period = current_period + FUTURE_PERIODS - 1;
        self.mint_periods_up_to(rwtxn, target_period)?;
        Ok(())
    }

    fn mint_periods_up_to(
        &self,
        rwtxn: &mut RwTxn<'_>,
        target_period: u32,
    ) -> Result<(), Error> {
        let mut highest_existing = 0u32;
        {
            let mut iter = self.period_decisions.iter(rwtxn)?;
            while let Some((period, _)) = iter.next()? {
                if period > highest_existing {
                    highest_existing = period;
                }
            }
        }

        for period in (highest_existing + 1)..=target_period {
            if self.period_decisions.try_get(rwtxn, &period)?.is_none() {
                let empty_period: BTreeSet<DecisionEntry> = BTreeSet::new();
                self.period_decisions.put(rwtxn, &period, &empty_period)?;
            }
        }

        Ok(())
    }

    pub fn get_highest_minted_period(
        &self,
        rotxn: &sneed::RoTxn,
    ) -> Result<Option<u32>, Error> {
        let mut highest: Option<u32> = None;
        let mut iter = self.period_decisions.iter(rotxn)?;
        while let Some((period, _)) = iter.next()? {
            highest = Some(match highest {
                Some(h) => h.max(period),
                None => period,
            });
        }
        Ok(highest)
    }

    pub fn delete_periods_above(
        &self,
        rwtxn: &mut RwTxn<'_>,
        max_period: u32,
    ) -> Result<(), Error> {
        let mut to_delete = Vec::new();
        {
            let mut iter = self.period_decisions.iter(rwtxn)?;
            while let Some((period, _)) = iter.next()? {
                if period > max_period {
                    to_delete.push(period);
                }
            }
        }
        for period in to_delete {
            self.period_decisions.delete(rwtxn, &period)?;
        }
        Ok(())
    }

    pub fn total_for(
        &self,
        _rotxn: &sneed::RoTxn,
        period: u32,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<u64, Error> {
        let current_period =
            self.get_current_period(current_ts, current_height, genesis_ts)?;
        Ok(self.calculate_available_decisions(period, current_period))
    }

    pub fn get_active_periods(
        &self,
        _rotxn: &sneed::RoTxn,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<Vec<(u32, u64)>, Error> {
        let current_period =
            self.get_current_period(current_ts, current_height, genesis_ts)?;
        let mut periods = Vec::new();

        for period in current_period..=(current_period + FUTURE_PERIODS - 1) {
            let decisions =
                self.calculate_available_decisions(period, current_period);
            if decisions > 0 {
                periods.push((period, decisions));
            }
        }

        Ok(periods)
    }

    pub fn is_testing_mode(&self) -> bool {
        self.config.is_blocks
    }

    pub fn get_testing_blocks_per_period(&self) -> u32 {
        self.config.quantity
    }

    pub fn block_height_to_testing_period(&self, block_height: u32) -> u32 {
        crate::state::voting::period_calculator::get_current_period(
            0,
            Some(block_height),
            0,
            &self.config,
        )
        .unwrap_or(0)
    }

    pub fn get_config(&self) -> &DecisionConfig {
        &self.config
    }

    pub fn period_to_string(&self, period_idx: u32) -> String {
        period_to_string(period_idx, &self.config)
    }

    pub fn is_period_ossified(
        &self,
        decision_period: u32,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> bool {
        let current_period = self
            .get_current_period(current_ts, current_height, genesis_ts)
            .unwrap_or(0);
        current_period > decision_period.saturating_add(7)
    }

    pub fn is_decision_ossified(
        &self,
        decision_id: DecisionId,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> bool {
        let period = decision_id.period_index();
        self.is_period_ossified(period, current_ts, current_height, genesis_ts)
    }

    pub fn is_decision_in_voting(
        &self,
        rotxn: &RoTxn,
        decision_id: DecisionId,
    ) -> Result<bool, Error> {
        Ok(self.get_decision_current_state(rotxn, decision_id)?
            == DecisionState::Voting)
    }

    fn decisions_needing_transition(
        &self,
        rotxn: &RoTxn,
        from_state: DecisionState,
        threshold_period: impl Fn(&DecisionId) -> u32,
        current_period: u32,
    ) -> Result<Vec<DecisionId>, Error> {
        let mut result = Vec::new();
        let mut iter = self.decision_state_histories.iter(rotxn)?;
        while let Some((decision_id, history)) = iter.next()? {
            if history.current_state() == from_state
                && current_period > threshold_period(&decision_id)
            {
                result.push(decision_id);
            }
        }
        Ok(result)
    }

    pub fn get_claimed_decisions_needing_voting(
        &self,
        rotxn: &RoTxn,
        current_period: u32,
    ) -> Result<Vec<DecisionId>, Error> {
        self.decisions_needing_transition(
            rotxn,
            DecisionState::Claimed,
            |id| id.period_index(),
            current_period,
        )
    }

    pub fn get_voting_decisions_needing_resolution(
        &self,
        rotxn: &RoTxn,
        current_period: u32,
    ) -> Result<Vec<DecisionId>, Error> {
        self.decisions_needing_transition(
            rotxn,
            DecisionState::Voting,
            |id| id.voting_period(),
            current_period,
        )
    }

    pub fn get_ossified_decisions(
        &self,
        rotxn: &sneed::RoTxn,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<Vec<DecisionEntry>, Error> {
        let mut ossified_entries = Vec::new();

        let mut iter = self.period_decisions.iter(rotxn)?;
        while let Some((period, entries)) = iter.next()? {
            if self.is_period_ossified(
                period,
                current_ts,
                current_height,
                genesis_ts,
            ) {
                ossified_entries.extend(entries.iter().cloned());
            }
        }

        Ok(ossified_entries)
    }

    pub fn validate_decision_claim(
        &self,
        rotxn: &RoTxn,
        decision_id: DecisionId,
        _decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<(), Error> {
        let period_index = decision_id.period_index();
        let decision_index = decision_id.decision_index();
        let current_period =
            self.get_current_period(current_ts, current_height, genesis_ts)?;

        if self.is_decision_ossified(
            decision_id,
            current_ts,
            current_height,
            genesis_ts,
        ) {
            return Err(Error::DecisionNotAvailable {
                decision_id,
                reason: format!("Decision period {period_index} is ossified"),
            });
        }

        if self.is_decision_in_voting(rotxn, decision_id)? {
            return Err(Error::DecisionNotAvailable {
                decision_id,
                reason: format!(
                    "Decision {decision_id:?} is in voting state - no new decisions can be claimed"
                ),
            });
        }

        if period_index > current_period + FUTURE_PERIODS - 1 {
            return Err(Error::DecisionNotAvailable {
                decision_id,
                reason: format!(
                    "Decision period {} exceeds maximum allowed period {} (current + 20)",
                    period_index,
                    current_period + FUTURE_PERIODS - 1
                ),
            });
        }

        if decision_id.is_standard() {
            let (start_period, end_period) =
                self.get_active_window(current_ts, current_height, genesis_ts)?;
            if period_index < start_period || period_index > end_period {
                return Err(Error::DecisionNotAvailable {
                    decision_id,
                    reason: format!(
                        "Period {period_index} is not in active \
                         window for new decisions \
                         ({start_period}-{end_period})"
                    ),
                });
            }

            let total_decisions = self.total_for(
                rotxn,
                period_index,
                current_ts,
                current_height,
                genesis_ts,
            )?;
            if decision_index as u64 >= total_decisions {
                return Err(Error::DecisionNotAvailable {
                    decision_id,
                    reason: format!(
                        "Standard decision index \
                         {decision_index} exceeds available \
                         decisions {total_decisions} for period \
                         {period_index}"
                    ),
                });
            }
        }

        let claimed_decisions =
            self.get_claimed_decision_ids_for_period(rotxn, period_index)?;

        if claimed_decisions.contains(&decision_id) {
            return Err(Error::DecisionAlreadyClaimed { decision_id });
        }

        Ok(())
    }

    pub fn claim_decision(
        &self,
        rwtxn: &mut RwTxn<'_>,
        decision_id: DecisionId,
        decision: Decision,
        claiming_txid: Txid,
        current_height: Option<u32>,
    ) -> Result<(), Error> {
        let period_index = decision_id.period_index();

        let mut period_decisions = self
            .period_decisions
            .try_get(rwtxn, &period_index)?
            .unwrap_or_default();

        let new_entry = DecisionEntry {
            decision_id,
            decision: Some(decision),
            claiming_txid,
        };
        period_decisions.insert(new_entry);
        self.period_decisions
            .put(rwtxn, &period_index, &period_decisions)?;

        let block_height = current_height.unwrap_or(0);
        let decision_history = DecisionStateHistory::new(block_height);

        self.decision_state_histories.put(
            rwtxn,
            &decision_id,
            &decision_history,
        )?;

        Ok(())
    }

    pub fn get_decision_entry(
        &self,
        rotxn: &sneed::RoTxn,
        decision_id: DecisionId,
    ) -> Result<Option<DecisionEntry>, Error> {
        let period = decision_id.period_index();
        if let Some(period_decisions) =
            self.period_decisions.try_get(rotxn, &period)?
        {
            let search_entry = DecisionEntry {
                decision_id,
                decision: None,
                claiming_txid: Txid([0u8; 32]),
            };

            if let Some(found) = period_decisions.get(&search_entry) {
                return Ok(Some(found.clone()));
            }

            for entry in period_decisions.range(search_entry..) {
                if entry.decision_id == decision_id {
                    return Ok(Some(entry.clone()));
                }
                if entry.decision_id > decision_id {
                    break;
                }
            }
        }
        Ok(None)
    }

    pub fn get_available_decisions_in_period(
        &self,
        rotxn: &sneed::RoTxn,
        period_index: u32,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<Vec<DecisionId>, Error> {
        let total_decisions = self.total_for(
            rotxn,
            period_index,
            current_ts,
            current_height,
            genesis_ts,
        )?;

        let max_decision_index = total_decisions;

        let mut available_decisions =
            Vec::with_capacity(max_decision_index as usize);

        let claimed_decisions =
            self.get_claimed_decision_ids_for_period(rotxn, period_index)?;

        for decision_index in 0..max_decision_index {
            let decision_id =
                DecisionId::new(true, period_index, decision_index as u32)?;

            if !claimed_decisions.contains(&decision_id) {
                available_decisions.push(decision_id);
            }
        }

        Ok(available_decisions)
    }

    pub fn get_claimed_decisions_in_period(
        &self,
        rotxn: &sneed::RoTxn,
        period_index: u32,
    ) -> Result<Vec<DecisionEntry>, Error> {
        let mut claimed_decisions = Vec::new();

        if let Some(period_decisions) =
            self.period_decisions.try_get(rotxn, &period_index)?
        {
            for entry in &period_decisions {
                if entry.decision.is_some() {
                    claimed_decisions.push(entry.clone());
                }
            }
        }

        Ok(claimed_decisions)
    }

    pub fn claimed_count_in_period(
        &self,
        rotxn: &sneed::RoTxn,
        period_index: u32,
    ) -> Result<u64, Error> {
        let claimed_decisions =
            self.get_claimed_decision_ids_for_period(rotxn, period_index)?;
        Ok(claimed_decisions.len() as u64)
    }

    pub fn get_standard_claimed_count_in_period(
        &self,
        rotxn: &sneed::RoTxn,
        period_index: u32,
    ) -> Result<u64, Error> {
        let claimed_ids =
            self.get_claimed_decision_ids_for_period(rotxn, period_index)?;
        Ok(claimed_ids.iter().filter(|id| id.is_standard()).count() as u64)
    }

    pub fn get_listing_fee_info(
        &self,
        rotxn: &sneed::RoTxn,
        period_index: u32,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<(u64, u64, u64), Error> {
        let current_period =
            self.get_current_period(current_ts, current_height, genesis_ts)?;
        let available =
            self.calculate_available_decisions(period_index, current_period);
        let claimed_standard =
            self.get_standard_claimed_count_in_period(rotxn, period_index)?;
        let next_fee = calculate_listing_fee(claimed_standard, available)
            .unwrap_or(u64::MAX);
        Ok((next_fee, claimed_standard, available))
    }

    pub fn get_all_claimed_decisions(
        &self,
        rotxn: &sneed::RoTxn,
    ) -> Result<Vec<DecisionEntry>, Error> {
        let mut all_claimed_decisions = Vec::new();

        let mut iter = self.period_decisions.iter(rotxn)?;
        while let Some((_period, period_decisions)) = iter.next()? {
            for entry in &period_decisions {
                if entry.decision.is_some() {
                    all_claimed_decisions.push(entry.clone());
                }
            }
        }

        Ok(all_claimed_decisions)
    }

    /// Returns periods that have decisions currently in voting state.
    /// Each tuple is (claim_period, voting_decision_count, total_decisions).
    /// claim_period = period_index = the period in which the decisions were claimed.
    pub fn get_voting_periods(
        &self,
        rotxn: &sneed::RoTxn,
        _current_ts: u64,
        _current_height: Option<u32>,
        _genesis_ts: u64,
    ) -> Result<Vec<(u32, u64, u64)>, Error> {
        use std::collections::HashMap;

        let mut period_voting_counts: HashMap<u32, u64> = HashMap::new();

        let mut iter = self.decision_state_histories.iter(rotxn)?;
        while let Some((decision_id, history)) = iter.next()? {
            if history.current_state() == DecisionState::Voting {
                let claim_period = decision_id.period_index();
                *period_voting_counts.entry(claim_period).or_insert(0) += 1;
            }
        }

        let mut voting_periods: Vec<(u32, u64, u64)> = period_voting_counts
            .into_iter()
            .map(|(period, count)| {
                let total_decisions = 500u64;
                (period, count, total_decisions)
            })
            .collect();

        voting_periods.sort_by_key(|(period, _, _)| *period);
        Ok(voting_periods)
    }

    pub fn get_period_summary(
        &self,
        rotxn: &sneed::RoTxn,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<super::super::type_aliases::PeriodSummary, Error> {
        let active_periods = self.get_active_periods(
            rotxn,
            current_ts,
            current_height,
            genesis_ts,
        )?;
        let voting_periods_full = self.get_voting_periods(
            rotxn,
            current_ts,
            current_height,
            genesis_ts,
        )?;
        let voting_periods = voting_periods_full
            .into_iter()
            .map(|(period, claimed, _total)| (period, claimed))
            .collect();

        Ok((active_periods, voting_periods))
    }

    pub fn revert_decision_claim(
        &self,
        rwtxn: &mut RwTxn<'_>,
        decision_id: DecisionId,
    ) -> Result<(), Error> {
        let period_index = decision_id.period_index();

        let mut period_decisions = self
            .period_decisions
            .try_get(rwtxn, &period_index)?
            .unwrap_or_default();

        let target_decision = period_decisions
            .iter()
            .find(|entry| entry.decision_id == decision_id)
            .cloned();

        if let Some(decision_to_remove) = target_decision {
            period_decisions.remove(&decision_to_remove);

            if period_decisions.is_empty() {
                self.period_decisions.delete(rwtxn, &period_index)?;
            } else {
                self.period_decisions.put(
                    rwtxn,
                    &period_index,
                    &period_decisions,
                )?;
            }
        } else {
            tracing::debug!(
                "Attempted to revert decision {:?} that wasn't found",
                decision_id
            );
        }

        Ok(())
    }

    pub fn get_decision_state_history(
        &self,
        rotxn: &RoTxn,
        decision_id: DecisionId,
    ) -> Result<Option<DecisionStateHistory>, Error> {
        Ok(self.decision_state_histories.try_get(rotxn, &decision_id)?)
    }

    pub fn get_decision_current_state(
        &self,
        rotxn: &RoTxn,
        decision_id: DecisionId,
    ) -> Result<DecisionState, Error> {
        Ok(self
            .get_decision_state_history(rotxn, decision_id)?
            .map(|h| h.current_state())
            .unwrap_or(DecisionState::Created))
    }

    pub fn transition_decision_to_voting(
        &self,
        rwtxn: &mut RwTxn,
        decision_id: DecisionId,
        block_height: u32,
    ) -> Result<(), Error> {
        let mut history = self
            .get_decision_state_history(rwtxn, decision_id)?
            .ok_or(Error::InvalidDecisionId {
                reason: format!(
                    "Decision {decision_id:?} has no state history"
                ),
            })?;

        history.transition_to_voting(block_height)?;
        self.decision_state_histories
            .put(rwtxn, &decision_id, &history)?;
        Ok(())
    }

    pub fn transition_decision_to_resolved(
        &self,
        rwtxn: &mut RwTxn,
        decision_id: DecisionId,
        block_height: u32,
        consensus_outcome: f64,
    ) -> Result<(), Error> {
        let mut history = self
            .get_decision_state_history(rwtxn, decision_id)?
            .ok_or(Error::InvalidDecisionId {
                reason: format!(
                    "Decision {decision_id:?} has no state history"
                ),
            })?;

        history.transition_to_resolved(consensus_outcome, block_height)?;
        self.decision_state_histories
            .put(rwtxn, &decision_id, &history)?;
        Ok(())
    }

    pub fn rollback_decision_states_to_height(
        &self,
        rwtxn: &mut RwTxn,
        height: u32,
    ) -> Result<(), Error> {
        let mut decisions_to_update = Vec::new();
        let mut decisions_to_unclaim: Vec<DecisionId> = Vec::new();

        {
            let mut iter = self.decision_state_histories.iter(rwtxn)?;
            while let Some((decision_id, mut history)) = iter.next()? {
                let was_claimed =
                    history.current_state() != DecisionState::Created;
                history.rollback_to_height(height);
                let is_now_created =
                    history.current_state() == DecisionState::Created;

                if was_claimed && is_now_created {
                    decisions_to_unclaim.push(decision_id);
                }
                decisions_to_update.push((decision_id, history));
            }
        }

        for (decision_id, history) in decisions_to_update {
            self.decision_state_histories
                .put(rwtxn, &decision_id, &history)?;
        }

        for decision_id in decisions_to_unclaim {
            let period_index = decision_id.period_index();

            if let Some(mut period_decisions) =
                self.period_decisions.try_get(rwtxn, &period_index)?
            {
                period_decisions.retain(|s| s.decision_id != decision_id);
                if period_decisions.is_empty() {
                    self.period_decisions.delete(rwtxn, &period_index)?;
                } else {
                    self.period_decisions.put(
                        rwtxn,
                        &period_index,
                        &period_decisions,
                    )?;
                }
            }
        }

        Ok(())
    }

    pub fn get_available_decisions(
        &self,
        _rotxn: &RoTxn,
        period: u32,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<u64, Error> {
        let current_period =
            self.get_current_period(current_ts, current_height, genesis_ts)?;
        Ok(self.calculate_available_decisions(period, current_period))
    }
}

impl DecisionValidationInterface for Dbs {
    fn validate_decision_claim(
        &self,
        rotxn: &RoTxn,
        decision_id: DecisionId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<(), Error> {
        self.validate_decision_claim(
            rotxn,
            decision_id,
            decision,
            current_ts,
            current_height,
            genesis_ts,
        )
    }

    fn try_get_height(&self, _rotxn: &RoTxn) -> Result<Option<u32>, Error> {
        Ok(None)
    }

    fn try_get_genesis_timestamp(
        &self,
        _rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error> {
        Ok(None)
    }

    fn try_get_mainchain_timestamp(
        &self,
        _rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error> {
        Ok(None)
    }

    fn get_standard_claimed_count_in_period(
        &self,
        rotxn: &RoTxn,
        period_index: u32,
    ) -> Result<u64, Error> {
        self.get_standard_claimed_count_in_period(rotxn, period_index)
    }

    fn get_available_decisions(
        &self,
        rotxn: &RoTxn,
        period: u32,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<u64, Error> {
        self.get_available_decisions(
            rotxn,
            period,
            current_ts,
            current_height,
            genesis_ts,
        )
    }
}
