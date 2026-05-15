use crate::state::Error;
use crate::types::Txid;
use crate::validation::DecisionValidationInterface;
use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use sneed::{DatabaseUnique, Env, RoTxn, RwTxn};
use std::collections::BTreeSet;

use super::pricing::{
    self, GENESIS_P_PERIOD_SATS, PeriodPricing, apply_reprice_with_floor,
};
use super::types::{
    Decision, DecisionConfig, DecisionEntry, DecisionId, DecisionState,
    DecisionStateHistory, FUTURE_PERIODS, INITIAL_DECISIONS_PER_PERIOD,
    period_to_string,
};

const DECISIONS_DECLINING_RATE: u64 = 25;

pub type PeriodPricingUndoEntry = (u32, Option<PeriodPricing>);

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
    period_pricing:
        DatabaseUnique<SerdeBincode<u32>, SerdeBincode<PeriodPricing>>,
    period_pricing_undo: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<Vec<PeriodPricingUndoEntry>>,
    >,
    config: DecisionConfig,
}

impl Dbs {
    pub const NUM_DBS: u32 = 4;

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
            period_pricing: DatabaseUnique::create(
                env,
                rwtxn,
                "period_pricing",
            )?,
            period_pricing_undo: DatabaseUnique::create(
                env,
                rwtxn,
                "period_pricing_undo",
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
        let target = current_period + FUTURE_PERIODS - 1;
        self.mint_periods_up_to(rwtxn, target)?;
        for period in current_period..=target {
            if self.period_pricing.try_get(rwtxn, &period)?.is_some() {
                continue;
            }
            let offset = period - current_period;
            let mints = (FUTURE_PERIODS as u64).saturating_sub(offset as u64);
            let pricing = PeriodPricing {
                p_period: GENESIS_P_PERIOD_SATS,
                p_floor: GENESIS_P_PERIOD_SATS / 16,
                mints,
                claimed: 0,
                last_reprice_block: block_height,
                window_open_claimed: 0,
                window_open_mints: mints,
            };
            self.period_pricing.put(rwtxn, &period, &pricing)?;
        }
        Ok(())
    }

    /// Snapshot the entire `period_pricing` table into the undo log for
    /// `block_height`. Must be called BEFORE any modifications at this height.
    pub fn snapshot_period_pricing(
        &self,
        rwtxn: &mut RwTxn<'_>,
        block_height: u32,
    ) -> Result<(), Error> {
        let mut snapshots: Vec<PeriodPricingUndoEntry> = Vec::new();
        {
            let mut iter = self.period_pricing.iter(rwtxn)?;
            while let Some((period, p)) = iter.next()? {
                snapshots.push((period, Some(p)));
            }
        }
        self.period_pricing_undo
            .put(rwtxn, &block_height, &snapshots)?;
        Ok(())
    }

    /// Restore `period_pricing` from the undo log for `block_height` and
    /// delete the undo entry. Used during disconnect.
    pub fn restore_period_pricing_undo(
        &self,
        rwtxn: &mut RwTxn<'_>,
        block_height: u32,
    ) -> Result<(), Error> {
        let snapshots =
            match self.period_pricing_undo.try_get(rwtxn, &block_height)? {
                Some(v) => v,
                None => return Ok(()),
            };
        let snap_keys: std::collections::HashSet<u32> =
            snapshots.iter().map(|(p, _)| *p).collect();

        let mut current_keys: Vec<u32> = Vec::new();
        {
            let mut iter = self.period_pricing.iter(rwtxn)?;
            while let Some((period, _)) = iter.next()? {
                current_keys.push(period);
            }
        }
        for k in &current_keys {
            if !snap_keys.contains(k) {
                self.period_pricing.delete(rwtxn, k)?;
            }
        }
        for (period, prior) in snapshots {
            if let Some(pp) = prior {
                self.period_pricing.put(rwtxn, &period, &pp)?;
            }
        }
        self.period_pricing_undo.delete(rwtxn, &block_height)?;
        Ok(())
    }

    /// Apply per-block pricing updates: mint new periods on transition,
    /// increment `mints` for surviving periods on transition (with forced
    /// reprice), or fire scheduled reprice when the interval has elapsed.
    /// Caller must call `snapshot_period_pricing` first.
    pub fn process_block_pricing(
        &self,
        rwtxn: &mut RwTxn<'_>,
        block_height: u32,
        ts_secs: u64,
        genesis_ts: u64,
    ) -> Result<(), Error> {
        let new_current =
            self.get_current_period(ts_secs, Some(block_height), genesis_ts)?;
        let target = new_current + FUTURE_PERIODS - 1;
        let prev_highest = self.get_highest_minted_period(rwtxn)?.unwrap_or(0);
        // After genesis the invariant is `prev_highest == prev_current + FUTURE_PERIODS - 1`,
        // so `prev_current = prev_highest + 1 - FUTURE_PERIODS`. The
        // `saturating_sub` covers the genesis-adjacent case where
        // `prev_highest == 0` (no prior mint), in which case we treat the
        // chain as having just transitioned into `new_current`.
        let prev_current = (prev_highest + 1).saturating_sub(FUTURE_PERIODS);
        debug_assert!(
            prev_current <= new_current,
            "current_period must monotonically advance: prev={prev_current} new={new_current}"
        );

        self.mint_periods_up_to(rwtxn, target)?;

        let did_transition = new_current > prev_current;

        if did_transition {
            for new_period in (prev_highest + 1)..=target {
                let p_seed = self.compute_p_average(rwtxn, new_current)?;
                let pricing = PeriodPricing::new_seeded(p_seed, block_height);
                self.period_pricing.put(rwtxn, &new_period, &pricing)?;
            }
            let surviving_end = prev_highest.min(target);
            if new_current <= surviving_end {
                for period in new_current..=surviving_end {
                    let mut pricing =
                        match self.period_pricing.try_get(rwtxn, &period)? {
                            Some(p) => p,
                            None => continue,
                        };
                    let transitions = new_current - prev_current;
                    pricing.mints =
                        pricing.mints.saturating_add(transitions as u64);
                    apply_reprice_with_floor(&mut pricing, block_height);
                    self.period_pricing.put(rwtxn, &period, &pricing)?;
                }
            }
        } else {
            let interval = pricing::reprice_interval(&self.config);
            let any_due = self.any_period_due_for_reprice(
                rwtxn,
                block_height,
                interval,
                new_current,
            )?;
            if any_due {
                let active_end = new_current + FUTURE_PERIODS - 1;
                for period in new_current..=active_end {
                    let mut pricing =
                        match self.period_pricing.try_get(rwtxn, &period)? {
                            Some(p) => p,
                            None => continue,
                        };
                    apply_reprice_with_floor(&mut pricing, block_height);
                    self.period_pricing.put(rwtxn, &period, &pricing)?;
                }
            }
        }

        Ok(())
    }

    fn any_period_due_for_reprice(
        &self,
        rotxn: &RoTxn,
        block_height: u32,
        interval: u32,
        current_period: u32,
    ) -> Result<bool, Error> {
        let active_end = current_period + FUTURE_PERIODS - 1;
        for period in current_period..=active_end {
            if let Some(p) = self.period_pricing.try_get(rotxn, &period)?
                && block_height.saturating_sub(p.last_reprice_block) >= interval
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn compute_p_average(
        &self,
        rotxn: &RoTxn,
        current_period: u32,
    ) -> Result<u64, Error> {
        let mut sum: u128 = 0;
        let mut count: u128 = 0;
        for offset in 1..FUTURE_PERIODS {
            let period = current_period + offset;
            if let Some(p) = self.period_pricing.try_get(rotxn, &period)? {
                sum += p.p_period as u128;
                count += 1;
            }
        }
        match sum.checked_div(count) {
            Some(avg) => Ok(avg as u64),
            None => Ok(GENESIS_P_PERIOD_SATS),
        }
    }

    pub fn get_period_pricing(
        &self,
        rotxn: &RoTxn,
        period: u32,
    ) -> Result<Option<PeriodPricing>, Error> {
        Ok(self.period_pricing.try_get(rotxn, &period)?)
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

        let empty_period: BTreeSet<DecisionEntry> = BTreeSet::new();
        for period in (highest_existing + 1)..=target_period {
            self.period_decisions.put(rwtxn, &period, &empty_period)?;
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

    /// Delete all `period_decisions` entries above `max_period`. The
    /// `period_pricing` table is rolled back separately via the undo log
    /// (`restore_period_pricing_undo`), so it is intentionally not touched
    /// here.
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
        for period in &to_delete {
            self.period_decisions.delete(rwtxn, period)?;
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

    pub fn is_period_settled(
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

    pub fn is_decision_settled(
        &self,
        decision_id: DecisionId,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> bool {
        let period = decision_id.period_index();
        self.is_period_settled(period, current_ts, current_height, genesis_ts)
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

    pub fn get_settled_decisions(
        &self,
        rotxn: &sneed::RoTxn,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<Vec<DecisionEntry>, Error> {
        let mut settled_entries = Vec::new();

        let mut iter = self.period_decisions.iter(rotxn)?;
        while let Some((period, entries)) = iter.next()? {
            if self.is_period_settled(
                period,
                current_ts,
                current_height,
                genesis_ts,
            ) {
                settled_entries.extend(entries.iter().cloned());
            }
        }

        Ok(settled_entries)
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

        if self.is_decision_settled(
            decision_id,
            current_ts,
            current_height,
            genesis_ts,
        ) {
            return Err(Error::DecisionNotAvailable {
                decision_id,
                reason: format!("Decision period {period_index} is settled"),
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

            let pricing = self
                .period_pricing
                .try_get(rotxn, &period_index)?
                .ok_or(Error::DecisionNotAvailable {
                    decision_id,
                    reason: format!(
                        "no pricing record for period {period_index}"
                    ),
                })?;
            if !pricing::slot_unlocked(decision_index, pricing.mints) {
                return Err(Error::DecisionNotAvailable {
                    decision_id,
                    reason: format!(
                        "decision_index {decision_index} not yet \
                         unlocked in period {period_index} \
                         (mints={})",
                        pricing.mints
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

        if decision_id.is_standard() {
            let mut pricing = self
                .period_pricing
                .try_get(rwtxn, &period_index)?
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: format!(
                        "no pricing record for period {period_index}"
                    ),
                })?;
            pricing.claimed = pricing.claimed.saturating_add(1);
            self.period_pricing.put(rwtxn, &period_index, &pricing)?;
        }

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
        if total_decisions == 0 {
            return Ok(Vec::new());
        }

        let Some(pricing) =
            self.period_pricing.try_get(rotxn, &period_index)?
        else {
            return Ok(Vec::new());
        };

        let claimed_decisions =
            self.get_claimed_decision_ids_for_period(rotxn, period_index)?;

        let unlocked_per_tier =
            pricing.mints * pricing::SLOTS_PER_TIER_PER_MINT;
        let mut available_decisions = Vec::with_capacity(
            (unlocked_per_tier as usize) * (pricing::TIER_COUNT),
        );

        for tier in 0..pricing::TIER_COUNT as u64 {
            for pos in 0..unlocked_per_tier {
                let idx = (tier * pricing::SLOTS_PER_TIER + pos) as u32;
                let decision_id = DecisionId::new(true, period_index, idx)?;
                if !claimed_decisions.contains(&decision_id) {
                    available_decisions.push(decision_id);
                }
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
    ) -> Result<Option<PeriodPricing>, Error> {
        Ok(self.period_pricing.try_get(rotxn, &period_index)?)
    }

    /// Fee for claiming a single decision_id at its deterministic tier
    /// (`tier = decision_index / 100`). Validates that the slot is unlocked
    /// and the index is in range.
    pub fn fee_for_decision_id(
        &self,
        rotxn: &sneed::RoTxn,
        decision_id: DecisionId,
    ) -> Result<u64, Error> {
        let period_index = decision_id.period_index();
        let pricing = self
            .period_pricing
            .try_get(rotxn, &period_index)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!("no pricing record for period {period_index}"),
            })?;
        pricing::fee_for_index(
            pricing.p_period,
            pricing.mints,
            decision_id.decision_index(),
        )
        .map_err(|e| Error::InvalidTransaction {
            reason: e.to_string(),
        })
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

    fn fee_for_decision_id(
        &self,
        rotxn: &RoTxn,
        decision_id: DecisionId,
    ) -> Result<u64, Error> {
        self.fee_for_decision_id(rotxn, decision_id)
    }
}
