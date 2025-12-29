use crate::state::Error;
use crate::types::{BITCOIN_GENESIS_TIMESTAMP, SECONDS_PER_QUARTER};
use crate::validation::SlotValidationInterface;
use borsh::BorshSerialize;
use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use serde::{Deserialize, Serialize};
use sneed::{DatabaseUnique, Env, RoTxn, RwTxn};
use std::collections::BTreeSet;

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
    BorshSerialize,
)]
pub struct SlotId([u8; 3]);

const MAX_PERIOD_INDEX: u32 = (1 << 10) - 1;
const MAX_SLOT_INDEX: u32 = (1 << 14) - 1;
const STANDARD_SLOT_MAX: u32 = 499;
const NONSTANDARD_SLOT_MIN: u32 = 500;
const PERIOD_SHIFT: u32 = 14;
const SLOT_MASK: u32 = MAX_SLOT_INDEX;

/// Validates that period and index are within valid bounds.
/// Shared by SlotId::new() and SlotId::from_bytes().
fn validate_slot_bounds(period: u32, index: u32) -> Result<(), Error> {
    if period > MAX_PERIOD_INDEX {
        return Err(Error::InvalidSlotId {
            reason: format!(
                "Period {period} exceeds maximum {MAX_PERIOD_INDEX}"
            ),
        });
    }
    if index > MAX_SLOT_INDEX {
        return Err(Error::InvalidSlotId {
            reason: format!(
                "Slot index {index} exceeds maximum {MAX_SLOT_INDEX}"
            ),
        });
    }
    Ok(())
}

impl SlotId {
    #[inline(always)]
    const fn as_u32(self) -> u32 {
        ((self.0[0] as u32) << 16)
            | ((self.0[1] as u32) << 8)
            | (self.0[2] as u32)
    }

    pub fn new(period: u32, index: u32) -> Result<Self, Error> {
        validate_slot_bounds(period, index)?;
        let combined = (period << PERIOD_SHIFT) | index;
        let bytes = [
            (combined >> 16) as u8,
            (combined >> 8) as u8,
            combined as u8,
        ];
        Ok(SlotId(bytes))
    }

    #[inline(always)]
    pub const fn period_index(self) -> u32 {
        self.as_u32() >> PERIOD_SHIFT
    }

    #[inline(always)]
    pub const fn slot_index(self) -> u32 {
        self.as_u32() & SLOT_MASK
    }

    pub fn as_bytes(self) -> [u8; 3] {
        self.0
    }

    pub fn from_bytes(bytes: [u8; 3]) -> Result<Self, Error> {
        let slot = SlotId(bytes);
        let combined = slot.as_u32();
        validate_slot_bounds(combined >> PERIOD_SHIFT, combined & SLOT_MASK)?;
        Ok(slot)
    }

    pub fn from_hex(slot_id_hex: &str) -> Result<Self, Error> {
        if slot_id_hex.len() != 6 {
            return Err(Error::InvalidSlotId {
                reason: "Slot ID hex must be exactly 6 characters (3 bytes)"
                    .to_string(),
            });
        }

        let mut slot_id_bytes = [0u8; 3];
        for (i, chunk) in slot_id_hex.as_bytes().chunks_exact(2).enumerate() {
            let hex_str = std::str::from_utf8(chunk).map_err(|_| {
                Error::InvalidSlotId {
                    reason: "Invalid slot ID hex format".to_string(),
                }
            })?;

            slot_id_bytes[i] =
                u8::from_str_radix(hex_str, 16).map_err(|_| {
                    Error::InvalidSlotId {
                        reason: "Invalid slot ID hex format".to_string(),
                    }
                })?;
        }

        Self::from_bytes(slot_id_bytes)
    }

    pub fn to_hex(self) -> String {
        hex::encode(self.0)
    }

    #[inline(always)]
    pub const fn voting_period(self) -> u32 {
        self.period_index() + 1
    }
}

#[derive(
    Clone, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd,
)]
pub struct Decision {
    pub id: [u8; 32],
    pub market_maker_pubkey_hash: [u8; 20],
    pub slot_id_bytes: [u8; 3],
    pub is_standard: bool,
    pub is_scaled: bool,
    pub question: String,
    pub min: Option<u16>,
    pub max: Option<u16>,
}

impl Decision {
    pub fn new(
        market_maker_pubkey_hash: [u8; 20],
        slot_id_bytes: [u8; 3],
        is_standard: bool,
        is_scaled: bool,
        question: String,
        min: Option<u16>,
        max: Option<u16>,
    ) -> Result<Self, Error> {
        if question.len() > 1000 {
            return Err(Error::InvalidSlotId {
                reason: "Question must be 1000 bytes or less".to_string(),
            });
        }

        match (min, max, is_scaled) {
            (Some(min_val), Some(max_val), true) => {
                if min_val >= max_val {
                    return Err(Error::InvalidRange);
                }
            }
            (None, None, false) => {}
            _ => return Err(Error::InconsistentDecisionType),
        }

        let mut decision = Decision {
            id: [0; 32],
            market_maker_pubkey_hash,
            slot_id_bytes,
            is_standard,
            is_scaled,
            question,
            min,
            max,
        };

        decision.id = decision.compute_id_hash();

        Ok(decision)
    }

    fn compute_id_hash(&self) -> [u8; 32] {
        use crate::types::hashes;

        let hash_data = (
            &self.market_maker_pubkey_hash,
            &self.slot_id_bytes,
            self.is_standard,
            self.is_scaled,
            &self.question,
            self.min,
            self.max,
        );

        hashes::hash(&hash_data)
    }
}

#[derive(
    Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd,
)]
pub enum SlotState {
    Created,
    Claimed,
    Voting,
    Resolved,
    Invalid,
}

impl SlotState {
    pub fn can_transition_to(&self, new_state: SlotState) -> bool {
        use SlotState::*;
        matches!(
            (self, new_state),
            (Created, Claimed)
                | (Claimed, Voting)
                | (Voting, Resolved)
                | (_, Invalid)
        )
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, SlotState::Resolved | SlotState::Invalid)
    }

    pub fn allows_voting(&self) -> bool {
        matches!(self, SlotState::Voting)
    }

    pub fn has_consensus(&self) -> bool {
        matches!(self, SlotState::Resolved)
    }
}

#[derive(
    Clone, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd,
)]
pub struct SlotStateHistory {
    pub voting_period: Option<u32>,
    pub state_history: Vec<(SlotState, u32)>,
}

impl SlotStateHistory {
    pub fn new_created(_slot_id: SlotId, height: u32) -> Self {
        Self {
            voting_period: None,
            state_history: vec![(SlotState::Created, height)],
        }
    }

    pub fn new(_slot_id: SlotId, initial_height: u64, _timestamp: u64) -> Self {
        let height_u32 = initial_height as u32;
        Self::new_created(_slot_id, height_u32)
    }

    pub fn current_state(&self) -> SlotState {
        self.state_history
            .last()
            .map(|(s, _)| *s)
            .unwrap_or(SlotState::Created)
    }

    pub fn transition_to_claimed(&mut self, height: u32) -> Result<(), Error> {
        let current = self.current_state();
        if !current.can_transition_to(SlotState::Claimed) {
            return Err(Error::InvalidSlotState {
                reason: format!(
                    "Cannot transition to Claimed from {current:?}"
                ),
            });
        }
        self.state_history.push((SlotState::Claimed, height));
        Ok(())
    }

    pub fn transition_to_voting(
        &mut self,
        voting_period: u32,
        height: u32,
    ) -> Result<(), Error> {
        let current = self.current_state();
        if !current.can_transition_to(SlotState::Voting) {
            return Err(Error::InvalidSlotState {
                reason: format!("Cannot transition to Voting from {current:?}"),
            });
        }
        self.voting_period = Some(voting_period);
        self.state_history.push((SlotState::Voting, height));
        Ok(())
    }

    pub fn transition_to_resolved(
        &mut self,
        consensus_outcome: f64,
        height: u32,
    ) -> Result<(), Error> {
        let current = self.current_state();
        if !current.can_transition_to(SlotState::Resolved) {
            return Err(Error::InvalidSlotState {
                reason: format!(
                    "Cannot transition to Resolved from {current:?}"
                ),
            });
        }

        if !(0.0..=1.0).contains(&consensus_outcome) {
            return Err(Error::InvalidSlotState {
                reason: format!(
                    "Consensus outcome {consensus_outcome} outside valid range [0.0, 1.0]"
                ),
            });
        }

        self.state_history.push((SlotState::Resolved, height));
        Ok(())
    }

    pub fn get_voting_period(&self) -> Option<u32> {
        self.voting_period
    }

    pub fn can_accept_votes(&self) -> bool {
        self.current_state() == SlotState::Voting
    }

    pub fn has_consensus(&self) -> bool {
        self.current_state().has_consensus()
    }

    pub fn has_reached_state(&self, state: SlotState) -> bool {
        self.state_history.iter().any(|(s, _)| *s == state)
    }

    pub fn state_at_height(&self, height: u64) -> SlotState {
        let height_u32 = height as u32;
        self.state_history
            .iter()
            .rev()
            .find(|(_, h)| *h <= height_u32)
            .map(|(s, _)| *s)
            .unwrap_or(SlotState::Created)
    }

    pub fn rollback_to_height(&mut self, height: u64) {
        let height_u32 = height as u32;
        self.state_history.retain(|(_, h)| *h <= height_u32);
    }

    pub fn get_state_height(&self, state: SlotState) -> Option<u64> {
        self.state_history
            .iter()
            .find(|(s, _)| *s == state)
            .map(|(_, h)| *h as u64)
    }

    pub fn transition_to_claimed_with_timestamp(
        &mut self,
        block_height: u64,
        _timestamp: u64,
    ) -> Result<(), Error> {
        self.transition_to_claimed(block_height as u32)
    }

    pub fn transition_to_voting_with_timestamp(
        &mut self,
        block_height: u64,
        _timestamp: u64,
        voting_period: u32,
    ) -> Result<(), Error> {
        self.transition_to_voting(voting_period, block_height as u32)
    }
}

#[derive(
    Clone, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd,
)]
pub struct Slot {
    pub slot_id: SlotId,
    pub decision: Option<Decision>,
}

const FUTURE_PERIODS: u32 = 20;
const SLOTS_DECLINING_RATE: u64 = 25;
const INITIAL_SLOTS_PER_PERIOD: u64 = 500;

#[derive(Clone, Debug)]
pub struct SlotConfig {
    pub testing_mode: bool,
    pub testing_blocks_per_period: u32,
}

impl Default for SlotConfig {
    fn default() -> Self {
        Self {
            testing_mode: true,
            testing_blocks_per_period: 10,
        }
    }
}

impl SlotConfig {
    pub fn production() -> Self {
        Self {
            testing_mode: false,
            testing_blocks_per_period: 1,
        }
    }

    pub fn testing(blocks_per_period: u32) -> Self {
        if blocks_per_period == 0 {
            panic!("blocks_per_period must be > 0");
        }
        Self {
            testing_mode: true,
            testing_blocks_per_period: blocks_per_period,
        }
    }
}

/// Convert timestamp to period index (single source of truth).
/// Used by RPC and other layers for consistent period calculation.
#[inline]
pub fn timestamp_to_period(timestamp: u64) -> u32 {
    if timestamp < BITCOIN_GENESIS_TIMESTAMP {
        return 0;
    }
    let elapsed_seconds = timestamp - BITCOIN_GENESIS_TIMESTAMP;
    (elapsed_seconds / SECONDS_PER_QUARTER) as u32
}

/// Convert period index to human-readable name (single source of truth).
/// Returns "Genesis" for period 0, otherwise "Q{quarter} Y{year}".
#[inline]
pub fn period_to_name(period: u32) -> String {
    if period == 0 {
        return "Genesis".to_string();
    }
    let year = 2009 + (period - 1) / 4;
    let quarter = ((period - 1) % 4) + 1;
    format!("Q{quarter} Y{year}")
}

pub fn quarter_to_string(quarter_idx: u32, config: &SlotConfig) -> String {
    if config.testing_mode {
        format!("Testing Period {quarter_idx}")
    } else {
        let year = quarter_idx / 4;
        let quarter = quarter_idx % 4;
        let quarter_name = match quarter {
            0 => "Q1",
            1 => "Q2",
            2 => "Q3",
            3 => "Q4",
            _ => unreachable!(),
        };
        format!("{quarter_name} {year}")
    }
}

fn get_current_period(
    timestamp: u64,
    block_height: Option<u32>,
    config: &SlotConfig,
) -> Result<u32, Error> {
    if config.testing_mode {
        let height = block_height.unwrap_or(0);
        // Block heights are 0-indexed and directly map to periods
        // Heights 0-9 = period 1, 10-19 = period 2, etc.
        if config.testing_blocks_per_period == 0 {
            Ok(0)
        } else {
            Ok((height / config.testing_blocks_per_period) + 1)
        }
    } else {
        Ok(timestamp_to_period(timestamp))
    }
}

#[derive(Clone)]
pub struct Dbs {
    period_slots:
        DatabaseUnique<SerdeBincode<u32>, SerdeBincode<BTreeSet<Slot>>>,
    slot_state_histories:
        DatabaseUnique<SerdeBincode<SlotId>, SerdeBincode<SlotStateHistory>>,
    config: SlotConfig,
}

impl Dbs {
    pub const NUM_DBS: u32 = 2;

    /// Derive claimed slot IDs from period_slots (a slot is claimed if decision.is_some())
    fn get_claimed_slot_ids_for_period(
        &self,
        rotxn: &RoTxn,
        period_index: u32,
    ) -> Result<BTreeSet<SlotId>, Error> {
        let period_slots = self
            .period_slots
            .try_get(rotxn, &period_index)?
            .unwrap_or_default();
        Ok(period_slots
            .iter()
            .filter(|slot| slot.decision.is_some())
            .map(|slot| slot.slot_id)
            .collect())
    }

    pub fn new(env: &Env, rwtxn: &mut RwTxn<'_>) -> Result<Self, Error> {
        Self::new_with_config(env, rwtxn, SlotConfig::default())
    }

    pub fn new_with_config(
        env: &Env,
        rwtxn: &mut RwTxn<'_>,
        config: SlotConfig,
    ) -> Result<Self, Error> {
        Ok(Self {
            period_slots: DatabaseUnique::create(env, rwtxn, "period_slots")?,
            slot_state_histories: DatabaseUnique::create(
                env,
                rwtxn,
                "slot_state_histories",
            )?,
            config,
        })
    }

    pub fn get_current_period(
        &self,
        ts_secs: u64,
        block_height: Option<u32>,
    ) -> Result<u32, Error> {
        get_current_period(ts_secs, block_height, &self.config)
    }

    fn get_active_window(
        &self,
        ts_secs: u64,
        block_height: Option<u32>,
    ) -> Result<(u32, u32), Error> {
        let current = self.get_current_period(ts_secs, block_height)?;
        Ok((current, current + FUTURE_PERIODS - 1))
    }

    #[inline]
    const fn calculate_available_slots(
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

        INITIAL_SLOTS_PER_PERIOD
            .saturating_sub((offset as u64) * SLOTS_DECLINING_RATE)
    }

    pub fn mint_genesis(
        &self,
        rwtxn: &mut RwTxn<'_>,
        ts_secs: u64,
        block_height: u32,
    ) -> Result<(), Error> {
        let current_period =
            self.get_current_period(ts_secs, Some(block_height))?;
        self.mint_periods_up_to(rwtxn, current_period + FUTURE_PERIODS - 1)?;
        Ok(())
    }

    pub fn mint_up_to(
        &self,
        rwtxn: &mut RwTxn<'_>,
        ts_secs: u64,
        block_height: u32,
    ) -> Result<(), Error> {
        let current_period =
            self.get_current_period(ts_secs, Some(block_height))?;
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
            let mut iter = self.period_slots.iter(rwtxn)?;
            while let Some((period, _)) = iter.next()? {
                if period > highest_existing {
                    highest_existing = period;
                }
            }
        }

        for period in (highest_existing + 1)..=target_period {
            if self.period_slots.try_get(rwtxn, &period)?.is_none() {
                let empty_period: BTreeSet<Slot> = BTreeSet::new();
                self.period_slots.put(rwtxn, &period, &empty_period)?;
            }
        }

        Ok(())
    }

    pub fn total_for(
        &self,
        _rotxn: &sneed::RoTxn,
        period: u32,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<u64, Error> {
        let current_period =
            self.get_current_period(current_ts, current_height)?;
        Ok(self.calculate_available_slots(period, current_period))
    }

    pub fn get_active_periods(
        &self,
        _rotxn: &sneed::RoTxn,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<Vec<(u32, u64)>, Error> {
        let current_period =
            self.get_current_period(current_ts, current_height)?;
        let mut periods = Vec::new();

        for period in current_period..=(current_period + FUTURE_PERIODS - 1) {
            let slots = self.calculate_available_slots(period, current_period);
            if slots > 0 {
                periods.push((period, slots));
            }
        }

        Ok(periods)
    }

    pub fn is_testing_mode(&self) -> bool {
        self.config.testing_mode
    }

    pub fn get_testing_blocks_per_period(&self) -> u32 {
        self.config.testing_blocks_per_period
    }

    pub fn block_height_to_testing_period(&self, block_height: u32) -> u32 {
        if self.config.testing_blocks_per_period == 0 {
            0
        } else {
            (block_height / self.config.testing_blocks_per_period) + 1
        }
    }

    pub fn get_config(&self) -> &SlotConfig {
        &self.config
    }

    pub fn quarter_to_string(&self, quarter_idx: u32) -> String {
        quarter_to_string(quarter_idx, &self.config)
    }

    pub fn is_period_ossified(
        &self,
        slot_period: u32,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> bool {
        let current_period = self
            .get_current_period(current_ts, current_height)
            .unwrap_or(0);
        current_period > slot_period.saturating_add(7)
    }

    pub fn is_slot_ossified(
        &self,
        slot_id: SlotId,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> bool {
        let period = slot_id.period_index();
        self.is_period_ossified(period, current_ts, current_height)
    }

    pub fn is_slot_in_voting(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
    ) -> Result<bool, Error> {
        Ok(self.get_slot_current_state(rotxn, slot_id)? == SlotState::Voting)
    }

    pub fn get_ossified_slots(
        &self,
        rotxn: &sneed::RoTxn,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<Vec<Slot>, Error> {
        let mut ossified_slots = Vec::new();

        let mut iter = self.period_slots.iter(rotxn)?;
        while let Some((period, slots)) = iter.next()? {
            if self.is_period_ossified(period, current_ts, current_height) {
                ossified_slots.extend(slots.iter().cloned());
            }
        }

        Ok(ossified_slots)
    }

    pub fn validate_slot_claim(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<(), Error> {
        let period_index = slot_id.period_index();
        let slot_index = slot_id.slot_index();
        let current_period =
            self.get_current_period(current_ts, current_height)?;

        if self.is_slot_ossified(slot_id, current_ts, current_height) {
            return Err(Error::SlotNotAvailable {
                slot_id,
                reason: format!("Slot period {period_index} is ossified"),
            });
        }

        if self.is_slot_in_voting(rotxn, slot_id)? {
            return Err(Error::SlotNotAvailable {
                slot_id,
                reason: format!(
                    "Slot {slot_id:?} is in voting state - no new slots can be claimed"
                ),
            });
        }

        if period_index > current_period + FUTURE_PERIODS - 1 {
            return Err(Error::SlotNotAvailable {
                slot_id,
                reason: format!(
                    "Slot period {} exceeds maximum allowed period {} (current + 20)",
                    period_index,
                    current_period + FUTURE_PERIODS - 1
                ),
            });
        }

        if decision.is_standard {
            if slot_index > STANDARD_SLOT_MAX {
                return Err(Error::SlotNotAvailable {
                    slot_id,
                    reason: format!(
                        "Standard slots must have index <= {STANDARD_SLOT_MAX}, got {slot_index}"
                    ),
                });
            }

            let (start_period, end_period) =
                self.get_active_window(current_ts, current_height)?;
            if period_index < start_period || period_index > end_period {
                return Err(Error::SlotNotAvailable {
                    slot_id,
                    reason: format!(
                        "Period {period_index} is not in active window for new slots ({start_period}-{end_period})"
                    ),
                });
            }

            let total_slots = self.total_for(
                rotxn,
                period_index,
                current_ts,
                current_height,
            )?;
            if slot_index as u64 >= total_slots {
                return Err(Error::SlotNotAvailable {
                    slot_id,
                    reason: format!(
                        "Standard slot index {slot_index} exceeds available slots {total_slots} for period {period_index}"
                    ),
                });
            }
        } else if slot_index < NONSTANDARD_SLOT_MIN {
            return Err(Error::SlotNotAvailable {
                slot_id,
                reason: format!(
                    "Non-standard slots must have index >= {NONSTANDARD_SLOT_MIN}, got {slot_index}"
                ),
            });
        }

        let claimed_slots =
            self.get_claimed_slot_ids_for_period(rotxn, period_index)?;

        if claimed_slots.contains(&slot_id) {
            return Err(Error::SlotAlreadyClaimed { slot_id });
        }

        Ok(())
    }

    pub fn claim_slot(
        &self,
        rwtxn: &mut RwTxn<'_>,
        slot_id: SlotId,
        decision: Decision,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<(), Error> {
        self.validate_slot_claim(
            rwtxn,
            slot_id,
            &decision,
            current_ts,
            current_height,
        )?;

        let period_index = slot_id.period_index();

        let mut period_slots = self
            .period_slots
            .try_get(rwtxn, &period_index)?
            .unwrap_or_default();

        let new_slot = Slot {
            slot_id,
            decision: Some(decision),
        };
        period_slots.insert(new_slot);
        self.period_slots.put(rwtxn, &period_index, &period_slots)?;

        let block_height = current_height.unwrap_or(0) as u64;
        let mut slot_history =
            SlotStateHistory::new(slot_id, block_height, current_ts);

        slot_history
            .transition_to_claimed_with_timestamp(block_height, current_ts)?;

        self.slot_state_histories
            .put(rwtxn, &slot_id, &slot_history)?;

        Ok(())
    }

    pub fn get_slot(
        &self,
        rotxn: &sneed::RoTxn,
        slot_id: SlotId,
    ) -> Result<Option<Slot>, Error> {
        let period = slot_id.period_index();
        if let Some(period_slots) = self.period_slots.try_get(rotxn, &period)? {
            let search_slot = Slot {
                slot_id,
                decision: None,
            };

            if let Some(found_slot) = period_slots.get(&search_slot) {
                return Ok(Some(found_slot.clone()));
            }

            for slot in period_slots.range(search_slot..) {
                if slot.slot_id == slot_id {
                    return Ok(Some(slot.clone()));
                }
                if slot.slot_id > slot_id {
                    break;
                }
            }
        }
        Ok(None)
    }

    pub fn get_available_slots_in_period(
        &self,
        rotxn: &sneed::RoTxn,
        period_index: u32,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<Vec<SlotId>, Error> {
        let total_slots =
            self.total_for(rotxn, period_index, current_ts, current_height)?;

        let max_slot_index =
            std::cmp::min(total_slots, (STANDARD_SLOT_MAX + 1) as u64);

        let mut available_slots = Vec::with_capacity(max_slot_index as usize);

        let claimed_slots =
            self.get_claimed_slot_ids_for_period(rotxn, period_index)?;

        for slot_index in 0..max_slot_index {
            let slot_id = SlotId::new(period_index, slot_index as u32)?;

            if !claimed_slots.contains(&slot_id) {
                available_slots.push(slot_id);
            }
        }

        Ok(available_slots)
    }

    pub fn get_claimed_slots_in_period(
        &self,
        rotxn: &sneed::RoTxn,
        period_index: u32,
    ) -> Result<Vec<Slot>, Error> {
        let mut claimed_slots = Vec::new();

        if let Some(period_slots) =
            self.period_slots.try_get(rotxn, &period_index)?
        {
            for slot in &period_slots {
                if slot.decision.is_some() {
                    claimed_slots.push(slot.clone());
                }
            }
        }

        Ok(claimed_slots)
    }

    pub fn get_claimed_slot_count_in_period(
        &self,
        rotxn: &sneed::RoTxn,
        period_index: u32,
    ) -> Result<u64, Error> {
        let claimed_slots =
            self.get_claimed_slot_ids_for_period(rotxn, period_index)?;
        Ok(claimed_slots.len() as u64)
    }

    pub fn get_all_claimed_slots(
        &self,
        rotxn: &sneed::RoTxn,
    ) -> Result<Vec<Slot>, Error> {
        let mut all_claimed_slots = Vec::new();

        let mut iter = self.period_slots.iter(rotxn)?;
        while let Some((_period, period_slots)) = iter.next()? {
            for slot in &period_slots {
                if slot.decision.is_some() {
                    all_claimed_slots.push(slot.clone());
                }
            }
        }

        Ok(all_claimed_slots)
    }

    pub fn get_voting_periods(
        &self,
        rotxn: &sneed::RoTxn,
        _current_ts: u64,
        _current_height: Option<u32>,
    ) -> Result<Vec<(u32, u64, u64)>, Error> {
        use std::collections::HashMap;

        let mut period_voting_counts: HashMap<u32, u64> = HashMap::new();

        let mut iter = self.slot_state_histories.iter(rotxn)?;
        while let Some((slot_id, history)) = iter.next()? {
            if history.current_state() == SlotState::Voting {
                let period = slot_id.period_index();
                *period_voting_counts.entry(period).or_insert(0) += 1;
            }
        }

        let mut voting_periods: Vec<(u32, u64, u64)> = period_voting_counts
            .into_iter()
            .map(|(period, count)| {
                let total_slots = 500u64;
                (period, count, total_slots)
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
    ) -> Result<super::type_aliases::PeriodSummary, Error> {
        let active_periods =
            self.get_active_periods(rotxn, current_ts, current_height)?;
        let voting_periods_full =
            self.get_voting_periods(rotxn, current_ts, current_height)?;
        let voting_periods = voting_periods_full
            .into_iter()
            .map(|(period, claimed, _total)| (period, claimed))
            .collect();

        Ok((active_periods, voting_periods))
    }

    pub fn revert_claim_slot(
        &self,
        rwtxn: &mut RwTxn<'_>,
        slot_id: SlotId,
    ) -> Result<(), Error> {
        let period_index = slot_id.period_index();

        let mut period_slots = self
            .period_slots
            .try_get(rwtxn, &period_index)?
            .unwrap_or_default();

        // Find and remove the slot using BTreeSet operations
        let target_slot = period_slots
            .iter()
            .find(|slot| slot.slot_id == slot_id)
            .cloned();

        if let Some(slot_to_remove) = target_slot {
            period_slots.remove(&slot_to_remove);

            // Update period_slots database
            if period_slots.is_empty() {
                self.period_slots.delete(rwtxn, &period_index)?;
            } else {
                self.period_slots.put(rwtxn, &period_index, &period_slots)?;
            }
        } else {
            tracing::debug!(
                "Attempted to revert slot {:?} that wasn't found",
                slot_id
            );
        }

        Ok(())
    }

    pub fn get_slot_state_history(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
    ) -> Result<Option<SlotStateHistory>, Error> {
        Ok(self.slot_state_histories.try_get(rotxn, &slot_id)?)
    }

    pub fn get_slot_current_state(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
    ) -> Result<SlotState, Error> {
        Ok(self
            .get_slot_state_history(rotxn, slot_id)?
            .map(|h| h.current_state())
            .unwrap_or(SlotState::Created))
    }

    pub fn transition_slot_to_voting(
        &self,
        rwtxn: &mut RwTxn,
        slot_id: SlotId,
        block_height: u64,
        timestamp: u64,
    ) -> Result<(), Error> {
        let mut history = self.get_slot_state_history(rwtxn, slot_id)?.ok_or(
            Error::InvalidSlotId {
                reason: format!("Slot {slot_id:?} has no state history"),
            },
        )?;

        let voting_period = slot_id.voting_period();
        history.transition_to_voting_with_timestamp(
            block_height,
            timestamp,
            voting_period,
        )?;
        self.slot_state_histories.put(rwtxn, &slot_id, &history)?;
        Ok(())
    }

    pub fn transition_slot_to_resolved(
        &self,
        rwtxn: &mut RwTxn,
        slot_id: SlotId,
        block_height: u64,
        _timestamp: u64,
        consensus_outcome: f64,
    ) -> Result<(), Error> {
        let mut history = self.get_slot_state_history(rwtxn, slot_id)?.ok_or(
            Error::InvalidSlotId {
                reason: format!("Slot {slot_id:?} has no state history"),
            },
        )?;

        history
            .transition_to_resolved(consensus_outcome, block_height as u32)?;
        self.slot_state_histories.put(rwtxn, &slot_id, &history)?;
        Ok(())
    }

    pub fn get_slots_in_state(
        &self,
        rotxn: &RoTxn,
        state: SlotState,
    ) -> Result<Vec<SlotId>, Error> {
        let mut slots_in_state = Vec::new();

        let mut iter = self.slot_state_histories.iter(rotxn)?;
        while let Some((slot_id, history)) = iter.next()? {
            if history.current_state() == state {
                slots_in_state.push(slot_id);
            }
        }

        Ok(slots_in_state)
    }

    pub fn slot_has_consensus(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
    ) -> Result<bool, Error> {
        Ok(self
            .get_slot_state_history(rotxn, slot_id)?
            .map(|h| h.current_state().has_consensus())
            .unwrap_or(false))
    }

    pub fn rollback_slot_states_to_height(
        &self,
        rwtxn: &mut RwTxn,
        height: u64,
    ) -> Result<(), Error> {
        let mut slots_to_update = Vec::new();
        // Track slots that need to be removed from period_slots
        // because they rolled back to Created state (before being claimed)
        let mut slots_to_unclaim: Vec<SlotId> = Vec::new();

        {
            let mut iter = self.slot_state_histories.iter(rwtxn)?;
            while let Some((slot_id, mut history)) = iter.next()? {
                let was_claimed = history.current_state() != SlotState::Created;
                history.rollback_to_height(height);
                let is_now_created =
                    history.current_state() == SlotState::Created;

                // If slot was claimed but rolled back to Created, track it for removal
                if was_claimed && is_now_created {
                    slots_to_unclaim.push(slot_id);
                }
                slots_to_update.push((slot_id, history));
            }
        }

        // Update slot state histories
        for (slot_id, history) in slots_to_update {
            self.slot_state_histories.put(rwtxn, &slot_id, &history)?;
        }

        // Remove unclaimed slots from period_slots
        for slot_id in slots_to_unclaim {
            let period_index = slot_id.period_index();

            if let Some(mut period_slots) =
                self.period_slots.try_get(rwtxn, &period_index)?
            {
                period_slots.retain(|s| s.slot_id != slot_id);
                if period_slots.is_empty() {
                    self.period_slots.delete(rwtxn, &period_index)?;
                } else {
                    self.period_slots.put(
                        rwtxn,
                        &period_index,
                        &period_slots,
                    )?;
                }
            }
        }

        Ok(())
    }
}

impl SlotValidationInterface for Dbs {
    fn validate_slot_claim(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<(), Error> {
        self.validate_slot_claim(
            rotxn,
            slot_id,
            decision,
            current_ts,
            current_height,
        )
    }

    fn try_get_height(&self, _rotxn: &RoTxn) -> Result<Option<u32>, Error> {
        Ok(None)
    }
}
