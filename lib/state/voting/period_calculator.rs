use crate::state::{
    Error,
    slots::{Dbs as SlotsDbs, SlotConfig, SlotId},
    voting::types::{VotingPeriod, VotingPeriodId, VotingPeriodStatus},
};
use sneed::RoTxn;
use std::collections::HashMap;

/// Returns the 1-based period number for a given timestamp/block_height.
///
/// This is the **single source of truth** for "what period are we in".
/// In blocks mode: `height / quantity + 1`.
/// In time mode: `(timestamp - genesis_ts) / quantity + 1`.
pub fn get_current_period(
    timestamp: u64,
    block_height: Option<u32>,
    genesis_ts: u64,
    config: &SlotConfig,
) -> Result<u32, Error> {
    if config.is_blocks {
        let height = block_height.unwrap_or(0);
        Ok(height
            .checked_div(config.quantity)
            .map(|v| v + 1)
            .unwrap_or(0))
    } else {
        if genesis_ts == 0 || timestamp < genesis_ts {
            return Ok(0);
        }
        Ok(((timestamp - genesis_ts) / config.quantity as u64) as u32 + 1)
    }
}

/// Display/consensus: returns (start, end) boundaries for a period.
/// Used by GUI timers, RPC, and VotingPeriod construction for consensus.
pub fn calculate_period_boundaries(
    period_index: u32,
    config: &SlotConfig,
    genesis_ts: u64,
) -> (u64, u64) {
    if config.is_blocks {
        if period_index == 0 {
            (0, 0)
        } else {
            let start_block = (period_index - 1) * config.quantity;
            let end_block = period_index * config.quantity;
            (start_block as u64, end_block as u64)
        }
    } else {
        let period_duration = config.quantity as u64;
        if period_index == 0 {
            (0, 0)
        } else {
            let start =
                genesis_ts + (period_index - 1) as u64 * period_duration;
            let end = start + period_duration;
            (start, end)
        }
    }
}

/// Display only: determines period status from boundary timestamps.
/// Used by RPC period queries and GUI. Not used by the block transition path.
pub fn calculate_period_status(
    start_timestamp: u64,
    end_timestamp: u64,
    current_timestamp: u64,
    has_outcomes: bool,
) -> VotingPeriodStatus {
    if has_outcomes {
        VotingPeriodStatus::Resolved
    } else if current_timestamp >= end_timestamp {
        VotingPeriodStatus::Closed
    } else if current_timestamp >= start_timestamp {
        VotingPeriodStatus::Active
    } else {
        VotingPeriodStatus::Pending
    }
}

pub fn get_decision_slots_for_period(
    rotxn: &RoTxn,
    voting_period_id: VotingPeriodId,
    slots_db: &SlotsDbs,
) -> Result<Vec<SlotId>, Error> {
    let voting_period = voting_period_id.as_u32();

    if voting_period == 0 {
        return Err(Error::InvalidTransaction {
            reason: "Voting period 0 does not exist".to_string(),
        });
    }

    let claim_period = voting_period - 1;
    let all_slots = slots_db.get_all_claimed_slots(rotxn)?;

    let mut decision_slots = Vec::new();
    for slot in all_slots {
        if slot.slot_id.period_index() == claim_period {
            decision_slots.push(slot.slot_id);
        }
    }

    Ok(decision_slots)
}

/// Vote validation / RPC / statistics: constructs a full VotingPeriod struct.
/// Used by `cast_vote` validation and RPC queries. Not used by block transitions.
#[allow(clippy::too_many_arguments)]
pub fn calculate_voting_period(
    rotxn: &RoTxn,
    period_id: VotingPeriodId,
    current_height: u32,
    current_timestamp: u64,
    config: &SlotConfig,
    slots_db: &SlotsDbs,
    has_outcomes: bool,
    genesis_ts: u64,
) -> Result<VotingPeriod, Error> {
    let (start_boundary, end_boundary) =
        calculate_period_boundaries(period_id.as_u32(), config, genesis_ts);

    let decision_slots =
        get_decision_slots_for_period(rotxn, period_id, slots_db)?;

    let effective_current = if config.is_blocks {
        current_height as u64
    } else {
        current_timestamp
    };

    let status = calculate_period_status(
        start_boundary,
        end_boundary,
        effective_current,
        has_outcomes,
    );

    let mut period = VotingPeriod::new(
        period_id,
        start_boundary,
        end_boundary,
        decision_slots,
    );

    period.status = status;

    Ok(period)
}

/// Display only: returns all periods that have claimed slots, with computed statuses.
/// Used by GUI/RPC period listing. Not used by block transitions.
pub fn get_all_active_periods(
    rotxn: &RoTxn,
    slots_db: &SlotsDbs,
    config: &SlotConfig,
    current_timestamp: u64,
    current_height: u32,
    voting_db: &crate::state::voting::database::VotingDatabases,
    genesis_ts: u64,
) -> Result<HashMap<VotingPeriodId, VotingPeriod>, Error> {
    let all_slots = slots_db.get_all_claimed_slots(rotxn)?;
    let mut period_map: HashMap<u32, Vec<SlotId>> = HashMap::new();

    for slot in all_slots {
        let voting_period = slot.slot_id.voting_period();
        period_map
            .entry(voting_period)
            .or_default()
            .push(slot.slot_id);
    }

    let mut result = HashMap::new();
    for (period_index, decision_slots) in period_map {
        let period_id = VotingPeriodId::new(period_index);
        let has_outcomes = voting_db.has_consensus(rotxn, period_id)?;

        let (start_boundary, end_boundary) =
            calculate_period_boundaries(period_index, config, genesis_ts);

        let effective_current = if config.is_blocks {
            current_height as u64
        } else {
            current_timestamp
        };

        let status = calculate_period_status(
            start_boundary,
            end_boundary,
            effective_current,
            has_outcomes,
        );

        let period = VotingPeriod {
            id: period_id,
            start_timestamp: start_boundary,
            end_timestamp: end_boundary,
            status,
            decision_slots,
        };

        result.insert(period_id, period);
    }

    Ok(result)
}

pub fn validate_transition(
    period: &VotingPeriod,
    new_status: VotingPeriodStatus,
    current_timestamp: u64,
) -> Result<(), Error> {
    match (period.status, new_status) {
        (VotingPeriodStatus::Pending, VotingPeriodStatus::Active) => {
            validate_pending_to_active(period, current_timestamp)
        }
        (VotingPeriodStatus::Active, VotingPeriodStatus::Closed) => {
            validate_active_to_closed(period, current_timestamp)
        }
        (VotingPeriodStatus::Closed, VotingPeriodStatus::Resolved) => {
            validate_closed_to_resolved(period)
        }
        (status, new) if status == new => Ok(()),
        _ => Err(Error::InvalidTransaction {
            reason: format!(
                "Invalid voting period state transition: {:?} -> {:?}",
                period.status, new_status
            ),
        }),
    }
}

fn validate_pending_to_active(
    period: &VotingPeriod,
    current_timestamp: u64,
) -> Result<(), Error> {
    if current_timestamp < period.start_timestamp {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Cannot activate period before start time (current: {}, start: {})",
                current_timestamp, period.start_timestamp
            ),
        });
    }

    Ok(())
}

fn validate_active_to_closed(
    period: &VotingPeriod,
    current_timestamp: u64,
) -> Result<(), Error> {
    if current_timestamp < period.end_timestamp {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Cannot close period before end time (current: {}, end: {})",
                current_timestamp, period.end_timestamp
            ),
        });
    }

    Ok(())
}

fn validate_closed_to_resolved(_period: &VotingPeriod) -> Result<(), Error> {
    Ok(())
}

pub fn can_accept_votes(period: &VotingPeriod) -> bool {
    period.status == VotingPeriodStatus::Active
}

pub fn can_close(period: &VotingPeriod) -> bool {
    period.status == VotingPeriodStatus::Active
}

pub fn can_resolve(period: &VotingPeriod) -> bool {
    period.status == VotingPeriodStatus::Closed
}

pub fn is_terminal(period: &VotingPeriod) -> bool {
    period.status == VotingPeriodStatus::Resolved
}

#[cfg(test)]
mod tests {
    use super::super::types::VotingPeriodId;
    use super::*;

    fn create_test_period(status: VotingPeriodStatus) -> VotingPeriod {
        let mut period =
            VotingPeriod::new(VotingPeriodId::new(1), 1000, 2000, vec![]);
        period.status = status;
        period
    }

    #[test]
    fn test_valid_transitions() {
        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(
            validate_transition(&period, VotingPeriodStatus::Active, 1000)
                .is_ok()
        );

        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(
            validate_transition(&period, VotingPeriodStatus::Closed, 2000)
                .is_ok()
        );

        let period = create_test_period(VotingPeriodStatus::Closed);
        assert!(
            validate_transition(&period, VotingPeriodStatus::Resolved, 2000)
                .is_ok()
        );
    }

    #[test]
    fn test_invalid_transitions() {
        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(
            validate_transition(&period, VotingPeriodStatus::Closed, 1000)
                .is_err()
        );

        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(
            validate_transition(&period, VotingPeriodStatus::Resolved, 2000)
                .is_err()
        );
    }

    #[test]
    fn test_early_activation() {
        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(
            validate_transition(&period, VotingPeriodStatus::Active, 999)
                .is_err()
        );
    }

    #[test]
    fn test_early_close() {
        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(
            validate_transition(&period, VotingPeriodStatus::Closed, 1999)
                .is_err()
        );
    }

    #[test]
    fn test_can_accept_votes() {
        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(can_accept_votes(&period));

        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(!can_accept_votes(&period));

        let period = create_test_period(VotingPeriodStatus::Closed);
        assert!(!can_accept_votes(&period));

        let period = create_test_period(VotingPeriodStatus::Resolved);
        assert!(!can_accept_votes(&period));
    }

    #[test]
    fn test_can_close() {
        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(can_close(&period));

        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(!can_close(&period));

        let period = create_test_period(VotingPeriodStatus::Closed);
        assert!(!can_close(&period));
    }

    #[test]
    fn test_can_resolve() {
        let period = create_test_period(VotingPeriodStatus::Closed);
        assert!(can_resolve(&period));

        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(!can_resolve(&period));

        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(!can_resolve(&period));

        let period = create_test_period(VotingPeriodStatus::Resolved);
        assert!(!can_resolve(&period));
    }

    #[test]
    fn test_is_terminal() {
        let period = create_test_period(VotingPeriodStatus::Resolved);
        assert!(is_terminal(&period));

        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(!is_terminal(&period));

        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(!is_terminal(&period));

        let period = create_test_period(VotingPeriodStatus::Closed);
        assert!(!is_terminal(&period));
    }

    // --- get_current_period tests ---

    #[test]
    fn test_get_current_period_blocks_mode() {
        let config = SlotConfig::testing(10);

        // Block 0-9 → period 1
        assert_eq!(get_current_period(0, Some(0), 0, &config).unwrap(), 1);
        assert_eq!(get_current_period(0, Some(5), 0, &config).unwrap(), 1);
        assert_eq!(get_current_period(0, Some(9), 0, &config).unwrap(), 1);

        // Block 10-19 → period 2
        assert_eq!(get_current_period(0, Some(10), 0, &config).unwrap(), 2);
        assert_eq!(get_current_period(0, Some(19), 0, &config).unwrap(), 2);

        // Block 20-29 → period 3
        assert_eq!(get_current_period(0, Some(20), 0, &config).unwrap(), 3);
    }

    #[test]
    fn test_get_current_period_time_mode() {
        let config = SlotConfig::default(); // 120 seconds per period
        let genesis_ts = 1000;

        // Before genesis
        assert_eq!(
            get_current_period(999, None, genesis_ts, &config).unwrap(),
            0
        );

        // Genesis to genesis+119 → period 1
        assert_eq!(
            get_current_period(1000, None, genesis_ts, &config).unwrap(),
            1
        );
        assert_eq!(
            get_current_period(1119, None, genesis_ts, &config).unwrap(),
            1
        );

        // genesis+120 to genesis+239 → period 2
        assert_eq!(
            get_current_period(1120, None, genesis_ts, &config).unwrap(),
            2
        );
    }

    #[test]
    fn test_get_current_period_zero_genesis() {
        let config = SlotConfig::default();
        // genesis_ts == 0 → period 0
        assert_eq!(get_current_period(500, None, 0, &config).unwrap(), 0);
    }

    // --- Transition rule tests ---

    #[test]
    fn test_claimed_to_voting_transition_rule() {
        // With SlotConfig::testing(10), a slot claimed in period 1 (period_index=1)
        // should transition to Voting when current_period > 1.
        let config = SlotConfig::testing(10);

        // Block 9 → period 1; current_period(1) > period_index(1) is false
        let period_at_9 = get_current_period(0, Some(9), 0, &config).unwrap();
        assert_eq!(period_at_9, 1);
        assert!(period_at_9 <= 1); // No transition

        // Block 10 → period 2; current_period(2) > period_index(1) is true
        let period_at_10 = get_current_period(0, Some(10), 0, &config).unwrap();
        assert_eq!(period_at_10, 2);
        assert!(period_at_10 > 1); // Transition to Voting
    }

    #[test]
    fn test_voting_to_resolved_transition_rule() {
        // With SlotConfig::testing(10), a slot with period_index=1 has voting_period=2.
        // It should resolve when current_period > 2.
        let config = SlotConfig::testing(10);

        // Block 19 → period 2; current_period(2) > voting_period(2) is false
        let period_at_19 = get_current_period(0, Some(19), 0, &config).unwrap();
        assert_eq!(period_at_19, 2);
        assert!(period_at_19 <= 2); // No resolution

        // Block 20 → period 3; current_period(3) > voting_period(2) is true
        let period_at_20 = get_current_period(0, Some(20), 0, &config).unwrap();
        assert_eq!(period_at_20, 3);
        assert!(period_at_20 > 2); // Resolution
    }

    #[test]
    fn test_catch_up_scenario() {
        // If current_period jumps from 1 to 5, slots from periods 1-3 should all
        // be eligible for resolution (their voting_periods are 2-4, all < 5).
        let config = SlotConfig::testing(10);

        let period_at_40 = get_current_period(0, Some(40), 0, &config).unwrap();
        assert_eq!(period_at_40, 5);

        // Slot period_index=1, voting_period=2: 5 > 2 → resolve
        assert!(period_at_40 > 2);
        // Slot period_index=2, voting_period=3: 5 > 3 → resolve
        assert!(period_at_40 > 3);
        // Slot period_index=3, voting_period=4: 5 > 4 → resolve
        assert!(period_at_40 > 4);
        // Slot period_index=4, voting_period=5: 5 > 5 → false, not yet
        assert!(period_at_40 <= 5);
    }

    #[test]
    fn test_full_lifecycle_blocks_mode() {
        // Manual trace with SlotConfig::testing(10):
        // A slot claimed at block 5 (period_index=1) should:
        let config = SlotConfig::testing(10);

        // Stay Claimed through block 9 (current_period=1, 1 > 1 is false)
        for h in 5..=9 {
            let cp = get_current_period(0, Some(h), 0, &config).unwrap();
            assert_eq!(cp, 1, "at block {h}");
            assert!(cp <= 1, "should NOT transition at block {h}");
        }

        // Transition to Voting at block 10 (current_period=2, 2 > 1 is true)
        let cp = get_current_period(0, Some(10), 0, &config).unwrap();
        assert_eq!(cp, 2);
        assert!(cp > 1, "should transition to Voting at block 10");

        // Stay Voting through block 19 (current_period=2, 2 > 2 is false)
        for h in 10..=19 {
            let cp = get_current_period(0, Some(h), 0, &config).unwrap();
            assert_eq!(cp, 2, "at block {h}");
            assert!(cp <= 2, "should NOT resolve at block {h}");
        }

        // Resolve at block 20 (current_period=3, 3 > 2 is true)
        let cp = get_current_period(0, Some(20), 0, &config).unwrap();
        assert_eq!(cp, 3);
        assert!(cp > 2, "should resolve at block 20");
    }
}
