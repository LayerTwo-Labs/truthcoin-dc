use crate::state::{
    Error,
    slots::{Dbs as SlotsDbs, SlotConfig, SlotId},
    voting::types::{VotingPeriod, VotingPeriodId, VotingPeriodStatus},
};
use sneed::RoTxn;
use std::collections::HashMap;

pub fn calculate_period_boundaries(
    period_index: u32,
    config: &SlotConfig,
) -> (u64, u64) {
    if config.testing_mode {
        if period_index == 0 {
            (0, 0)
        } else {
            let start_block =
                (period_index - 1) * config.testing_blocks_per_period;
            let end_block = period_index * config.testing_blocks_per_period;
            (start_block as u64, end_block as u64)
        }
    } else {
        use crate::math::voting::constants::PRODUCTION_PERIOD_DURATION_SECONDS;
        let start = period_index as u64 * PRODUCTION_PERIOD_DURATION_SECONDS;
        let end = start + PRODUCTION_PERIOD_DURATION_SECONDS;
        (start, end)
    }
}

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
        return Ok(Vec::new());
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

pub fn calculate_voting_period(
    rotxn: &RoTxn,
    period_id: VotingPeriodId,
    current_height: u32,
    current_timestamp: u64,
    config: &SlotConfig,
    slots_db: &SlotsDbs,
    has_outcomes: bool,
) -> Result<VotingPeriod, Error> {
    let (start_boundary, end_boundary) =
        calculate_period_boundaries(period_id.as_u32(), config);

    let decision_slots =
        get_decision_slots_for_period(rotxn, period_id, slots_db)?;

    let effective_current = if config.testing_mode {
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

pub fn get_all_active_periods(
    rotxn: &RoTxn,
    slots_db: &SlotsDbs,
    config: &SlotConfig,
    current_timestamp: u64,
    current_height: u32,
    voting_db: &crate::state::voting::database::VotingDatabases,
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
            calculate_period_boundaries(period_index, config);

        let effective_current = if config.testing_mode {
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
}
