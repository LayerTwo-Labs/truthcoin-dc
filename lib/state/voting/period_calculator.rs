use crate::state::{
    Error,
    decisions::{Dbs as DecisionsDbs, DecisionConfig, DecisionId},
    voting::types::{VotingPeriod, VotingPeriodId, VotingPeriodStatus},
};
use sneed::RoTxn;
use std::collections::HashMap;

const SECONDS_PER_DAY: u64 = 86_400;

/// Calendar quarters since 1970 Q1 of a unix timestamp, in UTC
pub(crate) fn utc_quarter(timestamp: u64) -> u64 {
    let days = timestamp / SECONDS_PER_DAY + 719_468;
    let era = days / 146_097;
    let day_of_era = days % 146_097;
    let year_of_era = (day_of_era - day_of_era / 1_460 + day_of_era / 36_524
        - day_of_era / 146_096)
        / 365;
    let day_of_year =
        day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = era * 400 + year_of_era + u64::from(month <= 2);
    (year - 1970) * 4 + (month - 1) / 3
}

/// Unix timestamp of 00:00 UTC on the first day of a quarter counted as in
/// [`utc_quarter`]
pub(crate) fn utc_quarter_start(quarter: u64) -> u64 {
    let month = (quarter % 4) * 3 + 1;
    let year = 1970 + quarter / 4 - u64::from(month <= 2);
    let era = year / 400;
    let year_of_era = year % 400;
    let shifted_month = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * shifted_month + 2) / 5;
    let day_of_era =
        year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    (era * 146_097 + day_of_era - 719_468) * SECONDS_PER_DAY
}

/// Returns the 1-based period number for a given timestamp/block_height.
///
/// This is the **single source of truth** for "what period are we in".
/// In blocks mode: `height / quantity + 1`.
/// In time mode: the calendar quarter (UTC) of `timestamp`, counting the
/// quarter of `genesis_ts` as period 1.
pub fn get_current_period(
    timestamp: u64,
    block_height: Option<u32>,
    genesis_ts: u64,
    config: &DecisionConfig,
) -> Result<u32, Error> {
    if config.is_blocks {
        let height = block_height.unwrap_or(0);
        Ok(height
            .checked_div(config.quantity)
            .map(|v| v + 1)
            .unwrap_or(0))
    } else {
        if genesis_ts == 0 {
            return Ok(0);
        }
        Ok(utc_quarter(timestamp)
            .checked_sub(utc_quarter(genesis_ts))
            .map_or(0, |quarters| quarters as u32 + 1))
    }
}

/// Display/consensus: returns (start, end) boundaries for a period.
/// Used by GUI timers, RPC, and VotingPeriod construction for consensus.
/// In time mode a period is a calendar quarter, from 00:00 UTC on its first
/// day to 00:00 UTC on the first day of the next quarter.
pub fn calculate_period_boundaries(
    period_index: u32,
    config: &DecisionConfig,
    genesis_ts: u64,
) -> (u64, u64) {
    if period_index == 0 {
        return (0, 0);
    }
    if config.is_blocks {
        let start_block = (period_index - 1) * config.quantity;
        let end_block = period_index * config.quantity;
        (start_block as u64, end_block as u64)
    } else {
        let quarter = utc_quarter(genesis_ts) + u64::from(period_index) - 1;
        (utc_quarter_start(quarter), utc_quarter_start(quarter + 1))
    }
}

/// Whether the reveal window following `voting_period` has elapsed, using the
/// same clock as [`get_current_period`].
pub fn reveal_window_closed(
    voting_period: u32,
    timestamp: u64,
    block_height: Option<u32>,
    genesis_ts: u64,
    config: &DecisionConfig,
) -> bool {
    let (_, end) =
        calculate_period_boundaries(voting_period, config, genesis_ts);
    let now = if config.is_blocks {
        u64::from(block_height.unwrap_or(0))
    } else if genesis_ts == 0 {
        return false;
    } else {
        timestamp
    };
    now >= end + u64::from(config.reveal_window())
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

pub fn get_decision_ids_for_period(
    rotxn: &RoTxn,
    voting_period_id: VotingPeriodId,
    decisions_db: &DecisionsDbs,
) -> Result<Vec<DecisionId>, Error> {
    let voting_period = voting_period_id.as_u32();

    if voting_period == 0 {
        return Err(Error::InvalidTransaction {
            reason: "Voting period 0 does not exist".to_string(),
        });
    }

    Ok(decisions_db
        .get_claimed_decisions_in_period(rotxn, voting_period - 1)?
        .into_iter()
        .map(|entry| entry.decision_id)
        .collect())
}

/// Vote validation / RPC / statistics: constructs a full VotingPeriod struct.
/// Used by `cast_vote` validation and RPC queries. Not used by block transitions.
#[allow(clippy::too_many_arguments)]
pub fn calculate_voting_period(
    rotxn: &RoTxn,
    period_id: VotingPeriodId,
    current_height: u32,
    current_timestamp: u64,
    config: &DecisionConfig,
    decisions_db: &DecisionsDbs,
    has_outcomes: bool,
    genesis_ts: u64,
) -> Result<VotingPeriod, Error> {
    let (start_boundary, end_boundary) =
        calculate_period_boundaries(period_id.as_u32(), config, genesis_ts);

    let decision_ids =
        get_decision_ids_for_period(rotxn, period_id, decisions_db)?;

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
        decision_ids,
    );

    period.status = status;

    Ok(period)
}

/// Display only: returns all periods that have claimed decisions, with computed statuses.
/// Used by GUI/RPC period listing. Not used by block transitions.
pub fn get_all_active_periods(
    rotxn: &RoTxn,
    decisions_db: &DecisionsDbs,
    config: &DecisionConfig,
    current_timestamp: u64,
    current_height: u32,
    voting_db: &crate::state::voting::database::VotingDatabases,
    genesis_ts: u64,
) -> Result<HashMap<VotingPeriodId, VotingPeriod>, Error> {
    let all_decisions = decisions_db.get_all_claimed_decisions(rotxn)?;
    let mut period_map: HashMap<u32, Vec<DecisionId>> = HashMap::new();

    for entry in all_decisions {
        let voting_period = entry.decision_id.voting_period();
        period_map
            .entry(voting_period)
            .or_default()
            .push(entry.decision_id);
    }

    let mut result = HashMap::new();
    for (period_index, decision_ids) in period_map {
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
            decision_ids,
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
            validate_closed_to_resolved()
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

fn validate_closed_to_resolved() -> Result<(), Error> {
    Ok(())
}

pub fn can_accept_votes(period: &VotingPeriod) -> bool {
    period.status == VotingPeriodStatus::Active
}

#[cfg(test)]
mod tests {
    use super::super::types::VotingPeriodId;
    use super::*;
    use std::num::NonZeroU32;

    const TEN: NonZeroU32 = NonZeroU32::new(10).unwrap();

    fn can_close(period: &VotingPeriod) -> bool {
        period.status == VotingPeriodStatus::Active
    }

    fn can_resolve(period: &VotingPeriod) -> bool {
        period.status == VotingPeriodStatus::Closed
    }

    fn is_terminal(period: &VotingPeriod) -> bool {
        period.status == VotingPeriodStatus::Resolved
    }

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
    fn test_current_period_blocks() {
        let config = DecisionConfig::testing(TEN);

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

    const OCT_1_2026: u64 = 1_790_812_800;
    const JAN_1_2027: u64 = 1_798_761_600;
    const APR_1_2027: u64 = 1_806_537_600;
    const JUL_1_2027: u64 = 1_814_400_000;
    const OCT_1_2027: u64 = 1_822_348_800;
    const OCT_29_2026: u64 = 1_793_232_000;

    #[test]
    fn test_current_period_time() {
        let config = DecisionConfig::production();
        let genesis_ts = OCT_29_2026 + 12 * 3600;
        let period = |timestamp| {
            get_current_period(timestamp, None, genesis_ts, &config).unwrap()
        };

        assert_eq!(period(OCT_1_2026 - 1), 0);
        assert_eq!(period(OCT_1_2026), 1);
        assert_eq!(period(genesis_ts), 1);
        assert_eq!(period(JAN_1_2027 - 1), 1);
        assert_eq!(period(JAN_1_2027), 2);
        assert_eq!(period(APR_1_2027 - 1), 2);
        assert_eq!(period(APR_1_2027), 3);
        assert_eq!(period(JUL_1_2027), 4);
        assert_eq!(period(OCT_1_2027 - 1), 4);
        assert_eq!(period(OCT_1_2027), 5);
    }

    /// Production periods are UTC calendar quarters from the genesis quarter
    #[test]
    fn periods_are_utc_quarters() {
        let config = DecisionConfig::production();
        let genesis_ts = OCT_29_2026 + 12 * 3600;
        let starts = [OCT_1_2026, JAN_1_2027, APR_1_2027, JUL_1_2027];
        for (period, bounds) in (1..).zip(starts.windows(2)) {
            assert_eq!(
                calculate_period_boundaries(period, &config, genesis_ts),
                (bounds[0], bounds[1]),
            );
        }
        assert_eq!(
            crate::state::decisions::period_to_name(1, genesis_ts),
            "Q4 Y2026"
        );
        assert_eq!(
            crate::state::decisions::period_to_name(2, genesis_ts),
            "Q1 Y2027"
        );
    }

    /// Quarter start and index conversions round-trip through leap years
    #[test]
    fn quarter_math_leap_years() {
        const APR_1_2028: u64 = 1_838_160_000;
        const MAR_1_2100: u64 = 4_107_542_400;
        const APR_1_2100: u64 = 4_110_220_800;

        assert_eq!(utc_quarter_start(utc_quarter(APR_1_2028)), APR_1_2028);
        assert_eq!(utc_quarter(APR_1_2028 - 1), utc_quarter(APR_1_2028) - 1);
        assert_eq!(utc_quarter(MAR_1_2100), utc_quarter(APR_1_2100) - 1);
        assert_eq!(utc_quarter_start(utc_quarter(MAR_1_2100) + 1), APR_1_2100);
        for quarter in 0..(4 * 400) {
            let start = utc_quarter_start(quarter);
            assert_eq!(start % 86_400, 0);
            assert_eq!(utc_quarter(start), quarter);
            assert_eq!(
                utc_quarter(utc_quarter_start(quarter + 1) - 1),
                quarter
            );
        }
    }

    #[test]
    fn test_zero_genesis_period() {
        let config = DecisionConfig::default();
        // genesis_ts == 0 → period 0
        assert_eq!(get_current_period(500, None, 0, &config).unwrap(), 0);
    }

    // --- Transition rule tests ---

    #[test]
    fn test_claimed_to_voting() {
        // With DecisionConfig::testing(TEN), a decision claimed in period 1 (period_index=1)
        // should transition to Voting when current_period > 1.
        let config = DecisionConfig::testing(TEN);

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
    fn test_voting_to_resolved() {
        // With DecisionConfig::testing(TEN), a decision with period_index=1 has voting_period=2.
        // It should resolve when current_period > 2.
        let config = DecisionConfig::testing(TEN);

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
        // If current_period jumps from 1 to 5, decisions from periods 1-3 should all
        // be eligible for resolution (their voting_periods are 2-4, all < 5).
        let config = DecisionConfig::testing(TEN);

        let period_at_40 = get_current_period(0, Some(40), 0, &config).unwrap();
        assert_eq!(period_at_40, 5);

        // Decision period_index=1, voting_period=2: 5 > 2 → resolve
        assert!(period_at_40 > 2);
        // Decision period_index=2, voting_period=3: 5 > 3 → resolve
        assert!(period_at_40 > 3);
        // Decision period_index=3, voting_period=4: 5 > 4 → resolve
        assert!(period_at_40 > 4);
        // Decision period_index=4, voting_period=5: 5 > 5 → false, not yet
        assert!(period_at_40 <= 5);
    }

    #[test]
    fn test_full_lifecycle_blocks_mode() {
        // Manual trace with DecisionConfig::testing(TEN):
        // A decision claimed at block 5 (period_index=1) should:
        let config = DecisionConfig::testing(TEN);

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

        // Reveal at block 20 (current_period=3, 3 > 2 is true)
        let cp = get_current_period(0, Some(20), 0, &config).unwrap();
        assert_eq!(cp, 3);
        assert!(cp > 2, "should reveal at block 20");
        assert!(!reveal_window_closed(2, 0, Some(20), 0, &config));

        // Resolve at block 21, one block after the period ends
        assert!(reveal_window_closed(2, 0, Some(21), 0, &config));
    }

    /// In time mode the reveal window closes one week after the period ends
    #[test]
    fn reveal_window_one_week() {
        let config = DecisionConfig::default();
        let genesis_ts = 1_000;
        let (_, end) = calculate_period_boundaries(2, &config, genesis_ts);
        let week = 7 * 24 * 3600;

        assert!(!reveal_window_closed(2, end, None, genesis_ts, &config));
        assert!(!reveal_window_closed(
            2,
            end + week - 1,
            None,
            genesis_ts,
            &config
        ));
        assert!(reveal_window_closed(
            2,
            end + week,
            None,
            genesis_ts,
            &config
        ));
        assert!(!reveal_window_closed(2, end + week, None, 0, &config));
    }

    #[test]
    fn catch_up_closes_reveal() {
        let config = DecisionConfig::testing(TEN);
        let cp = get_current_period(0, Some(40), 0, &config).unwrap();
        assert!(cp > 2);
        assert!(reveal_window_closed(2, 0, Some(40), 0, &config));
    }
}
