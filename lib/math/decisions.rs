//! Decision listing-fee math: deterministic ID→tier mapping, tier pricing,
//! exponential reprice curve. Pure functions over primitive inputs — no
//! `state/` imports.
//!
//! Each period's 500-slot index space is partitioned into 5 fixed tiers of
//! 100 slots each. The decision_index field of `DecisionId` directly encodes
//! the tier:
//!
//! ```text
//!   tier 0 (0.25×p): decision_index   0..=99
//!   tier 1 (0.50×p): decision_index 100..=199
//!   tier 2 (1.00×p): decision_index 200..=299
//!   tier 3 (2.00×p): decision_index 300..=399
//!   tier 4 (4.00×p): decision_index 400..=499
//! ```
//!
//! A slot is "unlocked" iff `(decision_index % 100) < mints × 5`. Each period
//! transition increases `mints` by 1, unlocking 5 more slots in each tier.

pub const TIER_COUNT: usize = 5;
pub const SLOTS_PER_TIER_PER_MINT: u64 = 5;
pub const SLOTS_PER_TIER: u64 = 100;
pub const SLOTS_PER_PERIOD_MAX: u64 = SLOTS_PER_TIER * TIER_COUNT as u64;
pub const MID_PERIOD_REPRICES: u32 = 5;
pub const TIER_MULTIPLIERS_NUM: [u64; TIER_COUNT] = [1, 2, 4, 8, 16];
pub const TIER_MULTIPLIERS_DEN: u64 = 4;
pub const GENESIS_P_PERIOD_SATS: u64 = 10_000;
pub const SECONDS_PER_BLOCK: u64 = 600;

#[derive(Clone, Debug, thiserror::Error)]
pub enum ListingFeeError {
    #[error("decision_index {0} is out of range (max 499)")]
    IndexOutOfRange(u32),
    #[error(
        "slot at decision_index {decision_index} is not yet minted (mints={mints})"
    )]
    SlotNotUnlocked { decision_index: u32, mints: u64 },
    #[error("listing fee arithmetic overflow")]
    Overflow,
}

pub fn reprice_interval(blocks_per_period: u32) -> u32 {
    (blocks_per_period / MID_PERIOD_REPRICES).max(1)
}

pub fn blocks_per_period(is_blocks: bool, quantity: u32) -> u32 {
    if is_blocks {
        quantity
    } else {
        ((quantity as u64) / SECONDS_PER_BLOCK) as u32
    }
}

/// Map a decision_index to its fixed tier (0..=4). Returns
/// `IndexOutOfRange` if the index falls outside the period's slot space.
pub fn tier_for_index(decision_index: u32) -> Result<usize, ListingFeeError> {
    if (decision_index as u64) >= SLOTS_PER_PERIOD_MAX {
        return Err(ListingFeeError::IndexOutOfRange(decision_index));
    }
    Ok((decision_index as u64 / SLOTS_PER_TIER) as usize)
}

/// Position of a slot within its tier (0..=99). Together with
/// `tier_for_index` this identifies the slot in the tier-grid view.
pub fn position_in_tier(decision_index: u32) -> u64 {
    (decision_index as u64) % SLOTS_PER_TIER
}

/// True iff the slot at `decision_index` has been unlocked by the current
/// `mints` count. Each mint event unlocks 5 more positions in every tier.
pub fn slot_unlocked(decision_index: u32, mints: u64) -> bool {
    if (decision_index as u64) >= SLOTS_PER_PERIOD_MAX {
        return false;
    }
    position_in_tier(decision_index) < mints * SLOTS_PER_TIER_PER_MINT
}

pub fn slot_price(p_period: u64, tier: usize) -> u64 {
    let num = TIER_MULTIPLIERS_NUM[tier];
    p_period
        .saturating_mul(num)
        .checked_div(TIER_MULTIPLIERS_DEN)
        .unwrap_or(u64::MAX)
}

pub fn tier_prices(p_period: u64) -> [u64; TIER_COUNT] {
    let mut out = [0u64; TIER_COUNT];
    for (tier, slot) in out.iter_mut().enumerate() {
        *slot = slot_price(p_period, tier);
    }
    out
}

/// Fee for claiming a single slot at `decision_index` given the period's
/// current `p_period` and `mints`. Validates that the slot is unlocked.
pub fn fee_for_index(
    p_period: u64,
    mints: u64,
    decision_index: u32,
) -> Result<u64, ListingFeeError> {
    let tier = tier_for_index(decision_index)?;
    if !slot_unlocked(decision_index, mints) {
        return Err(ListingFeeError::SlotNotUnlocked {
            decision_index,
            mints,
        });
    }
    Ok(slot_price(p_period, tier))
}

pub fn reprice(p_old: u64, sales: u64, available_at_open: u64) -> u64 {
    if available_at_open == 0 {
        return p_old;
    }
    if sales == 0 {
        return p_old / 2;
    }
    if sales == available_at_open {
        return p_old.saturating_mul(2);
    }
    if sales.saturating_mul(2) == available_at_open {
        return p_old;
    }
    let q32 = 1i128 << 32;
    let e_fp = (sales as i128 * 2 * q32 / available_at_open as i128) - q32;
    let factor_fp = pow2_fixed_q32(e_fp);
    ((p_old as u128 * factor_fp) >> 32) as u64
}

fn pow2_fixed_q32(e_fp: i128) -> u128 {
    let q32: i128 = 1i128 << 32;
    let exp_int = e_fp.div_euclid(q32);
    let frac = e_fp.rem_euclid(q32) as u128;

    let frac_factor_q62 = pow2_frac_q62(frac);
    let mut result_q32 = frac_factor_q62 >> 30;

    if exp_int >= 0 {
        if exp_int >= 64 {
            return u128::MAX;
        }
        result_q32 <<= exp_int as u32;
    } else {
        let shift = (-exp_int) as u32;
        if shift >= 64 {
            return 0;
        }
        result_q32 >>= shift;
    }
    result_q32
}

fn pow2_frac_q62(frac_q32: u128) -> u128 {
    let q62: u128 = 1u128 << 62;
    let frac_q62 = frac_q32 << 30;
    let ln2_q62: u128 = 0x_2C5C_85FD_F473_DE6Bu128;
    let x_q62 = (frac_q62 * ln2_q62) >> 62;

    let mut term = q62;
    let mut sum = q62;
    for k in 1u128..12 {
        term = (term * x_q62) >> 62;
        term /= k;
        sum += term;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_price_anchors() {
        let p = 12_400u64;
        assert_eq!(slot_price(p, 0), 3_100);
        assert_eq!(slot_price(p, 1), 6_200);
        assert_eq!(slot_price(p, 2), 12_400);
        assert_eq!(slot_price(p, 3), 24_800);
        assert_eq!(slot_price(p, 4), 49_600);
    }

    #[test]
    fn tier_for_index_boundaries() {
        assert_eq!(tier_for_index(0).unwrap(), 0);
        assert_eq!(tier_for_index(99).unwrap(), 0);
        assert_eq!(tier_for_index(100).unwrap(), 1);
        assert_eq!(tier_for_index(199).unwrap(), 1);
        assert_eq!(tier_for_index(200).unwrap(), 2);
        assert_eq!(tier_for_index(399).unwrap(), 3);
        assert_eq!(tier_for_index(400).unwrap(), 4);
        assert_eq!(tier_for_index(499).unwrap(), 4);
        assert!(matches!(
            tier_for_index(500),
            Err(ListingFeeError::IndexOutOfRange(500))
        ));
        assert!(matches!(
            tier_for_index(65535),
            Err(ListingFeeError::IndexOutOfRange(_))
        ));
    }

    #[test]
    fn slot_unlocked_progression() {
        // mints=1: only positions 0..=4 of each tier unlocked
        for tier in 0..5u32 {
            for pos in 0..5u32 {
                assert!(slot_unlocked(tier * 100 + pos, 1));
            }
            for pos in 5..100u32 {
                assert!(!slot_unlocked(tier * 100 + pos, 1));
            }
        }
        // mints=20: all 500 slots unlocked
        for idx in 0..500u32 {
            assert!(slot_unlocked(idx, 20));
        }
        // out-of-range never unlocked
        assert!(!slot_unlocked(500, 20));
    }

    #[test]
    fn fee_for_index_basic() {
        // mints=1, p=10_000: tier 0 = 2_500, tier 4 = 40_000
        assert_eq!(fee_for_index(10_000, 1, 0).unwrap(), 2_500);
        assert_eq!(fee_for_index(10_000, 1, 4).unwrap(), 2_500);
        assert_eq!(fee_for_index(10_000, 1, 100).unwrap(), 5_000);
        assert_eq!(fee_for_index(10_000, 1, 400).unwrap(), 40_000);
    }

    #[test]
    fn fee_for_index_locked_rejected() {
        // mints=1: position 5 not yet unlocked
        assert!(matches!(
            fee_for_index(10_000, 1, 5),
            Err(ListingFeeError::SlotNotUnlocked { .. })
        ));
        assert!(matches!(
            fee_for_index(10_000, 1, 99),
            Err(ListingFeeError::SlotNotUnlocked { .. })
        ));
        // mints=2: position 5..9 now unlocked
        assert_eq!(fee_for_index(10_000, 2, 5).unwrap(), 2_500);
        assert_eq!(fee_for_index(10_000, 2, 9).unwrap(), 2_500);
        assert!(matches!(
            fee_for_index(10_000, 2, 10),
            Err(ListingFeeError::SlotNotUnlocked { .. })
        ));
    }

    #[test]
    fn fee_for_index_out_of_range() {
        assert!(matches!(
            fee_for_index(10_000, 20, 500),
            Err(ListingFeeError::IndexOutOfRange(500))
        ));
    }

    #[test]
    fn reprice_anchor_zero_sales() {
        assert_eq!(reprice(10_000, 0, 25), 5_000);
        assert_eq!(reprice(7, 0, 100), 3);
    }

    #[test]
    fn reprice_anchor_half_sales() {
        assert_eq!(reprice(10_000, 12, 24), 10_000);
        assert_eq!(reprice(10_000, 50, 100), 10_000);
    }

    #[test]
    fn reprice_anchor_full_sales() {
        assert_eq!(reprice(10_000, 25, 25), 20_000);
    }

    #[test]
    fn reprice_zero_available_no_change() {
        assert_eq!(reprice(10_000, 0, 0), 10_000);
    }

    #[test]
    fn reprice_quarter_sales_within_one_sat() {
        let p = 1_000_000u64;
        let r = reprice(p, 25, 100);
        let expected = (p as f64 * 2f64.powf(2.0 * 0.25 - 1.0)) as u64;
        let diff = (r as i64 - expected as i64).unsigned_abs();
        assert!(diff <= 1, "got {r}, expected {expected}");
    }

    #[test]
    fn reprice_three_quarter_sales_within_one_sat() {
        let p = 1_000_000u64;
        let r = reprice(p, 75, 100);
        let expected = (p as f64 * 2f64.powf(2.0 * 0.75 - 1.0)) as u64;
        let diff = (r as i64 - expected as i64).unsigned_abs();
        assert!(diff <= 1, "got {r}, expected {expected}");
    }

    #[test]
    fn reprice_log_symmetry() {
        let p = 1_000_000u64;
        for k in 1..100 {
            let up = reprice(p, k, 100);
            let down = reprice(p, 100 - k, 100);
            let prod = up as u128 * down as u128;
            let target = p as u128 * p as u128;
            let diff = prod.abs_diff(target);
            let tol = target / 1_000;
            assert!(
                diff <= tol,
                "k={k} up={up} down={down} prod={prod} target={target}"
            );
        }
    }

    #[test]
    fn interval_signet() {
        let bp = blocks_per_period(false, 86_400);
        assert_eq!(bp, 144);
        assert_eq!(reprice_interval(bp), 28);
    }

    #[test]
    fn interval_production() {
        let bp = blocks_per_period(false, (3600 * 24 * 91) as u32);
        assert_eq!(bp, 13_104);
        assert_eq!(reprice_interval(bp), 2_620);
    }

    #[test]
    fn interval_blocks_mode() {
        assert_eq!(reprice_interval(blocks_per_period(true, 144)), 28);
        assert_eq!(reprice_interval(blocks_per_period(true, 4)), 1);
    }
}
