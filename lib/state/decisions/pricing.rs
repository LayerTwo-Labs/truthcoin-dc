use serde::{Deserialize, Serialize};

use crate::math::decisions as math;

use super::types::DecisionConfig;

pub use crate::math::decisions::{
    GENESIS_P_PERIOD_SATS, ListingFeeError, SLOTS_PER_PERIOD_MAX,
    SLOTS_PER_TIER, SLOTS_PER_TIER_PER_MINT, TIER_COUNT, TIER_MULTIPLIERS_DEN,
    TIER_MULTIPLIERS_NUM, fee_for_index, slot_price, slot_unlocked,
    tier_for_index,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PeriodPricing {
    pub p_period: u64,
    pub p_floor: u64,
    pub mints: u64,
    pub claimed: u64,
    pub last_reprice_block: u32,
    pub window_open_claimed: u64,
    pub window_open_mints: u64,
}

impl PeriodPricing {
    pub fn new_seeded(p_seed: u64, block_height: u32) -> Self {
        Self {
            p_period: p_seed,
            p_floor: p_seed / 16,
            mints: 1,
            claimed: 0,
            last_reprice_block: block_height,
            window_open_claimed: 0,
            window_open_mints: 1,
        }
    }

    /// Total slots minted into this period so far (across all 5 tiers).
    pub fn period_capacity(&self) -> u64 {
        self.mints * SLOTS_PER_TIER_PER_MINT * TIER_COUNT as u64
    }
}

pub fn reprice_interval(cfg: &DecisionConfig) -> u32 {
    let bp = math::blocks_per_period(cfg.is_blocks, cfg.quantity);
    math::reprice_interval(bp)
}

pub fn apply_reprice_with_floor(
    pricing: &mut PeriodPricing,
    block_height: u32,
) {
    if pricing.last_reprice_block == block_height {
        return;
    }
    let sales = pricing.claimed.saturating_sub(pricing.window_open_claimed);
    let available_at_open = pricing
        .window_open_mints
        .saturating_mul(SLOTS_PER_TIER_PER_MINT * TIER_COUNT as u64)
        .saturating_sub(pricing.window_open_claimed);
    let new_p = math::reprice(pricing.p_period, sales, available_at_open)
        .max(pricing.p_floor);
    pricing.p_period = new_p;
    pricing.last_reprice_block = block_height;
    pricing.window_open_claimed = pricing.claimed;
    pricing.window_open_mints = pricing.mints;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reprice_interval_signet() {
        let cfg = DecisionConfig {
            is_blocks: false,
            quantity: 86_400,
        };
        assert_eq!(reprice_interval(&cfg), 28);
    }

    #[test]
    fn reprice_interval_production() {
        let cfg = DecisionConfig {
            is_blocks: false,
            quantity: (3600 * 24 * 91) as u32,
        };
        assert_eq!(reprice_interval(&cfg), 2_620);
    }

    #[test]
    fn reprice_interval_blocks_mode() {
        let cfg = DecisionConfig {
            is_blocks: true,
            quantity: 144,
        };
        assert_eq!(reprice_interval(&cfg), 28);
    }

    #[test]
    fn period_pricing_seeded() {
        let p = PeriodPricing::new_seeded(8_000, 100);
        assert_eq!(p.p_period, 8_000);
        assert_eq!(p.p_floor, 500);
        assert_eq!(p.mints, 1);
        assert_eq!(p.period_capacity(), 25);
        assert_eq!(p.claimed, 0);
    }

    #[test]
    fn floor_clamps_halving() {
        let mut p = PeriodPricing::new_seeded(10_000, 0);
        p.p_floor = 1_000;
        for h in 1..=10 {
            apply_reprice_with_floor(&mut p, h);
        }
        assert!(p.p_period >= p.p_floor);
        assert_eq!(p.p_period, 1_000);
    }

    #[test]
    fn apply_reprice_idempotent_at_open_block() {
        let mut p = PeriodPricing::new_seeded(8_000, 50);
        apply_reprice_with_floor(&mut p, 50);
        assert_eq!(p.p_period, 8_000);
    }

    #[test]
    fn reprice_uses_claimed_counter() {
        let mut p = PeriodPricing::new_seeded(10_000, 0);
        // Sell all 25 unlocked slots in window.
        p.claimed = 25;
        apply_reprice_with_floor(&mut p, 24);
        assert_eq!(p.p_period, 20_000);
        assert_eq!(p.window_open_claimed, 25);
    }
}
