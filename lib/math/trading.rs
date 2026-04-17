//! All LMSR and trading fee calculations. No code outside `lib/math/`
//! should call `lmsr.rs` directly.

use super::lmsr::{LmsrError, LmsrService};
use super::safe_math::{Rounding, SatoshiError, to_sats};
use ndarray::Array1;

/// Minimum trading fee in satoshis.
/// Every trade must pay at least this fee to the market author.
/// This ensures fee UTXOs always have meaningful value for tracking/cleanup.
pub const MIN_TRADING_FEE_SATS: u64 = 1000;

pub const TRADE_MINER_FEE_SATS: u64 = 1000;

fn shares_to_f64(shares: &Array1<i64>) -> Array1<f64> {
    shares.mapv(|s| s as f64)
}

pub fn calculate_prices(
    shares: &Array1<i64>,
    beta: f64,
) -> Result<Array1<f64>, LmsrError> {
    LmsrService::calculate_prices(&shares_to_f64(shares), beta)
}

pub fn calculate_update_cost(
    current_shares: &Array1<i64>,
    new_shares: &Array1<i64>,
    beta: f64,
) -> Result<f64, LmsrError> {
    LmsrService::calculate_update_cost(
        &shares_to_f64(current_shares),
        &shares_to_f64(new_shares),
        beta,
    )
}

pub fn calculate_treasury(
    shares: &Array1<i64>,
    beta: f64,
) -> Result<f64, LmsrError> {
    LmsrService::calculate_treasury(&shares_to_f64(shares), beta)
}

pub fn calculate_amp_b_cost(
    shares: &Array1<i64>,
    current_b: f64,
    new_b: f64,
) -> Result<f64, LmsrError> {
    let shares_f64 = shares_to_f64(shares);
    let current_cost = LmsrService::calculate_treasury(&shares_f64, current_b)?;
    let new_cost = LmsrService::calculate_treasury(&shares_f64, new_b)?;
    Ok(new_cost - current_cost)
}

pub fn validate_lmsr_parameters(
    beta: f64,
    shares: &Array1<i64>,
) -> Result<(), LmsrError> {
    LmsrService::validate_lmsr_parameters(beta, &shares_to_f64(shares))
}

/// `total_cost_sats` = `base_cost_sats + trading_fee_sats`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuyCost {
    pub base_cost_sats: u64,
    pub trading_fee_sats: u64,
    pub total_cost_sats: u64,
}

/// `net_proceeds_sats` = `gross_proceeds_sats - trading_fee_sats`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SellProceeds {
    pub gross_proceeds_sats: u64,
    pub trading_fee_sats: u64,
    pub net_proceeds_sats: u64,
}

/// Calculate buy cost, derive total for conservation.
pub fn calculate_buy_cost(
    base_cost_f64: f64,
    trading_fee_pct: f64,
) -> Result<BuyCost, SatoshiError> {
    let base_cost_sats = to_sats(base_cost_f64, Rounding::Nearest)?;

    let fee_f64 = base_cost_f64 * trading_fee_pct;
    let calculated_fee_sats = to_sats(fee_f64, Rounding::Nearest)?;
    let trading_fee_sats = calculated_fee_sats.max(MIN_TRADING_FEE_SATS);

    let total_cost_sats = base_cost_sats + trading_fee_sats;

    Ok(BuyCost {
        base_cost_sats,
        trading_fee_sats,
        total_cost_sats,
    })
}

/// Calculate sell proceeds, derive net for conservation.
pub fn calculate_sell_proceeds(
    gross_proceeds_f64: f64,
    trading_fee_pct: f64,
) -> Result<SellProceeds, SatoshiError> {
    let gross_proceeds_sats = to_sats(gross_proceeds_f64, Rounding::Nearest)?;

    let fee_f64 = gross_proceeds_f64 * trading_fee_pct;
    let calculated_fee_sats = to_sats(fee_f64, Rounding::Nearest)?;
    let trading_fee_sats = calculated_fee_sats.max(MIN_TRADING_FEE_SATS);

    if trading_fee_sats > gross_proceeds_sats {
        return Err(SatoshiError::Overflow(gross_proceeds_f64));
    }

    let net_proceeds_sats = gross_proceeds_sats - trading_fee_sats;

    Ok(SellProceeds {
        gross_proceeds_sats,
        trading_fee_sats,
        net_proceeds_sats,
    })
}

/// `liquidity = beta * ln(num_outcomes)`
pub fn calculate_lmsr_liquidity(beta: f64, num_outcomes: usize) -> f64 {
    if num_outcomes <= 1 {
        return 0.0;
    }
    beta * (num_outcomes as f64).ln()
}

/// `beta = liquidity / ln(num_outcomes)`
pub fn derive_beta_from_liquidity(
    liquidity_sats: u64,
    num_outcomes: usize,
) -> f64 {
    if num_outcomes <= 1 {
        return crate::state::markets::DEFAULT_MARKET_BETA;
    }
    let ln_outcomes = (num_outcomes as f64).ln();
    if ln_outcomes <= 0.0 {
        return crate::state::markets::DEFAULT_MARKET_BETA;
    }
    liquidity_sats as f64 / ln_outcomes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buy_cost_uses_nearest_rounding() {
        let cost1 = calculate_buy_cost(1_000_000.4, 0.01).unwrap();
        assert_eq!(cost1.base_cost_sats, 1_000_000);

        let cost2 = calculate_buy_cost(1_000_000.5, 0.01).unwrap();
        assert_eq!(cost2.base_cost_sats, 1_000_001);
    }

    #[test]
    fn test_buy_cost_conservation() {
        let cost = calculate_buy_cost(100_000.0, 0.01).unwrap();
        assert_eq!(
            cost.total_cost_sats,
            cost.base_cost_sats + cost.trading_fee_sats
        );
    }

    #[test]
    fn test_sell_proceeds_uses_nearest_rounding() {
        let proceeds1 = calculate_sell_proceeds(1_000_000.4, 0.01).unwrap();
        assert_eq!(proceeds1.gross_proceeds_sats, 1_000_000);

        let proceeds2 = calculate_sell_proceeds(1_000_000.5, 0.01).unwrap();
        assert_eq!(proceeds2.gross_proceeds_sats, 1_000_001);
    }

    #[test]
    fn test_sell_proceeds_conservation() {
        let proceeds = calculate_sell_proceeds(100_000.0, 0.01).unwrap();
        assert_eq!(
            proceeds.gross_proceeds_sats,
            proceeds.net_proceeds_sats + proceeds.trading_fee_sats
        );
    }

    #[test]
    fn test_sell_proceeds_fee_exceeds_gross() {
        let result = calculate_sell_proceeds(500.0, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_minimum_trading_fee_enforced_buy() {
        let cost = calculate_buy_cost(100.0, 0.01).unwrap();
        assert_eq!(cost.trading_fee_sats, MIN_TRADING_FEE_SATS);
        assert_eq!(cost.total_cost_sats, 100 + MIN_TRADING_FEE_SATS);
    }

    #[test]
    fn test_minimum_trading_fee_enforced_sell() {
        let proceeds = calculate_sell_proceeds(10_000.0, 0.01).unwrap();
        assert_eq!(proceeds.trading_fee_sats, MIN_TRADING_FEE_SATS);
        assert_eq!(proceeds.net_proceeds_sats, 10_000 - MIN_TRADING_FEE_SATS);
    }

    #[test]
    fn test_buy_sell_roundtrip_fair() {
        let base_cost = 1_000_000.0;
        let fee_pct = 0.01;

        let buy = calculate_buy_cost(base_cost, fee_pct).unwrap();
        let sell = calculate_sell_proceeds(base_cost, fee_pct).unwrap();

        assert_eq!(buy.trading_fee_sats, sell.trading_fee_sats);
    }

    #[test]
    fn test_beta_liquidity_roundtrip() {
        let original_beta = 100_000_000.0;
        let num_outcomes = 4;

        let liquidity = calculate_lmsr_liquidity(original_beta, num_outcomes);
        let derived_beta =
            derive_beta_from_liquidity(liquidity as u64, num_outcomes);

        assert!((original_beta - derived_beta).abs() < 1.0);
    }

    #[test]
    fn test_liquidity_edge_cases() {
        assert_eq!(calculate_lmsr_liquidity(100.0, 1), 0.0);
        assert_eq!(calculate_lmsr_liquidity(100.0, 0), 0.0);

        let liq = calculate_lmsr_liquidity(100.0, 2);
        assert!((liq - 69.3).abs() < 0.5);
    }

    #[test]
    fn test_calculate_prices_i64() {
        use ndarray::array;
        let shares = array![0i64, 0i64];
        let prices = calculate_prices(&shares, 100.0).unwrap();
        assert!((prices[0] - 0.5).abs() < 1e-6);
        assert!((prices[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_update_cost_i64() {
        use ndarray::array;
        let current = array![0i64, 0i64];
        let new = array![10i64, 0i64];
        let cost = calculate_update_cost(&current, &new, 100.0).unwrap();
        assert!(cost > 0.0);
    }
}
