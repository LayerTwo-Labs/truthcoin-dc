//! Single source of truth for all trading fee and cost calculations.
//!
//! All conversions use `Rounding::Nearest` for fairness and consistency.
//! Conservation is ensured by deriving totals: `total = part1 + part2`.
//!
//! This module consolidates duplicate calculations from:
//! - `lib/validation.rs` (buy/sell validation)
//! - `lib/state/block.rs` (block application)
//! - `app/rpc_server.rs` (RPC endpoints)
//! - `app/gui/markets/buy_shares.rs` (GUI preview)
//! - `app/gui/markets/sell_shares.rs` (GUI preview)

use super::lmsr::LmsrService;
use super::safe_math::{Rounding, SatoshiError, to_sats};
use ndarray::Array1;

/// Minimum trading fee in satoshis.
/// Every trade must pay at least this fee to the market author.
/// This ensures fee UTXOs always have meaningful value for tracking/cleanup.
pub const MIN_TRADING_FEE_SATS: u64 = 1000;

/// Result of buy cost calculation.
///
/// `total_cost_sats` is derived as `base_cost_sats + trading_fee_sats`
/// to ensure exact conservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuyCost {
    pub base_cost_sats: u64,
    pub trading_fee_sats: u64,
    /// DERIVED: base_cost_sats + trading_fee_sats (ensures conservation)
    pub total_cost_sats: u64,
}

/// Result of sell proceeds calculation.
///
/// `net_proceeds_sats` is derived as `gross_proceeds_sats - trading_fee_sats`
/// to ensure exact conservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SellProceeds {
    pub gross_proceeds_sats: u64,
    pub trading_fee_sats: u64,
    /// DERIVED: gross_proceeds_sats - trading_fee_sats (ensures conservation)
    pub net_proceeds_sats: u64,
}

/// Calculate buy cost with `Rounding::Nearest`, derive total for conservation.
///
/// # Arguments
/// * `base_cost_f64` - The raw LMSR cost in satoshis (as f64)
/// * `trading_fee_pct` - Trading fee as a decimal (e.g., 0.005 for 0.5%)
///
/// # Returns
/// `BuyCost` with base, fee, and derived total (base + fee)
///
/// # Example
/// ```ignore
/// let cost = calculate_buy_cost(1000.5, 0.01)?;
/// assert_eq!(cost.total_cost_sats, cost.base_cost_sats + cost.trading_fee_sats);
/// ```
pub fn calculate_buy_cost(
    base_cost_f64: f64,
    trading_fee_pct: f64,
) -> Result<BuyCost, SatoshiError> {
    // Convert base cost using Nearest rounding
    let base_cost_sats = to_sats(base_cost_f64, Rounding::Nearest)?;

    // Calculate fee on the f64 value, convert with Nearest rounding
    // Enforce minimum fee to ensure fee UTXOs always have meaningful value
    let fee_f64 = base_cost_f64 * trading_fee_pct;
    let calculated_fee_sats = to_sats(fee_f64, Rounding::Nearest)?;
    let trading_fee_sats = calculated_fee_sats.max(MIN_TRADING_FEE_SATS);

    // Derive total to ensure conservation: total = base + fee exactly
    let total_cost_sats = base_cost_sats + trading_fee_sats;

    Ok(BuyCost {
        base_cost_sats,
        trading_fee_sats,
        total_cost_sats,
    })
}

/// Calculate sell proceeds with `Rounding::Nearest`, derive net for conservation.
///
/// # Arguments
/// * `gross_proceeds_f64` - The raw LMSR proceeds in satoshis (as f64)
/// * `trading_fee_pct` - Trading fee as a decimal (e.g., 0.005 for 0.5%)
///
/// # Returns
/// `SellProceeds` with gross, fee, and derived net (gross - fee)
///
/// # Errors
/// Returns error if fee would exceed gross proceeds (trade too small)
///
/// # Example
/// ```ignore
/// let proceeds = calculate_sell_proceeds(1000.5, 0.01)?;
/// assert_eq!(proceeds.gross_proceeds_sats, proceeds.net_proceeds_sats + proceeds.trading_fee_sats);
/// ```
pub fn calculate_sell_proceeds(
    gross_proceeds_f64: f64,
    trading_fee_pct: f64,
) -> Result<SellProceeds, SatoshiError> {
    // Convert gross proceeds using Nearest rounding
    let gross_proceeds_sats = to_sats(gross_proceeds_f64, Rounding::Nearest)?;

    // Calculate fee on the f64 value for consistency with buy, convert with Nearest rounding
    // Enforce minimum fee to ensure fee UTXOs always have meaningful value
    let fee_f64 = gross_proceeds_f64 * trading_fee_pct;
    let calculated_fee_sats = to_sats(fee_f64, Rounding::Nearest)?;
    let trading_fee_sats = calculated_fee_sats.max(MIN_TRADING_FEE_SATS);

    // Check that fee doesn't exceed gross proceeds
    if trading_fee_sats > gross_proceeds_sats {
        return Err(SatoshiError::Overflow(gross_proceeds_f64));
    }

    // Derive net to ensure conservation: gross = net + fee exactly
    let net_proceeds_sats = gross_proceeds_sats - trading_fee_sats;

    Ok(SellProceeds {
        gross_proceeds_sats,
        trading_fee_sats,
        net_proceeds_sats,
    })
}

/// Calculate LMSR liquidity from beta.
///
/// Formula: `liquidity = beta * ln(num_outcomes)`
///
/// This is the minimum treasury required for a market with the given beta
/// and number of outcomes.
pub fn calculate_lmsr_liquidity(beta: f64, num_outcomes: usize) -> f64 {
    if num_outcomes <= 1 {
        return 0.0;
    }
    beta * (num_outcomes as f64).ln()
}

/// Derive beta from target liquidity.
///
/// Formula: `beta = liquidity / ln(num_outcomes)`
///
/// # Arguments
/// * `liquidity_sats` - Target liquidity in satoshis
/// * `num_outcomes` - Number of market outcomes
///
/// # Returns
/// The beta value, or a default if calculation is not possible
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

/// Calculate normalized post-trade price for a specific outcome.
///
/// This function calculates what the price will be AFTER a trade is executed,
/// accounting for valid outcome combinations (for multi-decision markets).
///
/// # Arguments
/// * `new_shares` - Share quantities after the trade
/// * `beta` - Market liquidity parameter
/// * `outcome_index` - The state index of the outcome to get price for
/// * `valid_indices` - List of valid state indices (outcomes that haven't been invalidated)
///
/// # Returns
/// Normalized price (0.0 to 1.0) for the specified outcome
pub fn calculate_post_trade_price(
    new_shares: &Array1<f64>,
    beta: f64,
    outcome_index: usize,
    valid_indices: &[usize],
) -> f64 {
    // Calculate raw prices for new share state
    let new_prices = match LmsrService::calculate_prices(new_shares, beta) {
        Ok(prices) => prices,
        Err(_) => return 0.0,
    };

    // Extract prices for valid outcomes only
    let valid_prices: Vec<f64> = valid_indices
        .iter()
        .map(|idx| new_prices.get(*idx).copied().unwrap_or(0.0))
        .collect();

    let valid_sum: f64 = valid_prices.iter().sum();

    if valid_sum <= 0.0 {
        return 0.0;
    }

    // Find the display index for our outcome in the valid indices
    let display_idx =
        valid_indices.iter().position(|idx| *idx == outcome_index);

    // Return normalized price
    display_idx
        .and_then(|di| valid_prices.get(di).copied())
        .unwrap_or(0.0)
        / valid_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buy_cost_uses_nearest_rounding() {
        // Use large enough values so percentage fee exceeds minimum
        // 1_000_000.4 should round to 1_000_000, 1_000_000.5 to 1_000_001
        let cost1 = calculate_buy_cost(1_000_000.4, 0.01).unwrap();
        assert_eq!(cost1.base_cost_sats, 1_000_000);

        let cost2 = calculate_buy_cost(1_000_000.5, 0.01).unwrap();
        assert_eq!(cost2.base_cost_sats, 1_000_001);
    }

    #[test]
    fn test_buy_cost_conservation() {
        let cost = calculate_buy_cost(100_000.0, 0.01).unwrap();
        // Conservation: total = base + fee
        assert_eq!(
            cost.total_cost_sats,
            cost.base_cost_sats + cost.trading_fee_sats
        );
    }

    #[test]
    fn test_sell_proceeds_uses_nearest_rounding() {
        // Use large enough values so percentage fee exceeds minimum
        let proceeds1 = calculate_sell_proceeds(1_000_000.4, 0.01).unwrap();
        assert_eq!(proceeds1.gross_proceeds_sats, 1_000_000);

        let proceeds2 = calculate_sell_proceeds(1_000_000.5, 0.01).unwrap();
        assert_eq!(proceeds2.gross_proceeds_sats, 1_000_001);
    }

    #[test]
    fn test_sell_proceeds_conservation() {
        let proceeds = calculate_sell_proceeds(100_000.0, 0.01).unwrap();
        // Conservation: gross = net + fee
        assert_eq!(
            proceeds.gross_proceeds_sats,
            proceeds.net_proceeds_sats + proceeds.trading_fee_sats
        );
    }

    #[test]
    fn test_sell_proceeds_fee_exceeds_gross() {
        // Small proceeds that can't cover minimum fee should error
        let result = calculate_sell_proceeds(500.0, 0.01); // Min fee 1000 > gross 500
        assert!(result.is_err());
    }

    #[test]
    fn test_minimum_trading_fee_enforced_buy() {
        // Small trade where percentage fee < minimum
        let cost = calculate_buy_cost(100.0, 0.01).unwrap(); // 1% of 100 = 1 sat
        // Should enforce minimum fee of 1000 sats
        assert_eq!(cost.trading_fee_sats, MIN_TRADING_FEE_SATS);
        assert_eq!(cost.total_cost_sats, 100 + MIN_TRADING_FEE_SATS);
    }

    #[test]
    fn test_minimum_trading_fee_enforced_sell() {
        // Trade where percentage fee < minimum but gross can cover it
        let proceeds = calculate_sell_proceeds(10_000.0, 0.01).unwrap(); // 1% of 10k = 100 sat
        // Should enforce minimum fee of 1000 sats
        assert_eq!(proceeds.trading_fee_sats, MIN_TRADING_FEE_SATS);
        assert_eq!(proceeds.net_proceeds_sats, 10_000 - MIN_TRADING_FEE_SATS);
    }

    #[test]
    fn test_buy_sell_roundtrip_fair() {
        // Test that buying and selling the same amount doesn't systematically favor either party
        // Use large enough values so percentage fee exceeds minimum
        let base_cost = 1_000_000.0;
        let fee_pct = 0.01;

        let buy = calculate_buy_cost(base_cost, fee_pct).unwrap();
        let sell = calculate_sell_proceeds(base_cost, fee_pct).unwrap();

        // Both should use same rounding, so fees should be equal
        assert_eq!(buy.trading_fee_sats, sell.trading_fee_sats);
    }

    #[test]
    fn test_beta_liquidity_roundtrip() {
        let original_beta = 100_000_000.0;
        let num_outcomes = 4;

        let liquidity = calculate_lmsr_liquidity(original_beta, num_outcomes);
        let derived_beta =
            derive_beta_from_liquidity(liquidity as u64, num_outcomes);

        // Should be approximately equal (within 1 sat of rounding)
        assert!((original_beta - derived_beta).abs() < 1.0);
    }

    #[test]
    fn test_liquidity_edge_cases() {
        // Single outcome should return 0
        assert_eq!(calculate_lmsr_liquidity(100.0, 1), 0.0);
        assert_eq!(calculate_lmsr_liquidity(100.0, 0), 0.0);

        // Two outcomes: liquidity = beta * ln(2) ≈ beta * 0.693
        let liq = calculate_lmsr_liquidity(100.0, 2);
        assert!((liq - 69.3).abs() < 0.5);
    }
}
