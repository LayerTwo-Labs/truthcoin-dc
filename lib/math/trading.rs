//! All LMSR and trading fee calculations. No code outside `lib/math/`
//! should call `lmsr.rs` directly.

use super::lmsr::{LmsrError, LmsrService};
use super::safe_math::{Rounding, SatoshiError, to_sats};
use crate::types::ShareDelta;
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

pub fn validate_lmsr_parameters(
    beta: f64,
    shares: &Array1<i64>,
) -> Result<(), LmsrError> {
    LmsrService::validate_lmsr_parameters(beta, &shares_to_f64(shares))
}

#[derive(Debug, thiserror::Error)]
pub enum TradeQuoteError {
    #[error("trade moves no shares")]
    Empty,
    #[error("trade legs must list each outcome once, in ascending order")]
    NotCanonical,
    #[error("trade leg for outcome {0} moves no shares")]
    ZeroLeg(u32),
    #[error("outcome {outcome_index} is out of range for {outcomes} outcomes")]
    OutcomeOutOfRange { outcome_index: u32, outcomes: usize },
    #[error("trade overflows the shares of outcome {0}")]
    ShareOverflow(u32),
    #[error("trade leaves negative market shares for outcome {0}")]
    NegativeShares(u32),
    #[error("LMSR error: {0:?}")]
    Lmsr(LmsrError),
    #[error(transparent)]
    Sats(#[from] SatoshiError),
}

impl From<LmsrError> for TradeQuoteError {
    fn from(err: LmsrError) -> Self {
        Self::Lmsr(err)
    }
}

/// Each outcome once, in ascending order, within `outcomes`, and moving a
/// non-zero number of shares
pub fn check_deltas(
    deltas: &[ShareDelta],
    outcomes: usize,
) -> Result<(), TradeQuoteError> {
    if deltas.is_empty() {
        return Err(TradeQuoteError::Empty);
    }
    if deltas
        .windows(2)
        .any(|pair| pair[0].outcome_index >= pair[1].outcome_index)
    {
        return Err(TradeQuoteError::NotCanonical);
    }
    for delta in deltas {
        if delta.outcome_index as usize >= outcomes {
            return Err(TradeQuoteError::OutcomeOutOfRange {
                outcome_index: delta.outcome_index,
                outcomes,
            });
        }
        if delta.shares == 0 {
            return Err(TradeQuoteError::ZeroLeg(delta.outcome_index));
        }
    }
    Ok(())
}

/// The cost of moving a market's share vector by a set of legs
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TradeQuote {
    /// LMSR cost of the buy legs, rounded up
    pub buy_cost_sats: u64,
    /// LMSR proceeds of the sell legs, rounded down
    pub sell_gross_sats: u64,
    /// Trading fee on the gross of both, at least [`MIN_TRADING_FEE_SATS`]
    pub fee_sats: u64,
    /// What the trader pays the market, or receives when negative
    pub net_cost_sats: i64,
    pub new_shares: Array1<i64>,
}

impl TradeQuote {
    /// The buy cost and sell proceeds together, the trade's volume
    pub fn gross_sats(&self) -> u64 {
        self.buy_cost_sats.saturating_add(self.sell_gross_sats)
    }
}

/// Price a trade: the buy legs first, then the sell legs, so the buy cost
/// less the sell proceeds is the LMSR cost of the whole move. The fee is
/// charged on the gross of both, once per trade.
pub fn quote_trade(
    shares: &Array1<i64>,
    deltas: &[ShareDelta],
    beta: f64,
    trading_fee_pct: f64,
) -> Result<TradeQuote, TradeQuoteError> {
    check_deltas(deltas, shares.len())?;
    let mut bought = shares.clone();
    let mut new_shares = shares.clone();
    for delta in deltas {
        let index = delta.outcome_index as usize;
        let moved = shares[index]
            .checked_add(delta.shares)
            .ok_or(TradeQuoteError::ShareOverflow(delta.outcome_index))?;
        if moved < 0 {
            return Err(TradeQuoteError::NegativeShares(delta.outcome_index));
        }
        new_shares[index] = moved;
        if delta.shares > 0 {
            bought[index] = moved;
        }
    }
    let buy_cost = calculate_update_cost(shares, &bought, beta)?;
    let sell_gross = calculate_update_cost(&new_shares, &bought, beta)?;
    let buy_cost_sats = to_sats(buy_cost, Rounding::Up)?;
    let sell_gross_sats = to_sats(sell_gross, Rounding::Down)?;
    let fee_sats =
        to_sats((buy_cost + sell_gross) * trading_fee_pct, Rounding::Nearest)?
            .max(MIN_TRADING_FEE_SATS);
    let signed = |sats: u64| {
        i64::try_from(sats).map_err(|_| SatoshiError::Overflow(sats as f64))
    };
    let net_cost_sats = signed(buy_cost_sats)?
        .checked_sub(signed(sell_gross_sats)?)
        .and_then(|net| net.checked_add(signed(fee_sats).ok()?))
        .ok_or(SatoshiError::Overflow(buy_cost + sell_gross))?;
    Ok(TradeQuote {
        buy_cost_sats,
        sell_gross_sats,
        fee_sats,
        net_cost_sats,
        new_shares,
    })
}

/// `liquidity = beta * ln(num_outcomes)`
pub fn calculate_lmsr_liquidity(beta: f64, num_outcomes: usize) -> f64 {
    if num_outcomes <= 1 {
        return 0.0;
    }
    beta * libm::log(num_outcomes as f64)
}

/// `beta = liquidity / ln(num_outcomes)`
pub fn derive_beta_from_liquidity(
    liquidity_sats: u64,
    num_outcomes: usize,
) -> f64 {
    if num_outcomes <= 1 {
        return crate::state::markets::DEFAULT_MARKET_BETA;
    }
    let ln_outcomes = libm::log(num_outcomes as f64);
    if ln_outcomes <= 0.0 {
        return crate::state::markets::DEFAULT_MARKET_BETA;
    }
    liquidity_sats as f64 / ln_outcomes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leg(outcome_index: u32, shares: i64) -> ShareDelta {
        ShareDelta {
            outcome_index,
            shares,
        }
    }

    fn lmsr_cost(from: &[i64], to: &[i64], beta: f64) -> f64 {
        calculate_update_cost(
            &Array1::from_vec(from.to_vec()),
            &Array1::from_vec(to.to_vec()),
            beta,
        )
        .unwrap()
    }

    const BETA: f64 = 10_000.0;

    #[test]
    fn single_leg_cost() {
        let shares = Array1::from_vec(vec![100, 200]);
        let buy = quote_trade(&shares, &[leg(0, 500)], BETA, 0.01).unwrap();
        let cost = lmsr_cost(&[100, 200], &[600, 200], BETA);
        assert_eq!(buy.buy_cost_sats, cost.ceil() as u64);
        assert_eq!(buy.sell_gross_sats, 0);
        assert_eq!(
            buy.fee_sats,
            ((cost * 0.01).round() as u64).max(MIN_TRADING_FEE_SATS)
        );
        assert_eq!(
            buy.net_cost_sats,
            (buy.buy_cost_sats + buy.fee_sats) as i64
        );

        let sell = quote_trade(&shares, &[leg(1, -150)], BETA, 0.01).unwrap();
        let proceeds = lmsr_cost(&[100, 50], &[100, 200], BETA);
        assert_eq!(sell.sell_gross_sats, proceeds.floor() as u64);
        assert_eq!(sell.buy_cost_sats, 0);
        assert_eq!(
            sell.net_cost_sats,
            sell.fee_sats as i64 - sell.sell_gross_sats as i64
        );
    }

    /// A multi-leg buy costs the single LMSR move and pays one fee.
    #[test]
    fn vector_trade_cost() {
        let s45 = Array1::from_vec(vec![12_000, 34_000, 90_000, 13_000]);
        let deltas = [leg(1, 10_000), leg(2, 10_000)];
        let quote = quote_trade(&s45, &deltas, BETA, 0.0).unwrap();
        let whole = lmsr_cost(
            &[12_000, 34_000, 90_000, 13_000],
            &[12_000, 44_000, 100_000, 13_000],
            BETA,
        );
        let sequential = lmsr_cost(
            &[12_000, 34_000, 90_000, 13_000],
            &[12_000, 44_000, 90_000, 13_000],
            BETA,
        ) + lmsr_cost(
            &[12_000, 44_000, 90_000, 13_000],
            &[12_000, 44_000, 100_000, 13_000],
            BETA,
        );
        assert!((whole - sequential).abs() < 1e-6);
        assert_eq!(quote.buy_cost_sats, whole.ceil() as u64);
        assert_eq!(quote.fee_sats, MIN_TRADING_FEE_SATS);
        assert_eq!(
            quote.new_shares.to_vec(),
            vec![12_000, 44_000, 100_000, 13_000]
        );
    }

    /// Buying a complete set costs exactly its share count.
    #[test]
    fn complete_set_cost() {
        let shares = Array1::from_vec(vec![500, 2_000, 0]);
        let k = 7_000;
        let quote =
            quote_trade(&shares, &[leg(0, k), leg(1, k), leg(2, k)], BETA, 0.0)
                .unwrap();
        assert_eq!(quote.buy_cost_sats, k as u64);
    }

    /// A buy-and-sell trade nets the LMSR move and pays fee on gross.
    #[test]
    fn mixed_trade_cost() {
        let shares = Array1::from_vec(vec![20_000, 45_000, 100_000, 13_000]);
        let quote =
            quote_trade(&shares, &[leg(0, -2_000), leg(1, 1_000)], BETA, 0.02)
                .unwrap();
        let net = lmsr_cost(
            &[20_000, 45_000, 100_000, 13_000],
            &[18_000, 46_000, 100_000, 13_000],
            BETA,
        );
        let buy = lmsr_cost(
            &[20_000, 45_000, 100_000, 13_000],
            &[20_000, 46_000, 100_000, 13_000],
            BETA,
        );
        let sell = buy - net;
        assert_eq!(quote.buy_cost_sats, buy.ceil() as u64);
        assert_eq!(quote.sell_gross_sats, sell.floor() as u64);
        assert_eq!(
            quote.fee_sats,
            (((buy + sell) * 0.02).round() as u64).max(MIN_TRADING_FEE_SATS)
        );
        assert!(quote.fee_sats > ((net.abs() * 0.02).round() as u64));
        assert_eq!(
            quote.net_cost_sats,
            quote.buy_cost_sats as i64 - quote.sell_gross_sats as i64
                + quote.fee_sats as i64
        );
    }

    #[test]
    fn trade_legs_must_be_canonical() {
        let shares = Array1::from_vec(vec![100, 100, 100]);
        let quote =
            |deltas: &[ShareDelta]| quote_trade(&shares, deltas, BETA, 0.01);
        assert!(matches!(quote(&[]), Err(TradeQuoteError::Empty)));
        assert!(matches!(
            quote(&[leg(1, 5), leg(0, 5)]),
            Err(TradeQuoteError::NotCanonical)
        ));
        assert!(matches!(
            quote(&[leg(1, 5), leg(1, 5)]),
            Err(TradeQuoteError::NotCanonical)
        ));
        assert!(matches!(
            quote(&[leg(0, 0)]),
            Err(TradeQuoteError::ZeroLeg(0))
        ));
        assert!(matches!(
            quote(&[leg(3, 5)]),
            Err(TradeQuoteError::OutcomeOutOfRange { .. })
        ));
        assert!(matches!(
            quote(&[leg(2, -101)]),
            Err(TradeQuoteError::NegativeShares(2))
        ));
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

    /// Payouts never exceed the treasury when beta is liquidity-base derived.
    #[test]
    fn fixed_redemption_solvent() {
        use ndarray::Array1;

        let mut seed: u64 = 0x1234_5678_9abc_def0;
        let mut rng = |bound: u64| -> u64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % bound.max(1)
        };

        let betas = [50u64, 200, 1443];

        for scenario in 0..4 {
            let n: usize = if scenario == 3 { 4 } else { 2 };
            let winner: usize = if n == 4 { 2 } else { 1 };

            for _trial in 0..3000 {
                let liquidity_base = betas[rng(3) as usize]
                    * (calculate_lmsr_liquidity(1.0, n).round() as u64).max(1);
                let beta = derive_beta_from_liquidity(liquidity_base, n);

                let mut shares: Array1<i64> = Array1::zeros(n);
                let mut treasury: u64 =
                    calculate_treasury(&shares, beta).unwrap().ceil() as u64;

                let steps = 3 + rng(40);
                for _ in 0..steps {
                    let j = if rng(100) < 75 {
                        winner
                    } else {
                        rng(n as u64) as usize
                    };
                    let d = 1 + rng(2000) as i64;
                    let mut new_shares = shares.clone();
                    new_shares[j] += d;
                    let base =
                        calculate_update_cost(&shares, &new_shares, beta)
                            .unwrap();
                    let cost = to_sats(base, Rounding::Up).unwrap();
                    treasury += cost;
                    shares = new_shares;
                }

                let mut p = vec![0.0f64; n];
                match scenario {
                    1 => {
                        p[0] = 0.5;
                        p[1] = 0.5;
                    }
                    2 => {
                        let o = 0.01 + (rng(98) as f64) / 100.0;
                        p[0] = 1.0 - o;
                        p[1] = o;
                    }
                    _ => p[winner] = 1.0,
                }

                let total_payout: u64 = (0..n)
                    .map(|i| (shares[i] as f64 * p[i]).round().max(0.0) as u64)
                    .sum();

                assert!(
                    total_payout <= treasury,
                    "insolvent: scenario {scenario} payout {total_payout} > treasury {treasury}"
                );
            }
        }
    }

    #[test]
    fn test_calculate_update_cost_i64() {
        use ndarray::array;
        let current = array![0i64, 0i64];
        let new = array![10i64, 0i64];
        let cost = calculate_update_cost(&current, &new, 100.0).unwrap();
        assert!(cost > 0.0);
    }

    #[test]
    fn test_beta_math_uses_libm_log() {
        for num_outcomes in [2usize, 3, 14, 48, 256] {
            let expected = libm::log(num_outcomes as f64);
            assert_eq!(
                calculate_lmsr_liquidity(1.0, num_outcomes).to_bits(),
                expected.to_bits()
            );
            assert_eq!(
                derive_beta_from_liquidity(1_000_000, num_outcomes).to_bits(),
                (1_000_000.0_f64 / expected).to_bits()
            );
        }
    }
}
