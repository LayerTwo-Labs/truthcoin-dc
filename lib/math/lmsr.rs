/// Optimized LMSR Mathematical Operations using ndarray for Bitcoin Hivemind Sidechain
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::fmt;

use super::safe_math::{self, BasisPoints, Rounding};

const SATOSHI_PRECISION: f64 = 100_000_000.0;
const LMSR_PRECISION: f64 = 1e-8;
const MAX_BETA: f64 = 1e12;
const MIN_BETA: f64 = 1e-6;

/// Minimum number of outcomes for a valid market (binary)
pub const MIN_OUTCOMES: usize = 2;
/// Maximum number of outcomes
pub const MAX_OUTCOMES: usize = 256;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LmsrError {
    InvalidBeta {
        beta: f64,
        min: f64,
        max: f64,
    },
    ShareQuantityOverflow,
    InvalidCostCalculation,
    InsufficientTreasury {
        required: u64,
        available: u64,
    },
    InvalidOutcomeCount {
        count: usize,
        min: usize,
        max: usize,
    },
    PrecisionLoss,
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidFeeRate {
        rate: f64,
        reason: String,
    },
}

impl fmt::Display for LmsrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LmsrError::InvalidBeta { beta, min, max } => {
                write!(f, "Beta {beta} outside valid range [{min}, {max}]")
            }
            LmsrError::ShareQuantityOverflow => {
                write!(f, "Share quantity overflow")
            }
            LmsrError::InvalidCostCalculation => {
                write!(f, "Invalid cost calculation")
            }
            LmsrError::InsufficientTreasury {
                required,
                available,
            } => write!(
                f,
                "Insufficient treasury: required {required}, available {available}"
            ),
            LmsrError::InvalidOutcomeCount { count, min, max } => write!(
                f,
                "Invalid outcome count {count}: must be between {min} and {max}"
            ),
            LmsrError::PrecisionLoss => write!(f, "Numerical precision loss"),
            LmsrError::DimensionMismatch { expected, actual } => write!(
                f,
                "Dimension mismatch: expected {expected}, got {actual}"
            ),
            LmsrError::InvalidFeeRate { rate, reason } => {
                write!(f, "Invalid fee rate {rate}: {reason}")
            }
        }
    }
}

impl std::error::Error for LmsrError {}

#[derive(Debug, Clone, PartialEq)]
pub struct LmsrState {
    pub beta: f64,
    pub shares: Array1<f64>,
    pub treasury_balance: u64,
    pub trading_fee: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LmsrTradeResult {
    pub cost_satoshis: i64,
    pub fee_satoshis: u64,
    pub new_shares: Array1<f64>,
    pub new_treasury_balance: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LmsrPriceInfo {
    pub prices: Array1<f64>,
    pub marginal_costs: Array1<u64>,
}

pub struct LmsrContext<'a> {
    lmsr: &'a Lmsr,
    state: &'a LmsrState,
    cached_prices: Array1<f64>,
    cached_base_cost: f64,
}

impl<'a> LmsrContext<'a> {
    pub fn new(
        lmsr: &'a Lmsr,
        state: &'a LmsrState,
    ) -> Result<Self, LmsrError> {
        let prices = lmsr.validate_state(state)?;
        let base_cost = lmsr.cost_function(state.beta, &state.shares.view())?;

        Ok(Self {
            lmsr,
            state,
            cached_prices: prices,
            cached_base_cost: base_cost,
        })
    }

    pub fn prices(&self) -> &Array1<f64> {
        &self.cached_prices
    }

    pub fn calculate_buy_cost_fast(
        &self,
        outcome: usize,
        shares_to_buy: f64,
    ) -> Result<LmsrTradeResult, LmsrError> {
        self.lmsr.calculate_buy_cost_with_base(
            self.state,
            outcome,
            shares_to_buy,
            self.cached_base_cost,
        )
    }
}

pub struct Lmsr {
    max_outcomes: usize,
}

impl Lmsr {
    pub fn new(max_outcomes: usize) -> Self {
        Self { max_outcomes }
    }

    /// Core LMSR calculation: C = b × ln(Σ exp(qᵢ/b))
    fn calculate_core_exp_ln(
        &self,
        beta: f64,
        shares: &ArrayView1<f64>,
    ) -> Result<(f64, f64, Array1<f64>), LmsrError> {
        if beta <= MIN_BETA || beta >= MAX_BETA {
            return Err(LmsrError::InvalidBeta {
                beta,
                min: MIN_BETA,
                max: MAX_BETA,
            });
        }

        if shares.is_empty() {
            return Ok((0.0, 0.0, Array1::zeros(0)));
        }

        // Shift by max for numerical stability (prevents exp overflow)
        let max_share = shares.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let shifted_shares = (shares - max_share) / beta;
        let exp_shares = shifted_shares.mapv(f64::exp);

        if !exp_shares.iter().all(|x| x.is_finite()) {
            return Err(LmsrError::ShareQuantityOverflow);
        }

        let sum_exp = exp_shares.sum();

        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            return Err(LmsrError::InvalidCostCalculation);
        }

        // Hanson LMSR cost function: C = b × ln(Σ exp(qᵢ/b))
        let cost = beta * (sum_exp.ln() + max_share / beta);

        if !cost.is_finite() {
            return Err(LmsrError::InvalidCostCalculation);
        }

        Ok((cost, sum_exp, exp_shares))
    }

    /// Calculate the LMSR cost function value.
    /// Works for any market size from 2 to 256 outcomes.
    pub fn cost_function(
        &self,
        beta: f64,
        shares: &ArrayView1<f64>,
    ) -> Result<f64, LmsrError> {
        let (cost, _, _) = self.calculate_core_exp_ln(beta, shares)?;
        Ok(cost)
    }

    /// Calculate outcome prices (probability distribution).
    /// Works for any market size from 2 to 256 outcomes.
    pub fn calculate_prices(
        &self,
        beta: f64,
        shares: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, LmsrError> {
        let (_, sum_exp, exp_shares) =
            self.calculate_core_exp_ln(beta, shares)?;

        if sum_exp == 0.0 {
            return Ok(Array1::zeros(0));
        }

        let prices = exp_shares / sum_exp;
        let price_sum: f64 = prices.sum();
        if (price_sum - 1.0).abs() > LMSR_PRECISION {
            return Err(LmsrError::PrecisionLoss);
        }

        Ok(prices)
    }

    pub fn calculate_buy_cost(
        &self,
        state: &LmsrState,
        outcome: usize,
        shares_to_buy: f64,
    ) -> Result<LmsrTradeResult, LmsrError> {
        let _prices = self.validate_state(state)?;
        let cost_before =
            self.cost_function(state.beta, &state.shares.view())?;
        self.calculate_buy_cost_with_base(
            state,
            outcome,
            shares_to_buy,
            cost_before,
        )
    }

    /// Internal helper: calculate buy cost given a precomputed base cost.
    /// Used by both `calculate_buy_cost` and `LmsrContext::calculate_buy_cost_fast`.
    fn calculate_buy_cost_with_base(
        &self,
        state: &LmsrState,
        outcome: usize,
        shares_to_buy: f64,
        cost_before: f64,
    ) -> Result<LmsrTradeResult, LmsrError> {
        if shares_to_buy <= 0.0 {
            return Err(LmsrError::ShareQuantityOverflow);
        }

        if outcome >= state.shares.len() {
            return Err(LmsrError::DimensionMismatch {
                expected: state.shares.len(),
                actual: outcome + 1,
            });
        }

        let mut new_shares = state.shares.clone();
        new_shares[outcome] += shares_to_buy;

        let cost_after = self.cost_function(state.beta, &new_shares.view())?;
        let trade_cost = cost_after - cost_before;
        let cost_satoshis = self.cost_to_satoshis(trade_cost)?;

        if cost_satoshis <= 0 {
            return Err(LmsrError::InvalidCostCalculation);
        }

        let fee_satoshis =
            self.calculate_fee(cost_satoshis as u64, state.trading_fee)?;
        let total_cost = cost_satoshis + (fee_satoshis as i64);

        if total_cost > (state.treasury_balance as i64) {
            return Err(LmsrError::InsufficientTreasury {
                required: total_cost as u64,
                available: state.treasury_balance,
            });
        }

        let new_treasury_balance = state.treasury_balance - (total_cost as u64);

        Ok(LmsrTradeResult {
            cost_satoshis: total_cost,
            fee_satoshis,
            new_shares,
            new_treasury_balance,
        })
    }

    pub fn calculate_sell_payout(
        &self,
        state: &LmsrState,
        outcome: usize,
        shares_to_sell: f64,
    ) -> Result<LmsrTradeResult, LmsrError> {
        let _prices = self.validate_state(state)?;

        if shares_to_sell <= 0.0 {
            return Err(LmsrError::ShareQuantityOverflow);
        }

        if outcome >= state.shares.len() {
            return Err(LmsrError::DimensionMismatch {
                expected: state.shares.len(),
                actual: outcome + 1,
            });
        }

        if shares_to_sell > state.shares[outcome] {
            return Err(LmsrError::ShareQuantityOverflow);
        }

        let mut new_shares = state.shares.clone();
        new_shares[outcome] -= shares_to_sell;

        let cost_before =
            self.cost_function(state.beta, &state.shares.view())?;
        let cost_after = self.cost_function(state.beta, &new_shares.view())?;

        let payout = cost_before - cost_after;
        let payout_satoshis = self.cost_to_satoshis(payout)?;

        if payout_satoshis <= 0 {
            return Err(LmsrError::InvalidCostCalculation);
        }

        let fee_satoshis =
            self.calculate_fee(payout_satoshis as u64, state.trading_fee)?;
        let net_payout = payout_satoshis - (fee_satoshis as i64);

        let new_treasury_balance = state.treasury_balance + (net_payout as u64);

        Ok(LmsrTradeResult {
            cost_satoshis: -net_payout,
            fee_satoshis,
            new_shares,
            new_treasury_balance,
        })
    }

    pub fn calculate_current_prices(
        &self,
        state: &LmsrState,
    ) -> Result<LmsrPriceInfo, LmsrError> {
        // Use context for single validation instead of N+1
        let ctx = LmsrContext::new(self, state)?;

        let mut marginal_costs = Array1::zeros(state.shares.len());
        for outcome in 0..state.shares.len() {
            match ctx.calculate_buy_cost_fast(outcome, 1.0) {
                Ok(result) => {
                    marginal_costs[outcome] = result.cost_satoshis as u64;
                }
                Err(_) => {
                    marginal_costs[outcome] = u64::MAX;
                }
            }
        }

        Ok(LmsrPriceInfo {
            prices: ctx.cached_prices.clone(),
            marginal_costs,
        })
    }

    pub fn calculate_required_treasury(
        &self,
        beta: f64,
        outcome_count: usize,
    ) -> Result<u64, LmsrError> {
        if beta <= MIN_BETA || beta >= MAX_BETA {
            return Err(LmsrError::InvalidBeta {
                beta,
                min: MIN_BETA,
                max: MAX_BETA,
            });
        }

        if outcome_count < MIN_OUTCOMES || outcome_count > self.max_outcomes {
            return Err(LmsrError::InvalidOutcomeCount {
                count: outcome_count,
                min: MIN_OUTCOMES,
                max: self.max_outcomes,
            });
        }

        let max_cost = beta * (outcome_count as f64).ln();

        if !max_cost.is_finite() || max_cost < 0.0 {
            return Err(LmsrError::InvalidCostCalculation);
        }

        safe_math::to_sats(max_cost, Rounding::Up)
            .map_err(|_| LmsrError::InvalidCostCalculation)
    }

    pub fn validate_state(
        &self,
        state: &LmsrState,
    ) -> Result<Array1<f64>, LmsrError> {
        if state.trading_fee < 0.0 || state.trading_fee >= 1.0 {
            return Err(LmsrError::InvalidCostCalculation);
        }

        self.validate_params(state.beta, &state.shares.view())
    }

    /// Validate LMSR parameters without requiring full LmsrState
    pub fn validate_params(
        &self,
        beta: f64,
        shares: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, LmsrError> {
        if beta <= MIN_BETA || beta >= MAX_BETA {
            return Err(LmsrError::InvalidBeta {
                beta,
                min: MIN_BETA,
                max: MAX_BETA,
            });
        }

        if !shares.iter().all(|&x| x >= 0.0 && x.is_finite()) {
            return Err(LmsrError::ShareQuantityOverflow);
        }

        if shares.len() < MIN_OUTCOMES || shares.len() > self.max_outcomes {
            return Err(LmsrError::InvalidOutcomeCount {
                count: shares.len(),
                min: MIN_OUTCOMES,
                max: self.max_outcomes,
            });
        }

        let prices = self.calculate_prices(beta, shares)?;
        let price_sum: f64 = prices.sum();
        if (price_sum - 1.0).abs() > LMSR_PRECISION {
            return Err(LmsrError::PrecisionLoss);
        }

        Ok(prices)
    }

    fn cost_to_satoshis(&self, cost: f64) -> Result<i64, LmsrError> {
        let satoshis = cost * SATOSHI_PRECISION;

        if !satoshis.is_finite() {
            return Err(LmsrError::InvalidCostCalculation);
        }

        safe_math::to_sats_signed(satoshis, Rounding::Up)
            .map_err(|_| LmsrError::InvalidCostCalculation)
    }

    fn calculate_fee(
        &self,
        cost_satoshis: u64,
        fee_rate: f64,
    ) -> Result<u64, LmsrError> {
        let bp = BasisPoints::from_f64(fee_rate).map_err(|e| {
            LmsrError::InvalidFeeRate {
                rate: fee_rate,
                reason: e.to_string(),
            }
        })?;
        Ok(bp.apply_to_u64(cost_satoshis))
    }
}

impl Default for Lmsr {
    fn default() -> Self {
        Self::new(MAX_OUTCOMES)
    }
}

pub struct LmsrService;

impl LmsrService {
    pub fn calculate_treasury(
        shares: &Array1<f64>,
        beta: f64,
    ) -> Result<f64, LmsrError> {
        let lmsr = Lmsr::new(shares.len());
        lmsr.cost_function(beta, &shares.view())
    }

    pub fn validate_lmsr_parameters(
        beta: f64,
        shares: &Array1<f64>,
    ) -> Result<(), LmsrError> {
        let lmsr = Lmsr::new(shares.len());
        let _prices = lmsr.validate_params(beta, &shares.view())?;
        Ok(())
    }

    pub fn calculate_prices(
        shares: &Array1<f64>,
        beta: f64,
    ) -> Result<Array1<f64>, LmsrError> {
        let lmsr = Lmsr::new(shares.len());
        lmsr.calculate_prices(beta, &shares.view())
    }

    pub fn calculate_update_cost(
        current_shares: &Array1<f64>,
        new_shares: &Array1<f64>,
        beta: f64,
    ) -> Result<f64, LmsrError> {
        if current_shares.len() != new_shares.len() {
            return Err(LmsrError::DimensionMismatch {
                expected: current_shares.len(),
                actual: new_shares.len(),
            });
        }

        let lmsr = Lmsr::new(current_shares.len());
        let current_cost = lmsr.cost_function(beta, &current_shares.view())?;
        let new_cost = lmsr.cost_function(beta, &new_shares.view())?;

        Ok(new_cost - current_cost)
    }
}

pub fn calculate_cost(shares: &[f64], beta: f64) -> Result<f64, LmsrError> {
    let shares_array = Array1::from_vec(shares.to_vec());
    LmsrService::calculate_treasury(&shares_array, beta)
}

pub fn calculate_prices(
    shares: &[f64],
    beta: f64,
) -> Result<Vec<f64>, LmsrError> {
    let shares_array = Array1::from_vec(shares.to_vec());
    let lmsr = Lmsr::new(shares.len());
    let prices = lmsr.calculate_prices(beta, &shares_array.view())?;
    Ok(prices.to_vec())
}

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::uninlined_format_args)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cost_calculation() {
        let lmsr = Lmsr::default();
        let state = LmsrState {
            beta: 14400.0,
            shares: array![10.0, 5.0],
            treasury_balance: 100000000,
            trading_fee: 0.01,
        };

        let result = lmsr.calculate_buy_cost(&state, 0, 1.0).unwrap();
        assert!(result.cost_satoshis > 0);
        assert!(result.fee_satoshis > 0);
        assert_eq!(result.new_shares[0], 11.0);
    }

    #[test]
    fn test_price_calculation() {
        let lmsr = Lmsr::default();
        let shares = array![10.0, 5.0];
        let prices = lmsr.calculate_prices(100.0, &shares.view()).unwrap();

        let price_sum: f64 = prices.sum();
        assert!((price_sum - 1.0).abs() < LMSR_PRECISION);
        assert!(prices[0] > prices[1]);
    }

    #[test]
    fn test_numerical_stability() {
        let lmsr = Lmsr::default();
        let shares = array![1000.0, 999.0, 998.0];
        let beta = 14400.0;

        let result = lmsr.calculate_prices(beta, &shares.view());
        assert!(result.is_ok());

        let prices = result.unwrap();
        assert!(
            prices
                .iter()
                .all(|&p| p.is_finite() && (0.0..=1.0).contains(&p))
        );
    }
}
