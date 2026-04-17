/// Optimized LMSR Mathematical Operations using ndarray for Bitcoin Hivemind Sidechain
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::fmt;

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

        let max_share = shares.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let shifted_shares = (shares - max_share) / beta;
        let exp_shares = shifted_shares.mapv(libm::exp);

        if !exp_shares.iter().all(|x| x.is_finite()) {
            return Err(LmsrError::ShareQuantityOverflow);
        }

        let sum_exp = exp_shares.iter().fold(0.0_f64, |acc, &x| acc + x);

        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            return Err(LmsrError::InvalidCostCalculation);
        }

        let cost = beta * (libm::log(sum_exp) + max_share / beta);

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
        let price_sum: f64 = prices.iter().fold(0.0_f64, |acc, &x| acc + x);
        if (price_sum - 1.0).abs() > LMSR_PRECISION {
            return Err(LmsrError::PrecisionLoss);
        }

        Ok(prices)
    }

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
        let price_sum: f64 = prices.iter().fold(0.0_f64, |acc, &x| acc + x);
        if (price_sum - 1.0).abs() > LMSR_PRECISION {
            return Err(LmsrError::PrecisionLoss);
        }

        Ok(prices)
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
        lmsr.validate_params(beta, &shares.view())?;
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

    #[test]
    fn test_deterministic_cost_bits() {
        let lmsr = Lmsr::default();
        let shares = array![0.5, 0.3, 0.2];
        let beta = 100.0;
        let cost = lmsr.cost_function(beta, &shares.view()).unwrap();
        const EXPECTED: u64 = 0x405b_8c74_fb4a_283e;
        assert_eq!(
            cost.to_bits(),
            EXPECTED,
            "LMSR cost bit-pattern drift: got {:#018x}",
            cost.to_bits()
        );
    }

    #[test]
    fn test_deterministic_prices_bits() {
        let lmsr = Lmsr::default();
        let shares = array![1.0, 2.0, 3.0];
        let beta = 50.0;
        let prices = lmsr.calculate_prices(beta, &shares.view()).unwrap();
        const EXPECTED: [u64; 3] = [
            0x3fd4_e87a_5e3a_3a49,
            0x3fd5_549a_f03f_08e4,
            0x3fd5_c2ea_b186_bcd3,
        ];
        let actual: Vec<u64> = prices.iter().map(|p| p.to_bits()).collect();
        assert_eq!(
            actual,
            EXPECTED.to_vec(),
            "LMSR prices bit-pattern drift: got {:#018x?}",
            actual
        );
    }
}
