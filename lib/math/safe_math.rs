//! Satoshi conversion utilities with standardized rounding.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum SatoshiError {
    #[error("Non-finite value: {0}")]
    NonFinite(f64),
    #[error("Negative value not allowed: {0}")]
    Negative(f64),
    #[error("Value exceeds maximum: {0}")]
    Overflow(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    Up,      // ceil - for costs/fees
    Down,    // floor - for payouts
    Nearest, // round - for neutral calculations
}

/// Convert f64 to u64 satoshis with validation and rounding.
pub fn to_sats(value: f64, mode: Rounding) -> Result<u64, SatoshiError> {
    if !value.is_finite() {
        return Err(SatoshiError::NonFinite(value));
    }
    if value < 0.0 {
        return Err(SatoshiError::Negative(value));
    }
    let rounded = match mode {
        Rounding::Up => value.ceil(),
        Rounding::Down => value.floor(),
        Rounding::Nearest => value.round(),
    };
    if rounded > u64::MAX as f64 {
        return Err(SatoshiError::Overflow(value));
    }
    Ok(rounded as u64)
}

/// Convert f64 to i64 satoshis (for values that can be negative).
pub fn to_sats_signed(value: f64, mode: Rounding) -> Result<i64, SatoshiError> {
    if !value.is_finite() {
        return Err(SatoshiError::NonFinite(value));
    }
    let rounded = match mode {
        Rounding::Up => {
            if value >= 0.0 {
                value.ceil()
            } else {
                value.floor()
            }
        }
        Rounding::Down => {
            if value >= 0.0 {
                value.floor()
            } else {
                value.ceil()
            }
        }
        Rounding::Nearest => value.round(),
    };
    if rounded > i64::MAX as f64 || rounded < i64::MIN as f64 {
        return Err(SatoshiError::Overflow(value));
    }
    Ok(rounded as i64)
}

/// Rate as basis points (0-10000 = 0%-100%). Uses integer arithmetic to avoid f64 precision issues.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
pub struct BasisPoints(u32);

impl BasisPoints {
    pub const ZERO: BasisPoints = BasisPoints(0);
    pub const MAX: u32 = 10000;
    pub const ONE_HUNDRED_PERCENT: BasisPoints = BasisPoints(10000);
    /// VoteCoin redistribution rate: 10% = 1000 basis points
    pub const REDISTRIBUTION_RATE: BasisPoints = BasisPoints(1000);

    pub fn new(value: u32) -> Result<Self, SatoshiError> {
        if value > Self::MAX {
            return Err(SatoshiError::Overflow(value as f64));
        }
        Ok(BasisPoints(value))
    }

    pub fn from_f64(rate: f64) -> Result<Self, SatoshiError> {
        if !rate.is_finite() {
            return Err(SatoshiError::NonFinite(rate));
        }
        if !(0.0..=1.0).contains(&rate) {
            return Err(SatoshiError::Overflow(rate));
        }
        Self::new((rate * 10000.0).round() as u32)
    }

    #[inline]
    pub fn apply_to_u64(&self, amount: u64) -> u64 {
        ((amount as u128 * self.0 as u128) / 10000) as u64
    }

    #[inline]
    pub fn apply_to_i64(&self, amount: i64) -> i64 {
        ((amount as i128 * self.0 as i128) / 10000) as i64
    }

    #[inline]
    pub fn value(&self) -> u32 {
        self.0
    }

    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.0 as f64 / 10000.0
    }
}

impl Default for BasisPoints {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for BasisPoints {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}bp ({}%)", self.0, self.0 as f64 / 100.0)
    }
}

/// NaN-safe f64 comparison. NaN values sort as equal to each other and less than all other values.
#[inline]
pub fn safe_cmp_f64(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// Canonical rounding function for precision-controlled values.
/// Non-finite values (NaN, Infinity) pass through unchanged.
#[inline]
pub fn round_to_precision(value: f64, decimals: u32) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let multiplier = 10f64.powi(decimals as i32);
    (value * multiplier).round() / multiplier
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_sats_rounding_up() {
        assert_eq!(to_sats(100.0, Rounding::Up).unwrap(), 100);
        assert_eq!(to_sats(100.1, Rounding::Up).unwrap(), 101);
        assert_eq!(to_sats(100.9, Rounding::Up).unwrap(), 101);
        assert_eq!(to_sats(0.0, Rounding::Up).unwrap(), 0);
    }

    #[test]
    fn test_to_sats_rounding_down() {
        assert_eq!(to_sats(100.0, Rounding::Down).unwrap(), 100);
        assert_eq!(to_sats(100.1, Rounding::Down).unwrap(), 100);
        assert_eq!(to_sats(100.9, Rounding::Down).unwrap(), 100);
        assert_eq!(to_sats(0.0, Rounding::Down).unwrap(), 0);
    }

    #[test]
    fn test_to_sats_rounding_nearest() {
        assert_eq!(to_sats(100.0, Rounding::Nearest).unwrap(), 100);
        assert_eq!(to_sats(100.4, Rounding::Nearest).unwrap(), 100);
        assert_eq!(to_sats(100.5, Rounding::Nearest).unwrap(), 101); // round half to even: 100.5 -> 100 or 101
        assert_eq!(to_sats(100.6, Rounding::Nearest).unwrap(), 101);
    }

    #[test]
    fn test_to_sats_negative_error() {
        assert!(matches!(
            to_sats(-1.0, Rounding::Up),
            Err(SatoshiError::Negative(_))
        ));
    }

    #[test]
    fn test_to_sats_non_finite_error() {
        assert!(matches!(
            to_sats(f64::NAN, Rounding::Up),
            Err(SatoshiError::NonFinite(_))
        ));
        assert!(matches!(
            to_sats(f64::INFINITY, Rounding::Up),
            Err(SatoshiError::NonFinite(_))
        ));
        assert!(matches!(
            to_sats(f64::NEG_INFINITY, Rounding::Up),
            Err(SatoshiError::NonFinite(_))
        ));
    }

    #[test]
    fn test_to_sats_signed_positive() {
        assert_eq!(to_sats_signed(100.3, Rounding::Up).unwrap(), 101);
        assert_eq!(to_sats_signed(100.3, Rounding::Down).unwrap(), 100);
        assert_eq!(to_sats_signed(100.5, Rounding::Nearest).unwrap(), 101);
    }

    #[test]
    fn test_to_sats_signed_negative() {
        // Up rounds away from zero: -100.3 -> -101
        assert_eq!(to_sats_signed(-100.3, Rounding::Up).unwrap(), -101);
        // Down rounds toward zero: -100.3 -> -100
        assert_eq!(to_sats_signed(-100.3, Rounding::Down).unwrap(), -100);
        // Nearest: -100.5 -> -101 (or -100, depending on tie-breaking)
        assert_eq!(to_sats_signed(-100.6, Rounding::Nearest).unwrap(), -101);
    }

    #[test]
    fn test_to_sats_signed_non_finite_error() {
        assert!(matches!(
            to_sats_signed(f64::NAN, Rounding::Up),
            Err(SatoshiError::NonFinite(_))
        ));
    }

    // BasisPoints tests
    #[test]
    fn test_basis_points_new() {
        assert!(BasisPoints::new(0).is_ok());
        assert!(BasisPoints::new(5000).is_ok());
        assert!(BasisPoints::new(10000).is_ok());
        assert!(BasisPoints::new(10001).is_err());
    }

    #[test]
    fn test_basis_points_from_f64() {
        assert_eq!(BasisPoints::from_f64(0.0).unwrap().value(), 0);
        assert_eq!(BasisPoints::from_f64(0.1).unwrap().value(), 1000); // 10%
        assert_eq!(BasisPoints::from_f64(0.5).unwrap().value(), 5000); // 50%
        assert_eq!(BasisPoints::from_f64(1.0).unwrap().value(), 10000); // 100%

        // Invalid values
        assert!(BasisPoints::from_f64(-0.1).is_err());
        assert!(BasisPoints::from_f64(1.1).is_err());
        assert!(BasisPoints::from_f64(f64::NAN).is_err());
        assert!(BasisPoints::from_f64(f64::INFINITY).is_err());
    }

    #[test]
    fn test_basis_points_apply_to_u64() {
        let rate = BasisPoints::new(1000).unwrap(); // 10%
        assert_eq!(rate.apply_to_u64(1000), 100);
        assert_eq!(rate.apply_to_u64(0), 0);
        assert_eq!(rate.apply_to_u64(10), 1);

        let half = BasisPoints::new(5000).unwrap(); // 50%
        assert_eq!(half.apply_to_u64(1000), 500);

        let full = BasisPoints::ONE_HUNDRED_PERCENT;
        assert_eq!(full.apply_to_u64(1000), 1000);

        // Test with large values (no overflow)
        let rate = BasisPoints::new(10000).unwrap();
        assert_eq!(rate.apply_to_u64(u64::MAX), u64::MAX);
    }

    #[test]
    fn test_basis_points_apply_to_i64() {
        let rate = BasisPoints::new(1000).unwrap(); // 10%
        assert_eq!(rate.apply_to_i64(1000), 100);
        assert_eq!(rate.apply_to_i64(-1000), -100);
        assert_eq!(rate.apply_to_i64(0), 0);
    }

    #[test]
    fn test_basis_points_to_f64() {
        assert!((BasisPoints::new(1000).unwrap().to_f64() - 0.1).abs() < 1e-10);
        assert!((BasisPoints::new(5000).unwrap().to_f64() - 0.5).abs() < 1e-10);
        assert!(
            (BasisPoints::new(10000).unwrap().to_f64() - 1.0).abs() < 1e-10
        );
    }

    #[test]
    fn test_basis_points_display() {
        let bp = BasisPoints::new(1000).unwrap();
        assert_eq!(format!("{bp}"), "1000bp (10%)");
    }

    // safe_cmp_f64 tests
    #[test]
    fn test_safe_cmp_f64_normal() {
        assert_eq!(safe_cmp_f64(1.0, 2.0), Ordering::Less);
        assert_eq!(safe_cmp_f64(2.0, 1.0), Ordering::Greater);
        assert_eq!(safe_cmp_f64(1.0, 1.0), Ordering::Equal);
    }

    #[test]
    fn test_safe_cmp_f64_nan() {
        // NaN compared to NaN
        assert_eq!(safe_cmp_f64(f64::NAN, f64::NAN), Ordering::Equal);

        // NaN compared to normal
        assert_eq!(safe_cmp_f64(f64::NAN, 1.0), Ordering::Less);
        assert_eq!(safe_cmp_f64(1.0, f64::NAN), Ordering::Greater);

        // NaN compared to infinity
        assert_eq!(safe_cmp_f64(f64::NAN, f64::INFINITY), Ordering::Less);
        assert_eq!(safe_cmp_f64(f64::INFINITY, f64::NAN), Ordering::Greater);
    }

    #[test]
    fn test_safe_cmp_f64_sorting() {
        let mut values = [3.0, f64::NAN, 1.0, f64::NAN, 2.0];
        values.sort_by(|a, b| safe_cmp_f64(*a, *b));
        // NaN values should be at the beginning
        assert!(values[0].is_nan());
        assert!(values[1].is_nan());
        assert_eq!(values[2], 1.0);
        assert_eq!(values[3], 2.0);
        assert_eq!(values[4], 3.0);
    }
}
