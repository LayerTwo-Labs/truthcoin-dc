use std::{fmt, num::ParseIntError, str::FromStr};

use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    ToSchema,
)]
#[serde(transparent)]
#[schema(value_type = u64)]
pub struct Votecoin(pub u64);

impl Votecoin {
    pub const ZERO: Self = Self(0);
    pub const TOTAL: Self = Self(1_000_000);

    pub const fn checked_add(self, other: Self) -> Option<Self> {
        match self.0.checked_add(other.0) {
            Some(units) => Some(Self(units)),
            None => None,
        }
    }

    pub const fn checked_sub(self, other: Self) -> Option<Self> {
        match self.0.checked_sub(other.0) {
            Some(units) => Some(Self(units)),
            None => None,
        }
    }

    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for Votecoin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl FromStr for Votecoin {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse().map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::Votecoin;

    /// JSON integer and borsh u64; fractions are refused
    #[test]
    fn votecoin_encoding() {
        let amount = Votecoin(142_857);

        assert_eq!(serde_json::to_string(&amount).unwrap(), "142857");
        assert_eq!(serde_json::from_str::<Votecoin>("142857").unwrap(), amount);
        assert_eq!(borsh::to_vec(&amount).unwrap(), 142_857u64.to_le_bytes());
        assert!(serde_json::from_str::<Votecoin>("0.5").is_err());
        assert_eq!("7".parse::<Votecoin>().unwrap(), Votecoin(7));
    }

    #[test]
    fn votecoin_arithmetic_is_checked() {
        assert_eq!(Votecoin(5).checked_sub(Votecoin(3)), Some(Votecoin(2)));
        assert_eq!(Votecoin(3).checked_sub(Votecoin(5)), None);
        assert_eq!(Votecoin(u64::MAX).checked_add(Votecoin(1)), None);
        assert!(Votecoin::ZERO.is_zero());
    }
}
