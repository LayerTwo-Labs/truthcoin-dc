use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};

use crate::error::InvalidDecisionId;

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    PartialOrd,
    Ord,
    Deserialize,
    Serialize,
    BorshSerialize,
)]
pub struct DecisionId([u8; 3]);

const MAX_PERIOD_INDEX: u32 = (1 << 7) - 1; // 127
const MAX_DECISION_INDEX: u32 = (1 << 16) - 1; // 65535
const STANDARD_BIT: u32 = 23;
const PERIOD_SHIFT: u32 = 16;
const PERIOD_MASK: u32 = 0x7F; // 7 bits
const DECISION_MASK: u32 = MAX_DECISION_INDEX; // 16 bits

fn validate_decision_bounds(
    period: u32,
    index: u32,
) -> Result<(), InvalidDecisionId> {
    if period > MAX_PERIOD_INDEX {
        return Err(InvalidDecisionId {
            reason: format!(
                "Period {period} exceeds maximum {MAX_PERIOD_INDEX}"
            ),
        });
    }
    if index > MAX_DECISION_INDEX {
        return Err(InvalidDecisionId {
            reason: format!(
                "Decision index {index} exceeds maximum \
                 {MAX_DECISION_INDEX}"
            ),
        });
    }
    Ok(())
}

impl DecisionId {
    #[inline(always)]
    const fn as_u32(self) -> u32 {
        ((self.0[0] as u32) << 16)
            | ((self.0[1] as u32) << 8)
            | (self.0[2] as u32)
    }

    pub fn new(
        is_standard: bool,
        period: u32,
        index: u32,
    ) -> Result<Self, InvalidDecisionId> {
        validate_decision_bounds(period, index)?;
        let standard_bit = if is_standard { 1u32 } else { 0u32 };
        let combined =
            (standard_bit << STANDARD_BIT) | (period << PERIOD_SHIFT) | index;
        let bytes = [
            (combined >> 16) as u8,
            (combined >> 8) as u8,
            combined as u8,
        ];
        Ok(DecisionId(bytes))
    }

    #[inline(always)]
    pub const fn is_standard(self) -> bool {
        (self.as_u32() >> STANDARD_BIT) & 1 == 1
    }

    #[inline(always)]
    pub const fn period_index(self) -> u32 {
        (self.as_u32() >> PERIOD_SHIFT) & PERIOD_MASK
    }

    #[inline(always)]
    pub const fn decision_index(self) -> u32 {
        self.as_u32() & DECISION_MASK
    }

    pub fn as_bytes(self) -> [u8; 3] {
        self.0
    }

    pub fn from_bytes(bytes: [u8; 3]) -> Result<Self, InvalidDecisionId> {
        let id = DecisionId(bytes);
        validate_decision_bounds(id.period_index(), id.decision_index())?;
        Ok(id)
    }

    pub fn from_hex(hex_str: &str) -> Result<Self, InvalidDecisionId> {
        if hex_str.len() != 6 {
            return Err(InvalidDecisionId {
                reason: "Decision ID hex must be exactly 6 characters \
                     (3 bytes)"
                    .to_string(),
            });
        }

        let mut bytes = [0u8; 3];
        for (i, chunk) in
            hex_str.as_bytes().as_chunks::<2>().0.iter().enumerate()
        {
            let s =
                std::str::from_utf8(chunk).map_err(|_| InvalidDecisionId {
                    reason: "Invalid decision ID hex format".to_string(),
                })?;
            bytes[i] =
                u8::from_str_radix(s, 16).map_err(|_| InvalidDecisionId {
                    reason: "Invalid decision ID hex format".to_string(),
                })?;
        }

        Self::from_bytes(bytes)
    }

    pub fn to_hex(self) -> String {
        const_hex::encode(self.0)
    }

    #[inline(always)]
    pub const fn voting_period(self) -> u32 {
        self.period_index() + 1
    }
}

#[derive(
    Clone, Debug, Deserialize, Serialize, BorshSerialize, utoipa::ToSchema,
)]
pub enum DecisionType {
    Binary,
    Scaled { min: f64, max: f64, increment: f64 },
    Category { options: Vec<String> },
}

impl PartialEq for DecisionType {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for DecisionType {}

impl PartialOrd for DecisionType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DecisionType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use DecisionType::{Binary, Category, Scaled};
        use std::cmp::Ordering;
        match (self, other) {
            (Binary, Binary) => Ordering::Equal,
            (
                Scaled {
                    min: a,
                    max: b,
                    increment: c,
                },
                Scaled {
                    min: d,
                    max: e,
                    increment: f,
                },
            ) => a.total_cmp(d).then(b.total_cmp(e)).then(c.total_cmp(f)),
            (Category { options: a }, Category { options: b }) => a.cmp(b),
            (Binary, _) => Ordering::Less,
            (_, Binary) => Ordering::Greater,
            (Scaled { .. }, _) => Ordering::Less,
            (_, Scaled { .. }) => Ordering::Greater,
        }
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Deserialize,
    Serialize,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    utoipa::ToSchema,
)]
pub enum DecisionState {
    Created,
    Claimed,
    Voting,
    Resolved,
    Invalid,
}

impl DecisionState {
    pub fn can_transition_to(&self, new_state: DecisionState) -> bool {
        use DecisionState::*;
        matches!(
            (self, new_state),
            (Claimed, Voting) | (Voting, Resolved) | (_, Invalid)
        )
    }

    pub fn allows_voting(&self) -> bool {
        matches!(self, DecisionState::Voting)
    }

    pub fn has_consensus(&self) -> bool {
        matches!(self, DecisionState::Resolved)
    }
}
