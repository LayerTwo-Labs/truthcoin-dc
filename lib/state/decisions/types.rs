use crate::state::Error;
use crate::types::{SECONDS_PER_QUARTER, Txid};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;

pub use truthcoin_dc_types::decision::{DecisionId, DecisionType};

fn is_step_aligned(delta: f64, step: f64) -> bool {
    let k = (delta / step).round();
    let tolerance = (step.abs() * 1e-9).max(1e-12);
    (delta - k * step).abs() <= tolerance
}

#[derive(
    Clone, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd,
)]
pub struct Decision {
    pub market_maker_pubkey_hash: [u8; 20],
    pub decision_type: DecisionType,
    pub header: String,
    pub description: String,
    pub option_0_label: Option<String>,
    pub option_1_label: Option<String>,
    pub tags: Vec<String>,
}

impl Decision {
    pub fn new(
        market_maker_pubkey_hash: [u8; 20],
        decision_type: DecisionType,
        header: String,
        description: String,
        option_0_label: Option<String>,
        option_1_label: Option<String>,
        tags: Vec<String>,
    ) -> Result<Self, Error> {
        if header.len() > 100 {
            return Err(Error::InvalidDecisionId {
                reason: "Header must be 100 bytes or less".to_string(),
            });
        }
        if description.len() > 2000 {
            return Err(Error::InvalidDecisionId {
                reason: "Description must be 2000 bytes or less".to_string(),
            });
        }
        if tags.len() > 10 {
            return Err(Error::InvalidDecisionId {
                reason: "Maximum 10 tags allowed".to_string(),
            });
        }
        for tag in &tags {
            if tag.len() > 50 {
                return Err(Error::InvalidDecisionId {
                    reason: format!("Tag '{tag}' exceeds 50 byte limit"),
                });
            }
        }

        match &decision_type {
            DecisionType::Binary => {}
            DecisionType::Scaled {
                min,
                max,
                increment,
            } => {
                if !min.is_finite()
                    || !max.is_finite()
                    || !increment.is_finite()
                    || min >= max
                    || *increment <= 0.0
                {
                    return Err(Error::InvalidScaled {
                        reason: format!(
                            "invalid bounds: min={min}, max={max}, \
                             increment={increment}"
                        ),
                    });
                }
                if !is_step_aligned(max - min, *increment) {
                    return Err(Error::InvalidScaled {
                        reason: format!(
                            "(max - min) = {} is not an integer multiple of \
                             increment {increment}",
                            max - min
                        ),
                    });
                }
            }
            DecisionType::Category { options } => {
                if options.len() < 2 {
                    return Err(Error::InvalidDecisionId {
                        reason: "Category decisions require \
                                 at least 2 options"
                            .to_string(),
                    });
                }
            }
        }

        Ok(Decision {
            market_maker_pubkey_hash,
            decision_type,
            header,
            description,
            option_0_label,
            option_1_label,
            tags,
        })
    }

    pub fn is_scaled(&self) -> bool {
        matches!(self.decision_type, DecisionType::Scaled { .. })
    }

    pub fn is_categorical(&self) -> bool {
        matches!(self.decision_type, DecisionType::Category { .. })
    }

    pub fn option_count(&self) -> Option<usize> {
        match &self.decision_type {
            DecisionType::Category { options } => Some(options.len()),
            _ => None,
        }
    }

    pub fn inconclusive_index(&self) -> Option<u16> {
        self.option_count().map(|n| n as u16)
    }

    pub fn get_category_labels(&self) -> Option<&[String]> {
        match &self.decision_type {
            DecisionType::Category { options } => Some(options),
            _ => None,
        }
    }

    pub fn scale_min(&self) -> Option<f64> {
        match &self.decision_type {
            DecisionType::Scaled { min, .. } => Some(*min),
            _ => None,
        }
    }

    pub fn scale_max(&self) -> Option<f64> {
        match &self.decision_type {
            DecisionType::Scaled { max, .. } => Some(*max),
            _ => None,
        }
    }

    pub fn id(&self) -> [u8; 32] {
        use crate::types::hashes;

        let hash_data = (
            &self.market_maker_pubkey_hash,
            &self.decision_type,
            &self.header,
            &self.description,
            &self.option_0_label,
            &self.option_1_label,
            &self.tags,
        );

        hashes::hash(&hash_data)
    }

    pub fn get_binary_labels(&self) -> (String, String) {
        let label0 = self.option_0_label.as_deref().unwrap_or("No").to_string();
        let label1 =
            self.option_1_label.as_deref().unwrap_or("Yes").to_string();
        (label0, label1)
    }

    pub fn normalize_value(&self, user_value: f64) -> f64 {
        if user_value.is_nan() {
            return f64::NAN;
        }

        match &self.decision_type {
            DecisionType::Scaled { min, max, .. } => {
                let range = max - min;
                if range == 0.0 {
                    return 0.5;
                }
                (user_value - min) / range
            }
            DecisionType::Binary | DecisionType::Category { .. } => user_value,
        }
    }

    pub fn denormalize_value(&self, internal_value: f64) -> f64 {
        if internal_value.is_nan() {
            return f64::NAN;
        }

        match &self.decision_type {
            DecisionType::Scaled { min, max, .. } => {
                let range = max - min;
                internal_value * range + min
            }
            DecisionType::Binary | DecisionType::Category { .. } => {
                internal_value
            }
        }
    }

    pub fn validate_and_normalize(
        &self,
        user_value: f64,
    ) -> Result<f64, Error> {
        if user_value.is_nan() {
            return Ok(f64::NAN);
        }

        match &self.decision_type {
            DecisionType::Scaled {
                min,
                max,
                increment,
            } => {
                if user_value < *min || user_value > *max {
                    return Err(Error::InvalidVoteValue {
                        reason: format!(
                            "Value {user_value} is outside \
                             allowed range [{min}, {max}]"
                        ),
                    });
                }
                if !is_step_aligned(user_value - min, *increment) {
                    return Err(Error::InvalidVoteValue {
                        reason: format!(
                            "Value {user_value} is not a multiple of \
                             increment {increment} from min {min}"
                        ),
                    });
                }
            }
            DecisionType::Binary => {
                if !(0.0..=1.0).contains(&user_value) {
                    return Err(Error::InvalidVoteValue {
                        reason: format!(
                            "Binary vote value {user_value} \
                             must be between 0.0 and 1.0"
                        ),
                    });
                }
            }
            DecisionType::Category { options } => {
                let idx = user_value as usize;
                if user_value < 0.0
                    || user_value != (idx as f64)
                    || idx > options.len()
                {
                    return Err(Error::InvalidVoteValue {
                        reason: format!(
                            "Categorical vote value \
                             {user_value} must be a valid \
                             option index (0..={}, where {} = Inconclusive)",
                            options.len(),
                            options.len()
                        ),
                    });
                }
            }
        }

        Ok(self.normalize_value(user_value))
    }

    pub fn get_display_range(&self) -> (f64, f64) {
        match &self.decision_type {
            DecisionType::Scaled { min, max, .. } => (*min, *max),
            DecisionType::Binary => (0.0, 1.0),
            DecisionType::Category { options } => (0.0, options.len() as f64),
        }
    }
}

pub use crate::types::decision::DecisionState;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct DecisionStateHistory {
    pub state_history: Vec<(DecisionState, u32, Option<f64>)>,
}

impl DecisionStateHistory {
    pub fn new_claimed(height: u32) -> Self {
        Self {
            state_history: vec![(DecisionState::Claimed, height, None)],
        }
    }

    pub fn new(initial_height: u32) -> Self {
        Self::new_claimed(initial_height)
    }

    pub fn current_state(&self) -> DecisionState {
        self.state_history
            .last()
            .map(|(s, _, _)| *s)
            .unwrap_or(DecisionState::Created)
    }

    pub fn transition_to_voting(&mut self, height: u32) -> Result<(), Error> {
        let current = self.current_state();
        if !current.can_transition_to(DecisionState::Voting) {
            return Err(Error::InvalidDecisionState {
                reason: format!("Cannot transition to Voting from {current:?}"),
            });
        }
        self.state_history
            .push((DecisionState::Voting, height, None));
        Ok(())
    }

    pub fn transition_to_resolved(
        &mut self,
        consensus_outcome: f64,
        height: u32,
    ) -> Result<(), Error> {
        let current = self.current_state();
        if !current.can_transition_to(DecisionState::Resolved) {
            return Err(Error::InvalidDecisionState {
                reason: format!(
                    "Cannot transition to Resolved from {current:?}"
                ),
            });
        }

        if consensus_outcome < -1.0 {
            return Err(Error::InvalidDecisionState {
                reason: format!(
                    "Consensus outcome {consensus_outcome} \
                     is invalid (< -1.0)"
                ),
            });
        }

        self.state_history.push((
            DecisionState::Resolved,
            height,
            Some(consensus_outcome),
        ));
        Ok(())
    }

    pub fn can_accept_votes(&self) -> bool {
        self.current_state() == DecisionState::Voting
    }

    pub fn has_consensus(&self) -> bool {
        self.current_state().has_consensus()
    }

    pub fn consensus_outcome(&self) -> Option<f64> {
        self.state_history
            .iter()
            .find(|(s, _, _)| *s == DecisionState::Resolved)
            .and_then(|(_, _, outcome)| *outcome)
    }

    pub fn has_reached_state(&self, state: DecisionState) -> bool {
        self.state_history.iter().any(|(s, _, _)| *s == state)
    }

    pub fn state_at_height(&self, height: u32) -> DecisionState {
        self.state_history
            .iter()
            .rev()
            .find(|(_, h, _)| *h <= height)
            .map(|(s, _, _)| *s)
            .unwrap_or(DecisionState::Created)
    }

    pub fn rollback_to_height(&mut self, height: u32) {
        self.state_history.retain(|(_, h, _)| *h <= height);
    }

    pub fn get_state_height(&self, state: DecisionState) -> Option<u32> {
        self.state_history
            .iter()
            .find(|(s, _, _)| *s == state)
            .map(|(_, h, _)| *h)
    }
}

#[derive(
    Clone, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd,
)]
pub struct DecisionEntry {
    pub decision_id: DecisionId,
    pub decision: Option<Decision>,
    /// Txid of the transaction that claimed this decision
    pub claiming_txid: Txid,
}

pub(crate) const FUTURE_PERIODS: u32 = 20;
pub(crate) const INITIAL_DECISIONS_PER_PERIOD: u64 = 500;

#[derive(Clone, Debug)]
pub struct DecisionConfig {
    pub is_blocks: bool,
    pub quantity: u32,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            is_blocks: false,
            quantity: 86400,
        }
    }
}

impl DecisionConfig {
    pub fn production() -> Self {
        Self {
            is_blocks: false,
            quantity: SECONDS_PER_QUARTER as u32,
        }
    }

    pub fn testing(blocks_per_period: NonZeroU32) -> Self {
        Self {
            is_blocks: true,
            quantity: blocks_per_period.get(),
        }
    }

    pub fn is_blocks_mode(&self) -> bool {
        self.is_blocks
    }

    pub fn blocks_per_period(&self) -> Option<u32> {
        if self.is_blocks {
            Some(self.quantity)
        } else {
            None
        }
    }

    pub fn seconds_per_period(&self) -> Option<u64> {
        if !self.is_blocks {
            Some(self.quantity as u64)
        } else {
            None
        }
    }
}

/// Convert period index to human-readable name (single source of truth).
/// Returns "Genesis" for period 0, otherwise "Q{quarter} Y{year}".
#[inline]
pub fn period_to_name(period: u32) -> String {
    if period == 0 {
        return "Genesis".to_string();
    }
    let year = 2009 + (period - 1) / 4;
    let quarter = ((period - 1) % 4) + 1;
    format!("Q{quarter} Y{year}")
}

pub fn period_to_string(period_idx: u32, config: &DecisionConfig) -> String {
    if config.is_blocks {
        format!("Testing Period {period_idx}")
    } else {
        period_to_name(period_idx)
    }
}
