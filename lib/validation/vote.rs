use crate::state::Error;
use crate::types::FilledTransaction;
use sneed::RoTxn;

pub struct VoteValidator;

impl VoteValidator {
    pub fn convert_vote_value(
        vote_value: f64,
    ) -> crate::state::voting::types::VoteValue {
        Self::convert_vote_value_with_decision(vote_value, None)
    }

    pub fn convert_vote_value_with_decision(
        vote_value: f64,
        decision: Option<&crate::state::decisions::Decision>,
    ) -> crate::state::voting::types::VoteValue {
        use crate::state::voting::types::VoteValue;

        if vote_value.is_nan() {
            return VoteValue::Abstain;
        }

        if let Some(d) = decision
            && d.is_categorical()
        {
            let idx = vote_value.round() as u16;
            return VoteValue::Categorical(idx);
        }

        if vote_value == 0.0 || vote_value == 1.0 {
            VoteValue::Binary(vote_value == 1.0)
        } else {
            VoteValue::Scalar(vote_value)
        }
    }

    fn validate_voter_eligibility(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voter_address: &crate::types::Address,
    ) -> Result<(), Error> {
        let reputation =
            state.reputation().get_reputation(rotxn, voter_address)?;
        if reputation <= 0.0 {
            return Err(Error::InvalidTransaction {
                reason: "Voter has no reputation (voting rights)".to_string(),
            });
        }
        Ok(())
    }

    fn validate_decision_entry(
        state: &crate::state::State,
        rotxn: &RoTxn,
        decision_id: crate::state::decisions::DecisionId,
    ) -> Result<crate::state::decisions::Decision, Error> {
        let decision_entry = state
            .decisions()
            .get_decision_entry(rotxn, decision_id)?
            .ok_or_else(|| Error::InvalidDecisionId {
                reason: format!("Decision {decision_id:?} does not exist"),
            })?;

        let decision = decision_entry.decision.ok_or_else(|| {
            Error::InvalidDecisionId {
                reason: format!("Decision {decision_id:?} has no decision"),
            }
        })?;

        Ok(decision)
    }

    fn validate_vote_value(
        decision: &crate::state::decisions::Decision,
        vote_value: f64,
    ) -> Result<(), Error> {
        if vote_value.is_nan() {
            return Ok(());
        }

        if decision.is_categorical() {
            let max_idx = decision.option_count().unwrap_or(0) as f64;
            let idx = vote_value as usize;
            let is_integer_in_range = vote_value >= 0.0
                && vote_value == (idx as f64)
                && vote_value <= max_idx;
            if !is_integer_in_range {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Categorical vote value {vote_value} must be an integer \
                         index in 0..={max_idx} (where {max_idx} = Inconclusive)"
                    ),
                });
            }
            return Ok(());
        }

        if !(0.0..=1.0).contains(&vote_value) {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Vote value {vote_value} outside valid normalized range [0.0, 1.0]"
                ),
            });
        }

        Ok(())
    }

    fn validate_voting_period(
        state: &crate::state::State,
        rotxn: &sneed::RoTxn,
        decision_id: crate::state::decisions::DecisionId,
    ) -> Result<(), Error> {
        if !state
            .decisions()
            .is_decision_in_voting(rotxn, decision_id)?
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Decision {decision_id:?} is not in voting period"
                ),
            });
        }
        Ok(())
    }

    fn validate_no_duplicate_vote(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voter_address: crate::types::Address,
        period_id: crate::state::voting::types::VotingPeriodId,
        decision_id: crate::state::decisions::DecisionId,
    ) -> Result<(), Error> {
        if state
            .voting()
            .databases()
            .get_vote(rotxn, period_id, voter_address, decision_id)?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: "Duplicate vote: voter already voted on this decision in this period"
                    .to_string(),
            });
        }
        Ok(())
    }

    pub fn validate_vote_submission(
        state: &crate::state::State,
        rotxn: &RoTxn,
        filled_tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::state::{
            decisions::DecisionId, voting::types::VotingPeriodId,
        };

        let vote_data = filled_tx.submit_vote().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a vote submission transaction".to_string(),
            }
        })?;

        let voter_address = vote_data.voter;

        let has_voter_input = filled_tx
            .spent_utxos
            .iter()
            .any(|utxo| utxo.address == voter_address);
        let has_actor_proof = filled_tx.actor_address == Some(voter_address);
        if !has_voter_input && !has_actor_proof {
            return Err(Error::InvalidTransaction {
                reason: format!("No proof of voter {voter_address} identity"),
            });
        }

        Self::validate_voter_eligibility(state, rotxn, &voter_address)?;

        let decision_id = DecisionId::from_bytes(vote_data.decision_id_bytes)?;
        let decision =
            Self::validate_decision_entry(state, rotxn, decision_id)?;

        Self::validate_vote_value(&decision, vote_data.vote_value)?;

        Self::validate_voting_period(state, rotxn, decision_id)?;

        // Voting period is deterministically derived from decision: voting_period = period_index + 1
        let period_id = VotingPeriodId::new(decision_id.voting_period());
        Self::validate_no_duplicate_vote(
            state,
            rotxn,
            voter_address,
            period_id,
            decision_id,
        )?;

        Ok(())
    }

    pub fn validate_reputation_transfer(
        state: &crate::state::State,
        rotxn: &RoTxn,
        filled_tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        let transfer_data =
            filled_tx.transfer_reputation().ok_or_else(|| {
                Error::InvalidTransaction {
                    reason: "Not a reputation transfer transaction".to_string(),
                }
            })?;

        let sender_address = transfer_data.sender;

        let has_sender_input = filled_tx
            .spent_utxos
            .iter()
            .any(|utxo| utxo.address == sender_address);
        if !has_sender_input {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Reputation transfer from {sender_address} must spend \
                     a Bitcoin UTXO owned by the sender (replay protection)"
                ),
            });
        }

        if transfer_data.amount <= 0.0 {
            return Err(Error::InvalidTransaction {
                reason: "Transfer amount must be positive".to_string(),
            });
        }

        if !transfer_data.amount.is_finite() {
            return Err(Error::InvalidTransaction {
                reason: "Transfer amount must be a finite number".to_string(),
            });
        }

        if transfer_data.amount
            != crate::math::voting::constants::round_reputation(
                transfer_data.amount,
            )
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Transfer amount {} exceeds reputation precision \
                     (max {} decimal digits)",
                    transfer_data.amount,
                    crate::math::voting::constants::REPUTATION_PRECISION_DECIMALS,
                ),
            });
        }

        if sender_address == transfer_data.dest {
            return Err(Error::InvalidTransaction {
                reason: "Cannot transfer reputation to self".to_string(),
            });
        }

        let sender_reputation =
            state.reputation().get_reputation(rotxn, &sender_address)?;
        if sender_reputation < transfer_data.amount {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Insufficient reputation: have \
                     {sender_reputation}, need {}",
                    transfer_data.amount
                ),
            });
        }

        Ok(())
    }

    pub fn validate_ballot(
        state: &crate::state::State,
        rotxn: &RoTxn,
        filled_tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::state::{
            decisions::DecisionId, voting::types::VotingPeriodId,
        };

        let ballot_data = filled_tx.submit_ballot().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a ballot submission transaction".to_string(),
            }
        })?;

        let voter_address = ballot_data.voter;

        let has_voter_input = filled_tx
            .spent_utxos
            .iter()
            .any(|utxo| utxo.address == voter_address);
        let has_actor_proof = filled_tx.actor_address == Some(voter_address);
        if !has_voter_input && !has_actor_proof {
            return Err(Error::InvalidTransaction {
                reason: format!("No proof of voter {voter_address} identity"),
            });
        }

        Self::validate_voter_eligibility(state, rotxn, &voter_address)?;

        let mut seen_votes =
            std::collections::HashSet::<(VotingPeriodId, DecisionId)>::new();
        for (idx, vote_item) in ballot_data.votes.iter().enumerate() {
            let decision_id =
                DecisionId::from_bytes(vote_item.decision_id_bytes)?;

            // Voting period is deterministically derived from decision: voting_period = period_index + 1
            let period_id = VotingPeriodId::new(decision_id.voting_period());

            if !seen_votes.insert((period_id, decision_id)) {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Ballot item {idx}: duplicate vote for decision {decision_id:?} in period {period_id:?}"
                    ),
                });
            }

            let decision =
                Self::validate_decision_entry(state, rotxn, decision_id)
                    .map_err(|e| match e {
                        Error::InvalidDecisionId { reason } => {
                            Error::InvalidDecisionId {
                                reason: format!("Ballot item {idx}: {reason}"),
                            }
                        }
                        other => other,
                    })?;

            Self::validate_vote_value(&decision, vote_item.vote_value)
                .map_err(|e| match e {
                    Error::InvalidTransaction { reason } => {
                        Error::InvalidTransaction {
                            reason: format!("Ballot item {idx}: {reason}"),
                        }
                    }
                    other => other,
                })?;

            Self::validate_voting_period(state, rotxn, decision_id).map_err(
                |e| match e {
                    Error::InvalidTransaction { reason } => {
                        Error::InvalidTransaction {
                            reason: format!("Ballot item {idx}: {reason}"),
                        }
                    }
                    other => other,
                },
            )?;

            Self::validate_no_duplicate_vote(
                state,
                rotxn,
                voter_address,
                period_id,
                decision_id,
            )
            .map_err(|e| match e {
                Error::InvalidTransaction { reason } => {
                    Error::InvalidTransaction {
                        reason: format!("Ballot item {idx}: {reason}"),
                    }
                }
                other => other,
            })?;
        }

        Ok(())
    }
}

pub struct PeriodValidator;

impl PeriodValidator {
    pub fn validate_decision_in_period(
        period: &crate::state::voting::types::VotingPeriod,
        decision_id: crate::state::decisions::DecisionId,
    ) -> Result<(), Error> {
        if !period.decision_ids.contains(&decision_id) {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Decision {:?} not available in period {:?}",
                    decision_id, period.id
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::voting::types::VoteValue;

    #[test]
    fn convert_nan_returns_abstain() {
        assert_eq!(
            VoteValidator::convert_vote_value(f64::NAN),
            VoteValue::Abstain
        );
    }

    #[test]
    fn convert_zero_returns_binary_false() {
        assert_eq!(
            VoteValidator::convert_vote_value(0.0),
            VoteValue::Binary(false)
        );
    }

    #[test]
    fn convert_one_returns_binary_true() {
        assert_eq!(
            VoteValidator::convert_vote_value(1.0),
            VoteValue::Binary(true)
        );
    }

    #[test]
    fn convert_fractional_returns_scalar() {
        assert_eq!(
            VoteValidator::convert_vote_value(0.5),
            VoteValue::Scalar(0.5)
        );
    }

    #[test]
    fn convert_categorical_decision() {
        use crate::state::decisions::{Decision, DecisionType};

        let decision = Decision::new(
            [0u8; 20],
            DecisionType::Category {
                options: vec!["A".into(), "B".into(), "C".into()],
            },
            "Test".into(),
            String::new(),
            None,
            None,
            vec![],
        )
        .unwrap();

        assert_eq!(
            VoteValidator::convert_vote_value_with_decision(
                2.0,
                Some(&decision)
            ),
            VoteValue::Categorical(2)
        );
    }

    #[test]
    fn vote_value_in_range_ok() {
        use crate::state::decisions::{Decision, DecisionType};

        let d = Decision::new(
            [0u8; 20],
            DecisionType::Binary,
            "T".into(),
            String::new(),
            None,
            None,
            vec![],
        )
        .unwrap();

        assert!(VoteValidator::validate_vote_value(&d, 0.0).is_ok());
        assert!(VoteValidator::validate_vote_value(&d, 0.5).is_ok());
        assert!(VoteValidator::validate_vote_value(&d, 1.0).is_ok());
        assert!(VoteValidator::validate_vote_value(&d, f64::NAN).is_ok());
    }

    #[test]
    fn vote_value_out_of_range_err() {
        use crate::state::decisions::{Decision, DecisionType};

        let d = Decision::new(
            [0u8; 20],
            DecisionType::Binary,
            "T".into(),
            String::new(),
            None,
            None,
            vec![],
        )
        .unwrap();

        assert!(VoteValidator::validate_vote_value(&d, 1.5).is_err());
        assert!(VoteValidator::validate_vote_value(&d, -0.1).is_err());
    }

    #[test]
    fn categorical_vote_accepts_inconclusive_index() {
        use crate::state::decisions::{Decision, DecisionType};

        let d = Decision::new(
            [0u8; 20],
            DecisionType::Category {
                options: vec!["A".into(), "B".into(), "C".into()],
            },
            "T".into(),
            String::new(),
            None,
            None,
            vec![],
        )
        .unwrap();

        assert!(VoteValidator::validate_vote_value(&d, 0.0).is_ok());
        assert!(VoteValidator::validate_vote_value(&d, 2.0).is_ok());
        assert!(VoteValidator::validate_vote_value(&d, 3.0).is_ok());
        assert!(VoteValidator::validate_vote_value(&d, f64::NAN).is_ok());
    }

    #[test]
    fn categorical_vote_rejects_out_of_range() {
        use crate::state::decisions::{Decision, DecisionType};

        let d = Decision::new(
            [0u8; 20],
            DecisionType::Category {
                options: vec!["A".into(), "B".into(), "C".into()],
            },
            "T".into(),
            String::new(),
            None,
            None,
            vec![],
        )
        .unwrap();

        assert!(VoteValidator::validate_vote_value(&d, 4.0).is_err());
        assert!(VoteValidator::validate_vote_value(&d, -1.0).is_err());
        assert!(VoteValidator::validate_vote_value(&d, 1.5).is_err());
    }

    #[test]
    fn transfer_amount_at_precision_is_accepted() {
        use crate::math::voting::constants::round_reputation;
        let amount = 0.000_000_000_1_f64;
        assert_eq!(amount, round_reputation(amount));
    }

    #[test]
    fn transfer_amount_above_precision_is_rejected() {
        use crate::math::voting::constants::round_reputation;
        let amount = 0.000_000_000_01_f64;
        assert_ne!(amount, round_reputation(amount));
    }

    #[test]
    fn transfer_amount_subprecision_is_rejected() {
        use crate::math::voting::constants::round_reputation;
        let amount = 1e-15_f64;
        assert_ne!(amount, round_reputation(amount));
    }

    #[test]
    fn transfer_amount_whole_value_is_accepted() {
        use crate::math::voting::constants::round_reputation;
        let amount = 0.5_f64;
        assert_eq!(amount, round_reputation(amount));
    }
}
