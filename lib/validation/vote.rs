use crate::state::Error;
use crate::types::FilledTransaction;
use sneed::RoTxn;
use std::collections::HashSet;

pub(crate) fn ballot_key(
    transaction: &crate::types::Transaction,
) -> Option<(u32, crate::types::Address, bool)> {
    use crate::types::TxData;

    match &transaction.data {
        Some(TxData::SubmitBallot {
            voter,
            voting_period,
            ..
        }) => Some((*voting_period, *voter, false)),
        Some(TxData::RevealBallot {
            voter,
            voting_period,
            ..
        }) => Some((*voting_period, *voter, true)),
        Some(
            TxData::ClaimDecision(_)
            | TxData::CreateMarket { .. }
            | TxData::Trade { .. }
            | TxData::TransferReputation { .. }
            | TxData::AmplifyBeta { .. },
        )
        | None => None,
    }
}

/// Reject a block with two ballots of one voter in the same period and
/// phase, or with a sealed ballot and a reputation transfer that touches its
/// voter, since each is checked against the state before the block.
pub(crate) fn check_ballot_conflicts(
    transactions: &[crate::types::Transaction],
) -> Result<(), Error> {
    use crate::types::TxData;

    let mut seen = HashSet::new();
    let mut sealing_voters = HashSet::new();
    let mut transfer_parties = HashSet::new();
    for transaction in transactions {
        if let Some(key) = ballot_key(transaction) {
            if !seen.insert(key) {
                return Err(Error::InvalidTransaction {
                    reason: "Duplicate ballot in block".to_string(),
                });
            }
            let (_, voter, is_reveal) = key;
            if !is_reveal {
                sealing_voters.insert(voter);
            }
        }
        if let Some(TxData::TransferReputation { sender, dest, .. }) =
            &transaction.data
        {
            transfer_parties.insert(*sender);
            transfer_parties.insert(*dest);
        }
    }
    match sealing_voters.intersection(&transfer_parties).next() {
        Some(voter) => Err(Error::InvalidTransaction {
            reason: format!(
                "Block seals a ballot of {voter} and transfers its reputation"
            ),
        }),
        None => Ok(()),
    }
}

pub const MIN_BALLOT_FEERATE_SATS_PER_BYTE: u64 = 1;

/// Minimum fee of a ballot commit or reveal, by the serialized size of its
/// inputs, outputs and data. The utreexo proof is left out, because the node
/// regenerates it as the chain grows.
pub fn min_ballot_fee(
    transaction: &crate::types::Transaction,
) -> Result<bitcoin::Amount, bincode::Error> {
    let size = bincode::serialized_size(&(
        &transaction.inputs,
        &transaction.outputs,
        &transaction.data,
    ))?;
    Ok(bitcoin::Amount::from_sat(
        size * MIN_BALLOT_FEERATE_SATS_PER_BYTE,
    ))
}

fn validate_ballot_fee(filled_tx: &FilledTransaction) -> Result<(), Error> {
    let fee = super::tx_fee(filled_tx)?;
    let min_fee = min_ballot_fee(&filled_tx.transaction)
        .map_err(Error::BincodeSerialize)?;
    if fee < min_fee {
        return Err(Error::InvalidTransaction {
            reason: format!("Ballot fee {fee} is below the minimum {min_fee}"),
        });
    }
    Ok(())
}

pub struct VoteValidator;

impl VoteValidator {
    pub fn convert_vote_value(
        vote_value: Option<f64>,
    ) -> crate::state::voting::types::VoteValue {
        Self::convert_vote_value_with_decision(vote_value, None)
    }

    pub fn convert_vote_value_with_decision(
        vote_value: Option<f64>,
        decision: Option<&crate::state::decisions::Decision>,
    ) -> crate::state::voting::types::VoteValue {
        use crate::state::voting::types::VoteValue;

        let Some(vote_value) = vote_value else {
            return VoteValue::Abstain;
        };

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
        if reputation.is_zero() {
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
            return Err(Error::InvalidTransaction {
                reason: "Vote value is NaN; omit the value to abstain"
                    .to_string(),
            });
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

    pub(crate) fn validate_actor_identity(
        filled_tx: &FilledTransaction,
        actor_address: crate::types::Address,
    ) -> Result<(), Error> {
        let has_actor_input = filled_tx
            .spent_utxos
            .iter()
            .any(|utxo| utxo.address == actor_address);
        let has_actor_proof = filled_tx.actor_address == Some(actor_address);
        if !has_actor_input && !has_actor_proof {
            return Err(Error::InvalidTransaction {
                reason: format!("No proof of {actor_address} identity"),
            });
        }
        Ok(())
    }

    pub fn period_decisions_in_state(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voting_period: u32,
        decision_state: crate::state::decisions::DecisionState,
    ) -> Result<HashSet<crate::state::decisions::DecisionId>, Error> {
        let decision_ids =
            crate::state::voting::period_calculator::get_decision_ids_for_period(
                rotxn,
                crate::state::voting::types::VotingPeriodId::new(voting_period),
                state.decisions(),
            )?;
        let mut in_state = HashSet::new();
        for decision_id in decision_ids {
            if state
                .decisions()
                .get_decision_current_state(rotxn, decision_id)?
                == decision_state
            {
                in_state.insert(decision_id);
            }
        }
        Ok(in_state)
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

        if transfer_data.amount.is_zero() {
            return Err(Error::InvalidTransaction {
                reason: "Transfer amount must be positive".to_string(),
            });
        }

        if sender_address == transfer_data.dest {
            return Err(Error::InvalidTransaction {
                reason: "Cannot transfer reputation to self".to_string(),
            });
        }

        Self::ensure_reputation_unlocked(
            state,
            rotxn,
            &[sender_address, transfer_data.dest],
        )?;

        let sender_reputation =
            state.reputation().get_reputation(rotxn, &sender_address)?;
        let receiver_reputation = state
            .reputation()
            .get_reputation(rotxn, &transfer_data.dest)?;
        Self::balances_after_transfer(
            sender_reputation,
            receiver_reputation,
            transfer_data.amount,
        )?;

        Ok(())
    }

    pub(crate) fn ensure_reputation_unlocked(
        state: &crate::state::State,
        rotxn: &RoTxn,
        addresses: &[crate::types::Address],
    ) -> Result<(), Error> {
        for period in state.decisions().get_open_voting_periods(rotxn)? {
            let period_id =
                crate::state::voting::types::VotingPeriodId::new(period);
            for &address in addresses {
                if state
                    .voting()
                    .databases()
                    .get_sealed_ballot(rotxn, period_id, address)?
                    .is_some()
                {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "Reputation of {address} is locked by its ballot \
                             for period {period}"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    pub(crate) fn balances_after_transfer(
        sender_reputation: crate::types::Votecoin,
        receiver_reputation: crate::types::Votecoin,
        amount: crate::types::Votecoin,
    ) -> Result<(crate::types::Votecoin, crate::types::Votecoin), Error> {
        let sender_after =
            sender_reputation.checked_sub(amount).ok_or_else(|| {
                Error::InvalidTransaction {
                    reason: format!(
                        "Insufficient votecoin: have {sender_reputation}, \
                         need {amount}"
                    ),
                }
            })?;
        let receiver_after = receiver_reputation
            .checked_add(amount)
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!(
                    "Votecoin balance overflow: {receiver_reputation} + \
                         {amount}"
                ),
            })?;
        Ok((sender_after, receiver_after))
    }

    pub fn validate_ballot(
        state: &crate::state::State,
        rotxn: &RoTxn,
        filled_tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::state::decisions::DecisionState;

        let ballot = filled_tx.submit_ballot().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a ballot submission transaction".to_string(),
            }
        })?;

        Self::validate_actor_identity(filled_tx, ballot.voter)?;
        Self::validate_voter_eligibility(state, rotxn, &ballot.voter)?;
        validate_ballot_fee(filled_tx)?;

        if Self::period_decisions_in_state(
            state,
            rotxn,
            ballot.voting_period,
            DecisionState::Voting,
        )?
        .is_empty()
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Voting period {} is not open for ballots",
                    ballot.voting_period
                ),
            });
        }

        Self::validate_first_ballot(
            state,
            rotxn,
            ballot.voter,
            ballot.voting_period,
        )
    }

    fn validate_first_ballot(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voter: crate::types::Address,
        voting_period: u32,
    ) -> Result<(), Error> {
        if state
            .voting()
            .databases()
            .get_sealed_ballot(
                rotxn,
                crate::state::voting::types::VotingPeriodId::new(voting_period),
                voter,
            )?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Voter {voter} already submitted a ballot for period \
                     {voting_period}"
                ),
            });
        }
        Ok(())
    }

    /// The ballot votes at most once on each decision of `expected`, with a
    /// valid value or an abstain, and on every standard decision of it
    fn validate_ballot_votes(
        state: &crate::state::State,
        rotxn: &RoTxn,
        votes: &[crate::types::BallotItem],
        expected: &HashSet<crate::state::decisions::DecisionId>,
    ) -> Result<(), Error> {
        use crate::state::decisions::DecisionId;

        let mut seen = HashSet::new();
        for (idx, vote_item) in votes.iter().enumerate() {
            let decision_id =
                DecisionId::from_bytes(vote_item.decision_id_bytes)?;
            if !expected.contains(&decision_id) {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Ballot item {idx}: decision {} is not part of the \
                         ballot's period",
                        decision_id.to_hex()
                    ),
                });
            }
            if !seen.insert(decision_id) {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Ballot item {idx}: duplicate vote for decision {}",
                        decision_id.to_hex()
                    ),
                });
            }
            if let Some(vote_value) = vote_item.vote_value {
                let decision =
                    Self::validate_decision_entry(state, rotxn, decision_id)?;
                Self::validate_vote_value(&decision, vote_value)?;
            }
        }
        if let Some(missing) = expected
            .iter()
            .find(|id| id.is_standard() && !seen.contains(id))
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Ballot omits standard decision {} of its period",
                    missing.to_hex()
                ),
            });
        }
        Ok(())
    }

    /// Check a ballot before sealing it, against the rules its reveal must
    /// pass
    pub fn prepare_ballot(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voter: crate::types::Address,
        voting_period: u32,
        votes: &[crate::types::BallotItem],
    ) -> Result<(), Error> {
        use crate::state::decisions::DecisionState;

        Self::validate_voter_eligibility(state, rotxn, &voter)?;
        let open = Self::period_decisions_in_state(
            state,
            rotxn,
            voting_period,
            DecisionState::Voting,
        )?;
        if open.is_empty() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Voting period {voting_period} is not open for ballots"
                ),
            });
        }
        Self::validate_ballot_votes(state, rotxn, votes, &open)?;
        Self::validate_first_ballot(state, rotxn, voter, voting_period)
    }

    pub fn validate_reveal(
        state: &crate::state::State,
        rotxn: &RoTxn,
        filled_tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::state::{
            decisions::DecisionState, voting::types::VotingPeriodId,
        };

        let reveal = filled_tx.reveal_ballot().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a ballot reveal transaction".to_string(),
            }
        })?;

        Self::validate_actor_identity(filled_tx, reveal.voter)?;
        validate_ballot_fee(filled_tx)?;

        let revealing = Self::period_decisions_in_state(
            state,
            rotxn,
            reveal.voting_period,
            DecisionState::Revealing,
        )?;
        if revealing.is_empty() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Voting period {} is not revealing",
                    reveal.voting_period
                ),
            });
        }

        let sealed_ballot = state
            .voting()
            .databases()
            .get_sealed_ballot(
                rotxn,
                VotingPeriodId::new(reveal.voting_period),
                reveal.voter,
            )?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!(
                    "No sealed ballot for voter {} in period {}",
                    reveal.voter, reveal.voting_period
                ),
            })?;
        if sealed_ballot.revealed {
            return Err(Error::InvalidTransaction {
                reason: "Ballot is already revealed".to_string(),
            });
        }
        if reveal.commitment() != sealed_ballot.commitment {
            return Err(Error::InvalidTransaction {
                reason: "Reveal does not match the sealed ballot".to_string(),
            });
        }

        Self::validate_ballot_votes(state, rotxn, &reveal.votes, &revealing)
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

    /// A missing vote value converts to an abstention
    #[test]
    fn convert_none() {
        assert_eq!(VoteValidator::convert_vote_value(None), VoteValue::Abstain);
    }

    /// 0.0 converts to a binary false vote
    #[test]
    fn convert_zero() {
        assert_eq!(
            VoteValidator::convert_vote_value(Some(0.0)),
            VoteValue::Binary(false)
        );
    }

    /// 1.0 converts to a binary true vote
    #[test]
    fn convert_one() {
        assert_eq!(
            VoteValidator::convert_vote_value(Some(1.0)),
            VoteValue::Binary(true)
        );
    }

    /// A fractional value converts to a scalar vote
    #[test]
    fn convert_fractional() {
        assert_eq!(
            VoteValidator::convert_vote_value(Some(0.5)),
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
                Some(2.0),
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
        assert!(VoteValidator::validate_vote_value(&d, f64::NAN).is_err());
    }

    /// The index past the last option (inconclusive) is a valid vote
    #[test]
    fn categorical_inconclusive_ok() {
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
    }

    /// Indices at and just below the option limit validate and convert intact
    #[test]
    fn categorical_limit_indices() {
        use crate::state::decisions::{Decision, DecisionType};

        let decision = Decision::new(
            [0; 20],
            DecisionType::Category {
                options: vec![
                    String::new();
                    crate::state::decisions::MAX_CATEGORY_OPTIONS
                ],
            },
            String::new(),
            String::new(),
            None,
            None,
            vec![],
        )
        .unwrap();
        let limit = crate::state::decisions::MAX_CATEGORY_OPTIONS as u16;
        for index in [limit - 1, limit] {
            let value = f64::from(index);
            assert!(
                VoteValidator::validate_vote_value(&decision, value).is_ok()
            );
            assert_eq!(
                VoteValidator::convert_vote_value_with_decision(
                    Some(value),
                    Some(&decision),
                ),
                VoteValue::Categorical(index),
            );
        }
        assert!(
            VoteValidator::validate_vote_value(
                &decision,
                f64::from(limit) + 1.0
            )
            .is_err()
        );
    }

    #[test]
    fn categorical_out_of_range_err() {
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

    /// Debits the sender, credits the receiver, refuses overdraft and overflow
    #[test]
    fn transfer_balances() {
        use crate::types::Votecoin;

        let after = VoteValidator::balances_after_transfer;

        assert_eq!(
            after(Votecoin(300_000), Votecoin::ZERO, Votecoin(100_000))
                .unwrap(),
            (Votecoin(200_000), Votecoin(100_000))
        );
        assert_eq!(
            after(Votecoin(250_000), Votecoin(500_000), Votecoin(250_000))
                .unwrap(),
            (Votecoin::ZERO, Votecoin(750_000))
        );
        assert_eq!(
            after(Votecoin(1), Votecoin::ZERO, Votecoin(1)).unwrap(),
            (Votecoin::ZERO, Votecoin(1))
        );
        assert!(after(Votecoin(1), Votecoin::ZERO, Votecoin(2)).is_err());
        assert!(after(Votecoin(1), Votecoin(u64::MAX), Votecoin(1)).is_err());
    }

    struct VotingFixture {
        _dir: tempfile::TempDir,
        env: sneed::Env,
        state: crate::state::State,
        voter: crate::types::Address,
        decision_ids: [crate::state::decisions::DecisionId; 3],
    }

    const VOTING_PERIOD: u32 = 2;

    fn voting_fixture() -> VotingFixture {
        use crate::state::decisions::{Decision, DecisionId, DecisionType};

        let dir = tempfile::tempdir().unwrap();
        let env_path = dir.path().join("data.mdb");
        std::fs::create_dir_all(&env_path).unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024)
            .max_dbs(crate::state::State::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, &env_path) }.unwrap();
        let state = crate::state::State::new(&env, None, true).unwrap();

        let voter = crate::types::Address([1; 20]);
        let decision_ids = [
            DecisionId::new(true, VOTING_PERIOD - 1, 0).unwrap(),
            DecisionId::new(true, VOTING_PERIOD - 1, 1).unwrap(),
            DecisionId::new(false, VOTING_PERIOD - 1, 0).unwrap(),
        ];

        let mut rwtxn = env.write_txn().unwrap();
        state
            .decisions()
            .mint_genesis(&mut rwtxn, 1000, 0, 1000)
            .unwrap();
        state
            .reputation()
            .set_reputation(&mut rwtxn, &voter, crate::types::Votecoin::TOTAL)
            .unwrap();
        for decision_id in decision_ids {
            let decision = Decision::new(
                [0u8; 20],
                DecisionType::Binary,
                "T".into(),
                String::new(),
                None,
                None,
                vec![],
            )
            .unwrap();
            state
                .decisions()
                .claim_decision(
                    &mut rwtxn,
                    decision_id,
                    decision,
                    crate::types::Txid([0; 32]),
                    None,
                )
                .unwrap();
            state
                .decisions()
                .transition_decision_to_voting(&mut rwtxn, decision_id, 0)
                .unwrap();
        }
        rwtxn.commit().unwrap();

        VotingFixture {
            _dir: dir,
            env,
            state,
            voter,
            decision_ids,
        }
    }

    impl VotingFixture {
        fn seal(&self, commitment: crate::types::Hash, revealed: bool) {
            let mut rwtxn = self.env.write_txn().unwrap();
            self.state
                .voting()
                .databases()
                .put_sealed_ballot(
                    &mut rwtxn,
                    crate::state::voting::types::VotingPeriodId::new(
                        VOTING_PERIOD,
                    ),
                    self.voter,
                    &crate::state::voting::types::SealedBallot {
                        commitment,
                        revealed,
                    },
                )
                .unwrap();
            rwtxn.commit().unwrap();
        }

        fn close_voting(&self) {
            let mut rwtxn = self.env.write_txn().unwrap();
            for decision_id in self.decision_ids {
                self.state
                    .decisions()
                    .transition_decision_to_revealing(
                        &mut rwtxn,
                        decision_id,
                        1,
                    )
                    .unwrap();
            }
            rwtxn.commit().unwrap();
        }

        fn submit(&self, fee_sats: u64) -> FilledTransaction {
            filled_tx(
                self.voter,
                crate::types::TransactionData::SubmitBallot {
                    voter: self.voter,
                    voting_period: VOTING_PERIOD,
                    commitment: [7; 32],
                },
                fee_sats,
            )
        }

        fn reveal(
            &self,
            votes: Vec<crate::types::BallotItem>,
        ) -> crate::types::RevealBallot {
            crate::types::RevealBallot {
                voter: self.voter,
                voting_period: VOTING_PERIOD,
                votes,
                salt: [9; 32],
            }
        }

        fn item(
            &self,
            idx: usize,
            vote_value: Option<f64>,
        ) -> crate::types::BallotItem {
            crate::types::BallotItem {
                decision_id_bytes: self.decision_ids[idx].as_bytes(),
                vote_value,
            }
        }
    }

    const FEE_SATS: u64 = 10_000;

    /// A transaction that spends one coin of `voter` and pays all of it as
    /// fee
    fn filled_tx(
        voter: crate::types::Address,
        data: crate::types::TransactionData,
        fee_sats: u64,
    ) -> FilledTransaction {
        use crate::types::{OutPoint, Output, OutputContent, PointedOutputRef};

        let input = Output {
            address: voter,
            content: OutputContent::Value(bitcoin::Amount::from_sat(fee_sats)),
        };
        let outpoint = OutPoint::Regular {
            txid: crate::types::Txid([3; 32]),
            vout: 0,
        };
        let input_hash = crate::types::hash(&PointedOutputRef {
            outpoint,
            output: &input,
        });
        FilledTransaction {
            transaction: crate::types::Transaction {
                inputs: vec![(outpoint, input_hash)].into(),
                data: Some(data),
                ..Default::default()
            },
            spent_utxos: vec![input],
            actor_address: Some(voter),
        }
    }

    fn reveal_tx_with_fee(
        reveal: &crate::types::RevealBallot,
        fee_sats: u64,
    ) -> FilledTransaction {
        filled_tx(
            reveal.voter,
            crate::types::TransactionData::RevealBallot {
                voter: reveal.voter,
                voting_period: reveal.voting_period,
                votes: reveal.votes.clone(),
                salt: reveal.salt,
            },
            fee_sats,
        )
    }

    fn reveal_tx(reveal: &crate::types::RevealBallot) -> FilledTransaction {
        reveal_tx_with_fee(reveal, FEE_SATS)
    }

    fn min_fee_sats(tx: &FilledTransaction) -> u64 {
        super::min_ballot_fee(&tx.transaction).unwrap().to_sat()
    }

    #[test]
    fn one_seal_per_period() {
        let f = voting_fixture();
        let validate = |tx: &FilledTransaction| {
            let rotxn = f.env.read_txn().unwrap();
            VoteValidator::validate_ballot(&f.state, &rotxn, tx, None)
        };

        validate(&f.submit(FEE_SATS)).unwrap();
        f.seal([1; 32], false);
        assert!(validate(&f.submit(FEE_SATS)).is_err());
    }

    #[test]
    fn sealing_closes_with_voting() {
        let f = voting_fixture();
        f.close_voting();
        let rotxn = f.env.read_txn().unwrap();
        assert!(
            VoteValidator::validate_ballot(
                &f.state,
                &rotxn,
                &f.submit(FEE_SATS),
                None
            )
            .is_err()
        );
    }

    /// Seal and reveal txs must pay the minimum fee for their size
    #[test]
    fn ballot_minimum_fee() {
        let f = voting_fixture();
        let min_seal_fee = min_fee_sats(&f.submit(FEE_SATS));
        let validate_seal = |fee_sats| {
            let rotxn = f.env.read_txn().unwrap();
            VoteValidator::validate_ballot(
                &f.state,
                &rotxn,
                &f.submit(fee_sats),
                None,
            )
        };
        assert!(min_seal_fee > 0);
        assert!(validate_seal(min_seal_fee - 1).is_err());
        validate_seal(min_seal_fee).unwrap();

        let reveal = f.reveal(vec![f.item(0, Some(1.0)), f.item(1, None)]);
        f.seal(reveal.commitment(), false);
        f.close_voting();
        let min_reveal_fee = min_fee_sats(&reveal_tx(&reveal));
        let validate_reveal = |fee_sats| {
            let rotxn = f.env.read_txn().unwrap();
            VoteValidator::validate_reveal(
                &f.state,
                &rotxn,
                &reveal_tx_with_fee(&reveal, fee_sats),
                None,
            )
        };
        assert!(min_reveal_fee > min_seal_fee);
        assert!(validate_reveal(min_reveal_fee - 1).is_err());
        validate_reveal(min_reveal_fee).unwrap();
    }

    /// A reveal is only valid once voting has closed
    #[test]
    fn reveal_after_voting() {
        let f = voting_fixture();
        let reveal = f.reveal(vec![f.item(0, Some(1.0)), f.item(1, None)]);
        f.seal(reveal.commitment(), false);
        let validate = || {
            let rotxn = f.env.read_txn().unwrap();
            VoteValidator::validate_reveal(
                &f.state,
                &rotxn,
                &reveal_tx(&reveal),
                None,
            )
        };

        assert!(validate().is_err());
        f.close_voting();
        validate().unwrap();
    }

    /// A reveal must match an unrevealed seal's commitment
    #[test]
    fn reveal_opens_seal() {
        let f = voting_fixture();
        let reveal = f.reveal(vec![f.item(0, Some(1.0)), f.item(1, Some(0.0))]);
        f.close_voting();
        let validate = |reveal: &crate::types::RevealBallot| {
            let rotxn = f.env.read_txn().unwrap();
            VoteValidator::validate_reveal(
                &f.state,
                &rotxn,
                &reveal_tx(reveal),
                None,
            )
        };

        assert!(validate(&reveal).is_err());
        f.seal(reveal.commitment(), false);
        let forged = crate::types::RevealBallot {
            salt: [8; 32],
            ..reveal.clone()
        };
        assert!(validate(&forged).is_err());
        let flipped = crate::types::RevealBallot {
            votes: vec![f.item(0, Some(0.0)), f.item(1, Some(0.0))],
            ..reveal.clone()
        };
        assert!(validate(&flipped).is_err());
        validate(&reveal).unwrap();
        f.seal(reveal.commitment(), true);
        assert!(validate(&reveal).is_err());
    }

    /// A ballot may omit a non-standard decision but not a standard one
    #[test]
    fn omit_nonstandard_decision() {
        let f = voting_fixture();
        let prepare = |votes: Vec<crate::types::BallotItem>| {
            let rotxn = f.env.read_txn().unwrap();
            VoteValidator::prepare_ballot(
                &f.state,
                &rotxn,
                f.voter,
                VOTING_PERIOD,
                &votes,
            )
        };

        prepare(vec![f.item(0, Some(1.0)), f.item(1, None)]).unwrap();
        prepare(vec![
            f.item(0, Some(1.0)),
            f.item(1, None),
            f.item(2, Some(0.0)),
        ])
        .unwrap();
        assert!(prepare(vec![f.item(0, Some(1.0)), f.item(2, None)]).is_err());
        assert!(
            prepare(vec![
                f.item(0, Some(1.0)),
                f.item(1, None),
                f.item(2, None),
                f.item(2, None),
            ])
            .is_err()
        );
    }

    /// A reveal must cover every decision exactly once with valid values
    #[test]
    fn reveal_covers_decisions() {
        let f = voting_fixture();
        f.close_voting();
        let validate = |votes: Vec<crate::types::BallotItem>| {
            let reveal = f.reveal(votes);
            f.seal(reveal.commitment(), false);
            let rotxn = f.env.read_txn().unwrap();
            VoteValidator::validate_reveal(
                &f.state,
                &rotxn,
                &reveal_tx(&reveal),
                None,
            )
        };

        assert!(validate(vec![f.item(0, Some(1.0))]).is_err());
        assert!(
            validate(vec![f.item(0, Some(1.0)), f.item(0, Some(1.0))]).is_err()
        );
        assert!(
            validate(vec![f.item(0, Some(1.5)), f.item(1, Some(1.0))]).is_err()
        );
        validate(vec![f.item(0, None), f.item(1, None)]).unwrap();
    }

    /// prepare_ballot applies the reveal rules and refuses a sealed voter
    #[test]
    fn prepare_ballot_rules() {
        let f = voting_fixture();
        let prepare = |votes: Vec<crate::types::BallotItem>| {
            let rotxn = f.env.read_txn().unwrap();
            VoteValidator::prepare_ballot(
                &f.state,
                &rotxn,
                f.voter,
                VOTING_PERIOD,
                &votes,
            )
        };
        let full = || vec![f.item(0, Some(1.0)), f.item(1, None)];

        prepare(full()).unwrap();
        assert!(prepare(vec![f.item(0, Some(1.0))]).is_err());
        assert!(
            prepare(vec![f.item(0, Some(f64::NAN)), f.item(1, None)]).is_err()
        );
        assert!(prepare(vec![f.item(0, Some(1.5)), f.item(1, None)]).is_err());

        f.seal([1; 32], false);
        assert!(prepare(full()).is_err());
    }

    /// A block holds one ballot per voter, period and phase
    #[test]
    fn one_ballot_per_block() {
        use crate::types::{Address, Transaction, TxData};

        let voter = Address([1; 20]);
        let seal = |voter| Transaction {
            data: Some(TxData::SubmitBallot {
                voter,
                voting_period: VOTING_PERIOD,
                commitment: [1; 32],
            }),
            ..Default::default()
        };
        let reveal = Transaction {
            data: Some(TxData::RevealBallot {
                voter,
                voting_period: VOTING_PERIOD - 1,
                votes: vec![],
                salt: [0; 32],
            }),
            ..Default::default()
        };
        super::check_ballot_conflicts(&[
            seal(voter),
            reveal,
            seal(Address([2; 20])),
            Transaction::default(),
        ])
        .unwrap();
        assert!(
            super::check_ballot_conflicts(&[seal(voter), seal(voter)]).is_err()
        );
    }

    /// A block cannot seal a ballot and move the voter's reputation
    #[test]
    fn seal_transfer_conflict() {
        use crate::types::{Address, Transaction, TxData};

        let voter = Address([1; 20]);
        let other = Address([2; 20]);
        let seal = Transaction {
            data: Some(TxData::SubmitBallot {
                voter,
                voting_period: VOTING_PERIOD,
                commitment: [1; 32],
            }),
            ..Default::default()
        };
        let transfer = |sender, dest| Transaction {
            data: Some(TxData::TransferReputation {
                sender,
                dest,
                amount: crate::types::Votecoin(100_000),
            }),
            ..Default::default()
        };
        assert!(
            super::check_ballot_conflicts(&[
                seal.clone(),
                transfer(voter, other)
            ])
            .is_err()
        );
        assert!(
            super::check_ballot_conflicts(&[
                transfer(other, voter),
                seal.clone()
            ])
            .is_err()
        );
        super::check_ballot_conflicts(&[
            seal,
            transfer(other, Address([3; 20])),
        ])
        .unwrap();
    }
}
