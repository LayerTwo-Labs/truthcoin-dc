pub mod decision;
pub mod market;
pub mod vote;

use std::collections::HashSet;

use sneed::RoTxn;

use crate::{
    math::trading::TRADE_MINER_FEE_SATS,
    state::{
        Error, State,
        decisions::{Decision, DecisionId},
    },
    types::{
        AmountOverflowError, ComputeFeeError, FilledTransaction, Output,
        OutputContent, Transaction, TransactionData,
    },
};

pub use decision::DecisionValidator;
pub use market::{MarketStateValidator, MarketValidator};
pub use vote::{PeriodValidator, VoteValidator};

pub trait DecisionValidationInterface {
    fn validate_decision_claim(
        &self,
        rotxn: &RoTxn,
        decision_id: DecisionId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<(), Error>;

    fn try_get_height(&self, rotxn: &RoTxn) -> Result<Option<u32>, Error>;

    fn try_get_genesis_timestamp(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error>;

    fn try_get_mainchain_timestamp(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error>;

    fn get_standard_claimed_count_in_period(
        &self,
        rotxn: &RoTxn,
        period_index: u32,
    ) -> Result<u64, Error>;

    fn get_available_decisions(
        &self,
        rotxn: &RoTxn,
        period: u32,
        current_ts: u64,
        current_height: Option<u32>,
        genesis_ts: u64,
    ) -> Result<u64, Error>;

    fn fee_for_decision_id(
        &self,
        rotxn: &RoTxn,
        decision_id: DecisionId,
    ) -> Result<u64, Error>;
}

/// Value in minus value out
pub(crate) fn tx_fee(tx: &FilledTransaction) -> Result<bitcoin::Amount, Error> {
    tx.get_fee().map_err(|err| match err {
        ComputeFeeError::Underfunded => Error::NotEnoughValueIn,
        ComputeFeeError::ValueInOverflow(err)
        | ComputeFeeError::ValueOutOverflow(err) => err.into(),
    })
}

/// Fee that a transaction pays to the block producer
pub fn miner_fee(
    filled_tx: &FilledTransaction,
) -> Result<bitcoin::Amount, Error> {
    if filled_tx.is_trade() || filled_tx.is_amplify_beta() {
        return Ok(bitcoin::Amount::from_sat(TRADE_MINER_FEE_SATS));
    }
    tx_fee(filled_tx)
}

/// Run the market checks for a transaction. Returns the miner fee of a trade
/// or an amplify beta transaction, and `None` for any other transaction.
pub(crate) fn validate_market_transaction(
    state: &State,
    rotxn: &RoTxn,
    tx: &FilledTransaction,
    archive: &crate::archive::Archive,
    override_height: Option<u32>,
) -> Result<Option<bitcoin::Amount>, Error> {
    if tx.is_claim_decision() {
        DecisionValidator::validate_complete_decision_claim(
            state,
            rotxn,
            tx,
            override_height,
        )?;
    }
    if tx.is_create_market() {
        MarketValidator::validate_market_creation(
            state,
            rotxn,
            tx,
            override_height,
        )?;
    }
    if tx.is_trade() {
        MarketValidator::validate_trade(
            state,
            archive,
            rotxn,
            tx,
            override_height,
        )?;
        return Ok(Some(bitcoin::Amount::from_sat(TRADE_MINER_FEE_SATS)));
    }
    if tx.is_amplify_beta() {
        MarketValidator::validate_amplify_beta(
            state,
            rotxn,
            tx,
            override_height,
        )?;
        return Ok(Some(bitcoin::Amount::from_sat(TRADE_MINER_FEE_SATS)));
    }
    if tx.is_submit_vote() {
        VoteValidator::validate_vote_submission(
            state,
            rotxn,
            tx,
            override_height,
        )?;
    }
    if tx.is_submit_ballot() {
        VoteValidator::validate_ballot(state, rotxn, tx, override_height)?;
    }
    if tx.is_transfer_reputation() {
        VoteValidator::validate_reputation_transfer(
            state,
            rotxn,
            tx,
            override_height,
        )?;
    }
    Ok(None)
}

/// Reject a block claiming the same decision id more than once, across both
/// `ClaimDecision` transactions and `CreateMarket` new-claim payloads.
pub fn check_duplicate_decision_claims(
    transactions: &[Transaction],
) -> Result<(), Error> {
    let mut claimed_decision_ids = HashSet::new();
    for tx in transactions {
        let payloads = match &tx.data {
            Some(TransactionData::ClaimDecision(payload)) => {
                std::slice::from_ref(payload)
            }
            Some(TransactionData::CreateMarket { new_claims, .. }) => {
                new_claims.as_slice()
            }
            _ => &[],
        };
        for payload in payloads {
            for entry in &payload.decisions {
                if !claimed_decision_ids.insert(entry.decision_id_bytes) {
                    return Err(Error::DuplicateDecisionClaim(
                        entry.decision_id_bytes,
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Reject market funds in the coinbase
pub fn validate_coinbase_outputs(coinbase: &[Output]) -> Result<(), Error> {
    for output in coinbase {
        match &output.content {
            OutputContent::Value(_) | OutputContent::Withdrawal { .. } => {}
            OutputContent::MarketFunds { .. } => {
                return Err(Error::BadCoinbaseOutputContent);
            }
        }
    }
    Ok(())
}

/// Reject a coinbase worth more than the miner fees of the applied
/// transactions
pub fn validate_fees(
    coinbase_value: bitcoin::Amount,
    filled_txs: &[FilledTransaction],
    skipped_indices: &HashSet<usize>,
) -> Result<(), Error> {
    let mut actual_total_fees = bitcoin::Amount::ZERO;
    for (idx, filled_tx) in filled_txs.iter().enumerate() {
        if skipped_indices.contains(&idx) {
            continue;
        }
        actual_total_fees = actual_total_fees
            .checked_add(miner_fee(filled_tx)?)
            .ok_or(AmountOverflowError)?;
    }
    if coinbase_value > actual_total_fees {
        return Err(Error::NotEnoughFees);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::types::{
        Address, ClaimDecisionPayload, DecisionClaimEntry, Output,
        OutputContent, Transaction, TransactionData,
    };

    fn bitcoin_output(sats: u64) -> Output {
        Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::Value(bitcoin::Amount::from_sat(sats)),
        }
    }

    #[test]
    fn coinbase_bitcoin_output_always_valid() {
        let outputs = vec![bitcoin_output(5000)];
        assert!(super::validate_coinbase_outputs(&outputs).is_ok());
    }

    #[test]
    fn coinbase_market_funds_always_rejected() {
        let outputs = vec![Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::MarketFunds {
                market_id: [0u8; 6],
                amount: bitcoin::Amount::from_sat(1000),
                is_fee: false,
            },
        }];
        assert!(super::validate_coinbase_outputs(&outputs).is_err());
    }

    #[test]
    fn validate_fees_exact_match_ok() {
        let fees = bitcoin::Amount::from_sat(0);
        assert!(super::validate_fees(fees, &[], &HashSet::new()).is_ok());
    }

    #[test]
    fn validate_fees_coinbase_exceeds_fees_rejected() {
        let coinbase = bitcoin::Amount::from_sat(1000);
        assert!(super::validate_fees(coinbase, &[], &HashSet::new()).is_err());
    }

    fn claim_tx(ids: &[[u8; 3]]) -> Transaction {
        use crate::state::decisions::DecisionType;
        let decisions = ids
            .iter()
            .map(|id| DecisionClaimEntry {
                decision_id_bytes: *id,
                header: String::new(),
                description: String::new(),
                option_0_label: None,
                option_1_label: None,
                option_labels: None,
                tags: None,
            })
            .collect();
        Transaction {
            data: Some(TransactionData::ClaimDecision(ClaimDecisionPayload {
                decision_type: DecisionType::Binary,
                decisions,
            })),
            ..Transaction::default()
        }
    }

    #[test]
    fn miner_fee_excludes_amplify_beta_deposit() {
        use crate::{
            math::trading::TRADE_MINER_FEE_SATS,
            state::markets::MarketId,
            types::{FilledTransaction, TxData},
        };

        let amount = 50_000;
        let amplify = FilledTransaction {
            transaction: Transaction {
                data: Some(TxData::AmplifyBeta {
                    market_id: MarketId::new([1; 6]),
                    amount,
                    market_author: Address::ALL_ZEROS,
                }),
                ..Transaction::default()
            },
            spent_utxos: vec![bitcoin_output(amount + TRADE_MINER_FEE_SATS)],
            actor_address: None,
        };
        assert_eq!(
            super::miner_fee(&amplify).unwrap(),
            bitcoin::Amount::from_sat(TRADE_MINER_FEE_SATS)
        );
    }

    #[test]
    fn duplicate_decision_claim_across_txs_rejected() {
        let txs = vec![claim_tx(&[[1, 2, 3]]), claim_tx(&[[1, 2, 3]])];
        assert!(super::check_duplicate_decision_claims(&txs).is_err());
    }

    #[test]
    fn distinct_decision_claims_ok() {
        let txs = vec![claim_tx(&[[1, 2, 3]]), claim_tx(&[[4, 5, 6]])];
        assert!(super::check_duplicate_decision_claims(&txs).is_ok());
    }
}
