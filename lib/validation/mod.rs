pub mod decision;
pub mod market;
pub mod offer;
pub mod vote;

use std::collections::{BTreeMap, HashSet};

use sneed::RoTxn;

use crate::{
    math::trading::TRADE_MINER_FEE_SATS,
    state::{
        Error, State,
        decisions::{Decision, DecisionId},
    },
    types::{
        AmountOverflowError, ComputeFeeError, FilledTransaction, Hash,
        OutPoint, Output, OutputContent, Transaction, TransactionData,
    },
};

pub use decision::DecisionValidator;
pub use market::{MarketStateValidator, MarketValidator};
pub use offer::OfferValidator;
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

    fn get_nonstandard_claimed_count_in_period(
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

pub const MIN_FEERATE_SATS_PER_BYTE: u64 = 1;

/// Serialized size of one authorization: a verifying key and a signature
pub const AUTHORIZATION_SIZE: u64 = 104;

/// Size of a transaction for its fee: the serialized size of its inputs,
/// outputs and data, plus one authorization per input and its actor proof.
/// The utreexo proof is left out, because the node regenerates it as the
/// chain grows.
pub fn tx_size(
    transaction: &Transaction,
    actor_proof: bool,
) -> Result<u64, bincode::Error> {
    let body = bincode::serialized_size(&(
        &transaction.inputs,
        &transaction.outputs,
        &transaction.data,
    ))?;
    let authorizations =
        transaction.inputs.len() as u64 + u64::from(actor_proof);
    Ok(body + authorizations * AUTHORIZATION_SIZE)
}

/// Size that spending the output at `outpoint` adds to a transaction: its
/// input and authorization
pub fn input_size(outpoint: &OutPoint) -> Result<u64, bincode::Error> {
    Ok(bincode::serialized_size(&(outpoint, Hash::default()))?
        + AUTHORIZATION_SIZE)
}

/// Minimum fee of a transaction, at the minimum fee rate
pub fn min_tx_fee(
    transaction: &Transaction,
    actor_proof: bool,
) -> Result<bitcoin::Amount, bincode::Error> {
    Ok(bitcoin::Amount::from_sat(
        tx_size(transaction, actor_proof)? * MIN_FEERATE_SATS_PER_BYTE,
    ))
}

/// Miner fee of a trade or an amplify beta transaction: the flat trade fee,
/// or the minimum fee for its size when that is larger
pub fn trade_miner_fee(
    transaction: &Transaction,
    actor_proof: bool,
) -> Result<bitcoin::Amount, bincode::Error> {
    Ok(min_tx_fee(transaction, actor_proof)?
        .max(bitcoin::Amount::from_sat(TRADE_MINER_FEE_SATS)))
}

/// Add a listing fee owed to the pool of `period`
pub(crate) fn add_listing_fee(
    listing_fees: &mut BTreeMap<u32, u64>,
    period: u32,
    fee: u64,
) -> Result<(), Error> {
    let owed = listing_fees.entry(period).or_insert(0);
    *owed = owed
        .checked_add(fee)
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Listing fee overflow".to_string(),
        })?;
    Ok(())
}

/// The listing fee outputs of a transaction pay each period's pool at least
/// the listing fees owed there, one output per period. A claim quoted
/// before a price drop overpays, and the excess stays in the pool.
pub(crate) fn validate_listing_outputs(
    tx: &FilledTransaction,
    listing_fees: &BTreeMap<u32, u64>,
) -> Result<(), Error> {
    use crate::state::markets::payouts::generate_listing_pool_address;

    let mut paid = BTreeMap::new();
    for output in &tx.transaction.outputs {
        let OutputContent::ListingFees { period, amount } = &output.content
        else {
            continue;
        };
        if output.address != generate_listing_pool_address(*period) {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Listing fees of period {period} are not paid to its pool"
                ),
            });
        }
        if paid.insert(*period, amount.to_sat()).is_some() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "More than one listing fee output for period {period}"
                ),
            });
        }
    }
    let owed: BTreeMap<u32, u64> = listing_fees
        .iter()
        .filter(|(_, fee)| **fee > 0)
        .map(|(period, fee)| (*period, *fee))
        .collect();
    if paid.len() != owed.len()
        || owed
            .iter()
            .any(|(period, fee)| paid.get(period).is_none_or(|paid| paid < fee))
    {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Listing fee outputs {paid:?} do not cover the listing fees \
                 owed {owed:?}"
            ),
        });
    }
    Ok(())
}

/// The transaction pays `required` plus the minimum fee rate to the miner
pub(crate) fn validate_fee_floor(
    tx: &FilledTransaction,
    required: bitcoin::Amount,
) -> Result<(), Error> {
    let fee = tx_fee(tx)?;
    let floor = min_tx_fee(&tx.transaction, tx.actor_address.is_some())
        .map_err(Error::BincodeSerialize)?
        .checked_add(required)
        .ok_or(AmountOverflowError)?;
    if fee < floor {
        return Err(Error::InvalidTransaction {
            reason: format!("Fee {fee} is below the minimum {floor}"),
        });
    }
    Ok(())
}

/// Fee that a transaction pays to the block producer
pub fn miner_fee(
    filled_tx: &FilledTransaction,
) -> Result<bitcoin::Amount, Error> {
    if filled_tx.is_trade() || filled_tx.is_amplify_beta() {
        return trade_miner_fee(
            &filled_tx.transaction,
            filled_tx.actor_address.is_some(),
        )
        .map_err(Error::BincodeSerialize);
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
    if !(tx.is_claim_decision() || tx.is_create_market())
        && tx
            .transaction
            .outputs
            .iter()
            .any(|output| output.content.is_listing_fees())
    {
        return Err(Error::InvalidTransaction {
            reason: "Only a decision claim or market creation may pay \
                     listing fees"
                .to_string(),
        });
    }
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
        return Ok(Some(
            trade_miner_fee(&tx.transaction, tx.actor_address.is_some())
                .map_err(Error::BincodeSerialize)?,
        ));
    }
    if tx.is_amplify_beta() {
        MarketValidator::validate_amplify_beta(
            state,
            rotxn,
            tx,
            override_height,
        )?;
        return Ok(Some(
            trade_miner_fee(&tx.transaction, tx.actor_address.is_some())
                .map_err(Error::BincodeSerialize)?,
        ));
    }
    if tx.is_submit_ballot() {
        VoteValidator::validate_ballot(state, rotxn, tx, override_height)?;
    }
    if tx.is_reveal_ballot() {
        VoteValidator::validate_reveal(state, rotxn, tx, override_height)?;
    }
    if tx.is_transfer_reputation() {
        VoteValidator::validate_reputation_transfer(
            state,
            rotxn,
            tx,
            override_height,
        )?;
    }
    if tx.is_take_offer() {
        OfferValidator::validate_take_offer(state, rotxn, tx)?;
    }
    if tx.is_cancel_offer() {
        OfferValidator::validate_cancel_offer(state, rotxn, tx)?;
    }
    Ok(None)
}

pub fn check_duplicate_offer_closes(
    transactions: &[Transaction],
) -> Result<(), Error> {
    let mut closed = HashSet::new();
    for offer_id in transactions
        .iter()
        .filter_map(|tx| tx.data.as_ref()?.closed_offer_id())
    {
        if !closed.insert(offer_id) {
            return Err(Error::InvalidTransaction {
                reason: format!("Block closes offer {offer_id} more than once"),
            });
        }
    }
    Ok(())
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

/// Reject market funds and listing fees in the coinbase
pub fn validate_coinbase_outputs(coinbase: &[Output]) -> Result<(), Error> {
    for output in coinbase {
        match &output.content {
            OutputContent::Value(_) | OutputContent::Withdrawal { .. } => {}
            OutputContent::MarketFunds { .. }
            | OutputContent::ListingFees { .. } => {
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

    /// A coinbase may always pay a bitcoin output
    #[test]
    fn coinbase_bitcoin_output() {
        let outputs = vec![bitcoin_output(5000)];
        assert!(super::validate_coinbase_outputs(&outputs).is_ok());
    }

    /// A coinbase may never pay market funds
    #[test]
    fn coinbase_market_funds() {
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

    /// A coinbase may never pay listing fees
    #[test]
    fn coinbase_listing_fees() {
        let outputs = vec![Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::ListingFees {
                period: 1,
                amount: bitcoin::Amount::from_sat(1000),
            },
        }];
        assert!(super::validate_coinbase_outputs(&outputs).is_err());
    }

    /// A transaction that claims no decision may not pay listing fees
    #[test]
    fn transfer_listing_fees() {
        use crate::{
            archive::Archive,
            state::State,
            types::{FilledTransaction, OutPoint, Txid},
        };

        let dir = tempfile::tempdir().unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024)
            .max_dbs(State::NUM_DBS + Archive::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, dir.path()) }.unwrap();
        let state = State::new(&env, None, true).unwrap();
        let archive = Archive::new(&env).unwrap();
        let tx = FilledTransaction {
            transaction: Transaction {
                inputs: vec![(
                    OutPoint::Regular {
                        txid: Txid([1; 32]),
                        vout: 0,
                    },
                    [0; 32],
                )]
                .into(),
                outputs: vec![Output {
                    address:
                        crate::state::markets::payouts::generate_listing_pool_address(1),
                    content: OutputContent::ListingFees {
                        period: 1,
                        amount: bitcoin::Amount::from_sat(1000),
                    },
                }]
                .into(),
                ..Default::default()
            },
            spent_utxos: vec![bitcoin_output(2000)],
            actor_address: None,
        };
        let rotxn = env.read_txn().unwrap();
        assert!(
            super::validate_market_transaction(
                &state, &rotxn, &tx, &archive, None
            )
            .is_err()
        );
    }

    #[test]
    fn validate_fees_exact_match_ok() {
        let fees = bitcoin::Amount::from_sat(0);
        assert!(super::validate_fees(fees, &[], &HashSet::new()).is_ok());
    }

    #[test]
    fn coinbase_exceeds_fees() {
        let coinbase = bitcoin::Amount::from_sat(1000);
        assert!(super::validate_fees(coinbase, &[], &HashSet::new()).is_err());
    }

    fn claim_tx(ids: &[[u8; 4]]) -> Transaction {
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
    fn block_closes_offer_once() {
        use crate::types::{TxData, VotecoinOffer, authorization::SigningKey};

        let key = SigningKey::new(&mut rand::rng());
        let offer = |nonce| VotecoinOffer {
            seller: Address([1; 20]),
            amount: crate::types::Votecoin(100_000),
            price: bitcoin::Amount::from_sat(1000),
            pay_to: Address([1; 20]),
            voting_period: 1,
            nonce,
        };
        let take = |nonce| Transaction {
            data: Some(TxData::TakeOffer {
                offer: Box::new(super::offer::sign_test_offer(
                    &key,
                    offer(nonce),
                )),
                buyer: Address([2; 20]),
            }),
            ..Transaction::default()
        };
        let cancel = |nonce| Transaction {
            data: Some(TxData::CancelOffer {
                offer: offer(nonce),
            }),
            ..Transaction::default()
        };

        assert!(
            super::check_duplicate_offer_closes(&[take(1), take(1)]).is_err()
        );
        assert!(
            super::check_duplicate_offer_closes(&[take(1), cancel(1)]).is_err()
        );
        super::check_duplicate_offer_closes(&[take(1), take(2), cancel(3)])
            .unwrap();
    }

    /// The miner fee excludes an amplify-beta deposit
    #[test]
    fn miner_fee_beta_deposit() {
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

    /// The fee size counts one authorization per input and the actor
    /// proof, each the size of a real authorization
    #[test]
    fn size_counts_signatures() {
        use crate::types::{
            OutPoint, Txid,
            authorization::{Authorization, SigningKey, sign_tx},
        };

        let key = SigningKey::new(&mut rand::rng());
        let transaction = Transaction {
            inputs: vec![
                (
                    OutPoint::Regular {
                        txid: Txid([1; 32]),
                        vout: 0,
                    },
                    [0; 32],
                );
                2
            ]
            .into(),
            ..Transaction::default()
        };
        let authorization = Authorization {
            verifying_key: (&key).into(),
            signature: sign_tx(rand::rng(), &key, &transaction).unwrap(),
        };
        assert_eq!(
            bincode::serialized_size(&authorization).unwrap(),
            super::AUTHORIZATION_SIZE
        );
        let bare = bincode::serialized_size(&(
            &transaction.inputs,
            &transaction.outputs,
            &transaction.data,
        ))
        .unwrap();
        assert_eq!(
            super::tx_size(&transaction, false).unwrap(),
            bare + 2 * super::AUTHORIZATION_SIZE
        );
        assert_eq!(
            super::tx_size(&transaction, true).unwrap(),
            bare + 3 * super::AUTHORIZATION_SIZE
        );
    }

    /// A trade pays the flat trade fee, or its minimum fee once larger
    #[test]
    fn trade_fee_by_size() {
        use crate::{
            math::trading::TRADE_MINER_FEE_SATS,
            state::markets::MarketId,
            types::{BlockHash, OutPoint, ShareDelta, TxData, Txid},
        };

        let trade = |inputs: u32| Transaction {
            inputs: (0..inputs)
                .map(|vout| {
                    (
                        OutPoint::Regular {
                            txid: Txid([1; 32]),
                            vout,
                        },
                        [0; 32],
                    )
                })
                .collect::<Vec<_>>()
                .into(),
            data: Some(TxData::Trade {
                market_id: MarketId::new([1; 6]),
                deltas: vec![ShareDelta {
                    outcome_index: 0,
                    shares: 1,
                }],
                trader: Address::ALL_ZEROS,
                max_cost: 1_000,
                tx_pow_nonce: None,
                prev_block_hash: BlockHash::default(),
            }),
            ..Transaction::default()
        };
        assert_eq!(
            super::trade_miner_fee(&trade(1), false).unwrap(),
            bitcoin::Amount::from_sat(TRADE_MINER_FEE_SATS)
        );
        let large = trade(20);
        let min_fee = super::min_tx_fee(&large, false).unwrap();
        assert!(min_fee.to_sat() > TRADE_MINER_FEE_SATS);
        assert_eq!(super::trade_miner_fee(&large, false).unwrap(), min_fee);
    }

    #[test]
    fn duplicate_claim_across_txs() {
        let txs = vec![claim_tx(&[[1, 2, 3, 4]]), claim_tx(&[[1, 2, 3, 4]])];
        assert!(super::check_duplicate_decision_claims(&txs).is_err());
    }

    #[test]
    fn distinct_decision_claims_ok() {
        let txs = vec![claim_tx(&[[1, 2, 3, 4]]), claim_tx(&[[5, 6, 7, 8]])];
        assert!(super::check_duplicate_decision_claims(&txs).is_ok());
    }
}
