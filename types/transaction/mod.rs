use std::{borrow::Borrow, collections::HashSet};

use bitcoin::amount::CheckedSum;
use borsh::{self, BorshSerialize};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    address::Address,
    authorization::Authorization,
    decision::DecisionType,
    error,
    hashes::{
        self, Hash, InputsMerkleRoot, M6id, OutputsMerkleRoot, TxMerkleRoot,
        Txid, hash_with_scratch_buffer,
    },
    market::{DimensionSpec, MarketId},
};

pub mod inputs;
pub use inputs::Inputs;
pub mod outpoint;
pub use outpoint::{OutPoint, OutPointKey};
pub mod output;
pub use output::{
    Content as OutputContent, Output, Pointed as PointedOutput,
    PointedOutputRef,
};
pub mod outputs;
pub use outputs::Outputs;

pub trait GetAddress {
    fn get_address(&self) -> Address;
}

pub trait GetValue {
    fn get_value(&self) -> bitcoin::Amount;
}

/// Reference to a tx input.
#[derive(
    Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize, ToSchema,
)]
pub enum InPoint {
    /// Transaction input
    Regular {
        txid: Txid,
        // index of the spend in the inputs to spend_tx
        vin: u32,
    },
    // Created by mainchain withdrawals
    Withdrawal {
        m6id: M6id,
    },
    /// Removed from the state without a spend (wallet sync for stale UTXOs)
    Redistribution,
}

/// Struct representing a single vote in a batch vote transaction
#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Debug,
    Deserialize,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct BallotItem {
    /// 3 byte decision ID
    pub decision_id_bytes: [u8; 3],
    /// The vote value (0.0-1.0 for binary, scaled range for scaled decisions)
    pub vote_value: f64,
}

#[derive(
    BorshSerialize, Clone, Debug, Deserialize, PartialEq, Serialize, ToSchema,
)]
pub struct DecisionClaimEntry {
    pub decision_id_bytes: [u8; 3],
    pub header: String,
    pub description: String,
    pub option_0_label: Option<String>,
    pub option_1_label: Option<String>,
    pub option_labels: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
}

#[derive(
    BorshSerialize, Clone, Debug, Deserialize, PartialEq, Serialize, ToSchema,
)]
pub struct ClaimDecisionPayload {
    #[schema(value_type = String)]
    pub decision_type: DecisionType,
    pub decisions: Vec<DecisionClaimEntry>,
}

#[allow(clippy::enum_variant_names)]
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
#[schema(as = TxData)]
pub enum TransactionData {
    /// Claim one or more decisions.
    /// Binary: 1 entry. Scaled: 1 entry. Category: 2+ entries.
    ClaimDecision(ClaimDecisionPayload),
    /// Create a prediction market using dimension bracket notation.
    /// The initial LMSR beta is derived from the treasury output amount
    /// (`beta = treasury / ln(num_outcomes)`), so no beta field is needed.
    CreateMarket {
        title: String,
        description: String,
        #[schema(value_type = Vec<String>)]
        dimension_specs: Vec<DimensionSpec>,
        new_claims: Vec<ClaimDecisionPayload>,
        trading_fee: Option<f64>,
        tx_pow_hash_selector: Option<u8>,
        tx_pow_ordering: Option<u8>,
        tx_pow_difficulty: Option<u8>,
    },
    /// Trade shares in a prediction market (buy or sell).
    /// `outcome_index` is the position within the market's tradeable outcomes
    /// (i.e. index into `Market::get_valid_state_combos()`), not a full-state
    /// combo index. Abstain/invalid states are voter-only and never tradeable.
    Trade {
        market_id: MarketId,
        outcome_index: u32,
        shares: i64,
        trader: Address,
        limit_sats: u64,
        tx_pow_nonce: Option<u64>,
        prev_block_hash: hashes::BlockHash,
    },
    /// Submit a vote for a decision
    SubmitVote {
        voter: Address,
        decision_id_bytes: [u8; 3],
        vote_value: f64,
        voting_period: u32,
    },
    /// Submit multiple votes efficiently
    SubmitBallot {
        voter: Address,
        votes: Vec<BallotItem>,
        voting_period: u32,
    },
    TransferReputation {
        sender: Address,
        dest: Address,
        amount: f64,
    },
    /// Amplify a market's LMSR beta by funding its treasury.
    /// Only the market author may submit this transaction.
    AmplifyBeta {
        market_id: MarketId,
        amount: u64,
        market_author: Address,
    },
}

pub type TxData = TransactionData;

impl TxData {
    pub fn is_claim_decision(&self) -> bool {
        matches!(self, Self::ClaimDecision(_))
    }

    pub fn is_create_market(&self) -> bool {
        matches!(self, Self::CreateMarket { .. })
    }

    pub fn is_trade(&self) -> bool {
        matches!(self, Self::Trade { .. })
    }

    pub fn is_submit_vote(&self) -> bool {
        matches!(self, Self::SubmitVote { .. })
    }

    pub fn is_submit_ballot(&self) -> bool {
        matches!(self, Self::SubmitBallot { .. })
    }

    pub fn is_transfer_reputation(&self) -> bool {
        matches!(self, Self::TransferReputation { .. })
    }

    pub fn is_amplify_beta(&self) -> bool {
        matches!(self, Self::AmplifyBeta { .. })
    }
}

/// Borrowed view over a `CreateMarket` transaction's data.
#[derive(Clone, Debug)]
pub struct MarketCreationView<'a> {
    pub title: &'a str,
    pub description: &'a str,
    pub dimension_specs: &'a [DimensionSpec],
    pub trading_fee: Option<f64>,
    pub tx_pow_hash_selector: Option<u8>,
    pub tx_pow_ordering: Option<u8>,
    pub tx_pow_difficulty: Option<u8>,
    pub new_claims: &'a [ClaimDecisionPayload],
}

/// Struct describing a trade operation (buy or sell shares).
/// Sign of shares determines direction: positive = buy, negative = sell.
/// `outcome_index` is the position within the market's tradeable outcomes
/// (i.e. index into `Market::get_valid_state_combos()`), not a full-state
/// combo index.
#[derive(Clone, Debug, PartialEq)]
pub struct Trade {
    pub market_id: MarketId,
    pub outcome_index: u32,
    pub shares: i64,
    pub trader: Address,
    pub limit_sats: u64,
    pub tx_pow_nonce: Option<u64>,
    pub prev_block_hash: hashes::BlockHash,
}

impl Trade {
    /// Returns true if this is a buy trade (positive shares)
    pub fn is_buy(&self) -> bool {
        self.shares > 0
    }

    /// Returns true if this is a sell trade (negative shares)
    pub fn is_sell(&self) -> bool {
        self.shares < 0
    }

    /// Returns the absolute number of shares
    pub fn shares_abs(&self) -> u64 {
        self.shares.unsigned_abs()
    }
}

/// Struct describing a vote submission
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubmitVote {
    pub voter: Address,
    pub decision_id_bytes: [u8; 3],
    pub vote_value: f64,
    pub voting_period: u32,
}

/// Struct describing a ballot submission
#[derive(Clone, Debug, PartialEq)]
pub struct SubmitBallot {
    pub voter: Address,
    pub votes: Vec<BallotItem>,
    pub voting_period: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferReputation {
    pub sender: Address,
    pub dest: Address,
    pub amount: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AmplifyBeta {
    pub market_id: MarketId,
    pub amount: u64,
    pub market_author: Address,
}

#[derive(
    BorshSerialize, Clone, Debug, Default, Deserialize, Serialize, ToSchema,
)]
pub struct Transaction {
    #[schema(value_type = Vec<(OutPoint, String)>)]
    pub inputs: Inputs<(OutPoint, Hash)>,
    pub outputs: Outputs,
    pub data: Option<TransactionData>,
}

impl Transaction {
    pub fn txid(&self) -> Txid {
        hash_with_scratch_buffer(self).into()
    }

    /// Canonical encoding as bytes. The canonical encoding is used for hashing,
    /// but other encodings may be used at eg. networking, rpc levels.
    pub fn canonical_bytes(&self) -> borsh::io::Result<Vec<u8>> {
        borsh::to_vec(&self)
    }

    /// Canonical size in bytes. The canonical encoding is used for hashing,
    /// but other encodings may be used at eg. networking, rpc levels.
    #[inline(always)]
    pub fn canonical_size(&self) -> borsh::io::Result<u64> {
        borsh::object_length(self).map(|size| size as u64)
    }

    pub(crate) fn compute_merkle_root(
        &self,
    ) -> Result<TxMerkleRoot, outputs::error::ComputeMerkleRoot> {
        let Self {
            inputs,
            outputs,
            data,
        } = self;
        // Borsh encoding for hashing
        #[derive(BorshSerialize)]
        struct HashComponents {
            inputs_commitment: InputsMerkleRoot,
            outputs_commitment: OutputsMerkleRoot,
            data_commitment: Hash,
        }
        let res = hash_with_scratch_buffer(&HashComponents {
            inputs_commitment: inputs.compute_merkle_root(),
            outputs_commitment: outputs.compute_merkle_root()?,
            data_commitment: hashes::hash(data),
        });
        Ok(res.into())
    }
}

/// Representation of a spent output
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub struct SpentOutput {
    pub output: Output,
    pub inpoint: InPoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilledTransaction {
    pub transaction: Transaction,
    pub spent_utxos: Vec<Output>,
    /// Address proven by the actor proof, if any
    #[serde(default)]
    pub actor_address: Option<Address>,
}

impl FilledTransaction {
    pub fn get_value_in(
        &self,
    ) -> Result<bitcoin::Amount, error::AmountOverflow> {
        self.spent_utxos
            .iter()
            .map(GetValue::get_value)
            .checked_sum()
            .ok_or(error::AmountOverflow)
    }

    pub fn get_value_out(
        &self,
    ) -> Result<bitcoin::Amount, error::AmountOverflow> {
        self.transaction
            .outputs
            .iter()
            .map(GetValue::get_value)
            .checked_sum()
            .ok_or(error::AmountOverflow)
    }

    pub fn get_fee(&self) -> Result<bitcoin::Amount, error::ComputeFee> {
        let value_in = self
            .get_value_in()
            .map_err(error::ComputeFee::ValueInOverflow)?;
        let value_out = self
            .get_value_out()
            .map_err(error::ComputeFee::ValueOutOverflow)?;
        if value_in < value_out {
            Err(error::ComputeFee::Underfunded)
        } else {
            Ok(value_in - value_out)
        }
    }

    pub fn inputs(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &Hash, &Output)> {
        self.transaction.inputs.iter().zip(&self.spent_utxos).map(
            |((outpoint, utxo_hash), output)| (outpoint, utxo_hash, output),
        )
    }

    /// Accessor for tx data
    pub fn data(&self) -> &Option<TxData> {
        &self.transaction.data
    }

    /// Accessor for txid
    pub fn txid(&self) -> Txid {
        self.transaction.txid()
    }

    pub fn is_claim_decision(&self) -> bool {
        self.data().as_ref().is_some_and(TxData::is_claim_decision)
    }

    pub fn is_create_market(&self) -> bool {
        self.data().as_ref().is_some_and(TxData::is_create_market)
    }

    pub fn is_trade(&self) -> bool {
        self.data().as_ref().is_some_and(TxData::is_trade)
    }

    pub fn is_submit_vote(&self) -> bool {
        self.data().as_ref().is_some_and(TxData::is_submit_vote)
    }

    pub fn is_submit_ballot(&self) -> bool {
        self.data().as_ref().is_some_and(TxData::is_submit_ballot)
    }

    pub fn is_transfer_reputation(&self) -> bool {
        self.data()
            .as_ref()
            .is_some_and(TxData::is_transfer_reputation)
    }

    pub fn is_amplify_beta(&self) -> bool {
        self.data().as_ref().is_some_and(TxData::is_amplify_beta)
    }

    pub fn claim_decision(&self) -> Option<&ClaimDecisionPayload> {
        match &self.transaction.data {
            Some(TransactionData::ClaimDecision(payload)) => Some(payload),
            _ => None,
        }
    }

    /// If the tx is a market creation, returns a borrowed view over its data.
    pub fn as_market_creation(&self) -> Option<MarketCreationView<'_>> {
        match &self.transaction.data {
            Some(TransactionData::CreateMarket {
                title,
                description,
                dimension_specs,
                new_claims,
                trading_fee,
                tx_pow_hash_selector,
                tx_pow_ordering,
                tx_pow_difficulty,
            }) => Some(MarketCreationView {
                title: title.as_str(),
                description: description.as_str(),
                dimension_specs: dimension_specs.as_slice(),
                trading_fee: *trading_fee,
                tx_pow_hash_selector: *tx_pow_hash_selector,
                tx_pow_ordering: *tx_pow_ordering,
                tx_pow_difficulty: *tx_pow_difficulty,
                new_claims: new_claims.as_slice(),
            }),
            _ => None,
        }
    }

    /// If the tx is a trade, returns the corresponding [`Trade`].
    pub fn trade(&self) -> Option<Trade> {
        match &self.transaction.data {
            Some(TransactionData::Trade {
                market_id,
                outcome_index,
                shares,
                trader,
                limit_sats,
                tx_pow_nonce,
                prev_block_hash,
            }) => Some(Trade {
                market_id: *market_id,
                outcome_index: *outcome_index,
                shares: *shares,
                trader: *trader,
                limit_sats: *limit_sats,
                tx_pow_nonce: *tx_pow_nonce,
                prev_block_hash: *prev_block_hash,
            }),
            _ => None,
        }
    }

    /// If the tx is a vote submission, returns the corresponding [`SubmitVote`].
    pub fn submit_vote(&self) -> Option<SubmitVote> {
        match &self.transaction.data {
            Some(TransactionData::SubmitVote {
                voter,
                decision_id_bytes,
                vote_value,
                voting_period,
            }) => Some(SubmitVote {
                voter: *voter,
                decision_id_bytes: *decision_id_bytes,
                vote_value: *vote_value,
                voting_period: *voting_period,
            }),
            _ => None,
        }
    }

    pub fn submit_ballot(&self) -> Option<SubmitBallot> {
        match &self.transaction.data {
            Some(TransactionData::SubmitBallot {
                voter,
                votes,
                voting_period,
            }) => Some(SubmitBallot {
                voter: *voter,
                votes: votes.clone(),
                voting_period: *voting_period,
            }),
            _ => None,
        }
    }

    pub fn transfer_reputation(&self) -> Option<TransferReputation> {
        match &self.transaction.data {
            Some(TransactionData::TransferReputation {
                sender,
                dest,
                amount,
            }) => Some(TransferReputation {
                sender: *sender,
                dest: *dest,
                amount: *amount,
            }),
            _ => None,
        }
    }

    pub fn amplify_beta(&self) -> Option<AmplifyBeta> {
        match &self.transaction.data {
            Some(TransactionData::AmplifyBeta {
                market_id,
                amount,
                market_author,
            }) => Some(AmplifyBeta {
                market_id: *market_id,
                amount: *amount,
                market_author: *market_author,
            }),
            _ => None,
        }
    }
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Authorized<T> {
    pub transaction: T,
    /// Authorizations are called witnesses in Bitcoin.
    pub authorizations: Vec<Authorization>,
    /// Signature of the market actor (trader, voter or sender) when no input
    /// belongs to that actor
    #[serde(default)]
    pub actor_proof: Option<Authorization>,
}

pub type AuthorizedTransaction = Authorized<Transaction>;

impl AuthorizedTransaction {
    /// Return an iterator over all addresses relevant to the transaction
    pub fn relevant_addresses(&self) -> HashSet<Address> {
        let input_addrs =
            self.authorizations.iter().map(|auth| auth.get_address());
        let actor_addrs =
            self.actor_proof.iter().map(|auth| auth.get_address());
        let output_addrs =
            self.transaction.outputs.iter().map(|output| output.address);
        input_addrs.chain(actor_addrs).chain(output_addrs).collect()
    }
}

impl<T> Borrow<T> for Authorized<T> {
    fn borrow(&self) -> &T {
        &self.transaction
    }
}

impl From<Authorized<FilledTransaction>> for AuthorizedTransaction {
    fn from(tx: Authorized<FilledTransaction>) -> Self {
        Self {
            transaction: tx.transaction.transaction,
            authorizations: tx.authorizations,
            actor_proof: tx.actor_proof,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        address::Address,
        transaction::{
            FilledTransaction, GetValue, Output, OutputContent, Outputs,
            Transaction,
        },
    };

    // a withdrawal output must be funded for both its payout and its mainchain
    // fee, since both leave the treasury
    #[test]
    fn withdrawal_value_includes_main_fee() {
        let value = bitcoin::Amount::from_sat(1000);
        let main_fee = bitcoin::Amount::from_sat(300);
        let main_address = "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
            .parse::<bitcoin::Address<bitcoin::address::NetworkUnchecked>>()
            .unwrap();
        let withdrawal = Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::Withdrawal {
                value,
                main_fee,
                main_address,
            },
        };
        assert_eq!(withdrawal.get_value(), value + main_fee);

        let value_output = |amount| Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::Value(amount),
        };
        let withdrawal_tx = |funding| FilledTransaction {
            transaction: Transaction {
                outputs: Outputs(vec![withdrawal.clone()]),
                ..Default::default()
            },
            spent_utxos: vec![value_output(funding)],
            actor_address: None,
        };

        // inputs covering only the payout are insufficient
        assert!(withdrawal_tx(value).get_fee().is_err());
        // inputs covering payout plus mainchain fee fully fund it
        assert_eq!(
            withdrawal_tx(value + main_fee).get_fee().unwrap(),
            bitcoin::Amount::ZERO
        );
    }

    #[test]
    fn claim_decision_variant_wire_format() -> anyhow::Result<()> {
        use crate::{
            decision::DecisionType,
            transaction::{
                ClaimDecisionPayload, DecisionClaimEntry, TransactionData,
            },
        };

        let payload = ClaimDecisionPayload {
            decision_type: DecisionType::Binary,
            decisions: vec![DecisionClaimEntry {
                decision_id_bytes: [0x01, 0x02, 0x03],
                header: "h".to_string(),
                description: "d".to_string(),
                option_0_label: None,
                option_1_label: None,
                option_labels: None,
                tags: None,
            }],
        };

        let payload_bytes = borsh::to_vec(&payload)?;
        let variant_bytes =
            borsh::to_vec(&TransactionData::ClaimDecision(payload.clone()))?;

        anyhow::ensure!(
            !variant_bytes.is_empty(),
            "variant encoding must not be empty"
        );
        anyhow::ensure!(
            variant_bytes[0] == 0u8,
            "ClaimDecision must be variant index 0 (got {})",
            variant_bytes[0]
        );
        anyhow::ensure!(
            &variant_bytes[1..] == payload_bytes.as_slice(),
            "tuple-variant body must equal raw payload encoding"
        );

        Ok(())
    }
}
