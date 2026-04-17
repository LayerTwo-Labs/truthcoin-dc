use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    io::Cursor,
};

use bitcoin::amount::CheckedSum as _;
use borsh::{self, BorshDeserialize, BorshSerialize};
use heed::{BoxedError, BytesDecode, BytesEncode};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use utoipa::{PartialSchema, ToSchema};

use crate::{
    authorization::Authorization,
    state::{
        decisions::DecisionType,
        markets::{DimensionSpec, MarketId},
    },
    types::{
        AmountOverflowError, GetAddress, GetBitcoinValue,
        address::Address,
        hashes::{self, AssetId, M6id, MerkleRoot, Txid},
        serde_hexstr_human_readable,
    },
};

mod output;
pub use output::{
    AssetContent as AssetOutputContent, AssetOutput,
    BitcoinContent as BitcoinOutputContent, BitcoinOutput,
    Content as OutputContent, FilledContent as FilledOutputContent,
    FilledOutput, Output, Pointed as PointedOutput, SpentOutput,
    WithdrawalContent as WithdrawalOutputContent,
};

fn borsh_serialize_bitcoin_outpoint<W>(
    block_hash: &bitcoin::OutPoint,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    let bitcoin::OutPoint { txid, vout } = block_hash;
    let txid_bytes: &[u8; 32] = txid.as_ref();
    borsh::BorshSerialize::serialize(&(txid_bytes, vout), writer)
}

fn borsh_deserialize_bitcoin_outpoint<R>(
    reader: &mut R,
) -> borsh::io::Result<bitcoin::OutPoint>
where
    R: borsh::io::Read,
{
    use bitcoin::hashes::Hash as _;
    let (txid_bytes, vout): ([u8; 32], u32) =
        <([u8; 32], u32) as BorshDeserialize>::deserialize_reader(reader)?;
    Ok(bitcoin::OutPoint {
        txid: bitcoin::Txid::from_byte_array(txid_bytes),
        vout,
    })
}

#[derive(
    BorshDeserialize,
    BorshSerialize,
    Clone,
    Copy,
    Debug,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    ToSchema,
)]
pub enum OutPoint {
    // Created by transactions.
    Regular {
        txid: Txid,
        vout: u32,
    },
    // Created by block bodies.
    Coinbase {
        merkle_root: MerkleRoot,
        vout: u32,
    },
    // Created by mainchain deposits.
    #[schema(value_type = crate::types::schema::BitcoinOutPoint)]
    Deposit(
        #[borsh(
            serialize_with = "borsh_serialize_bitcoin_outpoint",
            deserialize_with = "borsh_deserialize_bitcoin_outpoint"
        )]
        bitcoin::OutPoint,
    ),
    /// Market funds UTXO - treasury (is_fee=false) or author fees (is_fee=true)
    /// Unified type that replaces the separate Market and MarketAuthorFee variants
    MarketFunds {
        market_id: [u8; 6],
        block_height: u32,
        is_fee: bool,
    },
    Payout {
        hash: MerkleRoot,
        vout: u32,
    },
}

impl std::fmt::Display for OutPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Regular { txid, vout } => write!(f, "regular {txid} {vout}"),
            Self::Coinbase { merkle_root, vout } => {
                write!(f, "coinbase {merkle_root} {vout}")
            }
            Self::Deposit(bitcoin::OutPoint { txid, vout }) => {
                write!(f, "deposit {txid} {vout}")
            }
            Self::MarketFunds {
                market_id,
                block_height,
                is_fee,
            } => {
                let type_str = if *is_fee { "market_fee" } else { "market" };
                write!(
                    f,
                    "{} {} {}",
                    type_str,
                    hex::encode(market_id),
                    block_height
                )
            }
            Self::Payout { hash, vout } => {
                write!(f, "payout {hash} {vout}")
            }
        }
    }
}

const OUTPOINT_KEY_SIZE: usize = 37;

/// Fixed-width key for OutPoint based on its canonical Borsh encoding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OutPointKey([u8; OUTPOINT_KEY_SIZE]);

impl OutPointKey {
    /// Encode an OutPoint into a fixed-width lexicographically sortable key
    #[inline]
    pub fn from_outpoint(op: &OutPoint) -> Self {
        let mut key = [0u8; OUTPOINT_KEY_SIZE];
        let mut cursor = Cursor::new(&mut key[..]);
        BorshSerialize::serialize(op, &mut cursor)
            .expect("serializing OutPoint into key buffer should never fail");
        assert!(
            cursor.position() as usize <= OUTPOINT_KEY_SIZE,
            "OutPoint serialized to {} bytes, exceeding max of {}",
            cursor.position(),
            OUTPOINT_KEY_SIZE,
        );
        Self(key)
    }

    /// Get the raw key bytes
    #[inline]
    pub fn as_bytes(&self) -> &[u8; OUTPOINT_KEY_SIZE] {
        &self.0
    }

    /// Decode OutPointKey back to OutPoint
    #[inline]
    pub fn to_outpoint(&self) -> OutPoint {
        let mut cursor = Cursor::new(&self.0[..]);
        OutPoint::deserialize_reader(&mut cursor)
            .expect("deserializing OutPointKey should never fail")
    }
}

impl From<OutPoint> for OutPointKey {
    #[inline]
    fn from(op: OutPoint) -> Self {
        Self::from_outpoint(&op)
    }
}

impl From<&OutPoint> for OutPointKey {
    #[inline]
    fn from(op: &OutPoint) -> Self {
        OutPointKey::from_outpoint(op)
    }
}

impl From<OutPointKey> for OutPoint {
    #[inline]
    fn from(key: OutPointKey) -> Self {
        key.to_outpoint()
    }
}

impl From<&OutPointKey> for OutPoint {
    #[inline]
    fn from(key: &OutPointKey) -> Self {
        key.to_outpoint()
    }
}

impl Ord for OutPointKey {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for OutPointKey {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl AsRef<[u8]> for OutPointKey {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl<'a> BytesEncode<'a> for OutPointKey {
    type EItem = OutPointKey;

    #[inline]
    fn bytes_encode(
        item: &'a Self::EItem,
    ) -> Result<std::borrow::Cow<'a, [u8]>, BoxedError> {
        Ok(std::borrow::Cow::Borrowed(item.as_ref()))
    }
}

impl<'a> BytesDecode<'a> for OutPointKey {
    type DItem = OutPointKey;

    #[inline]
    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        if bytes.len() != OUTPOINT_KEY_SIZE {
            return Err("OutPointKey must be exactly 37 bytes".into());
        }
        let mut key = [0u8; OUTPOINT_KEY_SIZE];
        key.copy_from_slice(bytes);
        let mut cursor = Cursor::new(&key[..]);
        OutPoint::deserialize_reader(&mut cursor)
            .map_err(|err| -> BoxedError { Box::new(err) })?;
        Ok(OutPointKey(key))
    }
}

#[cfg(test)]
mod tests {
    use super::{OUTPOINT_KEY_SIZE, OutPoint, OutPointKey};
    use bitcoin::hashes::Hash as _;

    #[test]
    fn check_outpoint_key_size() -> anyhow::Result<()> {
        let variants = [
            OutPoint::Regular {
                txid: Default::default(),
                vout: u32::MAX,
            },
            OutPoint::Coinbase {
                merkle_root: Default::default(),
                vout: u32::MAX,
            },
            OutPoint::Deposit(bitcoin::OutPoint {
                txid: bitcoin::Txid::from_byte_array([0; 32]),
                vout: u32::MAX,
            }),
        ];

        for op in variants {
            let serialized = borsh::to_vec(&op)?;
            anyhow::ensure!(
                serialized.len() == OUTPOINT_KEY_SIZE,
                "unexpected serialized size: {}",
                serialized.len()
            );

            let key = OutPointKey::from(op);
            let decoded = OutPoint::from(key);
            anyhow::ensure!(decoded == op);
        }

        let market_funds = OutPoint::MarketFunds {
            market_id: [0xAB; 6],
            block_height: 42,
            is_fee: true,
        };
        let mf_serialized = borsh::to_vec(&market_funds)?;
        anyhow::ensure!(
            mf_serialized.len() <= OUTPOINT_KEY_SIZE,
            "MarketFunds serialized to {} bytes, exceeding max {}",
            mf_serialized.len(),
            OUTPOINT_KEY_SIZE,
        );
        let mf_key = OutPointKey::from(market_funds);
        let mf_decoded = OutPoint::from(mf_key);
        anyhow::ensure!(mf_decoded == market_funds);

        let payout = OutPoint::Payout {
            hash: Default::default(),
            vout: u32::MAX,
        };
        let payout_serialized = borsh::to_vec(&payout)?;
        anyhow::ensure!(
            payout_serialized.len() == OUTPOINT_KEY_SIZE,
            "Payout serialized to {} bytes, expected {}",
            payout_serialized.len(),
            OUTPOINT_KEY_SIZE,
        );
        let payout_key = OutPointKey::from(payout);
        let payout_decoded = OutPoint::from(payout_key);
        anyhow::ensure!(payout_decoded == payout);

        Ok(())
    }
}

/// Reference to a tx input.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
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
    /// Consumed by system operations (wallet sync for stale UTXOs)
    Redistribution,
}

pub type TxInputs = Vec<OutPoint>;

pub type TxOutputs = Vec<Output>;

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

#[allow(clippy::enum_variant_names)]
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
#[schema(as = TxData)]
pub enum TransactionData {
    /// Claim one or more decisions.
    /// Binary: 1 entry. Scaled: 1 entry. Category: 2+ entries.
    ClaimDecision {
        #[schema(value_type = String)]
        decision_type: DecisionType,
        decisions: Vec<DecisionClaimEntry>,
    },
    /// Create a prediction market using dimension bracket notation.
    /// The initial LMSR beta is derived from the treasury output amount
    /// (`beta = treasury / ln(num_outcomes)`), so no beta field is needed.
    CreateMarket {
        title: String,
        description: String,
        #[schema(value_type = Vec<String>)]
        dimension_specs: Vec<DimensionSpec>,
        trading_fee: Option<f64>,
        category_txids: Option<Vec<[u8; 32]>>,
        residual_names: Option<Vec<String>>,
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
        matches!(self, Self::ClaimDecision { .. })
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

#[derive(Clone, Debug, PartialEq)]
pub struct ClaimDecision {
    pub decision_type: DecisionType,
    pub decisions: Vec<DecisionClaimEntry>,
}

/// Struct describing a market creation using dimension specifications
#[derive(Clone, Debug, PartialEq)]
pub struct CreateMarket {
    pub title: String,
    pub description: String,
    pub dimension_specs: Vec<DimensionSpec>,
    pub trading_fee: Option<f64>,
    pub category_txids: Option<Vec<[u8; 32]>>,
    pub residual_names: Option<Vec<String>>,
    pub tx_pow_hash_selector: Option<u8>,
    pub tx_pow_ordering: Option<u8>,
    pub tx_pow_difficulty: Option<u8>,
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
    #[schema(schema_with = TxInputs::schema)]
    pub inputs: TxInputs,
    #[schema(schema_with = TxOutputs::schema)]
    pub outputs: TxOutputs,
    #[serde(with = "serde_hexstr_human_readable")]
    #[schema(value_type = String)]
    pub memo: Vec<u8>,
    pub data: Option<TransactionData>,
}

impl Transaction {
    pub fn new(inputs: TxInputs, outputs: TxOutputs) -> Self {
        Self {
            inputs,
            outputs,
            memo: Vec::new(),
            data: None,
        }
    }

    /// Return an iterator over asset outputs with index
    pub fn indexed_asset_outputs(
        &self,
    ) -> impl Iterator<Item = (usize, AssetOutput)> + '_ {
        self.outputs.iter().enumerate().filter_map(|(idx, output)| {
            let asset_output: AssetOutput =
                Option::<AssetOutput>::from(output.clone())?;
            Some((idx, asset_output))
        })
    }

    /// `true` if the tx data corresponds to a regular tx
    pub fn is_regular(&self) -> bool {
        self.data.is_none()
    }

    pub fn txid(&self) -> Txid {
        hashes::hash_with_scratch_buffer(self).into()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FilledTransaction {
    pub transaction: Transaction,
    pub spent_utxos: Vec<FilledOutput>,
    #[serde(default)]
    pub actor_address: Option<Address>,
}

impl FilledTransaction {
    /// Accessor for tx data
    pub fn data(&self) -> &Option<TxData> {
        &self.transaction.data
    }

    /// Accessor for tx inputs
    pub fn inputs(&self) -> &TxInputs {
        &self.transaction.inputs
    }

    /// Accessor for tx outputs
    pub fn outputs(&self) -> &TxOutputs {
        &self.transaction.outputs
    }

    pub fn is_claim_decision(&self) -> bool {
        match &self.transaction.data {
            Some(tx_data) => tx_data.is_claim_decision(),
            None => false,
        }
    }

    pub fn is_create_market(&self) -> bool {
        match &self.transaction.data {
            Some(tx_data) => tx_data.is_create_market(),
            None => false,
        }
    }

    pub fn is_submit_vote(&self) -> bool {
        match &self.transaction.data {
            Some(tx_data) => tx_data.is_submit_vote(),
            None => false,
        }
    }

    pub fn is_submit_ballot(&self) -> bool {
        match &self.transaction.data {
            Some(tx_data) => tx_data.is_submit_ballot(),
            None => false,
        }
    }

    pub fn is_transfer_reputation(&self) -> bool {
        match &self.transaction.data {
            Some(tx_data) => tx_data.is_transfer_reputation(),
            None => false,
        }
    }

    pub fn claim_decision(&self) -> Option<ClaimDecision> {
        match &self.transaction.data {
            Some(TransactionData::ClaimDecision {
                decision_type,
                decisions,
            }) => Some(ClaimDecision {
                decision_type: decision_type.clone(),
                decisions: decisions.clone(),
            }),
            _ => None,
        }
    }

    /// If the tx is a market creation, returns the corresponding [`CreateMarket`].
    pub fn create_market(&self) -> Option<CreateMarket> {
        match &self.transaction.data {
            Some(TransactionData::CreateMarket {
                title,
                description,
                dimension_specs,
                trading_fee,
                category_txids,
                residual_names,
                tx_pow_hash_selector,
                tx_pow_ordering,
                tx_pow_difficulty,
            }) => Some(CreateMarket {
                title: title.clone(),
                description: description.clone(),
                dimension_specs: dimension_specs.clone(),
                trading_fee: *trading_fee,
                category_txids: category_txids.clone(),
                residual_names: residual_names.clone(),
                tx_pow_hash_selector: *tx_pow_hash_selector,
                tx_pow_ordering: *tx_pow_ordering,
                tx_pow_difficulty: *tx_pow_difficulty,
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

    pub fn is_trade(&self) -> bool {
        match &self.transaction.data {
            Some(tx_data) => tx_data.is_trade(),
            None => false,
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

    pub fn is_amplify_beta(&self) -> bool {
        match &self.transaction.data {
            Some(tx_data) => tx_data.is_amplify_beta(),
            None => false,
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

    /// Accessor for txid
    pub fn txid(&self) -> Txid {
        self.transaction.txid()
    }

    /// Return an iterator over spent outpoints/outputs
    pub fn spent_inputs(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.inputs().iter().zip(self.spent_utxos.iter())
    }

    /// Returns the total Bitcoin value spent
    pub fn spent_bitcoin_value(
        &self,
    ) -> Result<bitcoin::Amount, AmountOverflowError> {
        self.spent_utxos
            .iter()
            .map(GetBitcoinValue::get_bitcoin_value)
            .checked_sum()
            .ok_or(AmountOverflowError)
    }

    /// Returns the total Bitcoin value in the outputs
    pub fn bitcoin_value_out(
        &self,
    ) -> Result<bitcoin::Amount, AmountOverflowError> {
        self.outputs()
            .iter()
            .map(GetBitcoinValue::get_bitcoin_value)
            .checked_sum()
            .ok_or(AmountOverflowError)
    }

    /// Returns the difference between the value spent and value out, if it is
    /// non-negative.
    pub fn bitcoin_fee(
        &self,
    ) -> Result<Option<bitcoin::Amount>, AmountOverflowError> {
        let spent_value = self.spent_bitcoin_value()?;
        let value_out = self.bitcoin_value_out()?;
        if spent_value < value_out {
            Ok(None)
        } else {
            Ok(Some(spent_value - value_out))
        }
    }

    pub fn spent_assets(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.spent_inputs()
            .filter(|(_, filled_output)| filled_output.is_bitcoin())
    }

    /** Return a vector of pairs consisting of an [`AssetId`] and the combined
     *  input value for that asset.
     *  The vector is ordered such that assets occur in the same order
     *  as they first occur in the inputs. */
    pub fn unique_spent_assets(&self) -> Vec<(AssetId, u64)> {
        let mut combined_value = HashMap::<AssetId, u64>::new();
        let spent_asset_values = || {
            self.spent_assets()
                .filter_map(|(_, output)| output.asset_value())
        };
        spent_asset_values().for_each(|(asset, value)| {
            *combined_value.entry(asset).or_default() += value;
        });
        spent_asset_values()
            .unique_by(|(asset, _)| *asset)
            .map(|(asset, _)| (asset, combined_value[&asset]))
            .collect()
    }

    /** Returns an iterator over total value for each asset that must
     *  appear in the outputs, in order.
     *  The total output value can possibly over/underflow in a transaction,
     *  so the total output values are [`Option<u64>`],
     *  where `None` indicates over/underflow. */
    fn output_asset_total_values(
        &self,
    ) -> impl Iterator<Item = (AssetId, Option<u64>)> + '_ {
        self.unique_spent_assets()
            .into_iter()
            .map(|(asset, total_value)| (asset, Some(total_value)))
            .filter(|(_, amount)| *amount != Some(0))
    }

    /** Returns the max value of Bitcoin that can occur in the outputs.
     *  The total output value can possibly over/underflow in a transaction,
     *  so the total output values are [`Option<bitcoin::Amount>`],
     *  where `None` indicates over/underflow. */
    fn output_bitcoin_max_value(&self) -> Option<bitcoin::Amount> {
        self.output_asset_total_values()
            .map(|(asset_id, value)| match asset_id {
                AssetId::Bitcoin => value.map(bitcoin::Amount::from_sat),
            })
            .next()
            .unwrap_or(Some(bitcoin::Amount::ZERO))
    }

    /// Compute the filled outputs.
    /// Returns None if the outputs cannot be filled because the tx is invalid.
    ///
    /// Transaction validation ensures that all iterators over expected output amounts
    /// are fully consumed during processing. If any iterator has remaining unconsumed
    /// elements after processing all outputs, the transaction is considered invalid
    /// per Bitcoin Hivemind transaction consistency requirements.
    pub fn filled_outputs(&self) -> Option<Vec<FilledOutput>> {
        let mut output_bitcoin_max_value = self.output_bitcoin_max_value()?;

        self.outputs()
            .iter()
            .map(|output| {
                let content = match output.content.clone() {
                    OutputContent::Bitcoin(value) => {
                        let new_max =
                            output_bitcoin_max_value.checked_sub(value.0)?;
                        output_bitcoin_max_value = new_max;
                        FilledOutputContent::Bitcoin(value)
                    }
                    OutputContent::Withdrawal(withdrawal) => {
                        FilledOutputContent::BitcoinWithdrawal(withdrawal)
                    }
                    OutputContent::MarketFunds {
                        market_id,
                        amount,
                        is_fee,
                    } => {
                        let new_max =
                            output_bitcoin_max_value.checked_sub(amount.0)?;
                        output_bitcoin_max_value = new_max;
                        FilledOutputContent::MarketFunds {
                            market_id,
                            amount,
                            is_fee,
                        }
                    }
                };
                Some(FilledOutput {
                    address: output.address,
                    content,
                    memo: output.memo.clone(),
                })
            })
            .collect()
    }
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize)]
pub struct Authorized<T> {
    pub transaction: T,
    /// Authorizations are called witnesses in Bitcoin.
    pub authorizations: Vec<Authorization>,
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
