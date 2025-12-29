use std::collections::{HashMap, HashSet};

use bitcoin::amount::CheckedSum as _;
use borsh::BorshSerialize;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use utoipa::{PartialSchema, ToSchema};

use crate::{
    authorization::Authorization,
    state::markets::MarketId,
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

#[derive(
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
        #[borsh(serialize_with = "borsh_serialize_bitcoin_outpoint")]
        bitcoin::OutPoint,
    ),
    // Market treasury UTXO - tracks the single source of truth for market funds
    Market {
        market_id: [u8; 6],
        block_height: u32,
    },
    // Market author fee UTXO - accumulates trading fees for the market creator
    MarketAuthorFee {
        market_id: [u8; 6],
        block_height: u32,
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
            Self::Market {
                market_id,
                block_height,
            } => {
                write!(f, "market {} {}", hex::encode(market_id), block_height)
            }
            Self::MarketAuthorFee {
                market_id,
                block_height,
            } => {
                write!(
                    f,
                    "market_author_fee {} {}",
                    hex::encode(market_id),
                    block_height
                )
            }
        }
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
    /// Consumed by votecoin redistribution during consensus
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
pub struct VoteBatchItem {
    /// 3 byte slot ID
    pub slot_id_bytes: [u8; 3],
    /// The vote value (0.0-1.0 for binary, scaled range for scaled decisions)
    pub vote_value: f64,
}

#[allow(clippy::enum_variant_names)]
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
#[schema(as = TxData)]
pub enum TransactionData {
    /// Claim a decision slot
    ClaimDecisionSlot {
        /// 3 byte slot ID
        slot_id_bytes: [u8; 3],
        /// Whether it is a Standard slot or not
        is_standard: bool,
        /// Scaled (true) or Binary (false) decision
        is_scaled: bool,
        /// Human readable question for voters (<1000 bytes)
        question: String,
        /// Min value (only for scaled decisions)
        min: Option<u16>,
        /// Max value (only for scaled decisions)
        max: Option<u16>,
    },
    /// Create a prediction market
    CreateMarket {
        /// Market title
        title: String,
        /// Market description
        description: String,
        /// Decision slot IDs (hex-encoded)
        decision_slots: Vec<String>,
        /// Market type: "independent" or "categorical"
        market_type: String,
        /// Has residual outcome (for categorical markets)
        has_residual: Option<bool>,
        /// LMSR beta parameter (required - liquidity calculated as b * ln(num_states))
        b: f64,
        /// Trading fee percentage
        trading_fee: Option<f64>,
        /// Categorization tags
        tags: Option<Vec<String>>,
    },
    /// Create a multidimensional prediction market with mixed dimension types
    CreateMarketDimensional {
        /// Market title
        title: String,
        /// Market description
        description: String,
        /// Dimension specification: "[slot1,[slot2,slot3],slot4]"
        dimensions: String,
        /// LMSR beta parameter (required - liquidity calculated as b * ln(num_states))
        b: f64,
        /// Trading fee percentage
        trading_fee: Option<f64>,
        /// Categorization tags
        tags: Option<Vec<String>>,
    },
    /// Buy shares in a prediction market by spending Bitcoin on L2
    BuyShares {
        /// Market ID standardized across all transaction types per Bitcoin Hivemind specifications
        market_id: MarketId,
        /// Outcome index to buy shares for
        outcome_index: u32,
        /// Number of shares to buy
        shares_to_buy: f64,
        /// Maximum cost willing to pay (in satoshis)
        max_cost: u64,
    },
    /// Submit a vote for a decision in the current voting period
    SubmitVote {
        /// 3 byte slot ID of the decision being voted on
        slot_id_bytes: [u8; 3],
        /// The vote value (0.0-1.0 for binary, scaled range for scaled decisions)
        vote_value: f64,
        /// The voting period this vote belongs to
        voting_period: u32,
    },
    /// Register as a voter (one-time setup)
    RegisterVoter {
        /// Reserved for future voter metadata
        initial_data: [u8; 32],
    },
    /// Submit multiple votes efficiently
    SubmitVoteBatch {
        /// List of votes to submit
        votes: Vec<VoteBatchItem>,
        /// The voting period these votes belong to
        voting_period: u32,
    },
}

pub type TxData = TransactionData;

impl TxData {
    /// `true` if the tx data corresponds to a decision slot claim
    pub fn is_claim_decision_slot(&self) -> bool {
        matches!(self, Self::ClaimDecisionSlot { .. })
    }

    /// `true` if the tx data corresponds to market creation
    pub fn is_create_market(&self) -> bool {
        matches!(self, Self::CreateMarket { .. })
    }

    /// `true` if the tx data corresponds to dimensional market creation
    pub fn is_create_market_dimensional(&self) -> bool {
        matches!(self, Self::CreateMarketDimensional { .. })
    }

    /// `true` if the tx data corresponds to buying shares
    pub fn is_buy_shares(&self) -> bool {
        matches!(self, Self::BuyShares { .. })
    }

    /// `true` if the tx data corresponds to submitting a vote
    pub fn is_submit_vote(&self) -> bool {
        matches!(self, Self::SubmitVote { .. })
    }

    /// `true` if the tx data corresponds to registering a voter
    pub fn is_register_voter(&self) -> bool {
        matches!(self, Self::RegisterVoter { .. })
    }

    /// `true` if the tx data corresponds to submitting a batch of votes
    pub fn is_submit_vote_batch(&self) -> bool {
        matches!(self, Self::SubmitVoteBatch { .. })
    }
}

/// Struct describing a decision slot claim
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimDecisionSlot {
    /// 3 byte slot ID
    pub slot_id_bytes: [u8; 3],
    /// Whether it is a Standard slot or not
    pub is_standard: bool,
    /// Scaled (true) or Binary (false) decision
    pub is_scaled: bool,
    /// Human readable question for voters (<1000 bytes)
    pub question: String,
    /// Min value (only for scaled decisions)
    pub min: Option<u16>,
    /// Max value (only for scaled decisions)
    pub max: Option<u16>,
}

/// Struct describing a market creation
#[derive(Clone, Debug, PartialEq)]
pub struct CreateMarket {
    /// Market title
    pub title: String,
    /// Market description
    pub description: String,
    /// Decision slot IDs (hex-encoded)
    pub decision_slots: Vec<String>,
    /// Market type: "independent" or "categorical"
    pub market_type: String,
    /// Has residual outcome (for categorical markets)
    pub has_residual: Option<bool>,
    /// LMSR beta parameter (required - liquidity calculated as b * ln(num_states))
    pub b: f64,
    /// Trading fee percentage
    pub trading_fee: Option<f64>,
    /// Categorization tags
    pub tags: Option<Vec<String>>,
}

/// Struct describing a dimensional market creation
#[derive(Clone, Debug, PartialEq)]
pub struct CreateMarketDimensional {
    /// Market title
    pub title: String,
    /// Market description
    pub description: String,
    /// Dimension specification: "[slot1,[slot2,slot3],slot4]"
    pub dimensions: String,
    /// LMSR beta parameter (required - liquidity calculated as b * ln(num_states))
    pub b: f64,
    /// Trading fee percentage
    pub trading_fee: Option<f64>,
    /// Categorization tags
    pub tags: Option<Vec<String>>,
}

/// Struct describing a share buy operation
#[derive(Clone, Debug, PartialEq)]
pub struct BuyShares {
    /// Market ID standardized across all transaction types per Bitcoin Hivemind specifications
    pub market_id: MarketId,
    /// Outcome index to buy shares for
    pub outcome_index: u32,
    /// Number of shares to buy
    pub shares_to_buy: f64,
    /// Maximum cost willing to pay (in satoshis)
    pub max_cost: u64,
}

/// Struct describing a vote submission
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubmitVote {
    /// 3 byte slot ID of the decision being voted on
    pub slot_id_bytes: [u8; 3],
    /// The vote value (0.0-1.0 for binary, scaled range for scaled decisions)
    pub vote_value: f64,
    /// The voting period this vote belongs to
    pub voting_period: u32,
}

/// Struct describing voter registration
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisterVoter {
    /// Reserved for future voter metadata
    pub initial_data: [u8; 32],
}

/// Struct describing a batch vote submission
#[derive(Clone, Debug, PartialEq)]
pub struct SubmitVoteBatch {
    /// List of votes to submit
    pub votes: Vec<VoteBatchItem>,
    /// The voting period these votes belong to
    pub voting_period: u32,
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

    /// Return an iterator over Votecoin outputs
    pub fn votecoin_outputs(&self) -> impl Iterator<Item = &Output> {
        self.outputs.iter().filter(|output| output.is_votecoin())
    }

    /// `true` if the tx data corresponds to a regular tx
    pub fn is_regular(&self) -> bool {
        self.data.is_none()
    }

    pub fn txid(&self) -> Txid {
        hashes::hash(self).into()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FilledTransaction {
    pub transaction: Transaction,
    pub spent_utxos: Vec<FilledOutput>,
}

impl FilledTransaction {
    // Return an iterator over Votecoin outputs
    pub fn votecoin_outputs(&self) -> impl Iterator<Item = &Output> {
        self.transaction.votecoin_outputs()
    }

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

    /// `true` if the tx data corresponds to a decision slot claim
    pub fn is_claim_decision_slot(&self) -> bool {
        match &self.transaction.data {
            Some(tx_data) => tx_data.is_claim_decision_slot(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to market creation
    pub fn is_create_market(&self) -> bool {
        match &self.transaction.data {
            Some(tx_data) => tx_data.is_create_market(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to submitting a vote
    pub fn is_submit_vote(&self) -> bool {
        match &self.transaction.data {
            Some(tx_data) => tx_data.is_submit_vote(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to registering a voter
    pub fn is_register_voter(&self) -> bool {
        match &self.transaction.data {
            Some(tx_data) => tx_data.is_register_voter(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to submitting a batch of votes
    pub fn is_submit_vote_batch(&self) -> bool {
        match &self.transaction.data {
            Some(tx_data) => tx_data.is_submit_vote_batch(),
            None => false,
        }
    }

    /// If the tx is a decision slot claim, returns the corresponding [`ClaimDecisionSlot`].
    pub fn claim_decision_slot(&self) -> Option<ClaimDecisionSlot> {
        match &self.transaction.data {
            Some(TransactionData::ClaimDecisionSlot {
                slot_id_bytes,
                is_standard,
                is_scaled,
                question,
                min,
                max,
            }) => Some(ClaimDecisionSlot {
                slot_id_bytes: *slot_id_bytes,
                is_standard: *is_standard,
                is_scaled: *is_scaled,
                question: question.clone(),
                min: *min,
                max: *max,
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
                decision_slots,
                market_type,
                has_residual,
                b,
                trading_fee,
                tags,
            }) => Some(CreateMarket {
                title: title.clone(),
                description: description.clone(),
                decision_slots: decision_slots.clone(),
                market_type: market_type.clone(),
                has_residual: *has_residual,
                b: *b,
                trading_fee: *trading_fee,
                tags: tags.clone(),
            }),
            _ => None,
        }
    }

    /// If the tx is a dimensional market creation, returns the corresponding [`CreateMarketDimensional`].
    pub fn create_market_dimensional(&self) -> Option<CreateMarketDimensional> {
        match &self.transaction.data {
            Some(TransactionData::CreateMarketDimensional {
                title,
                description,
                dimensions,
                b,
                trading_fee,
                tags,
            }) => Some(CreateMarketDimensional {
                title: title.clone(),
                description: description.clone(),
                dimensions: dimensions.clone(),
                b: *b,
                trading_fee: *trading_fee,
                tags: tags.clone(),
            }),
            _ => None,
        }
    }

    /// If the tx is a share buy, returns the corresponding [`BuyShares`].
    pub fn buy_shares(&self) -> Option<BuyShares> {
        match &self.transaction.data {
            Some(TransactionData::BuyShares {
                market_id,
                outcome_index,
                shares_to_buy,
                max_cost,
            }) => Some(BuyShares {
                market_id: market_id.clone(),
                outcome_index: *outcome_index,
                shares_to_buy: *shares_to_buy,
                max_cost: *max_cost,
            }),
            _ => None,
        }
    }

    /// If the tx is a vote submission, returns the corresponding [`SubmitVote`].
    pub fn submit_vote(&self) -> Option<SubmitVote> {
        match &self.transaction.data {
            Some(TransactionData::SubmitVote {
                slot_id_bytes,
                vote_value,
                voting_period,
            }) => Some(SubmitVote {
                slot_id_bytes: *slot_id_bytes,
                vote_value: *vote_value,
                voting_period: *voting_period,
            }),
            _ => None,
        }
    }

    /// If the tx is a voter registration, returns the corresponding [`RegisterVoter`].
    pub fn register_voter(&self) -> Option<RegisterVoter> {
        match &self.transaction.data {
            Some(TransactionData::RegisterVoter { initial_data }) => {
                Some(RegisterVoter {
                    initial_data: *initial_data,
                })
            }
            _ => None,
        }
    }

    /// If the tx is a vote batch submission, returns the corresponding [`SubmitVoteBatch`].
    pub fn submit_vote_batch(&self) -> Option<SubmitVoteBatch> {
        match &self.transaction.data {
            Some(TransactionData::SubmitVoteBatch {
                votes,
                voting_period,
            }) => Some(SubmitVoteBatch {
                votes: votes.clone(),
                voting_period: *voting_period,
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

    /// Return an iterator over spent Votecoin
    pub fn spent_votecoin(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.spent_inputs()
            .filter(|(_, filled_output)| filled_output.is_votecoin())
    }

    /** Return an iterator over spent assets (Bitcoin and Votecoin) */
    pub fn spent_assets(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.spent_inputs().filter(|(_, filled_output)| {
            filled_output.is_bitcoin() || filled_output.is_votecoin()
        })
    }

    /** Returns the total amount of Votecoin spent in this transaction.
     *  Since Votecoin has a fixed supply with no subtypes, this returns
     *  a simple total amount. */
    pub fn unique_spent_votecoin(&self) -> Option<u32> {
        let mut total_votecoin: u64 = 0;
        for (_, output) in self.spent_votecoin() {
            if let Some(amount) = output.votecoin() {
                total_votecoin += amount as u64;
            }
        }
        if total_votecoin > 0 {
            // Convert back to u32, should not overflow since Votecoin uses u32
            total_votecoin.try_into().ok()
        } else {
            None
        }
    }

    /** Return a vector of pairs consisting of an [`AssetId`] and the combined
     *  input value for that asset.
     *  The vector is ordered such that assets occur in the same order
     *  as they first occur in the inputs. */
    pub fn unique_spent_assets(&self) -> Vec<(AssetId, u64)> {
        // Combined value for each asset
        let mut combined_value = HashMap::<AssetId, u64>::new();
        let spent_asset_values = || {
            self.spent_assets()
                .filter_map(|(_, output)| output.asset_value())
        };
        // Increment combined value for the asset
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
            .find_map(|(asset_id, value)| match asset_id {
                AssetId::Bitcoin => Some(value.map(bitcoin::Amount::from_sat)),
                _ => None,
            })
            .unwrap_or(Some(bitcoin::Amount::ZERO))
    }

    /** Returns an iterator over total amount for each LP token that must
     *  appear in the outputs, in order.
     *  The total output value can possibly over/underflow,
     *  so the total output values are [`Option<u64>`],
     *  where `None` indicates over/underflow. */
    fn output_lp_token_total_amounts(
        &self,
    ) -> impl Iterator<Item = (AssetId, AssetId, Option<u64>)> + '_ {
        std::iter::empty()
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
        let mut output_lp_token_total_amounts =
            self.output_lp_token_total_amounts().peekable();

        let outputs = self
            .outputs()
            .iter()
            .map(|output| {
                let content = match output.content.clone() {
                    OutputContent::Votecoin(value) => {
                        FilledOutputContent::Votecoin(value)
                    }
                    OutputContent::Bitcoin(value) => {
                        let new_max =
                            output_bitcoin_max_value.checked_sub(value.0);
                        new_max?;

                        output_bitcoin_max_value = new_max.unwrap();
                        FilledOutputContent::Bitcoin(value)
                    }
                    OutputContent::Withdrawal(withdrawal) => {
                        FilledOutputContent::BitcoinWithdrawal(withdrawal)
                    }
                    OutputContent::MarketTreasury { market_id, amount } => {
                        let new_max =
                            output_bitcoin_max_value.checked_sub(amount.0);
                        new_max?;

                        output_bitcoin_max_value = new_max.unwrap();
                        FilledOutputContent::MarketTreasury {
                            market_id,
                            amount,
                        }
                    }
                    OutputContent::MarketAuthorFee { market_id, amount } => {
                        let new_max =
                            output_bitcoin_max_value.checked_sub(amount.0);
                        new_max?;

                        output_bitcoin_max_value = new_max.unwrap();
                        FilledOutputContent::MarketAuthorFee {
                            market_id,
                            amount,
                        }
                    }
                };
                Some(FilledOutput {
                    address: output.address,
                    content,
                    memo: output.memo.clone(),
                })
            })
            .collect::<Option<Vec<FilledOutput>>>();

        let outputs = outputs?;

        // Validate that all LP token iterators are fully consumed.
        // If there are remaining unconsumed elements, the transaction has
        // inconsistent LP token allocations and must be marked invalid.
        if output_lp_token_total_amounts.peek().is_some() {
            return None;
        }

        Some(outputs)
    }
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize)]
pub struct Authorized<T> {
    pub transaction: T,
    /// Authorizations are called witnesses in Bitcoin.
    pub authorizations: Vec<Authorization>,
}

pub type AuthorizedTransaction = Authorized<Transaction>;

impl AuthorizedTransaction {
    /// Return an iterator over all addresses relevant to the transaction
    pub fn relevant_addresses(&self) -> HashSet<Address> {
        let input_addrs =
            self.authorizations.iter().map(|auth| auth.get_address());
        let output_addrs =
            self.transaction.outputs.iter().map(|output| output.address);
        input_addrs.chain(output_addrs).collect()
    }
}

impl From<Authorized<FilledTransaction>> for AuthorizedTransaction {
    fn from(tx: Authorized<FilledTransaction>) -> Self {
        Self {
            transaction: tx.transaction.transaction,
            authorizations: tx.authorizations,
        }
    }
}
