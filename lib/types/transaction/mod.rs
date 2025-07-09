use std::{
    borrow::Borrow,
    cmp::Ordering,
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
    types::{
        AmountOverflowError, TruthcoinData, TruthcoinDataUpdates, GetAddress,
        GetBitcoinValue,
        address::Address,
        hashes::{
            self, AssetId, TruthcoinId, DutchAuctionId, Hash, M6id, MerkleRoot,
            Txid,
        },
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
        debug_assert_eq!(cursor.position() as usize, OUTPOINT_KEY_SIZE);
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

// Database key encoding traits for direct LMDB usage
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
}

pub type TxInputs = Vec<OutPoint>;

pub type TxOutputs = Vec<Output>;

/// Parameters of a Dutch Auction
#[derive(
    BorshSerialize, Clone, Copy, Debug, Deserialize, Serialize, ToSchema,
)]
#[cfg_attr(feature = "clap", derive(clap::Parser))]
pub struct DutchAuctionParams {
    /// Block height at which the auction starts
    #[cfg_attr(feature = "clap", arg(long))]
    pub start_block: u32,
    /// Auction duration, in blocks
    #[cfg_attr(feature = "clap", arg(long))]
    pub duration: u32,
    /// The asset to be auctioned
    #[cfg_attr(feature = "clap", arg(long))]
    pub base_asset: AssetId,
    /// The amount of the base asset to be auctioned
    #[cfg_attr(feature = "clap", arg(long))]
    pub base_amount: u64,
    /// The asset in which the auction is to be quoted
    #[cfg_attr(feature = "clap", arg(long))]
    pub quote_asset: AssetId,
    /// Initial price
    #[cfg_attr(feature = "clap", arg(long))]
    pub initial_price: u64,
    /// Final price
    #[cfg_attr(feature = "clap", arg(long))]
    pub final_price: u64,
}

#[allow(clippy::enum_variant_names)]
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
#[schema(as = TxData)]
pub enum TransactionData {
    /// Burn an AMM position
    AmmBurn {
        /// Amount of the lexicographically ordered first Truthcoin to receive
        amount0: u64,
        /// Amount of the lexicographically ordered second Truthcoin to receive
        amount1: u64,
        /// Amount of the LP token to burn
        lp_token_burn: u64,
    },
    /// Mint an AMM position
    AmmMint {
        /// Amount of the lexicographically ordered first Truthcoin to deposit
        amount0: u64,
        /// Amount of the lexicographically ordered second Truthcoin to deposit
        amount1: u64,
        /// Amount of the LP token to receive
        lp_token_mint: u64,
    },
    /// AMM swap
    AmmSwap {
        /// Amount to spend
        amount_spent: u64,
        /// Amount to receive
        amount_receive: u64,
        /// Pair asset to swap for
        pair_asset: AssetId,
    },
    TruthcoinReservation {
        /// Commitment to the Truthcoin that will be registered
        #[serde(with = "serde_hexstr_human_readable")]
        #[schema(value_type = String)]
        commitment: Hash,
    },
    TruthcoinRegistration {
        /// Reveal of the name hash
        #[serde(with = "serde_hexstr_human_readable")]
        #[schema(value_type = String)]
        name_hash: Hash,
        /// Reveal of the nonce used for the Truthcoin reservation commitment
        #[serde(with = "serde_hexstr_human_readable")]
        #[schema(value_type = String)]
        revealed_nonce: Hash,
        /// Initial Truthcoin data
        truthcoin_data: Box<TruthcoinData>,
        /// Amount to mint
        initial_supply: u64,
    },
    /// Mint more of a Truthcoin
    TruthcoinMint(u64),
    TruthcoinUpdate(Box<TruthcoinDataUpdates>),
    DutchAuctionCreate(DutchAuctionParams),
    DutchAuctionBid {
        auction_id: DutchAuctionId,
        /// Asset to receive in the auction
        receive_asset: AssetId,
        /// Quantity to purchase in the auction
        quantity: u64,
        /// Total bid size, in terms of the quote asset
        bid_size: u64,
    },
    DutchAuctionCollect {
        /// Base asset
        asset_offered: AssetId,
        /// Quote asset
        asset_receive: AssetId,
        /// Amount of the offered base asset
        amount_offered_remaining: u64,
        /// Amount of the received quote asset
        amount_received: u64,
    },
}

pub type TxData = TransactionData;

impl TxData {
    /// `true` if the tx data corresponds to an AMM burn
    pub fn is_amm_burn(&self) -> bool {
        matches!(self, Self::AmmBurn { .. })
    }

    /// `true` if the tx data corresponds to an AMM mint
    pub fn is_amm_mint(&self) -> bool {
        matches!(self, Self::AmmMint { .. })
    }

    /// `true` if the tx data corresponds to an AMM swap
    pub fn is_amm_swap(&self) -> bool {
        matches!(self, Self::AmmSwap { .. })
    }

    /// `true` if the tx data corresponds to a Dutch auction bid
    pub fn is_dutch_auction_bid(&self) -> bool {
        matches!(self, Self::DutchAuctionBid { .. })
    }

    /// `true` if the tx data corresponds to a Dutch auction creation
    pub fn is_dutch_auction_create(&self) -> bool {
        matches!(self, Self::DutchAuctionCreate(_))
    }

    /// `true` if the tx data corresponds to a Dutch auction collect
    pub fn is_dutch_auction_collect(&self) -> bool {
        matches!(self, Self::DutchAuctionCollect { .. })
    }

    /// `true` if the tx data corresponds to a registration
    pub fn is_registration(&self) -> bool {
        matches!(self, Self::TruthcoinRegistration { .. })
    }

    /// `true` if the tx data corresponds to a reservation
    pub fn is_reservation(&self) -> bool {
        matches!(self, Self::TruthcoinReservation { .. })
    }

    /// `true` if the tx data corresponds to an update
    pub fn is_update(&self) -> bool {
        matches!(self, Self::TruthcoinUpdate(_))
    }
}

/// Struct describing an AMM burn
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AmmBurn {
    pub asset0: AssetId,
    pub asset1: AssetId,
    /// Amount of asset 0 received
    pub amount0: u64,
    /// Amount of asset 1 received
    pub amount1: u64,
    /// Amount of LP token burned
    pub lp_token_burn: u64,
}

/// Struct describing an AMM mint
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AmmMint {
    pub asset0: AssetId,
    pub asset1: AssetId,
    /// Amount of asset 0 deposited
    pub amount0: u64,
    /// Amount of asset 1 deposited
    pub amount1: u64,
    /// Amount of LP token received
    pub lp_token_mint: u64,
}

/// Struct describing an AMM swap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AmmSwap {
    pub asset_spend: AssetId,
    pub asset_receive: AssetId,
    /// Amount of spend asset spent
    pub amount_spend: u64,
    //// Amount of receive asset received
    pub amount_receive: u64,
}

/// Struct describing a Dutch auction bid
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DutchAuctionBid {
    pub auction_id: DutchAuctionId,
    pub asset_spend: AssetId,
    pub asset_receive: AssetId,
    /// Amount of spend asset spent
    pub amount_spend: u64,
    //// Amount of receive asset received
    pub amount_receive: u64,
}

/// Struct describing a Dutch auction collect
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DutchAuctionCollect {
    pub auction_id: DutchAuctionId,
    pub asset_offered: AssetId,
    pub asset_receive: AssetId,
    /// Amount of offered asset remaining
    pub amount_offered_remaining: u64,
    //// Amount of receive asset received
    pub amount_received: u64,
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

    /// Return an iterator over Truthcoin outputs
    pub fn truthcoin_outputs(&self) -> impl Iterator<Item = &Output> {
        self.outputs.iter().filter(|output| output.is_truthcoin())
    }

    /// Return an iterator over Truthcoin control outputs
    pub fn truthcoin_control_outputs(&self) -> impl Iterator<Item = &Output> {
        self.outputs
            .iter()
            .filter(|output| output.is_truthcoin_control())
    }

    /// `true` if the tx data corresponds to an AMM burn
    pub fn is_amm_burn(&self) -> bool {
        match &self.data {
            Some(tx_data) => tx_data.is_amm_burn(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to an AMM mint
    pub fn is_amm_mint(&self) -> bool {
        match &self.data {
            Some(tx_data) => tx_data.is_amm_mint(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to an AMM swap
    pub fn is_amm_swap(&self) -> bool {
        match &self.data {
            Some(tx_data) => tx_data.is_amm_swap(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to a Dutch auction bid
    pub fn is_dutch_auction_bid(&self) -> bool {
        match &self.data {
            Some(tx_data) => tx_data.is_dutch_auction_bid(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to a Dutch auction creation
    pub fn is_dutch_auction_create(&self) -> bool {
        match &self.data {
            Some(tx_data) => tx_data.is_dutch_auction_create(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to a Dutch auction collect
    pub fn is_dutch_auction_collect(&self) -> bool {
        match &self.data {
            Some(tx_data) => tx_data.is_dutch_auction_collect(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to a Truthcoin registration
    pub fn is_registration(&self) -> bool {
        match &self.data {
            Some(tx_data) => tx_data.is_registration(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to a regular tx
    pub fn is_regular(&self) -> bool {
        self.data.is_none()
    }

    /// `true` if the tx data corresponds to a reservation
    pub fn is_reservation(&self) -> bool {
        match &self.data {
            Some(tx_data) => tx_data.is_reservation(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to an update
    pub fn is_update(&self) -> bool {
        match &self.data {
            Some(tx_data) => tx_data.is_update(),
            None => false,
        }
    }

    /// If the tx is a truthcoin registration, returns the registered name hash
    pub fn registration_name_hash(&self) -> Option<Hash> {
        match self.data {
            Some(TxData::TruthcoinRegistration { name_hash, .. }) => {
                Some(name_hash)
            }
            _ => None,
        }
    }

    /** If the tx is a truthcoin registration, returns the implied reservation
     * commitment */
    pub fn implied_reservation_commitment(&self) -> Option<Hash> {
        match self.data {
            Some(TxData::TruthcoinRegistration {
                name_hash,
                revealed_nonce,
                ..
            }) => {
                let implied_commitment =
                    blake3::keyed_hash(&revealed_nonce, &name_hash).into();
                Some(implied_commitment)
            }
            _ => None,
        }
    }

    /// Return an iterator over reservation outputs
    pub fn reservation_outputs(
        &self,
    ) -> impl DoubleEndedIterator<Item = &Output> {
        self.outputs.iter().filter(|output| output.is_reservation())
    }

    pub fn txid(&self) -> Txid {
        hashes::hash_with_scratch_buffer(self).into()
    }

    /// If the tx is a truthcoin reservation, returns the reservation commitment
    pub fn reservation_commitment(&self) -> Option<Hash> {
        match self.data {
            Some(TxData::TruthcoinReservation { commitment }) => {
                Some(commitment)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FilledTransaction {
    pub transaction: Transaction,
    pub spent_utxos: Vec<FilledOutput>,
}

impl FilledTransaction {
    // Return an iterator over Truthcoin outputs
    pub fn truthcoin_outputs(&self) -> impl Iterator<Item = &Output> {
        self.transaction.truthcoin_outputs()
    }

    // Return an iterator over Truthcoin control outputs
    pub fn truthcoin_control_outputs(&self) -> impl Iterator<Item = &Output> {
        self.transaction.truthcoin_control_outputs()
    }

    /// Accessor for tx data
    pub fn data(&self) -> &Option<TxData> {
        &self.transaction.data
    }

    /** If the tx is a truthcoin registration, returns the implied reservation
     * commitment */
    pub fn implied_reservation_commitment(&self) -> Option<Hash> {
        self.transaction.implied_reservation_commitment()
    }

    /// Accessor for tx outputs
    pub fn inputs(&self) -> &TxInputs {
        &self.transaction.inputs
    }

    /// `true` if the tx data corresponds to an AMM burn
    pub fn is_amm_burn(&self) -> bool {
        self.transaction.is_amm_burn()
    }

    /// `true` if the tx data corresponds to an AMM mint
    pub fn is_amm_mint(&self) -> bool {
        self.transaction.is_amm_mint()
    }

    /// `true` if the tx data corresponds to an AMM swap
    pub fn is_amm_swap(&self) -> bool {
        self.transaction.is_amm_swap()
    }

    /// `true` if the tx data corresponds to a Dutch auction bid
    pub fn is_dutch_auction_bid(&self) -> bool {
        self.transaction.is_dutch_auction_bid()
    }

    /// `true` if the tx data corresponds to a Dutch auction creation
    pub fn is_dutch_auction_create(&self) -> bool {
        self.transaction.is_dutch_auction_create()
    }

    /// `true` if the tx data corresponds to a Dutch auction collect
    pub fn is_dutch_auction_collect(&self) -> bool {
        self.transaction.is_dutch_auction_collect()
    }

    /// `true` if the tx data corresponds to a Truthcoin registration
    pub fn is_registration(&self) -> bool {
        self.transaction.is_registration()
    }

    /// `true` if the tx data corresponds to a regular tx
    pub fn is_regular(&self) -> bool {
        self.transaction.is_regular()
    }

    /// `true` if the tx data corresponds to a Truthcoin reservation
    pub fn is_reservation(&self) -> bool {
        self.transaction.is_reservation()
    }

    /// `true` if the tx data corresponds to a Truthcoin update
    pub fn is_update(&self) -> bool {
        self.transaction.is_update()
    }

    /// Accessor for tx outputs
    pub fn outputs(&self) -> &TxOutputs {
        &self.transaction.outputs
    }

    /** If the tx is an AMM burn, returns the LP token's
     *  corresponding [`AmmBurn`]. */
    pub fn amm_burn(&self) -> Option<AmmBurn> {
        match self.transaction.data {
            Some(TransactionData::AmmBurn {
                amount0,
                amount1,
                lp_token_burn,
            }) => {
                let unique_spent_lp_tokens = self.unique_spent_lp_tokens();
                let (asset0, asset1, _) = unique_spent_lp_tokens.first()?;
                Some(AmmBurn {
                    asset0: *asset0,
                    asset1: *asset1,
                    amount0,
                    amount1,
                    lp_token_burn,
                })
            }
            _ => None,
        }
    }

    /// If the tx is an AMM mint, returns the corresponding [`AmmMint`].
    pub fn amm_mint(&self) -> Option<AmmMint> {
        match self.transaction.data {
            Some(TransactionData::AmmMint {
                amount0,
                amount1,
                lp_token_mint,
            }) => match self.unique_spent_assets().get(0..=1) {
                Some([(first_asset, _), (second_asset, _)]) => {
                    let mut assets = [first_asset, second_asset];
                    assets.sort();
                    let [asset0, asset1] = assets;
                    Some(AmmMint {
                        asset0: *asset0,
                        asset1: *asset1,
                        amount0,
                        amount1,
                        lp_token_mint,
                    })
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// If the tx is an AMM swap, returns the corresponding [`AmmSwap`].
    pub fn amm_swap(&self) -> Option<AmmSwap> {
        match self.transaction.data {
            Some(TransactionData::AmmSwap {
                amount_spent,
                amount_receive,
                pair_asset,
            }) => {
                let (spent_asset, _) = *self.unique_spent_assets().first()?;
                Some(AmmSwap {
                    asset_spend: spent_asset,
                    asset_receive: pair_asset,
                    amount_spend: amount_spent,
                    amount_receive,
                })
            }
            _ => None,
        }
    }

    /** If the tx is a valid Truthcoin mint,
     *  returns the Truthcoin ID and mint amount */
    pub fn truthcoin_mint(&self) -> Option<(TruthcoinId, u64)> {
        match self.transaction.data {
            Some(TransactionData::TruthcoinMint(amount)) => {
                let (_, control_output) =
                    self.spent_truthcoin_controls().next_back()?;
                let truthcoin = control_output.get_truthcoin()?;
                Some((truthcoin, amount))
            }
            _ => None,
        }
    }

    /** If the tx is a Dutch auction bid,
     *  returns the corresponding [`DutchAuctionBid`]. */
    pub fn dutch_auction_bid(&self) -> Option<DutchAuctionBid> {
        match self.transaction.data {
            Some(TransactionData::DutchAuctionBid {
                auction_id,
                receive_asset,
                quantity,
                bid_size,
            }) => {
                let unique_spent_assets = self.unique_spent_assets();
                let (asset_spend, _) = unique_spent_assets.first()?;
                Some(DutchAuctionBid {
                    auction_id,
                    asset_spend: *asset_spend,
                    asset_receive: receive_asset,
                    amount_spend: bid_size,
                    amount_receive: quantity,
                })
            }
            _ => None,
        }
    }

    /** If the tx is a Dutch auction creation,
     *  returns the corresponding [`DutchAuctionParams`]. */
    pub fn dutch_auction_create(&self) -> Option<DutchAuctionParams> {
        match self.transaction.data {
            Some(TransactionData::DutchAuctionCreate(dutch_auction_params)) => {
                Some(dutch_auction_params)
            }
            _ => None,
        }
    }

    /** If the tx is a Dutch auction collect,
     *  returns the corresponding [`DutchAuctionCollect`]. */
    pub fn dutch_auction_collect(&self) -> Option<DutchAuctionCollect> {
        match self.transaction.data {
            Some(TransactionData::DutchAuctionCollect {
                asset_offered,
                asset_receive,
                amount_offered_remaining,
                amount_received,
            }) => {
                let mut spent_dutch_auction_receipts =
                    self.spent_dutch_auction_receipts();
                let auction_id = spent_dutch_auction_receipts
                    .next()?
                    .1
                    .dutch_auction_receipt()?;
                Some(DutchAuctionCollect {
                    auction_id,
                    asset_offered,
                    asset_receive,
                    amount_offered_remaining,
                    amount_received,
                })
            }
            _ => None,
        }
    }

    /// If the tx is a Truthcoin registration, returns the registered name hash
    pub fn registration_name_hash(&self) -> Option<Hash> {
        self.transaction.registration_name_hash()
    }

    /// Return an iterator over Truthcoin reservation outputs
    pub fn reservation_outputs(&self) -> impl Iterator<Item = &Output> {
        self.transaction.reservation_outputs()
    }

    /// If the tx is a Truthcoin reservation, returns the reservation commitment
    pub fn reservation_commitment(&self) -> Option<Hash> {
        self.transaction.reservation_commitment()
    }

    /// Rccessor for txid
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
    /// Return an iterator over spent reservations
    pub fn spent_reservations(
        &self,
    ) -> impl Iterator<Item = (&OutPoint, &FilledOutput)> {
        self.spent_inputs()
            .filter(|(_, filled_output)| filled_output.is_reservation())
    }

    /// Return an iterator over spent Truthcoin
    pub fn spent_truthcoin(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.spent_inputs()
            .filter(|(_, filled_output)| filled_output.is_truthcoin())
    }

    /** Return an iterator over spent assets (Bitcoin, Truthcoin,
     * and Truthcoin control coins) */
    pub fn spent_assets(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.spent_inputs().filter(|(_, filled_output)| {
            filled_output.is_bitcoin()
                || filled_output.is_truthcoin()
                || filled_output.is_truthcoin_control()
        })
    }

    /** Return a vector of pairs consisting of a Truthcoin and the combined
     *  input value for that Truthcoin.
     *  The vector is ordered such that Truthcoin occur in the same order
     *  as they first occur in the inputs. */
    pub fn unique_spent_truthcoin(&self) -> Vec<(TruthcoinId, u64)> {
        // Combined value for each Truthcoin
        let mut combined_value = HashMap::<TruthcoinId, u64>::new();
        let spent_truthcoin_values = || {
            self.spent_truthcoin()
                .filter_map(|(_, output)| output.truthcoin_value())
        };
        // Increment combined value for the Truthcoin
        spent_truthcoin_values().for_each(|(truthcoin, value)| {
            *combined_value.entry(truthcoin).or_default() += value;
        });
        spent_truthcoin_values()
            .unique_by(|(truthcoin, _)| *truthcoin)
            .map(|(truthcoin, _)| (truthcoin, combined_value[&truthcoin]))
            .collect()
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

    /// Return an iterator over spent Truthcoin control coins
    pub fn spent_truthcoin_controls(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.spent_inputs()
            .filter(|(_, filled_output)| filled_output.is_truthcoin_control())
    }

    /// Return an iterator over spent Dutch auction receipts
    pub fn spent_dutch_auction_receipts(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.spent_inputs().filter(|(_, filled_output)| {
            filled_output.is_dutch_auction_receipt()
        })
    }

    /// Return an iterator over spent AMM LP tokens
    pub fn spent_lp_tokens(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.spent_inputs()
            .filter(|(_, filled_output)| filled_output.is_lp_token())
    }

    /** Return a vector of pairs consisting of an LP token's corresponding
     *  asset pair and the combined input amount for that LP token.
     *  The vector is ordered such that LP tokens occur in the same order
     *  as they first occur in the inputs. */
    pub fn unique_spent_lp_tokens(&self) -> Vec<(AssetId, AssetId, u64)> {
        // Combined amount for each LP token
        let mut combined_amounts = HashMap::<(AssetId, AssetId), u64>::new();
        let spent_lp_token_amounts = || {
            self.spent_lp_tokens()
                .filter_map(|(_, output)| output.lp_token_amount())
        };
        // Increment combined amount for the Truthcoin
        spent_lp_token_amounts().for_each(|(asset0, asset1, amount)| {
            *combined_amounts.entry((asset0, asset1)).or_default() += amount;
        });
        spent_lp_token_amounts()
            .unique_by(|(asset0, asset1, _)| (*asset0, *asset1))
            .map(|(asset0, asset1, _)| {
                (asset0, asset1, combined_amounts[&(asset0, asset1)])
            })
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
        /* If this tx is a Truthcoin registration, this is the Truthcoin ID and
         * value of the output corresponding to the newly created Truthcoin,
         * which must be the second-to-last registration output.
         * ie. If there are `n >= 2` outputs `0..(n-1)`,
         * output `(n-1)` is the Truthcoin control coin,
         * and output `(n-2)` is the Truthcoin mint.
         * Note that there may be no Truthcoin mint,
         * in the case that the initial supply is zero. */
        let new_truthcoin_value: Option<(TruthcoinId, u64)> =
            match self.transaction.data {
                Some(TransactionData::TruthcoinRegistration {
                    name_hash,
                    initial_supply,
                    ..
                }) if initial_supply != 0 => {
                    Some((TruthcoinId(name_hash), initial_supply))
                }
                _ => None,
            };
        let truthcoin_mint = self.truthcoin_mint();
        let (mut amm_burn0, mut amm_burn1) = match self.amm_burn() {
            Some(AmmBurn {
                asset0,
                asset1,
                amount0,
                amount1,
                lp_token_burn: _,
            }) => (Some((asset0, amount0)), Some((asset1, amount1))),
            None => (None, None),
        };
        let (mut amm_mint0, mut amm_mint1) = match self.amm_mint() {
            Some(AmmMint {
                asset0,
                asset1,
                amount0,
                amount1,
                lp_token_mint: _,
            }) => (Some((asset0, amount0)), Some((asset1, amount1))),
            None => (None, None),
        };
        let (mut amm_swap_spend, mut amm_swap_receive) = match self.amm_swap() {
            Some(AmmSwap {
                asset_spend,
                asset_receive,
                amount_spend,
                amount_receive,
            }) => (
                Some((asset_spend, amount_spend)),
                Some((asset_receive, amount_receive)),
            ),
            None => (None, None),
        };
        let (mut dutch_auction_bid_spend, mut dutch_auction_bid_receive) =
            match self.dutch_auction_bid() {
                Some(DutchAuctionBid {
                    auction_id: _,
                    asset_spend,
                    asset_receive,
                    amount_spend,
                    amount_receive,
                }) => (
                    Some((asset_spend, amount_spend)),
                    Some((asset_receive, amount_receive)),
                ),
                None => (None, None),
            };
        let mut dutch_auction_create_spend =
            self.dutch_auction_create().map(|auction_params| {
                (auction_params.base_asset, auction_params.base_amount)
            });
        let (mut dutch_auction_collect0, mut dutch_auction_collect1) =
            match self.dutch_auction_collect() {
                Some(DutchAuctionCollect {
                    auction_id: _,
                    asset_offered,
                    asset_receive,
                    amount_offered_remaining,
                    amount_received,
                }) => (
                    Some((asset_offered, amount_offered_remaining)),
                    Some((asset_receive, amount_received)),
                ),
                None => (None, None),
            };
        self.unique_spent_assets()
            .into_iter()
            .map(move |(asset, total_value)| {
                let total_value = if let Some((mint_truthcoin, mint_amount)) =
                    truthcoin_mint
                    && AssetId::Truthcoin(mint_truthcoin) == asset
                {
                    total_value.checked_add(mint_amount)
                } else if let Some((burn_asset, burn_amount)) = amm_burn0
                    && burn_asset == asset
                {
                    amm_burn0 = None;
                    total_value.checked_add(burn_amount)
                } else if let Some((burn_asset, burn_amount)) = amm_burn1
                    && burn_asset == asset
                {
                    amm_burn1 = None;
                    total_value.checked_add(burn_amount)
                } else if let Some((mint_asset, mint_amount)) = amm_mint0
                    && mint_asset == asset
                {
                    amm_mint0 = None;
                    total_value.checked_sub(mint_amount)
                } else if let Some((mint_asset, mint_amount)) = amm_mint1
                    && mint_asset == asset
                {
                    amm_mint1 = None;
                    total_value.checked_sub(mint_amount)
                } else if let Some((swap_spend_asset, swap_spend_amount)) =
                    amm_swap_spend
                    && swap_spend_asset == asset
                {
                    amm_swap_spend = None;
                    total_value.checked_sub(swap_spend_amount)
                } else if let Some((swap_receive_asset, swap_receive_amount)) =
                    amm_swap_receive
                    && swap_receive_asset == asset
                {
                    amm_swap_receive = None;
                    total_value.checked_add(swap_receive_amount)
                } else if let Some((receive_asset, receive_amount)) =
                    dutch_auction_bid_receive
                    && receive_asset == asset
                {
                    dutch_auction_bid_receive = None;
                    total_value.checked_add(receive_amount)
                } else if let Some((spend_asset, spend_amount)) =
                    dutch_auction_bid_spend
                    && spend_asset == asset
                {
                    dutch_auction_bid_spend = None;
                    total_value.checked_sub(spend_amount)
                } else if let Some((spend_asset, spend_amount)) =
                    dutch_auction_create_spend
                    && spend_asset == asset
                {
                    dutch_auction_create_spend = None;
                    total_value.checked_sub(spend_amount)
                } else if let Some((receive_asset, receive_amount)) =
                    dutch_auction_collect0
                    && receive_asset == asset
                {
                    dutch_auction_collect0 = None;
                    total_value.checked_add(receive_amount)
                } else if let Some((receive_asset, receive_amount)) =
                    dutch_auction_collect1
                    && receive_asset == asset
                {
                    dutch_auction_collect1 = None;
                    total_value.checked_add(receive_amount)
                } else {
                    Some(total_value)
                };
                (asset, total_value)
            })
            .filter(|(_, amount)| *amount != Some(0))
            .chain(amm_burn0.map(|(burn_asset, burn_amount)| {
                (burn_asset, Some(burn_amount))
            }))
            .chain(amm_burn1.map(|(burn_asset, burn_amount)| {
                (burn_asset, Some(burn_amount))
            }))
            .chain(amm_mint0.map(|(mint_asset, _)|
                    /* If the Truthcoin are not already accounted for,
                    * indicate an underflow */
                    (mint_asset, None)))
            .chain(amm_mint1.map(|(mint_asset, _)|
                    /* If the Truthcoin are not already accounted for,
                    * indicate an underflow */
                    (mint_asset, None)))
            .chain(amm_swap_spend.map(|(spend_asset, _)|
                    /* If the Truthcoin are not already accounted for,
                    * indicate an underflow */
                    (spend_asset, None)))
            .chain(amm_swap_receive.map(|(receive_asset, receive_amount)| {
                (receive_asset, Some(receive_amount))
            }))
            .chain(dutch_auction_bid_receive.map(
                |(receive_asset, receive_amount)| {
                    (receive_asset, Some(receive_amount))
                },
            ))
            .chain(dutch_auction_bid_spend.map(|(spend_asset, _)|
                    /* If the Truthcoin are not already accounted for,
                    * indicate an underflow */
                    (spend_asset, None)))
            .chain(dutch_auction_create_spend.map(|(spend_asset, _)|
                    /* If the Truthcoin are not already accounted for,
                    * indicate an underflow */
                    (spend_asset, None)))
            .chain(dutch_auction_collect0.map(
                |(receive_asset, receive_amount)| {
                    (receive_asset, Some(receive_amount))
                },
            ))
            .chain(dutch_auction_collect1.map(
                |(receive_asset, receive_amount)| {
                    (receive_asset, Some(receive_amount))
                },
            ))
            .chain(new_truthcoin_value.map(|(truthcoin, total_value)| {
                (AssetId::Truthcoin(truthcoin), Some(total_value))
            }))
            .chain(new_truthcoin_value.map(|(truthcoin, _)| {
                (AssetId::TruthcoinControl(truthcoin), Some(1))
            }))
    }

    /** Returns an iterator over total value for each Truthcoin that must
     *  appear in the outputs, in order.
     *  The total output value can possibly over/underflow in a transaction,
     *  so the total output values are [`Option<u64>`],
     *  where `None` indicates over/underflow. */
    fn output_truthcoin_total_values(
        &self,
    ) -> impl Iterator<Item = (TruthcoinId, Option<u64>)> + '_ {
        self.output_asset_total_values()
            .filter_map(|(asset_id, value)| match asset_id {
                AssetId::Truthcoin(truthcoin_id) => Some((truthcoin_id, value)),
                _ => None,
            })
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
        /* If this tx is an AMM burn, this is the corresponding Truthcoin IDs
        and token amount of the output corresponding to the newly created
        AMM LP position. */
        let mut amm_burn: Option<AmmBurn> = self.amm_burn();
        /* If this tx is an AMM mint, this is the corresponding Truthcoin IDs
        and token amount of the output corresponding to the newly created
        AMM LP position. */
        let mut amm_mint: Option<AmmMint> = self.amm_mint();
        self.unique_spent_lp_tokens()
            .into_iter()
            .map(move |(asset0, asset1, total_amount)| {
                let total_value = if let Some(AmmBurn {
                    asset0: burn_asset0,
                    asset1: burn_asset1,
                    amount0: _,
                    amount1: _,
                    lp_token_burn,
                }) = amm_burn
                    && (burn_asset0, burn_asset1) == (asset0, asset1)
                {
                    amm_burn = None;
                    total_amount.checked_sub(lp_token_burn)
                } else if let Some(AmmMint {
                    asset0: mint_asset0,
                    asset1: mint_asset1,
                    amount0: _,
                    amount1: _,
                    lp_token_mint,
                }) = amm_mint
                    && (mint_asset0, mint_asset1) == (asset0, asset1)
                {
                    amm_mint = None;
                    total_amount.checked_add(lp_token_mint)
                } else {
                    Some(total_amount)
                };
                (asset0, asset1, total_value)
            })
            .chain(amm_burn.map(|amm_burn| {
                /* If the LP tokens are not already accounted for,
                 * indicate an underflow */
                (amm_burn.asset0, amm_burn.asset1, None)
            }))
            .chain(amm_mint.map(|amm_mint| {
                (
                    amm_mint.asset0,
                    amm_mint.asset1,
                    Some(amm_mint.lp_token_mint),
                )
            }))
    }

    /// Compute the filled content for Truthcoin reservation outputs.
    fn filled_truthcoin_control_output_content(
        &self,
    ) -> impl Iterator<Item = FilledOutputContent> + '_ {
        self.output_asset_total_values()
            .filter_map(|(asset, _)| match asset {
                AssetId::TruthcoinControl(truthcoin_id) => {
                    Some(FilledOutputContent::TruthcoinControl(truthcoin_id))
                }
                _ => None,
            })
    }

    /// Compute the filled content for Dutch auction receipt outputs.
    // WARNING: do not expose DoubleEndedIterator.
    fn filled_dutch_auction_receipts(
        &self,
    ) -> impl Iterator<Item = FilledOutputContent> + '_ {
        /* If this tx is a Dutch auction creation, this is the content of the
         * output corresponding to the newly created Dutch auction receipt,
         * which is the last Dutch auction receipt output. */
        let new_dutch_auction_receipt_content =
            if self.is_dutch_auction_create() {
                let auction_id = DutchAuctionId(self.txid());
                Some(FilledOutputContent::DutchAuctionReceipt(auction_id))
            } else {
                None
            };
        let mut spent_dutch_auction_receipts =
            self.spent_dutch_auction_receipts();
        /* If this tx is a Dutch auction collect,
        the first auction receipt is burned */
        if self.is_dutch_auction_collect() {
            let _ = spent_dutch_auction_receipts.next();
        }
        spent_dutch_auction_receipts
            .map(|(_, filled_output)| filled_output.content())
            .cloned()
            .chain(new_dutch_auction_receipt_content)
    }

    /// compute the filled content for Truthcoin reservation outputs
    /// WARNING: do not expose DoubleEndedIterator.
    fn filled_reservation_output_content(
        &self,
    ) -> impl Iterator<Item = FilledOutputContent> + '_ {
        // If this tx is a Truthcoin reservation, this is the content of the
        // output corresponding to the newly created Truthcoin reservation,
        // which must be the final reservation output.
        let new_reservation_content: Option<FilledOutputContent> =
            self.reservation_commitment().map(|commitment| {
                FilledOutputContent::TruthcoinReservation(
                    self.txid(),
                    commitment,
                )
            });
        // used to track if the reservation that should be burned as part
        // of a registration tx
        let mut reservation_to_burn: Option<Hash> =
            self.implied_reservation_commitment();
        self.spent_reservations()
            .map(|(_, filled_output)| filled_output.content())
            // In the event of a registration, the first corresponding
            // reservation does not occur in the output
            .filter(move |content| {
                if let Some(implied_commitment) = reservation_to_burn {
                    if matches!(
                        content,
                        FilledOutputContent::TruthcoinReservation(_, commitment)
                            if *commitment == implied_commitment)
                    {
                        reservation_to_burn = None;
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            })
            .cloned()
            .chain(new_reservation_content)
    }

    /// compute the filled outputs.
    /// returns None if the outputs cannot be filled because the tx is invalid
    // FIXME: Invalidate tx if any iterator is incomplete
    pub fn filled_outputs(&self) -> Option<Vec<FilledOutput>> {
        let mut output_bitcoin_max_value = self.output_bitcoin_max_value()?;
        let mut output_truthcoin_total_values =
            self.output_truthcoin_total_values().peekable();
        let mut output_lp_token_total_amounts =
            self.output_lp_token_total_amounts().peekable();
        let mut filled_truthcoin_control_output_content =
            self.filled_truthcoin_control_output_content();
        let mut filled_dutch_auction_receipts =
            self.filled_dutch_auction_receipts();
        let mut filled_reservation_output_content =
            self.filled_reservation_output_content();
        self.outputs()
            .iter()
            .map(|output| {
                let content = match output.content.clone() {
                    OutputContent::AmmLpToken(amount) => {
                        let (asset0, asset1, remaining_amount) =
                            output_lp_token_total_amounts.peek_mut()?;
                        let remaining_amount = remaining_amount.as_mut()?;
                        let filled_content = FilledOutputContent::AmmLpToken {
                            asset0: *asset0,
                            asset1: *asset1,
                            amount,
                        };
                        match amount.cmp(remaining_amount) {
                            Ordering::Greater => {
                                // Invalid tx, return `None`
                                return None;
                            }
                            Ordering::Equal => {
                                // Advance the iterator to the next LP token
                                let _ = output_lp_token_total_amounts.next()?;
                            }
                            Ordering::Less => {
                                // Decrement the remaining value for the current LP token
                                *remaining_amount -= amount;
                            }
                        }
                        filled_content
                    }
                    OutputContent::Truthcoin(value) => {
                        let (truthcoin, remaining_value) =
                            output_truthcoin_total_values.peek_mut()?;
                        let remaining_value = remaining_value.as_mut()?;
                        let filled_content =
                            FilledOutputContent::Truthcoin(*truthcoin, value);
                        match value.cmp(remaining_value) {
                            Ordering::Greater => {
                                // Invalid tx, return `None`
                                return None;
                            }
                            Ordering::Equal => {
                                // Advance the iterator to the next Truthcoin
                                let _ = output_truthcoin_total_values.next()?;
                            }
                            Ordering::Less => {
                                // Decrement the remaining value for the current Truthcoin
                                *remaining_value -= value;
                            }
                        }
                        filled_content
                    }
                    OutputContent::TruthcoinControl => {
                        filled_truthcoin_control_output_content.next()?.clone()
                    }
                    OutputContent::TruthcoinReservation => {
                        filled_reservation_output_content.next()?.clone()
                    }
                    OutputContent::DutchAuctionReceipt => {
                        filled_dutch_auction_receipts.next()?
                    }
                    OutputContent::Bitcoin(value) => {
                        output_bitcoin_max_value =
                            output_bitcoin_max_value.checked_sub(value.0)?;
                        FilledOutputContent::Bitcoin(value)
                    }
                    OutputContent::Withdrawal(withdrawal) => {
                        FilledOutputContent::BitcoinWithdrawal(withdrawal)
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
        }
    }
}
