use std::io::Cursor;

use borsh::{BorshDeserialize, BorshSerialize};
#[cfg(feature = "heed")]
use heed::{BoxedError, BytesDecode, BytesEncode};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::hashes::{MerkleRoot, Txid};

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
    #[schema(value_type = crate::schema::BitcoinOutPoint)]
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
                    const_hex::encode(market_id),
                    block_height
                )
            }
            Self::Payout { hash, vout } => {
                write!(f, "payout {hash} {vout}")
            }
        }
    }
}

pub(crate) const OUTPOINT_KEY_SIZE: usize = 37;

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

#[cfg(feature = "heed")]
impl<'a> BytesEncode<'a> for OutPointKey {
    type EItem = OutPointKey;

    #[inline]
    fn bytes_encode(
        item: &'a Self::EItem,
    ) -> Result<std::borrow::Cow<'a, [u8]>, BoxedError> {
        Ok(std::borrow::Cow::Borrowed(item.as_ref()))
    }
}

#[cfg(feature = "heed")]
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
