//! Versioned executed-market commitments for the isolated oracle test chain.

use borsh::BorshSerialize;
use merkle_cbt::{CBMT, MerkleProof, merkle_tree::Merge};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{BlockHash, Hash, hashes, market::MarketId};

pub const MAGIC: [u8; 8] = *b"TCORCL01";
pub const PRICE_SCALE: u64 = 1_000_000_000_000;
// CBMT uses u32 heap indices (2 * leaf_count - 2).
pub const MAX_MARKETS: usize = 1 << 30;

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketPrice {
    pub market_id: MarketId,
    #[schema(value_type = Vec<u8>)]
    pub terms_hash: Hash,
    /// Native MarketState discriminant: 1 trading, 3 cancelled, 4 invalid, 5 settled.
    pub status: u8,
    pub updated_height: u32,
    pub liquidity_base_sats: u64,
    /// Native get_valid_state_combos (tradeable outcome) order. Scale is 10^12.
    /// Only status 5 denotes final payouts; other prices are indicative LMSR marks.
    pub prices: Vec<u64>,
}

impl MarketPrice {
    pub fn hash(&self) -> Hash {
        hashes::hash(&(*b"TCORACLE/leaf/v1", self))
    }
}

pub struct Branch;
impl Merge for Branch {
    type Item = Hash;
    fn merge(left: &Hash, right: &Hash) -> Hash {
        hashes::hash(&(*b"TCORACLE/node/v1", left, right))
    }
}
pub type Tree = CBMT<Hash, Branch>;

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Commitment {
    pub previous_height: Option<u32>,
    pub previous_hash: Option<BlockHash>,
    pub market_count: u32,
    #[schema(value_type = Vec<u8>)]
    pub root: Hash,
}

impl Commitment {
    pub fn memo(&self) -> Vec<u8> {
        // All fields have infallible fixed-width Borsh encodings.
        borsh::to_vec(&(MAGIC, self)).expect("fixed oracle commitment")
    }
}

/// Current executed state, to be committed by the *next* block. This is not
/// authenticated until its memo is proven under a validated oracletest header.
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct PriceProof {
    pub commitment: Commitment,
    pub market: MarketPrice,
    pub index: u32,
    #[schema(value_type = Vec<Vec<u8>>)]
    pub siblings: Vec<Hash>,
}

impl PriceProof {
    pub fn verify(&self) -> bool {
        let count = self.commitment.market_count;
        if count == 0
            || count as usize > MAX_MARKETS
            || self.index < count - 1
            || self.index > 2 * count - 2
            || self.siblings.len() > 31
        {
            return false;
        }
        MerkleProof::<Hash, Branch>::new(
            vec![self.index],
            self.siblings.clone(),
        )
        .verify(&self.commitment.root, &[self.market.hash()])
    }
}
