use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::hashes::{self, BlockHash, MerkleRoot};

pub mod body;
pub use body::Body;

#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct Header {
    pub merkle_root: MerkleRoot,
    pub prev_side_hash: Option<BlockHash>,
    #[borsh(serialize_with = "crate::borsh_serialize_bitcoin_block_hash")]
    #[schema(value_type = crate::schema::BitcoinBlockHash)]
    pub prev_main_hash: bitcoin::BlockHash,
}

impl Header {
    pub fn hash(&self) -> BlockHash {
        hashes::hash_with_scratch_buffer(self).into()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Block {
    pub header: Header,
    pub body: Body,
    pub height: u32,
}

#[cfg(test)]
mod block_json_tests {
    use bitcoin::hashes::Hash as _;

    use super::{Block, Body, Header};
    use crate::MerkleRoot;

    #[test]
    fn block_json_nests_the_header_and_the_body() {
        let block = Block {
            header: Header {
                merkle_root: MerkleRoot::from([1; 32]),
                prev_side_hash: None,
                prev_main_hash: bitcoin::BlockHash::from_byte_array([2; 32]),
            },
            body: Body {
                coinbase: Vec::new(),
                transactions: Vec::new(),
                authorizations: Vec::new(),
                actor_proofs: Vec::new(),
            },
            height: 7,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(
            json["header"]["prev_main_hash"],
            serde_json::to_value(block.header.prev_main_hash).unwrap()
        );
        assert!(json["body"]["transactions"].is_array());
        assert_eq!(json["height"], 7);
        assert!(json.get("prev_main_hash").is_none());
        let decoded: Block = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(&decoded).unwrap(), json);
    }
}
