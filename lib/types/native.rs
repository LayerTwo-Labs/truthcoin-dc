//! Chain-local share movement and conditional delivery. No foreign-chain state.
use super::{Address, BlockHash};
use crate::state::markets::MarketId;
use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
pub type NativeId = [u8; 32];

/// Signed as part of the ordinary transaction. Consuming its inputs prevents replay.
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct NativeOperationV3 {
    pub genesis_hash: BlockHash,
    pub valid_from_parent: u32,
    pub valid_before_parent: u32,
    #[schema(value_type = Vec<u8>)]
    pub reference: NativeId,
    pub action: NativeActionV3,
}
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub enum NativeActionV3 {
    MoveShares {
        owner: Address,
        recipient: Address,
        market_id: MarketId,
        outcome_index: u32,
        shares: i64,
    },
    LockShares {
        owner: Address,
        claim_address: Address,
        refund_address: Address,
        market_id: MarketId,
        outcome_index: u32,
        shares: i64,
        #[schema(value_type = Vec<u8>)]
        hashlock: NativeId,
        claim_before_parent: u32,
        #[serde(default)]
        mutable_rights: bool,
    },
    ResolveEscrow {
        original_owner: Address,
        #[schema(value_type = Vec<u8>)]
        escrow_id: NativeId,
        resolution: EscrowResolutionV3,
    },
    AssignEscrow {
        original_owner: Address,
        #[schema(value_type = Vec<u8>)]
        escrow_id: NativeId,
        new_claim_address: Address,
        new_refund_address: Address,
    },
}
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub enum EscrowResolutionV3 {
    Claim {
        #[schema(value_type = Vec<u8>)]
        preimage: NativeId,
    },
    Refund,
}
#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Serialize,
    PartialEq,
    Eq,
    ToSchema,
)]
pub enum EscrowAssetV1 {
    Shares,
    /// The integer native cash successor after market settlement, including 0.
    NativeCash(u64),
}

#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Serialize,
    PartialEq,
    Eq,
    ToSchema,
)]
pub enum EscrowStatusV1 {
    Locked,
    Claimed {
        #[schema(value_type = Vec<u8>)]
        transaction_id: NativeId,
    },
    Refunded {
        #[schema(value_type = Vec<u8>)]
        transaction_id: NativeId,
    },
}

#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Serialize,
    PartialEq,
    Eq,
    ToSchema,
)]
pub struct ShareEscrowV1 {
    #[schema(value_type = Vec<u8>)]
    pub escrow_id: NativeId,
    pub owner: Address,
    pub claim_address: Address,
    pub refund_address: Address,
    pub market_id: MarketId,
    pub outcome_index: u32,
    pub shares: i64,
    #[schema(value_type = Vec<u8>)]
    pub hashlock: NativeId,
    pub claim_before_parent: u32,
    #[schema(value_type = Vec<u8>)]
    pub reference: NativeId,
    pub asset: EscrowAssetV1,
    pub status: EscrowStatusV1,
    /// Set only at creation; assignment cannot change this value.
    pub mutable_rights: bool,
}

pub fn escrow_id(genesis: BlockHash, transaction_id: NativeId) -> NativeId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"TRUTHCOIN_NATIVE_ESCROW_V3\0");
    hasher.update(&genesis.0);
    hasher.update(&transaction_id);
    *hasher.finalize().as_bytes()
}
