//! General, chain-local share delivery and hash/time escrow operations.
//!
//! These types deliberately contain no foreign-chain identifiers or verifiers.
//! All encodings are versioned; append variants instead of reordering them.
use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::{Address, Authorization, BlockHash, GetAddress};
use crate::state::markets::MarketId;

pub type NativeId = [u8; 32];

/// A recipient's fill-once permission for a sponsor to buy exact shares.
/// The sponsor pays the actual LMSR price and receives all unused cash.
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct BuyIntentV1 {
    pub genesis_hash: BlockHash,
    pub recipient: Address,
    pub market_id: MarketId,
    pub outcome_index: u32,
    pub shares: i64,
    #[schema(value_type = Vec<u8>)]
    pub nonce: NativeId,
    #[schema(value_type = Vec<u8>)]
    pub reference: NativeId,
    pub valid_from_parent: u32,
    pub valid_before_parent: u32,
}

impl BuyIntentV1 {
    pub fn signing_bytes(&self) -> std::io::Result<Vec<u8>> {
        let mut bytes = b"TRUTHCOIN_BUY_INTENT_V1\0".to_vec();
        bytes.push(super::THIS_SIDECHAIN);
        bytes.extend(borsh::to_vec(self)?);
        Ok(bytes)
    }

    pub fn verify(&self, authorization: &Authorization) -> bool {
        if authorization.get_address() != self.recipient {
            return false;
        }
        self.signing_bytes().is_ok_and(|bytes| {
            crate::authorization::verify(
                authorization.signature,
                &authorization.verifying_key,
                crate::authorization::Dst::NativeIntent,
                &bytes,
            )
        })
    }
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub enum NativeOperationV1 {
    BuyForIntent {
        intent: BuyIntentV1,
        authorization: Authorization,
        change_address: Address,
        limit_sats: u64,
        tx_pow_nonce: Option<u64>,
        prev_block_hash: BlockHash,
    },
    TransferShares {
        owner: Address,
        recipient: Address,
        market_id: MarketId,
        outcome_index: u32,
        shares: i64,
        #[schema(value_type = Vec<u8>)]
        nonce: NativeId,
        #[schema(value_type = Vec<u8>)]
        reference: NativeId,
        valid_from_parent: u32,
        valid_before_parent: u32,
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
        #[schema(value_type = Vec<u8>)]
        nonce: NativeId,
        #[schema(value_type = Vec<u8>)]
        reference: NativeId,
    },
    /// Anyone may relay a claim; the destination was fixed by LockShares.
    ClaimEscrow {
        #[schema(value_type = Vec<u8>)]
        escrow_id: NativeId,
        #[schema(value_type = Vec<u8>)]
        preimage: NativeId,
    },
    /// Anyone may relay a refund after expiry; its destination is immutable.
    RefundEscrow {
        #[schema(value_type = Vec<u8>)]
        escrow_id: NativeId,
    },
}

impl NativeOperationV1 {
    pub fn actor(&self) -> Option<Address> {
        match self {
            Self::TransferShares { owner, .. }
            | Self::LockShares { owner, .. } => Some(*owner),
            _ => None,
        }
    }

    pub fn nonce(&self) -> Option<(Address, NativeId)> {
        match self {
            Self::BuyForIntent { intent, .. } => {
                Some((intent.recipient, intent.nonce))
            }
            Self::TransferShares { owner, nonce, .. }
            | Self::LockShares { owner, nonce, .. } => Some((*owner, *nonce)),
            _ => None,
        }
    }
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
}

#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Debug,
    Deserialize,
    Serialize,
    PartialEq,
    Eq,
    ToSchema,
)]
pub enum NativeEffectKindV1 {
    Bought,
    Transferred,
    Locked,
    Claimed,
    Refunded,
}

/// An executed effect, never an inclusion-only or soft-skipped receipt.
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
pub struct NativeEffectV1 {
    #[schema(value_type = Vec<u8>)]
    pub transaction_id: NativeId,
    pub sidechain_height: u32,
    pub parent_height: u32,
    #[schema(value_type = Vec<u8>)]
    pub reference: NativeId,
    pub kind: NativeEffectKindV1,
    #[schema(value_type = Option<Vec<u8>>)]
    pub escrow_id: Option<NativeId>,
    pub owner: Address,
    pub recipient: Address,
    pub market_id: MarketId,
    pub outcome_index: u32,
    pub shares: i64,
    pub asset: EscrowAssetV1,
}

pub fn escrow_id(transaction_id: NativeId) -> NativeId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"TRUTHCOIN_NATIVE_ESCROW_V1\0");
    hasher.update(&transaction_id);
    *hasher.finalize().as_bytes()
}

/// Local bidder preview. This is not proof of inclusion or execution.
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct NativeBmmCandidateV1 {
    pub header: super::Header,
    pub body: super::Body,
    pub parent_height: u32,
    pub preview_effects: Vec<NativeEffectV1>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intent_signature_binds_genesis_recipient_quantity_and_reference() {
        let key = ed25519_dalek::SigningKey::from_bytes(&[3; 32]);
        let vk: crate::types::VerifyingKey = key.verifying_key().into();
        let intent = BuyIntentV1 {
            genesis_hash: [1; 32].into(),
            recipient: crate::authorization::get_address(&vk),
            market_id: MarketId::new([2; 6]),
            outcome_index: 1,
            shares: 100,
            nonce: [3; 32],
            reference: [4; 32],
            valid_from_parent: 10,
            valid_before_parent: 20,
        };
        let auth = Authorization {
            verifying_key: vk,
            signature: crate::authorization::sign(
                &key,
                crate::authorization::Dst::NativeIntent,
                &intent.signing_bytes().unwrap(),
            ),
        };
        assert!(intent.verify(&auth));
        let mut changed = intent.clone();
        changed.genesis_hash = [2; 32].into();
        assert!(!changed.verify(&auth));
        let mut changed = intent.clone();
        changed.recipient = Address([0; 20]);
        assert!(!changed.verify(&auth));
        let mut changed = intent.clone();
        changed.shares += 1;
        assert!(!changed.verify(&auth));
        let mut changed = intent.clone();
        changed.reference[0] ^= 1;
        assert!(!changed.verify(&auth));
        let mut changed = intent.clone();
        changed.valid_before_parent += 1;
        assert!(!changed.verify(&auth));
        let wrong_domain = Authorization {
            verifying_key: vk,
            signature: crate::authorization::sign(
                &key,
                crate::authorization::Dst::Transaction,
                &intent.signing_bytes().unwrap(),
            ),
        };
        assert!(!intent.verify(&wrong_domain));
    }
}
