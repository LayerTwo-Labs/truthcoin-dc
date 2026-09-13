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

/// General assignment or exact subdivision of existing conditional rights.
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub enum EscrowMutationV1 {
    Split {
        #[schema(value_type = Vec<u8>)]
        escrow_id: NativeId,
        split_shares: i64,
        #[schema(value_type = Vec<u8>)]
        first_reference: NativeId,
        #[schema(value_type = Vec<u8>)]
        second_reference: NativeId,
    },
    Assign {
        #[schema(value_type = Vec<u8>)]
        escrow_id: NativeId,
        new_claim_address: Address,
        new_refund_address: Address,
        #[schema(value_type = Vec<u8>)]
        reference: NativeId,
    },
}

impl EscrowMutationV1 {
    pub fn escrow_id(&self) -> NativeId {
        match self {
            Self::Split { escrow_id, .. } | Self::Assign { escrow_id, .. } => {
                *escrow_id
            }
        }
    }
}

/// Both current rights holders sign these exact bytes. The current claim
/// signer's nonce is consumed only by successful execution.
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct EscrowMutationIntentV1 {
    pub genesis_hash: BlockHash,
    #[schema(value_type = Vec<u8>)]
    pub nonce: NativeId,
    pub mutation: EscrowMutationV1,
}

impl EscrowMutationIntentV1 {
    pub fn signing_bytes(&self) -> std::io::Result<Vec<u8>> {
        let mut bytes = b"TRUTHCOIN_ESCROW_MUTATION_V1\0".to_vec();
        bytes.push(super::THIS_SIDECHAIN);
        bytes.extend(borsh::to_vec(self)?);
        Ok(bytes)
    }
    pub fn verify(&self, authorization: &Authorization) -> bool {
        self.signing_bytes().is_ok_and(|bytes| {
            crate::authorization::verify(
                authorization.signature,
                &authorization.verifying_key,
                crate::authorization::Dst::NativeEscrowMutation,
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
    /// Anyone may relay a claim to the current jointly authorized destination.
    ClaimEscrow {
        #[schema(value_type = Vec<u8>)]
        escrow_id: NativeId,
        #[schema(value_type = Vec<u8>)]
        preimage: NativeId,
    },
    /// Anyone may relay a refund after expiry to the current signed destination.
    RefundEscrow {
        #[schema(value_type = Vec<u8>)]
        escrow_id: NativeId,
    },
    MutateEscrow {
        intent: EscrowMutationIntentV1,
        claim_authorization: Authorization,
        refund_authorization: Option<Authorization>,
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
            Self::MutateEscrow {
                intent,
                claim_authorization,
                ..
            } => Some((claim_authorization.get_address(), intent.nonce)),
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
    Split {
        #[schema(value_type = Vec<u8>)]
        transaction_id: NativeId,
        #[schema(value_type = Vec<u8>)]
        first_child_id: NativeId,
        #[schema(value_type = Vec<u8>)]
        second_child_id: NativeId,
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
    Assigned {
        previous_claim_address: Address,
        previous_refund_address: Address,
    },
    Split {
        first_child: ShareEscrowV1,
        second_child: ShareEscrowV1,
    },
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
    /// Historical post-operation escrow state, independent of later mutations.
    pub escrow_snapshot: Option<ShareEscrowV1>,
}

pub fn child_escrow_id(transaction_id: NativeId, child_index: u8) -> NativeId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"TRUTHCOIN_NATIVE_ESCROW_CHILD_V1\0");
    hasher.update(&transaction_id);
    hasher.update(&[child_index]);
    *hasher.finalize().as_bytes()
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
    fn mutation_signature_binds_chain_nonce_destinations_and_reference() {
        let key = ed25519_dalek::SigningKey::from_bytes(&[19; 32]);
        let intent = EscrowMutationIntentV1 {
            genesis_hash: [1; 32].into(),
            nonce: [2; 32],
            mutation: EscrowMutationV1::Assign {
                escrow_id: [3; 32],
                new_claim_address: Address([4; 20]),
                new_refund_address: Address([5; 20]),
                reference: [6; 32],
            },
        };
        let auth = Authorization {
            verifying_key: key.verifying_key().into(),
            signature: crate::authorization::sign(
                &key,
                crate::authorization::Dst::NativeEscrowMutation,
                &intent.signing_bytes().unwrap(),
            ),
        };
        assert!(intent.verify(&auth));
        let mut changed = intent.clone();
        changed.genesis_hash = [9; 32].into();
        assert!(!changed.verify(&auth));
        let mut changed = intent.clone();
        changed.nonce[0] ^= 1;
        assert!(!changed.verify(&auth));
        for field in 0..4 {
            let mut changed = intent.clone();
            if let EscrowMutationV1::Assign {
                escrow_id,
                new_claim_address,
                new_refund_address,
                reference,
            } = &mut changed.mutation
            {
                match field {
                    0 => escrow_id[0] ^= 1,
                    1 => *new_claim_address = Address([8; 20]),
                    2 => *new_refund_address = Address([8; 20]),
                    _ => reference[0] ^= 1,
                }
            }
            assert!(!changed.verify(&auth));
        }
        let wrong_domain = Authorization {
            verifying_key: key.verifying_key().into(),
            signature: crate::authorization::sign(
                &key,
                crate::authorization::Dst::NativeIntent,
                &intent.signing_bytes().unwrap(),
            ),
        };
        assert!(!intent.verify(&wrong_domain));
    }

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
