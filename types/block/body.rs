use std::collections::HashMap;

use bitcoin::amount::CheckedSum as _;
use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    AmountOverflowError, Authorization, AuthorizedTransaction,
    GetBitcoinValue as _, MalformedBodyError, OutPoint, Output, Transaction,
    hashes::{self, Hash, MerkleRoot},
};

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Body {
    pub coinbase: Vec<Output>,
    pub transactions: Vec<Transaction>,
    pub authorizations: Vec<Authorization>,
    #[serde(default)]
    pub actor_proofs: Vec<Option<Authorization>>,
}

impl Body {
    /// Size limit in bytes
    pub const MAX_SIZE: usize = 8 * 1024 * 1024;

    pub fn new(
        authorized_transactions: Vec<AuthorizedTransaction>,
        coinbase: Vec<Output>,
    ) -> Self {
        let mut authorizations = Vec::with_capacity(
            authorized_transactions
                .iter()
                .map(|t| t.transaction.inputs.len())
                .sum(),
        );
        let mut transactions =
            Vec::with_capacity(authorized_transactions.len());
        let mut actor_proofs =
            Vec::with_capacity(authorized_transactions.len());
        for at in authorized_transactions.into_iter() {
            authorizations.extend(at.authorizations);
            actor_proofs.push(at.actor_proof);
            transactions.push(at.transaction);
        }
        Self {
            coinbase,
            transactions,
            authorizations,
            actor_proofs,
        }
    }

    pub fn authorized_transactions(
        &self,
    ) -> Result<Vec<AuthorizedTransaction>, MalformedBodyError> {
        let mut authorizations_iter = self.authorizations.iter();
        let mut actor_proofs_iter = self.actor_proofs.iter();
        self.transactions
            .iter()
            .map(|tx| {
                let mut authorizations = Vec::with_capacity(tx.inputs.len());
                for _ in 0..tx.inputs.len() {
                    let auth =
                        authorizations_iter.next().ok_or(MalformedBodyError)?;
                    authorizations.push(auth.clone());
                }
                let actor_proof = actor_proofs_iter.next().cloned().flatten();
                Ok(AuthorizedTransaction {
                    transaction: tx.clone(),
                    authorizations,
                    actor_proof,
                })
            })
            .collect()
    }

    pub fn compute_merkle_root(
        coinbase: &[Output],
        txs: &[Transaction],
    ) -> MerkleRoot {
        let coinbase_hash: Hash = hashes::hash_with_scratch_buffer(coinbase);
        let mut leaves: Vec<Hash> = std::iter::once(coinbase_hash)
            .chain(txs.iter().map(|tx| tx.txid().into()))
            .collect();
        while leaves.len() > 1 {
            let mut next_level = Vec::with_capacity(leaves.len().div_ceil(2));
            for pair in leaves.chunks(2) {
                let left = pair[0].as_ref();
                let right = if pair.len() == 2 {
                    pair[1].as_ref()
                } else {
                    pair[0].as_ref()
                };
                let mut combined = [0u8; 64];
                combined[..32].copy_from_slice(left);
                combined[32..].copy_from_slice(right);
                next_level.push(*blake3::hash(&combined).as_bytes());
            }
            leaves = next_level;
        }
        leaves[0].into()
    }

    pub fn get_inputs(&self) -> Vec<OutPoint> {
        self.transactions
            .iter()
            .flat_map(|tx| tx.inputs.iter())
            .copied()
            .collect()
    }

    pub fn get_outputs(&self) -> HashMap<OutPoint, Output> {
        let mut outputs = HashMap::new();
        let merkle_root =
            Body::compute_merkle_root(&self.coinbase, &self.transactions);
        for (vout, output) in self.coinbase.iter().enumerate() {
            let vout = vout as u32;
            let outpoint = OutPoint::Coinbase { merkle_root, vout };
            outputs.insert(outpoint, output.clone());
        }
        for transaction in &self.transactions {
            let txid = transaction.txid();
            for (vout, output) in transaction.outputs.iter().enumerate() {
                let vout = vout as u32;
                let outpoint = OutPoint::Regular { txid, vout };
                outputs.insert(outpoint, output.clone());
            }
        }
        outputs
    }

    pub fn get_coinbase_value(
        &self,
    ) -> Result<bitcoin::Amount, AmountOverflowError> {
        self.coinbase
            .iter()
            .map(|output| output.get_bitcoin_value())
            .checked_sum()
            .ok_or(AmountOverflowError)
    }
}
