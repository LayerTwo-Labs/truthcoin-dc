use std::collections::HashMap;

use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    authorization::Authorization,
    block::coinbase::Coinbase,
    error,
    hashes::{self, CoinbaseTxid, Hash, MerkleRoot},
    transaction::{
        AuthorizedTransaction, FilledTransaction, GetValue, OutPoint, Output,
        Transaction,
    },
};

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Body {
    pub coinbase: Coinbase,
    pub transactions: Vec<Transaction>,
    pub authorizations: Vec<Authorization>,
    /// One optional actor proof per transaction
    #[serde(default)]
    pub actor_proofs: Vec<Option<Authorization>>,
}

impl Body {
    /// Size limit in bytes
    pub const MAX_SIZE: usize = 8 * 1024 * 1024;

    pub fn new(
        authorized_transactions: Vec<AuthorizedTransaction>,
        coinbase: Coinbase,
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

    pub fn authorized_transactions(&self) -> Vec<AuthorizedTransaction> {
        let mut authorizations_iter = self.authorizations.iter();
        let mut actor_proofs_iter = self.actor_proofs.iter();
        self.transactions
            .iter()
            .map(|tx| {
                let mut authorizations = Vec::with_capacity(tx.inputs.len());
                for _ in 0..tx.inputs.len() {
                    let auth = authorizations_iter.next().unwrap();
                    authorizations.push(auth.clone());
                }
                let actor_proof = actor_proofs_iter.next().cloned().flatten();
                AuthorizedTransaction {
                    transaction: tx.clone(),
                    authorizations,
                    actor_proof,
                }
            })
            .collect()
    }

    pub fn compute_merkle_root(
        coinbase: &Coinbase,
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
            .flat_map(|tx| tx.inputs.iter().map(|(outpoint, _)| outpoint))
            .copied()
            .collect()
    }

    pub fn get_outputs(
        coinbase_txid: CoinbaseTxid,
        coinbase: &Coinbase,
        txs: &[FilledTransaction],
    ) -> HashMap<OutPoint, Output> {
        let mut res = HashMap::new();
        for (vout, output) in coinbase.outputs.iter().enumerate() {
            let vout = vout as u32;
            let outpoint = OutPoint::Coinbase {
                txid: coinbase_txid,
                vout,
            };
            res.insert(outpoint, output.clone());
        }
        for tx in txs {
            let txid = tx.transaction.txid();
            for (vout, output) in tx.transaction.outputs.iter().enumerate() {
                let vout = vout as u32;
                let outpoint = OutPoint::Regular { txid, vout };
                res.insert(outpoint, output.clone());
            }
        }
        res
    }

    pub fn get_coinbase_value(
        &self,
    ) -> Result<bitcoin::Amount, error::AmountOverflow> {
        use bitcoin::amount::CheckedSum as _;
        self.coinbase
            .outputs
            .iter()
            .map(|output| output.get_value())
            .checked_sum()
            .ok_or(error::AmountOverflow)
    }

    /// Calculate total number of inputs across all transactions in a block body
    pub fn inputs_len(&self) -> usize {
        self.transactions.iter().map(|t| t.inputs.len()).sum()
    }
}
