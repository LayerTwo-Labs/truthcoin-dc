use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use fallible_iterator::FallibleIterator as _;
use futures::{Stream, StreamExt};
use heed::types::SerdeBincode;
use sneed::{
    DatabaseUnique, DbError, EnvError, RoTxn, RwTxn, RwTxnError, UnitKey, db,
    env, rwtxn,
};
use tokio_stream::{StreamMap, wrappers::WatchStream};

use crate::{
    types::{
        Address, AuthorizedTransaction, InPoint, OutPoint, Output, Transaction,
        Txid, VERSION, Version,
    },
    util::Watchable,
};

#[allow(clippy::duplicated_attributes)]
#[derive(thiserror::Error, transitive::Transitive, Debug)]
#[transitive(from(db::error::Delete, DbError))]
#[transitive(from(db::error::Put, DbError))]
#[transitive(from(db::error::TryGet, DbError))]
#[transitive(from(env::error::CreateDb, EnvError))]
#[transitive(from(env::error::WriteTxn, EnvError))]
#[transitive(from(rwtxn::error::Commit, RwTxnError))]
pub enum Error {
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] EnvError),
    #[error("Database write error")]
    DbWrite(#[from] RwTxnError),
    #[error("Missing transaction {0}")]
    MissingTransaction(Txid),
    #[error("can't add transaction, utxo double spent")]
    UtxoDoubleSpent,
    #[error("can't add transaction, decision {0} already claimed in mempool")]
    DecisionAlreadyClaimedInMempool(String),
    #[error("mempool full: {current} transactions (max {max})")]
    MempoolFull { current: usize, max: usize },
}

#[derive(Clone)]
pub struct MemPool {
    pub transactions:
        DatabaseUnique<SerdeBincode<Txid>, SerdeBincode<AuthorizedTransaction>>,
    pub spent_utxos:
        DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<InPoint>>,
    /// Associates relevant txs to each address
    address_to_txs:
        DatabaseUnique<SerdeBincode<Address>, SerdeBincode<HashSet<Txid>>>,
    /// Tracks pending decision claims: decision_id_bytes -> claiming txid
    pending_decision_claims:
        DatabaseUnique<SerdeBincode<[u8; 3]>, SerdeBincode<Txid>>,
    trade_insertion_order:
        DatabaseUnique<SerdeBincode<u64>, SerdeBincode<Txid>>,
    trade_order_counter: DatabaseUnique<UnitKey, SerdeBincode<u64>>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

const MAX_MEMPOOL_TRANSACTIONS: usize = 5000;

impl MemPool {
    pub const NUM_DBS: u32 = 7;

    pub fn new(env: &sneed::Env) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn()?;
        let transactions =
            DatabaseUnique::create(env, &mut rwtxn, "transactions")?;
        let spent_utxos =
            DatabaseUnique::create(env, &mut rwtxn, "spent_utxos")?;
        let address_to_txs =
            DatabaseUnique::create(env, &mut rwtxn, "address_to_txs")?;
        let pending_decision_claims =
            DatabaseUnique::create(env, &mut rwtxn, "pending_decision_claims")?;
        let trade_insertion_order =
            DatabaseUnique::create(env, &mut rwtxn, "trade_insertion_order")?;
        let trade_order_counter =
            DatabaseUnique::create(env, &mut rwtxn, "trade_order_counter")?;
        let version =
            DatabaseUnique::create(env, &mut rwtxn, "mempool_version")?;
        if version.try_get(&rwtxn, &())?.is_none() {
            version.put(&mut rwtxn, &(), &*VERSION)?;
        }
        rwtxn.commit()?;
        Ok(Self {
            transactions,
            spent_utxos,
            address_to_txs,
            pending_decision_claims,
            trade_insertion_order,
            trade_order_counter,
            _version: version,
        })
    }

    /// Stores STXOs, checking for double spends
    fn put_stxos<Iter>(
        &self,
        rwtxn: &mut RwTxn,
        stxos: Iter,
    ) -> Result<(), Error>
    where
        Iter: IntoIterator<Item = (OutPoint, InPoint)>,
    {
        stxos.into_iter().try_for_each(|(outpoint, inpoint)| {
            if self.spent_utxos.try_get(rwtxn, &outpoint)?.is_some() {
                Err(Error::UtxoDoubleSpent)
            } else {
                self.spent_utxos.put(rwtxn, &outpoint, &inpoint)?;
                Ok(())
            }
        })
    }

    /// Delete STXOs
    fn delete_stxos<'a, Iter>(
        &self,
        rwtxn: &mut RwTxn,
        stxos: Iter,
    ) -> Result<(), Error>
    where
        Iter: IntoIterator<Item = &'a OutPoint>,
    {
        stxos.into_iter().try_for_each(|stxo| {
            self.spent_utxos.delete(rwtxn, stxo)?;
            Ok(())
        })
    }

    /// Associates the [`Txid`] with the [`Address`],
    /// by inserting into `address_to_txs`.
    fn assoc_txid_with_address(
        &self,
        rwtxn: &mut RwTxn,
        txid: Txid,
        address: &Address,
    ) -> Result<(), Error> {
        let mut associated_txs = self
            .address_to_txs
            .try_get(rwtxn, address)?
            .unwrap_or_default();
        associated_txs.insert(txid);
        self.address_to_txs.put(rwtxn, address, &associated_txs)?;
        Ok(())
    }

    /// Associates the [`Transaction`]'s [`Txid`] with all relevant
    /// [`Address`]es, by inserting into `address_to_txs`.
    fn index_tx_addresses(
        &self,
        rwtxn: &mut RwTxn,
        tx: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        let txid = tx.transaction.txid();
        tx.relevant_addresses().into_iter().try_for_each(|addr| {
            self.assoc_txid_with_address(rwtxn, txid, &addr)
        })
    }

    /// Unassociates the [`Txid`] with the [`Address`],
    /// by deleting from `address_to_txs`.
    fn unassoc_txid_with_address(
        &self,
        rwtxn: &mut RwTxn,
        txid: &Txid,
        address: &Address,
    ) -> Result<(), Error> {
        let Some(mut associated_txs) =
            self.address_to_txs.try_get(rwtxn, address)?
        else {
            return Ok(());
        };
        associated_txs.remove(txid);
        if !associated_txs.is_empty() {
            self.address_to_txs.put(rwtxn, address, &associated_txs)?;
        } else {
            self.address_to_txs.delete(rwtxn, address)?;
        }
        Ok(())
    }

    /// Unassociates the [`Transaction`]'s [`Txid`] with all relevant
    /// [`Address`]es, by deleting from `address_to_txs`.
    fn unindex_tx_addresses(
        &self,
        rwtxn: &mut RwTxn,
        tx: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        let txid = tx.transaction.txid();
        tx.relevant_addresses().into_iter().try_for_each(|addr| {
            self.unassoc_txid_with_address(rwtxn, &txid, &addr)
        })
    }

    fn is_trade_tx(transaction: &AuthorizedTransaction) -> bool {
        matches!(
            &transaction.transaction.data,
            Some(crate::types::TransactionData::Trade { .. })
        )
    }

    /// Extract decision IDs being claimed by this transaction
    fn get_claimed_decision_ids(transaction: &Transaction) -> Vec<[u8; 3]> {
        use crate::types::TransactionData;

        let mut decision_ids = Vec::new();
        if let Some(ref data) = transaction.data
            && let TransactionData::ClaimDecision { decisions, .. } = data
        {
            for entry in decisions {
                decision_ids.push(entry.decision_id_bytes);
            }
        }
        decision_ids
    }

    /// Check if any decisions are already claimed in mempool, and add them.
    ///
    /// # Atomicity
    /// This method uses a check-then-act pattern that relies on LMDB write
    /// transaction exclusivity - only one write transaction can be active at
    /// a time across all threads, preventing TOCTOU races between the check
    /// and add phases.
    fn put_decision_claims(
        &self,
        rwtxn: &mut RwTxn,
        txid: Txid,
        decision_ids: &[[u8; 3]],
    ) -> Result<(), Error> {
        // First check for conflicts
        for decision_id in decision_ids {
            if let Some(existing_txid) =
                self.pending_decision_claims.try_get(rwtxn, decision_id)?
                && existing_txid != txid
            {
                return Err(Error::DecisionAlreadyClaimedInMempool(
                    hex::encode(decision_id),
                ));
            }
        }
        // No conflicts, add all claims
        for decision_id in decision_ids {
            self.pending_decision_claims
                .put(rwtxn, decision_id, &txid)?;
        }
        Ok(())
    }

    /// Remove decision claims for a transaction
    fn delete_decision_claims(
        &self,
        rwtxn: &mut RwTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        let decision_ids =
            Self::get_claimed_decision_ids(&transaction.transaction);
        for decision_id in decision_ids {
            self.pending_decision_claims.delete(rwtxn, &decision_id)?;
        }
        Ok(())
    }

    fn tx_count(&self, rotxn: &RoTxn) -> Result<usize, Error> {
        let count = self
            .transactions
            .iter(rotxn)
            .map_err(DbError::from)?
            .count()
            .map_err(DbError::from)?;
        Ok(count)
    }

    pub fn put(
        &self,
        rwtxn: &mut RwTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        let txid = transaction.transaction.txid();
        tracing::debug!("adding transaction {txid} to mempool");

        let current_count = self.tx_count(rwtxn)?;
        if current_count >= MAX_MEMPOOL_TRANSACTIONS {
            return Err(Error::MempoolFull {
                current: current_count,
                max: MAX_MEMPOOL_TRANSACTIONS,
            });
        }

        // Check for duplicate decision claims first
        let claimed_decisions =
            Self::get_claimed_decision_ids(&transaction.transaction);
        if !claimed_decisions.is_empty() {
            self.put_decision_claims(rwtxn, txid, &claimed_decisions)?;
        }

        let stxos = {
            let txid = transaction.transaction.txid();
            transaction.transaction.inputs.iter().enumerate().map(
                move |(vin, outpoint)| {
                    (
                        *outpoint,
                        InPoint::Regular {
                            txid,
                            vin: vin as u32,
                        },
                    )
                },
            )
        };
        let () = self.put_stxos(rwtxn, stxos)?;
        self.transactions.put(rwtxn, &txid, transaction)?;
        let () = self.index_tx_addresses(rwtxn, transaction)?;

        if Self::is_trade_tx(transaction) {
            let counter =
                self.trade_order_counter.try_get(rwtxn, &())?.unwrap_or(0);
            let next = counter + 1;
            self.trade_insertion_order.put(rwtxn, &next, &txid)?;
            self.trade_order_counter.put(rwtxn, &(), &next)?;
        }

        Ok(())
    }

    pub fn delete(&self, rwtxn: &mut RwTxn, txid: Txid) -> Result<(), Error> {
        let mut pending_deletes = VecDeque::from([txid]);
        while let Some(txid) = pending_deletes.pop_front() {
            if let Some(tx) = self.transactions.try_get(rwtxn, &txid)? {
                let () = self.delete_stxos(rwtxn, &tx.transaction.inputs)?;
                let () = self.unindex_tx_addresses(rwtxn, &tx)?;
                let () = self.delete_decision_claims(rwtxn, &tx)?;

                if Self::is_trade_tx(&tx) {
                    self.delete_trade_order(rwtxn, &txid)?;
                }

                self.transactions.delete(rwtxn, &txid)?;
                for vout in 0..tx.transaction.outputs.len() {
                    let outpoint = OutPoint::Regular {
                        txid,
                        vout: vout as u32,
                    };
                    if let Some(InPoint::Regular {
                        txid: child_txid, ..
                    }) = self.spent_utxos.try_get(rwtxn, &outpoint)?
                    {
                        pending_deletes.push_back(child_txid);
                    }
                }
            }
        }
        Ok(())
    }

    /// Evict mempool transactions that conflict with `confirmed_tx` on a
    /// `decision_id`. Used after a block confirms a decision-claim to drop
    /// zombie competitors that lost the propagation race or a reorg.
    ///
    /// Returns the txids that were evicted (empty if `confirmed_tx` is not a
    /// `ClaimDecision` or no conflicts exist). Cascades through `delete()`,
    /// so any descendants of evicted zombies are also removed.
    pub fn evict_decision_claim_conflicts(
        &self,
        rwtxn: &mut RwTxn,
        confirmed_tx: &Transaction,
    ) -> Result<Vec<Txid>, Error> {
        let confirmed_txid = confirmed_tx.txid();
        let mut evicted = Vec::new();
        for decision_id in Self::get_claimed_decision_ids(confirmed_tx) {
            if let Some(zombie_txid) =
                self.pending_decision_claims.try_get(rwtxn, &decision_id)?
                && zombie_txid != confirmed_txid
            {
                tracing::info!(
                    decision_id = %hex::encode(decision_id),
                    %zombie_txid,
                    %confirmed_txid,
                    "evicting zombie decision-claim conflict"
                );
                self.delete(rwtxn, zombie_txid)?;
                evicted.push(zombie_txid);
            }
        }
        Ok(evicted)
    }

    fn delete_trade_order(
        &self,
        rwtxn: &mut RwTxn,
        txid: &Txid,
    ) -> Result<(), Error> {
        let mut iter = self
            .trade_insertion_order
            .iter(rwtxn)
            .map_err(DbError::from)?;
        let mut key_to_delete = None;
        while let Some((counter, stored_txid)) =
            iter.next().map_err(DbError::from)?
        {
            if stored_txid == *txid {
                key_to_delete = Some(counter);
                break;
            }
        }
        drop(iter);
        if let Some(key) = key_to_delete {
            self.trade_insertion_order.delete(rwtxn, &key)?;
        }
        Ok(())
    }

    pub fn take(
        &self,
        rotxn: &RoTxn,
        number: usize,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        self.transactions
            .iter(rotxn)
            .map_err(DbError::from)?
            .take(number)
            .map(|(_, transaction)| Ok(transaction))
            .collect()
            .map_err(|err| DbError::from(err).into())
    }

    pub fn take_all(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        self.transactions
            .iter(rotxn)
            .map_err(DbError::from)?
            .map(|(_, transaction)| Ok(transaction))
            .collect()
            .map_err(|err| DbError::from(err).into())
    }

    pub fn take_trades_ordered(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        let mut trades = Vec::new();
        let mut iter = self
            .trade_insertion_order
            .iter(rotxn)
            .map_err(DbError::from)?;
        while let Some((_counter, txid)) = iter.next().map_err(DbError::from)? {
            if let Some(tx) = self.transactions.try_get(rotxn, &txid)? {
                trades.push(tx);
            }
        }
        Ok(trades)
    }

    /// Get [`Txid`]s relevant to a particular address
    fn get_txids_relevant_to_address(
        &self,
        rotxn: &RoTxn,
        addr: &Address,
    ) -> Result<HashSet<Txid>, Error> {
        let res = self
            .address_to_txs
            .try_get(rotxn, addr)?
            .unwrap_or_default();
        Ok(res)
    }

    /// Get [`Transaction`]s relevant to a particular address
    fn get_txs_relevant_to_address(
        &self,
        rotxn: &RoTxn,
        addr: &Address,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        self.get_txids_relevant_to_address(rotxn, addr)?
            .into_iter()
            .map(|txid| {
                self.transactions
                    .try_get(rotxn, &txid)?
                    .ok_or(Error::MissingTransaction(txid))
            })
            .collect()
    }

    /// Get unconfirmed UTXOs relevant to a particular address
    pub fn get_unconfirmed_utxos(
        &self,
        rotxn: &RoTxn,
        addr: &Address,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let relevant_txs = self.get_txs_relevant_to_address(rotxn, addr)?;
        let res = relevant_txs
            .into_iter()
            .flat_map(|tx| {
                let txid = tx.transaction.txid();
                tx.transaction.outputs.into_iter().enumerate().filter_map(
                    move |(vout, output)| {
                        if output.address == *addr {
                            Some((
                                OutPoint::Regular {
                                    txid,
                                    vout: vout as u32,
                                },
                                output,
                            ))
                        } else {
                            None
                        }
                    },
                )
            })
            .collect();
        Ok(res)
    }

    pub fn pending_decision_claim_ids(
        &self,
        rotxn: &RoTxn,
    ) -> Result<BTreeSet<[u8; 3]>, Error> {
        self.pending_decision_claims
            .iter(rotxn)
            .map_err(DbError::from)?
            .map(|(decision_id, _txid)| Ok(decision_id))
            .collect()
            .map_err(DbError::from)
            .map_err(Error::from)
    }
}

impl Watchable<()> for MemPool {
    type WatchStream = impl Stream<Item = ()>;

    /// Get a signal that notifies whenever the mempool changes
    fn watch(&self) -> Self::WatchStream {
        let watchables = [
            self.transactions.watch().clone(),
            self.spent_utxos.watch().clone(),
            self.pending_decision_claims.watch().clone(),
        ];
        let streams = StreamMap::from_iter(
            watchables.into_iter().map(WatchStream::new).enumerate(),
        );
        let streams_len = streams.len();
        streams.ready_chunks(streams_len).map(|signals| {
            assert_ne!(signals.len(), 0);
            #[allow(clippy::unused_unit)]
            ()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::decisions::DecisionType;
    use crate::types::{
        Authorized, DecisionClaimEntry, TransactionData, hashes::Hash,
    };
    use sneed::Env;

    fn make_env() -> (Env, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let env_path = dir.path().join("data.mdb");
        std::fs::create_dir_all(&env_path).unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(64 * 1024 * 1024).max_dbs(MemPool::NUM_DBS);
        let env = unsafe { Env::open(&opts, &env_path) }.unwrap();
        (env, dir)
    }

    fn input_outpoint(seed: u8) -> OutPoint {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        OutPoint::Regular {
            txid: Txid(Hash::from(bytes)),
            vout: 0,
        }
    }

    fn claim_entry(decision_id: [u8; 3]) -> DecisionClaimEntry {
        DecisionClaimEntry {
            decision_id_bytes: decision_id,
            header: "h".to_string(),
            description: String::new(),
            option_0_label: None,
            option_1_label: None,
            option_labels: None,
            tags: None,
        }
    }

    fn claim_tx(
        input_seed: u8,
        decision_ids: &[[u8; 3]],
    ) -> AuthorizedTransaction {
        let entries: Vec<DecisionClaimEntry> =
            decision_ids.iter().copied().map(claim_entry).collect();
        let tx = Transaction {
            inputs: vec![input_outpoint(input_seed)],
            outputs: vec![],
            memo: Vec::new(),
            data: Some(TransactionData::ClaimDecision {
                decision_type: DecisionType::Binary,
                decisions: entries,
            }),
        };
        Authorized {
            transaction: tx,
            authorizations: vec![],
            actor_proof: None,
        }
    }

    fn regular_tx(input_seed: u8) -> AuthorizedTransaction {
        let tx = Transaction {
            inputs: vec![input_outpoint(input_seed)],
            outputs: vec![],
            memo: Vec::new(),
            data: None,
        };
        Authorized {
            transaction: tx,
            authorizations: vec![],
            actor_proof: None,
        }
    }

    #[test]
    fn evict_basic_conflict() {
        let (env, _dir) = make_env();
        let mempool = MemPool::new(&env).unwrap();
        let did: [u8; 3] = [0x42, 0, 0];

        let tx_a = claim_tx(1, &[did]);
        let tx_b = claim_tx(2, &[did]);
        let txid_a = tx_a.transaction.txid();
        let txid_b = tx_b.transaction.txid();

        let mut rwtxn = env.write_txn().unwrap();
        mempool.put(&mut rwtxn, &tx_b).unwrap();
        assert_eq!(
            mempool
                .pending_decision_claims
                .try_get(&rwtxn, &did)
                .unwrap(),
            Some(txid_b),
        );

        let evicted = mempool
            .evict_decision_claim_conflicts(&mut rwtxn, &tx_a.transaction)
            .unwrap();

        assert_eq!(evicted, vec![txid_b]);
        assert!(
            mempool
                .pending_decision_claims
                .try_get(&rwtxn, &did)
                .unwrap()
                .is_none()
        );
        assert!(
            mempool
                .transactions
                .try_get(&rwtxn, &txid_b)
                .unwrap()
                .is_none()
        );
        assert!(
            mempool
                .transactions
                .try_get(&rwtxn, &txid_a)
                .unwrap()
                .is_none(),
            "tx_a was never inserted; helper should not insert it"
        );
        rwtxn.commit().unwrap();
    }

    #[test]
    fn evict_noop_when_in_block_tx_equals_pending_claim() {
        let (env, _dir) = make_env();
        let mempool = MemPool::new(&env).unwrap();
        let did: [u8; 3] = [1, 2, 3];

        let tx_a = claim_tx(7, &[did]);
        let txid_a = tx_a.transaction.txid();

        let mut rwtxn = env.write_txn().unwrap();
        mempool.put(&mut rwtxn, &tx_a).unwrap();

        let evicted = mempool
            .evict_decision_claim_conflicts(&mut rwtxn, &tx_a.transaction)
            .unwrap();

        assert!(evicted.is_empty());
        assert_eq!(
            mempool
                .pending_decision_claims
                .try_get(&rwtxn, &did)
                .unwrap(),
            Some(txid_a),
        );
        assert!(
            mempool
                .transactions
                .try_get(&rwtxn, &txid_a)
                .unwrap()
                .is_some()
        );
        rwtxn.commit().unwrap();
    }

    #[test]
    fn evict_noop_when_tx_is_not_claim_decision() {
        let (env, _dir) = make_env();
        let mempool = MemPool::new(&env).unwrap();
        let did: [u8; 3] = [9, 9, 9];

        let zombie = claim_tx(3, &[did]);
        let txid_zombie = zombie.transaction.txid();
        let confirmed_regular = regular_tx(4);

        let mut rwtxn = env.write_txn().unwrap();
        mempool.put(&mut rwtxn, &zombie).unwrap();

        let evicted = mempool
            .evict_decision_claim_conflicts(
                &mut rwtxn,
                &confirmed_regular.transaction,
            )
            .unwrap();

        assert!(evicted.is_empty());
        assert_eq!(
            mempool
                .pending_decision_claims
                .try_get(&rwtxn, &did)
                .unwrap(),
            Some(txid_zombie),
            "regular tx should not touch decision-claim state"
        );
        rwtxn.commit().unwrap();
    }

    #[test]
    fn evict_multi_decision_partial_overlap_clears_all_zombie_rows() {
        let (env, _dir) = make_env();
        let mempool = MemPool::new(&env).unwrap();
        let x: [u8; 3] = [1, 1, 1];
        let y: [u8; 3] = [2, 2, 2];
        let z: [u8; 3] = [3, 3, 3];

        let zombie = claim_tx(5, &[x, z]);
        let txid_zombie = zombie.transaction.txid();
        let confirmed = claim_tx(6, &[x, y]);

        let mut rwtxn = env.write_txn().unwrap();
        mempool.put(&mut rwtxn, &zombie).unwrap();

        let evicted = mempool
            .evict_decision_claim_conflicts(&mut rwtxn, &confirmed.transaction)
            .unwrap();

        assert_eq!(evicted, vec![txid_zombie]);
        assert!(
            mempool
                .pending_decision_claims
                .try_get(&rwtxn, &x)
                .unwrap()
                .is_none()
        );
        assert!(
            mempool
                .pending_decision_claims
                .try_get(&rwtxn, &z)
                .unwrap()
                .is_none(),
            "cascade through delete() must clear all of zombie's claim rows, \
             not just the directly-conflicting one"
        );
        rwtxn.commit().unwrap();
    }

    // Cascade-to-children behavior is intentionally not tested here: it lives
    // entirely in `MemPool::delete()` (already in production) and the helper
    // simply forwards to it. Building the parent/child UTXO graph requires
    // assembling typed Outputs and is covered indirectly by the existing
    // `delete` test surface in higher-level integration coverage.
}
