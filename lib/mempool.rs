use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    path::PathBuf,
};

use fallible_iterator::FallibleIterator as _;
use futures::{Stream, StreamExt as _};
use heed::types::SerdeBincode;
use sneed::{
    DatabaseUnique, DbError, EnvError, RoTxn, RwTxn, RwTxnError, UnitKey, db,
};
use tokio_stream::{StreamMap, wrappers::WatchStream};

use crate::{
    types::{
        Accumulator, Address, AuthorizedTransaction, OutPoint, Output,
        Transaction, Txid, UtreexoError, VERSION, Version,
    },
    util::Watchable,
};

/// Longest chain of unconfirmed transactions the mempool accepts. Bitcoin
/// Core holds the same number in `DEFAULT_ANCESTOR_LIMIT`.
pub const MAX_UNCONFIRMED_ANCESTORS: usize = 25;

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, thiserror::Error, transitive::Transitive)]
#[transitive(from(db::error::Delete, DbError))]
#[transitive(from(db::error::Put, DbError))]
#[transitive(from(db::error::TryGet, DbError))]
pub enum Error {
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] EnvError),
    #[error("Database write error")]
    DbWrite(#[from] RwTxnError),
    #[error(
        "Incompatible DB version ({}). Please clear the DB (`{}`) and re-sync",
        .version,
        .db_path.display()
    )]
    IncompatibleVersion { version: Version, db_path: PathBuf },
    #[error("can't add transaction, decision {0} already claimed in mempool")]
    DecisionAlreadyClaimedInMempool(String),
    #[error("Missing transaction {0}")]
    MissingTransaction(Txid),
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
    #[error("can't add transaction, utxo double spent")]
    UtxoDoubleSpent,
    #[error(
        "can't add transaction, it has {count} unconfirmed ancestors and the \
         limit is {MAX_UNCONFIRMED_ANCESTORS}"
    )]
    TooManyAncestors { count: usize },
}

#[derive(Clone)]
pub struct MemPool {
    pub transactions:
        DatabaseUnique<SerdeBincode<Txid>, SerdeBincode<AuthorizedTransaction>>,
    pub spent_utxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<Txid>>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
    /// Associates relevant txs to each address
    address_to_txs:
        DatabaseUnique<SerdeBincode<Address>, SerdeBincode<HashSet<Txid>>>,
    /// Pending decision claims: decision id bytes to the claiming txid
    pending_decision_claims:
        DatabaseUnique<SerdeBincode<[u8; 3]>, SerdeBincode<Txid>>,
    trade_insertion_order:
        DatabaseUnique<SerdeBincode<u64>, SerdeBincode<Txid>>,
    trade_order_counter: DatabaseUnique<UnitKey, SerdeBincode<u64>>,
}

impl MemPool {
    pub const NUM_DBS: u32 = 7;

    pub fn new<Tls>(env: &sneed::Env<Tls>) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
        let transactions =
            DatabaseUnique::create(env, &mut rwtxn, "transactions")
                .map_err(EnvError::from)?;
        let spent_utxos =
            DatabaseUnique::create(env, &mut rwtxn, "spent_utxos")
                .map_err(EnvError::from)?;
        let version =
            DatabaseUnique::create(env, &mut rwtxn, "mempool_version")
                .map_err(EnvError::from)?;
        match version.try_get(&rwtxn, &())? {
            Some(db_version)
                if db_version
                    < Version {
                        major: 0,
                        minor: 20,
                        patch: 0,
                    } =>
            {
                return Err(Error::IncompatibleVersion {
                    version: db_version,
                    db_path: env.path().to_path_buf(),
                });
            }
            Some(_) => (),
            None => version
                .put(&mut rwtxn, &(), &*VERSION)
                .map_err(DbError::from)?,
        };
        let address_to_txs =
            DatabaseUnique::create(env, &mut rwtxn, "address_to_txs")
                .map_err(EnvError::from)?;
        let pending_decision_claims =
            DatabaseUnique::create(env, &mut rwtxn, "pending_decision_claims")
                .map_err(EnvError::from)?;
        let trade_insertion_order =
            DatabaseUnique::create(env, &mut rwtxn, "trade_insertion_order")
                .map_err(EnvError::from)?;
        let trade_order_counter =
            DatabaseUnique::create(env, &mut rwtxn, "trade_order_counter")
                .map_err(EnvError::from)?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(Self {
            transactions,
            spent_utxos,
            _version: version,
            address_to_txs,
            pending_decision_claims,
            trade_insertion_order,
            trade_order_counter,
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
        if let Some(ref data) = transaction.data {
            match data {
                TransactionData::ClaimDecision(payload) => {
                    for entry in &payload.decisions {
                        decision_ids.push(entry.decision_id_bytes);
                    }
                }
                TransactionData::CreateMarket { new_claims, .. } => {
                    for payload in new_claims {
                        for entry in &payload.decisions {
                            decision_ids.push(entry.decision_id_bytes);
                        }
                    }
                }
                _ => {}
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
                    const_hex::encode(decision_id),
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

    pub fn put(
        &self,
        txn: &mut RwTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        let ancestors = self.ancestors(txn, &transaction.transaction)?;
        if ancestors.len() >= MAX_UNCONFIRMED_ANCESTORS {
            return Err(Error::TooManyAncestors {
                count: ancestors.len(),
            });
        }
        self.insert(txn, transaction)
    }

    /// Take back a transaction that a disconnected block carried. The chain
    /// accepted it once, so the ancestor limit does not apply.
    pub fn put_disconnected(
        &self,
        txn: &mut RwTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        self.insert(txn, transaction)
    }

    fn insert(
        &self,
        txn: &mut RwTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        let txid = transaction.transaction.txid();
        tracing::debug!("adding transaction {txid} to mempool");
        let claimed_decisions =
            Self::get_claimed_decision_ids(&transaction.transaction);
        if !claimed_decisions.is_empty() {
            self.put_decision_claims(txn, txid, &claimed_decisions)?;
        }
        for (outpoint, _) in &transaction.transaction.inputs {
            if self
                .spent_utxos
                .try_get(txn, outpoint)
                .map_err(DbError::from)?
                .is_some()
            {
                return Err(Error::UtxoDoubleSpent);
            }
            self.spent_utxos
                .put(txn, outpoint, &txid)
                .map_err(DbError::from)?;
        }
        self.transactions
            .put(txn, &txid, transaction)
            .map_err(DbError::from)?;
        let () = self.index_tx_addresses(txn, transaction)?;
        if Self::is_trade_tx(transaction) {
            let counter =
                self.trade_order_counter.try_get(txn, &())?.unwrap_or(0);
            let next = counter + 1;
            self.trade_insertion_order.put(txn, &next, &txid)?;
            self.trade_order_counter.put(txn, &(), &next)?;
        }
        Ok(())
    }

    pub fn delete(&self, rwtxn: &mut RwTxn, txid: Txid) -> Result<(), Error> {
        let mut pending_deletes = VecDeque::from([txid]);
        while let Some(txid) = pending_deletes.pop_front() {
            if let Some(tx) = self
                .transactions
                .try_get(rwtxn, &txid)
                .map_err(DbError::from)?
            {
                for (outpoint, _) in &tx.transaction.inputs {
                    self.spent_utxos
                        .delete(rwtxn, outpoint)
                        .map_err(DbError::from)?;
                }
                let () = self.unindex_tx_addresses(rwtxn, &tx)?;
                let () = self.delete_decision_claims(rwtxn, &tx)?;
                if Self::is_trade_tx(&tx) {
                    self.delete_trade_order(rwtxn, &txid)?;
                }
                self.transactions
                    .delete(rwtxn, &txid)
                    .map_err(DbError::from)?;
                for vout in 0..tx.transaction.outputs.len() {
                    let outpoint = OutPoint::Regular {
                        txid,
                        vout: vout as u32,
                    };
                    if let Some(child_txid) = self
                        .spent_utxos
                        .try_get(rwtxn, &outpoint)
                        .map_err(DbError::from)?
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
                    decision_id = %const_hex::encode(decision_id),
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

    /// Remove a transaction that a block confirms, and keep its children. A
    /// child of a confirmed parent spends a confirmed output, so it stays
    /// valid.
    pub fn delete_confirmed(
        &self,
        rwtxn: &mut RwTxn,
        txid: Txid,
    ) -> Result<(), Error> {
        let Some(tx) = self
            .transactions
            .try_get(rwtxn, &txid)
            .map_err(DbError::from)?
        else {
            return Ok(());
        };
        for (outpoint, _) in &tx.transaction.inputs {
            self.spent_utxos
                .delete(rwtxn, outpoint)
                .map_err(DbError::from)?;
        }
        let () = self.unindex_tx_addresses(rwtxn, &tx)?;
        let () = self.delete_decision_claims(rwtxn, &tx)?;
        if Self::is_trade_tx(&tx) {
            self.delete_trade_order(rwtxn, &txid)?;
        }
        self.transactions
            .delete(rwtxn, &txid)
            .map_err(DbError::from)?;
        Ok(())
    }

    /// The mempool transactions that `transaction` spends from, directly or
    /// through another mempool transaction.
    pub fn ancestors(
        &self,
        rotxn: &RoTxn,
        transaction: &Transaction,
    ) -> Result<HashSet<Txid>, Error> {
        let mut found = HashSet::new();
        let mut pending: VecDeque<Txid> = transaction
            .inputs
            .iter()
            .filter_map(|(outpoint, _)| parent_txid(outpoint))
            .collect();
        while let Some(txid) = pending.pop_front() {
            if found.contains(&txid) {
                continue;
            }
            let Some(tx) = self
                .transactions
                .try_get(rotxn, &txid)
                .map_err(DbError::from)?
            else {
                continue;
            };
            found.insert(txid);
            pending.extend(
                tx.transaction
                    .inputs
                    .iter()
                    .filter_map(|(outpoint, _)| parent_txid(outpoint)),
            );
        }
        Ok(found)
    }

    /// The outputs this mempool holds that `transaction` spends. The confirmed
    /// UTXO set holds none of them.
    pub fn unconfirmed_outputs(
        &self,
        rotxn: &RoTxn,
        transaction: &Transaction,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let mut res = HashMap::new();
        for (outpoint, _) in &transaction.inputs {
            let OutPoint::Regular { txid, vout } = outpoint else {
                continue;
            };
            let Some(parent) = self
                .transactions
                .try_get(rotxn, txid)
                .map_err(DbError::from)?
            else {
                continue;
            };
            let Some(output) =
                parent.transaction.outputs.as_slice().get(*vout as usize)
            else {
                continue;
            };
            res.insert(*outpoint, output.clone());
        }
        Ok(res)
    }

    /// Transactions with a parent before its child. `limit` caps how many the
    /// walk returns, and the result stays closed under parents, so a shorter
    /// walk never gives a child whose parent it left out. The read still
    /// covers the whole mempool; the limit bounds the walk and the clones.
    pub fn topological(
        &self,
        rotxn: &RoTxn,
        limit: Option<usize>,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        let txs: Vec<(Txid, AuthorizedTransaction)> = self
            .transactions
            .iter(rotxn)
            .map_err(DbError::from)?
            .collect()
            .map_err(DbError::from)?;
        let by_txid: HashMap<Txid, &AuthorizedTransaction> =
            txs.iter().map(|(txid, tx)| (*txid, tx)).collect();
        let mut order = Vec::with_capacity(txs.len());
        let mut placed = HashSet::with_capacity(txs.len());
        for (txid, _) in &txs {
            if limit.is_some_and(|limit| order.len() >= limit) {
                break;
            }
            let mut stack = vec![*txid];
            while let Some(txid) = stack.last().copied() {
                if limit.is_some_and(|limit| order.len() >= limit) {
                    break;
                }
                if placed.contains(&txid) {
                    stack.pop();
                    continue;
                }
                let Some(tx) = by_txid.get(&txid) else {
                    stack.pop();
                    continue;
                };
                let parent =
                    tx.transaction.inputs.iter().find_map(|(outpoint, _)| {
                        parent_txid(outpoint).filter(|parent| {
                            by_txid.contains_key(parent)
                                && !placed.contains(parent)
                        })
                    });
                match parent {
                    Some(parent) => stack.push(parent),
                    None => {
                        placed.insert(txid);
                        order.push((*tx).clone());
                        stack.pop();
                    }
                }
            }
        }
        Ok(order)
    }

    /// The unconfirmed outputs that pay one of `addresses` and that this
    /// wallet made on its own.
    ///
    /// `confirmed` names the outputs the wallet already holds from the chain.
    /// Bitcoin Core takes an unconfirmed output only when the wallet funded
    /// every input of the transaction that made it, and it walks the parents
    /// to the last confirmed one. This copies that rule, so an unconfirmed
    /// payment from a stranger never appears here.
    ///
    /// The wallet counts these outputs in its balance whatever the
    /// `--spend-zero-conf-change` option says. The option decides only whether
    /// the wallet may put them in a new transaction.
    pub fn own_unconfirmed_utxos(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<Address>,
        confirmed: &HashSet<OutPoint>,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let mut res = HashMap::new();
        // A parent comes first, so its trust and its ancestors are known by the
        // time the walk reaches the child.
        let mut trusted: HashSet<OutPoint> = HashSet::new();
        let mut ancestors: HashMap<Txid, HashSet<Txid>> = HashMap::new();
        for tx in self.topological(rotxn, None)? {
            let txid = tx.transaction.txid();
            let mut is_trusted = true;
            let mut tx_ancestors = HashSet::new();
            for (outpoint, _) in &tx.transaction.inputs {
                if let Some(parent) = parent_txid(outpoint)
                    && let Some(parent_ancestors) = ancestors.get(&parent)
                {
                    tx_ancestors.extend(parent_ancestors.iter().copied());
                    tx_ancestors.insert(parent);
                }
                if confirmed.contains(outpoint) || trusted.contains(outpoint) {
                    continue;
                }
                is_trusted = false;
            }
            ancestors.insert(txid, tx_ancestors);
            if !is_trusted {
                continue;
            }
            for (vout, output) in tx.transaction.outputs.iter().enumerate() {
                if !addresses.contains(&output.address) {
                    continue;
                }
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                trusted.insert(outpoint);
                if self
                    .spent_utxos
                    .try_get(rotxn, &outpoint)
                    .map_err(DbError::from)?
                    .is_none()
                {
                    res.insert(outpoint, output.clone());
                }
            }
        }
        Ok(res)
    }

    /// The transaction in this mempool that spends `outpoint`, if there is
    /// one.
    pub fn spender(
        &self,
        rotxn: &RoTxn,
        outpoint: &OutPoint,
    ) -> Result<Option<Txid>, Error> {
        let txid = self
            .spent_utxos
            .try_get(rotxn, outpoint)
            .map_err(DbError::from)?;
        Ok(txid)
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

    /// regenerate utreexo proofs for all txs in the mempool
    ///
    /// A transaction whose inputs can no longer be proven against the
    /// accumulator (eg. because they were spent by a just-connected block via
    /// a conflicting transaction) is no longer valid. Such a transaction is
    /// evicted from the mempool, along with its descendants, rather than
    /// propagating an error that would abort block connect/disconnect.
    pub fn regenerate_proofs(
        &self,
        rwtxn: &mut RwTxn,
        accumulator: &Accumulator,
    ) -> Result<(), Error> {
        let txids: Vec<_> = self
            .transactions
            .iter_keys(rwtxn)
            .map_err(DbError::from)?
            .collect()
            .map_err(DbError::from)?;
        for txid in txids {
            // The tx may already have been evicted as a descendant of an
            // earlier invalidated tx.
            let Some(mut tx) = self
                .transactions
                .try_get(rwtxn, &txid)
                .map_err(DbError::from)?
            else {
                continue;
            };
            let unconfirmed =
                self.unconfirmed_outputs(rwtxn, &tx.transaction)?;
            let targets: Vec<_> = tx
                .transaction
                .inputs
                .iter()
                .filter(|(outpoint, _)| !unconfirmed.contains_key(outpoint))
                .map(|(_, utxo_hash)| utxo_hash.into())
                .collect();
            match accumulator.prove(&targets) {
                Ok(proof) => {
                    tx.transaction.proof = proof;
                    self.transactions
                        .put(rwtxn, &txid, &tx)
                        .map_err(DbError::from)?;
                }
                Err(_) => {
                    tracing::debug!(
                        "evicting mempool transaction {txid}: inputs no \
                         longer in accumulator"
                    );
                    let () = self.delete(rwtxn, txid)?;
                }
            }
        }
        Ok(())
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
    type WatchStream = std::pin::Pin<Box<dyn Stream<Item = ()> + Send>>;

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
        Box::pin(streams.ready_chunks(streams_len).map(|signals| {
            assert_ne!(signals.len(), 0);
            #[allow(clippy::unused_unit)]
            ()
        }))
    }
}

/// The mempool transaction that could have made this outpoint. A coinbase,
/// a deposit, a market funds or a payout outpoint names no transaction.
fn parent_txid(outpoint: &OutPoint) -> Option<Txid> {
    match outpoint {
        OutPoint::Regular { txid, .. } => Some(*txid),
        OutPoint::Coinbase { .. }
        | OutPoint::Deposit(_)
        | OutPoint::MarketFunds { .. }
        | OutPoint::Payout { .. } => None,
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
            inputs: vec![(input_outpoint(input_seed), [0; 32])].into(),
            proof: Default::default(),
            outputs: vec![].into(),
            data: Some(TransactionData::ClaimDecision(
                crate::types::ClaimDecisionPayload {
                    decision_type: DecisionType::Binary,
                    decisions: entries,
                },
            )),
        };
        Authorized {
            transaction: tx,
            authorizations: vec![],
            actor_proof: None,
        }
    }

    fn regular_tx(input_seed: u8) -> AuthorizedTransaction {
        let tx = Transaction {
            inputs: vec![(input_outpoint(input_seed), [0; 32])].into(),
            proof: Default::default(),
            outputs: vec![].into(),
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

#[cfg(test)]
mod test {
    use bitcoin::hashes::Hash as _;

    use super::*;
    use crate::types::{
        Address, OutputContent, PointedOutput,
        authorization::{SigningKey, get_address},
        hash,
    };

    fn temp_env(
        test_name: &str,
    ) -> anyhow::Result<(temp_dir::TempDir, sneed::Env)> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let temp_dir = temp_dir::TempDir::with_prefix(format!(
            "{test_name}-{nanos}-{}",
            std::process::id()
        ))?;
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(16 * 1024 * 1024).max_dbs(MemPool::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, temp_dir.path()) }?;
        Ok((temp_dir, env))
    }

    fn value_output(address: Address, sats: u64) -> Output {
        Output {
            address,
            content: OutputContent::Value(bitcoin::Amount::from_sat(sats)),
        }
    }

    fn deposit_outpoint(seed: u8) -> OutPoint {
        OutPoint::Deposit(bitcoin::OutPoint {
            txid: bitcoin::Txid::from_byte_array([seed; 32]),
            vout: 0,
        })
    }

    /// Build a transaction that spends `outpoint`, worth `output` after it.
    /// The mempool never checks a signature, so the authorization is empty.
    fn spend(
        outpoint: OutPoint,
        spent: &Output,
        output: Output,
    ) -> AuthorizedTransaction {
        let utxo_hash = hash(&PointedOutput {
            outpoint,
            output: spent.clone(),
        });
        AuthorizedTransaction {
            authorizations: Vec::new(),
            transaction: Transaction {
                inputs: vec![(outpoint, utxo_hash)].into(),
                proof: Default::default(),
                outputs: vec![output].into(),
                data: None,
            },
            actor_proof: None,
        }
    }

    /// A chain of `len` transactions, each spending the one before it.
    fn chain(
        address: Address,
        start: OutPoint,
        start_output: Output,
        len: usize,
    ) -> Vec<AuthorizedTransaction> {
        let mut txs = Vec::with_capacity(len);
        let mut outpoint = start;
        let mut spent = start_output;
        for i in 0..len {
            let output = value_output(address, 10_000 - i as u64 - 1);
            let tx = spend(outpoint, &spent, output.clone());
            outpoint = OutPoint::Regular {
                txid: tx.transaction.txid(),
                vout: 0,
            };
            spent = output;
            txs.push(tx);
        }
        txs
    }

    fn owner() -> (SigningKey, Address) {
        let key = SigningKey::new(&mut rand::rng());
        let address = get_address(&(&key).into());
        (key, address)
    }

    #[test]
    fn topological_puts_a_parent_before_its_child() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("topological")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let start = deposit_outpoint(0x01);
        let start_output = value_output(address, 10_000);
        let txs = chain(address, start, start_output, 4);

        let mut rwtxn = env.write_txn()?;
        // Insert the children first, so key order cannot pass the test by
        // accident.
        for tx in txs.iter().rev() {
            mempool.put(&mut rwtxn, tx)?;
        }
        rwtxn.commit()?;

        let rotxn = env.read_txn()?;
        let order: Vec<_> = mempool
            .topological(&rotxn, None)?
            .into_iter()
            .map(|tx| tx.transaction.txid())
            .collect();
        let expected: Vec<_> =
            txs.iter().map(|tx| tx.transaction.txid()).collect();
        anyhow::ensure!(
            order == expected,
            "expected {expected:?}, got {order:?}"
        );
        Ok(())
    }

    #[test]
    fn the_mempool_refuses_a_chain_past_the_limit() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("ancestor_limit")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let start = deposit_outpoint(0x02);
        let start_output = value_output(address, 10_000);
        let txs =
            chain(address, start, start_output, MAX_UNCONFIRMED_ANCESTORS + 1);

        let mut rwtxn = env.write_txn()?;
        for tx in txs.iter().take(MAX_UNCONFIRMED_ANCESTORS) {
            mempool.put(&mut rwtxn, tx)?;
        }
        let last = mempool.put(&mut rwtxn, &txs[MAX_UNCONFIRMED_ANCESTORS]);
        anyhow::ensure!(
            matches!(last, Err(Error::TooManyAncestors { count }) if count
                == MAX_UNCONFIRMED_ANCESTORS),
            "expected TooManyAncestors, got {last:?}",
        );
        Ok(())
    }

    #[test]
    fn a_confirmed_parent_leaves_its_child_behind() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("delete_confirmed")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let start = deposit_outpoint(0x03);
        let start_output = value_output(address, 10_000);
        let txs = chain(address, start, start_output, 2);

        let mut rwtxn = env.write_txn()?;
        for tx in &txs {
            mempool.put(&mut rwtxn, tx)?;
        }
        mempool.delete_confirmed(&mut rwtxn, txs[0].transaction.txid())?;
        rwtxn.commit()?;

        let rotxn = env.read_txn()?;
        let left: Vec<_> = mempool
            .take_all(&rotxn)?
            .into_iter()
            .map(|tx| tx.transaction.txid())
            .collect();
        anyhow::ensure!(
            left == vec![txs[1].transaction.txid()],
            "the child must stay, got {left:?}",
        );
        Ok(())
    }

    #[test]
    fn a_double_spending_parent_takes_its_child() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("delete_cascades")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let start = deposit_outpoint(0x04);
        let start_output = value_output(address, 10_000);
        let txs = chain(address, start, start_output, 2);

        let mut rwtxn = env.write_txn()?;
        for tx in &txs {
            mempool.put(&mut rwtxn, tx)?;
        }
        mempool.delete(&mut rwtxn, txs[0].transaction.txid())?;
        rwtxn.commit()?;

        let rotxn = env.read_txn()?;
        anyhow::ensure!(mempool.take_all(&rotxn)?.is_empty());
        Ok(())
    }

    #[test]
    fn own_change_is_spendable_and_a_stranger_output_is_not()
    -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("trust")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let stranger =
            get_address(&(&SigningKey::new(&mut rand::rng())).into());

        // The wallet holds one confirmed coin and spends it. The change is
        // its own, so it is trusted.
        let mine = deposit_outpoint(0x05);
        let mine_output = value_output(address, 10_000);
        let own = spend(mine, &mine_output, value_output(address, 9_000));
        // A stranger spends a coin the wallet never held, and pays the wallet.
        let theirs = deposit_outpoint(0x06);
        let theirs_output = value_output(stranger, 5_000);
        let gift = spend(theirs, &theirs_output, value_output(address, 4_000));

        let mut rwtxn = env.write_txn()?;
        mempool.put(&mut rwtxn, &own)?;
        mempool.put(&mut rwtxn, &gift)?;
        rwtxn.commit()?;

        let addresses = HashSet::from([address]);
        let confirmed = HashSet::from([mine]);
        let own_outpoint = OutPoint::Regular {
            txid: own.transaction.txid(),
            vout: 0,
        };
        let gift_outpoint = OutPoint::Regular {
            txid: gift.transaction.txid(),
            vout: 0,
        };

        let rotxn = env.read_txn()?;
        let spendable =
            mempool.own_unconfirmed_utxos(&rotxn, &addresses, &confirmed)?;
        anyhow::ensure!(
            spendable.keys().collect::<Vec<_>>() == vec![&own_outpoint],
            "only own change is trusted, got {spendable:?}",
        );
        anyhow::ensure!(
            !spendable.contains_key(&gift_outpoint),
            "an unconfirmed payment from someone else waits for a block",
        );
        Ok(())
    }

    #[test]
    fn an_untrusted_ancestor_stops_the_whole_chain() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("trust_chain")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let stranger =
            get_address(&(&SigningKey::new(&mut rand::rng())).into());

        let theirs = deposit_outpoint(0x07);
        let theirs_output = value_output(stranger, 5_000);
        let gift_output = value_output(address, 4_000);
        let gift = spend(theirs, &theirs_output, gift_output.clone());
        let gift_outpoint = OutPoint::Regular {
            txid: gift.transaction.txid(),
            vout: 0,
        };
        let child =
            spend(gift_outpoint, &gift_output, value_output(address, 3_000));

        let mut rwtxn = env.write_txn()?;
        mempool.put(&mut rwtxn, &gift)?;
        mempool.put(&mut rwtxn, &child)?;
        rwtxn.commit()?;

        let rotxn = env.read_txn()?;
        let spendable = mempool.own_unconfirmed_utxos(
            &rotxn,
            &HashSet::from([address]),
            &HashSet::new(),
        )?;
        anyhow::ensure!(
            spendable.is_empty(),
            "a child of a stranger's transaction is not trusted, got \
             {spendable:?}",
        );
        Ok(())
    }

    /// A chain at the limit stays visible, so the balance shows the money.
    /// The mempool refuses the next link, and the user reads a clear error.
    #[test]
    fn a_chain_at_the_limit_stays_visible() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("ancestor_boundary")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let start = deposit_outpoint(0x09);
        let start_output = value_output(address, 10_000);
        let txs =
            chain(address, start, start_output, MAX_UNCONFIRMED_ANCESTORS);

        let mut rwtxn = env.write_txn()?;
        for tx in &txs {
            mempool.put(&mut rwtxn, tx)?;
        }
        rwtxn.commit()?;

        let last = &txs[MAX_UNCONFIRMED_ANCESTORS - 1];
        let tip = OutPoint::Regular {
            txid: last.transaction.txid(),
            vout: 0,
        };
        let rotxn = env.read_txn()?;
        anyhow::ensure!(
            mempool.ancestors(&rotxn, &last.transaction)?.len()
                == MAX_UNCONFIRMED_ANCESTORS - 1,
        );
        let visible = mempool.own_unconfirmed_utxos(
            &rotxn,
            &HashSet::from([address]),
            &HashSet::from([start]),
        )?;
        anyhow::ensure!(
            visible.keys().collect::<Vec<_>>() == vec![&tip],
            "the money the chain holds must stay visible, got {visible:?}",
        );
        drop(rotxn);

        // A child of the last link would carry 25 ancestors.
        let child = spend(
            tip,
            &value_output(address, 10_000 - MAX_UNCONFIRMED_ANCESTORS as u64),
            value_output(address, 1),
        );
        let mut rwtxn = env.write_txn()?;
        let refused = mempool.put(&mut rwtxn, &child);
        anyhow::ensure!(
            matches!(refused, Err(Error::TooManyAncestors { count }) if count
                == MAX_UNCONFIRMED_ANCESTORS),
            "the mempool must refuse the next link, got {refused:?}",
        );
        Ok(())
    }

    #[test]
    fn a_spent_unconfirmed_output_is_not_offered() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("trust_spent")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let start = deposit_outpoint(0x08);
        let start_output = value_output(address, 10_000);
        let txs = chain(address, start, start_output, 2);

        let mut rwtxn = env.write_txn()?;
        for tx in &txs {
            mempool.put(&mut rwtxn, tx)?;
        }
        rwtxn.commit()?;

        let rotxn = env.read_txn()?;
        let spendable = mempool.own_unconfirmed_utxos(
            &rotxn,
            &HashSet::from([address]),
            &HashSet::from([start]),
        )?;
        let tip = OutPoint::Regular {
            txid: txs[1].transaction.txid(),
            vout: 0,
        };
        anyhow::ensure!(
            spendable.keys().collect::<Vec<_>>() == vec![&tip],
            "only the last output of the chain is unspent, got {spendable:?}",
        );
        Ok(())
    }
}
