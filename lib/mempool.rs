use std::collections::{HashMap, HashSet, VecDeque};

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
        Address, AuthorizedTransaction, InPoint, OutPoint, Output, Txid,
        VERSION, Version,
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
    #[error("can't add transaction, slot {0} already claimed in mempool")]
    SlotAlreadyClaimedInMempool(String),
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
    /// Tracks pending slot claims: slot_id_bytes -> claiming txid
    pending_slot_claims:
        DatabaseUnique<SerdeBincode<[u8; 3]>, SerdeBincode<Txid>>,
    trade_insertion_order:
        DatabaseUnique<SerdeBincode<u64>, SerdeBincode<Txid>>,
    trade_order_counter: DatabaseUnique<UnitKey, SerdeBincode<u64>>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

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
        let pending_slot_claims =
            DatabaseUnique::create(env, &mut rwtxn, "pending_slot_claims")?;
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
            pending_slot_claims,
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
    fn assoc_tx_with_relevant_addresses(
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
    fn unassoc_tx_with_relevant_addresses(
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

    /// Extract slot IDs being claimed by this transaction
    fn get_claimed_slot_ids(
        transaction: &AuthorizedTransaction,
    ) -> Vec<[u8; 3]> {
        use crate::types::TransactionData;

        let mut slot_ids = Vec::new();
        if let Some(ref data) = transaction.transaction.data {
            match data {
                TransactionData::ClaimDecisionSlot {
                    slot_id_bytes, ..
                } => {
                    slot_ids.push(*slot_id_bytes);
                }
                TransactionData::ClaimCategorySlots { slots, .. } => {
                    for (slot_id_bytes, _) in slots {
                        slot_ids.push(*slot_id_bytes);
                    }
                }
                _ => {}
            }
        }
        slot_ids
    }

    /// Check if any slots are already claimed in mempool, and add them.
    ///
    /// # Atomicity
    /// This method uses a check-then-act pattern that relies on LMDB write
    /// transaction exclusivity - only one write transaction can be active at
    /// a time across all threads, preventing TOCTOU races between the check
    /// and add phases.
    fn put_slot_claims(
        &self,
        rwtxn: &mut RwTxn,
        txid: Txid,
        slot_ids: &[[u8; 3]],
    ) -> Result<(), Error> {
        // First check for conflicts
        for slot_id in slot_ids {
            if let Some(existing_txid) =
                self.pending_slot_claims.try_get(rwtxn, slot_id)?
                && existing_txid != txid
            {
                return Err(Error::SlotAlreadyClaimedInMempool(hex::encode(
                    slot_id,
                )));
            }
        }
        // No conflicts, add all claims
        for slot_id in slot_ids {
            self.pending_slot_claims.put(rwtxn, slot_id, &txid)?;
        }
        Ok(())
    }

    /// Remove slot claims for a transaction
    fn delete_slot_claims(
        &self,
        rwtxn: &mut RwTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        let slot_ids = Self::get_claimed_slot_ids(transaction);
        for slot_id in slot_ids {
            self.pending_slot_claims.delete(rwtxn, &slot_id)?;
        }
        Ok(())
    }

    pub fn put(
        &self,
        rwtxn: &mut RwTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        let txid = transaction.transaction.txid();
        tracing::debug!("adding transaction {txid} to mempool");

        // Check for duplicate slot claims first
        let claimed_slots = Self::get_claimed_slot_ids(transaction);
        if !claimed_slots.is_empty() {
            self.put_slot_claims(rwtxn, txid, &claimed_slots)?;
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
        let () = self.assoc_tx_with_relevant_addresses(rwtxn, transaction)?;

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
                let () = self.unassoc_tx_with_relevant_addresses(rwtxn, &tx)?;
                let () = self.delete_slot_claims(rwtxn, &tx)?;

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
}

impl Watchable<()> for MemPool {
    type WatchStream = impl Stream<Item = ()>;

    /// Get a signal that notifies whenever the mempool changes
    fn watch(&self) -> Self::WatchStream {
        let watchables = [
            self.transactions.watch().clone(),
            self.spent_utxos.watch().clone(),
            self.pending_slot_claims.watch().clone(),
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
