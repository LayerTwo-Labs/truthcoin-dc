use std::{
    borrow::BorrowMut,
    collections::{HashMap, HashSet},
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
};

use fallible_iterator::{FallibleIterator, IteratorExt};
use futures::Stream;
use sneed::{DbError, Env, EnvError, RoTxn, RwTxn, RwTxnError};
use tokio::sync::Mutex;
use tonic::transport::Channel;
use truthcoin_dc_types::{
    M6id, WithdrawalBundleStatus, state::WithdrawalBundleInfo,
};

use crate::{
    archive::Archive,
    math::trading,
    mempool::{self, MemPool},
    net::{DialKnownPeersHandle, Net},
    state::{
        self, State,
        markets::{
            MarketId,
            block::{TradeApplyResult, TradeSimulation},
        },
    },
    types::{
        Accumulator, Address, AmountUnderflowError, Authorized,
        AuthorizedTransaction, BlockHash, BmmResult, Body, FilledTransaction,
        Header, InPoint, MainchainSyncProgress, Network, OutPoint, OutPointKey,
        Output, SpentOutput, Tip, Transaction, TxIn, Txid, WithdrawalBundle,
        authorization::{BatchVerificationContext, rand_core::CryptoRng},
        net::{Peer, PeerAddress, ResolvedPeerAddress},
        proto::{self, mainchain},
        state::TwoWayPegEvent,
    },
    util::Watchable,
};

pub(crate) mod error;
pub use error::Error;
mod mainchain_task;
use mainchain_task::MainchainTaskHandle;
mod net_task;
use net_task::NetTaskHandle;

pub type FilledTransactionWithPosition =
    (Authorized<FilledTransaction>, Option<TxIn>);

#[derive(Debug)]
pub struct Config {
    pub datadir: PathBuf,
    pub bind_addr: SocketAddr,
    pub magic_bytes_override: Option<crate::net::peer_message::MagicBytes>,
    pub network: Network,
    pub add_peers: HashSet<PeerAddress>,
    pub server_names: HashSet<String>,
    /// Blocks per voting period, for a test network
    pub decision_config_testing: Option<u32>,
}

/// Handles for spawned tasks / task sets
#[derive(Clone)]
struct TaskHandles {
    _dial_known_peers: Arc<DialKnownPeersHandle>,
    mainchain: MainchainTaskHandle,
    net: NetTaskHandle,
}

#[derive(Clone)]
pub struct Node<MainchainTransport = Channel> {
    archive: Archive,
    batch_verification_ctxt: BatchVerificationContext,
    cusf_mainchain: mainchain::ValidatorClient<MainchainTransport>,
    cusf_mainchain_block_producer:
        Option<Arc<Mutex<mainchain::BlockProducerClient<MainchainTransport>>>>,
    env: sneed::Env<heed::WithoutTls>,
    mempool: MemPool,
    net: Net,
    state: State,
    task_handles: TaskHandles,
}

impl<MainchainTransport> Node<MainchainTransport>
where
    MainchainTransport: proto::Transport,
{
    pub fn new<R>(
        config: Config,
        cusf_mainchain: mainchain::ValidatorClient<MainchainTransport>,
        cusf_mainchain_block_producer: Option<
            mainchain::BlockProducerClient<MainchainTransport>,
        >,
        rng: &mut R,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<Self, Error>
    where
        mainchain::ValidatorClient<MainchainTransport>: Clone,
        MainchainTransport: Send + 'static,
        <MainchainTransport as tonic::client::GrpcService<
            tonic::body::Body,
        >>::Future: Send,
        R: CryptoRng,
{
        let Config {
            datadir,
            bind_addr,
            magic_bytes_override,
            network,
            add_peers,
            server_names,
            decision_config_testing,
        } = config;
        let env_path = datadir.join("data.mdb");
        // let _ = std::fs::remove_dir_all(&env_path);
        std::fs::create_dir_all(&env_path)?;
        let env = {
            use heed::EnvFlags;
            let mut env_open_opts =
                heed::EnvOpenOptions::new().read_txn_without_tls();
            env_open_opts
                .map_size(128 * 1024 * 1024 * 1024) // 128 GB
                .max_dbs(
                    Archive::NUM_DBS
                        + MemPool::NUM_DBS
                        + Net::NUM_DBS
                        + State::NUM_DBS,
                );
            // Apply LMDB "fast" flags consistent with our benchmark setup:
            // - WRITE_MAP lets us write directly into the memory map instead of
            //   copying into LMDB's page buffer, reducing syscall overhead for
            //   write-heavy workloads.
            // - MAP_ASYNC hands dirty-page flushing to the kernel so commits do
            //   not block waiting for msync, keeping latencies tight.
            // - NO_SYNC and NO_META_SYNC skip fsync calls for data and
            //   metadata; this trades durability for throughput, which is
            //   acceptable here because the state can be reconstructed from the
            //   canonical chain if a crash occurs.
            // - NO_READ_AHEAD disables kernel readahead that would otherwise
            //   touch cold pages we immediately overwrite, improving random
            //   access behaviour on SSDs used in testing.
            // - NO_TLS stops LMDB from relying on thread-local storage for
            //   reader slots so transactions can be moved across Tokio tasks.
            // WRITE_MAP/MAP_ASYNC/NO_READ_AHEAD are gated off on Windows:
            // LMDB's writable-mmap path returns ERROR_INVALID_HANDLE on commit
            // there.
            #[cfg(not(windows))]
            let fast_flags = EnvFlags::WRITE_MAP
                | EnvFlags::MAP_ASYNC
                | EnvFlags::NO_SYNC
                | EnvFlags::NO_META_SYNC
                | EnvFlags::NO_READ_AHEAD;
            #[cfg(windows)]
            let fast_flags = EnvFlags::NO_SYNC | EnvFlags::NO_META_SYNC;
            unsafe { env_open_opts.flags(fast_flags) };
            unsafe { Env::open(&env_open_opts, &env_path) }
                .map_err(EnvError::from)?
        };
        let state = State::new(&env, decision_config_testing)?;
        let archive = Archive::new(&env)?;
        let mempool = MemPool::new(&env)?;
        let (mainchain_task_handle, mainchain_task_event_rx) =
            MainchainTaskHandle::new(
                env.clone(),
                archive.clone(),
                cusf_mainchain.clone(),
            );
        let batch_verification_ctxt = BatchVerificationContext::new(rng);
        let (net, peer_info_rx, dial_known_peers_handle) = Net::new(
            runtime.handle(),
            &env,
            archive.clone(),
            batch_verification_ctxt,
            magic_bytes_override,
            network,
            mempool.clone(),
            state.clone(),
            bind_addr,
            add_peers,
            server_names,
        )?;
        let net_task_handle = NetTaskHandle::new(
            runtime,
            env.clone(),
            archive.clone(),
            mainchain_task_handle.clone(),
            mainchain_task_event_rx,
            mempool.clone(),
            net.clone(),
            peer_info_rx,
            state.clone(),
        );
        let task_handles = TaskHandles {
            _dial_known_peers: Arc::new(dial_known_peers_handle),
            mainchain: mainchain_task_handle,
            net: net_task_handle,
        };
        let cusf_mainchain_block_producer = cusf_mainchain_block_producer
            .map(|block_producer| Arc::new(Mutex::new(block_producer)));
        Ok(Self {
            archive,
            batch_verification_ctxt,
            cusf_mainchain,
            cusf_mainchain_block_producer,
            env,
            mempool,
            net,
            state,
            task_handles,
        })
    }

    pub fn env(&self) -> &Env<heed::WithoutTls> {
        &self.env
    }

    pub fn archive(&self) -> &Archive {
        &self.archive
    }

    /// Borrow the CUSF mainchain client
    #[inline(always)]
    pub fn with_cusf_mainchain<F, Output>(&self, f: F) -> Output
    where
        F: FnOnce(&mainchain::ValidatorClient<MainchainTransport>) -> Output,
    {
        f(&self.cusf_mainchain)
    }

    pub fn dns_resolver(&self) -> &Arc<hickory_resolver::TokioResolver> {
        &self.net.dns_resolver
    }

    /// Invalidate a block.
    /// This will delete the header and body, and mark invalid, the specified
    /// block and any descendants.
    /// The node will re-org to a previous valid block in the active chain.
    pub fn invalidate_block(&self, block_hash: BlockHash) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
        let Some(header) = self.archive.try_get_header(&rwtxn, block_hash)?
        else {
            return Ok(());
        };
        // check if the specified block is in the active chain
        let tip = self.state.try_get_tip(&rwtxn)?;
        let in_active_chain = if let Some(tip) = tip {
            self.archive.is_descendant(&rwtxn, block_hash, tip)?
        } else {
            false
        };
        // re-org if necessary
        if in_active_chain {
            while self.state.try_get_tip(&rwtxn)? != header.prev_side_hash {
                net_task::disconnect_tip_(
                    &mut rwtxn,
                    &self.archive,
                    &self.mempool,
                    &self.state,
                )?;
            }
        }
        // invalidate within archive
        let () = self.archive.invalidate_block(&mut rwtxn, block_hash)?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn try_get_height(&self) -> Result<Option<u32>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.state.try_get_height(&rotxn)?)
    }

    pub fn try_get_best_hash(&self) -> Result<Option<BlockHash>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.state.try_get_tip(&rotxn)?)
    }

    /// Regenerate proofs and submit transaction
    pub fn submit_transaction<Tx>(
        &self,
        mut transaction: Tx,
    ) -> Result<(), Error>
    where
        Tx: BorrowMut<AuthorizedTransaction>,
    {
        {
            let mut rotxn = self.env.write_txn().map_err(EnvError::from)?;
            let unconfirmed = self.mempool.unconfirmed_outputs(
                &rotxn,
                &transaction.borrow().transaction,
            )?;
            self.state.regenerate_proof(
                &rotxn,
                &unconfirmed,
                &mut transaction.borrow_mut().transaction,
            )?;
            self.state.validate_transaction(
                &rotxn,
                &self.batch_verification_ctxt,
                &unconfirmed,
                transaction.borrow(),
                &self.archive,
            )?;
            self.mempool.put(&mut rotxn, transaction.borrow())?;
            let () = self.update_mempool_market(
                &mut rotxn,
                &transaction.borrow().transaction,
            )?;
            rotxn.commit().map_err(RwTxnError::from)?;
        }
        self.net.push_tx(Default::default(), transaction.borrow());
        Ok(())
    }

    pub fn get_all_utxos(&self) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        self.state
            .get_utxos(&rotxn)
            .map_err(|err| DbError::from(err).into())
    }

    pub fn get_latest_failed_withdrawal_bundle_height(
        &self,
    ) -> Result<Option<u32>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let res = self
            .state
            .get_latest_failed_withdrawal_bundle(&rotxn)
            .map_err(DbError::from)?
            .map(|(height, _)| height);
        Ok(res)
    }

    pub fn get_unconfirmed_spent_utxos<'a, OutPoints>(
        &self,
        outpoints: OutPoints,
    ) -> Result<Vec<(OutPoint, InPoint)>, Error>
    where
        OutPoints: IntoIterator<Item = &'a OutPoint>,
    {
        let rotxn = self.env.read_txn()?;
        let mut spent = vec![];
        for outpoint in outpoints {
            let Some(txid) = self
                .mempool
                .spent_utxos
                .try_get(&rotxn, outpoint)
                .map_err(mempool::Error::from)?
            else {
                continue;
            };
            let tx = self
                .mempool
                .transactions
                .try_get(&rotxn, &txid)
                .map_err(mempool::Error::from)?
                .ok_or(mempool::Error::MissingTransaction(txid))?;
            if let Some(vin) = tx
                .transaction
                .inputs
                .iter()
                .position(|(spent_outpoint, _)| spent_outpoint == outpoint)
            {
                let inpoint = InPoint::Regular {
                    txid,
                    vin: vin as u32,
                };
                spent.push((*outpoint, inpoint));
            }
        }
        Ok(spent)
    }

    pub fn get_unconfirmed_utxos_by_addresses(
        &self,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn()?;
        let mut res = HashMap::new();
        let () = addresses.iter().try_for_each(|addr| {
            let utxos = self.mempool.get_unconfirmed_utxos(&rotxn, addr)?;
            res.extend(utxos);
            Result::<(), Error>::Ok(())
        })?;
        Ok(res)
    }

    pub fn get_spent_utxos(
        &self,
        outpoints: &[OutPoint],
    ) -> Result<Vec<(OutPoint, SpentOutput)>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let mut spent = vec![];
        for outpoint in outpoints {
            let key = OutPointKey::from(outpoint);
            if let Some(output) = self
                .state
                .stxos
                .try_get(&rotxn, &key)
                .map_err(DbError::from)?
            {
                spent.push((*outpoint, output));
            }
        }
        Ok(spent)
    }

    /// What the mempool means for a wallet: the unconfirmed outputs the
    /// wallet made on its own, and the confirmed outputs from `confirmed` that
    /// a mempool transaction already spends.
    pub fn get_mempool_view(
        &self,
        addresses: &HashSet<Address>,
        confirmed: &HashSet<OutPoint>,
    ) -> Result<(HashMap<OutPoint, Output>, HashSet<OutPoint>), Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let unconfirmed = self
            .mempool
            .own_unconfirmed_utxos(&rotxn, addresses, confirmed)?;
        let mut spent = HashSet::new();
        for outpoint in confirmed {
            if self.mempool.spender(&rotxn, outpoint)?.is_some() {
                spent.insert(*outpoint);
            }
        }
        Ok((unconfirmed, spent))
    }

    pub fn get_stxos_by_addresses(
        &self,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, SpentOutput>, Error> {
        let rotxn = self.env.read_txn()?;
        let stxos = self
            .state
            .get_stxos_by_addresses(&rotxn, addresses)
            .map_err(DbError::from)?;
        Ok(stxos)
    }

    pub fn get_utxos_by_addresses(
        &self,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let utxos = self
            .state
            .get_utxos_by_addresses(&rotxn, addresses)
            .map_err(DbError::from)?;
        Ok(utxos)
    }

    pub fn try_get_tip(&self) -> Result<Option<BlockHash>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let tip = self.state.try_get_tip(&rotxn)?;
        Ok(tip)
    }

    pub fn get_tip_accumulator(&self) -> Result<Accumulator, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.state.get_accumulator(&rotxn)?)
    }

    pub fn regenerate_proof(&self, tx: &mut Transaction) -> Result<(), Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let unconfirmed = self.mempool.unconfirmed_outputs(&rotxn, tx)?;
        let () = self.state.regenerate_proof(&rotxn, &unconfirmed, tx)?;
        Ok(())
    }

    pub fn try_get_accumulator(
        &self,
        block_hash: BlockHash,
    ) -> Result<Option<Accumulator>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.try_get_accumulator(&rotxn, block_hash)?)
    }

    pub fn get_accumulator(
        &self,
        block_hash: BlockHash,
    ) -> Result<Accumulator, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.get_accumulator(&rotxn, block_hash)?)
    }

    pub fn try_get_header(
        &self,
        block_hash: BlockHash,
    ) -> Result<Option<Header>, Error> {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.try_get_header(&txn, block_hash)?)
    }

    pub fn get_header(&self, block_hash: BlockHash) -> Result<Header, Error> {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.get_header(&txn, block_hash)?)
    }

    fn try_get_block_hash_at(
        &self,
        rotxn: &RoTxn,
        height: u32,
    ) -> Result<Option<BlockHash>, Error> {
        let Some(tip) = self.state.try_get_tip(rotxn)? else {
            return Ok(None);
        };
        let Some(tip_height) = self.state.try_get_height(rotxn)? else {
            return Ok(None);
        };
        if tip_height >= height {
            self.archive
                .ancestors(rotxn, tip)
                .nth((tip_height - height) as usize)
                .map_err(Error::from)
        } else {
            Ok(None)
        }
    }

    /// Get the block hash at the specified height in the active chain,
    /// if it exists
    pub fn try_get_block_hash(
        &self,
        height: u32,
    ) -> Result<Option<BlockHash>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        self.try_get_block_hash_at(&rotxn, height)
    }

    /// Get the coin movements that a block applied outside its body, in the
    /// order the node applied them
    pub fn get_two_way_peg_events(
        &self,
        block_hash: BlockHash,
    ) -> Result<Vec<TwoWayPegEvent>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let height = self.archive.get_height(&rotxn, block_hash)?;
        // The events are keyed by height, so a block off the active chain
        // would read another block's events.
        if self.try_get_block_hash_at(&rotxn, height)? != Some(block_hash) {
            return Err(Error::NotInActiveChain { block_hash });
        }
        Ok(self.state.get_two_way_peg_events(&rotxn, height)?)
    }

    pub fn try_get_body(
        &self,
        block_hash: BlockHash,
    ) -> Result<Option<Body>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.try_get_body(&rotxn, block_hash)?)
    }

    pub fn get_body(&self, block_hash: BlockHash) -> Result<Body, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.get_body(&rotxn, block_hash)?)
    }

    pub fn get_best_main_verification(
        &self,
        hash: BlockHash,
    ) -> Result<bitcoin::BlockHash, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let hash = self.archive.get_best_main_verification(&rotxn, hash)?;
        Ok(hash)
    }

    pub fn get_bmm_inclusions(
        &self,
        block_hash: BlockHash,
    ) -> Result<Vec<bitcoin::BlockHash>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let bmm_inclusions = self
            .archive
            .get_bmm_results(&rotxn, block_hash)?
            .into_iter()
            .filter_map(|(block_hash, bmm_res)| match bmm_res {
                BmmResult::Verified => Some(block_hash),
                BmmResult::Failed => None,
            })
            .collect();
        Ok(bmm_inclusions)
    }

    pub fn get_all_transactions(
        &self,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let transactions = self.mempool.take_all(&rotxn)?;
        Ok(transactions)
    }

    /// Get total sidechain wealth in Bitcoin
    pub fn get_sidechain_wealth(&self) -> Result<bitcoin::Amount, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.state.sidechain_wealth(&rotxn)?)
    }

    pub fn get_transactions(
        &self,
        number: usize,
    ) -> Result<(Vec<Authorized<FilledTransaction>>, bitcoin::Amount), Error>
    {
        let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
        // Take non-trade txs first, a parent before its child, then trade txs
        // in insertion order
        let transactions = self.mempool.topological(&rwtxn, Some(number))?;
        let trade_txs = self.mempool.take_trades_ordered(&rwtxn)?;
        let trade_txids: HashSet<_> =
            trade_txs.iter().map(|tx| tx.transaction.txid()).collect();
        let transactions: Vec<_> = transactions
            .into_iter()
            .filter(|tx| !trade_txids.contains(&tx.transaction.txid()))
            .chain(trade_txs)
            .take(number)
            .collect();
        let mut fee = bitcoin::Amount::ZERO;
        let mut returned_transactions = vec![];
        let mut spent_utxos = HashSet::new();
        let mut trades = TradeSimulation::default();
        // Outputs the transactions already taken for this block make.
        let mut block_outputs = HashMap::<OutPoint, Output>::new();
        for transaction in transactions {
            let inputs: HashSet<_> =
                transaction.transaction.inputs.iter().copied().collect();
            if !spent_utxos.is_disjoint(&inputs) {
                // UTXO double spent
                self.mempool
                    .delete(&mut rwtxn, transaction.transaction.txid())?;
                continue;
            }
            // A child whose parent this block does not carry waits for a
            // later block
            if !self
                .mempool
                .unconfirmed_outputs(&rwtxn, &transaction.transaction)?
                .keys()
                .all(|outpoint| block_outputs.contains_key(outpoint))
            {
                continue;
            }
            if self
                .state
                .validate_transaction(
                    &rwtxn,
                    &self.batch_verification_ctxt,
                    &block_outputs,
                    &transaction,
                    &self.archive,
                )
                .is_err()
            {
                self.mempool
                    .delete(&mut rwtxn, transaction.transaction.txid())?;
                continue;
            }
            let filled_transaction = self.state.fill_authorized_transaction(
                &rwtxn,
                &block_outputs,
                transaction,
            )?;
            match trades.apply(
                &self.state,
                &self.archive,
                &rwtxn,
                &filled_transaction.transaction,
            )? {
                TradeApplyResult::Applied => (),
                TradeApplyResult::Skipped { reason } => {
                    tracing::debug!(
                        txid = %filled_transaction.transaction.txid(),
                        %reason,
                        "Skip trade for this block"
                    );
                    continue;
                }
            }
            block_outputs.extend(
                filled_transaction
                    .transaction
                    .transaction
                    .outputs_by_outpoint(),
            );
            fee = fee
                .checked_add(crate::validation::miner_fee(
                    &filled_transaction.transaction,
                )?)
                .ok_or(AmountUnderflowError)?;
            spent_utxos.extend(
                filled_transaction
                    .transaction
                    .transaction
                    .inputs
                    .iter()
                    .cloned(),
            );
            returned_transactions.push(filled_transaction);
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok((returned_transactions, fee))
    }

    /// Get a transaction if it exists in the active chain or mempool.
    /// Returns the transaction and the block it was included in, if it exists
    /// in the active chain.
    pub fn try_get_transaction(
        &self,
        txid: Txid,
    ) -> Result<Option<(Transaction, Option<BlockHash>)>, Error> {
        let rotxn = self.env.read_txn()?;
        let tip = self.state.try_get_tip(&rotxn)?;
        if let Some(tip) = tip
            && let Some((block_hash, txin)) = self
                .archive
                .get_tx_inclusions(&rotxn, txid)?
                .into_iter()
                .map(Ok)
                .transpose_into_fallible()
                .find(|(block_hash, _idx)| {
                    self.archive.is_descendant(&rotxn, *block_hash, tip)
                })?
        {
            let body = self.archive.get_body(&rotxn, block_hash)?;
            let tx = body.transactions.into_iter().nth(txin as usize).unwrap();
            Ok(Some((tx, Some(block_hash))))
        } else if let Some(auth_tx) = self
            .mempool
            .transactions
            .try_get(&rotxn, &txid)
            .map_err(mempool::Error::from)?
        {
            Ok(Some((auth_tx.transaction, None)))
        } else {
            Ok(None)
        }
    }

    pub fn try_get_withdrawal_bundle(
        &self,
        m6id: &M6id,
    ) -> Result<Option<(WithdrawalBundleInfo, WithdrawalBundleStatus)>, Error>
    {
        let rotxn = self.env.read_txn()?;
        let res = self
            .state
            .try_get_withdrawal_bundle(&rotxn, m6id)
            .map_err(state::Error::from)?;
        Ok(res)
    }

    pub fn try_get_filled_transaction(
        &self,
        txid: Txid,
    ) -> Result<Option<FilledTransactionWithPosition>, Error> {
        let rotxn = self.env.read_txn()?;
        let tip = self.state.try_get_tip(&rotxn)?;
        let inclusions = self.archive.get_tx_inclusions(&rotxn, txid)?;
        if let Some((block_hash, idx)) = inclusions
            .into_iter()
            .map(Ok)
            .transpose_into_fallible()
            .find(|(block_hash, _)| {
                if let Some(tip) = tip {
                    self.archive.is_descendant(&rotxn, *block_hash, tip)
                } else {
                    Ok(true)
                }
            })?
        {
            let body = self.archive.get_body(&rotxn, block_hash)?;
            let auth_txs = body.authorized_transactions();
            let auth_tx =
                auth_txs.into_iter().nth(idx as usize).ok_or_else(|| {
                    Error::State(Box::new(state::Error::InvalidTransaction {
                        reason: format!(
                            "tx index {idx} out of bounds in \
                                 block {block_hash}"
                        ),
                    }))
                })?;
            let filled_tx = self
                .state
                .fill_transaction_from_stxos(&rotxn, auth_tx.transaction)?;
            let auth_tx = Authorized {
                transaction: filled_tx,
                authorizations: auth_tx.authorizations,
                actor_proof: auth_tx.actor_proof,
            };
            let txin = TxIn { block_hash, idx };
            let res = (auth_tx, Some(txin));
            return Ok(Some(res));
        }
        if let Some(auth_tx) = self
            .mempool
            .transactions
            .try_get(&rotxn, &txid)
            .map_err(mempool::Error::from)?
        {
            let unconfirmed = self
                .mempool
                .unconfirmed_outputs(&rotxn, &auth_tx.transaction)?;
            match self.state.fill_authorized_transaction(
                &rotxn,
                &unconfirmed,
                auth_tx,
            ) {
                Ok(filled_tx) => {
                    let res = (filled_tx, None);
                    Ok(Some(res))
                }
                Err(state::Error::NoUtxo { .. }) => Ok(None),
                Err(err) => Err(err.into()),
            }
        } else {
            Ok(None)
        }
    }

    pub fn try_get_pending_withdrawal_bundle(
        &self,
    ) -> Result<Option<WithdrawalBundle>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let bundle = self
            .state
            .try_get_pending_withdrawal_bundle(&rotxn)?
            .map(|(bundle, _)| bundle);
        Ok(bundle)
    }

    pub fn remove_from_mempool(&self, txid: Txid) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
        let () = self.mempool.delete(&mut rwtxn, txid)?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn connect_peer(&self, addr: ResolvedPeerAddress) -> Result<(), Error> {
        let peer_addr = addr.as_peer_address().to_owned();
        let () =
            self.net
                .connect_peer(self.env.clone(), addr)
                .map_err(|err| crate::net::Error::ConnectPeer {
                    peer_addr,
                    source: err,
                })?;
        Ok(())
    }

    pub fn forget_peer(&self, addr: &PeerAddress) -> Result<bool, Error> {
        let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
        let res = self.net.forget_peer(&mut rwtxn, addr)?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(res)
    }

    pub fn get_active_peers(&self) -> Vec<Peer> {
        self.net.get_active_peers()
    }

    /// Get the progress of the startup sync with the mainchain
    pub fn mainchain_sync_progress(&self) -> MainchainSyncProgress {
        self.task_handles.mainchain.sync_progress()
    }

    pub async fn request_mainchain_ancestor_infos(
        &self,
        block_hash: bitcoin::BlockHash,
    ) -> Result<bool, Error> {
        let mainchain_task::Response::AncestorInfos(_, res): mainchain_task::Response = self
            .task_handles
            .mainchain
            .request_oneshot(mainchain_task::Request::AncestorInfos(
                block_hash,
            ))
            .map_err(|_| Error::SendMainchainTaskRequest)?
            .await
            .map_err(|_| Error::ReceiveMainchainTaskResponse)?;
        res.map_err(Error::MainchainAncestors)
    }

    /// Attempt to submit a block.
    /// Returns `Ok(true)` if the block was accepted successfully as the new tip.
    /// Returns `Ok(false)` if the block could not be submitted for some reason,
    /// or was rejected as the new tip.
    pub async fn submit_block(
        &self,
        main_block_hash: bitcoin::BlockHash,
        header: &Header,
        body: &Body,
    ) -> Result<bool, Error> {
        let block_hash = header.hash();
        // Store the header, if ancestors exist
        if let Some(parent) = header.prev_side_hash
            && self.try_get_header(parent)?.is_none()
        {
            tracing::error!(%block_hash,
                "Rejecting block {block_hash} due to missing ancestor headers",
            );
            return Ok(false);
        }
        // Request mainchain header/infos if they do not exist
        let mainchain_task::Response::AncestorInfos(_, res): mainchain_task::Response = self
            .task_handles
            .mainchain
            .request_oneshot(mainchain_task::Request::AncestorInfos(
                main_block_hash,
            ))
            .map_err(|_| Error::SendMainchainTaskRequest)?
            .await
            .map_err(|_| Error::ReceiveMainchainTaskResponse)?;
        if !res.map_err(Error::MainchainAncestors)? {
            return Ok(false);
        };
        // Write header
        tracing::trace!("Storing header: {block_hash}");
        {
            let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
            let () = self.archive.put_header(&mut rwtxn, header)?;
            rwtxn.commit().map_err(RwTxnError::from)?;
        }
        tracing::trace!("Stored header: {block_hash}");
        // Check BMM
        {
            let rotxn = self.env.read_txn().map_err(EnvError::from)?;
            match self.archive.get_bmm_result(
                &rotxn,
                block_hash,
                main_block_hash,
            )? {
                BmmResult::Verified => (),
                BmmResult::Failed => {
                    tracing::error!(%block_hash,
                        "Rejecting block {block_hash} due to failing BMM verification",
                    );
                    return Ok(false);
                }
            }
        }
        // Check that ancestor bodies exist, and store body
        {
            let rotxn = self.env.read_txn().map_err(EnvError::from)?;
            let tip = self.state.try_get_tip(&rotxn)?;
            let common_ancestor = if let Some(tip) = tip {
                self.archive.last_common_ancestor(&rotxn, tip, block_hash)?
            } else {
                None
            };
            let missing_bodies = self.archive.get_missing_bodies(
                &rotxn,
                block_hash,
                common_ancestor,
            )?;
            if !(missing_bodies.is_empty()
                || missing_bodies == vec![block_hash])
            {
                tracing::error!(%block_hash,
                    "Rejecting block {block_hash} due to missing ancestor bodies",
                );
                return Ok(false);
            }
            drop(rotxn);
            if missing_bodies == vec![block_hash] {
                let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
                let () = self.archive.put_body(&mut rwtxn, block_hash, body)?;
                rwtxn.commit().map_err(RwTxnError::from)?;
            }
        }
        // Submit new tip
        let new_tip = Tip {
            block_hash,
            main_block_hash,
        };
        if !self.task_handles.net.new_tip_ready_confirm(new_tip).await? {
            tracing::warn!(%block_hash, "Not ready to reorg");
            return Ok(false);
        };
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let bundle = self.state.try_get_pending_withdrawal_bundle(&rotxn)?;
        if let Some((bundle, _)) = bundle {
            let m6id = bundle.compute_m6id();
            if let Some(cusf_mainchain_block_producer) =
                self.cusf_mainchain_block_producer.as_ref()
            {
                {
                    let mut cusf_mainchain_block_producer_lock =
                        cusf_mainchain_block_producer.lock().await;
                    let () = cusf_mainchain_block_producer_lock
                        .propose_withdrawal_bundle(bundle.tx())
                        .await?;
                }
                tracing::trace!(%m6id, "Proposed withdrawal bundle");
            } else {
                // Without the mainchain's block producer service there is
                // nowhere to send the bundle, and it stays pending for every
                // block that follows. Say so, or the withdrawal simply never
                // completes and nothing explains why.
                tracing::warn!(
                    %m6id,
                    "Withdrawal bundle is pending, but the mainchain node \
                     does not serve BlockProducerService, so the bundle \
                     cannot be proposed and the withdrawal cannot complete",
                );
            }
        }
        Ok(true)
    }

    /// Get a notification whenever the tip changes
    pub fn watch_state(&self) -> impl Stream<Item = ()> {
        self.state.watch()
    }

    /// Get a notification whenever the tip or the mempool changes
    pub fn watch(&self) -> std::pin::Pin<Box<dyn Stream<Item = ()> + Send>> {
        Box::pin(futures::stream::select(
            self.state.watch(),
            self.mempool.watch(),
        ))
    }

    pub fn get_height(&self, block_hash: BlockHash) -> Result<u32, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.get_height(&rotxn, block_hash)?)
    }

    /// Get UTXOs for addresses along with their mempool spent status.
    /// This is atomic - both queries use the same read transaction.
    #[allow(clippy::type_complexity)]
    pub fn get_utxos_with_mempool_status(
        &self,
        addresses: &HashSet<Address>,
    ) -> Result<(HashMap<OutPoint, Output>, Vec<(OutPoint, Txid)>), Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;

        // Get confirmed UTXOs from state
        let utxos = self
            .state
            .get_utxos_by_addresses(&rotxn, addresses)
            .map_err(DbError::from)?;

        // Check which are spent in mempool (same transaction - atomic)
        let mut spent_in_mempool = vec![];
        for outpoint in utxos.keys() {
            if let Some(txid) = self
                .mempool
                .spent_utxos
                .try_get(&rotxn, outpoint)
                .map_err(mempool::Error::from)?
            {
                spent_in_mempool.push((*outpoint, txid));
            }
        }

        Ok((utxos, spent_in_mempool))
    }

    pub fn read_txn(
        &self,
    ) -> Result<sneed::RoTxn<'_, heed::WithoutTls>, Error> {
        self.env.read_txn().map_err(Into::into)
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn get_mempool_shares(
        &self,
        market_id: &MarketId,
    ) -> Result<Option<ndarray::Array1<i64>>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.state.get_mempool_shares(&rotxn, market_id)?)
    }

    pub fn get_pending_decision_claim_ids(
        &self,
    ) -> Result<std::collections::BTreeSet<[u8; 3]>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.mempool.pending_decision_claim_ids(&rotxn)?)
    }

    /// Track the shares that a mempool trade moves
    fn update_mempool_market(
        &self,
        rwtxn: &mut RwTxn,
        transaction: &Transaction,
    ) -> Result<(), Error> {
        if let Some(crate::types::TxData::Trade {
            market_id,
            outcome_index,
            shares,
            ..
        }) = transaction.data.as_ref()
        {
            if *shares > 0 {
                self.update_mempool_buy(
                    rwtxn,
                    *market_id,
                    *outcome_index,
                    *shares,
                )?;
            } else {
                self.update_mempool_sell(
                    rwtxn,
                    *market_id,
                    *outcome_index,
                    shares.unsigned_abs() as i64,
                )?;
            }
        }
        Ok(())
    }

    fn update_mempool_buy(
        &self,
        rwtxn: &mut RwTxn,
        market_id: MarketId,
        outcome_index: u32,
        shares_to_buy: i64,
    ) -> Result<(), Error> {
        use crate::state;

        let market = self
            .state
            .markets()
            .get_market(rwtxn, &market_id)?
            .ok_or_else(|| {
                Error::State(Box::new(state::Error::InvalidDecisionId {
                    reason: format!("Market {market_id:?} does not exist"),
                }))
            })?;

        if market.state() != crate::state::markets::MarketState::Trading {
            return Ok(());
        }

        if outcome_index as usize >= market.shares().len() {
            return Err(Error::State(Box::new(
                state::Error::InvalidDecisionId {
                    reason: format!(
                        "Outcome index {} exceeds market outcomes {}",
                        outcome_index,
                        market.shares().len()
                    ),
                },
            )));
        }

        let current_shares = if let Some(existing_mempool_shares) =
            self.state.get_mempool_shares(rwtxn, &market_id)?
        {
            existing_mempool_shares
        } else {
            market.shares().clone()
        };

        let mut new_shares = current_shares.clone();
        new_shares[outcome_index as usize] += shares_to_buy;

        let beta = self.derive_market_beta(&market)?;
        trading::validate_lmsr_parameters(beta, &new_shares).map_err(|e| {
            Error::State(Box::new(state::Error::InvalidDecisionId {
                reason: format!(
                    "Invalid LMSR state after mempool update: {e:?}"
                ),
            }))
        })?;

        self.state
            .put_mempool_shares(rwtxn, &market_id, &new_shares)
            .map_err(|e| Error::State(Box::new(e)))?;

        tracing::debug!(
            "Updated mempool shares for market {}: outcome {} increased by {} shares",
            const_hex::encode(market_id),
            outcome_index,
            shares_to_buy
        );

        Ok(())
    }

    fn update_mempool_sell(
        &self,
        rwtxn: &mut RwTxn,
        market_id: MarketId,
        outcome_index: u32,
        shares_to_sell: i64,
    ) -> Result<(), Error> {
        use crate::state;

        let market = self
            .state
            .markets()
            .get_market(rwtxn, &market_id)?
            .ok_or_else(|| {
                Error::State(Box::new(state::Error::InvalidDecisionId {
                    reason: format!("Market {market_id:?} does not exist"),
                }))
            })?;

        if market.state() != crate::state::markets::MarketState::Trading {
            return Ok(());
        }

        if outcome_index as usize >= market.shares().len() {
            return Err(Error::State(Box::new(
                state::Error::InvalidDecisionId {
                    reason: format!(
                        "Outcome index {} exceeds market outcomes {}",
                        outcome_index,
                        market.shares().len()
                    ),
                },
            )));
        }

        let current_shares = if let Some(existing_mempool_shares) =
            self.state.get_mempool_shares(rwtxn, &market_id)?
        {
            existing_mempool_shares
        } else {
            market.shares().clone()
        };

        let mut new_shares = current_shares.clone();
        new_shares[outcome_index as usize] -= shares_to_sell;

        let beta = self.derive_market_beta(&market)?;
        trading::validate_lmsr_parameters(beta, &new_shares).map_err(|e| {
            Error::State(Box::new(state::Error::InvalidDecisionId {
                reason: format!(
                    "Invalid LMSR state after mempool sell update: {e:?}"
                ),
            }))
        })?;

        self.state
            .put_mempool_shares(rwtxn, &market_id, &new_shares)
            .map_err(|e| Error::State(Box::new(e)))?;

        tracing::debug!(
            "Updated mempool shares for market {}: outcome {} decreased by {} shares (sell)",
            const_hex::encode(market_id),
            outcome_index,
            shares_to_sell
        );

        Ok(())
    }

    /// Returns the confirmed market treasury value in sats.
    pub fn get_effective_market_treasury_sats(
        &self,
        market_id: &crate::state::markets::MarketId,
    ) -> Result<u64, Error> {
        let rotxn = self.env.read_txn()?;
        self.state
            .markets()
            .get_market_funds_sats(&rotxn, &self.state, market_id, false)
            .map_err(|e| Error::State(Box::new(e)))
    }

    /// Derive the current effective LMSR beta for a market.
    /// `beta = liquidity_base_sats / ln(num_outcomes)`, i.e. the creation seed
    /// plus confirmed `AmplifyBeta` deposits (trade proceeds excluded). Pending
    /// mempool deposits are not counted until they confirm.
    fn derive_market_beta(
        &self,
        market: &crate::state::Market,
    ) -> Result<f64, Error> {
        Ok(trading::derive_beta_from_liquidity(
            market.liquidity_base_sats,
            market.shares().len(),
        ))
    }

    pub fn get_market_beta(
        &self,
        market: &crate::state::Market,
    ) -> Result<f64, Error> {
        self.derive_market_beta(market)
    }

    pub fn get_all_decision_periods(&self) -> Result<Vec<(u32, u64)>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.get_all_decision_periods(&rotxn)?)
    }

    pub fn get_decisions_for_period(&self, period: u32) -> Result<u64, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.get_decisions_for_period(&rotxn, period)?)
    }

    pub fn get_genesis_timestamp(&self) -> Result<Option<u64>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.try_get_genesis_timestamp(&rotxn)?)
    }

    pub fn get_mainchain_timestamp(&self) -> Result<u64, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.try_get_mainchain_timestamp(&rotxn)?.unwrap_or(0))
    }

    pub fn get_decision_entry(
        &self,
        decision_id: crate::state::decisions::DecisionId,
    ) -> Result<Option<crate::state::decisions::DecisionEntry>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .decisions()
            .get_decision_entry(&rotxn, decision_id)?)
    }

    pub fn get_available_decisions_in_period(
        &self,
        period_id: crate::state::voting::types::VotingPeriodId,
    ) -> Result<Vec<crate::state::decisions::DecisionId>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .get_available_decisions_in_period(&rotxn, period_id.as_u32())?)
    }

    pub fn get_claimed_decisions_in_period(
        &self,
        period_id: crate::state::voting::types::VotingPeriodId,
    ) -> Result<Vec<crate::state::decisions::DecisionEntry>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .decisions()
            .get_claimed_decisions_in_period(&rotxn, period_id.as_u32())?)
    }

    pub fn is_decision_in_voting(
        &self,
        decision_id: crate::state::decisions::DecisionId,
    ) -> Result<bool, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.is_decision_in_voting(&rotxn, decision_id)?)
    }

    pub fn get_settled_decisions(
        &self,
    ) -> Result<Vec<crate::state::decisions::DecisionEntry>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.get_settled_decisions(&rotxn)?)
    }

    pub fn get_voting_periods(&self) -> Result<Vec<(u32, u64, u64)>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.get_voting_periods(&rotxn)?)
    }

    pub fn get_period_summary(
        &self,
    ) -> Result<crate::state::type_aliases::PeriodSummary, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.get_period_summary(&rotxn)?)
    }

    pub fn claimed_count_in_period(
        &self,
        period_id: crate::state::voting::types::VotingPeriodId,
    ) -> Result<u64, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .claimed_count_in_period(&rotxn, period_id.as_u32())?)
    }

    pub fn get_listing_fee_info(
        &self,
        period: u32,
    ) -> Result<Option<crate::state::decisions::PeriodPricing>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .decisions()
            .get_listing_fee_info(&rotxn, period)?)
    }

    pub fn fee_for_decision_id(
        &self,
        decision_id: crate::state::decisions::DecisionId,
    ) -> Result<u64, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .decisions()
            .fee_for_decision_id(&rotxn, decision_id)?)
    }

    pub fn is_decisions_testing_mode(&self) -> bool {
        self.state.decisions().is_testing_mode()
    }

    pub fn get_decisions_testing_config(&self) -> u32 {
        self.state.decisions().get_testing_blocks_per_period()
    }

    pub fn get_decision_config(
        &self,
    ) -> &crate::state::decisions::DecisionConfig {
        self.state.decisions().get_config()
    }

    pub fn get_current_period(&self) -> Result<u32, Error> {
        let rotxn = self.env.read_txn()?;
        let block_height = self.state.try_get_height(&rotxn)?;
        let genesis_ts =
            self.state.try_get_genesis_timestamp(&rotxn)?.unwrap_or(0);
        let mainchain_ts =
            self.state.try_get_mainchain_timestamp(&rotxn)?.unwrap_or(0);
        Ok(self.state.decisions().get_current_period(
            mainchain_ts,
            block_height,
            genesis_ts,
        )?)
    }

    pub fn get_decisions_db(&self) -> &crate::state::decisions::Dbs {
        self.state.decisions()
    }

    pub fn block_height_to_testing_period(&self, block_height: u32) -> u32 {
        self.state
            .decisions()
            .block_height_to_testing_period(block_height)
    }

    pub fn get_all_markets(&self) -> Result<Vec<crate::state::Market>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.markets().get_all_markets(&rotxn)?)
    }

    pub fn get_all_markets_with_states(
        &self,
    ) -> Result<Vec<(crate::state::Market, crate::state::MarketState)>, Error>
    {
        let rotxn = self.env.read_txn()?;
        let markets = self.state.markets().get_all_markets(&rotxn)?;
        let result = markets
            .into_iter()
            .map(|market| {
                let state = market.state();
                (market, state)
            })
            .collect();
        Ok(result)
    }

    pub fn get_markets_by_state(
        &self,
        state: crate::state::MarketState,
    ) -> Result<Vec<crate::state::Market>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.markets().get_markets_by_state(&rotxn, state)?)
    }

    pub fn get_market_by_id(
        &self,
        market_id: &crate::state::MarketId,
    ) -> Result<Option<crate::state::Market>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.markets().get_market(&rotxn, market_id)?)
    }

    pub fn get_market_by_id_with_state(
        &self,
        market_id: &crate::state::MarketId,
    ) -> Result<Option<(crate::state::Market, crate::state::MarketState)>, Error>
    {
        let rotxn = self.env.read_txn()?;
        if let Some(market) =
            self.state.markets().get_market(&rotxn, market_id)?
        {
            let state = market.state();
            Ok(Some((market, state)))
        } else {
            Ok(None)
        }
    }

    pub fn try_get_market_price_history(
        &self,
        market_id: &MarketId,
    ) -> Result<
        Option<Vec<state::markets::price_history::MarketPricePoint>>,
        Error,
    > {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(state::markets::price_history::try_get_market_price_history(
            &self.state,
            &self.archive,
            &rotxn,
            market_id,
        )?)
    }

    pub fn get_markets_batch(
        &self,
        market_ids: &[crate::state::MarketId],
    ) -> Result<
        std::collections::HashMap<crate::state::MarketId, crate::state::Market>,
        Error,
    > {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.markets().get_markets_batch(&rotxn, market_ids)?)
    }

    pub fn get_market_decisions(
        &self,
        market: &crate::state::Market,
    ) -> Result<
        std::collections::HashMap<
            crate::state::decisions::DecisionId,
            crate::state::decisions::Decision,
        >,
        Error,
    > {
        let rotxn = self.env.read_txn()?;
        let mut decisions = std::collections::HashMap::new();

        for &decision_id in &market.decision_ids {
            if let Some(entry) = self
                .state
                .decisions()
                .get_decision_entry(&rotxn, decision_id)?
                && let Some(decision) = entry.decision
            {
                decisions.insert(decision_id, decision);
            }
        }

        Ok(decisions)
    }

    pub fn get_user_share_positions(
        &self,
        address: &crate::types::Address,
    ) -> Result<Vec<(crate::state::MarketId, u32, i64)>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .markets()
            .get_user_share_positions(&rotxn, address)?)
    }

    pub fn get_market_user_positions(
        &self,
        address: &crate::types::Address,
        market_id: &crate::state::MarketId,
    ) -> Result<Vec<(u32, i64)>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .markets()
            .get_market_user_positions(&rotxn, address, market_id)?)
    }

    /// Get share positions for multiple addresses for a specific market/outcome.
    /// Returns a map of address -> shares for addresses that have positions.
    pub fn get_wallet_positions_for_market_outcome(
        &self,
        addresses: &std::collections::HashSet<crate::types::Address>,
        market_id: &crate::state::MarketId,
        outcome_index: u32,
    ) -> Result<std::collections::HashMap<crate::types::Address, i64>, Error>
    {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .markets()
            .get_wallet_positions_for_market_outcome(
                &rotxn,
                addresses,
                market_id,
                outcome_index,
            )?)
    }

    pub fn get_all_share_accounts(
        &self,
    ) -> Result<crate::state::type_aliases::AllShareAccounts, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.markets().get_all_share_accounts(&rotxn)?)
    }

    pub fn get_market_treasury_sats(
        &self,
        market_id: &crate::state::MarketId,
    ) -> Result<u64, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.markets().get_market_funds_sats(
            &rotxn,
            &self.state,
            market_id,
            false,
        )?)
    }

    pub fn voting_state(&self) -> &crate::state::voting::VotingSystem {
        self.state.voting()
    }

    pub fn reputation(&self) -> &crate::state::reputation::ReputationDbs {
        self.state.reputation()
    }

    pub fn get_last_block_timestamp(&self) -> Result<u64, Error> {
        self.get_mainchain_timestamp()
    }

    pub fn resolve_voting_period(
        &self,
        period_id: crate::state::voting::types::VotingPeriodId,
    ) -> Result<Vec<crate::state::voting::types::DecisionOutcome>, Error> {
        let rotxn = self.env.read_txn()?;
        let outcomes = self
            .state
            .voting()
            .resolve_period_decisions(&rotxn, period_id)?;
        Ok(outcomes)
    }

    pub fn get_consensus_outcomes(
        &self,
        period_id: crate::state::voting::types::VotingPeriodId,
    ) -> Result<
        std::collections::HashMap<crate::state::decisions::DecisionId, f64>,
        Error,
    > {
        let rotxn = self.env.read_txn()?;
        self.state
            .voting()
            .databases()
            .get_consensus_outcomes_for_period(&rotxn, period_id)
            .map_err(Into::into)
    }

    /// Trigger a sync/reorg to a specific block hash.
    /// The block must exist in the archive (received via P2P or locally mined).
    /// Returns true if reorg was successful, false if not needed or failed.
    pub async fn sync_to_tip(
        &self,
        block_hash: crate::types::BlockHash,
    ) -> Result<bool, Error> {
        // Get the new tip info synchronously, then await the reorg
        let new_tip = {
            let rotxn = self.env.read_txn()?;

            let main_block_hash = self
                .archive
                .get_best_main_verification(&rotxn, block_hash)?;

            Tip {
                block_hash,
                main_block_hash,
            }
        }; // rotxn is dropped here before the await

        // Trigger the reorg via net_task
        self.task_handles
            .net
            .new_tip_ready_confirm(new_tip)
            .await
            .map_err(Error::from)
    }
}
