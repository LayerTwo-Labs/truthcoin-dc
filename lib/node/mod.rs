use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Debug,
    net::SocketAddr,
    path::Path,
    sync::Arc,
};

use bitcoin::amount::CheckedSum as _;
use fallible_iterator::FallibleIterator;
use futures::future::BoxFuture;
use heed::EnvFlags;
use sneed::{DbError, Env, EnvError, RwTxnError, env};
use tokio::sync::Mutex;
use tonic::transport::Channel;

use ndarray::Array1;

use crate::{
    archive::{self, Archive},
    math::trading,
    mempool::{self, MemPool},
    net::{self, Net, Peer},
    state::{self, State, markets::MarketId},
    types::{
        Address, AmountOverflowError, AmountUnderflowError, Authorized,
        AuthorizedTransaction, Block, BlockHash, BmmResult, Body, FilledOutput,
        FilledTransaction, GetBitcoinValue, Header, InPoint, Network, OutPoint,
        OutPointKey, Output, SpentOutput, Tip, Transaction, TxData, TxIn, Txid,
        WithdrawalBundle,
        proto::{self, mainchain},
    },
    util::Watchable,
};
use sneed::RwTxn;

mod mainchain_task;
mod net_task;

use mainchain_task::MainchainTaskHandle;
use net_task::NetTaskHandle;
#[cfg(feature = "zmq")]
use net_task::ZmqPubHandler;

#[allow(clippy::duplicated_attributes)]
#[derive(thiserror::Error, transitive::Transitive, Debug)]
#[transitive(from(env::error::OpenEnv, EnvError))]
#[transitive(from(env::error::ReadTxn, EnvError))]
#[transitive(from(env::error::WriteTxn, EnvError))]
pub enum Error {
    #[error("address parse error")]
    AddrParse(#[from] std::net::AddrParseError),
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error("archive error")]
    Archive(#[from] archive::Error),
    #[error("CUSF mainchain proto error")]
    CusfMainchain(#[from] proto::Error),
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] EnvError),
    #[error("Database write error")]
    DbWrite(#[from] RwTxnError),
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error("error requesting mainchain ancestors")]
    MainchainAncestors(#[source] mainchain_task::ResponseError),
    #[error("malformed body")]
    MalformedBody(#[from] crate::types::MalformedBodyError),
    #[error("mempool error")]
    MemPool(#[from] mempool::Error),
    #[error("net error")]
    Net(#[source] Box<net::Error>),
    #[error("net task error")]
    NetTask(#[source] Box<net_task::Error>),
    #[error("No CUSF mainchain wallet client")]
    NoCusfMainchainWalletClient,
    #[error("peer info stream closed")]
    PeerInfoRxClosed,
    #[error("Receive mainchain task response cancelled")]
    ReceiveMainchainTaskResponse,
    #[error("Send mainchain task request failed")]
    SendMainchainTaskRequest,
    #[error("state error: {0}")]
    State(#[source] Box<state::Error>),
    #[error("Utreexo error: {0}")]
    Utreexo(String),
    #[error("Verify BMM error")]
    VerifyBmm(anyhow::Error),
    #[cfg(feature = "zmq")]
    #[error("ZMQ error")]
    Zmq(#[from] zeromq::ZmqError),
}

impl From<net::Error> for Error {
    fn from(err: net::Error) -> Self {
        Self::Net(Box::new(err))
    }
}

impl From<net_task::Error> for Error {
    fn from(err: net_task::Error) -> Self {
        Self::NetTask(Box::new(err))
    }
}

impl From<state::Error> for Error {
    fn from(err: state::Error) -> Self {
        Self::State(Box::new(err))
    }
}

pub type FilledTransactionWithPosition =
    (Authorized<FilledTransaction>, Option<TxIn>);

#[derive(Clone)]
pub struct Node<MainchainTransport = Channel> {
    archive: Archive,
    cusf_mainchain: Arc<Mutex<mainchain::ValidatorClient<MainchainTransport>>>,
    cusf_mainchain_wallet:
        Option<Arc<Mutex<mainchain::WalletClient<MainchainTransport>>>>,
    env: sneed::Env,
    mainchain_task: MainchainTaskHandle,
    mempool: MemPool,
    net: Net,
    net_task: NetTaskHandle,
    state: State,
    #[cfg(feature = "zmq")]
    zmq_pub_handler: Arc<ZmqPubHandler>,
}

impl<MainchainTransport> Node<MainchainTransport>
where
    MainchainTransport: proto::Transport,
{
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        bind_addr: SocketAddr,
        datadir: &Path,
        network: Network,
        cusf_mainchain: mainchain::ValidatorClient<MainchainTransport>,
        cusf_mainchain_wallet: Option<
            mainchain::WalletClient<MainchainTransport>,
        >,
        runtime: &tokio::runtime::Runtime,
        decision_config_testing: Option<u32>,
        #[cfg(feature = "zmq")] zmq_addr: SocketAddr,
    ) -> Result<Self, Error>
    where
        mainchain::ValidatorClient<MainchainTransport>: Clone,
        MainchainTransport: Send + 'static,
        <MainchainTransport as tonic::client::GrpcService<
            tonic::body::Body,
        >>::Future: Send,
    {
        let env_path = datadir.join("data.mdb");
        std::fs::create_dir_all(&env_path)?;
        let env = {
            let mut env_open_opts = heed::EnvOpenOptions::new();
            env_open_opts
                .map_size(128 * 1024 * 1024 * 1024) // 128 GB
                .max_dbs(
                    State::NUM_DBS
                        + Archive::NUM_DBS
                        + MemPool::NUM_DBS
                        + Net::NUM_DBS,
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
            let fast_flags = EnvFlags::WRITE_MAP
                | EnvFlags::MAP_ASYNC
                | EnvFlags::NO_SYNC
                | EnvFlags::NO_META_SYNC
                | EnvFlags::NO_READ_AHEAD
                | EnvFlags::NO_TLS;
            unsafe { env_open_opts.flags(fast_flags) };
            unsafe { Env::open(&env_open_opts, &env_path) }?
        };
        let state = State::new(&env, decision_config_testing)?;
        #[cfg(feature = "zmq")]
        let zmq_pub_handler = Arc::new(ZmqPubHandler::new(zmq_addr).await?);
        let archive = Archive::new(&env)?;
        let mempool = MemPool::new(&env)?;
        let (mainchain_task, mainchain_task_response_rx) =
            MainchainTaskHandle::new(
                env.clone(),
                archive.clone(),
                cusf_mainchain.clone(),
            );
        let (net, peer_info_rx) =
            Net::new(&env, archive.clone(), network, state.clone(), bind_addr)?;
        let cusf_mainchain_wallet =
            cusf_mainchain_wallet.map(|wallet| Arc::new(Mutex::new(wallet)));
        let net_task = NetTaskHandle::new(
            runtime,
            env.clone(),
            archive.clone(),
            mainchain_task.clone(),
            mainchain_task_response_rx,
            mempool.clone(),
            net.clone(),
            peer_info_rx,
            state.clone(),
            #[cfg(feature = "zmq")]
            zmq_pub_handler.clone(),
        );
        Ok(Self {
            archive,
            cusf_mainchain: Arc::new(Mutex::new(cusf_mainchain)),
            cusf_mainchain_wallet,
            env,
            mainchain_task,
            mempool,
            net,
            net_task,
            state,
            #[cfg(feature = "zmq")]
            zmq_pub_handler: zmq_pub_handler.clone(),
        })
    }

    pub fn env(&self) -> &Env {
        &self.env
    }

    pub fn archive(&self) -> &Archive {
        &self.archive
    }

    /// Borrow the CUSF mainchain client, and execute the provided future.
    /// The CUSF mainchain client will be locked while the future is running.
    pub async fn with_cusf_mainchain<F, Output>(&self, f: F) -> Output
    where
        F: for<'cusf_mainchain> FnOnce(
            &'cusf_mainchain mut mainchain::ValidatorClient<MainchainTransport>,
        )
            -> BoxFuture<'cusf_mainchain, Output>,
    {
        let mut cusf_mainchain_lock = self.cusf_mainchain.lock().await;
        let res = f(&mut cusf_mainchain_lock).await;
        drop(cusf_mainchain_lock);
        res
    }

    pub fn try_get_tip_height(&self) -> Result<Option<u32>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.try_get_height(&rotxn)?)
    }

    pub fn try_get_tip(&self) -> Result<Option<BlockHash>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.try_get_tip(&rotxn)?)
    }

    pub fn try_get_height(
        &self,
        block_hash: BlockHash,
    ) -> Result<Option<u32>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.archive.try_get_height(&rotxn, block_hash)?)
    }

    pub fn get_height(&self, block_hash: BlockHash) -> Result<u32, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.archive.get_height(&rotxn, block_hash)?)
    }

    pub fn get_tx_inclusions(
        &self,
        txid: Txid,
    ) -> Result<BTreeMap<BlockHash, u32>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.archive.get_tx_inclusions(&rotxn, txid)?)
    }

    pub fn is_descendant(
        &self,
        ancestor: BlockHash,
        descendant: BlockHash,
    ) -> Result<bool, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.archive.is_descendant(&rotxn, ancestor, descendant)?)
    }

    pub fn submit_transaction(
        &self,
        transaction: AuthorizedTransaction,
    ) -> Result<(), Error> {
        {
            let mut rwtxn = self.env.write_txn()?;
            self.state.validate_transaction(
                &self.archive,
                &rwtxn,
                &transaction,
            )?;
            self.mempool.put(&mut rwtxn, &transaction)?;

            if let Some(data) = transaction.transaction.data.as_ref() {
                match data {
                    crate::types::TxData::Trade {
                        market_id,
                        outcome_index,
                        shares,
                        ..
                    } => {
                        if *shares > 0 {
                            self.update_mempool_buy(
                                &mut rwtxn,
                                *market_id,
                                *outcome_index,
                                *shares,
                            )?;
                        } else {
                            self.update_mempool_sell(
                                &mut rwtxn,
                                *market_id,
                                *outcome_index,
                                shares.unsigned_abs() as i64,
                            )?;
                        }
                    }
                    crate::types::TxData::AmplifyBeta {
                        market_id,
                        amount,
                        ..
                    } => {
                        self.update_mempool_amplify_beta(
                            &mut rwtxn, *market_id, *amount,
                        )?;
                    }
                    crate::types::TxData::ClaimDecision { .. }
                    | crate::types::TxData::CreateMarket { .. }
                    | crate::types::TxData::SubmitVote { .. }
                    | crate::types::TxData::SubmitBallot { .. }
                    | crate::types::TxData::TransferReputation { .. } => {}
                }
            }

            rwtxn.commit().map_err(RwTxnError::from)?;
        }
        self.net.push_tx(Default::default(), transaction);
        Ok(())
    }

    pub fn get_mempool_shares(
        &self,
        market_id: &crate::state::markets::MarketId,
    ) -> Result<Option<ndarray::Array1<i64>>, Error> {
        let rotxn = self.env.read_txn()?;
        self.state
            .get_mempool_shares(&rotxn, market_id)
            .map_err(|e| Error::State(Box::new(e)))
    }

    /// Single unified watch stream - fires when state or mempool changes.
    /// State changes cover: new blocks, market updates, decision changes.
    /// Mempool changes cover: new tx submissions, tx confirmations.
    pub fn watch(
        &self,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = ()> + Send>> {
        Box::pin(futures::stream::select(
            self.state.watch(),
            self.mempool.watch(),
        ))
    }

    /// Watch for state changes only (new blocks)
    pub fn watch_state(&self) -> <State as Watchable<()>>::WatchStream {
        self.state.watch()
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

        let beta = self.derive_market_beta(rwtxn, &market_id, &market)?;
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
            hex::encode(market_id),
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

        let beta = self.derive_market_beta(rwtxn, &market_id, &market)?;
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
            hex::encode(market_id),
            outcome_index,
            shares_to_sell
        );

        Ok(())
    }

    fn update_mempool_amplify_beta(
        &self,
        rwtxn: &mut RwTxn,
        market_id: MarketId,
        amount: u64,
    ) -> Result<(), Error> {
        let existing_delta = self
            .state
            .get_mempool_treasury_delta(rwtxn, &market_id)
            .map_err(|e| Error::State(Box::new(e)))?;
        let new_delta =
            existing_delta.checked_add(amount).ok_or_else(|| {
                Error::State(Box::new(crate::state::Error::InvalidDecisionId {
                    reason: "Mempool treasury delta overflow".to_string(),
                }))
            })?;
        self.state
            .put_mempool_treasury_delta(rwtxn, &market_id, new_delta)
            .map_err(|e| Error::State(Box::new(e)))
    }

    pub fn get_mempool_treasury_delta(
        &self,
        market_id: &crate::state::markets::MarketId,
    ) -> Result<u64, Error> {
        let rotxn = self.env.read_txn()?;
        self.state
            .get_mempool_treasury_delta(&rotxn, market_id)
            .map_err(|e| Error::State(Box::new(e)))
    }

    /// Returns the effective treasury (confirmed treasury UTXO value plus
    /// any pending amplify_beta deposits in the mempool). This is the basis
    /// for the effective beta used in price calculations.
    pub fn get_effective_market_treasury_sats(
        &self,
        market_id: &crate::state::markets::MarketId,
    ) -> Result<u64, Error> {
        let rotxn = self.env.read_txn()?;
        let confirmed = self
            .state
            .markets()
            .get_market_funds_sats(&rotxn, &self.state, market_id, false)
            .map_err(|e| Error::State(Box::new(e)))?;
        let pending = self
            .state
            .get_mempool_treasury_delta(&rotxn, market_id)
            .map_err(|e| Error::State(Box::new(e)))?;
        Ok(confirmed.saturating_add(pending))
    }

    /// Derive the current effective LMSR beta for a market.
    /// `beta = (confirmed_treasury + pending_mempool_delta) / ln(num_outcomes)`
    fn derive_market_beta(
        &self,
        rotxn: &sneed::RoTxn,
        market_id: &crate::state::markets::MarketId,
        market: &crate::state::Market,
    ) -> Result<f64, Error> {
        let confirmed = self
            .state
            .markets()
            .get_market_funds_sats(rotxn, &self.state, market_id, false)
            .map_err(|e| Error::State(Box::new(e)))?;
        let pending = self
            .state
            .get_mempool_treasury_delta(rotxn, market_id)
            .map_err(|e| Error::State(Box::new(e)))?;
        Ok(trading::derive_beta_from_liquidity(
            confirmed.saturating_add(pending),
            market.shares().len(),
        ))
    }

    pub fn get_market_beta(
        &self,
        market_id: &crate::state::markets::MarketId,
        market: &crate::state::Market,
    ) -> Result<f64, Error> {
        let rotxn = self.env.read_txn()?;
        self.derive_market_beta(&rotxn, market_id, market)
    }

    pub fn get_all_utxos(
        &self,
    ) -> Result<HashMap<OutPoint, FilledOutput>, Error> {
        let rotxn = self.env.read_txn()?;
        self.state.get_utxos(&rotxn).map_err(Error::from)
    }

    pub fn get_latest_failed_bundle_height(
        &self,
    ) -> Result<Option<u32>, Error> {
        let rotxn = self.env.read_txn()?;
        let res = self
            .state
            .get_latest_failed_withdrawal_bundle(&rotxn)?
            .map(|(height, _)| height);
        Ok(res)
    }

    pub fn get_spent_utxos(
        &self,
        outpoints: &[OutPoint],
    ) -> Result<Vec<(OutPoint, SpentOutput)>, Error> {
        let rotxn = self.env.read_txn()?;
        let mut spent = vec![];
        for outpoint in outpoints {
            let outpoint_key = OutPointKey::from_outpoint(outpoint);
            if let Some(output) = self
                .state
                .stxos()
                .try_get(&rotxn, &outpoint_key)
                .map_err(state::Error::from)?
            {
                spent.push((*outpoint, output));
            }
        }
        Ok(spent)
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
            if let Some(inpoint) = self
                .mempool
                .spent_utxos
                .try_get(&rotxn, outpoint)
                .map_err(mempool::Error::from)?
            {
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

    pub fn get_utxos_by_addresses(
        &self,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, FilledOutput>, Error> {
        let rotxn = self.env.read_txn()?;
        let utxos = self.state.get_utxos_by_addresses(&rotxn, addresses)?;
        Ok(utxos)
    }

    /// Get UTXOs for addresses along with their mempool spent status.
    /// This is atomic - both queries use the same read transaction.
    #[allow(clippy::type_complexity)]
    pub fn get_utxos_with_mempool_status(
        &self,
        addresses: &HashSet<Address>,
    ) -> Result<
        (HashMap<OutPoint, FilledOutput>, Vec<(OutPoint, InPoint)>),
        Error,
    > {
        let rotxn = self.env.read_txn()?;

        // Get confirmed UTXOs from state
        let utxos = self.state.get_utxos_by_addresses(&rotxn, addresses)?;

        // Check which are spent in mempool (same transaction - atomic)
        let mut spent_in_mempool = vec![];
        for outpoint in utxos.keys() {
            if let Some(inpoint) = self
                .mempool
                .spent_utxos
                .try_get(&rotxn, outpoint)
                .map_err(mempool::Error::from)?
            {
                spent_in_mempool.push((*outpoint, inpoint));
            }
        }

        Ok((utxos, spent_in_mempool))
    }

    pub fn try_get_header(
        &self,
        block_hash: BlockHash,
    ) -> Result<Option<Header>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.archive.try_get_header(&rotxn, block_hash)?)
    }

    pub fn get_header(&self, block_hash: BlockHash) -> Result<Header, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.archive.get_header(&rotxn, block_hash)?)
    }

    pub fn try_get_block_hash(
        &self,
        height: u32,
    ) -> Result<Option<BlockHash>, Error> {
        let rotxn = self.env.read_txn()?;
        let Some(tip) = self.state.try_get_tip(&rotxn)? else {
            return Ok(None);
        };
        let Some(tip_height) = self.state.try_get_height(&rotxn)? else {
            return Ok(None);
        };
        if tip_height >= height {
            self.archive
                .ancestors(&rotxn, tip)
                .nth((tip_height - height) as usize)
                .map_err(Error::from)
        } else {
            Ok(None)
        }
    }

    pub fn try_get_body(
        &self,
        block_hash: BlockHash,
    ) -> Result<Option<Body>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.archive.try_get_body(&rotxn, block_hash)?)
    }

    pub fn get_body(&self, block_hash: BlockHash) -> Result<Body, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.archive.get_body(&rotxn, block_hash)?)
    }

    pub fn get_best_main_verification(
        &self,
        hash: BlockHash,
    ) -> Result<bitcoin::BlockHash, Error> {
        let rotxn = self.env.read_txn()?;
        let hash = self.archive.get_best_main_verification(&rotxn, hash)?;
        Ok(hash)
    }

    pub fn get_bmm_inclusions(
        &self,
        block_hash: BlockHash,
    ) -> Result<Vec<bitcoin::BlockHash>, Error> {
        let rotxn = self.env.read_txn()?;
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

    pub fn get_block(&self, block_hash: BlockHash) -> Result<Block, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.archive.get_block(&rotxn, block_hash)?)
    }

    pub fn get_all_transactions(
        &self,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        let rotxn = self.env.read_txn()?;
        let transactions = self.mempool.take_all(&rotxn)?;
        Ok(transactions)
    }

    pub fn get_sidechain_wealth(&self) -> Result<bitcoin::Amount, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.sidechain_wealth(&rotxn)?)
    }

    pub fn get_transactions(
        &self,
        number: usize,
    ) -> Result<(Vec<Authorized<FilledTransaction>>, bitcoin::Amount), Error>
    {
        use crate::math::trading::TRADE_MINER_FEE_SATS;

        let mut rwtxn = self.env.write_txn()?;

        // Take non-trade TXs first, then trade TXs in chronological order
        let all_txs = self.mempool.take(&rwtxn, number)?;
        let trade_txs = self.mempool.take_trades_ordered(&rwtxn)?;

        // Separate non-trade TXs (those not in the trade list)
        let trade_txids: HashSet<_> =
            trade_txs.iter().map(|tx| tx.transaction.txid()).collect();
        let non_trade_txs: Vec<_> = all_txs
            .into_iter()
            .filter(|tx| !trade_txids.contains(&tx.transaction.txid()))
            .collect();

        // Process non-trade TXs first, then trade TXs in insertion order
        let combined_txs =
            non_trade_txs.into_iter().chain(trade_txs).take(number);

        let mut fee = bitcoin::Amount::ZERO;
        let mut returned_transactions = vec![];
        let mut spent_utxos = HashSet::new();
        let mut cumulative_market_states: HashMap<MarketId, Array1<i64>> =
            HashMap::new();

        for transaction in combined_txs {
            let txid = transaction.transaction.txid();
            let is_trade = transaction
                .transaction
                .data
                .as_ref()
                .is_some_and(|d| d.is_trade());

            let inputs: HashSet<_> =
                transaction.transaction.inputs.iter().copied().collect();
            if !spent_utxos.is_disjoint(&inputs) {
                self.mempool.delete(&mut rwtxn, txid)?;
                continue;
            }
            let filled_transaction = match self
                .state
                .fill_authorized_transaction(&rwtxn, transaction.clone())
            {
                Ok(filled_tx) => filled_tx,
                Err(err) => {
                    tracing::warn!(
                        "Cannot fill transaction {} during block construction: {:?}. Removing from mempool.",
                        txid,
                        err
                    );
                    self.mempool.delete(&mut rwtxn, txid)?;
                    continue;
                }
            };

            match self.check_trade_slippage(
                &rwtxn,
                &filled_transaction,
                &mut cumulative_market_states,
            ) {
                Ok(true) => {}
                Ok(false) => {
                    tracing::info!(
                        "Skipping tx {} due to slippage - will retry next block",
                        txid
                    );
                    continue;
                }
                Err(e) => {
                    tracing::warn!(
                        "Slippage check error for {}: {:?} - skipping",
                        txid,
                        e
                    );
                    continue;
                }
            }

            // Compute fee: constant for trade TXs, input-output for others
            let tx_fee = if is_trade {
                bitcoin::Amount::from_sat(TRADE_MINER_FEE_SATS)
            } else {
                let value_in: bitcoin::Amount = filled_transaction
                    .transaction
                    .spent_utxos
                    .iter()
                    .map(GetBitcoinValue::get_bitcoin_value)
                    .checked_sum()
                    .ok_or(AmountOverflowError)?;
                let value_out: bitcoin::Amount = filled_transaction
                    .transaction
                    .transaction
                    .outputs
                    .iter()
                    .map(GetBitcoinValue::get_bitcoin_value)
                    .checked_sum()
                    .ok_or(AmountOverflowError)?;
                value_in.checked_sub(value_out).ok_or(AmountOverflowError)?
            };

            fee = fee.checked_add(tx_fee).ok_or(AmountUnderflowError)?;
            spent_utxos.extend(filled_transaction.transaction.inputs());
            returned_transactions.push(filled_transaction);
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok((returned_transactions, fee))
    }

    /// Check if a trade transaction passes slippage validation against cumulative market state.
    ///
    /// Returns:
    /// - `Ok(true)` if the tx passes slippage check (or is not a market trade)
    /// - `Ok(false)` if slippage exceeded (tx should be skipped, not deleted)
    /// - `Err(...)` if there's an error checking (tx should be skipped)
    ///
    /// For Trade transactions, this checks the cost/proceeds against
    /// the cumulative market state (accounting for prior txs in this block) and updates
    /// the cumulative state if the check passes.
    fn check_trade_slippage(
        &self,
        rotxn: &sneed::RoTxn,
        filled_tx: &Authorized<FilledTransaction>,
        cumulative_states: &mut HashMap<MarketId, Array1<i64>>,
    ) -> Result<bool, Error> {
        let tx_data = match &filled_tx.transaction.transaction.data {
            Some(data) => data,
            None => return Ok(true), // Non-data txs always pass
        };

        match tx_data {
            TxData::Trade {
                market_id,
                outcome_index,
                shares,
                trader,
                limit_sats,
                ..
            } => {
                let is_buy = *shares > 0;
                let shares_abs = shares.unsigned_abs();

                tracing::debug!(
                    "check_trade_slippage: Trade ({}) market={}, outcome={}, shares={}, limit={}",
                    if is_buy { "buy" } else { "sell" },
                    market_id,
                    outcome_index,
                    shares,
                    limit_sats
                );

                let market =
                    match self.state.markets().get_market(rotxn, market_id)? {
                        Some(m) => m,
                        None => {
                            tracing::warn!(
                                "Market {} not found for slippage check",
                                market_id
                            );
                            return Ok(false);
                        }
                    };

                let beta =
                    self.derive_market_beta(rotxn, market_id, &market)?;
                tracing::debug!(
                    "check_trade_slippage: found market with {} outcomes, beta={}",
                    market.shares().len(),
                    beta
                );

                // For sells, verify trader owns sufficient shares
                if !is_buy {
                    let seller_account = self
                        .state
                        .markets()
                        .get_user_share_account(rotxn, trader)?;

                    let owned_shares = seller_account
                        .as_ref()
                        .and_then(|account| {
                            account
                                .positions
                                .get(&(*market_id, *outcome_index))
                                .copied()
                        })
                        .unwrap_or(0);

                    tracing::debug!(
                        "check_trade_slippage: seller {} owns {} shares, trying to sell {}",
                        trader,
                        owned_shares,
                        shares_abs
                    );

                    if owned_shares < 0 || (owned_shares as u64) < shares_abs {
                        tracing::info!(
                            "Insufficient shares for sell: {} owns {} but trying to sell {} - skipping",
                            trader,
                            owned_shares,
                            shares_abs
                        );
                        return Ok(false);
                    }
                }

                // Get cumulative state (or confirmed state if first tx for this market)
                let current_shares = cumulative_states
                    .get(market_id)
                    .cloned()
                    .unwrap_or_else(|| market.shares().clone());
                tracing::debug!(
                    "check_trade_slippage: current_shares={:?}",
                    current_shares
                );

                // Calculate new shares after trade (sign of shares handles direction)
                let mut new_shares = current_shares.clone();
                new_shares[*outcome_index as usize] += *shares;
                tracing::debug!(
                    "check_trade_slippage: new_shares={:?}",
                    new_shares
                );

                if is_buy {
                    // Buy: cost = LMSR(current -> new)
                    let base_cost = trading::calculate_update_cost(
                        &current_shares,
                        &new_shares,
                        beta,
                    )
                    .map_err(|e| {
                        Error::State(Box::new(
                            state::Error::InvalidTransaction {
                                reason: format!(
                                    "LMSR calculation failed: {e:?}"
                                ),
                            },
                        ))
                    })?;
                    tracing::debug!(
                        "check_trade_slippage: base_cost={}",
                        base_cost
                    );

                    let buy_cost = trading::calculate_buy_cost(
                        base_cost,
                        market.trading_fee(),
                    )
                    .map_err(|e| {
                        Error::State(Box::new(
                            state::Error::InvalidTransaction {
                                reason: format!(
                                    "Buy cost calculation failed: {e}"
                                ),
                            },
                        ))
                    })?;
                    tracing::debug!(
                        "check_trade_slippage: total_cost={}, limit={}",
                        buy_cost.total_cost_sats,
                        limit_sats
                    );

                    if buy_cost.total_cost_sats > *limit_sats {
                        tracing::info!(
                            "Slippage exceeded for buy tx: cost {} sats > max {} sats",
                            buy_cost.total_cost_sats,
                            limit_sats
                        );
                        return Ok(false);
                    }
                    // Note: We intentionally do NOT check if embedded costs match recalculated costs.
                    // Multiple transactions targeting the same market in a block will have different
                    // cumulative costs. The slippage check (above) is sufficient - if the cost is
                    // within the user's limit_sats, the transaction is valid.
                } else {
                    // Sell: proceeds = LMSR(new -> current)
                    let gross_proceeds = trading::calculate_update_cost(
                        &new_shares,
                        &current_shares,
                        beta,
                    )
                    .map_err(|e| {
                        Error::State(Box::new(
                            state::Error::InvalidTransaction {
                                reason: format!(
                                    "LMSR calculation failed: {e:?}"
                                ),
                            },
                        ))
                    })?;

                    let sell_proceeds = trading::calculate_sell_proceeds(
                        gross_proceeds,
                        market.trading_fee(),
                    )
                    .map_err(|e| {
                        Error::State(Box::new(
                            state::Error::InvalidTransaction {
                                reason: format!(
                                    "Sell proceeds calculation failed: {e}"
                                ),
                            },
                        ))
                    })?;

                    // Only check slippage limit if limit_sats > 0
                    // (limit_sats = 0 means "no minimum proceeds requirement")
                    if *limit_sats > 0
                        && sell_proceeds.net_proceeds_sats < *limit_sats
                    {
                        tracing::info!(
                            "Slippage exceeded for sell tx: proceeds {} sats < min {} sats",
                            sell_proceeds.net_proceeds_sats,
                            limit_sats
                        );
                        return Ok(false);
                    }
                    // Note: We intentionally do NOT check if embedded costs match recalculated costs.
                    // Multiple transactions targeting the same market in a block will have different
                    // cumulative proceeds. The slippage check (above) is sufficient - if proceeds
                    // meet the user's limit_sats minimum, the transaction is valid.
                }

                // Update cumulative state for subsequent txs
                cumulative_states.insert(*market_id, new_shares);
                tracing::debug!("check_trade_slippage: Trade passed");
                Ok(true)
            }
            TxData::ClaimDecision { .. }
            | TxData::CreateMarket { .. }
            | TxData::SubmitVote { .. }
            | TxData::SubmitBallot { .. }
            | TxData::TransferReputation { .. }
            | TxData::AmplifyBeta { .. } => Ok(true),
        }
    }

    pub fn try_get_transaction(
        &self,
        txid: Txid,
    ) -> Result<Option<Transaction>, Error> {
        let rotxn = self.env.read_txn()?;
        if let Some((block_hash, txin)) = self
            .archive
            .get_tx_inclusions(&rotxn, txid)?
            .first_key_value()
        {
            let body = self.archive.get_body(&rotxn, *block_hash)?;
            let tx = body
                .transactions
                .into_iter()
                .nth(*txin as usize)
                .ok_or_else(|| {
                    Error::State(Box::new(state::Error::InvalidTransaction {
                        reason: format!(
                            "tx index {txin} out of bounds in \
                             block {block_hash}"
                        ),
                    }))
                })?;
            Ok(Some(tx))
        } else if let Some(auth_tx) = self
            .mempool
            .transactions
            .try_get(&rotxn, &txid)
            .map_err(mempool::Error::from)?
        {
            Ok(Some(auth_tx.transaction))
        } else {
            Ok(None)
        }
    }

    pub fn try_get_filled_transaction(
        &self,
        txid: Txid,
    ) -> Result<Option<FilledTransactionWithPosition>, Error> {
        let rotxn = self.env.read_txn()?;
        let tip = self.state.try_get_tip(&rotxn)?;
        let inclusions = self.archive.get_tx_inclusions(&rotxn, txid)?;
        if let Some((block_hash, idx)) =
            inclusions.into_iter().try_find(|(block_hash, _)| {
                if let Some(tip) = tip {
                    self.archive.is_descendant(&rotxn, *block_hash, tip)
                } else {
                    Ok(true)
                }
            })?
        {
            let body = self.archive.get_body(&rotxn, block_hash)?;
            let auth_txs = body.authorized_transactions()?;
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
            match self.state.fill_authorized_transaction(&rotxn, auth_tx) {
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

    pub fn get_pending_withdrawal_bundle(
        &self,
    ) -> Result<Option<WithdrawalBundle>, Error> {
        let rotxn = self.env.read_txn()?;
        let bundle = self
            .state
            .get_pending_withdrawal_bundle(&rotxn)?
            .map(|(bundle, _)| bundle);
        Ok(bundle)
    }

    pub fn remove_from_mempool(&self, txid: Txid) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        let () = self.mempool.delete(&mut rwtxn, txid)?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn connect_peer(&self, addr: SocketAddr) -> Result<(), Error> {
        self.net
            .connect_peer(self.env.clone(), addr)
            .map_err(Error::from)
    }

    pub fn forget_peer(&self, addr: &SocketAddr) -> Result<bool, Error> {
        let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
        let res = self.net.forget_peer(&mut rwtxn, addr)?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(res)
    }

    pub fn get_active_peers(&self) -> Vec<Peer> {
        self.net.get_active_peers()
    }

    pub async fn request_mainchain_ancestor_infos(
        &self,
        block_hash: bitcoin::BlockHash,
    ) -> Result<bool, Error> {
        let mainchain_task::Response::AncestorInfos(_, res): mainchain_task::Response = self
            .mainchain_task
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
        let Some(cusf_mainchain_wallet) = self.cusf_mainchain_wallet.as_ref()
        else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        let block_hash = header.hash();
        if let Some(parent) = header.prev_side_hash
            && self.try_get_header(parent)?.is_none()
        {
            tracing::error!(%block_hash,
                "Rejecting block {block_hash} due to missing ancestor headers",
            );
            return Ok(false);
        }
        let mainchain_task::Response::AncestorInfos(_, res): mainchain_task::Response = self
            .mainchain_task
            .request_oneshot(mainchain_task::Request::AncestorInfos(
                main_block_hash,
            ))
            .map_err(|_| Error::SendMainchainTaskRequest)?
            .await
            .map_err(|_| Error::ReceiveMainchainTaskResponse)?;
        if !res.map_err(Error::MainchainAncestors)? {
            tracing::error!(%block_hash, "submit_block: Mainchain ancestor infos check failed");
            return Ok(false);
        };
        tracing::trace!("Storing header: {block_hash}");
        {
            let mut rwtxn = self.env.write_txn()?;
            let () = self.archive.put_header(&mut rwtxn, header)?;
            // Check BMM in same transaction before committing
            // This prevents TOCTOU: other threads won't see the header until BMM is verified
            if self.archive.get_bmm_result(
                &rwtxn,
                block_hash,
                main_block_hash,
            )? == BmmResult::Failed
            {
                tracing::error!(%block_hash,
                    "Rejecting block {block_hash} due to failing BMM verification",
                );
                // Don't commit - transaction will be dropped and header discarded
                return Ok(false);
            }
            rwtxn.commit().map_err(RwTxnError::from)?;
        }
        tracing::trace!("Stored header: {block_hash}");
        {
            let rotxn = self.env.read_txn()?;
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
                let mut rwtxn = self.env.write_txn()?;
                let () = self.archive.put_body(&mut rwtxn, block_hash, body)?;
                rwtxn.commit().map_err(RwTxnError::from)?;
            }
        }
        let new_tip = Tip {
            block_hash,
            main_block_hash,
        };
        if !self.net_task.new_tip_ready_confirm(new_tip).await? {
            tracing::error!(%block_hash, "submit_block: new_tip_ready_confirm returned false - reorg failed!");
            return Ok(false);
        };
        let rotxn = self.env.read_txn()?;
        let bundle = self.state.get_pending_withdrawal_bundle(&rotxn)?;
        #[cfg(feature = "zmq")]
        {
            let height = self
                .state
                .try_get_height(&rotxn)?
                .ok_or_else(|| Error::State(Box::new(state::Error::NoTip)))?;
            let block_hash = header.hash();
            let mut zmq_msg = zeromq::ZmqMessage::from("hashblock");
            zmq_msg.push_back(bytes::Bytes::copy_from_slice(&block_hash.0));
            zmq_msg.push_back(bytes::Bytes::copy_from_slice(
                &height.to_le_bytes(),
            ));
            if let Err(e) = self.zmq_pub_handler.tx.unbounded_send(zmq_msg) {
                tracing::warn!(
                    "Failed to send ZMQ hashblock notification: {e}"
                );
            }
        }
        if let Some((bundle, _)) = bundle {
            let m6id = bundle.compute_m6id();
            let mut cusf_mainchain_wallet_lock =
                cusf_mainchain_wallet.lock().await;
            let () = cusf_mainchain_wallet_lock
                .broadcast_withdrawal_bundle(bundle.tx())
                .await?;
            drop(cusf_mainchain_wallet_lock);
            tracing::trace!(%m6id, "Broadcast withdrawal bundle");
        }
        Ok(true)
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

    pub fn get_ossified_decisions(
        &self,
    ) -> Result<Vec<crate::state::decisions::DecisionEntry>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.get_ossified_decisions(&rotxn)?)
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
    ) -> Result<(u64, u64, u64), Error> {
        let rotxn = self.env.read_txn()?;
        let current_ts =
            self.state.try_get_mainchain_timestamp(&rotxn)?.unwrap_or(0);
        let current_height = self.state.try_get_height(&rotxn)?;
        let genesis_ts =
            self.state.try_get_genesis_timestamp(&rotxn)?.unwrap_or(0);
        Ok(self.state.decisions().get_listing_fee_info(
            &rotxn,
            period,
            current_ts,
            current_height,
            genesis_ts,
        )?)
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

    pub fn read_txn(&self) -> Result<sneed::RoTxn<'_>, Error> {
        self.env.read_txn().map_err(Into::into)
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn voting_state(&self) -> &crate::state::voting::VotingSystem {
        self.state.voting()
    }

    pub fn reputation(&self) -> &crate::state::reputation::ReputationDbs {
        self.state.reputation()
    }

    pub fn get_tip_height(&self) -> Result<u32, Error> {
        self.try_get_tip_height()?.ok_or_else(|| {
            Error::State(Box::new(state::Error::InvalidTransaction {
                reason: "No tip height found".to_string(),
            }))
        })
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
        self.net_task
            .new_tip_ready_confirm(new_tip)
            .await
            .map_err(Error::from)
    }
}
