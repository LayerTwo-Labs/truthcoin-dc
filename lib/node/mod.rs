use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Debug,
    net::SocketAddr,
    path::Path,
    sync::Arc,
};

use bitcoin::amount::CheckedSum as _;
use fallible_iterator::FallibleIterator;
use futures::{Stream, future::BoxFuture};
use sneed::{DbError, Env, EnvError, RwTxnError, env};
use tokio::sync::Mutex;
use tonic::transport::Channel;

use crate::{
    archive::{self, Archive},
    mempool::{self, MemPool},
    net::{self, Net, Peer},
    state::{self, State, markets::MarketId},
    types::{
        Address, AmountOverflowError, AmountUnderflowError, Authorized,
        AuthorizedTransaction, Block, BlockHash, BmmResult, Body, FilledOutput,
        FilledTransaction, GetBitcoinValue, Header, InPoint, Network, OutPoint,
        Output, SpentOutput, Tip, Transaction, TxIn, Txid, WithdrawalBundle,
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
    #[error("mempool error")]
    MemPool(#[from] mempool::Error),
    #[error("net error")]
    Net(#[from] Box<net::Error>),
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
    #[error("state error")]
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
        #[cfg(feature = "zmq")] zmq_addr: SocketAddr,
    ) -> Result<Self, Error>
    where
        mainchain::ValidatorClient<MainchainTransport>: Clone,
        MainchainTransport: Send + 'static,
        <MainchainTransport as tonic::client::GrpcService<
            tonic::body::BoxBody,
        >>::Future: Send,
    {
        let env_path = datadir.join("data.mdb");
        std::fs::create_dir_all(&env_path)?;
        let env = {
            let mut env_open_opts = heed::EnvOpenOptions::new();
            env_open_opts.map_size(1024 * 1024 * 1024).max_dbs(
                State::NUM_DBS
                    + Archive::NUM_DBS
                    + MemPool::NUM_DBS
                    + Net::NUM_DBS,
            );
            unsafe { Env::open(&env_open_opts, &env_path) }?
        };
        let state = State::new(&env)?;
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

    pub fn votecoin_network_data(
        &self,
    ) -> Result<
        Vec<(
            crate::state::votecoin::VotecoinId,
            crate::state::votecoin::VotecoinNetworkData,
        )>,
        Error,
    > {
        let rotxn = self.env.read_txn()?;
        let res = self
            .state
            .votecoin()
            .votecoin()
            .iter(&rotxn)
            .map_err(state::Error::from)?
            .map_err(state::Error::from)
            .map(|(votecoin_id, votecoin_data)| {
                Ok((votecoin_id, votecoin_data.current()))
            })
            .collect()?;
        Ok(res)
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

    pub fn get_votecoin_data_at_block_height(
        &self,
        votecoin_id: &crate::state::votecoin::VotecoinId,
        height: u32,
    ) -> Result<crate::state::votecoin::VotecoinNetworkData, Error> {
        let rotxn = self.env.read_txn()?;
        self.state
            .votecoin()
            .try_get_votecoin_data_at_block_height(&rotxn, votecoin_id, height)
            .map_err(state::Error::from)?
            .ok_or_else(|| {
                Error::State(Box::new(state::Error::UnbalancedVotecoin {
                    inputs: 0,
                    outputs: 0,
                }))
            })
    }

    pub fn try_get_current_votecoin_data(
        &self,
        votecoin_id: &crate::state::votecoin::VotecoinId,
    ) -> Result<Option<crate::state::votecoin::VotecoinNetworkData>, Error>
    {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .votecoin()
            .try_get_current_votecoin_data(&rotxn, votecoin_id)?)
    }

    pub fn get_current_votecoin_data(
        &self,
        votecoin_id: &crate::state::votecoin::VotecoinId,
    ) -> Result<crate::state::votecoin::VotecoinNetworkData, Error> {
        let rotxn = self.env.read_txn()?;
        self.state
            .votecoin()
            .try_get_current_votecoin_data(&rotxn, votecoin_id)?
            .ok_or_else(|| {
                Error::State(Box::new(state::Error::UnbalancedVotecoin {
                    inputs: 0,
                    outputs: 0,
                }))
            })
    }

    pub fn submit_transaction(
        &self,
        transaction: AuthorizedTransaction,
    ) -> Result<(), Error> {
        {
            let mut rwtxn = self.env.write_txn()?;
            self.state.validate_transaction(&rwtxn, &transaction)?;
            self.mempool.put(&mut rwtxn, &transaction)?;

            if let Some(data) = transaction.transaction.data.as_ref()
                && let crate::types::TxData::BuyShares {
                    market_id,
                    outcome_index,
                    shares_to_buy,
                    ..
                } = data
            {
                self.update_mempool_buy(
                    &mut rwtxn,
                    market_id.clone(),
                    *outcome_index,
                    *shares_to_buy,
                )?;
            }

            rwtxn.commit().map_err(RwTxnError::from)?;
        }
        self.net.push_tx(Default::default(), transaction);
        Ok(())
    }

    pub fn get_mempool_shares(
        &self,
        market_id: &crate::state::markets::MarketId,
    ) -> Result<Option<ndarray::Array1<f64>>, Error> {
        let rotxn = self.env.read_txn()?;
        self.state
            .get_mempool_shares(&rotxn, market_id)
            .map_err(|e| Error::State(Box::new(e)))
    }

    fn update_mempool_buy(
        &self,
        rwtxn: &mut RwTxn,
        market_id: MarketId,
        outcome_index: u32,
        shares_to_buy: f64,
    ) -> Result<(), Error> {
        use crate::math::lmsr::LmsrService;
        use crate::state;

        let market = self
            .state
            .markets()
            .get_market(rwtxn, &market_id)?
            .ok_or_else(|| {
                Error::State(Box::new(state::Error::InvalidSlotId {
                    reason: format!("Market {market_id:?} does not exist"),
                }))
            })?;

        if market.state() != crate::state::markets::MarketState::Trading {
            return Ok(());
        }

        if outcome_index as usize >= market.shares().len() {
            return Err(Error::State(Box::new(state::Error::InvalidSlotId {
                reason: format!(
                    "Outcome index {} exceeds market outcomes {}",
                    outcome_index,
                    market.shares().len()
                ),
            })));
        }

        // Get existing mempool shares or use current market shares as base
        let current_shares = if let Some(existing_mempool_shares) =
            self.state.get_mempool_shares(rwtxn, &market_id)?
        {
            existing_mempool_shares
        } else {
            market.shares().clone()
        };

        let mut new_shares = current_shares.clone();
        new_shares[outcome_index as usize] += shares_to_buy;

        LmsrService::validate_lmsr_parameters(market.b(), &new_shares)
            .map_err(|e| {
                Error::State(Box::new(state::Error::InvalidSlotId {
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
            if let Some(output) = self
                .state
                .stxos()
                .try_get(&rotxn, outpoint)
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
        let mut rwtxn = self.env.write_txn()?;
        let transactions = self.mempool.take(&rwtxn, number)?;
        let mut fee = bitcoin::Amount::ZERO;
        let mut returned_transactions = vec![];
        let mut spent_utxos = HashSet::new();
        for transaction in transactions {
            let inputs: HashSet<_> =
                transaction.transaction.inputs.iter().copied().collect();
            if !spent_utxos.is_disjoint(&inputs) {
                // UTXO double spent
                self.mempool
                    .delete(&mut rwtxn, transaction.transaction.txid())?;
                continue;
            }
            // Transactions are validated before entering mempool in submit_transaction().
            // We only need to check if UTXOs are still available (done by fill below).
            let txid = transaction.transaction.txid();
            let filled_transaction = match self
                .state
                .fill_authorized_transaction(&rwtxn, transaction)
            {
                Ok(filled_tx) => filled_tx,
                Err(err) => {
                    tracing::warn!(
                        "Cannot fill transaction {} during block construction (missing UTXOs?): {:?}. Removing from mempool.",
                        txid,
                        err
                    );
                    self.mempool.delete(&mut rwtxn, txid)?;
                    continue;
                }
            };
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
            fee = fee
                .checked_add(
                    value_in
                        .checked_sub(value_out)
                        .ok_or(AmountOverflowError)?,
                )
                .ok_or(AmountUnderflowError)?;
            spent_utxos.extend(filled_transaction.transaction.inputs());
            returned_transactions.push(filled_transaction);
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok((returned_transactions, fee))
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
            let tx = body.transactions.into_iter().nth(*txin as usize).unwrap();
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
            let auth_txs = body.authorized_transactions();
            let auth_tx = auth_txs.into_iter().nth(idx as usize).unwrap();
            let filled_tx = self
                .state
                .fill_transaction_from_stxos(&rotxn, auth_tx.transaction)?;
            let auth_tx = Authorized {
                transaction: filled_tx,
                authorizations: auth_tx.authorizations,
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

    pub fn get_active_peers(&self) -> Vec<Peer> {
        self.net.get_active_peers()
    }

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
            tracing::warn!(%block_hash, "Not ready to reorg");
            return Ok(false);
        };
        let rotxn = self.env.read_txn()?;
        let bundle = self.state.get_pending_withdrawal_bundle(&rotxn)?;
        #[cfg(feature = "zmq")]
        {
            let height = self
                .state
                .try_get_height(&rotxn)?
                .expect("Height should exist for tip");
            let block_hash = header.hash();
            let mut zmq_msg = zeromq::ZmqMessage::from("hashblock");
            zmq_msg.push_back(bytes::Bytes::copy_from_slice(&block_hash.0));
            zmq_msg.push_back(bytes::Bytes::copy_from_slice(
                &height.to_le_bytes(),
            ));
            self.zmq_pub_handler.tx.unbounded_send(zmq_msg).unwrap();
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

    pub fn watch_state(&self) -> impl Stream<Item = ()> {
        self.state.watch()
    }

    pub fn get_all_slot_quarters(&self) -> Result<Vec<(u32, u64)>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.get_all_slot_quarters(&rotxn)?)
    }

    pub fn get_slots_for_quarter(&self, quarter: u32) -> Result<u64, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.get_slots_for_quarter(&rotxn, quarter)?)
    }

    pub fn get_slot(
        &self,
        slot_id: crate::state::slots::SlotId,
    ) -> Result<Option<crate::state::slots::Slot>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.slots().get_slot(&rotxn, slot_id)?)
    }

    pub fn get_available_slots_in_period(
        &self,
        period_id: crate::state::voting::types::VotingPeriodId,
    ) -> Result<Vec<crate::state::slots::SlotId>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .get_available_slots_in_period(&rotxn, period_id.as_u32())?)
    }

    pub fn get_claimed_slots_in_period(
        &self,
        period_id: crate::state::voting::types::VotingPeriodId,
    ) -> Result<Vec<crate::state::slots::Slot>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .slots()
            .get_claimed_slots_in_period(&rotxn, period_id.as_u32())?)
    }

    pub fn is_slot_in_voting(
        &self,
        slot_id: crate::state::slots::SlotId,
    ) -> Result<bool, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.is_slot_in_voting(&rotxn, slot_id)?)
    }

    pub fn get_ossified_slots(
        &self,
    ) -> Result<Vec<crate::state::slots::Slot>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.get_ossified_slots(&rotxn)?)
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

    pub fn get_claimed_slot_count_in_period(
        &self,
        period_id: crate::state::voting::types::VotingPeriodId,
    ) -> Result<u64, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .get_claimed_slot_count_in_period(&rotxn, period_id.as_u32())?)
    }

    pub fn is_slots_testing_mode(&self) -> bool {
        self.state.slots().is_testing_mode()
    }

    pub fn get_slots_testing_config(&self) -> u32 {
        self.state.slots().get_testing_blocks_per_period()
    }

    pub fn get_slot_config(&self) -> &crate::state::slots::SlotConfig {
        self.state.slots().get_config()
    }

    pub fn get_slots_db(&self) -> &crate::state::slots::Dbs {
        self.state.slots()
    }

    pub fn block_height_to_testing_period(&self, block_height: u32) -> u32 {
        self.state
            .slots()
            .block_height_to_testing_period(block_height)
    }

    pub fn quarter_to_string(&self, quarter: u32) -> String {
        self.state.slots().quarter_to_string(quarter)
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
            crate::state::slots::SlotId,
            crate::state::slots::Decision,
        >,
        Error,
    > {
        let rotxn = self.env.read_txn()?;
        let mut decisions = std::collections::HashMap::new();

        for &slot_id in &market.decision_slots {
            if let Some(slot) = self.state.slots().get_slot(&rotxn, slot_id)?
                && let Some(decision) = slot.decision
            {
                decisions.insert(slot_id, decision);
            }
        }

        Ok(decisions)
    }

    pub fn get_user_share_positions(
        &self,
        address: &crate::types::Address,
    ) -> Result<Vec<(crate::state::MarketId, u32, f64)>, Error> {
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
    ) -> Result<Vec<(u32, f64)>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self
            .state
            .markets()
            .get_market_user_positions(&rotxn, address, market_id)?)
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
        Ok(self.state.markets().get_market_treasury_sats(
            &rotxn,
            &self.state,
            market_id,
        )?)
    }

    pub fn read_txn(&self) -> Result<sneed::RoTxn<'_>, Error> {
        self.env.read_txn().map_err(Into::into)
    }

    pub fn voting_state(&self) -> &crate::state::voting::VotingSystem {
        self.state.voting()
    }

    pub fn get_votecoin_balance_for(
        &self,
        rotxn: &sneed::RoTxn,
        address: &crate::types::Address,
    ) -> Result<u32, Error> {
        self.state
            .get_votecoin_balance(rotxn, address)
            .map_err(Into::into)
    }

    pub fn get_tip_height(&self) -> Result<u32, Error> {
        self.try_get_tip_height()?.ok_or_else(|| {
            Error::State(Box::new(state::Error::InvalidTransaction {
                reason: "No tip height found".to_string(),
            }))
        })
    }

    pub fn get_last_block_timestamp(&self) -> Result<u64, Error> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| {
                Error::State(Box::new(state::Error::InvalidTransaction {
                    reason: format!("Failed to get current time: {e}"),
                }))
            })?;
        Ok(timestamp.as_secs())
    }

    pub fn resolve_voting_period(
        &self,
        period_id: crate::state::voting::types::VotingPeriodId,
    ) -> Result<Vec<crate::state::voting::types::DecisionOutcome>, Error> {
        let mut rwtxn = self.env.write_txn()?;
        let current_timestamp = self.get_last_block_timestamp()?;
        let current_height = self.get_tip_height()?;

        let config = self.state.slots().get_config();
        let slots_db = self.state.slots();

        let outcomes = self.state.voting().resolve_period_decisions(
            &mut rwtxn,
            period_id,
            current_timestamp,
            current_height as u64,
            &self.state,
            config,
            slots_db,
        )?;

        rwtxn
            .commit()
            .map_err(|e| Error::DbWrite(RwTxnError::Commit(e)))?;
        Ok(outcomes)
    }

    pub fn get_consensus_outcomes(
        &self,
        period_id: crate::state::voting::types::VotingPeriodId,
    ) -> Result<
        std::collections::HashMap<crate::state::slots::SlotId, f64>,
        Error,
    > {
        let rotxn = self.env.read_txn()?;
        self.state
            .voting()
            .databases()
            .get_consensus_outcomes_for_period(&rotxn, period_id)
            .map_err(Into::into)
    }
}

pub fn quarter_to_string(quarter: u32) -> String {
    let config = crate::state::slots::SlotConfig::production();
    crate::state::slots::quarter_to_string(quarter, &config)
}
