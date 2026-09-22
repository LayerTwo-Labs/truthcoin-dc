use std::{collections::HashMap, sync::Arc, time::Duration};

use fallible_iterator::FallibleIterator as _;
use futures::{StreamExt as _, TryFutureExt as _};
use parking_lot::RwLock;
use tokio::{spawn, sync::RwLock as TokioRwLock, task::JoinHandle};
use tokio_util::task::LocalPoolHandle;
use tonic_health::{
    ServingStatus,
    pb::{HealthCheckRequest, health_client::HealthClient},
};
use truthcoin_dc::{
    miner::{self, Miner},
    node::{self, Node},
    types::{
        self, Address, AmountOverflowError, BitcoinOutputContent, Body,
        FilledOutput, InPoint, OutPoint, Output, Transaction,
        proto::mainchain::{
            self,
            generated::{
                block_producer_service_server, mining_service_server,
                validator_service_server, wallet_service_server,
            },
        },
    },
    wallet::{self, Wallet},
};

use crate::cli::Config;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error("CUSF mainchain proto error: {0}")]
    CusfMainchain(#[from] truthcoin_dc::types::proto::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("miner error: {0}")]
    Miner(#[from] miner::Error),
    #[error("node error: {0}")]
    Node(#[source] Box<node::Error>),
    #[error("No CUSF mainchain wallet client")]
    NoCusfMainchainWalletClient,
    #[error("Failed to request mainchain ancestor info for {block_hash}")]
    RequestMainchainAncestorInfos { block_hash: bitcoin::BlockHash },
    #[error("wallet error: {0}")]
    Wallet(#[from] wallet::Error),
}

impl From<node::Error> for Error {
    fn from(err: node::Error) -> Self {
        Self::Node(Box::new(err))
    }
}

struct ProtoSupport {
    block_producer: bool,
    miner: bool,
    wallet: bool,
}

fn update_wallet(node: &Node, wallet: &Wallet) -> Result<(), Error> {
    let addresses = wallet.get_addresses()?;
    let unconfirmed_utxos =
        node.get_unconfirmed_utxos_by_addresses(&addresses)?;
    let utxos_from_state = node.get_utxos_by_addresses(&addresses)?;

    // Get current wallet UTXOs to detect which ones no longer exist in state
    let wallet_utxos = wallet.get_utxos()?;

    // Find UTXOs that are in wallet but NOT in state (these were removed, e.g., by redistribution)
    let mut utxos_to_remove = Vec::new();
    for outpoint in wallet_utxos.keys() {
        if !utxos_from_state.contains_key(outpoint) {
            utxos_to_remove.push(*outpoint);
        }
    }

    // Remove stale UTXOs from wallet (e.g., spent by redistribution)
    if !utxos_to_remove.is_empty() {
        tracing::debug!(
            "Removing {} stale UTXOs from wallet (spent by redistribution or otherwise removed from state)",
            utxos_to_remove.len()
        );
        wallet.spend_utxos(
            &utxos_to_remove
                .iter()
                .map(|outpoint| (*outpoint, InPoint::Redistribution))
                .collect::<Vec<_>>(),
        )?;
    }

    let confirmed_outpoints: Vec<_> = wallet_utxos.into_keys().collect();
    let confirmed_spent = node
        .get_spent_utxos(&confirmed_outpoints)?
        .into_iter()
        .map(|(outpoint, spent_output)| (outpoint, spent_output.inpoint));
    let unconfirmed_outpoints: Vec<_> =
        wallet.get_unconfirmed_utxos()?.into_keys().collect();
    // Check ALL state UTXOs against mempool.spent_utxos, not just wallet UTXOs.
    // This prevents "resurrecting" UTXOs that are spent in mempool but were
    // already moved to wallet.stxos in a previous update cycle.
    let state_outpoints: Vec<_> = utxos_from_state.keys().copied().collect();
    let unconfirmed_spent = node
        .get_unconfirmed_spent_utxos(
            state_outpoints.iter().chain(&unconfirmed_outpoints),
        )?
        .into_iter();
    let spent: Vec<_> = confirmed_spent.chain(unconfirmed_spent).collect();
    wallet.put_utxos(&utxos_from_state)?;
    wallet.put_unconfirmed_utxos(&unconfirmed_utxos)?;
    wallet.spend_utxos(&spent)?;
    Ok(())
}

fn update(
    node: &Node,
    utxos: &mut HashMap<OutPoint, FilledOutput>,
    unconfirmed_utxos: &mut HashMap<OutPoint, Output>,
    wallet: &Wallet,
) -> Result<(), Error> {
    let () = update_wallet(node, wallet)?;
    *utxos = wallet.get_utxos()?;
    *unconfirmed_utxos = wallet.get_unconfirmed_utxos()?;
    Ok(())
}

/// A block that is ready to be blind merged mined
pub struct BlockTemplate {
    /// Bribe to offer for this block
    pub bribe: bitcoin::Amount,
    pub header: types::Header,
    pub body: types::Body,
    pub height: u32,
    /// Fees collected by the transactions in the block
    pub fees: bitcoin::Amount,
}

#[derive(Clone)]
pub struct App {
    pub node: Arc<Node>,
    pub wallet: Wallet,
    pub miner: Option<Arc<TokioRwLock<Miner>>>,
    pub utxos: Arc<RwLock<HashMap<OutPoint, FilledOutput>>>,
    pub unconfirmed_utxos: Arc<RwLock<HashMap<OutPoint, Output>>>,
    pub runtime: Arc<tokio::runtime::Runtime>,
    task: Arc<JoinHandle<()>>,
    pub local_pool: LocalPoolHandle,
}

impl App {
    async fn task(
        node: Arc<Node>,
        utxos: Arc<RwLock<HashMap<OutPoint, FilledOutput>>>,
        unconfirmed_utxos: Arc<RwLock<HashMap<OutPoint, Output>>>,
        wallet: Wallet,
    ) -> Result<(), Error> {
        let mut state_changes = node.watch_state();
        while let Some(()) = state_changes.next().await {
            let () = update(
                &node,
                &mut utxos.write(),
                &mut unconfirmed_utxos.write(),
                &wallet,
            )?;
        }
        Ok(())
    }

    fn spawn_task(
        node: Arc<Node>,
        utxos: Arc<RwLock<HashMap<OutPoint, FilledOutput>>>,
        unconfirmed_utxos: Arc<RwLock<HashMap<OutPoint, Output>>>,
        wallet: Wallet,
    ) -> JoinHandle<()> {
        spawn(
            Self::task(node, utxos, unconfirmed_utxos, wallet).unwrap_or_else(
                |err| {
                    let err = anyhow::Error::from(err);
                    tracing::error!("{err:#}")
                },
            ),
        )
    }

    async fn check_status_serving(
        client: &mut HealthClient<tonic::transport::Channel>,
        service_name: &str,
    ) -> Result<bool, tonic::Status> {
        let health_check_request = HealthCheckRequest {
            service: service_name.to_string(),
        };
        match client.check(health_check_request).await {
            Ok(res) => {
                let expected_status = ServingStatus::Serving;
                let status = res.into_inner().status;
                let as_expected = status == expected_status as i32;
                if !as_expected {
                    tracing::warn!(
                        "Expected status {} for {}, got {}",
                        expected_status,
                        service_name,
                        status
                    );
                }
                Ok(as_expected)
            }
            Err(status) if status.code() == tonic::Code::NotFound => Ok(false),
            Err(e) => Err(e),
        }
    }

    async fn check_proto_support(
        transport: tonic::transport::channel::Channel,
    ) -> Result<ProtoSupport, tonic::Status> {
        let mut health_client = HealthClient::new(transport);
        let block_producer_service_name =
            block_producer_service_server::SERVICE_NAME;
        let mining_service_name = mining_service_server::SERVICE_NAME;
        let validator_service_name = validator_service_server::SERVICE_NAME;
        let wallet_service_name = wallet_service_server::SERVICE_NAME;
        if !Self::check_status_serving(
            &mut health_client,
            validator_service_name,
        )
        .await?
        {
            return Err(tonic::Status::aborted(format!(
                "{validator_service_name} is not supported in mainchain client",
            )));
        }
        tracing::info!("Verified existence of {}", validator_service_name);
        let has_block_producer_service = Self::check_status_serving(
            &mut health_client,
            block_producer_service_name,
        )
        .await?;
        let has_mining_service =
            Self::check_status_serving(&mut health_client, mining_service_name)
                .await?;
        let has_wallet_service =
            Self::check_status_serving(&mut health_client, wallet_service_name)
                .await?;
        tracing::info!(
            %has_block_producer_service,
            %has_mining_service,
            %has_wallet_service,
            "Checked existence of {}, {}, {}",
            block_producer_service_name,
            mining_service_name,
            wallet_service_name,
        );
        Ok(ProtoSupport {
            block_producer: has_block_producer_service,
            miner: has_mining_service,
            wallet: has_wallet_service,
        })
    }

    /// Ask the mainchain node for its services until it answers. The node may
    /// start before the mainchain node.
    async fn wait_for_proto_support(
        transport: tonic::transport::channel::Channel,
        url: &url::Url,
    ) -> ProtoSupport {
        const RETRY_DELAY: Duration = Duration::from_secs(5);

        loop {
            match Self::check_proto_support(transport.clone()).await {
                Ok(proto_support) => return proto_support,
                Err(status) => {
                    tracing::warn!(
                        %url, %status, "Waiting for CUSF mainchain service(s)"
                    );
                    tokio::time::sleep(RETRY_DELAY).await;
                }
            }
        }
    }

    pub fn new(config: &Config) -> Result<Self, Error> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let wallet = Wallet::new(&config.wallet_dir.join("wallet.mdb"))?;
        if let Some(seed_phrase_path) = &config.mnemonic_seed_phrase_path {
            let mnemonic = std::fs::read_to_string(seed_phrase_path)?;
            let () = wallet.set_seed_from_mnemonic(mnemonic.as_str())?;
        }
        tracing::info!(
            url = %config.mainchain_grpc_url,
            "Connecting to mainchain"
        );
        let rt_guard = runtime.enter();
        let transport = tonic::transport::channel::Channel::from_shared(
            config.mainchain_grpc_url.to_string(),
        )
        .unwrap()
        .concurrency_limit(256)
        .connect_lazy();
        let ProtoSupport {
            block_producer: has_block_producer,
            miner: has_miner,
            wallet: has_wallet,
        } = runtime.block_on(Self::wait_for_proto_support(
            transport.clone(),
            &config.mainchain_grpc_url,
        ));
        let cusf_mainchain_block_producer = has_block_producer
            .then(|| mainchain::BlockProducerClient::new(transport.clone()));
        let cusf_mainchain_miner =
            has_miner.then(|| mainchain::MiningClient::new(transport.clone()));
        let cusf_mainchain_wallet =
            has_wallet.then(|| mainchain::WalletClient::new(transport.clone()));
        let cusf_mainchain = mainchain::ValidatorClient::new(transport);
        let miner = cusf_mainchain_wallet
            .map(|wallet| {
                Miner::new(cusf_mainchain.clone(), cusf_mainchain_miner, wallet)
            })
            .transpose()?;
        let local_pool = LocalPoolHandle::new(1);
        let node = runtime.block_on(Node::new(
            config.net_addr,
            &config.datadir,
            config.network_magic_override,
            config.network,
            config.add_peers.clone(),
            config.server_names.clone(),
            cusf_mainchain,
            cusf_mainchain_block_producer,
            &mut rand::rng(),
            &runtime,
            config.decision_config_testing,
            #[cfg(feature = "zmq")]
            config.zmq_addr,
        ))?;
        let (unconfirmed_utxos, utxos) = {
            let mut utxos = wallet.get_utxos()?;
            let mut unconfirmed_utxos = wallet.get_unconfirmed_utxos()?;
            let transactions = node.get_all_transactions()?;
            for transaction in &transactions {
                for input in &transaction.transaction.inputs {
                    utxos.remove(input);
                    unconfirmed_utxos.remove(input);
                }
            }
            let unconfirmed_utxos = Arc::new(RwLock::new(unconfirmed_utxos));
            let utxos = Arc::new(RwLock::new(utxos));
            (unconfirmed_utxos, utxos)
        };
        let node = Arc::new(node);
        let miner = miner.map(|miner| Arc::new(TokioRwLock::new(miner)));
        let task = Self::spawn_task(
            node.clone(),
            utxos.clone(),
            unconfirmed_utxos.clone(),
            wallet.clone(),
        );
        drop(rt_guard);
        Ok(Self {
            node,
            wallet,
            miner,
            unconfirmed_utxos,
            utxos,
            runtime: Arc::new(runtime),
            task: Arc::new(task),
            local_pool,
        })
    }

    pub fn update(&self) -> Result<(), Error> {
        update(
            self.node.as_ref(),
            &mut self.utxos.write(),
            &mut self.unconfirmed_utxos.write(),
            &self.wallet,
        )
    }

    pub fn submit_transaction(
        &self,
        tx: &truthcoin_dc::types::AuthorizedTransaction,
    ) -> Result<(), Error> {
        self.node.submit_transaction(tx)?;
        let () = self.update()?;
        Ok(())
    }

    pub fn sign_and_send(&self, tx: Transaction) -> Result<(), Error> {
        let authorized_transaction = self.wallet.authorize(rand::rng(), tx)?;
        self.submit_transaction(&authorized_transaction)
    }

    pub async fn get_new_main_address(
        &self,
    ) -> Result<bitcoin::Address<bitcoin::address::NetworkChecked>, Error> {
        let Some(miner) = self.miner.as_ref() else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        let mut miner_write = miner.write().await;
        let cusf_mainchain = &mut miner_write.cusf_mainchain;
        let mainchain_info = cusf_mainchain.get_chain_info().await?;
        let cusf_mainchain_wallet = &mut miner_write.cusf_mainchain_wallet;
        let res = cusf_mainchain_wallet
            .create_new_address()
            .await?
            .require_network(mainchain_info.network)
            .unwrap();
        drop(miner_write);
        Ok(res)
    }

    pub fn get_new_main_address_blocking(
        &self,
    ) -> Result<bitcoin::Address<bitcoin::address::NetworkChecked>, Error> {
        self.runtime.block_on(self.get_new_main_address())
    }

    const EMPTY_BLOCK_BMM_BRIBE: bitcoin::Amount =
        bitcoin::Amount::from_sat(1000);

    /// Assemble a block to blind merge mine, without requesting BMM for it
    async fn build_block_template(
        &self,
        fee: Option<bitcoin::Amount>,
    ) -> Result<BlockTemplate, Error> {
        let prev_main_hash = self
            .node
            .with_cusf_mainchain(|cusf_mainchain| cusf_mainchain.clone())
            .get_chain_tip()
            .await?
            .block_hash;
        let tip_hash = self.node.try_get_tip()?;
        // If `prev_side_hash` is not the best tip to mine on, then mine an
        // empty block.
        // This is a temporary fix, ideally we always choose the best tip to
        // mine on
        let prev_side_hash = if let Some(tip_hash) = tip_hash {
            let tip_header = self.node.get_header(tip_hash)?;
            let archive = self.node.archive();
            let prev_main_hash_header_in_archive = {
                let rotxn =
                    self.node.env().read_txn().map_err(node::Error::from)?;
                archive
                    .try_get_main_header_info(&rotxn, &prev_main_hash)
                    .map_err(node::Error::from)?
                    .is_some()
            };
            if !prev_main_hash_header_in_archive {
                // Request mainchain header info
                if !self
                    .node
                    .request_mainchain_ancestor_infos(prev_main_hash)
                    .await?
                {
                    return Err(Error::RequestMainchainAncestorInfos {
                        block_hash: prev_main_hash,
                    });
                }
            }
            let rotxn =
                self.node.env().read_txn().map_err(node::Error::from)?;
            let last_common_main_ancestor = archive
                .last_common_main_ancestor(
                    &rotxn,
                    prev_main_hash,
                    tip_header.prev_main_hash,
                )
                .map_err(node::Error::from)?;
            if last_common_main_ancestor == tip_header.prev_main_hash {
                Some(tip_hash)
            } else {
                // Find a tip to mine on
                archive
                    .ancestor_headers(&rotxn, tip_hash)
                    .find_map(|(block_hash, header)| {
                        if header.prev_main_hash == last_common_main_ancestor {
                            Ok(None)
                        } else if archive.is_main_descendant(
                            &rotxn,
                            header.prev_main_hash,
                            last_common_main_ancestor,
                        )? {
                            Ok(Some(block_hash))
                        } else {
                            Ok(None)
                        }
                    })
                    .map_err(node::Error::from)?
            }
        } else {
            None
        };
        let (bribe, header, body, fees) = if prev_side_hash == tip_hash {
            const NUM_TRANSACTIONS: usize = 1000;
            let (txs, tx_fees) =
                self.node.get_transactions(NUM_TRANSACTIONS)?;
            let new_block_height =
                self.node.try_get_tip_height()?.map_or(0, |h| h + 1);
            let coinbase_address = if new_block_height == 0 {
                self.wallet.voter_address()?
            } else {
                // A template is built on every poll and mostly thrown
                // away, so it must not derive an address each time.
                self.wallet.get_receive_address()?
            };
            let coinbase =
                if tx_fees > bitcoin::Amount::ZERO || new_block_height == 0 {
                    vec![types::Output::new(
                        coinbase_address,
                        types::OutputContent::Bitcoin(BitcoinOutputContent(
                            tx_fees,
                        )),
                    )]
                } else {
                    Vec::new()
                };
            if new_block_height == 0 {
                tracing::info!(
                    "Genesis block: Reputation initialized during \
                     block connection"
                );
            }
            let coinbase = types::Coinbase {
                memo: Vec::new(),
                outputs: coinbase,
            };
            let merkle_root = Body::compute_merkle_root(
                &coinbase,
                &txs.iter()
                    .map(|tx| tx.transaction.transaction.clone())
                    .collect::<Vec<_>>(),
            );
            let body = Body::new(
                txs.into_iter().map(|tx| tx.into()).collect(),
                coinbase,
            );
            let header = types::Header {
                merkle_root,
                prev_side_hash,
                prev_main_hash,
            };
            let bribe = fee.unwrap_or_else(|| {
                if tx_fees > bitcoin::Amount::ZERO {
                    tx_fees
                } else {
                    Self::EMPTY_BLOCK_BMM_BRIBE
                }
            });
            (bribe, header, body, tx_fees)
        } else {
            let coinbase = types::Coinbase::default();
            let merkle_root = Body::compute_merkle_root(&coinbase, &[]);
            let body = Body::new(Vec::new(), coinbase);
            let header = types::Header {
                merkle_root,
                prev_side_hash,
                prev_main_hash,
            };
            let bribe = Self::EMPTY_BLOCK_BMM_BRIBE;
            (bribe, header, body, bitcoin::Amount::ZERO)
        };
        let height = match prev_side_hash {
            None => 0,
            Some(prev_side_hash) => self.node.get_height(prev_side_hash)? + 1,
        };
        Ok(BlockTemplate {
            bribe,
            header,
            body,
            height,
            fees,
        })
    }

    /// Assemble a block to blind merge mine. The caller requests BMM for
    /// `header.hash()` itself, and submits the block via `connect_block` once
    /// its BMM request is included in a mainchain block.
    pub async fn get_block_template(&self) -> Result<BlockTemplate, Error> {
        self.build_block_template(None).await
    }

    /// Connect a block for which a BMM request was included in the specified
    /// mainchain block. Returns `true` if it was accepted as the new tip.
    pub async fn connect_block(
        &self,
        block: types::Block,
        main_block_hash: bitcoin::BlockHash,
    ) -> Result<bool, Error> {
        let types::Block { header, body, .. } = block;
        let accepted = self
            .node
            .submit_block(main_block_hash, &header, &body)
            .await?;
        if accepted {
            let () = self.update()?;
        }
        Ok(accepted)
    }

    /// Attempt to mine a sidechain block
    pub async fn mine(
        &self,
        fee: Option<bitcoin::Amount>,
    ) -> Result<(), Error> {
        let Some(miner) = self.miner.as_ref() else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        let BlockTemplate {
            bribe,
            header,
            body,
            ..
        } = self.build_block_template(fee).await?;
        let mut miner_write = miner.write().await;
        miner_write
            .attempt_bmm(bribe.to_sat(), 0, header, body)
            .await?;
        if let Some((main_hash, header, body)) =
            miner_write.confirm_bmm().await.inspect_err(|err| {
                tracing::error!(
                    "{:#}",
                    truthcoin_dc::util::ErrorChain::new(err)
                )
            })?
        {
            tracing::info!(
                %main_hash,
                side_hash = %header.hash(),
                num_txs = body.transactions.len(),
                "mine: confirmed BMM, submitting block with {} transactions",
                body.transactions.len()
            );
            // Log transaction types in the block
            for (i, tx) in body.transactions.iter().enumerate() {
                let tx_type = match &tx.data {
                    None => "transfer",
                    Some(d) if d.is_trade() => "trade",
                    Some(d) if d.is_create_market() => "create_market",
                    Some(_) => "other",
                };
                tracing::info!(
                    "mine:   tx[{}] = {:?} type={}",
                    i,
                    tx.txid(),
                    tx_type
                );
            }
            match self
                .node
                .submit_block(main_hash, &header, &body)
                .await
                .inspect_err(|err| {
                    tracing::error!(
                        "{:#}",
                        truthcoin_dc::util::ErrorChain::new(err)
                    )
                })? {
                true => {
                    tracing::info!(
                         %main_hash, "mine: BMM accepted as new tip",
                    );
                }
                false => {
                    tracing::error!(
                        %main_hash, "mine: BMM NOT ACCEPTED as new tip - block rejected!",
                    );
                }
            }
        } else {
            tracing::info!(
                "mine: confirm_bmm returned None - no BMM confirmed on mainchain"
            );
        }
        let () = self.update()?;
        Ok(())
    }

    pub async fn deposit(
        &self,
        address: Address,
        amount: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<bitcoin::Txid, Error> {
        let Some(miner) = self.miner.as_ref() else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        let mut miner_write = miner.write().await;
        let txid = miner_write
            .cusf_mainchain_wallet
            .create_deposit_tx(address, amount.to_sat(), fee.to_sat())
            .await?;
        drop(miner_write);
        Ok(txid)
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.task.abort()
    }
}

#[cfg(test)]
mod test {
    use std::{net::SocketAddr, time::Duration};

    use tokio::time::timeout;
    use tonic_health::ServingStatus;
    use truthcoin_dc::types::proto::mainchain::generated::validator_service_server;

    use super::{App, ProtoSupport};

    fn transport(addr: SocketAddr) -> tonic::transport::channel::Channel {
        tonic::transport::channel::Channel::from_shared(format!(
            "http://{addr}"
        ))
        .unwrap()
        .connect_lazy()
    }

    async fn serve_validator_service(addr: SocketAddr) {
        let (health_reporter, health_service) =
            tonic_health::server::health_reporter();
        let () = health_reporter
            .set_service_status(
                validator_service_server::SERVICE_NAME,
                ServingStatus::Serving,
            )
            .await;
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(health_service)
                .serve(addr),
        );
    }

    /// The node may start before the mainchain node, so it waits for the
    /// validator service instead of an error.
    #[tokio::test]
    async fn wait_for_the_validator_service() -> anyhow::Result<()> {
        let reserved =
            reserve_port::ReservedSocketAddr::reserve_random_socket_addr()?;
        let addr = reserved.socket_addr();
        let url = format!("http://{addr}").parse()?;
        let mut proto_support =
            Box::pin(App::wait_for_proto_support(transport(addr), &url));
        assert!(
            timeout(Duration::from_secs(1), &mut proto_support)
                .await
                .is_err()
        );
        let () = serve_validator_service(addr).await;
        let ProtoSupport {
            block_producer,
            miner,
            wallet,
        } = timeout(Duration::from_secs(30), proto_support).await?;
        assert!(!block_producer && !miner && !wallet);
        Ok(())
    }
}
