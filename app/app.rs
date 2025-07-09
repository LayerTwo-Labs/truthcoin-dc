use std::{collections::HashMap, sync::Arc};

use fallible_iterator::FallibleIterator as _;
use futures::{StreamExt as _, TryFutureExt as _};
use parking_lot::RwLock;
use truthcoin_dc::{
    miner::{self, Miner},
    node::{self, Node},
    types::{
        self, Address, AmountOverflowError, BitcoinOutputContent, Body,
        FilledOutput, OutPoint, Output, Transaction,
        proto::mainchain::{
            self,
            generated::{validator_service_server, wallet_service_server},
        },
    },
    wallet::{self, Wallet},
};
use tokio::{spawn, sync::RwLock as TokioRwLock, task::JoinHandle};
use tokio_util::task::LocalPoolHandle;
use tonic_health::{
    ServingStatus,
    pb::{HealthCheckRequest, health_client::HealthClient},
};

use crate::cli::Config;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error("CUSF mainchain proto error")]
    CusfMainchain(#[from] truthcoin_dc::types::proto::Error),
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("miner error: {0}")]
    Miner(#[from] miner::Error),
    #[error("node error")]
    Node(#[source] Box<node::Error>),
    #[error("No CUSF mainchain wallet client")]
    NoCusfMainchainWalletClient,
    #[error("Failed to request mainchain ancestor info for {block_hash}")]
    RequestMainchainAncestorInfos { block_hash: bitcoin::BlockHash },
    #[error("Unable to verify existence of CUSF mainchain service(s) at {url}")]
    VerifyMainchainServices {
        url: Box<url::Url>,
        source: Box<tonic::Status>,
    },
    #[error("wallet error")]
    Wallet(#[from] wallet::Error),
}

impl From<node::Error> for Error {
    fn from(err: node::Error) -> Self {
        Self::Node(Box::new(err))
    }
}

fn update_wallet(node: &Node, wallet: &Wallet) -> Result<(), Error> {
    tracing::trace!("starting wallet update");
    let addresses = wallet.get_addresses()?;
    let unconfirmed_utxos =
        node.get_unconfirmed_utxos_by_addresses(&addresses)?;
    let utxos = node.get_utxos_by_addresses(&addresses)?;
    let confirmed_outpoints: Vec<_> = wallet.get_utxos()?.into_keys().collect();
    let confirmed_spent = node
        .get_spent_utxos(&confirmed_outpoints)?
        .into_iter()
        .map(|(outpoint, spent_output)| (outpoint, spent_output.inpoint));
    let unconfirmed_outpoints: Vec<_> =
        wallet.get_unconfirmed_utxos()?.into_keys().collect();
    let unconfirmed_spent = node
        .get_unconfirmed_spent_utxos(
            confirmed_outpoints.iter().chain(&unconfirmed_outpoints),
        )?
        .into_iter();
    let spent: Vec<_> = confirmed_spent.chain(unconfirmed_spent).collect();
    wallet.put_utxos(&utxos)?;
    wallet.put_unconfirmed_utxos(&unconfirmed_utxos)?;
    wallet.spend_utxos(&spent)?;
    tracing::debug!("finished wallet update");
    Ok(())
}

/// Update (unconfirmed) utxos & wallet
fn update(
    node: &Node,
    utxos: &mut HashMap<OutPoint, FilledOutput>,
    unconfirmed_utxos: &mut HashMap<OutPoint, Output>,
    wallet: &Wallet,
) -> Result<(), Error> {
    tracing::trace!("Updating wallet");
    let () = update_wallet(node, wallet)?;
    *utxos = wallet.get_utxos()?;
    *unconfirmed_utxos = wallet.get_unconfirmed_utxos()?;
    tracing::trace!("Updated wallet");
    Ok(())
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

    /// Check that a service has `Serving` status via gRPC health
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

    /// Returns `true` if validator service AND wallet service are available,
    /// `false` if only validator service is available, and error if validator
    /// service is unavailable.
    async fn check_proto_support(
        transport: tonic::transport::channel::Channel,
    ) -> Result<bool, tonic::Status> {
        let mut health_client = HealthClient::new(transport);
        let validator_service_name = validator_service_server::SERVICE_NAME;
        let wallet_service_name = wallet_service_server::SERVICE_NAME;
        // The validator service MUST exist. We therefore error out here directly.
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
        // The wallet service is optional.
        let has_wallet_service =
            Self::check_status_serving(&mut health_client, wallet_service_name)
                .await?;
        tracing::info!(
            "Checked existence of {}: {}",
            wallet_service_name,
            has_wallet_service
        );
        Ok(has_wallet_service)
    }

    pub fn new(config: &Config) -> Result<Self, Error> {
        // Node launches some tokio tasks for p2p networking, that is why we need a tokio runtime
        // here.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        tracing::info!(
            "Instantiating wallet with data directory: {}",
            config.datadir.display()
        );
        let wallet = Wallet::new(&config.datadir.join("wallet.mdb"))?;
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
        let (cusf_mainchain, cusf_mainchain_wallet) = if runtime
            .block_on(Self::check_proto_support(transport.clone()))
            .map_err(|err| Error::VerifyMainchainServices {
                url: Box::new(config.mainchain_grpc_url.clone()),
                source: Box::new(err),
            })? {
            (
                mainchain::ValidatorClient::new(transport.clone()),
                Some(mainchain::WalletClient::new(transport)),
            )
        } else {
            (mainchain::ValidatorClient::new(transport), None)
        };
        let miner = cusf_mainchain_wallet
            .clone()
            .map(|wallet| Miner::new(cusf_mainchain.clone(), wallet))
            .transpose()?;
        let local_pool = LocalPoolHandle::new(1);
        tracing::debug!("Initializing node...");
        let node = runtime.block_on(Node::new(
            config.net_addr,
            &config.datadir,
            config.network,
            cusf_mainchain,
            cusf_mainchain_wallet,
            &runtime,
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

    /// Update utxos & wallet
    fn update(&self) -> Result<(), Error> {
        update(
            self.node.as_ref(),
            &mut self.utxos.write(),
            &mut self.unconfirmed_utxos.write(),
            &self.wallet,
        )
    }

    pub fn sign_and_send(&self, tx: Transaction) -> Result<(), Error> {
        let authorized_transaction = self.wallet.authorize(tx)?;
        self.node.submit_transaction(authorized_transaction)?;
        let () = self.update()?;
        Ok(())
    }

    pub fn get_new_main_address(
        &self,
    ) -> Result<bitcoin::Address<bitcoin::address::NetworkChecked>, Error> {
        let Some(miner) = self.miner.as_ref() else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        let address = self.runtime.block_on({
            let miner = miner.clone();
            async move {
                let mut miner_write = miner.write().await;
                let cusf_mainchain = &mut miner_write.cusf_mainchain;
                let mainchain_info = cusf_mainchain.get_chain_info().await?;
                let cusf_mainchain_wallet =
                    &mut miner_write.cusf_mainchain_wallet;
                let res = cusf_mainchain_wallet
                    .create_new_address()
                    .await?
                    .require_network(mainchain_info.network)
                    .unwrap();
                drop(miner_write);
                Result::<_, Error>::Ok(res)
            }
        })?;
        Ok(address)
    }

    const EMPTY_BLOCK_BMM_BRIBE: bitcoin::Amount =
        bitcoin::Amount::from_sat(1000);

    pub async fn mine(
        &self,
        fee: Option<bitcoin::Amount>,
    ) -> Result<(), Error> {
        let Some(miner) = self.miner.as_ref() else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        let prev_main_hash = {
            let mut miner_write = miner.write().await;
            let prev_main_hash =
                miner_write.cusf_mainchain.get_chain_tip().await?.block_hash;
            drop(miner_write);
            prev_main_hash
        };
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
        let (bribe, header, body) = if prev_side_hash == tip_hash {
            const NUM_TRANSACTIONS: usize = 1000;
            let (txs, tx_fees) =
                self.node.get_transactions(NUM_TRANSACTIONS)?;
            let coinbase = match tx_fees {
                bitcoin::Amount::ZERO => Vec::new(),
                _ => vec![types::Output::new(
                    self.wallet.get_new_address()?,
                    types::OutputContent::Bitcoin(BitcoinOutputContent(
                        tx_fees,
                    )),
                )],
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
            (bribe, header, body)
        } else {
            let coinbase = Vec::new();
            let merkle_root = Body::compute_merkle_root(&coinbase, &[]);
            let body = Body::new(Vec::new(), coinbase);
            let header = types::Header {
                merkle_root,
                prev_side_hash,
                prev_main_hash,
            };
            let bribe = Self::EMPTY_BLOCK_BMM_BRIBE;
            (bribe, header, body)
        };
        let mut miner_write = miner.write().await;
        miner_write
            .attempt_bmm(bribe.to_sat(), 0, header, body)
            .await?;
        tracing::trace!("confirming bmm...");
        if let Some((main_hash, header, body)) =
            miner_write.confirm_bmm().await.inspect_err(|err| {
                tracing::error!(
                    "{:#}",
                    truthcoin_dc::util::ErrorChain::new(err)
                )
            })?
        {
            tracing::trace!(
                %main_hash,
                side_hash = %header.hash(),
                "mine: confirmed BMM, submitting block",
            );
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
                    tracing::debug!(
                         %main_hash, "mine: BMM accepted as new tip",
                    );
                }
                false => {
                    tracing::warn!(
                        %main_hash, "mine: BMM not accepted as new tip",
                    );
                }
            }
        }
        let () = self.update()?;
        Ok(())
    }

    pub fn deposit(
        &self,
        address: Address,
        amount: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<bitcoin::Txid, Error> {
        let Some(miner) = self.miner.as_ref() else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        self.runtime.block_on(async {
            let mut miner_write = miner.write().await;
            let txid = miner_write
                .cusf_mainchain_wallet
                .create_deposit_tx(address, amount.to_sat(), fee.to_sat())
                .await?;
            drop(miner_write);
            Ok(txid)
        })
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.task.abort()
    }
}
