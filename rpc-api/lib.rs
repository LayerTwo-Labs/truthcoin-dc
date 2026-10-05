//! RPC API

/// Exported for convenience
pub use typewit;

pub mod markets;
mod schema;

fn build_openapi(
    build: fn() -> utoipa::openapi::OpenApi,
) -> std::io::Result<utoipa::openapi::OpenApi> {
    // The generated doc builds every schema in one stack frame. That frame
    // does not fit a default 2MB thread stack.
    const STACK_SIZE: usize = 32 * 1024 * 1024;
    std::thread::Builder::new()
        .name("openapi-builder".to_owned())
        .stack_size(STACK_SIZE)
        .spawn(build)?
        .join()
        .map_err(|_panic| {
            std::io::Error::other("the OpenAPI builder thread panicked")
        })
}

/// Build the OpenAPI document that describes all RPC methods.
///
/// # Errors
/// Fails when the builder thread does not start, or panics.
pub fn openapi() -> std::io::Result<utoipa::openapi::OpenApi> {
    build_openapi(|| {
        use utoipa::OpenApi as _;
        let mut res = open_api::RpcDoc::openapi();
        res.merge(node::PrivateRpcDoc::openapi());
        res.merge(node::RpcDoc::openapi());
        res.merge(wallet::RpcDoc::openapi());
        res
    })
}

/// Build the OpenAPI document that describes the public RPC methods.
///
/// # Errors
/// Fails when the builder thread does not start, or panics.
pub fn public_openapi() -> std::io::Result<utoipa::openapi::OpenApi> {
    build_openapi(|| {
        use utoipa::OpenApi as _;
        let mut res = open_api::RpcDoc::openapi();
        res.merge(node::RpcDoc::openapi());
        res
    })
}

/// Build the OpenAPI document that describes the private RPC methods.
///
/// # Errors
/// Fails when the builder thread does not start, or panics.
pub fn private_openapi() -> std::io::Result<utoipa::openapi::OpenApi> {
    build_openapi(|| {
        use utoipa::OpenApi as _;
        let mut res = open_api::RpcDoc::openapi();
        res.merge(node::PrivateRpcDoc::openapi());
        res.merge(wallet::RpcDoc::openapi());
        res
    })
}

pub mod open_api {
    use jsonrpsee::{core::RpcResult, proc_macros::rpc};
    use l2l_openapi::open_api;

    use crate::schema;

    #[open_api]
    #[rpc(client, server)]
    pub trait Rpc {
        /// Get OpenAPI schema
        #[open_api_method(output_schema(PartialSchema = "schema::OpenApi"))]
        #[method(name = "openapi_schema")]
        async fn openapi_schema(&self) -> RpcResult<utoipa::openapi::OpenApi>;
    }
}

pub mod node {
    use std::collections::HashSet;

    use jsonrpsee::{core::RpcResult, proc_macros::rpc};
    use l2l_openapi::open_api;
    use serde::{Deserialize, Serialize};
    use truthcoin_dc_types::{
        Address, Authorization, Authorized, Block, BlockHash, BlockIndex,
        BlockIndexDeposit, BlockIndexSpend, BlockIndexTx, Body,
        ClaimDecisionPayload, Coinbase, CoinbaseTxid, DecisionClaimEntry,
        Header, InPoint, M6id, MainchainSyncPhase, MainchainSyncProgress,
        MerkleRoot, OutPoint, Output, OutputContent, PointedOutput,
        SpentOutput, Transaction, TxData, TxIn, Txid, WithdrawalBundle,
        WithdrawalBundleStatus,
        authorization::Signature,
        decision::DecisionType,
        market::MarketId,
        net::{Peer, PeerAddress, PeerConnectionStatus},
        state::WithdrawalBundleInfo,
        transaction::Outputs,
    };
    use typewit::const_marker::Bool;
    use utoipa::ToSchema;

    use crate::{
        markets::{
            BallotItem, CalculateInitialLiquidityRequest, ConsensusResults,
            DecisionContentInfo, DecisionDetails, DecisionFilter, DecisionInfo,
            DecisionListItem, DecisionListingFeeInfo, DecisionPeriodStatus,
            DecisionState, DecisionSummary, InitialLiquidityCalculation,
            MarketData, MarketDimension, MarketDimensionKind, MarketOutcome,
            MarketPricePoint, MarketResolution, MarketStatus, MarketSummary,
            ParticipationStats, PeriodPricingSummary, PeriodStats, ScoreChange,
            SharePosition, UserHoldings, VoteFilter, VoteInfo, VoterInfo,
            VoterInfoFull, VotingPeriodFull, WinningOutcome,
        },
        open_api, schema,
    };

    #[open_api]
    #[rpc(client, server, server_bounds(Self: open_api::RpcServer))]
    pub trait PrivateRpc {
        /// Connect to a peer
        #[method(name = "connect_peer")]
        async fn connect_peer(&self, addr: PeerAddress) -> RpcResult<()>;

        /// Delete peer from known_peers DB.
        /// Connections to the peer are not terminated.
        #[method(name = "forget_peer")]
        async fn forget_peer(&self, addr: PeerAddress) -> RpcResult<()>;

        /// Invalidate a block, potentially re-orging to a valid ancestor of
        /// the current tip.
        #[method(name = "invalidate_block")]
        async fn invalidate_block(
            &self,
            block_hash: BlockHash,
        ) -> RpcResult<()>;

        /// Remove a tx from the mempool
        #[method(name = "remove_from_mempool")]
        async fn remove_from_mempool(&self, txid: Txid) -> RpcResult<()>;

        /// Stop the node
        #[method(name = "stop")]
        async fn stop(&self);

        /// Trigger a sync to a specific tip block hash.
        /// The block must already exist in our archive (received via P2P).
        /// Returns true if reorg was successful, false if not needed or failed.
        #[method(name = "sync_to_tip")]
        async fn sync_to_tip(&self, block_hash: BlockHash) -> RpcResult<bool>;
    }

    /// Fee and position of a transaction in the active chain
    #[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
    pub struct TxInfo {
        pub confirmations: Option<u32>,
        pub fee_sats: u64,
        pub txin: Option<TxIn>,
    }

    #[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
    pub struct TransactionVerbose {
        #[serde(flatten)]
        pub tx: Transaction,
        #[serde(with = "const_hex")]
        pub canonical_bytes: Vec<u8>,
    }

    pub mod get_block {
        use jsonrpsee::{core::RpcResult, proc_macros::rpc};
        use serde::{Deserialize, Serialize, de::DeserializeOwned};
        use truthcoin_dc_types::{Authorization, Coinbase, Header};
        use utoipa::ToSchema;

        use crate::node::TransactionVerbose;

        #[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
        pub struct BodyVerbose {
            pub coinbase: Coinbase,
            pub transactions: Vec<TransactionVerbose>,
            pub authorizations: Vec<Authorization>,
            pub actor_proofs: Vec<Option<Authorization>>,
        }

        #[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
        pub struct BlockVerbose {
            pub header: Header,
            pub body: BodyVerbose,
        }

        pub mod verbosity {
            use serde::{Serialize, de::DeserializeOwned};
            use truthcoin_dc_types::Block;
            use typewit::const_marker::Bool;

            use crate::node::get_block::BlockVerbose;

            mod private {
                pub trait Sealed {}
            }

            pub trait Verbosity: Serialize + private::Sealed {
                type Response: DeserializeOwned + Serialize;
            }

            impl<const B: bool> private::Sealed for Bool<B> {}

            impl Verbosity for Bool<true> {
                type Response = BlockVerbose;
            }

            impl Verbosity for Bool<false> {
                type Response = Block;
            }

            impl private::Sealed for Option<Bool<false>> {}

            impl Verbosity for Option<Bool<false>> {
                type Response = <Bool<false> as Verbosity>::Response;
            }
        }
        pub use verbosity::Verbosity;

        #[rpc(client, server, server_bounds(
            V: DeserializeOwned + Verbosity,
            <V as Verbosity>::Response: Clone + 'static,
        ))]
        pub trait Rpc<V>
        where
            V: Verbosity,
        {
            /// Get the block with specified block hash, if it exists
            #[method(name = "get_block")]
            async fn get_block(
                &self,
                block_hash: truthcoin_dc_types::BlockHash,
                verbose: V,
            ) -> RpcResult<Option<V::Response>>;
        }

        pub mod untyped {
            use jsonrpsee::{
                core::{RpcResult, async_trait},
                proc_macros::rpc,
            };
            use l2l_openapi::open_api;
            use serde::Serialize;
            use truthcoin_dc_types::{
                Address, Authorization, Block, BlockHash, Body,
                ClaimDecisionPayload, Coinbase, CoinbaseTxid,
                DecisionClaimEntry, Header, MerkleRoot, Output, OutputContent,
                Transaction, TxData, Txid, authorization::Signature,
                market::MarketId, transaction::Outputs,
            };
            use typewit::const_marker::Bool;
            use utoipa::ToSchema;

            use crate::{
                markets::BallotItem,
                node::{
                    TransactionVerbose,
                    get_block::{
                        BlockVerbose, BodyVerbose, RpcServer as GetBlock,
                    },
                },
                schema,
            };

            mod private {
                pub trait Sealed {}
            }

            impl<S> private::Sealed for S where
                S: GetBlock<Bool<false>> + GetBlock<Bool<true>>
            {
            }

            #[derive(Clone, Serialize, ToSchema)]
            #[serde(untagged)]
            pub enum Response {
                NonVerbose(Block),
                Verbose(BlockVerbose),
            }

            /// This trait exists only as a bound, and should not be implemented
            /// manually
            #[open_api(ref_schemas[
                Address, Authorization, BallotItem, Block, BlockHash,
                BlockVerbose, Body, BodyVerbose, ClaimDecisionPayload, Coinbase,
                CoinbaseTxid, DecisionClaimEntry, Header, MarketId, MerkleRoot,
                Output, OutputContent, Outputs, Signature, Transaction,
                TransactionVerbose, TxData, Txid, schema::BitcoinAddr,
                schema::BitcoinBlockHash, schema::BitcoinOutPoint,
                schema::UtreexoNodeHash, schema::UtreexoProof,
            ])]
            #[rpc(server, server_bounds(Self: private::Sealed))]
            pub trait Rpc {
                /// Get the block with specified block hash, if it exists
                #[method(name = "get_block")]
                async fn get_block(
                    &self,
                    block_hash: truthcoin_dc_types::BlockHash,
                    verbose: Option<bool>,
                ) -> RpcResult<Option<Response>>;
            }

            #[async_trait]
            impl<S> RpcServer for S
            where
                S: GetBlock<Bool<false>> + GetBlock<Bool<true>>,
            {
                async fn get_block(
                    &self,
                    block_hash: truthcoin_dc_types::BlockHash,
                    verbose: Option<bool>,
                ) -> RpcResult<Option<Response>> {
                    match verbose {
                        Some(true) => {
                            <Self as GetBlock<Bool<true>>>::get_block(
                                self,
                                block_hash,
                                Bool::<true>,
                            )
                            .await
                            .map(|res| res.map(Response::Verbose))
                        }
                        Some(false) | None => {
                            <Self as GetBlock<Bool<false>>>::get_block(
                                self,
                                block_hash,
                                Bool::<false>,
                            )
                            .await
                            .map(|res| res.map(Response::NonVerbose))
                        }
                    }
                }
            }
        }
        pub use untyped::RpcDoc;
    }

    #[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
    pub struct GetTransactionResponse {
        pub tx: Transaction,
        /// Block hash, if in the active chain
        pub block_hash: Option<BlockHash>,
    }

    /// One transaction the mempool holds
    #[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
    pub struct MempoolTx {
        /// Blake3 over the canonical encoding
        pub txid: Txid,
        /// Canonical size in bytes
        pub size: u64,
        /// Borsh encoding, as hex
        pub raw: String,
        pub tx: Transaction,
    }

    #[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
    pub struct GetWithdrawalBundleResponse {
        pub info: WithdrawalBundleInfo,
        pub status: WithdrawalBundleStatus,
    }

    #[open_api(
        merge_apis[get_block::RpcDoc],
        ref_schemas[
            Address, Authorization, BallotItem, BlockHash, BlockIndexDeposit,
            BlockIndexSpend, BlockIndexTx, Body, ClaimDecisionPayload, Coinbase,
            CoinbaseTxid, ConsensusResults, DecisionClaimEntry,
            DecisionContentInfo, DecisionInfo, DecisionState, DecisionSummary,
            DecisionType, Header, InPoint, M6id, MainchainSyncPhase,
            MarketDimension, MarketDimensionKind, MarketId, MarketOutcome,
            MarketResolution, MarketStatus, MerkleRoot, OutPoint, Output,
            OutputContent, Outputs, ParticipationStats, PeerConnectionStatus,
            PeriodStats, ScoreChange, SharePosition, Signature, SpentOutput,
            Transaction, TxData, TxIn, Txid, WinningOutcome, WithdrawalBundle,
            WithdrawalBundleInfo, WithdrawalBundleStatus, schema::BitcoinAddr,
            schema::BitcoinBlockHash, schema::BitcoinOutPoint,
            schema::BitcoinTransaction, schema::SocketAddr,
            schema::UtreexoNodeHash, schema::UtreexoProof,
        ],
    )]
    #[rpc(
        client,
        client_bounds(
            Self:
                get_block::RpcClient<Bool<false>>
                + get_block::RpcClient<Bool<true>>
        ),
        server,
        server_bounds(
            Self: open_api::RpcServer + get_block::untyped::RpcServer,
        ),
    )]
    pub trait Rpc {
        /// Connect a block template for which a BMM request was included in the
        /// specified mainchain block. Returns `true` if it was accepted as the new
        /// tip.
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "connect_block")]
        async fn connect_block(
            &self,
            block: Block,
            #[open_api_method_arg(schema(
                PartialSchema = "schema::BitcoinBlockHash"
            ))]
            main_block_hash: bitcoin::BlockHash,
        ) -> RpcResult<bool>;

        /// Get the block hash at the specified height in the active chain,
        /// if it exists
        #[open_api_method(output_schema(
            PartialSchema = "schema::Optional<truthcoin_dc_types::BlockHash>"
        ))]
        #[method(name = "get_block_hash")]
        async fn get_block_hash(
            &self,
            height: u32,
        ) -> RpcResult<Option<truthcoin_dc_types::BlockHash>>;

        /// Get the transaction ids, sizes and encodings of a block, with the
        /// mainchain deposits and withdrawal bundle spends it applied
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "get_block_index")]
        async fn get_block_index(
            &self,
            block_hash: truthcoin_dc_types::BlockHash,
        ) -> RpcResult<BlockIndex>;

        /// Get mainchain blocks that commit to a specified block hash
        #[open_api_method(output_schema(
            PartialSchema = "schema::BitcoinBlockHash"
        ))]
        #[method(name = "get_bmm_inclusions")]
        async fn get_bmm_inclusions(
            &self,
            block_hash: truthcoin_dc_types::BlockHash,
        ) -> RpcResult<Vec<bitcoin::BlockHash>>;

        /// Get the best mainchain block hash known by Thunder
        #[open_api_method(output_schema(
            PartialSchema = "schema::Optional<schema::BitcoinBlockHash>"
        ))]
        #[method(name = "get_best_mainchain_block_hash")]
        async fn get_best_mainchain_block_hash(
            &self,
        ) -> RpcResult<Option<bitcoin::BlockHash>>;

        /// Get the best sidechain block hash known by Thunder
        #[open_api_method(output_schema(
            PartialSchema = "schema::Optional<truthcoin_dc_types::BlockHash>"
        ))]
        #[method(name = "get_best_sidechain_block_hash")]
        async fn get_best_sidechain_block_hash(
            &self,
        ) -> RpcResult<Option<truthcoin_dc_types::BlockHash>>;

        /// Get stxos for addresses
        #[method(name = "get_stxos")]
        async fn get_stxos(
            &self,
            addresses: HashSet<Address>,
        ) -> RpcResult<Vec<PointedOutput<SpentOutput>>>;

        /// Get transaction by txid
        #[method(name = "get_transaction")]
        async fn get_transaction(
            &self,
            txid: Txid,
        ) -> RpcResult<Option<GetTransactionResponse>>;

        /// Get utxos for addresses
        #[method(name = "get_utxos")]
        async fn get_utxos(
            &self,
            addresses: HashSet<Address>,
        ) -> RpcResult<Vec<PointedOutput>>;

        /// Get withdrawal bundle by M6id
        #[method(name = "get_withdrawal_bundle")]
        async fn get_withdrawal_bundle(
            &self,
            m6id: M6id,
        ) -> RpcResult<Option<GetWithdrawalBundleResponse>>;

        /// Get the current block count
        #[method(name = "getblockcount")]
        async fn getblockcount(&self) -> RpcResult<u32>;

        /// Get the height of the latest failed withdrawal bundle
        #[method(name = "latest_failed_withdrawal_bundle_height")]
        async fn latest_failed_withdrawal_bundle_height(
            &self,
        ) -> RpcResult<Option<u32>>;

        /// List the transactions the mempool holds, in no particular order.
        #[method(name = "list_mempool")]
        async fn list_mempool(&self) -> RpcResult<Vec<MempoolTx>>;

        /// List peers
        #[method(name = "list_peers")]
        async fn list_peers(&self) -> RpcResult<Vec<Peer>>;

        /// List all UTXOs
        #[method(name = "list_utxos")]
        async fn list_utxos(&self) -> RpcResult<Vec<PointedOutput>>;

        /// Get the progress of the startup sync with the mainchain
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "mainchain_sync_progress")]
        async fn mainchain_sync_progress(
            &self,
        ) -> RpcResult<MainchainSyncProgress>;

        /// Get pending withdrawal bundle
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "pending_withdrawal_bundle")]
        async fn pending_withdrawal_bundle(
            &self,
        ) -> RpcResult<Option<WithdrawalBundle>>;

        /// Get total sidechain wealth
        #[method(name = "sidechain_wealth")]
        async fn sidechain_wealth_sats(&self) -> RpcResult<u64>;

        /// Verify and broadcast a transaction
        #[method(name = "submit_transaction")]
        async fn submit_transaction(
            &self,
            transaction: Authorized<Transaction>,
        ) -> RpcResult<Txid>;

        /// Wait until the node reaches a specific block height (for sync)
        /// Returns the actual height reached (may be higher than requested)
        /// Times out after the specified milliseconds (default 10000ms)
        #[method(name = "await_block_height")]
        async fn await_block_height(
            &self,
            target_height: u32,
            timeout_ms: Option<u64>,
        ) -> RpcResult<u32>;

        /// Get information about a transaction in the current chain
        #[method(name = "get_transaction_info")]
        async fn get_transaction_info(
            &self,
            txid: Txid,
        ) -> RpcResult<Option<TxInfo>>;

        /// Submit a hex-encoded borsh-serialized `AuthorizedTransaction` directly
        /// to the mempool. Returns the transaction id on success.
        ///
        /// Intended for tests and advanced clients that need to submit a
        /// pre-signed transaction (e.g. to exercise validator paths like
        /// stale `prev_block_hash` rejection).
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "push_tx")]
        async fn push_tx(&self, tx_hex: String) -> RpcResult<Txid>;

        /// Calculate initial liquidity required for market creation
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "calculate_initial_liquidity")]
        async fn calculate_initial_liquidity(
            &self,
            request: CalculateInitialLiquidityRequest,
        ) -> RpcResult<InitialLiquidityCalculation>;

        /// Compute the listing fee (sats) for claiming a specific decision_id.
        /// The tier (and therefore the price multiplier) is determined by the
        /// id's decision_index field.
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "decision_fee_for_id")]
        async fn decision_fee_for_id(
            &self,
            decision_id_hex: String,
        ) -> RpcResult<u64>;

        /// Get a specific decision by ID (includes is_voting status)
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "decision_get")]
        async fn decision_get(
            &self,
            decision_id: String,
        ) -> RpcResult<Option<DecisionDetails>>;

        /// List decisions with optional filtering by period and state
        #[open_api_method(output_schema(ToSchema = "Vec<DecisionListItem>"))]
        #[method(name = "decision_list")]
        async fn decision_list(
            &self,
            filter: Option<DecisionFilter>,
        ) -> RpcResult<Vec<DecisionListItem>>;

        /// Get listing fee info for a period
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "decision_listing_fee")]
        async fn decision_listing_fee(
            &self,
            period: u32,
        ) -> RpcResult<DecisionListingFeeInfo>;

        /// Get decision system status and configuration
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "decision_status")]
        async fn decision_status(&self) -> RpcResult<DecisionPeriodStatus>;

        /// List open voting periods with the price the next claim would pay
        /// for the cheapest available unlocked slot in each. For the GUI's
        /// per-dimension period picker.
        #[open_api_method(output_schema(
            ToSchema = "Vec<PeriodPricingSummary>"
        ))]
        #[method(name = "list_open_periods_with_pricing")]
        async fn list_open_periods_with_pricing(
            &self,
        ) -> RpcResult<Vec<PeriodPricingSummary>>;

        /// Get detailed market information
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "market_get")]
        async fn market_get(
            &self,
            market_id: String,
        ) -> RpcResult<Option<MarketData>>;

        /// List all markets
        #[open_api_method(output_schema(ToSchema = "Vec<MarketSummary>"))]
        #[method(name = "market_list")]
        async fn market_list(&self) -> RpcResult<Vec<MarketSummary>>;

        /// Get the price history of a market from the active chain: one point
        /// for the creation block and one for each block that changed its
        /// prices, oldest first
        #[open_api_method(output_schema(ToSchema = "Vec<MarketPricePoint>"))]
        #[method(name = "market_price_history")]
        async fn market_price_history(
            &self,
            market_id: String,
        ) -> RpcResult<Vec<MarketPricePoint>>;

        /// Get share positions for an address (optionally filtered by market)
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "market_positions")]
        async fn market_positions(
            &self,
            address: Address,
            market_id: Option<String>,
        ) -> RpcResult<UserHoldings>;

        /// Query votes with filters (by voter, decision, or period)
        #[open_api_method(output_schema(ToSchema = "Vec<VoteInfo>"))]
        #[method(name = "vote_list")]
        async fn vote_list(
            &self,
            filter: VoteFilter,
        ) -> RpcResult<Vec<VoteInfo>>;

        /// Get full voting period information (null period_id = current)
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "vote_period")]
        async fn vote_period(
            &self,
            period_id: Option<u32>,
        ) -> RpcResult<Option<VotingPeriodFull>>;

        /// Get full voter information
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "vote_voter")]
        async fn vote_voter(
            &self,
            address: Address,
        ) -> RpcResult<Option<VoterInfoFull>>;

        /// List all registered voters
        #[open_api_method(output_schema(ToSchema = "Vec<VoterInfo>"))]
        #[method(name = "vote_voters")]
        async fn vote_voters(&self) -> RpcResult<Vec<VoterInfo>>;

        /// Get votecoin balance for an address
        #[open_api_method(output_schema(ToSchema = "f64"))]
        #[method(name = "votecoin_balance")]
        async fn votecoin_balance(&self, address: Address) -> RpcResult<f64>;
    }
}

pub mod wallet {
    use jsonrpsee::{core::RpcResult, proc_macros::rpc};
    use l2l_openapi::open_api;
    use serde::{Deserialize, Serialize};
    use truthcoin_dc_types::{
        Address, Authorization, Authorized, Block, BlockHash, Body,
        ClaimDecisionPayload, Coinbase, CoinbaseTxid, DecisionClaimEntry,
        EncryptionPubKey, Header, MerkleRoot, OutPoint, Output, OutputContent,
        PointedOutput, Transaction, TxData, Txid, VerifyingKey,
        authorization::{Dst, Signature},
        market::MarketId,
        transaction::Outputs,
        wallet::{Balance, TransferDests},
    };
    use utoipa::ToSchema;

    use crate::{
        markets::{
            BallotItem, ClaimedDecisionInfo, CreateTradeRequest,
            CreateTradeResponse, DecisionClaimItem, DecisionClaimRequest,
            DecisionClaimResponse, DimensionInput, MarketAmplifyBetaRequest,
            MarketBuyRequest, MarketBuyResponse, MarketCreateRequest,
            MarketCreateResponse, MarketSellRequest, MarketSellResponse,
        },
        open_api, schema,
    };

    #[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
    pub struct GetBlockTemplateResponse {
        /// Block hash to commit to in a BMM request
        pub critical_hash: BlockHash,
        /// Block to pass to `connect_block` once its BMM request is included in a
        /// mainchain block
        pub block: Block,
        /// Fees collected by the transactions in the block, in sats
        pub fees_sats: u64,
    }

    #[open_api(ref_schemas[
        Address, Authorization, BallotItem, Block, BlockHash, Body,
        ClaimDecisionPayload, ClaimedDecisionInfo, Coinbase, CoinbaseTxid,
        DecisionClaimEntry, DecisionClaimItem, DimensionInput, Header, MarketId,
        MerkleRoot, OutPoint, Output, OutputContent, Outputs, Signature,
        Transaction, TxData, Txid, schema::BitcoinAddr,
        schema::BitcoinBlockHash, schema::BitcoinOutPoint,
        schema::UtreexoNodeHash, schema::UtreexoProof,
    ])]
    #[rpc(client, server, server_bounds(Self: open_api::RpcServer))]
    pub trait Rpc {
        /// Get balance in sats
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "balance")]
        async fn balance(&self) -> RpcResult<Balance>;

        /// Deposit to address
        #[open_api_method(output_schema(
            PartialSchema = "schema::BitcoinTxid"
        ))]
        #[method(name = "create_deposit")]
        async fn create_deposit(
            &self,
            address: Address,
            value_sats: u64,
            fee_sats: u64,
        ) -> RpcResult<bitcoin::Txid>;

        /// Create a tx that transfers funds to the specified address
        #[method(name = "create_transfer")]
        async fn create_transfer(
            &self,
            dest: Address,
            value_sats: u64,
            fee_sats: u64,
        ) -> RpcResult<Txid>;

        /// Create a tx that transfers funds to each address in `dests`,
        /// which maps an address to a value in sats. The outputs come in
        /// address order, and the change output comes last.
        #[method(name = "create_transfer_many")]
        async fn create_transfer_many(
            &self,
            dests: TransferDests,
            fee_sats: u64,
        ) -> RpcResult<Txid>;

        /// Creates a tx that initiates a withdrawal to the specified mainchain
        /// address
        #[method(name = "create_withdrawal")]
        async fn create_withdrawal(
            &self,
            #[open_api_method_arg(schema(
                PartialSchema = "schema::BitcoinAddr"
            ))]
            mainchain_address: bitcoin::Address<
                bitcoin::address::NetworkUnchecked,
            >,
            amount_sats: u64,
            fee_sats: u64,
            mainchain_fee_sats: u64,
        ) -> RpcResult<Txid>;

        /// Format a deposit address
        #[method(name = "format_deposit_address")]
        async fn format_deposit_address(
            &self,
            address: Address,
        ) -> RpcResult<String>;

        /// Generate a mnemonic seed phrase
        #[method(name = "generate_mnemonic")]
        async fn generate_mnemonic(&self) -> RpcResult<String>;

        /// Assemble a block to blind merge mine, without requesting BMM for it.
        /// The caller requests BMM for `critical_hash` itself, then passes the
        /// block back to `connect_block`.
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "get_block_template")]
        async fn get_block_template(
            &self,
        ) -> RpcResult<GetBlockTemplateResponse>;

        /// Get a new address
        #[method(name = "get_new_address")]
        async fn get_new_address(&self) -> RpcResult<Address>;

        /// Get wallet addresses, sorted by base58 encoding
        #[method(name = "get_wallet_addresses")]
        async fn get_wallet_addresses(&self) -> RpcResult<Vec<Address>>;

        /// Get wallet UTXOs
        #[method(name = "get_wallet_utxos")]
        async fn get_wallet_utxos(&self) -> RpcResult<Vec<PointedOutput>>;

        /// Attempt to mine a sidechain block
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "mine")]
        async fn mine(&self, fee: Option<u64>) -> RpcResult<()>;

        /// Set the wallet seed from a mnemonic seed phrase
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "set_seed_from_mnemonic")]
        async fn set_seed_from_mnemonic(
            &self,
            mnemonic: String,
        ) -> RpcResult<()>;

        /// Sign a transaction, and optionally broadcast it.
        #[method(name = "sign_transaction")]
        async fn sign_transaction(
            &self,
            transaction: Transaction,
            broadcast: Option<bool>,
        ) -> RpcResult<Authorized<Transaction>>;

        #[method(name = "decrypt_msg")]
        async fn decrypt_msg(
            &self,
            encryption_pubkey: EncryptionPubKey,
            ciphertext: String,
        ) -> RpcResult<String>;

        #[method(name = "encrypt_msg")]
        async fn encrypt_msg(
            &self,
            encryption_pubkey: EncryptionPubKey,
            msg: String,
        ) -> RpcResult<String>;

        /// Generate new encryption key
        #[method(name = "get_new_encryption_key")]
        async fn get_new_encryption_key(&self) -> RpcResult<EncryptionPubKey>;

        /// Generate new verifying/signing key
        #[method(name = "get_new_verifying_key")]
        async fn get_new_verifying_key(&self) -> RpcResult<VerifyingKey>;

        /// List unconfirmed owned UTXOs
        #[method(name = "my_unconfirmed_utxos")]
        async fn my_unconfirmed_utxos(&self) -> RpcResult<Vec<PointedOutput>>;

        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "refresh_wallet")]
        async fn refresh_wallet(&self) -> RpcResult<()>;

        /// Sign an arbitrary message with the specified verifying key
        #[method(name = "sign_arbitrary_msg")]
        async fn sign_arbitrary_msg(
            &self,
            verifying_key: VerifyingKey,
            msg: String,
        ) -> RpcResult<Signature>;

        /// Sign an arbitrary message with the secret key for the specified address
        #[method(name = "sign_arbitrary_msg_as_addr")]
        async fn sign_arbitrary_msg_as_addr(
            &self,
            address: Address,
            msg: String,
        ) -> RpcResult<Authorization>;

        /// Verify a signature on a message against the specified verifying key.
        /// Returns `true` if the signature is valid
        #[method(name = "verify_signature")]
        async fn verify_signature(
            &self,
            signature: Signature,
            verifying_key: VerifyingKey,
            dst: Dst,
            msg: String,
        ) -> RpcResult<bool>;

        /// Get the voter address (index 0), used for reputation
        /// and voting identity
        #[method(name = "get_voter_address")]
        async fn get_voter_address(&self) -> RpcResult<Address>;

        /// Transfer votecoin to the specified address
        #[method(name = "transfer_votecoin")]
        async fn transfer_votecoin(
            &self,
            dest: Address,
            amount: f64,
            fee_sats: u64,
        ) -> RpcResult<Txid>;

        /// Claim one or more decisions.
        /// decision_type: "binary", "scaled", or "category"
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "decision_claim")]
        async fn decision_claim(
            &self,
            request: DecisionClaimRequest,
        ) -> RpcResult<DecisionClaimResponse>;

        /// Create a prediction market, optionally claiming new decisions in
        /// the same tx. Each dimension references either an existing claimed
        /// decision or carries new-claim metadata that will be allocated a
        /// slot and claimed before the market is built.
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "market_create")]
        async fn market_create(
            &self,
            request: MarketCreateRequest,
        ) -> RpcResult<MarketCreateResponse>;

        /// Buy shares (with dry_run support for cost calculation)
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "market_buy")]
        async fn market_buy(
            &self,
            request: MarketBuyRequest,
        ) -> RpcResult<MarketBuyResponse>;

        /// Sell shares (with dry_run support for proceeds calculation)
        /// Payout is created during block connection from market treasury
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "market_sell")]
        async fn market_sell(
            &self,
            request: MarketSellRequest,
        ) -> RpcResult<MarketSellResponse>;

        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "market_amplify_beta")]
        async fn market_amplify_beta(
            &self,
            request: MarketAmplifyBetaRequest,
        ) -> RpcResult<String>;

        /// Submit one or more votes (batch)
        #[open_api_method(output_schema(ToSchema = "String"))]
        #[method(name = "vote_submit")]
        async fn vote_submit(
            &self,
            votes: Vec<BallotItem>,
            fee_sats: u64,
        ) -> RpcResult<String>;

        /// Build and sign a Trade transaction with a caller-supplied
        /// `prev_block_hash`, returning the hex-encoded signed
        /// `AuthorizedTransaction` *without* submitting it to the mempool.
        ///
        /// Intended for tests that need to exercise the validator's
        /// chain-binding behavior (e.g. submitting a trade bound to an
        /// out-of-window block hash). The returned tx can be submitted via
        /// [`push_tx`].
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "create_trade")]
        async fn create_trade(
            &self,
            request: CreateTradeRequest,
        ) -> RpcResult<CreateTradeResponse>;
    }
}

#[cfg(test)]
mod test;
