pub mod open_api {
    use l2l_openapi::open_api;

    use crate::{RpcResult, rpc, schema};

    #[open_api]
    #[rpc(client, server)]
    pub trait Rpc {
        /// Get OpenRPC schema
        #[open_api_method(output_schema(ToSchema = "schema::OpenApi"))]
        #[method(name = "openapi_schema")]
        async fn openapi_schema(&self) -> RpcResult<utoipa::openapi::OpenApi>;
    }
}

pub mod node {
    use l2l_openapi::open_api;

    use crate::{
        Address, Authorization, Authorized, BallotItem, BitcoinOutputContent,
        Block, BlockHash, BlockIndex, BlockIndexDeposit, BlockIndexSpend,
        BlockIndexTx, Body, CalculateInitialLiquidityRequest,
        ClaimDecisionPayload, Coinbase, CoinbaseTxid, ConsensusResults,
        DecisionClaimEntry, DecisionContentInfo, DecisionDetails,
        DecisionFilter, DecisionInfo, DecisionListItem, DecisionListingFeeInfo,
        DecisionPeriodStatus, DecisionState, DecisionSummary, DecisionType,
        FilledOutput, FilledOutputContent, Header, InPoint,
        InitialLiquidityCalculation, M6id, MainchainSyncPhase,
        MainchainSyncProgress, MarketData, MarketDimension,
        MarketDimensionKind, MarketId, MarketOutcome, MarketResolution,
        MarketStatus, MarketSummary, MempoolTx, MerkleRoot, OutPoint, Output,
        OutputContent, ParticipationStats, Peer, PeerConnectionStatus,
        PeriodPricingSummary, PeriodStats, PointedOutput, PointedSpentOutput,
        RpcResult, ScoreChange, SharePosition, Signature, SocketAddr,
        SpentOutput, Transaction, TxData, TxIn, TxInfo, Txid, UserHoldings,
        VoteFilter, VoteInfo, VoterInfo, VoterInfoFull, VotingPeriodFull,
        WinningOutcome, WithdrawalBundle, WithdrawalOutputContent, rpc, schema,
        truthcoin_schema,
    };

    #[open_api]
    #[rpc(client, server, server_bounds(Self: super::open_api::RpcServer))]
    pub trait PrivateRpc {
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "connect_peer")]
        async fn connect_peer(
            &self,
            #[open_api_method_arg(schema(
                ToSchema = "truthcoin_schema::SocketAddr"
            ))]
            addr: SocketAddr,
        ) -> RpcResult<()>;

        /// Delete peer from known_peers DB.
        /// Connections to the peer are not terminated.
        #[method(name = "forget_peer")]
        async fn forget_peer(
            &self,
            #[open_api_method_arg(schema(
                ToSchema = "truthcoin_schema::SocketAddr"
            ))]
            addr: SocketAddr,
        ) -> RpcResult<()>;

        /// Invalidate a block and its descendants. If the tip descends from the
        /// block, re-org to the parent of the block.
        #[method(name = "invalidate_block")]
        async fn invalidate_block(
            &self,
            block_hash: BlockHash,
        ) -> RpcResult<()>;

        /// Remove a tx from the mempool
        #[open_api_method(output_schema(ToSchema))]
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

    #[open_api(ref_schemas[
        truthcoin_schema::BitcoinAddr, truthcoin_schema::BitcoinBlockHash,
        truthcoin_schema::BitcoinOutPoint, truthcoin_schema::BitcoinTransaction,
        truthcoin_schema::SocketAddr, Address, Authorization, BallotItem,
        BitcoinOutputContent, BlockHash, BlockIndexDeposit, BlockIndexSpend,
        BlockIndexTx, Body, ClaimDecisionPayload, Coinbase, CoinbaseTxid,
        ConsensusResults,
        DecisionClaimEntry, DecisionContentInfo, DecisionInfo, DecisionState,
        DecisionSummary, DecisionType, FilledOutput, FilledOutputContent,
        Header, InPoint, M6id, MainchainSyncPhase, MarketDimension,
        MarketDimensionKind, MarketId, MarketOutcome, MarketResolution,
        MarketStatus, MerkleRoot, OutPoint, Output, OutputContent,
        ParticipationStats, PeerConnectionStatus, PeriodStats, ScoreChange,
        SharePosition, Signature, SpentOutput, Transaction, TxData, TxIn, Txid,
        WinningOutcome, WithdrawalOutputContent,
    ])]
    #[rpc(client, server, server_bounds(Self: super::open_api::RpcServer))]
    pub trait Rpc {
        /// Wait until the node reaches a specific block height (for sync)
        /// Returns the actual height reached (may be higher than requested)
        /// Times out after the specified milliseconds (default 10000ms)
        #[method(name = "await_block_height")]
        async fn await_block_height(
            &self,
            target_height: u32,
            timeout_ms: Option<u64>,
        ) -> RpcResult<u32>;

        /// Calculate initial liquidity required for market creation
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "calculate_initial_liquidity")]
        async fn calculate_initial_liquidity(
            &self,
            request: CalculateInitialLiquidityRequest,
        ) -> RpcResult<InitialLiquidityCalculation>;

        /// Connect a block for which a BMM request was included in the specified
        /// mainchain block. Returns `true` if it was accepted as the new tip.
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "connect_block")]
        async fn connect_block(
            &self,
            block: Block,
            #[open_api_method_arg(schema(
                PartialSchema = "truthcoin_schema::BitcoinBlockHash"
            ))]
            main_block_hash: bitcoin::BlockHash,
        ) -> RpcResult<bool>;

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

        /// Get the best mainchain block hash known by Thunder
        #[open_api_method(output_schema(
            PartialSchema = "schema::Optional<truthcoin_schema::BitcoinBlockHash>"
        ))]
        #[method(name = "get_best_mainchain_block_hash")]
        async fn get_best_mainchain_block_hash(
            &self,
        ) -> RpcResult<Option<bitcoin::BlockHash>>;

        /// Get the best sidechain block hash known by Truthcoin
        #[open_api_method(output_schema(
            PartialSchema = "schema::Optional<BlockHash>"
        ))]
        #[method(name = "get_best_sidechain_block_hash")]
        async fn get_best_sidechain_block_hash(
            &self,
        ) -> RpcResult<Option<BlockHash>>;

        /// Get block data
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "get_block")]
        async fn get_block(&self, block_hash: BlockHash) -> RpcResult<Block>;

        /// Get the block hash at the specified height in the current chain,
        /// if it exists
        #[open_api_method(output_schema(
            PartialSchema = "schema::Optional<BlockHash>"
        ))]
        #[method(name = "get_block_hash")]
        async fn get_block_hash(
            &self,
            height: u32,
        ) -> RpcResult<Option<BlockHash>>;

        /// Get the transaction ids, sizes and encodings of a block, with the
        /// mainchain deposits and withdrawal bundle spends it applied
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "get_block_index")]
        async fn get_block_index(
            &self,
            block_hash: BlockHash,
        ) -> RpcResult<BlockIndex>;

        /// Get mainchain blocks that commit to a specified block hash
        #[open_api_method(output_schema(
            PartialSchema = "truthcoin_schema::BitcoinBlockHash"
        ))]
        #[method(name = "get_bmm_inclusions")]
        async fn get_bmm_inclusions(
            &self,
            block_hash: truthcoin_dc::types::BlockHash,
        ) -> RpcResult<Vec<bitcoin::BlockHash>>;

        /// Get stxos for addresses
        #[method(name = "get_stxos")]
        async fn get_stxos(
            &self,
            addresses: std::collections::HashSet<Address>,
        ) -> RpcResult<Vec<PointedSpentOutput>>;

        /// Get transaction by txid
        #[method(name = "get_transaction")]
        async fn get_transaction(
            &self,
            txid: Txid,
        ) -> RpcResult<Option<Transaction>>;

        /// Get information about a transaction in the current chain
        #[method(name = "get_transaction_info")]
        async fn get_transaction_info(
            &self,
            txid: Txid,
        ) -> RpcResult<Option<TxInfo>>;

        /// Get utxos for addresses
        #[method(name = "get_utxos")]
        async fn get_utxos(
            &self,
            addresses: std::collections::HashSet<Address>,
        ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>>;

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

        /// List peers
        #[method(name = "list_peers")]
        async fn list_peers(&self) -> RpcResult<Vec<Peer>>;

        /// List all UTXOs
        #[open_api_method(output_schema(
            ToSchema = "Vec<PointedOutput<FilledOutputContent>>"
        ))]
        #[method(name = "list_utxos")]
        async fn list_utxos(
            &self,
        ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>>;

        /// Get the progress of the sync with the mainchain
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "mainchain_sync_progress")]
        async fn mainchain_sync_progress(
            &self,
        ) -> RpcResult<MainchainSyncProgress>;

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

        /// Get share positions for an address (optionally filtered by market)
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "market_positions")]
        async fn market_positions(
            &self,
            address: Address,
            market_id: Option<String>,
        ) -> RpcResult<UserHoldings>;

        /// Get pending withdrawal bundle
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "pending_withdrawal_bundle")]
        async fn pending_withdrawal_bundle(
            &self,
        ) -> RpcResult<Option<WithdrawalBundle>>;

        /// Submit a hex-encoded borsh-serialized `AuthorizedTransaction` directly
        /// to the mempool. Returns the transaction id on success.
        ///
        /// Intended for tests and advanced clients that need to submit a
        /// pre-signed transaction (e.g. to exercise validator paths like
        /// stale `prev_block_hash` rejection).
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "push_tx")]
        async fn push_tx(&self, tx_hex: String) -> RpcResult<Txid>;

        /// Get total sidechain wealth in sats
        #[method(name = "sidechain_wealth")]
        async fn sidechain_wealth_sats(&self) -> RpcResult<u64>;

        /// Verify and broadcast a transaction
        #[method(name = "submit_transaction")]
        async fn submit_transaction(
            &self,
            transaction: Authorized<Transaction>,
        ) -> RpcResult<Txid>;

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
    use l2l_openapi::open_api;

    use crate::{
        Address, Authorization, Authorized, Balance, BallotItem,
        BitcoinOutputContent, Block, BlockHash, Body, ClaimDecisionPayload,
        ClaimedDecisionInfo, Coinbase, CoinbaseTxid, CreateTradeRequest,
        CreateTradeResponse, DecisionClaimEntry, DecisionClaimItem,
        DecisionClaimRequest, DecisionClaimResponse, DimensionInput, Dst,
        EncryptionPubKey, FilledOutput, FilledOutputContent,
        GetBlockTemplateResponse, Header, MarketAmplifyBetaRequest,
        MarketBuyRequest, MarketBuyResponse, MarketCreateRequest,
        MarketCreateResponse, MarketId, MarketSellRequest, MarketSellResponse,
        MerkleRoot, OutPoint, Output, OutputContent, PointedOutput, RpcResult,
        Signature, Transaction, TransferDests, TxData, Txid, VerifyingKey,
        WithdrawalOutputContent, rpc, schema, truthcoin_schema,
    };

    #[open_api(ref_schemas[
        truthcoin_schema::BitcoinAddr, truthcoin_schema::BitcoinBlockHash,
        truthcoin_schema::BitcoinOutPoint, Address, Authorization, BallotItem,
        BitcoinOutputContent, Block, BlockHash, Body, ClaimDecisionPayload,
        Coinbase, CoinbaseTxid,
        ClaimedDecisionInfo, DecisionClaimEntry, DecisionClaimItem,
        DimensionInput, FilledOutput, Header, MarketId, MerkleRoot, OutPoint,
        Output, OutputContent, Signature, Transaction, TxData, Txid,
        WithdrawalOutputContent,
    ])]
    #[rpc(client, server, server_bounds(Self: super::open_api::RpcServer))]
    pub trait Rpc {
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "bitcoin_balance")]
        async fn bitcoin_balance(&self) -> RpcResult<Balance>;

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

        #[method(name = "format_deposit_address")]
        async fn format_deposit_address(
            &self,
            address: Address,
        ) -> RpcResult<String>;

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

        /// Generate a new address
        #[method(name = "get_new_address")]
        async fn get_new_address(&self) -> RpcResult<Address>;

        /// Get the voter address (index 0), used for reputation
        /// and voting identity
        #[method(name = "get_voter_address")]
        async fn get_voter_address(&self) -> RpcResult<Address>;

        /// Generate new encryption key
        #[method(name = "get_new_encryption_key")]
        async fn get_new_encryption_key(&self) -> RpcResult<EncryptionPubKey>;

        /// Generate new verifying/signing key
        #[method(name = "get_new_verifying_key")]
        async fn get_new_verifying_key(&self) -> RpcResult<VerifyingKey>;

        /// Get wallet addresses, sorted by base58 encoding
        #[method(name = "get_wallet_addresses")]
        async fn get_wallet_addresses(&self) -> RpcResult<Vec<Address>>;

        /// Get wallet UTXOs
        #[method(name = "get_wallet_utxos")]
        async fn get_wallet_utxos(
            &self,
        ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>>;

        /// Attempt to mine a sidechain block
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "mine")]
        async fn mine(&self, fee: Option<u64>) -> RpcResult<()>;

        /// List unconfirmed owned UTXOs
        #[method(name = "my_unconfirmed_utxos")]
        async fn my_unconfirmed_utxos(&self) -> RpcResult<Vec<PointedOutput>>;

        /// Set the wallet seed from a mnemonic seed phrase
        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "set_seed_from_mnemonic")]
        async fn set_seed_from_mnemonic(
            &self,
            mnemonic: String,
        ) -> RpcResult<()>;

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

        /// Sign a transaction, and optionally broadcast it.
        #[method(name = "sign_transaction")]
        async fn sign_transaction(
            &self,
            transaction: Transaction,
            broadcast: Option<bool>,
        ) -> RpcResult<Authorized<Transaction>>;

        /// Transfer funds to the specified address
        #[method(name = "transfer")]
        async fn transfer(
            &self,
            dest: Address,
            value: u64,
            fee: u64,
            memo: Option<String>,
        ) -> RpcResult<Txid>;

        /// Transfer funds to each address in `dests`, which maps an address to a
        /// value in sats
        #[method(name = "transfer_many")]
        async fn transfer_many(
            &self,
            dests: TransferDests,
            fee_sats: u64,
        ) -> RpcResult<Txid>;

        /// Transfer votecoin to the specified address
        #[method(name = "transfer_votecoin")]
        async fn transfer_votecoin(
            &self,
            dest: Address,
            amount: f64,
            fee_sats: u64,
            memo: Option<String>,
        ) -> RpcResult<Txid>;

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

        /// Initiate a withdrawal to the specified mainchain address
        #[method(name = "withdraw")]
        async fn withdraw(
            &self,
            #[open_api_method_arg(schema(
                PartialSchema = "truthcoin_schema::BitcoinAddr"
            ))]
            mainchain_address: bitcoin::Address<
                bitcoin::address::NetworkUnchecked,
            >,
            amount_sats: u64,
            fee_sats: u64,
            mainchain_fee_sats: u64,
        ) -> RpcResult<Txid>;

        #[open_api_method(output_schema(ToSchema))]
        #[method(name = "refresh_wallet")]
        async fn refresh_wallet(&self) -> RpcResult<()>;

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
