#![allow(clippy::too_many_arguments)]

use std::net::SocketAddr;

use jsonrpsee::{core::RpcResult, proc_macros::rpc};
use l2l_openapi::open_api;

use serde::{Deserialize, Serialize};
use truthcoin_dc::{
    authorization::{Dst, Signature},
    net::{Peer, PeerConnectionStatus},
    types::{
        Address, AssetId, Authorization, BitcoinOutputContent, Block,
        BlockHash, Body, EncryptionPubKey, FilledOutputContent, Header,
        MerkleRoot, OutPoint, Output, OutputContent, PointedOutput,
        Transaction, TxData, TxIn, Txid, VerifyingKey, WithdrawalBundle,
        WithdrawalOutputContent, schema as truthcoin_schema,
    },
    wallet::Balance,
};
use utoipa::ToSchema;

mod schema;

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct TxInfo {
    pub confirmations: Option<u32>,
    pub fee_sats: u64,
    pub txin: Option<TxIn>,
}

pub use truthcoin_dc::state::decisions::DecisionState;

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionFilter {
    pub period: Option<u32>,
    pub status: Option<DecisionState>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionListItem {
    pub decision_id_hex: String,
    pub period_index: u32,
    pub decision_index: u32,
    pub state: DecisionState,
    pub decision: Option<DecisionInfo>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketBuyRequest {
    pub market_id: String,
    pub outcome_index: usize,
    pub shares_amount: i64,
    pub max_cost: Option<u64>,
    pub dry_run: Option<bool>,
}

/// Request to build and sign (but not submit) a Trade transaction with
/// a caller-supplied `prev_block_hash`.
///
/// `shares_amount` is positive for buy, negative for sell. For sells,
/// `trader_address` must be supplied and must own the shares.
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct CreateTradeRequest {
    pub market_id: String,
    pub outcome_index: usize,
    pub shares_amount: i64,
    pub limit_sats: u64,
    /// Address of the share-holder (required for sells; ignored for buys,
    /// in which case the wallet's first address is used as trader).
    pub trader_address: Option<Address>,
    /// Hex-encoded block hash to bind the trade's PoW preimage and
    /// chain-recency check to.
    pub prev_block_hash: String,
}

/// Hex-encoded signed `AuthorizedTransaction` plus its txid.
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct CreateTradeResponse {
    pub signed_tx_hex: String,
    pub txid: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionClaimItem {
    pub decision_id_hex: String,
    pub header: String,
    pub description: Option<String>,
    pub option_0_label: Option<String>,
    pub option_1_label: Option<String>,
    pub option_labels: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionClaimRequest {
    pub decision_type: String,
    pub decisions: Vec<DecisionClaimItem>,
    pub min: Option<i64>,
    pub max: Option<i64>,
    pub fee_sats: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketBuyResponse {
    pub txid: Option<String>,
    /// Total estimated cost in satoshis (LMSR cost + trading fee)
    pub cost_sats: u64,
    /// Trading fee that goes to market author
    pub trading_fee_sats: u64,
    pub new_price: f64,
}

/// Request to sell shares in a prediction market
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketSellRequest {
    pub market_id: String,
    pub outcome_index: usize,
    pub shares_amount: i64,
    /// Address holding the shares to sell (required)
    pub seller_address: Address,
    /// Minimum proceeds required (slippage protection)
    pub min_proceeds: Option<u64>,
    pub dry_run: Option<bool>,
}

/// Response from selling shares in a prediction market
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketSellResponse {
    /// Transaction ID for the sell transaction (None for dry runs)
    pub txid: Option<String>,
    /// Gross proceeds before trading fee (LMSR payout)
    pub proceeds_sats: u64,
    /// Trading fee deducted from proceeds
    pub trading_fee_sats: u64,
    /// Net proceeds seller will receive (proceeds_sats - trading_fee_sats)
    pub net_proceeds_sats: u64,
    pub new_price: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoteFilter {
    pub voter: Option<Address>,
    pub decision_id: Option<String>,
    pub period_id: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VotingPeriodFull {
    pub period_id: u32,
    pub status: String,
    pub start_height: u32,
    pub end_height: u32,
    pub start_time: u64,
    pub end_time: u64,
    pub decisions: Vec<DecisionSummary>,
    pub stats: PeriodStats,
    pub consensus: Option<ConsensusResults>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionSummary {
    pub decision_id_hex: String,
    pub header: String,
    pub is_standard: bool,
    pub is_scaled: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct PeriodStats {
    pub total_voters: u64,
    pub active_voters: u64,
    pub total_votes: u64,
    pub participation_rate: f64,
}

/// Results from the SVD consensus algorithm for a voting period.
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct ConsensusResults {
    /// Final consensus outcomes for each decision (decision_id_hex -> value).
    /// For scaled decisions, values are in real units (e.g., 270 electoral votes).
    /// For binary decisions, values are 0.0 or 1.0.
    pub outcomes: std::collections::HashMap<String, f64>,
    pub first_loading: Vec<f64>,
    pub certainty: f64,
    pub score_changes: std::collections::HashMap<String, ScoreChange>,
    pub outliers: Vec<String>,
    pub vote_matrix_dimensions: (usize, usize),
    pub algorithm_version: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoterInfoFull {
    pub address: String,
    pub votecoin_balance: f64,
    pub total_votes: u64,
    pub periods_active: u32,
    pub is_active: bool,
    pub current_period_participation: Option<ParticipationStats>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct ParticipationStats {
    pub period_id: u32,
    pub votes_cast: u32,
    pub decisions_available: u32,
    pub participation_rate: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionDetails {
    pub decision_id_hex: String,
    pub period_index: u32,
    pub decision_index: u32,
    pub content: DecisionContentInfo,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub enum DecisionContentInfo {
    Empty,
    Decision(DecisionInfo),
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionInfo {
    pub id: String,
    pub market_maker_pubkey_hash: String,
    pub is_standard: bool,
    pub is_scaled: bool,
    pub header: String,
    pub description: String,
    pub min: Option<i64>,
    pub max: Option<i64>,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionPeriodStatus {
    pub is_testing_mode: bool,
    pub blocks_per_period: u32,
    pub current_period: u32,
    pub current_period_name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionListingFeeInfo {
    pub next_fee: u64,
    pub claimed_standard_count: u64,
    pub available: u64,
    pub free_count: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketOutcome {
    pub name: String,
    pub current_price: f64,
    pub probability: f64,
    pub volume_sats: u64,
    /// The internal state array index used by market_buy/market_sell
    pub index: usize,
    /// The ordinal display position (0-based) among valid outcomes
    pub display_index: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketData {
    pub market_id: String,
    pub title: String,
    pub description: String,
    pub outcomes: Vec<MarketOutcome>,
    pub state: String,
    pub market_maker: String,
    pub expires_at: Option<u32>,
    pub beta: f64,
    pub trading_fee: f64,
    pub tags: Vec<String>,
    pub created_at_height: u32,
    pub treasury: f64,
    pub total_volume_sats: u64,
    pub liquidity: f64,
    pub decision_ids: Vec<String>,
    pub resolution: Option<MarketResolution>,
    pub tx_pow_hash_selector: u8,
    pub tx_pow_ordering: u8,
    pub tx_pow_difficulty: u8,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketResolution {
    pub winning_outcomes: Vec<WinningOutcome>,
    pub summary: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct WinningOutcome {
    pub outcome_index: usize,
    pub outcome_name: String,
    pub final_price: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketSummary {
    pub market_id: String,
    pub title: String,
    pub description: String,
    pub outcome_count: usize,
    pub state: String,
    pub volume_sats: u64,
    pub created_at_height: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SharePosition {
    pub market_id: String,
    pub outcome_index: usize,
    pub outcome_name: String,
    pub shares: i64,
    pub avg_purchase_price: f64,
    pub current_price: f64,
    pub current_value: f64,
    pub unrealized_pnl: f64,
    pub cost_basis: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct UserHoldings {
    pub address: String,
    pub positions: Vec<SharePosition>,
    pub total_value: f64,
    pub total_cost_basis: f64,
    pub total_unrealized_pnl: f64,
    pub active_markets: usize,
    pub last_updated_height: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct CreateMarketRequest {
    pub title: String,
    pub description: String,
    /// Dimension specification in bracket notation.
    /// Examples: "[004008]", "[004008,004009]", "[[004008,004009]]"
    pub dimensions: String,
    /// Advanced: LMSR liquidity parameter controlling price sensitivity.
    /// Higher beta = more liquid = smaller price moves per trade.
    /// Mutually exclusive with initial_liquidity - specify one or the other.
    pub beta: Option<f64>,
    pub trading_fee: Option<f64>,
    /// Initial liquidity in satoshis to fund the market (recommended).
    /// Beta is derived: β = liquidity / ln(num_outcomes)
    /// Mutually exclusive with beta - specify one or the other.
    pub initial_liquidity: Option<u64>,
    /// Txid(s) for categorical dimensions in hex format - required when using [[...]] notation
    /// Each txid must be a ClaimDecision transaction with Category type
    pub category_txids: Option<Vec<String>>,
    /// Names for residual outcomes in categorical dimensions (one per [[...]])
    /// e.g., ["Bengals"] when explicit decisions are Steelers/Ravens/Browns
    pub residual_names: Option<Vec<String>>,
    pub tx_pow_hash_selector: Option<u8>,
    pub tx_pow_ordering: Option<u8>,
    pub tx_pow_difficulty: Option<u8>,
    pub fee_sats: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct CalculateInitialLiquidityRequest {
    pub beta: f64,
    /// Dimension specification in bracket notation (alternative to num_outcomes)
    pub dimensions: Option<String>,
    /// Number of outcomes (alternative to dimensions)
    pub num_outcomes: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct InitialLiquidityCalculation {
    pub beta: f64,
    pub num_outcomes: usize,
    pub initial_liquidity_sats: u64,
    pub min_treasury_sats: u64,
    pub market_config: String,
    pub outcome_breakdown: String,
}

/// A single vote in a ballot submission.
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct BallotItem {
    pub decision_id: String,
    /// The vote value in real units (e.g., 270 for electoral votes).
    /// For scaled decisions with min/max bounds, enter the actual value
    /// within those bounds. The system handles internal normalization.
    /// For binary decisions, use 0.0 (No) or 1.0 (Yes).
    pub vote_value: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SubmitBallotRequest {
    pub votes: Vec<BallotItem>,
    pub fee_sats: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoterInfo {
    pub address: String,
    pub votecoin_balance: f64,
    pub total_votes: u64,
    pub is_active: bool,
}

/// Information about a recorded vote.
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoteInfo {
    pub voter_address: String,
    pub decision_id: String,
    /// The vote value in real units (e.g., 270 for electoral votes).
    /// For scaled decisions, this is the denormalized value within the
    /// decision's min/max bounds. For binary decisions, this is 0.0 or 1.0.
    pub vote_value: f64,
    pub period_id: u32,
    pub block_height: u32,
    pub txid: String,
    pub is_batch_vote: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct ScoreChange {
    pub old_score: f64,
    pub new_score: f64,
}

#[open_api(ref_schemas[
    truthcoin_schema::BitcoinAddr, truthcoin_schema::BitcoinBlockHash,
    truthcoin_schema::BitcoinTransaction, truthcoin_schema::BitcoinOutPoint,
    truthcoin_schema::SocketAddr, Address, AssetId, Authorization,
    BitcoinOutputContent, BlockHash, Body,
    CalculateInitialLiquidityRequest, DecisionClaimItem, DecisionClaimRequest,
    ConsensusResults, CreateMarketRequest, DecisionSummary,
    EncryptionPubKey, FilledOutputContent, Header, InitialLiquidityCalculation,
    MarketBuyRequest, MarketBuyResponse, MarketData, MarketOutcome,
    MarketSellRequest, MarketSellResponse, MarketSummary,
    MerkleRoot, OutPoint, Output, OutputContent,
    ParticipationStats, PeerConnectionStatus, PeriodStats,
    ScoreChange,
    SharePosition, Signature, DecisionDetails, DecisionFilter, DecisionListItem, DecisionListingFeeInfo, DecisionState, DecisionPeriodStatus,
    Transaction, TxData, Txid, TxIn, UserHoldings,
    BallotItem, VoteFilter, VoteInfo, VoterInfo, VoterInfoFull,
    VotingPeriodFull, WithdrawalOutputContent, VerifyingKey,
])]
#[rpc(client, server)]
pub trait Rpc {
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "bitcoin_balance")]
    async fn bitcoin_balance(&self) -> RpcResult<Balance>;

    #[open_api_method(output_schema(PartialSchema = "schema::BitcoinTxid"))]
    #[method(name = "create_deposit")]
    async fn create_deposit(
        &self,
        address: Address,
        value_sats: u64,
        fee_sats: u64,
    ) -> RpcResult<bitcoin::Txid>;

    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "connect_peer")]
    async fn connect_peer(
        &self,
        #[open_api_method_arg(schema(
            ToSchema = "truthcoin_schema::SocketAddr"
        ))]
        addr: SocketAddr,
    ) -> RpcResult<()>;

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

    #[method(name = "format_deposit_address")]
    async fn format_deposit_address(
        &self,
        address: Address,
    ) -> RpcResult<String>;

    #[method(name = "generate_mnemonic")]
    async fn generate_mnemonic(&self) -> RpcResult<String>;

    /// Get block data
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "get_block")]
    async fn get_block(&self, block_hash: BlockHash) -> RpcResult<Block>;

    /// Get mainchain blocks that commit to a specified block hash
    #[open_api_method(output_schema(
        PartialSchema = "truthcoin_schema::BitcoinBlockHash"
    ))]
    #[method(name = "get_bmm_inclusions")]
    async fn get_bmm_inclusions(
        &self,
        block_hash: truthcoin_dc::types::BlockHash,
    ) -> RpcResult<Vec<bitcoin::BlockHash>>;

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

    /// Get wallet addresses, sorted by base58 encoding
    #[method(name = "get_wallet_addresses")]
    async fn get_wallet_addresses(&self) -> RpcResult<Vec<Address>>;

    /// Get wallet UTXOs
    #[method(name = "get_wallet_utxos")]
    async fn get_wallet_utxos(
        &self,
    ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>>;

    /// Get the current block count
    #[method(name = "getblockcount")]
    async fn getblockcount(&self) -> RpcResult<u32>;

    /// Get the height of the latest failed withdrawal bundle
    #[method(name = "latest_failed_withdrawal_bundle_height")]
    async fn latest_failed_withdrawal_bundle_height(
        &self,
    ) -> RpcResult<Option<u32>>;

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

    /// Attempt to mine a sidechain block
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "mine")]
    async fn mine(&self, fee: Option<u64>) -> RpcResult<()>;

    /// List unconfirmed owned UTXOs
    #[method(name = "my_unconfirmed_utxos")]
    async fn my_unconfirmed_utxos(&self) -> RpcResult<Vec<PointedOutput>>;

    /// Get pending withdrawal bundle
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "pending_withdrawal_bundle")]
    async fn pending_withdrawal_bundle(
        &self,
    ) -> RpcResult<Option<WithdrawalBundle>>;

    /// Get OpenRPC schema
    #[open_api_method(output_schema(ToSchema = "schema::OpenApi"))]
    #[method(name = "openapi_schema")]
    async fn openapi_schema(&self) -> RpcResult<utoipa::openapi::OpenApi>;

    /// Remove a tx from the mempool
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "remove_from_mempool")]
    async fn remove_from_mempool(&self, txid: Txid) -> RpcResult<()>;

    /// Set the wallet seed from a mnemonic seed phrase
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "set_seed_from_mnemonic")]
    async fn set_seed_from_mnemonic(&self, mnemonic: String) -> RpcResult<()>;

    /// Get total sidechain wealth in sats
    #[method(name = "sidechain_wealth")]
    async fn sidechain_wealth_sats(&self) -> RpcResult<u64>;

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

    /// Stop the node
    #[method(name = "stop")]
    async fn stop(&self);

    /// Transfer funds to the specified address
    #[method(name = "transfer")]
    async fn transfer(
        &self,
        dest: Address,
        value: u64,
        fee: u64,
        memo: Option<String>,
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

    /// Wait until the node reaches a specific block height (for sync)
    /// Returns the actual height reached (may be higher than requested)
    /// Times out after the specified milliseconds (default 10000ms)
    #[method(name = "await_block_height")]
    async fn await_block_height(
        &self,
        target_height: u32,
        timeout_ms: Option<u64>,
    ) -> RpcResult<u32>;

    /// Trigger a sync to a specific tip block hash.
    /// The block must already exist in our archive (received via P2P).
    /// Returns true if reorg was successful, false if not needed or failed.
    #[method(name = "sync_to_tip")]
    async fn sync_to_tip(&self, block_hash: BlockHash) -> RpcResult<bool>;

    /// Get decision system status and configuration
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "decision_status")]
    async fn decision_status(&self) -> RpcResult<DecisionPeriodStatus>;

    /// List decisions with optional filtering by period and state
    #[open_api_method(output_schema(ToSchema = "Vec<DecisionListItem>"))]
    #[method(name = "decision_list")]
    async fn decision_list(
        &self,
        filter: Option<DecisionFilter>,
    ) -> RpcResult<Vec<DecisionListItem>>;

    /// Get a specific decision by ID (includes is_voting status)
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "decision_get")]
    async fn decision_get(
        &self,
        decision_id: String,
    ) -> RpcResult<Option<DecisionDetails>>;

    /// Claim one or more decisions.
    /// decision_type: "binary", "scaled", or "category"
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "decision_claim")]
    async fn decision_claim(
        &self,
        request: DecisionClaimRequest,
    ) -> RpcResult<Txid>;

    /// Get listing fee info for a period
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "decision_listing_fee")]
    async fn decision_listing_fee(
        &self,
        period: u32,
    ) -> RpcResult<DecisionListingFeeInfo>;

    /// Create a new prediction market
    #[method(name = "market_create")]
    async fn market_create(
        &self,
        request: CreateMarketRequest,
    ) -> RpcResult<String>;

    /// List all markets
    #[open_api_method(output_schema(ToSchema = "Vec<MarketSummary>"))]
    #[method(name = "market_list")]
    async fn market_list(&self) -> RpcResult<Vec<MarketSummary>>;

    /// Get detailed market information
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "market_get")]
    async fn market_get(
        &self,
        market_id: String,
    ) -> RpcResult<Option<MarketData>>;

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

    /// Get share positions for an address (optionally filtered by market)
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "market_positions")]
    async fn market_positions(
        &self,
        address: Address,
        market_id: Option<String>,
    ) -> RpcResult<UserHoldings>;

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

    /// Submit one or more votes (batch)
    #[open_api_method(output_schema(ToSchema = "String"))]
    #[method(name = "vote_submit")]
    async fn vote_submit(
        &self,
        votes: Vec<BallotItem>,
        fee_sats: u64,
    ) -> RpcResult<String>;

    /// Query votes with filters (by voter, decision, or period)
    #[open_api_method(output_schema(ToSchema = "Vec<VoteInfo>"))]
    #[method(name = "vote_list")]
    async fn vote_list(&self, filter: VoteFilter) -> RpcResult<Vec<VoteInfo>>;

    /// Get full voting period information (null period_id = current)
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "vote_period")]
    async fn vote_period(
        &self,
        period_id: Option<u32>,
    ) -> RpcResult<Option<VotingPeriodFull>>;

    /// Get votecoin balance for an address
    #[open_api_method(output_schema(ToSchema = "f64"))]
    #[method(name = "votecoin_balance")]
    async fn votecoin_balance(&self, address: Address) -> RpcResult<f64>;

    /// Calculate initial liquidity required for market creation
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "calculate_initial_liquidity")]
    async fn calculate_initial_liquidity(
        &self,
        request: CalculateInitialLiquidityRequest,
    ) -> RpcResult<InitialLiquidityCalculation>;

    /// Submit a hex-encoded borsh-serialized `AuthorizedTransaction` directly
    /// to the mempool. Returns the transaction id on success.
    ///
    /// Intended for tests and advanced clients that need to submit a
    /// pre-signed transaction (e.g. to exercise validator paths like
    /// stale `prev_block_hash` rejection).
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "push_tx")]
    async fn push_tx(&self, tx_hex: String) -> RpcResult<Txid>;

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
