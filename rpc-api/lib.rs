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

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SlotInfo {
    pub period: u32,
    pub slots: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub enum SlotState {
    Available,
    Claimed,
    Voting,
    Ossified,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SlotFilter {
    pub period: Option<u32>,
    pub status: Option<SlotState>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SlotListItem {
    pub slot_id_hex: String,
    pub period_index: u32,
    pub slot_index: u32,
    pub state: SlotState,
    pub decision: Option<DecisionInfo>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketBuyRequest {
    pub market_id: String,
    pub outcome_index: usize,
    pub shares_amount: f64,
    pub max_cost: Option<u64>,
    pub fee_sats: Option<u64>,
    pub dry_run: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketBuyResponse {
    pub txid: Option<String>,
    /// Total cost in satoshis (base_cost_sats + trading_fee_sats)
    pub cost_sats: u64,
    /// Base LMSR cost that goes to market treasury
    pub base_cost_sats: u64,
    /// Trading fee that goes to market author
    pub trading_fee_sats: u64,
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
    pub redistribution: Option<RedistributionInfo>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionSummary {
    pub slot_id_hex: String,
    pub question: String,
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

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct ConsensusResults {
    pub outcomes: std::collections::HashMap<String, f64>,
    pub first_loading: Vec<f64>,
    pub certainty: f64,
    pub reputation_updates: std::collections::HashMap<String, ReputationUpdate>,
    pub outliers: Vec<String>,
    pub vote_matrix_dimensions: (usize, usize),
    pub algorithm_version: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoterInfoFull {
    pub address: String,
    pub is_registered: bool,
    pub reputation: f64,
    pub votecoin_balance: u32,
    pub total_votes: u64,
    pub periods_active: u32,
    pub accuracy_score: f64,
    pub registered_at_height: u64,
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
pub struct SlotDetails {
    pub slot_id_hex: String,
    pub period_index: u32,
    pub slot_index: u32,
    pub content: SlotContentInfo,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub enum SlotContentInfo {
    Empty,
    Decision(DecisionInfo),
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionInfo {
    pub id: String,
    pub market_maker_pubkey_hash: String,
    pub is_standard: bool,
    pub is_scaled: bool,
    pub question: String,
    pub min: Option<u16>,
    pub max: Option<u16>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct ClaimedSlotSummary {
    pub slot_id_hex: String,
    pub period_index: u32,
    pub slot_index: u32,
    pub market_maker_pubkey_hash: String,
    pub is_standard: bool,
    pub is_scaled: bool,
    pub question_preview: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VotingPeriodInfo {
    pub period: u32,
    pub claimed_slots: u64,
    pub total_slots: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct RedistributionInfo {
    pub period_id: u32,
    pub total_redistributed: u64,
    pub winners_count: u32,
    pub losers_count: u32,
    pub unchanged_count: u32,
    pub conservation_check: i64,
    pub block_height: u64,
    pub is_applied: bool,
    pub slots_affected: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct AvailableSlotId {
    pub period_index: u32,
    pub slot_index: u32,
    pub slot_id_hex: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SlotStatus {
    pub is_testing_mode: bool,
    pub blocks_per_period: u32,
    pub current_period: u32,
    pub current_period_name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct OssifiedSlotInfo {
    pub slot_id_hex: String,
    pub period_index: u32,
    pub slot_index: u32,
    pub decision: Option<DecisionInfo>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketOutcome {
    pub name: String,
    pub current_price: f64,
    pub probability: f64,
    pub volume: f64,
    pub index: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketData {
    pub market_id: String,
    pub title: String,
    pub description: String,
    pub outcomes: Vec<MarketOutcome>,
    pub state: String,
    pub market_maker: String,
    pub expires_at: Option<u64>,
    pub beta: f64,
    pub trading_fee: f64,
    pub tags: Vec<String>,
    pub created_at_height: u64,
    pub treasury: f64,
    pub total_volume: f64,
    pub liquidity: f64,
    pub decision_slots: Vec<String>,
    pub resolution: Option<MarketResolution>,
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
    pub volume: f64,
    pub created_at_height: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SharePosition {
    pub market_id: String,
    pub outcome_index: usize,
    pub outcome_name: String,
    pub shares_held: f64,
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
    pub last_updated_height: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct CreateMarketRequest {
    pub title: String,
    pub description: String,
    pub market_type: String,
    pub decision_slots: Vec<String>,
    pub dimensions: Option<String>,
    pub has_residual: Option<bool>,
    pub beta: Option<f64>,
    pub trading_fee: Option<f64>,
    pub tags: Option<Vec<String>>,
    pub initial_liquidity: Option<u64>,
    pub fee_sats: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct CalculateInitialLiquidityRequest {
    pub beta: f64,
    pub market_type: String,
    pub decision_slots: Option<Vec<String>>,
    pub num_outcomes: Option<usize>,
    pub dimensions: Option<String>,
    pub has_residual: Option<bool>,
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

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct RegisterVoterRequest {
    pub reputation_bond_sats: Option<u64>,
    pub fee_sats: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SubmitVoteRequest {
    pub decision_id: String,
    pub vote_value: f64,
    pub fee_sats: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoteBatchItem {
    pub decision_id: String,
    pub vote_value: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SubmitVoteBatchRequest {
    pub votes: Vec<VoteBatchItem>,
    pub fee_sats: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoterInfo {
    pub address: String,
    pub reputation: f64,
    pub total_votes: u64,
    pub periods_active: u32,
    pub accuracy_score: f64,
    pub registered_at_height: u64,
    pub is_active: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoteInfo {
    pub voter_address: String,
    pub decision_id: String,
    pub vote_value: f64,
    pub period_id: u32,
    pub block_height: u64,
    pub txid: String,
    pub is_batch_vote: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VotingPeriodDetails {
    pub period_id: u32,
    pub start_time: u64,
    pub end_time: u64,
    pub status: String,
    pub decision_slots: Vec<String>,
    pub created_at_height: u64,
    pub total_voters: u64,
    pub active_voters: u64,
    pub total_votes: u64,
    pub consensus_reached: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoterParticipation {
    pub address: String,
    pub period_id: u32,
    pub votes_cast: u32,
    pub decisions_available: u32,
    pub participation_rate: f64,
    pub participated_in_consensus: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct ReputationUpdate {
    pub old_reputation: f64,
    pub new_reputation: f64,
    pub votecoin_proportion: f64,
    pub compliance_score: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VotingConsensusResults {
    pub period_id: u32,
    pub status: String,
    pub outcomes: std::collections::HashMap<String, f64>,
    pub first_loading: Vec<f64>,
    pub certainty: f64,
    pub reputation_updates: std::collections::HashMap<String, ReputationUpdate>,
    pub outliers: Vec<String>,
    pub vote_matrix_dimensions: (usize, usize),
    pub algorithm_version: String,
}

#[open_api(ref_schemas[
    truthcoin_schema::BitcoinAddr, truthcoin_schema::BitcoinBlockHash,
    truthcoin_schema::BitcoinTransaction, truthcoin_schema::BitcoinOutPoint,
    truthcoin_schema::SocketAddr, Address, AssetId, Authorization,
    BitcoinOutputContent, BlockHash, Body,
    CalculateInitialLiquidityRequest, ConsensusResults, CreateMarketRequest, DecisionSummary,
    EncryptionPubKey, FilledOutputContent, Header, InitialLiquidityCalculation,
    MarketBuyRequest, MarketBuyResponse, MarketData, MarketOutcome, MarketSummary,
    MerkleRoot, OutPoint, Output, OutputContent,
    ParticipationStats, PeerConnectionStatus, PeriodStats,
    RedistributionInfo, RegisterVoterRequest, ReputationUpdate,
    SharePosition, Signature, SlotDetails, SlotFilter, SlotInfo, SlotListItem, SlotState, SlotStatus,
    Transaction, TxData, Txid, TxIn, UserHoldings,
    VoteBatchItem, VoteFilter, VoteInfo, VoterInfo, VoterInfoFull,
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

    /// Get a new address
    #[method(name = "get_new_address")]
    async fn get_new_address(&self) -> RpcResult<Address>;

    /// Get new encryption key
    #[method(name = "get_new_encryption_key")]
    async fn get_new_encryption_key(&self) -> RpcResult<EncryptionPubKey>;

    /// Get new verifying/signing key
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

    /// List owned UTXOs
    #[method(name = "my_utxos")]
    async fn my_utxos(
        &self,
    ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>>;

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
        amount: u32,
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

    /// Get slot system status and configuration
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "slot_status")]
    async fn slot_status(&self) -> RpcResult<SlotStatus>;

    /// List slots with optional filtering by period and state
    #[open_api_method(output_schema(ToSchema = "Vec<SlotListItem>"))]
    #[method(name = "slot_list")]
    async fn slot_list(
        &self,
        filter: Option<SlotFilter>,
    ) -> RpcResult<Vec<SlotListItem>>;

    /// Get a specific slot by ID (includes is_voting status)
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "slot_get")]
    async fn slot_get(&self, slot_id: String)
    -> RpcResult<Option<SlotDetails>>;

    /// Claim a decision slot
    #[method(name = "slot_claim")]
    async fn slot_claim(
        &self,
        period_index: u32,
        slot_index: u32,
        is_standard: bool,
        is_scaled: bool,
        question: String,
        min: Option<u16>,
        max: Option<u16>,
        fee_sats: u64,
    ) -> RpcResult<Txid>;

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

    /// Get share positions for an address (optionally filtered by market)
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "market_positions")]
    async fn market_positions(
        &self,
        address: Address,
        market_id: Option<String>,
    ) -> RpcResult<UserHoldings>;

    /// Register as a voter
    #[open_api_method(output_schema(ToSchema = "String"))]
    #[method(name = "vote_register")]
    async fn vote_register(
        &self,
        request: RegisterVoterRequest,
    ) -> RpcResult<String>;

    /// Get full voter information (registration, reputation, participation)
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
        votes: Vec<VoteBatchItem>,
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

    /// Transfer votecoin
    #[method(name = "votecoin_transfer")]
    async fn votecoin_transfer(
        &self,
        dest: Address,
        amount: u32,
        fee_sats: u64,
        memo: Option<String>,
    ) -> RpcResult<Txid>;

    /// Get votecoin balance for an address
    #[open_api_method(output_schema(ToSchema = "u32"))]
    #[method(name = "votecoin_balance")]
    async fn votecoin_balance(&self, address: Address) -> RpcResult<u32>;

    /// Calculate initial liquidity required for market creation
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "calculate_initial_liquidity")]
    async fn calculate_initial_liquidity(
        &self,
        request: CalculateInitialLiquidityRequest,
    ) -> RpcResult<InitialLiquidityCalculation>;
}
