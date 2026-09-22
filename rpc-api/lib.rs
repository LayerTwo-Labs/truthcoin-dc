#![allow(clippy::too_many_arguments)]

use std::net::SocketAddr;

use jsonrpsee::{core::RpcResult, proc_macros::rpc};

use serde::{Deserialize, Serialize};
use truthcoin_dc::{
    authorization::{Dst, Signature},
    net::{Peer, PeerConnectionStatus},
    state::{decisions::DecisionType, markets::MarketId},
    types::{
        Address, Authorization, Authorized, BitcoinOutputContent, Block,
        BlockHash, BlockIndex, BlockIndexDeposit, BlockIndexSpend,
        BlockIndexTx, Body, ClaimDecisionPayload, Coinbase, CoinbaseTxid,
        DecisionClaimEntry, EncryptionPubKey, FilledOutput,
        FilledOutputContent, Header, InPoint, M6id, MainchainSyncPhase,
        MainchainSyncProgress, MerkleRoot, OutPoint, Output, OutputContent,
        PointedOutput, SpentOutput, Transaction, TxData, TxIn, Txid,
        VerifyingKey, WithdrawalBundle, WithdrawalOutputContent,
        schema as truthcoin_schema,
    },
    wallet::{Balance, TransferDests},
};
use utoipa::ToSchema;

mod schema;
#[cfg(test)]
mod test;

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

/// A spent output, and the outpoint that created it
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct PointedSpentOutput {
    pub outpoint: OutPoint,
    pub output: SpentOutput,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct TxInfo {
    pub confirmations: Option<u32>,
    pub fee_sats: u64,
    pub txin: Option<TxIn>,
}

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

/// One transaction the mempool holds
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MempoolTx {
    /// Blake3 over the borsh encoding
    pub txid: Txid,
    /// Borsh size in bytes
    pub size: u64,
    pub tx: Transaction,
    /// Borsh encoding, as hex
    pub raw: String,
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

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketAmplifyBetaRequest {
    pub market_id: String,
    pub amount_sats: u64,
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
    pub period_index: u32,
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
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// Step size for valid vote values. Honored only when
    /// `decision_type == "scaled"`. Defaults to `1.0` when omitted.
    /// `(max - min)` must be an integer multiple of `increment`.
    pub increment: Option<f64>,
    pub tx_fee_sats: u64,
    pub max_listing_fee_sats: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionClaimResponse {
    pub txid: Txid,
    pub decision_ids: Vec<String>,
    pub listing_fee_paid_sats: u64,
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
    pub decision_type: DecisionType,
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
    pub decision_type: DecisionType,
    pub header: String,
    pub description: String,
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
    pub p_period: u64,
    pub p_floor: u64,
    pub mints: u64,
    pub tier_prices: [u64; 5],
    pub last_reprice_block: u32,
    pub period_capacity: u64,
    pub claimed: u64,
}

#[derive(
    Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, ToSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum MarketStatus {
    Trading,
    Cancelled,
    Invalid,
    Settled,
}

impl From<truthcoin_dc::state::markets::MarketState> for MarketStatus {
    fn from(state: truthcoin_dc::state::markets::MarketState) -> Self {
        match state {
            truthcoin_dc::state::markets::MarketState::Trading => Self::Trading,
            truthcoin_dc::state::markets::MarketState::Cancelled => {
                Self::Cancelled
            }
            truthcoin_dc::state::markets::MarketState::Invalid => Self::Invalid,
            truthcoin_dc::state::markets::MarketState::Settled => Self::Settled,
        }
    }
}

impl std::fmt::Display for MarketStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::Trading => "trading",
            Self::Cancelled => "cancelled",
            Self::Invalid => "invalid",
            Self::Settled => "settled",
        };
        f.write_str(value)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketDimension {
    /// Position in the market's ordered dimension specification.
    pub dimension_index: usize,
    /// Hex-encoded decision ID backing this dimension.
    pub decision_id: String,
    /// Decision header, kept separate from outcome names.
    pub name: String,
    pub kind: MarketDimensionKind,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MarketDimensionKind {
    Binary {
        lower_label: String,
        upper_label: String,
    },
    Scaled {
        min: f64,
        max: f64,
        increment: f64,
        /// The current protocol does not carry a unit field, so this is null
        /// until one is added to the decision definition.
        unit: Option<String>,
        lower_label: String,
        upper_label: String,
    },
    Category {
        /// State labels in coordinate order.
        options: Vec<String>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketOutcome {
    /// Dense index used by market_buy and market_sell.
    pub outcome_index: usize,
    /// Human-readable description of this joint outcome.
    pub label: String,
    /// LMSR price for this outcome. For the current market implementation,
    /// this is also the outcome probability.
    pub price: f64,
    pub volume_sats: u64,
    /// Index into the full Cartesian state tensor, including non-tradeable
    /// abstain coordinates.
    pub full_state_index: usize,
    /// One coordinate per ordered market dimension.
    pub coordinates: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketData {
    pub market_id: String,
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub creator_address: String,
    pub created_at_height: u32,
    pub expires_at_height: Option<u32>,
    /// Ordered dimension metadata. The dimension decision IDs are the
    /// authoritative decision list for this market.
    pub dimensions: Vec<MarketDimension>,
    pub outcomes: Vec<MarketOutcome>,
    pub state: MarketStatus,
    /// Effective LMSR beta used for pricing, in satoshis per share unit.
    pub beta: f64,
    /// Author-funded LMSR liquidity base, in satoshis.
    pub liquidity_base_sats: u64,
    /// Exact market treasury balance, in satoshis.
    pub treasury_sats: u64,
    pub total_volume_sats: u64,
    /// Fractional trading fee applied to buys and sells.
    pub trading_fee_rate: f64,
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
    pub label: String,
    pub price: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketSummary {
    pub market_id: String,
    pub title: String,
    pub description: String,
    pub outcome_count: usize,
    pub state: MarketStatus,
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

/// Per-dimension input for `market_create`. Each dimension is either
/// a reference to an already-claimed decision (`Existing`) or a request to
/// claim a new decision (`New`). The wallet derives the `Single` vs
/// `Categorical` `DimensionSpec` from the decision's type.
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DimensionInput {
    /// Reuse an existing on-chain decision by ID (hex).
    Existing { id: String },
    /// Claim a new decision in this same tx.
    /// `decision_type` is "binary", "scaled", or "category".
    /// For "scaled", `min`/`max` are required; `increment` defaults to 1.0.
    /// For "category", `option_labels` (>= 2) are required.
    New {
        period_index: u32,
        decision_type: String,
        header: String,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        option_0_label: Option<String>,
        #[serde(default)]
        option_1_label: Option<String>,
        #[serde(default)]
        option_labels: Option<Vec<String>>,
        #[serde(default)]
        tags: Option<Vec<String>>,
        #[serde(default)]
        min: Option<f64>,
        #[serde(default)]
        max: Option<f64>,
        #[serde(default)]
        increment: Option<f64>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketCreateRequest {
    pub title: String,
    pub description: String,
    pub dimensions: Vec<DimensionInput>,
    pub beta: Option<f64>,
    pub trading_fee: Option<f64>,
    pub initial_liquidity: Option<u64>,
    pub tx_pow_hash_selector: Option<u8>,
    pub tx_pow_ordering: Option<u8>,
    pub tx_pow_difficulty: Option<u8>,
    pub tx_fee_sats: u64,
    /// Optional cap on the total listing fee paid for new claims.
    /// If the computed total exceeds this, the RPC errors before broadcasting.
    pub max_listing_fee_sats: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct ClaimedDecisionInfo {
    pub id: String,
    pub period_index: u32,
    pub listing_fee_paid_sats: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketCreateResponse {
    pub txid: Txid,
    pub market_id: String,
    pub claimed_decisions: Vec<ClaimedDecisionInfo>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct PeriodPricingSummary {
    pub period_index: u32,
    /// Sats charged for the next claim if dropped into this period
    /// (cheapest available unlocked slot's tier price).
    pub cheapest_available_slot_sats: u64,
    /// Tier index (0..=4) of the cheapest available slot.
    pub cheapest_available_tier: u8,
    /// Remaining unlocked slots per tier (length 5).
    pub slots_available_by_tier: Vec<u32>,
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
    /// For scaled decisions, the value must equal `min + k * increment`
    /// for some non-negative integer `k`, with `vote_value <= max`.
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
    /// For scaled decisions, the denormalized value aligned to the
    /// decision's increment. For binary decisions, this is 0.0 or 1.0.
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

#[allow(clippy::double_must_use)]
mod rpc;

pub use rpc::{node, open_api, wallet};
