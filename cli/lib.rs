use std::{
    net::{Ipv4Addr, SocketAddr},
    time::Duration,
};

use clap::{ArgAction, Parser, Subcommand};
use http::HeaderMap;
use jsonrpsee::{core::client::ClientT, http_client::HttpClientBuilder};
use serde::Serialize;
use tracing_subscriber::layer::SubscriberExt as _;
use truthcoin_dc::{
    authorization::{Dst, Signature},
    types::{
        Address, BlockHash, EncryptionPubKey, THIS_SIDECHAIN, Txid,
        VerifyingKey,
    },
};
use truthcoin_dc_app_rpc_api::RpcClient;
use url::{Host, Url};

/// Parse comma-separated input into filtered string vector
pub fn parse_comma_separated(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Validate that a list contains at least one item
pub fn validate_non_empty_list(
    list: &[String],
    name: &str,
) -> Result<(), String> {
    if list.is_empty() {
        Err(format!("Error: At least one {name} is required"))
    } else {
        Ok(())
    }
}

/// Extract decision slots from dimensions specification
/// Dimensions format: "[slot1,[slot2,slot3],slot4]" or "[slot1,slot2]"
pub fn extract_slots_from_dimensions(
    dimensions: &str,
) -> Result<Vec<String>, String> {
    use regex::Regex;

    // Remove outer brackets and whitespace
    let cleaned = dimensions
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']');

    // Extract all slot IDs using regex pattern
    let slot_pattern = Regex::new(r"[a-fA-F0-9]{6}")
        .map_err(|e| format!("Regex error: {e}"))?;
    let slots: Vec<String> = slot_pattern
        .find_iter(cleaned)
        .map(|m| m.as_str().to_string())
        .collect();

    if slots.is_empty() {
        return Err(
            "Error: No valid decision slots found in dimensions specification"
                .to_string(),
        );
    }

    Ok(slots)
}

/// Format transaction success messages consistently
pub fn format_tx_success(
    operation: &str,
    details: Option<&str>,
    txid: &str,
) -> String {
    match details {
        Some(details) => {
            format!("{operation} successful ({details}): {txid}")
        }
        None => format!("{operation} successful: {txid}"),
    }
}

/// Truncate strings for display with ellipsis
pub fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

/// Format slot display consistently
pub fn format_slot_info(slot_id: &str, period: u32, index: u32) -> String {
    format!("Slot {slot_id} (Period {period}, Index {index})")
}

pub fn json_response<T>(data: &T) -> anyhow::Result<String>
where
    T: Serialize,
{
    Ok(serde_json::to_string_pretty(data)?)
}

#[derive(Clone, Debug, Subcommand)]
#[command(arg_required_else_help(true))]
pub enum Command {
    // === SYSTEM COMMANDS ===
    /// Check node status and connection
    #[command(name = "status", alias = "stat", alias = "s")]
    Status,

    /// Stop the node
    #[command(name = "stop", alias = "shutdown")]
    Stop,

    /// Attempt to mine a sidechain block
    #[command(name = "mine", alias = "m")]
    Mine {
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Show OpenAPI schema
    #[command(name = "openapi-schema", alias = "schema")]
    OpenApiSchema,

    // === WALLET COMMANDS ===
    /// Get wallet balance in sats
    #[command(name = "balance", alias = "bal", alias = "b")]
    Balance,

    /// Get a new address
    #[command(name = "get-new-address", alias = "addr")]
    GetNewAddress,

    /// Get wallet addresses
    #[command(name = "get-wallet-addresses")]
    GetWalletAddresses,

    /// List owned UTXOs
    #[command(name = "my-utxos", alias = "utxos")]
    MyUtxos,

    /// List unconfirmed owned UTXOs
    #[command(name = "my-unconfirmed-utxos")]
    MyUnconfirmedUtxos,

    /// Get wallet UTXOs
    #[command(name = "get-wallet-utxos")]
    GetWalletUtxos,

    /// List all UTXOs
    #[command(name = "list-utxos")]
    ListUtxos,

    /// Transfer funds to address
    #[command(name = "transfer", alias = "send")]
    Transfer {
        dest: Address,
        #[arg(long)]
        value_sats: u64,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Initiate withdrawal to mainchain
    #[command(name = "withdraw")]
    Withdraw {
        mainchain_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        #[arg(long)]
        amount_sats: u64,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
        #[arg(long, default_value = "1000")]
        mainchain_fee_sats: u64,
    },

    /// Deposit to address
    #[command(name = "create-deposit", alias = "deposit")]
    CreateDeposit {
        address: Address,
        #[arg(long)]
        value_sats: u64,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Format a deposit address
    #[command(name = "format-deposit-address")]
    FormatDepositAddress { address: Address },

    /// Generate mnemonic seed phrase
    #[command(name = "generate-mnemonic", alias = "mnemonic")]
    GenerateMnemonic,

    /// Set wallet seed from mnemonic
    #[command(name = "set-seed-from-mnemonic")]
    SetSeedFromMnemonic { mnemonic: String },

    /// Get total sidechain wealth
    #[command(name = "sidechain-wealth")]
    SidechainWealth,

    // === BLOCKCHAIN COMMANDS ===
    /// Get current block count
    #[command(name = "get-block-count", alias = "height")]
    GetBlockCount,

    /// Get block data
    #[command(name = "get-block")]
    GetBlock { block_hash: BlockHash },

    /// Get best mainchain block hash
    #[command(name = "get-best-mainchain-block-hash")]
    GetBestMainchainBlockHash,

    /// Get best sidechain block hash
    #[command(name = "get-best-sidechain-block-hash")]
    GetBestSidechainBlockHash,

    /// Get mainchain BMM inclusions
    #[command(name = "get-bmm-inclusions")]
    GetBmmInclusions {
        block_hash: truthcoin_dc::types::BlockHash,
    },

    /// Get transaction by txid
    #[command(name = "get-transaction", alias = "get-tx")]
    GetTransaction { txid: Txid },

    /// Get transaction info
    #[command(name = "get-transaction-info")]
    GetTransactionInfo { txid: Txid },

    /// Get pending withdrawal bundle
    #[command(name = "pending-withdrawal-bundle")]
    PendingWithdrawalBundle,

    /// Get latest failed withdrawal bundle height
    #[command(name = "latest-failed-withdrawal-bundle-height")]
    LatestFailedWithdrawalBundleHeight,

    /// Remove transaction from mempool
    #[command(name = "remove-from-mempool")]
    RemoveFromMempool { txid: Txid },

    // === NETWORK COMMANDS ===
    /// Connect to a peer
    #[command(name = "connect-peer", alias = "connect")]
    ConnectPeer { addr: SocketAddr },

    /// List connected peers
    #[command(name = "list-peers", alias = "peers")]
    ListPeers,

    // === CRYPTOGRAPHY COMMANDS ===
    /// Get new encryption key
    #[command(name = "get-new-encryption-key")]
    GetNewEncryptionKey,

    /// Get new verifying key
    #[command(name = "get-new-verifying-key")]
    GetNewVerifyingKey,

    /// Encrypt message
    #[command(name = "encrypt-msg", alias = "encrypt")]
    EncryptMsg {
        #[arg(long)]
        encryption_pubkey: EncryptionPubKey,
        #[arg(long)]
        msg: String,
    },

    /// Decrypt message
    #[command(name = "decrypt-msg", alias = "decrypt")]
    DecryptMsg {
        #[arg(long)]
        encryption_pubkey: EncryptionPubKey,
        #[arg(long)]
        msg: String,
        #[arg(long)]
        utf8: bool,
    },

    /// Sign arbitrary message with verifying key
    #[command(name = "sign-arbitrary-msg", alias = "sign")]
    SignArbitraryMsg {
        #[arg(long)]
        verifying_key: VerifyingKey,
        #[arg(long)]
        msg: String,
    },

    /// Sign arbitrary message as address
    #[command(name = "sign-arbitrary-msg-as-addr")]
    SignArbitraryMsgAsAddr {
        #[arg(long)]
        address: Address,
        #[arg(long)]
        msg: String,
    },

    /// Verify signature
    #[command(name = "verify-signature", alias = "verify")]
    VerifySignature {
        #[arg(long)]
        signature: Signature,
        #[arg(long)]
        verifying_key: VerifyingKey,
        #[arg(long)]
        dst: Dst,
        #[arg(long)]
        msg: String,
    },

    // === SLOT COMMANDS (slot_*) ===
    /// Get slot system status
    #[command(name = "slot-status")]
    SlotStatus,

    /// List slots with optional filtering
    #[command(name = "slot-list", alias = "slots")]
    SlotList {
        /// Filter by period
        #[arg(long)]
        period: Option<u32>,
        /// Filter by status: available, claimed, voting, ossified
        #[arg(long)]
        status: Option<String>,
    },

    /// Get slot by ID
    #[command(name = "slot-get")]
    SlotGet {
        /// Slot ID (hex)
        slot_id: String,
    },

    /// Claim a decision slot
    #[command(name = "slot-claim", alias = "claim")]
    SlotClaim {
        #[arg(long)]
        period_index: u32,
        #[arg(long)]
        slot_index: u32,
        #[arg(long, action = ArgAction::Set)]
        is_standard: bool,
        #[arg(long, action = ArgAction::Set)]
        is_scaled: bool,
        #[arg(long)]
        question: String,
        #[arg(long)]
        min: Option<u16>,
        #[arg(long)]
        max: Option<u16>,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    // === MARKET COMMANDS (market_*) ===
    /// Create prediction market
    #[command(name = "market-create", alias = "cm")]
    MarketCreate {
        #[arg(long)]
        title: String,
        #[arg(long)]
        description: String,
        /// Dimensions: "[slot1]", "[slot1,slot2]", "[[slot1,slot2,slot3]]"
        #[arg(long)]
        dimensions: String,
        #[arg(long, default_value = "7.0")]
        beta: f64,
        #[arg(long, default_value = "0.005")]
        trading_fee: f64,
        #[arg(long)]
        tags: Option<String>,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// List all markets
    #[command(name = "market-list", alias = "markets")]
    MarketList,

    /// Get market details
    #[command(name = "market-get")]
    MarketGet { market_id: String },

    /// Buy/sell shares (use negative amount to sell)
    #[command(name = "market-buy", alias = "buy")]
    MarketBuy {
        #[arg(long)]
        market_id: String,
        #[arg(long)]
        outcome_index: usize,
        #[arg(long)]
        shares_amount: f64,
        #[arg(long)]
        max_cost: u64,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Get share positions for an address
    #[command(name = "market-positions", alias = "positions")]
    MarketPositions {
        #[arg(long)]
        address: Option<Address>,
        #[arg(long)]
        market_id: Option<String>,
    },

    /// Calculate share cost (dry run)
    #[command(name = "calculate-share-cost", alias = "calc-cost")]
    CalculateShareCost {
        #[arg(long)]
        market_id: String,
        #[arg(long)]
        outcome_index: usize,
        #[arg(long)]
        shares_amount: f64,
    },

    /// Calculate initial liquidity required
    #[command(name = "calculate-initial-liquidity")]
    CalculateInitialLiquidity {
        #[arg(long)]
        beta: f64,
        #[arg(long, default_value = "independent")]
        market_type: String,
        #[arg(long)]
        num_outcomes: Option<usize>,
        #[arg(long)]
        decision_slots: Option<String>,
        #[arg(long)]
        has_residual: Option<bool>,
        #[arg(long)]
        dimensions: Option<String>,
    },

    // === VOTE COMMANDS (vote_*) ===
    /// Register as a voter
    #[command(name = "vote-register", alias = "register")]
    VoteRegister {
        #[arg(long)]
        reputation_bond_sats: Option<u64>,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Get voter information
    #[command(name = "vote-voter")]
    VoteVoter {
        /// Voter address
        address: Address,
    },

    /// List all registered voters
    #[command(name = "vote-voters", alias = "voters")]
    VoteVoters,

    /// Submit vote(s) - use comma-separated "id:value" for batch
    #[command(name = "vote-submit", alias = "vote")]
    VoteSubmit {
        /// Single vote: --decision-id X --vote-value Y, or batch: --votes "id1:val1,id2:val2"
        #[arg(long)]
        decision_id: Option<String>,
        #[arg(long)]
        vote_value: Option<f64>,
        #[arg(long)]
        votes: Option<String>,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Query votes with filters
    #[command(name = "vote-list")]
    VoteList {
        #[arg(long)]
        voter: Option<Address>,
        #[arg(long)]
        decision_id: Option<String>,
        #[arg(long)]
        period_id: Option<u32>,
    },

    /// Get voting period info (omit period_id for current)
    #[command(name = "vote-period")]
    VotePeriod {
        #[arg(long)]
        period_id: Option<u32>,
    },

    // === VOTECOIN COMMANDS (votecoin_*) ===
    /// Transfer votecoin
    #[command(name = "votecoin-transfer")]
    VotecoinTransfer {
        dest: Address,
        #[arg(long)]
        amount: u32,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Get votecoin balance
    #[command(name = "votecoin-balance")]
    VotecoinBalance {
        /// Address to check
        address: Address,
    },
}

const DEFAULT_RPC_HOST: Host = Host::Ipv4(Ipv4Addr::LOCALHOST);

const DEFAULT_RPC_PORT: u16 = 6000 + THIS_SIDECHAIN as u16;

const DEFAULT_TIMEOUT_SECS: u64 = 60;

#[derive(Clone, Debug, Parser)]
#[command(
    author,
    version,
    about = "Truthcoin Drivechain CLI - Bitcoin Hivemind prediction market client",
    long_about = "
Truthcoin DC CLI - Command-line interface for Bitcoin Hivemind prediction markets

COMMANDS (matching RPC API namespaces):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM:     status, stop, mine, openapi-schema
WALLET:     balance, transfer, withdraw, create-deposit, my-utxos
BLOCKCHAIN: get-block-count, get-block, get-transaction, list-peers

slot_*:     slot-status, slot-list, slot-get, slot-claim
market_*:   market-create, market-list, market-get, market-buy, market-positions
vote_*:     vote-register, vote-voter, vote-voters, vote-submit, vote-list, vote-period
votecoin_*: votecoin-transfer, votecoin-balance

QUICK START:
    truthcoin_dc_app_cli status                    # Check node status
    truthcoin_dc_app_cli balance                   # Check wallet balance
    truthcoin_dc_app_cli market-list               # List markets
    truthcoin_dc_app_cli slot-list --status voting # List slots in voting

For command details: truthcoin_dc_app_cli <command> --help
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
    /// Host used for requests to the RPC server
    #[arg(default_value_t = DEFAULT_RPC_HOST, long, value_parser = Host::parse)]
    pub rpc_host: Host,
    /// Port used for requests to the RPC server
    #[arg(default_value_t = DEFAULT_RPC_PORT, long)]
    pub rpc_port: u16,
    /// Timeout for RPC requests in seconds.
    #[arg(default_value_t = DEFAULT_TIMEOUT_SECS, long = "timeout")]
    timeout_secs: u64,
    #[arg(short, long, help = "Enable verbose HTTP output")]
    pub verbose: bool,
}

impl Cli {
    pub fn new(
        command: Command,
        rpc_host: Option<Host>,
        rpc_port: Option<u16>,
        timeout_secs: Option<u64>,
        verbose: Option<bool>,
    ) -> Self {
        Self {
            command,
            rpc_host: rpc_host.unwrap_or(DEFAULT_RPC_HOST),
            rpc_port: rpc_port.unwrap_or(DEFAULT_RPC_PORT),
            timeout_secs: timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS),
            verbose: verbose.unwrap_or(false),
        }
    }

    fn rpc_url(&self) -> url::Url {
        Url::parse(&format!("http://{}:{}", self.rpc_host, self.rpc_port))
            .unwrap()
    }
}

async fn handle_command<RpcClient>(
    rpc_client: &RpcClient,
    command: Command,
) -> anyhow::Result<String>
where
    RpcClient: ClientT + Sync,
{
    Ok(match command {
        // === SYSTEM COMMANDS ===
        Command::Status => {
            let blockcount = rpc_client.getblockcount().await?;
            let peers = rpc_client.list_peers().await?;
            let balance = rpc_client.bitcoin_balance().await?;
            format!(
                "Node Status: ✓ Online\n\
                Block Count: {}\n\
                Connected Peers: {}\n\
                Wallet Balance: {} sats",
                blockcount,
                peers.len(),
                balance.total
            )
        }
        Command::Stop => {
            let () = rpc_client.stop().await?;
            "Node stopping...".to_string()
        }
        Command::Mine { fee_sats } => {
            let () = rpc_client.mine(Some(fee_sats)).await?;
            format!("Mining block with fee {fee_sats} sats")
        }
        Command::OpenApiSchema => {
            let openapi =
                <truthcoin_dc_app_rpc_api::RpcDoc as utoipa::OpenApi>::openapi(
                );
            openapi.to_pretty_json()?
        }

        // === WALLET COMMANDS ===
        Command::Balance => {
            let balance = rpc_client.bitcoin_balance().await?;
            json_response(&balance)?
        }
        Command::GetNewAddress => {
            let address = rpc_client.get_new_address().await?;
            format!("{address}")
        }
        Command::GetWalletAddresses => {
            let addresses = rpc_client.get_wallet_addresses().await?;
            json_response(&addresses)?
        }
        Command::MyUtxos => {
            let utxos = rpc_client.my_utxos().await?;
            json_response(&utxos)?
        }
        Command::MyUnconfirmedUtxos => {
            let utxos = rpc_client.my_unconfirmed_utxos().await?;
            json_response(&utxos)?
        }
        Command::GetWalletUtxos => {
            let utxos = rpc_client.get_wallet_utxos().await?;
            json_response(&utxos)?
        }
        Command::ListUtxos => {
            let utxos = rpc_client.list_utxos().await?;
            json_response(&utxos)?
        }
        Command::Transfer {
            dest,
            value_sats,
            fee_sats,
        } => {
            let txid = rpc_client
                .transfer(dest, value_sats, fee_sats, None)
                .await?;
            format_tx_success("Transfer", None, &txid.to_string())
        }
        Command::Withdraw {
            mainchain_address,
            amount_sats,
            fee_sats,
            mainchain_fee_sats,
        } => {
            let txid = rpc_client
                .withdraw(
                    mainchain_address,
                    amount_sats,
                    fee_sats,
                    mainchain_fee_sats,
                )
                .await?;
            format_tx_success("Withdrawal initiated", None, &txid.to_string())
        }
        Command::CreateDeposit {
            address,
            value_sats,
            fee_sats,
        } => {
            let txid = rpc_client
                .create_deposit(address, value_sats, fee_sats)
                .await?;
            format_tx_success("Deposit created", None, &txid.to_string())
        }
        Command::FormatDepositAddress { address } => {
            rpc_client.format_deposit_address(address).await?
        }
        Command::GenerateMnemonic => rpc_client.generate_mnemonic().await?,
        Command::SetSeedFromMnemonic { mnemonic } => {
            let () = rpc_client.set_seed_from_mnemonic(mnemonic).await?;
            "Wallet seed imported successfully".to_string()
        }
        Command::SidechainWealth => {
            let wealth = rpc_client.sidechain_wealth_sats().await?;
            format!("{wealth} sats")
        }

        // === BLOCKCHAIN COMMANDS ===
        Command::GetBlockCount => {
            let blockcount = rpc_client.getblockcount().await?;
            format!("{blockcount}")
        }
        Command::GetBlock { block_hash } => {
            let block = rpc_client.get_block(block_hash).await?;
            json_response(&block)?
        }
        Command::GetBestMainchainBlockHash => {
            let block_hash = rpc_client.get_best_mainchain_block_hash().await?;
            json_response(&block_hash)?
        }
        Command::GetBestSidechainBlockHash => {
            let block_hash = rpc_client.get_best_sidechain_block_hash().await?;
            json_response(&block_hash)?
        }
        Command::GetBmmInclusions { block_hash } => {
            let bmm_inclusions =
                rpc_client.get_bmm_inclusions(block_hash).await?;
            json_response(&bmm_inclusions)?
        }
        Command::GetTransaction { txid } => {
            let tx = rpc_client.get_transaction(txid).await?;
            json_response(&tx)?
        }
        Command::GetTransactionInfo { txid } => {
            let tx_info = rpc_client.get_transaction_info(txid).await?;
            json_response(&tx_info)?
        }
        Command::PendingWithdrawalBundle => {
            let withdrawal_bundle =
                rpc_client.pending_withdrawal_bundle().await?;
            json_response(&withdrawal_bundle)?
        }
        Command::LatestFailedWithdrawalBundleHeight => {
            let height =
                rpc_client.latest_failed_withdrawal_bundle_height().await?;
            json_response(&height)?
        }
        Command::RemoveFromMempool { txid } => {
            let () = rpc_client.remove_from_mempool(txid).await?;
            format!("Transaction {txid} removed from mempool")
        }

        // === NETWORK COMMANDS ===
        Command::ConnectPeer { addr } => {
            let () = rpc_client.connect_peer(addr).await?;
            format!("Connected to peer: {addr}")
        }
        Command::ListPeers => {
            let peers = rpc_client.list_peers().await?;
            json_response(&peers)?
        }

        // === CRYPTOGRAPHY COMMANDS ===
        Command::GetNewEncryptionKey => {
            let epk = rpc_client.get_new_encryption_key().await?;
            format!("{epk}")
        }
        Command::GetNewVerifyingKey => {
            let vk = rpc_client.get_new_verifying_key().await?;
            format!("{vk}")
        }
        Command::EncryptMsg {
            encryption_pubkey,
            msg,
        } => rpc_client.encrypt_msg(encryption_pubkey, msg).await?,
        Command::DecryptMsg {
            encryption_pubkey,
            msg,
            utf8,
        } => {
            let msg_hex =
                rpc_client.decrypt_msg(encryption_pubkey, msg).await?;
            if utf8 {
                let msg_bytes: Vec<u8> = hex::decode(msg_hex)?;
                String::from_utf8(msg_bytes)?
            } else {
                msg_hex
            }
        }
        Command::SignArbitraryMsg { verifying_key, msg } => {
            let signature =
                rpc_client.sign_arbitrary_msg(verifying_key, msg).await?;
            format!("{signature}")
        }
        Command::SignArbitraryMsgAsAddr { address, msg } => {
            let authorization =
                rpc_client.sign_arbitrary_msg_as_addr(address, msg).await?;
            serde_json::to_string_pretty(&authorization)?
        }
        Command::VerifySignature {
            signature,
            verifying_key,
            dst,
            msg,
        } => {
            let res = rpc_client
                .verify_signature(signature, verifying_key, dst, msg)
                .await?;
            format!("{res}")
        }

        // === SLOT COMMANDS (slot_*) ===
        Command::SlotStatus => {
            let status = rpc_client.slot_status().await?;
            json_response(&status)?
        }
        Command::SlotList { period, status } => {
            use truthcoin_dc_app_rpc_api::{SlotFilter, SlotState};
            let slot_status = status.map(|s| match s.to_lowercase().as_str() {
                "available" => SlotState::Available,
                "claimed" => SlotState::Claimed,
                "voting" => SlotState::Voting,
                "ossified" => SlotState::Ossified,
                _ => SlotState::Available,
            });
            let filter = if period.is_some() || slot_status.is_some() {
                Some(SlotFilter {
                    period,
                    status: slot_status,
                })
            } else {
                None
            };
            let slots = rpc_client.slot_list(filter).await?;
            json_response(&slots)?
        }
        Command::SlotGet { slot_id } => {
            let slot = rpc_client.slot_get(slot_id).await?;
            json_response(&slot)?
        }
        Command::SlotClaim {
            period_index,
            slot_index,
            is_standard,
            is_scaled,
            question,
            min,
            max,
            fee_sats,
        } => {
            let txid = rpc_client
                .slot_claim(
                    period_index,
                    slot_index,
                    is_standard,
                    is_scaled,
                    question,
                    min,
                    max,
                    fee_sats,
                )
                .await?;
            format!("Slot claimed: {txid}")
        }

        // === MARKET COMMANDS (market_*) ===
        Command::MarketCreate {
            title,
            description,
            dimensions,
            beta,
            trading_fee,
            tags,
            fee_sats,
        } => {
            use truthcoin_dc_app_rpc_api::CreateMarketRequest;
            let slots = match extract_slots_from_dimensions(&dimensions) {
                Ok(slots) => slots,
                Err(err) => return Ok(err),
            };
            let parsed_tags = tags.map(|t| parse_comma_separated(&t));
            let request = CreateMarketRequest {
                title: title.clone(),
                description,
                market_type: "dimensional".to_string(),
                decision_slots: slots,
                dimensions: Some(dimensions),
                has_residual: None,
                beta: Some(beta),
                trading_fee: Some(trading_fee),
                tags: parsed_tags,
                initial_liquidity: None,
                fee_sats,
            };
            let txid = rpc_client.market_create(request).await?;
            format!("Market '{title}' created: {txid}")
        }
        Command::MarketList => {
            let markets = rpc_client.market_list().await?;
            if markets.is_empty() {
                "No markets in Trading state found.".to_string()
            } else {
                let mut output = String::new();
                output.push_str("Markets in Trading State:\n");
                output.push_str("┌──────────────────┬──────────────────────────┬─────────────────────────┬──────────────┬────────────┐\n");
                output.push_str("│ Market ID        │ Title                    │ Outcomes                │ Volume       │ State      │\n");
                output.push_str("├──────────────────┼──────────────────────────┼─────────────────────────┼──────────────┼────────────┤\n");

                for market in &markets {
                    let short_id = market.market_id.clone();

                    // Truncate title to fit column
                    let short_title = if market.title.len() > 22 {
                        format!("{}...", &market.title[..19])
                    } else {
                        market.title.clone()
                    };

                    // Format outcome count for display
                    let outcomes_display =
                        format!("{} outcomes", market.outcome_count);
                    let short_outcomes = if outcomes_display.len() > 21 {
                        format!("{}...", &outcomes_display[..18])
                    } else {
                        outcomes_display
                    };

                    // Show volume
                    let volume_display = format!("{:.1}", market.volume);
                    let short_volume = if volume_display.len() > 12 {
                        format!("{}...", &volume_display[..9])
                    } else {
                        volume_display
                    };

                    output.push_str(&format!(
                        "│ {:16} │ {:24} │ {:23} │ {:12} │ {:10} │\n",
                        short_id,
                        short_title,
                        short_outcomes,
                        short_volume,
                        market.state
                    ));
                }

                output.push_str("└──────────────────┴──────────────────────────┴─────────────────────────┴──────────────┴────────────┘\n");
                output.push_str(&format!("\nTotal markets: {}", markets.len()));
                output
            }
        }
        Command::MarketGet { market_id } => {
            let market = rpc_client.market_get(market_id).await?;
            json_response(&market)?
        }
        Command::MarketBuy {
            market_id,
            outcome_index,
            shares_amount,
            max_cost,
            fee_sats,
        } => {
            use truthcoin_dc_app_rpc_api::MarketBuyRequest;
            let request = MarketBuyRequest {
                market_id: market_id.clone(),
                outcome_index,
                shares_amount,
                max_cost: Some(max_cost),
                fee_sats: Some(fee_sats),
                dry_run: Some(false),
            };
            let result = rpc_client.market_buy(request).await?;
            format!(
                "Successfully submitted buy shares transaction!\n\
                Market: {}\n\
                Outcome Index: {}\n\
                Shares: {:.4}\n\
                Cost: {} sats\n\
                Transaction ID: {}",
                market_id,
                outcome_index,
                shares_amount,
                result.cost_sats,
                result.txid.unwrap_or_default()
            )
        }
        Command::CalculateShareCost {
            market_id,
            outcome_index,
            shares_amount,
        } => {
            use truthcoin_dc_app_rpc_api::MarketBuyRequest;
            let request = MarketBuyRequest {
                market_id: market_id.clone(),
                outcome_index,
                shares_amount,
                max_cost: None,
                fee_sats: Some(0),
                dry_run: Some(true),
            };
            let result = rpc_client.market_buy(request).await?;

            format!(
                "Share Purchase Cost Calculation:\n\
                Market: {}\n\
                Outcome Index: {}\n\
                Shares Amount: {:.4}\n\
                Estimated Cost: {} sats ({:.8} BTC)",
                market_id,
                outcome_index,
                shares_amount,
                result.cost_sats,
                result.cost_sats as f64 / 100_000_000.0
            )
        }
        Command::MarketPositions { address, market_id } => {
            let addr = match address {
                Some(a) => a,
                None => rpc_client.get_new_address().await?,
            };
            let holdings = rpc_client.market_positions(addr, market_id).await?;
            json_response(&holdings)?
        }

        Command::CalculateInitialLiquidity {
            beta,
            market_type,
            num_outcomes,
            decision_slots,
            has_residual,
            dimensions,
        } => {
            use truthcoin_dc_app_rpc_api::CalculateInitialLiquidityRequest;

            // Parse decision slots if provided
            let parsed_slots =
                decision_slots.map(|s| parse_comma_separated(&s));

            let request = CalculateInitialLiquidityRequest {
                beta,
                market_type,
                decision_slots: parsed_slots,
                num_outcomes,
                dimensions,
                has_residual,
            };

            let result =
                rpc_client.calculate_initial_liquidity(request).await?;

            format!(
                "Initial Liquidity Calculation:\n\
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\
                Beta Parameter:       {:.2}\n\
                Number of Outcomes:   {}\n\
                Market Configuration: {}\n\
                \n\
                Calculation Details:\n\
                {}\n\
                \n\
                Required Initial Liquidity: {} sats\n\
                Minimum Treasury:          {} sats\n\
                \n\
                Formula Used: Initial Liquidity = β × ln(Number of States)\n\
                             = {:.2} × ln({}) = {:.6} ≈ {} sats\n\
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                result.beta,
                result.num_outcomes,
                result.market_config,
                result.outcome_breakdown,
                result.initial_liquidity_sats,
                result.min_treasury_sats,
                result.beta,
                result.num_outcomes,
                result.beta * (result.num_outcomes as f64).ln(),
                result.initial_liquidity_sats
            )
        }

        // === VOTE COMMANDS (vote_*) ===
        Command::VoteRegister {
            reputation_bond_sats,
            fee_sats,
        } => {
            use truthcoin_dc_app_rpc_api::RegisterVoterRequest;
            let request = RegisterVoterRequest {
                reputation_bond_sats,
                fee_sats,
            };
            let txid = rpc_client.vote_register(request).await?;
            format!("Voter registered: {txid}")
        }
        Command::VoteVoter { address } => {
            let voter = rpc_client.vote_voter(address).await?;
            json_response(&voter)?
        }
        Command::VoteVoters => {
            let voters = rpc_client.vote_voters().await?;
            json_response(&voters)?
        }
        Command::VoteSubmit {
            decision_id,
            vote_value,
            votes,
            fee_sats,
        } => {
            use truthcoin_dc_app_rpc_api::VoteBatchItem;
            let vote_items = if let Some(batch) = votes {
                // Parse batch format: "id1:val1,id2:val2"
                batch
                    .split(',')
                    .filter_map(|pair| {
                        let parts: Vec<&str> = pair.trim().split(':').collect();
                        if parts.len() == 2 {
                            let val: f64 = parts[1].parse().ok()?;
                            Some(VoteBatchItem {
                                decision_id: parts[0].to_string(),
                                vote_value: val,
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            } else if let (Some(id), Some(val)) = (decision_id, vote_value) {
                vec![VoteBatchItem {
                    decision_id: id,
                    vote_value: val,
                }]
            } else {
                return Ok(
                    "Error: provide --decision-id and --vote-value, or --votes"
                        .to_string(),
                );
            };
            let txid = rpc_client.vote_submit(vote_items, fee_sats).await?;
            format!("Vote submitted: {txid}")
        }
        Command::VoteList {
            voter,
            decision_id,
            period_id,
        } => {
            use truthcoin_dc_app_rpc_api::VoteFilter;
            let filter = VoteFilter {
                voter,
                decision_id,
                period_id,
            };
            let votes = rpc_client.vote_list(filter).await?;
            json_response(&votes)?
        }
        Command::VotePeriod { period_id } => {
            let period = rpc_client.vote_period(period_id).await?;
            json_response(&period)?
        }

        // === VOTECOIN COMMANDS (votecoin_*) ===
        Command::VotecoinTransfer {
            dest,
            amount,
            fee_sats,
        } => {
            let txid = rpc_client
                .votecoin_transfer(dest, amount, fee_sats, None)
                .await?;
            format!("Votecoin transferred: {txid}")
        }
        Command::VotecoinBalance { address } => {
            let balance = rpc_client.votecoin_balance(address).await?;
            format!("{balance}")
        }
    })
}

fn set_tracing_subscriber() -> anyhow::Result<()> {
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout()))
        .with_file(true)
        .with_line_number(true);

    let subscriber = tracing_subscriber::registry().with(stdout_layer);
    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}

impl Cli {
    pub async fn run(self) -> anyhow::Result<String> {
        if self.verbose {
            set_tracing_subscriber()?;
        }

        let request_id = uuid::Uuid::new_v4().as_simple().to_string();
        tracing::info!(%request_id);
        let builder = HttpClientBuilder::default()
            .request_timeout(Duration::from_secs(self.timeout_secs))
            .set_max_logging_length(1024)
            .set_headers(HeaderMap::from_iter([(
                http::header::HeaderName::from_static("x-request-id"),
                http::header::HeaderValue::from_str(&request_id)?,
            )]));
        let client = builder.build(self.rpc_url())?;

        let result = handle_command(&client, self.command).await?;
        Ok(result)
    }
}
