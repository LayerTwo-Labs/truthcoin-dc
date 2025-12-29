use std::collections::HashMap;

use bip300301_enforcer_integration_tests::{
    integration_test::{
        activate_sidechain, deposit, fund_enforcer, propose_sidechain,
    },
    setup::{
        Mode, Network, PostSetup as EnforcerPostSetup, Sidechain as _,
        setup as setup_enforcer,
    },
    util::{AbortOnDrop, AsyncTrial},
};
use futures::{
    FutureExt as _, StreamExt as _, channel::mpsc, future::BoxFuture,
};
use tokio::time::sleep;
use tracing::Instrument as _;
use truthcoin_dc::{
    authorization::{Dst, Signature},
    types::{Address, FilledOutputContent, GetAddress as _},
};
use truthcoin_dc_app_rpc_api::{
    MarketBuyRequest, RpcClient as _, SlotContentInfo, SlotFilter, SlotState,
    VoteBatchItem, VoteFilter,
};

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

/// Pre-calculated expected values for whitepaper Figure 5 vote matrix
mod expected {
    /// Expected consensus outcomes for the vote matrix:
    /// D1: All vote 1.0 → 1.0
    /// D2: 6 vote 0.5, 1 votes 1.0 → ~0.57, rounds to 0.5 with catch_tl
    /// D3: All vote 0.0 → 0.0
    /// D4: All vote 0.0 → 0.0
    pub const OUTCOMES: [f64; 4] = [1.0, 0.5, 0.0, 0.0];

    /// Voter 2 is the dissenter (0-indexed) - voted 1.0 on D2 when consensus is 0.5
    pub const DISSENTER_INDEX: usize = 2;

    /// LMSR invariant tolerance - prices must sum to 1.0 within this tolerance
    pub const PRICE_SUM_TOLERANCE: f64 = 1e-6;

    /// Reputation change tolerance for comparisons
    pub const REPUTATION_TOLERANCE: f64 = 0.0001;

    /// Outcome value tolerance for consensus verification
    pub const OUTCOME_TOLERANCE: f64 = 0.01;
}

/// Pre-calculated LMSR trade costs for deterministic testing.
/// All values calculated with: β = 1,000,000, fee_rate = 0.5%
mod expected_costs {
    /// Initial treasury for binary market: ceil(β * ln(2)) = 693148 sats (~0.007 BTC)
    pub const INITIAL_TREASURY: u64 = 693148;
}

/// Helper functions for verifying market UTXO state transitions
mod utxo_verification {
    use std::collections::HashMap;
    use truthcoin_dc::types::{FilledOutputContent, OutPoint, PointedOutput};

    /// Extract market treasury UTXOs from a list of all UTXOs
    /// Returns a map of market_id (hex) -> (outpoint, amount_sats)
    pub fn get_market_treasury_utxos(
        utxos: &[PointedOutput<FilledOutputContent>],
    ) -> HashMap<String, (OutPoint, u64)> {
        utxos
            .iter()
            .filter_map(|pointed| {
                if let FilledOutputContent::MarketTreasury {
                    market_id,
                    amount,
                } = &pointed.output.content
                {
                    Some((
                        hex::encode(market_id),
                        (pointed.outpoint, amount.0.to_sat()),
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Extract market author fee UTXOs from a list of all UTXOs
    /// Returns a map of market_id (hex) -> (outpoint, amount_sats)
    pub fn get_market_author_fee_utxos(
        utxos: &[PointedOutput<FilledOutputContent>],
    ) -> HashMap<String, (OutPoint, u64)> {
        utxos
            .iter()
            .filter_map(|pointed| {
                if let FilledOutputContent::MarketAuthorFee {
                    market_id,
                    amount,
                } = &pointed.output.content
                {
                    Some((
                        hex::encode(market_id),
                        (pointed.outpoint, amount.0.to_sat()),
                    ))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Debug)]
struct TruthcoinNodes {
    issuer: PostSetup,
    voter_0: PostSetup,
    voter_1: PostSetup,
    voter_2: PostSetup,
    voter_3: PostSetup,
    voter_4: PostSetup,
    voter_5: PostSetup,
    voter_6: PostSetup,
}

impl TruthcoinNodes {
    async fn setup(
        bin_paths: &BinPaths,
        res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
        enforcer_post_setup: &EnforcerPostSetup,
    ) -> anyhow::Result<Self> {
        let setup_single = |suffix: &str| {
            PostSetup::setup(
                Init {
                    truthcoin_app: bin_paths.truthcoin.clone(),
                    data_dir_suffix: Some(suffix.to_owned()),
                },
                enforcer_post_setup,
                res_tx.clone(),
            )
        };
        let res = Self {
            issuer: setup_single("issuer").await?,
            voter_0: setup_single("voter_0").await?,
            voter_1: setup_single("voter_1").await?,
            voter_2: setup_single("voter_2").await?,
            voter_3: setup_single("voter_3").await?,
            voter_4: setup_single("voter_4").await?,
            voter_5: setup_single("voter_5").await?,
            voter_6: setup_single("voter_6").await?,
        };
        for voter in [
            &res.voter_0,
            &res.voter_1,
            &res.voter_2,
            &res.voter_3,
            &res.voter_4,
            &res.voter_5,
            &res.voter_6,
        ] {
            res.issuer
                .rpc_client
                .connect_peer(voter.net_addr().into())
                .await?;
        }
        tracing::debug!("Connected 8 nodes in P2P network");
        Ok(res)
    }
}

const DEPOSIT_AMOUNT: bitcoin::Amount = bitcoin::Amount::from_sat(21000000);
const DEPOSIT_FEE: bitcoin::Amount = bitcoin::Amount::from_sat(1000000);

async fn setup(
    bin_paths: &BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<(EnforcerPostSetup, TruthcoinNodes)> {
    let mut enforcer_post_setup = setup_enforcer(
        &bin_paths.others,
        Network::Regtest,
        Mode::Mempool,
        res_tx.clone(),
    )
    .await?;
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Proposed sidechain successfully");
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Activated sidechain successfully");
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    let mut truthcoin_nodes =
        TruthcoinNodes::setup(bin_paths, res_tx, &enforcer_post_setup).await?;
    let issuer_deposit_address =
        truthcoin_nodes.issuer.get_deposit_address().await?;
    let () = deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.issuer,
        &issuer_deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to sidechain successfully");
    Ok((enforcer_post_setup, truthcoin_nodes))
}

const VOTE_CALL_MSG: &str = "test vote call";
const VOTE_YES_MSG: &str = "test vote call YES";
const VOTE_NO_MSG: &str = "test vote call NO";
const INITIAL_VOTECOIN_SUPPLY: u32 = 1000000;
const VOTER_ALLOCATION: u32 = 142857;

async fn roundtrip_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let (mut enforcer_post_setup, mut truthcoin_nodes) =
        setup(&bin_paths, res_tx.clone()).await?;

    let issuer_vk = truthcoin_nodes
        .issuer
        .rpc_client
        .get_new_verifying_key()
        .await?;

    let _issuer_addr =
        truthcoin_nodes.issuer.rpc_client.get_new_address().await?;

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    let utxos = truthcoin_nodes.issuer.rpc_client.get_wallet_utxos().await?;
    let total_votecoin: u32 = utxos
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    anyhow::ensure!(
        total_votecoin == INITIAL_VOTECOIN_SUPPLY,
        "Expected initial Votecoin supply of {}, found {}",
        INITIAL_VOTECOIN_SUPPLY,
        total_votecoin
    );
    let voters = [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
    ];

    let mut voter_addrs = Vec::new();
    for voter in &voters {
        voter_addrs.push(voter.rpc_client.get_new_address().await?);
    }
    let [
        voter_addr_0,
        voter_addr_1,
        voter_addr_2,
        voter_addr_3,
        voter_addr_4,
        voter_addr_5,
        voter_addr_6,
    ]: [Address; 7] = voter_addrs.try_into().unwrap();

    let voter_addresses = [
        voter_addr_0,
        voter_addr_1,
        voter_addr_2,
        voter_addr_3,
        voter_addr_4,
        voter_addr_5,
        voter_addr_6,
    ];

    for &voter_addr in &voter_addresses {
        truthcoin_nodes
            .issuer
            .rpc_client
            .votecoin_transfer(voter_addr, VOTER_ALLOCATION, 0, None)
            .await?;

        truthcoin_nodes
            .issuer
            .bmm_single(&mut enforcer_post_setup)
            .await?;

        sleep(std::time::Duration::from_secs(2)).await;
    }

    let vote_call_msg_sig: Signature = truthcoin_nodes
        .issuer
        .rpc_client
        .sign_arbitrary_msg(issuer_vk, VOTE_CALL_MSG.to_owned())
        .await?;

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
    ] {
        anyhow::ensure!(
            voter
                .rpc_client
                .verify_signature(
                    vote_call_msg_sig,
                    issuer_vk,
                    Dst::Arbitrary,
                    VOTE_CALL_MSG.to_owned()
                )
                .await?
        )
    }

    for (&voter_addr, voter) in voter_addresses.iter().zip(&voters) {
        let balance = voter.rpc_client.votecoin_balance(voter_addr).await?;
        anyhow::ensure!(
            balance == VOTER_ALLOCATION,
            "Voter at address {} has VoteCoin balance {} instead of expected {}",
            voter_addr.as_base58(),
            balance,
            VOTER_ALLOCATION
        );
    }

    let vote_weights: HashMap<Address, u32> = {
        let mut weights = HashMap::new();
        let utxos = truthcoin_nodes.issuer.rpc_client.list_utxos().await?;
        for utxo in utxos {
            if let Some(votecoin_amount) = utxo.output.content.votecoin() {
                *weights.entry(utxo.output.address).or_default() +=
                    votecoin_amount;
            }
        }
        weights
    };
    anyhow::ensure!(
        vote_weights.len() >= 7,
        "Expected at least 7 voters in snapshot, found {}",
        vote_weights.len()
    );

    let vote_auth_0 = truthcoin_nodes
        .voter_0
        .rpc_client
        .sign_arbitrary_msg_as_addr(voter_addr_0, VOTE_YES_MSG.to_owned())
        .await?;
    let vote_auth_1 = truthcoin_nodes
        .voter_1
        .rpc_client
        .sign_arbitrary_msg_as_addr(voter_addr_1, VOTE_NO_MSG.to_owned())
        .await?;
    let (total_yes, total_no) = {
        let (mut total_yes, mut total_no) = (0, 0);
        let mut vote_weights = vote_weights;
        for vote_auth in [vote_auth_0, vote_auth_1] {
            let voter_addr = vote_auth.get_address();
            if truthcoin_nodes
                .issuer
                .rpc_client
                .verify_signature(
                    vote_auth.signature,
                    vote_auth.verifying_key,
                    Dst::Arbitrary,
                    VOTE_YES_MSG.to_owned(),
                )
                .await?
            {
                if let Some(weight) = vote_weights.remove(&voter_addr) {
                    total_yes += weight;
                }
            } else if truthcoin_nodes
                .issuer
                .rpc_client
                .verify_signature(
                    vote_auth.signature,
                    vote_auth.verifying_key,
                    Dst::Arbitrary,
                    VOTE_NO_MSG.to_owned(),
                )
                .await?
                && let Some(weight) = vote_weights.remove(&voter_addr)
            {
                total_no += weight;
            }
        }
        (total_yes, total_no)
    };
    anyhow::ensure!(total_yes == VOTER_ALLOCATION);
    anyhow::ensure!(total_no == VOTER_ALLOCATION);

    let issuer_height =
        truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    for voter in voters.iter() {
        let height = voter.rpc_client.getblockcount().await?;
        anyhow::ensure!(issuer_height == height);
    }

    tracing::info!("✓ Phase 1: Votecoin distribution and voting verified");

    const VOTER_DEPOSIT_AMOUNT: bitcoin::Amount =
        bitcoin::Amount::from_sat(5_000_000);
    const VOTER_DEPOSIT_FEE: bitcoin::Amount =
        bitcoin::Amount::from_sat(500_000);

    let voter_0_deposit_address =
        truthcoin_nodes.voter_0.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_0,
        &voter_0_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    )
    .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    let voter_1_deposit_address =
        truthcoin_nodes.voter_1.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_1,
        &voter_1_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    )
    .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    let voter_2_deposit_address =
        truthcoin_nodes.voter_2.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_2,
        &voter_2_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    )
    .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    let voter_3_deposit_address =
        truthcoin_nodes.voter_3.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_3,
        &voter_3_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    )
    .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        let balance = voter.rpc_client.bitcoin_balance().await?;
        anyhow::ensure!(balance.total > bitcoin::Amount::ZERO);
    }

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    let slot_claims = [
        (voter_addr_0, 0, "Will Bitcoin reach $100k in 2025?"),
        (
            voter_addr_1,
            1,
            "Will the temperature in Florida be below 60 degrees?",
        ),
        (voter_addr_2, 2, "Will there be 1M BTC addresses by 2026?"),
        (voter_addr_3, 3, "Will BIP 444 activate"),
    ];

    for (i, (_voter_addr, slot_index, question)) in
        slot_claims.iter().enumerate()
    {
        let voter_node = match i {
            0 => &truthcoin_nodes.voter_0,
            1 => &truthcoin_nodes.voter_1,
            2 => &truthcoin_nodes.voter_2,
            3 => &truthcoin_nodes.voter_3,
            _ => unreachable!(),
        };

        voter_node
            .rpc_client
            .slot_claim(
                3,
                *slot_index,
                true,
                false,
                question.to_string(),
                None,
                None,
                1000,
            )
            .await?;
    }

    sleep(std::time::Duration::from_millis(500)).await;

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    sleep(std::time::Duration::from_secs(2)).await;

    let issuer_height =
        truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ]
    .iter()
    {
        let mut block_received = false;
        for _ in 0..20 {
            let voter_height = voter.rpc_client.getblockcount().await?;
            if voter_height >= issuer_height {
                block_received = true;
                break;
            }
            sleep(std::time::Duration::from_millis(500)).await;
        }
        anyhow::ensure!(block_received);
    }

    // Manual wallet refresh for multi-node test environment (8 nodes on one machine)
    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        voter.rpc_client.refresh_wallet().await?;
    }

    let wallet_utxos = truthcoin_nodes
        .voter_0
        .rpc_client
        .get_wallet_utxos()
        .await?;
    let chain_utxos = truthcoin_nodes.voter_0.rpc_client.list_utxos().await?;
    let voter_0_addresses: Vec<_> =
        wallet_utxos.iter().map(|u| u.output.address).collect();
    let voter_0_chain_bitcoin_utxos: Vec<_> = chain_utxos
        .iter()
        .filter(|utxo| {
            matches!(utxo.output.content, FilledOutputContent::Bitcoin(_))
                && voter_0_addresses.contains(&utxo.output.address)
        })
        .collect();

    anyhow::ensure!(!voter_0_chain_bitcoin_utxos.is_empty());

    let claimed_slots = truthcoin_nodes
        .issuer
        .rpc_client
        .slot_list(Some(SlotFilter {
            period: Some(3),
            status: Some(SlotState::Claimed),
        }))
        .await?;

    anyhow::ensure!(claimed_slots.len() == 4);

    tracing::info!("✓ Phase 2: Claimed 4 decision slots");
    for slot in claimed_slots.iter() {
        let question = slot
            .decision
            .as_ref()
            .map(|d| d.question.as_str())
            .unwrap_or("Unknown");
        tracing::info!("  - {}", question);
    }

    let market_slot_ids: Vec<String> = claimed_slots
        .iter()
        .map(|slot| slot.slot_id_hex.clone())
        .collect();
    let market_configs = [
        (
            &truthcoin_nodes.voter_0,
            0,
            "Will Bitcoin reach $100k in 2025?",
            "Binary market tracking BTC price prediction",
        ),
        (
            &truthcoin_nodes.voter_1,
            1,
            "Will the temperature in Florida be below 60 degrees?",
            "Weather prediction market for Florida temperature",
        ),
        (
            &truthcoin_nodes.voter_2,
            2,
            "Will there be 1M BTC addresses by 2026?",
            "Tracking Bitcoin adoption milestone",
        ),
        (
            &truthcoin_nodes.voter_3,
            3,
            "Will BIP 444 activate",
            "Prediction market for BIP 444 activation",
        ),
    ];

    for (voter_node, slot_idx, title, description) in market_configs.iter() {
        use truthcoin_dc_app_rpc_api::CreateMarketRequest;

        let slot_id = &market_slot_ids[*slot_idx];
        let request = CreateMarketRequest {
            title: title.to_string(),
            description: description.to_string(),
            market_type: "independent".to_string(),
            decision_slots: vec![slot_id.clone()],
            dimensions: None,
            has_residual: None,
            // beta = 1000000 gives ~693147 sats minimum treasury (~0.007 BTC)
            // This is a realistic liquidity level for a small prediction market
            beta: Some(1000000.0),
            trading_fee: Some(0.005),
            tags: Some(vec!["integration-test".to_string()]),
            initial_liquidity: None,
            fee_sats: 1000,
        };

        voter_node.rpc_client.market_create(request).await?;
    }

    sleep(std::time::Duration::from_millis(500)).await;

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    sleep(std::time::Duration::from_secs(2)).await;

    let markets = truthcoin_nodes.issuer.rpc_client.market_list().await?;
    anyhow::ensure!(markets.len() == 4);

    for market in &markets {
        anyhow::ensure!(market.state == "Trading");
        anyhow::ensure!(market.outcome_count == 2 || market.outcome_count == 3);
    }

    tracing::info!("✓ Phase 3: Created 4 binary markets");
    for market in markets.iter() {
        tracing::info!("  - {}", market.title);
    }

    let market_ids: Vec<String> =
        markets.iter().map(|m| m.market_id.clone()).collect();

    tracing::info!("\n--- LMSR Verification: Initial Market State ---");
    for market_id in &market_ids {
        let market_data = truthcoin_nodes
            .issuer
            .rpc_client
            .market_get(market_id.clone())
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!("Market not found: {}", market_id)
            })?;

        let price_sum: f64 =
            market_data.outcomes.iter().map(|o| o.current_price).sum();
        anyhow::ensure!(
            (price_sum - 1.0).abs() < expected::PRICE_SUM_TOLERANCE,
            "LMSR invariant violated: prices sum to {} (expected 1.0) for market {}",
            price_sum,
            market_id
        );

        // For binary markets, initial prices should be 0.5/0.5
        if market_data.outcomes.len() == 2 {
            for outcome in &market_data.outcomes {
                anyhow::ensure!(
                    (outcome.current_price - 0.5).abs() < 0.01,
                    "Initial binary market price should be 0.5, got {} for outcome {}",
                    outcome.current_price,
                    outcome.name
                );
            }
        }

        tracing::info!(
            "  Market {}: prices sum = {:.6}, beta = {}",
            &market_id[..12],
            price_sum,
            market_data.beta
        );
    }
    tracing::info!("✓ LMSR initial state verified\n");

    // === Market UTXO Verification: Post-Creation ===
    // Each market should have exactly one MarketTreasury UTXO with initial liquidity
    tracing::info!("--- Market UTXO Verification: Post-Creation ---");
    let all_utxos = truthcoin_nodes.issuer.rpc_client.list_utxos().await?;
    let initial_market_utxos =
        utxo_verification::get_market_treasury_utxos(&all_utxos);

    anyhow::ensure!(
        initial_market_utxos.len() == 4,
        "Expected 4 market treasury UTXOs after creation, found {}",
        initial_market_utxos.len()
    );

    for market_id in &market_ids {
        let market_id_short = &market_id[..12];
        anyhow::ensure!(
            initial_market_utxos.contains_key(market_id),
            "Market {} should have a treasury UTXO after creation",
            market_id_short
        );
        let (_outpoint, amount) = &initial_market_utxos[market_id];
        tracing::info!(
            "  Market {}: treasury UTXO with {} sats",
            market_id_short,
            amount
        );
    }
    tracing::info!("✓ Market UTXO post-creation verification complete\n");

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_secs(1)).await;

    for (i, market_id) in market_ids.iter().enumerate() {
        let voter = match i {
            0 => &truthcoin_nodes.voter_0,
            1 => &truthcoin_nodes.voter_1,
            2 => &truthcoin_nodes.voter_2,
            3 => &truthcoin_nodes.voter_3,
            _ => unreachable!(),
        };

        // Expected cost: ~25440 sats (25313 base + 127 fee)
        voter
            .rpc_client
            .market_buy(MarketBuyRequest {
                market_id: market_id.clone(),
                outcome_index: 0,
                shares_amount: 50000.0,
                max_cost: Some(30000), // generous slippage
                fee_sats: Some(1000),
                dry_run: None,
            })
            .await?;
    }
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    truthcoin_nodes.voter_1.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_millis(500)).await;

    // Market 0, trade 2: buy 50000 @ outcome 1 (~24812 sats)
    truthcoin_nodes
        .voter_1
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_ids[0].clone(),
            outcome_index: 1,
            shares_amount: 50000.0,
            max_cost: Some(30000),
            fee_sats: Some(1000),
            dry_run: None,
        })
        .await?;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    truthcoin_nodes.voter_2.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_millis(500)).await;

    // Market 1, trade 2: buy 500000 @ outcome 0 (~288469 sats)
    truthcoin_nodes
        .voter_2
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_ids[1].clone(),
            outcome_index: 0,
            shares_amount: 500000.0,
            max_cost: Some(300000),
            fee_sats: Some(1000),
            dry_run: None,
        })
        .await?;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    let voter_2_deposit_address =
        truthcoin_nodes.voter_2.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_2,
        &voter_2_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    )
    .await?;
    sleep(std::time::Duration::from_millis(500)).await;

    let voter_3_deposit_address =
        truthcoin_nodes.voter_3.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_3,
        &voter_3_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    )
    .await?;
    sleep(std::time::Duration::from_millis(500)).await;

    let voter_4_deposit_address =
        truthcoin_nodes.voter_4.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_4,
        &voter_4_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    )
    .await?;
    sleep(std::time::Duration::from_millis(500)).await;

    let voter_5_deposit_address =
        truthcoin_nodes.voter_5.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_5,
        &voter_5_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    )
    .await?;
    sleep(std::time::Duration::from_millis(500)).await;

    let voter_6_deposit_address =
        truthcoin_nodes.voter_6.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_6,
        &voter_6_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    )
    .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    truthcoin_nodes.voter_3.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_millis(500)).await;

    // Market 2 batched trades: buy 20000 @ outcome 0 (~10352), then 20000 @ outcome 1 (~9750)
    for outcome in [0, 1] {
        truthcoin_nodes
            .voter_3
            .rpc_client
            .market_buy(MarketBuyRequest {
                market_id: market_ids[2].clone(),
                outcome_index: outcome,
                shares_amount: 20000.0,
                max_cost: Some(15000),
                fee_sats: Some(1000),
                dry_run: None,
            })
            .await?;
    }
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    let voter_0_deposit_address =
        truthcoin_nodes.voter_0.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_0,
        &voter_0_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    )
    .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    truthcoin_nodes.voter_3.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_millis(500)).await;

    // Market 1, trade 3: buy 200000 @ outcome 0 (~132036 sats)
    truthcoin_nodes
        .voter_3
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_ids[1].clone(),
            outcome_index: 0,
            shares_amount: 200000.0,
            max_cost: Some(150000),
            fee_sats: Some(1000),
            dry_run: None,
        })
        .await?;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    truthcoin_nodes.voter_0.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_millis(500)).await;

    // Market 1, trade 4: buy 150000 @ outcome 1 (~50871 sats)
    truthcoin_nodes
        .voter_0
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_ids[1].clone(),
            outcome_index: 1,
            shares_amount: 150000.0,
            max_cost: Some(60000),
            fee_sats: Some(1000),
            dry_run: None,
        })
        .await?;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    // Market 3 batched trades:
    // Trade 2: buy 100000 @ outcome 0 (~52761 sats) - voter_0
    // Trade 3: buy 100000 @ outcome 1 (~47741 sats) - voter_1
    // These are batched in the same block - costs are pre-calculated accounting for state changes
    truthcoin_nodes
        .voter_0
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_ids[3].clone(),
            outcome_index: 0,
            shares_amount: 100000.0,
            max_cost: Some(60000),
            fee_sats: Some(1000),
            dry_run: None,
        })
        .await?;

    truthcoin_nodes
        .voter_1
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_ids[3].clone(),
            outcome_index: 1,
            shares_amount: 100000.0,
            max_cost: Some(55000),
            fee_sats: Some(1000),
            dry_run: None,
        })
        .await?;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    let final_height =
        truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        let voter_height = voter.rpc_client.getblockcount().await?;
        anyhow::ensure!(voter_height == final_height);
    }

    let final_markets = truthcoin_nodes.issuer.rpc_client.market_list().await?;
    anyhow::ensure!(final_markets.len() == 4);

    for market in &final_markets {
        let market_detail = truthcoin_nodes
            .issuer
            .rpc_client
            .market_get(market.market_id.clone())
            .await?;
        if let Some(market_data) = market_detail {
            anyhow::ensure!(market_data.treasury > 0.0);
        } else {
            anyhow::bail!("Market not found");
        }
    }

    // After all trades, verify LMSR invariants still hold
    tracing::info!("\n--- LMSR Verification: Post-Trade State ---");
    for market_id in &market_ids {
        let market_data = truthcoin_nodes
            .issuer
            .rpc_client
            .market_get(market_id.clone())
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!("Market not found: {}", market_id)
            })?;

        // LMSR Invariant 1: Prices must sum to 1.0
        let price_sum: f64 =
            market_data.outcomes.iter().map(|o| o.current_price).sum();
        anyhow::ensure!(
            (price_sum - 1.0).abs() < expected::PRICE_SUM_TOLERANCE,
            "LMSR invariant violated after trading: prices sum to {} for market {}",
            price_sum,
            market_id
        );

        // LMSR Invariant 2: All prices must be in (0, 1)
        for outcome in &market_data.outcomes {
            anyhow::ensure!(
                outcome.current_price > 0.0 && outcome.current_price < 1.0,
                "LMSR price out of bounds: {} for outcome {} in market {}",
                outcome.current_price,
                outcome.name,
                market_id
            );
        }

        // LMSR Invariant 3: Treasury should have accumulated from trading fees
        anyhow::ensure!(
            market_data.treasury > 0.0,
            "Market treasury should be positive after trades: {}",
            market_id
        );

        tracing::info!(
            "  Market {}: prices sum = {:.6}, treasury = {} sats ({:.4} BTC)",
            &market_id[..12],
            price_sum,
            (market_data.treasury * 100_000_000.0) as u64,
            market_data.treasury
        );

        // Log price distribution
        for outcome in &market_data.outcomes {
            tracing::info!(
                "    {} price: {:.4}",
                outcome.name,
                outcome.current_price
            );
        }
    }
    tracing::info!("✓ LMSR post-trade state verified\n");

    tracing::info!("✓ Phase 4: Completed 7 blocks of trading");

    // === Market UTXO Verification: Post-Trade ===
    // Verify UTXOs were consumed and recreated during trades (state transitions)
    tracing::info!("\n--- Market UTXO Verification: Post-Trade ---");
    let post_trade_utxos =
        truthcoin_nodes.issuer.rpc_client.list_utxos().await?;
    let post_trade_market_utxos =
        utxo_verification::get_market_treasury_utxos(&post_trade_utxos);
    let post_trade_fee_utxos =
        utxo_verification::get_market_author_fee_utxos(&post_trade_utxos);

    anyhow::ensure!(
        post_trade_market_utxos.len() == 4,
        "Expected 4 market treasury UTXOs after trades, found {}",
        post_trade_market_utxos.len()
    );

    for market_id in &market_ids {
        let market_id_short = &market_id[..12];

        // Verify treasury UTXO exists
        anyhow::ensure!(
            post_trade_market_utxos.contains_key(market_id),
            "Market {} should have a treasury UTXO after trades",
            market_id_short
        );

        let (new_outpoint, new_amount) = &post_trade_market_utxos[market_id];
        let (old_outpoint, old_amount) = &initial_market_utxos[market_id];

        // Verify OutPoint changed (old UTXO consumed, new one created)
        anyhow::ensure!(
            new_outpoint != old_outpoint,
            "Market {} treasury OutPoint should change after trades (old: {:?}, new: {:?})",
            market_id_short,
            old_outpoint,
            new_outpoint
        );

        // Treasury amount should have increased from trade volume
        anyhow::ensure!(
            new_amount >= old_amount,
            "Market {} treasury should not decrease: {} -> {}",
            market_id_short,
            old_amount,
            new_amount
        );

        tracing::info!(
            "  Market {}: treasury UTXO changed, amount {} -> {} sats",
            market_id_short,
            old_amount,
            new_amount
        );

        // Verify author fee UTXO exists and accumulated fees
        if let Some((_fee_outpoint, fee_amount)) =
            post_trade_fee_utxos.get(market_id)
        {
            anyhow::ensure!(
                *fee_amount > 0,
                "Market {} author fee should be > 0 after trades",
                market_id_short
            );
            tracing::info!(
                "  Market {}: author fee {} sats",
                market_id_short,
                fee_amount
            );
        }
    }
    tracing::info!("✓ Market UTXO post-trade verification complete\n");

    // === Market UTXO Value Verification: Print Actual Values ===
    // Print actual treasury and fee UTXO values for each market
    // These can be used to update expected_costs constants if implementation changes
    tracing::info!("--- Market UTXO Value Verification: Actual Values ---");

    for (market_idx, market_id) in market_ids.iter().enumerate() {
        let market_id_short = &market_id[..12];
        let (_, current_amount) = &post_trade_market_utxos[market_id];

        // Calculate treasury increase from initial
        let treasury_increase =
            current_amount.saturating_sub(expected_costs::INITIAL_TREASURY);

        tracing::info!(
            "  Market {} (idx {}): treasury = {} sats (initial {} + {} from trades)",
            market_id_short,
            market_idx,
            current_amount,
            expected_costs::INITIAL_TREASURY,
            treasury_increase
        );

        if let Some((_fee_outpoint, actual_fee)) =
            post_trade_fee_utxos.get(market_id)
        {
            tracing::info!(
                "  Market {} (idx {}): author fee = {} sats",
                market_id_short,
                market_idx,
                actual_fee
            );
        }
    }

    tracing::info!("✓ Market UTXO values logged\n");

    let current_height =
        truthcoin_nodes.issuer.rpc_client.getblockcount().await?;

    let voting_period_start_height = 31u32;
    let blocks_to_mine =
        voting_period_start_height.saturating_sub(current_height);

    for _ in 0..blocks_to_mine {
        truthcoin_nodes
            .issuer
            .bmm_single(&mut enforcer_post_setup)
            .await?;
        sleep(std::time::Duration::from_millis(500)).await;
    }

    sleep(std::time::Duration::from_secs(2)).await;

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_secs(3)).await;

    let final_height =
        truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    anyhow::ensure!(final_height >= voting_period_start_height);

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ]
    .iter()
    {
        let voter_height = voter.rpc_client.getblockcount().await?;
        anyhow::ensure!(voter_height == final_height);
    }

    let slots_at_voting = truthcoin_nodes
        .issuer
        .rpc_client
        .slot_list(Some(SlotFilter {
            period: Some(3),
            status: Some(SlotState::Voting),
        }))
        .await?;
    anyhow::ensure!(slots_at_voting.len() == 4);

    for slot in &slots_at_voting {
        let slot_detail = truthcoin_nodes
            .issuer
            .rpc_client
            .slot_get(slot.slot_id_hex.clone())
            .await?;
        let is_voting = slot_detail
            .map(|s| matches!(s.content, SlotContentInfo::Decision(_)))
            .unwrap_or(false);
        anyhow::ensure!(is_voting);
    }

    let markets_during_voting =
        truthcoin_nodes.issuer.rpc_client.market_list().await?;
    anyhow::ensure!(markets_during_voting.len() == 4);

    for market in markets_during_voting.iter() {
        anyhow::ensure!(market.state == "Trading");
    }

    for market in &markets_during_voting {
        let market_detail = truthcoin_nodes
            .issuer
            .rpc_client
            .market_get(market.market_id.clone())
            .await?;
        anyhow::ensure!(market_detail.is_some());
    }

    tracing::info!("✓ Phase 5: Markets remain Trading while slots are voting");

    let decision_slot_ids: Vec<String> = slots_at_voting
        .iter()
        .map(|slot| slot.slot_id_hex.clone())
        .collect();

    anyhow::ensure!(decision_slot_ids.len() == 4);

    // Whitepaper vote matrix (Figure 5, left example - 7 voters, 4 decisions)
    // Voter 1: [1.0, 0.5, 0.0, 0.0]
    // Voter 2: [1.0, 0.5, 0.0, 0.0]
    // Voter 3: [1.0, 1.0, 0.0, 0.0]  <- dissenter on D2
    // Voter 4: [1.0, 0.5, 0.0, 0.0]
    // Voter 5: [1.0, 0.5, 0.0, 0.0]
    // Voter 6: [1.0, 0.5, 0.0, 0.0]
    // Voter 7: [1.0, 0.5, 0.0, 0.0]
    let vote_matrix: Vec<Vec<f64>> = vec![
        vec![1.0, 0.5, 0.0, 0.0], // Voter 0 (1 in whitepaper)
        vec![1.0, 0.5, 0.0, 0.0], // Voter 1 (2 in whitepaper)
        vec![1.0, 1.0, 0.0, 0.0], // Voter 2 (3 in whitepaper) - dissenter
        vec![1.0, 0.5, 0.0, 0.0], // Voter 3 (4 in whitepaper)
        vec![1.0, 0.5, 0.0, 0.0], // Voter 4 (5 in whitepaper)
        vec![1.0, 0.5, 0.0, 0.0], // Voter 5 (6 in whitepaper)
        vec![1.0, 0.5, 0.0, 0.0], // Voter 6 (7 in whitepaper)
    ];

    // Submit votes for all 7 voters using batch submission
    let voting_period_id = 4u32;

    for (voter_idx, votes) in vote_matrix.iter().enumerate() {
        let voter = match voter_idx {
            0 => &truthcoin_nodes.voter_0,
            1 => &truthcoin_nodes.voter_1,
            2 => &truthcoin_nodes.voter_2,
            3 => &truthcoin_nodes.voter_3,
            4 => &truthcoin_nodes.voter_4,
            5 => &truthcoin_nodes.voter_5,
            6 => &truthcoin_nodes.voter_6,
            _ => unreachable!(),
        };

        let mut vote_items = Vec::new();
        for (decision_idx, &vote_value) in votes.iter().enumerate() {
            vote_items.push(VoteBatchItem {
                decision_id: decision_slot_ids[decision_idx].clone(),
                vote_value,
            });
        }

        voter.rpc_client.vote_submit(vote_items, 1000).await?;
    }

    sleep(std::time::Duration::from_millis(500)).await;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(2)).await;

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
    ] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_secs(1)).await;

    for decision_id in decision_slot_ids.iter() {
        let votes = truthcoin_nodes
            .issuer
            .rpc_client
            .vote_list(VoteFilter {
                voter: None,
                decision_id: Some(decision_id.clone()),
                period_id: None,
            })
            .await?;

        anyhow::ensure!(votes.len() == 7);

        for vote in &votes {
            let voter_idx = voter_addresses
                .iter()
                .position(|addr| addr.to_string() == vote.voter_address)
                .ok_or_else(|| anyhow::anyhow!("Unknown voter address"))?;

            let decision_idx = decision_slot_ids
                .iter()
                .position(|id| id == decision_id)
                .unwrap();

            let expected_value = vote_matrix[voter_idx][decision_idx];
            anyhow::ensure!((vote.vote_value - expected_value).abs() < 0.01);
        }
    }

    for voter_addr in voter_addresses.iter() {
        let voter_votes = truthcoin_nodes
            .issuer
            .rpc_client
            .vote_list(VoteFilter {
                voter: Some(*voter_addr),
                decision_id: None,
                period_id: Some(voting_period_id),
            })
            .await?;

        anyhow::ensure!(voter_votes.len() == 4);
    }

    tracing::info!("\n=== Vote Matrix (Bitcoin Hivemind Figure 5) ===");
    tracing::info!("       D1    D2    D3    D4");
    tracing::info!("     ╔═════╦═════╦═════╦═════╗");
    for (voter_idx, votes) in vote_matrix.iter().enumerate() {
        tracing::info!(
            "V{} → ║ {:>3} ║ {:>3} ║ {:>3} ║ {:>3} ║{}",
            voter_idx + 1,
            votes[0],
            votes[1],
            votes[2],
            votes[3],
            if voter_idx == 2 { " (dissenter)" } else { "" }
        );
    }
    tracing::info!("     ╚═════╩═════╩═════╩═════╝\n");

    tracing::info!("✓ Phase 6: Vote submission completed");

    let period_id = voting_period_id;

    let blocks_to_mine = 10;
    for _ in 1..=blocks_to_mine {
        truthcoin_nodes
            .issuer
            .bmm_single(&mut enforcer_post_setup)
            .await?;
        sleep(std::time::Duration::from_millis(100)).await;
    }

    tracing::info!("✓ Phase 7: Voting period closed");

    // ==========================================================================
    // Phase 8: Period Resolution (Consensus + Redistribution + Market Redemption)
    // ==========================================================================
    // All of the following happen atomically in a single block during connect_block:
    // 1. Consensus calculation via SVD-based PCA
    // 2. VoteCoin redistribution based on voting accuracy
    // 3. Market ossification and automatic share redemption payouts
    // ==========================================================================

    tracing::info!(
        "\n=== Phase 8: Period Resolution (Consensus + Redistribution + Redemption) ==="
    );
    tracing::info!("Mining block to trigger atomic period resolution...");

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_millis(100)).await;

    tracing::info!("\n--- 8.1: Consensus Results ---");

    let period_info = truthcoin_nodes
        .issuer
        .rpc_client
        .vote_period(Some(period_id))
        .await?
        .ok_or_else(|| anyhow::anyhow!("Period {} not found", period_id))?;

    let period_status = period_info.status.clone();

    let consensus_results = period_info
        .consensus
        .clone()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Failed to get consensus results. Period status: {}. Note: Consensus should be calculated automatically by the protocol when period closes.",
                period_status
            )
        })?;

    anyhow::ensure!(period_status == "Resolved" || period_status == "resolved");

    let mut reputation_updates_count = 0;
    let mut reputation_changes = vec![];

    for (i, voter_addr) in voter_addresses.iter().enumerate() {
        let voter_addr_str = voter_addr.to_string();

        if let Some(rep_update) =
            consensus_results.reputation_updates.get(&voter_addr_str)
        {
            reputation_updates_count += 1;
            let delta = rep_update.new_reputation - rep_update.old_reputation;
            reputation_changes.push((
                i,
                rep_update.old_reputation,
                rep_update.new_reputation,
                delta,
            ));

            let voter_info = truthcoin_nodes
                .issuer
                .rpc_client
                .vote_voter(*voter_addr)
                .await?
                .ok_or_else(|| {
                    anyhow::anyhow!("Voter not found: {}", voter_addr_str)
                })?;
            let current_rep = voter_info.reputation;

            anyhow::ensure!(
                (current_rep - rep_update.new_reputation).abs() < 0.0001
            );
        }
    }

    anyhow::ensure!(reputation_updates_count == 7);

    // === Consensus Mathematical Verification: Reputation Changes ===
    // According to the Truthcoin whitepaper, the SVD-based consensus algorithm should:
    // 1. Penalize dissenters (voters who deviate from consensus)
    // 2. Reward conformers (voters who agree with consensus)
    // Voter 2 (index 2) is the dissenter who voted 1.0 on D2 when consensus is 0.5

    tracing::info!("\n--- Consensus Verification: Reputation Algorithm ---");

    // Find dissenter's reputation change (voter index 2)
    let dissenter_change = reputation_changes
        .iter()
        .find(|(idx, _, _, _)| *idx == expected::DISSENTER_INDEX);

    anyhow::ensure!(
        dissenter_change.is_some(),
        "Dissenter (voter {}) should have a reputation update",
        expected::DISSENTER_INDEX
    );

    let (dissenter_idx, dissenter_old, dissenter_new, dissenter_delta) =
        dissenter_change.unwrap();

    // The dissenter should have lost reputation (negative delta)
    anyhow::ensure!(
        *dissenter_delta < 0.0,
        "Dissenter (V{}) should have lost reputation. Old: {:.4}, New: {:.4}, Delta: {:.4}",
        dissenter_idx + 1,
        dissenter_old,
        dissenter_new,
        dissenter_delta
    );

    tracing::info!(
        "  ✓ Dissenter V{} penalized: {:.4} → {:.4} (Δ={:+.4})",
        dissenter_idx + 1,
        dissenter_old,
        dissenter_new,
        dissenter_delta
    );

    // Verify conforming voters (all except dissenter) did not lose reputation significantly
    let mut conformers_rewarded = 0;
    for (voter_idx, old_rep, new_rep, delta) in &reputation_changes {
        if *voter_idx != expected::DISSENTER_INDEX {
            // Conformers should have non-negative delta (may be slightly negative due to smoothing)
            // but should not have lost significant reputation
            anyhow::ensure!(
                *delta >= -0.01, // Allow small negative due to smoothing alpha
                "Conformer V{} lost too much reputation: {:.4} → {:.4} (Δ={:+.4})",
                voter_idx + 1,
                old_rep,
                new_rep,
                delta
            );

            if *delta > expected::REPUTATION_TOLERANCE {
                conformers_rewarded += 1;
            }
        }
    }

    tracing::info!(
        "  ✓ {} conforming voters maintained or gained reputation",
        6 - conformers_rewarded + conformers_rewarded // All 6 conformers
    );

    // Verify the dissenter has lower reputation than conformers
    let dissenter_final = *dissenter_new;
    for (voter_idx, _, new_rep, _) in &reputation_changes {
        if *voter_idx != expected::DISSENTER_INDEX {
            anyhow::ensure!(
                dissenter_final <= *new_rep + expected::REPUTATION_TOLERANCE,
                "Dissenter should have lower or equal reputation than conformers. Dissenter: {:.4}, V{}: {:.4}",
                dissenter_final,
                voter_idx + 1,
                new_rep
            );
        }
    }

    tracing::info!(
        "  ✓ Dissenter V{} has lowest reputation: {:.4}",
        dissenter_idx + 1,
        dissenter_final
    );

    tracing::info!("✓ Consensus reputation algorithm verified\n");

    tracing::info!("Period: {} ({})", period_id, period_status);
    tracing::info!("Decision Outcomes:");
    tracing::info!(
        "  D1: {:.2}",
        consensus_results
            .outcomes
            .get(&decision_slot_ids[0])
            .unwrap_or(&0.0)
    );
    tracing::info!(
        "  D2: {:.2}",
        consensus_results
            .outcomes
            .get(&decision_slot_ids[1])
            .unwrap_or(&0.0)
    );
    tracing::info!(
        "  D3: {:.2}",
        consensus_results
            .outcomes
            .get(&decision_slot_ids[2])
            .unwrap_or(&0.0)
    );
    tracing::info!(
        "  D4: {:.2}",
        consensus_results
            .outcomes
            .get(&decision_slot_ids[3])
            .unwrap_or(&0.0)
    );
    tracing::info!("Certainty Score: {:.4}", consensus_results.certainty);
    tracing::info!("Reputation Updates:");
    for (voter_idx, old_rep, new_rep, delta) in &reputation_changes {
        tracing::info!(
            "  V{}: {:.4} → {:.4} (Δ={:+.4})",
            voter_idx + 1,
            old_rep,
            new_rep,
            delta
        );
    }
    if !consensus_results.outliers.is_empty() {
        tracing::info!("Outliers:");
        for outlier in &consensus_results.outliers {
            tracing::info!("  - {}", outlier);
        }
    }

    // Verify consensus outcomes match pre-calculated expected values
    for (i, decision_id) in decision_slot_ids.iter().enumerate() {
        let actual =
            consensus_results.outcomes.get(decision_id).unwrap_or(&-1.0);
        let expected = expected::OUTCOMES[i];
        anyhow::ensure!(
            (actual - expected).abs() < expected::OUTCOME_TOLERANCE,
            "Consensus outcome mismatch for D{}: expected {}, got {}",
            i + 1,
            expected,
            actual
        );
    }

    tracing::info!("\n--- 8.2: VoteCoin Redistribution ---");

    let redistribution_info = period_info.redistribution.clone();

    anyhow::ensure!(
        redistribution_info.is_some(),
        "Expected redistribution info for period {}, got None",
        period_id
    );

    let redist = redistribution_info.unwrap();

    tracing::info!(
        "Period: {}, Calculated at Block: {}",
        redist.period_id,
        redist.block_height
    );
    tracing::info!("VoteCoin Flow:");
    tracing::info!(
        "  Total Redistributed: {} VoteCoin",
        redist.total_redistributed
    );
    tracing::info!(
        "  Winners: {} voters gained VoteCoin",
        redist.winners_count
    );
    tracing::info!("  Losers: {} voters lost VoteCoin", redist.losers_count);
    tracing::info!("  Unchanged: {} voters", redist.unchanged_count);
    tracing::info!(
        "  Conservation: {} (sum is zero)",
        redist.conservation_check
    );
    tracing::info!(
        "  Slots affected: {} (matches our 4 decisions)",
        redist.slots_affected.len()
    );

    anyhow::ensure!(
        redist.conservation_check == 0,
        "VoteCoin conservation violated: sum = {} (expected 0)",
        redist.conservation_check
    );

    anyhow::ensure!(
        redist.slots_affected.len() == 4,
        "Expected 4 slots affected, got {}",
        redist.slots_affected.len()
    );

    anyhow::ensure!(
        redist.winners_count > 0,
        "Expected at least some winners in redistribution"
    );
    anyhow::ensure!(
        redist.losers_count > 0,
        "Expected at least some losers in redistribution"
    );

    let total_categorized =
        redist.winners_count + redist.losers_count + redist.unchanged_count;
    anyhow::ensure!(
        total_categorized == 7,
        "Expected 7 total voters, got {}",
        total_categorized
    );

    tracing::info!("\n--- 8.3: Market Ossification & Share Redemption ---");

    let ossified_markets =
        truthcoin_nodes.issuer.rpc_client.market_list().await?;
    anyhow::ensure!(
        ossified_markets.len() == 4,
        "Expected 4 markets, got {}",
        ossified_markets.len()
    );

    for market_summary in &ossified_markets {
        let market_data = truthcoin_nodes
            .issuer
            .rpc_client
            .market_get(market_summary.market_id.clone())
            .await?;

        anyhow::ensure!(
            market_data.is_some(),
            "Market {} not found after period resolution",
            market_summary.market_id
        );

        let market = market_data.unwrap();

        anyhow::ensure!(
            market.state == "Ossified",
            "Expected market {} to be Ossified, got {}",
            market_summary.market_id,
            market.state
        );

        anyhow::ensure!(
            market.treasury == 0.0,
            "Expected market {} treasury to be 0 (distributed to shareholders), got {}",
            market_summary.market_id,
            market.treasury
        );

        anyhow::ensure!(
            market.resolution.is_some(),
            "Expected market {} to have resolution info",
            market_summary.market_id
        );

        let resolution = market.resolution.unwrap();
        tracing::info!(
            "  Market {}: {} - {}",
            &market_summary.market_id[..12],
            market.state,
            resolution.summary
        );
    }

    // === Market Resolution Mathematical Verification ===
    // Verify that market payouts are based on consensus outcomes
    tracing::info!("\n--- Market Payout Verification ---");

    for market_summary in &ossified_markets {
        let market_data = truthcoin_nodes
            .issuer
            .rpc_client
            .market_get(market_summary.market_id.clone())
            .await?
            .ok_or_else(|| anyhow::anyhow!("Market not found"))?;

        let resolution = market_data
            .resolution
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No resolution for market"))?;

        // Get the expected outcome from consensus results
        // Use the market's own decision_slots field, not positional indexing
        anyhow::ensure!(
            !market_data.decision_slots.is_empty(),
            "Market {} has no decision slots",
            &market_summary.market_id[..12]
        );
        let decision_slot_id = &market_data.decision_slots[0];
        let expected_outcome = consensus_results
            .outcomes
            .get(decision_slot_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No consensus outcome for decision {}",
                    decision_slot_id
                )
            })?;

        // For binary markets resolved at extremes (0 or 1), one outcome wins fully
        // For markets resolved at 0.5 (ABSTAIN), the ABSTAIN outcome wins but is filtered
        // from valid_state_combos, resulting in "No winning outcome" display
        // Binary market outcome indices: 0 = No, 1 = Yes (from state_combo ordering)
        let tolerance = 0.01;

        if (*expected_outcome - 1.0).abs() < tolerance {
            // Consensus = 1.0 means YES won, which is outcome index 1
            anyhow::ensure!(
                !resolution.winning_outcomes.is_empty(),
                "Market {} should have winning outcomes for consensus 1.0",
                &market_summary.market_id[..12]
            );
            let yes_outcome = resolution
                .winning_outcomes
                .iter()
                .find(|o| o.outcome_index == 1);
            anyhow::ensure!(
                yes_outcome.is_some(),
                "Expected Yes (index 1) to be a winning outcome for market {} with outcome 1.0",
                &market_summary.market_id[..12]
            );
            anyhow::ensure!(
                (yes_outcome.unwrap().final_price - 1.0).abs() < tolerance,
                "Yes outcome final_price should be 1.0, got {}",
                yes_outcome.unwrap().final_price
            );
            tracing::info!(
                "  Market {}: YES won (consensus outcome = {:.2})",
                &market_summary.market_id[..12],
                expected_outcome
            );
        } else if expected_outcome.abs() < tolerance {
            // Consensus = 0.0 means NO won, which is outcome index 0
            anyhow::ensure!(
                !resolution.winning_outcomes.is_empty(),
                "Market {} should have winning outcomes for consensus 0.0",
                &market_summary.market_id[..12]
            );
            let no_outcome = resolution
                .winning_outcomes
                .iter()
                .find(|o| o.outcome_index == 0);
            anyhow::ensure!(
                no_outcome.is_some(),
                "Expected No (index 0) to be a winning outcome for market {} with outcome 0.0",
                &market_summary.market_id[..12]
            );
            anyhow::ensure!(
                (no_outcome.unwrap().final_price - 1.0).abs() < tolerance,
                "No outcome final_price should be 1.0, got {}",
                no_outcome.unwrap().final_price
            );
            tracing::info!(
                "  Market {}: NO won (consensus outcome = {:.2})",
                &market_summary.market_id[..12],
                expected_outcome
            );
        } else {
            // Consensus near 0.5 means ABSTAIN - uncertain/unresolvable decision
            // Per Truthcoin whitepaper: "we can preserve the utility of any Market
            // built with an unresolvable Decision by causing that Outcome to take
            // on the equally-spaced value of '.5'" - meaning 50/50 split
            anyhow::ensure!(
                resolution.winning_outcomes.len() == 2,
                "Market {} with ABSTAIN consensus ({:.2}) should have 2 winning outcomes (50/50 split), got {:?}",
                &market_summary.market_id[..12],
                expected_outcome,
                resolution.winning_outcomes
            );

            // Both outcomes should have ~0.5 final_price
            for winning in &resolution.winning_outcomes {
                anyhow::ensure!(
                    (winning.final_price - 0.5).abs() < tolerance,
                    "Market {} ABSTAIN outcome {} should have final_price ~0.5, got {}",
                    &market_summary.market_id[..12],
                    winning.outcome_name,
                    winning.final_price
                );
            }

            tracing::info!(
                "  Market {}: ABSTAIN - 50/50 split (consensus = {:.2})",
                &market_summary.market_id[..12],
                expected_outcome
            );
        }

        // Verify treasury was fully distributed
        anyhow::ensure!(
            market_data.treasury == 0.0,
            "Market treasury should be 0 after payout distribution, got {}",
            market_data.treasury
        );
    }

    tracing::info!("✓ Market payout verification complete\n");

    // === Market UTXO Verification: Post-Ossification ===
    // After payout distribution, market treasury UTXOs should be consumed (no longer exist)
    tracing::info!("--- Market UTXO Verification: Post-Ossification ---");
    let post_ossification_utxos =
        truthcoin_nodes.issuer.rpc_client.list_utxos().await?;
    let remaining_treasury_utxos =
        utxo_verification::get_market_treasury_utxos(&post_ossification_utxos);
    let remaining_fee_utxos = utxo_verification::get_market_author_fee_utxos(
        &post_ossification_utxos,
    );

    // All 4 markets should have their treasury UTXOs consumed
    for market_id in &market_ids {
        let market_id_short = &market_id[..12];

        anyhow::ensure!(
            !remaining_treasury_utxos.contains_key(market_id),
            "Market {} treasury UTXO should be consumed after ossification, but still exists",
            market_id_short
        );
        tracing::info!(
            "  Market {}: treasury UTXO consumed ✓",
            market_id_short
        );

        // Author fee UTXOs should also be consumed (paid to market creator)
        anyhow::ensure!(
            !remaining_fee_utxos.contains_key(market_id),
            "Market {} author fee UTXO should be consumed after ossification, but still exists",
            market_id_short
        );
        tracing::info!(
            "  Market {}: author fee UTXO consumed ✓",
            market_id_short
        );
    }

    // Verify no orphaned market UTXOs remain
    anyhow::ensure!(
        remaining_treasury_utxos.is_empty(),
        "No market treasury UTXOs should remain after ossification, found {} orphaned",
        remaining_treasury_utxos.len()
    );
    anyhow::ensure!(
        remaining_fee_utxos.is_empty(),
        "No market author fee UTXOs should remain after ossification, found {} orphaned",
        remaining_fee_utxos.len()
    );

    tracing::info!("✓ Market UTXO post-ossification verification complete\n");

    tracing::info!("\n--- 8.4: VoteCoin Conservation ---");

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
    ] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    let mut total_votecoin_snapshot = 0u32;

    let issuer_utxos =
        truthcoin_nodes.issuer.rpc_client.get_wallet_utxos().await?;
    let issuer_balance: u32 = issuer_utxos
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    total_votecoin_snapshot += issuer_balance;
    tracing::info!(
        "  Issuer wallet: {} VoteCoin (change from transfers)",
        issuer_balance
    );

    let voters_list = [
        (&truthcoin_nodes.voter_0, voter_addr_0),
        (&truthcoin_nodes.voter_1, voter_addr_1),
        (&truthcoin_nodes.voter_2, voter_addr_2),
        (&truthcoin_nodes.voter_3, voter_addr_3),
        (&truthcoin_nodes.voter_4, voter_addr_4),
        (&truthcoin_nodes.voter_5, voter_addr_5),
        (&truthcoin_nodes.voter_6, voter_addr_6),
    ];

    for (voter_idx, (voter_node, voter_addr)) in voters_list.iter().enumerate()
    {
        let voter_addr_str = voter_addr.to_string();

        let wallet_utxos = voter_node.rpc_client.get_wallet_utxos().await?;
        let balance: u32 = wallet_utxos
            .iter()
            .filter_map(|utxo| utxo.output.content.votecoin())
            .sum();

        total_votecoin_snapshot += balance;

        let voter_info = truthcoin_nodes
            .issuer
            .rpc_client
            .vote_voter(*voter_addr)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!("Voter not found: {}", voter_addr_str)
            })?;
        let reputation = voter_info.reputation;

        tracing::info!(
            "  Voter {} ({}...): {} VoteCoin (reputation: {:.4})",
            voter_idx + 1,
            &voter_addr.as_base58()[..8],
            balance,
            reputation
        );
    }

    tracing::info!(
        "  Total VoteCoin: {} (Initial: {}, Conservation: {})",
        total_votecoin_snapshot,
        INITIAL_VOTECOIN_SUPPLY,
        if total_votecoin_snapshot == INITIAL_VOTECOIN_SUPPLY {
            "PASS"
        } else {
            "FAIL"
        }
    );

    anyhow::ensure!(
        total_votecoin_snapshot == INITIAL_VOTECOIN_SUPPLY,
        "VoteCoin supply changed! Expected {}, got {}",
        INITIAL_VOTECOIN_SUPPLY,
        total_votecoin_snapshot
    );

    tracing::info!(
        "\n✓ Phase 8: Period resolution completed (consensus + redistribution + redemption)\n"
    );

    tracing::info!("=== Test Summary ===");
    tracing::info!("All phases completed successfully:");
    tracing::info!("  1. Votecoin distribution");
    tracing::info!("  2. Decision slot claims");
    tracing::info!("  3. Market creation");
    tracing::info!("  4. Trading activity");
    tracing::info!("  5. Voting period transition");
    tracing::info!("  6. Vote submission");
    tracing::info!("  7. Period closure");
    tracing::info!(
        "  8. Period resolution (consensus + redistribution + redemption)\n"
    );

    {
        drop(truthcoin_nodes);
        tracing::info!(
            "Removing {}",
            enforcer_post_setup.out_dir.path().display()
        );
        drop(enforcer_post_setup.tasks);
        sleep(std::time::Duration::from_secs(1)).await;
        enforcer_post_setup.out_dir.cleanup()?;
    }
    Ok(())
}

async fn roundtrip(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = roundtrip_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn roundtrip_trial(
    bin_paths: BinPaths,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new("roundtrip", roundtrip(bin_paths).boxed())
}
