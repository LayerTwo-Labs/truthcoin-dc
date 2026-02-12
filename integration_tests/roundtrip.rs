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
    CategoryClaimRequest, CategorySlotItem, CreateMarketRequest,
    MarketBuyRequest, MarketSellRequest, RpcClient as _, SlotContentInfo,
    SlotFilter, SlotState, VoteBatchItem, VoteFilter,
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

/// Pre-calculated LMSR values for deterministic testing.
/// All values calculated with fee_rate = 0.5%
mod expected_costs {
    /// Initial liquidity for test markets (~0.07 BTC)
    /// For binary markets, this gives β = 6931472 / ln(2) ≈ 10,000,001.18
    /// Increased 10x from original to reduce slippage sensitivity in tests
    pub const INITIAL_LIQUIDITY: u64 = 6_931_472;

    /// Explicit beta for advanced market creation path
    /// For binary markets, min treasury = β × ln(2) = 10,000,000 × 0.693... ≈ 6931472
    /// Increased 10x from original to reduce slippage sensitivity in tests
    pub const BETA: f64 = 10_000_000.0;
}

/// Expected values for Phase 2 comprehensive testing
/// All values are pre-calculated based on the vote matrix and Truthcoin consensus algorithms.
///
/// Consensus Algorithm Summary:
/// - Binary decisions: weighted_mean + catch_tl(tolerance=0.2)
///   - If mean < 0.4 → 0.0 (No)
///   - If mean > 0.6 → 1.0 (Yes)
///   - If 0.4 ≤ mean ≤ 0.6 → 0.5 (Abstain)
/// - Scaled decisions: weighted_median (continuous value in [0,1])
///
/// Final Price Calculation (calculate_final_prices):
/// - Binary: If consensus > 0.7 → Yes axis, < 0.3 → No axis, else split
/// - Scaled: Proportional [1-outcome, outcome]
/// - Multi-dim: Tensor product of per-dimension axes, normalized to sum=1.0
mod expected_phase2 {
    /// Tolerance for floating-point comparisons in assertions
    pub const FLOAT_TOLERANCE: f64 = 0.01;

    /// Scaled decision parameters and expected outcomes
    pub mod scaled {
        /// BTC price prediction range
        pub const BTC_PRICE_MIN: i64 = 10000;
        pub const BTC_PRICE_MAX: i64 = 200000;

        /// Expected consensus for scaled BTC price (weighted median)
        /// Votes: [0.75, 0.80, 0.70, 0.75, 0.78, 0.76, 0.74]
        /// Sorted: [0.70, 0.74, 0.75, 0.75, 0.76, 0.78, 0.80]
        /// With equal weights (1/7), median at 50% cumulative → 0.75
        pub const EXPECTED_BTC_CONSENSUS: f64 = 0.75;

        /// Expected ETH/BTC consensus
        /// Votes: [0.50, 0.60, 0.40, 0.50, 0.55, 0.52, 0.48]
        /// Sorted: [0.40, 0.48, 0.50, 0.50, 0.52, 0.55, 0.60]
        /// Median → 0.50
        pub const EXPECTED_ETH_BTC_CONSENSUS: f64 = 0.50;
    }

    /// Categorical decision expected outcomes
    pub mod categorical {
        /// 3-way categorical (Election): Candidate A wins
        /// cat_10 (A): 6/7 = 0.857 > 0.6 → 1.0
        /// cat_11 (B): 1/7 = 0.143 < 0.4 → 0.0
        /// cat_12 (C): 0/7 = 0.0 < 0.4 → 0.0
        pub const EXPECTED_CAT_10: f64 = 1.0;
        pub const EXPECTED_CAT_11: f64 = 0.0;
        pub const EXPECTED_CAT_12: f64 = 0.0;

        /// 4-way categorical (TVL): Ethereum wins
        /// cat_20 (ETH): 6/7 = 0.857 → 1.0
        /// cat_21 (SOL): 1/7 = 0.143 → 0.0
        /// cat_22 (ARB): 0/7 = 0.0 → 0.0
        /// cat_23 (Other): 0/7 = 0.0 → 0.0
        pub const EXPECTED_CAT_20: f64 = 1.0;
        pub const EXPECTED_CAT_21: f64 = 0.0;
        pub const EXPECTED_CAT_22: f64 = 0.0;
        pub const EXPECTED_CAT_23: f64 = 0.0;
    }

    /// Binary decision expected consensus outcomes
    /// catch_tl tolerance = 0.2: <0.4→0, >0.6→1, else→0.5
    pub mod binary {
        /// bin_30 (Inflation>3%): 5/7 = 0.714 > 0.6 → 1.0
        pub const EXPECTED_BIN_30: f64 = 1.0;
        /// bin_31 (Fed cuts): 5/7 = 0.714 > 0.6 → 1.0
        pub const EXPECTED_BIN_31: f64 = 1.0;
        /// bin_32 (Unemployment): 3/7 = 0.429, 0.4≤x≤0.6 → 0.5
        pub const EXPECTED_BIN_32: f64 = 0.5;
        /// bin_33 (GDP growth): 5/7 = 0.714 > 0.6 → 1.0
        pub const EXPECTED_BIN_33: f64 = 1.0;
        /// bin_34 (Housing): 4/7 = 0.571, 0.4≤x≤0.6 → 0.5
        pub const EXPECTED_BIN_34: f64 = 0.5;
        /// bin_35 (Consumer conf): 4/7 = 0.571 → 0.5
        pub const EXPECTED_BIN_35: f64 = 0.5;
        /// bin_36 (Retail): 4/7 = 0.571 → 0.5
        pub const EXPECTED_BIN_36: f64 = 0.5;
        /// bin_37 (Manufacturing): 4/7 = 0.571 → 0.5
        pub const EXPECTED_BIN_37: f64 = 0.5;
    }

    /// Market structure verification - expected outcome counts
    pub mod markets {
        /// Market A: Single scaled decision → 2 outcomes (No, Yes)
        pub const MARKET_A_OUTCOMES: usize = 2;
        /// Market B: 3-way categorical → 4 outcomes (3 options + residual)
        pub const MARKET_B_OUTCOMES: usize = 4;
        /// Market C: 4-way categorical + residual → 5 outcomes
        pub const MARKET_C_OUTCOMES: usize = 5;
        /// Market D: 2×2 binary (bin_30, bin_31) → 4 outcomes
        pub const MARKET_D_OUTCOMES: usize = 4;
        /// Market E: 2×2×2 binary (bin_30, bin_31, bin_32) → 8 outcomes
        pub const MARKET_E_OUTCOMES: usize = 8;
        /// Market F: 8-dimensional binary → 256 outcomes
        pub const MARKET_F_OUTCOMES: usize = 256;

        // === Mixed decision type markets ===

        /// Market G: Scaled × Binary (scaled_0 × bin_30) → 2×2 = 4 outcomes
        /// Tests proportional axis [0.25, 0.75] combined with binary axis
        pub const MARKET_G_OUTCOMES: usize = 4;

        /// Market H: Scaled × Categorical (scaled_0 × 3-way cat) → 2×4 = 8 outcomes
        /// Tests proportional axis combined with categorical (3 options + residual)
        pub const MARKET_H_OUTCOMES: usize = 8;

        /// Market I: Scaled × Scaled (scaled_0 × scaled_1) → 2×2 = 4 outcomes
        /// Tests two proportional axes: [0.25, 0.75] × [0.50, 0.50]
        pub const MARKET_I_OUTCOMES: usize = 4;

        /// Market J: Scaled × Binary × Binary × Categorical → 2×2×2×4 = 32 outcomes
        /// Tests ultimate mixed market with all dimension types
        pub const MARKET_J_OUTCOMES: usize = 32;
    }

    /// LMSR invariants
    pub mod lmsr {
        pub const PRICE_SUM_TOLERANCE: f64 = 1e-6;
    }
}

/// Assertion helpers for test verification
mod debug_helpers {
    use truthcoin_dc_app_rpc_api::MarketData;

    /// Log detailed market state for debugging (only called on assertion failures)
    fn log_market_detail(market: &MarketData, label: &str) {
        tracing::error!("=== {} Details ===", label);
        tracing::error!("  Market ID: {}", market.market_id);
        tracing::error!(
            "  State: {}, Treasury: {}",
            market.state,
            market.treasury
        );
        tracing::error!("  Outcome count: {}", market.outcomes.len());
        let price_sum: f64 =
            market.outcomes.iter().map(|o| o.current_price).sum();
        tracing::error!("  Price sum: {:.10}", price_sum);
    }

    /// Assert with detailed logging on failure
    pub fn assert_float_eq(
        actual: f64,
        expected: f64,
        tolerance: f64,
        context: &str,
    ) -> anyhow::Result<()> {
        let diff = (actual - expected).abs();
        if diff >= tolerance {
            tracing::error!("=== ASSERTION FAILED ===");
            tracing::error!("  Context: {}", context);
            tracing::error!("  Expected: {:.10}", expected);
            tracing::error!("  Actual:   {:.10}", actual);
            tracing::error!("  Diff:     {:.10}", diff);
            tracing::error!("  Tolerance: {:.10}", tolerance);
            anyhow::bail!(
                "{}: expected {}, got {} (diff {} >= tolerance {})",
                context,
                expected,
                actual,
                diff,
                tolerance
            );
        }
        Ok(())
    }

    /// Assert outcome count with detailed logging
    pub fn assert_outcome_count(
        market: &MarketData,
        expected: usize,
        market_label: &str,
    ) -> anyhow::Result<()> {
        if market.outcomes.len() != expected {
            tracing::error!("=== OUTCOME COUNT MISMATCH ===");
            log_market_detail(market, market_label);
            anyhow::bail!(
                "{} outcome count mismatch: expected {}, got {}",
                market_label,
                expected,
                market.outcomes.len()
            );
        }
        Ok(())
    }

    /// Assert LMSR price sum invariant with detailed logging
    pub fn assert_lmsr_invariant(
        market: &MarketData,
        market_label: &str,
    ) -> anyhow::Result<()> {
        let price_sum: f64 =
            market.outcomes.iter().map(|o| o.current_price).sum();
        let tolerance = super::expected_phase2::lmsr::PRICE_SUM_TOLERANCE;

        if (price_sum - 1.0).abs() >= tolerance {
            tracing::error!("=== LMSR INVARIANT VIOLATED ===");
            log_market_detail(market, market_label);
            anyhow::bail!(
                "{} LMSR invariant violated: prices sum to {}, expected 1.0",
                market_label,
                price_sum
            );
        }
        Ok(())
    }

    /// Assert ossified market treasury is zero with detailed logging
    pub fn assert_treasury_zero(
        market: &MarketData,
        market_id: &str,
    ) -> anyhow::Result<()> {
        if market.state == "Ossified" && market.treasury != 0.0 {
            tracing::error!("=== TREASURY NOT ZERO FOR OSSIFIED MARKET ===");
            log_market_detail(market, &format!("Market {market_id}"));
            anyhow::bail!(
                "Ossified market {} should have zero treasury, got {}",
                market_id,
                market.treasury
            );
        }
        Ok(())
    }

    /// Assert VoteCoin conservation with detailed logging
    pub fn assert_votecoin_conservation(
        actual: u32,
        expected: u32,
        voter_balances: &[(String, u32)],
    ) -> anyhow::Result<()> {
        if actual != expected {
            tracing::error!("=== VOTECOIN CONSERVATION VIOLATED ===");
            tracing::error!("  Expected total: {}", expected);
            tracing::error!("  Actual total:   {}", actual);
            tracing::error!(
                "  Difference:     {}",
                (actual as i64 - expected as i64).abs()
            );
            tracing::error!("  Individual balances:");
            for (name, balance) in voter_balances {
                tracing::error!("    {}: {}", name, balance);
            }
            anyhow::bail!(
                "VoteCoin conservation violated! Expected {}, got {}",
                expected,
                actual
            );
        }
        Ok(())
    }
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
                if let FilledOutputContent::MarketFunds {
                    market_id,
                    amount,
                    is_fee: false,
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
                if let FilledOutputContent::MarketFunds {
                    market_id,
                    amount,
                    is_fee: true,
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
        Ok(res)
    }
}

// Increased 10x to match 10x increase in market liquidity (INITIAL_LIQUIDITY/BETA)
const DEPOSIT_AMOUNT: bitcoin::Amount = bitcoin::Amount::from_sat(210_000_000);
const DEPOSIT_FEE: bitcoin::Amount = bitcoin::Amount::from_sat(1_000_000);

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
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
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

    // Increased 10x to match 10x increase in market liquidity (INITIAL_LIQUIDITY/BETA)
    const VOTER_DEPOSIT_AMOUNT: bitcoin::Amount =
        bitcoin::Amount::from_sat(50_000_000);
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

    // Extra deposit for voter_1 to ensure they have funds for voting after trading
    let voter_1_deposit_address_2 =
        truthcoin_nodes.voter_1.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_1,
        &voter_1_deposit_address_2,
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

    // Create markets using both valid parameter paths:
    // - Market 0: initial_liquidity only (primary user-facing path)
    // - Market 1: beta only (advanced path)
    // - Market 2: initial_liquidity only (different amount)
    // - Market 3: beta only (same beta, verifies consistency)
    // Note: "neither" path uses DEFAULT_MARKET_BETA=7.0 which is too low for trading tests
    for (idx, (voter_node, slot_idx, title, description)) in
        market_configs.iter().enumerate()
    {
        use truthcoin_dc_app_rpc_api::CreateMarketRequest;

        let slot_id = &market_slot_ids[*slot_idx];
        let dimensions = format!("[{slot_id}]");

        let (beta, initial_liquidity) = match idx {
            // Market 0: initial_liquidity only (primary user-facing)
            0 => (None, Some(expected_costs::INITIAL_LIQUIDITY)),
            // Market 1: beta only (advanced)
            1 => (Some(expected_costs::BETA), None),
            // Market 2: initial_liquidity only (higher liquidity)
            2 => (None, Some(expected_costs::INITIAL_LIQUIDITY * 2)),
            // Market 3: beta only (same beta as market 1)
            _ => (Some(expected_costs::BETA), None),
        };

        let request = CreateMarketRequest {
            title: title.to_string(),
            description: description.to_string(),
            dimensions,
            beta,
            trading_fee: Some(0.005),
            tags: Some(vec!["integration-test".to_string()]),
            initial_liquidity,
            category_txids: None,
            residual_names: None,
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

    // Verify LMSR invariants for initial market state
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
    }

    // Verify each market has a treasury UTXO after creation
    let all_utxos = truthcoin_nodes.issuer.rpc_client.list_utxos().await?;
    let initial_market_utxos =
        utxo_verification::get_market_treasury_utxos(&all_utxos);

    anyhow::ensure!(
        initial_market_utxos.len() == 4,
        "Expected 4 market treasury UTXOs after creation, found {}",
        initial_market_utxos.len()
    );

    for market_id in &market_ids {
        anyhow::ensure!(
            initial_market_utxos.contains_key(market_id),
            "Market {} should have a treasury UTXO after creation",
            market_id
        );
    }

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
        // With 10x higher beta, slippage is much lower so this max_cost has plenty of room
        voter
            .rpc_client
            .market_buy(MarketBuyRequest {
                market_id: market_id.clone(),
                outcome_index: 0,
                shares_amount: 50000,
                max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
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
            shares_amount: 50000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
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
            shares_amount: 500000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
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
                shares_amount: 20000,
                max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
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
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
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
            shares_amount: 200000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
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
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
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
            shares_amount: 150000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
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
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
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
            shares_amount: 100000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;

    truthcoin_nodes
        .voter_1
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_ids[3].clone(),
            outcome_index: 1,
            shares_amount: 100000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
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
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
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
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
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

    // Verify LMSR invariants still hold after all trades
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
            "LMSR invariant violated after trading: prices sum to {} for market {}",
            price_sum,
            market_id
        );

        for outcome in &market_data.outcomes {
            anyhow::ensure!(
                outcome.current_price > 0.0 && outcome.current_price < 1.0,
                "LMSR price out of bounds: {} for outcome {} in market {}",
                outcome.current_price,
                outcome.name,
                market_id
            );
        }

        anyhow::ensure!(
            market_data.treasury > 0.0,
            "Market treasury should be positive after trades: {}",
            market_id
        );
    }

    tracing::info!("✓ Phase 4: Completed 7 blocks of trading");

    // Verify UTXOs were consumed and recreated during trades
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
        anyhow::ensure!(
            post_trade_market_utxos.contains_key(market_id),
            "Market {} should have a treasury UTXO after trades",
            market_id
        );

        let (new_outpoint, new_amount) = &post_trade_market_utxos[market_id];
        let (old_outpoint, old_amount) = &initial_market_utxos[market_id];

        anyhow::ensure!(
            new_outpoint != old_outpoint,
            "Market {} treasury OutPoint should change after trades",
            market_id
        );

        anyhow::ensure!(
            new_amount >= old_amount,
            "Market {} treasury should not decrease: {} -> {}",
            market_id,
            old_amount,
            new_amount
        );

        if let Some((_fee_outpoint, fee_amount)) =
            post_trade_fee_utxos.get(market_id)
        {
            anyhow::ensure!(
                *fee_amount > 0,
                "Market {} author fee should be > 0 after trades",
                market_id
            );
        }
    }

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
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
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
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
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

    let voters = [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
    ];
    let voting_period_id = 4u32;

    for (voter_idx, votes) in vote_matrix.iter().enumerate() {
        let voter = voters[voter_idx];

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

    for voter in voters {
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

    // Mine block to trigger period resolution
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_millis(100)).await;

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

    // Verify dissenter was penalized (voter 2 voted 1.0 on D2 when consensus is 0.5)
    let dissenter_change = reputation_changes
        .iter()
        .find(|(idx, _, _, _)| *idx == expected::DISSENTER_INDEX);

    anyhow::ensure!(
        dissenter_change.is_some(),
        "Dissenter (voter {}) should have a reputation update",
        expected::DISSENTER_INDEX
    );

    let (_dissenter_idx, _dissenter_old, dissenter_new, dissenter_delta) =
        dissenter_change.unwrap();

    anyhow::ensure!(
        *dissenter_delta < 0.0,
        "Dissenter should have lost reputation, delta: {:.4}",
        dissenter_delta
    );

    // Verify conforming voters did not lose significant reputation
    for (voter_idx, _old_rep, _new_rep, delta) in &reputation_changes {
        if *voter_idx != expected::DISSENTER_INDEX {
            anyhow::ensure!(
                *delta >= -0.01,
                "Conformer V{} lost too much reputation (Δ={:+.4})",
                voter_idx + 1,
                delta
            );
        }
    }

    // Verify the dissenter has lower reputation than conformers
    let dissenter_final = *dissenter_new;
    for (voter_idx, _, new_rep, _) in &reputation_changes {
        if *voter_idx != expected::DISSENTER_INDEX {
            anyhow::ensure!(
                dissenter_final <= *new_rep + expected::REPUTATION_TOLERANCE,
                "Dissenter should have lower or equal reputation than conformers"
            );
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

    // Verify VoteCoin redistribution
    let redistribution_info = period_info.redistribution.clone();
    anyhow::ensure!(
        redistribution_info.is_some(),
        "Expected redistribution info for period {}, got None",
        period_id
    );

    let redist = redistribution_info.unwrap();
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

    // Verify market ossification and share redemption
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
            "Expected market {} treasury to be 0, got {}",
            market_summary.market_id,
            market.treasury
        );
        anyhow::ensure!(
            market.resolution.is_some(),
            "Expected market {} to have resolution info",
            market_summary.market_id
        );
    }

    // Verify market payouts based on consensus outcomes
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

        anyhow::ensure!(
            !market_data.decision_slots.is_empty(),
            "Market {} has no decision slots",
            &market_summary.market_id
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

        let tolerance = 0.01;

        if (*expected_outcome - 1.0).abs() < tolerance {
            anyhow::ensure!(
                !resolution.winning_outcomes.is_empty(),
                "Market {} should have winning outcomes for consensus 1.0",
                &market_summary.market_id
            );
            let yes_outcome = resolution
                .winning_outcomes
                .iter()
                .find(|o| o.outcome_index == 1);
            anyhow::ensure!(
                yes_outcome.is_some(),
                "Expected Yes (index 1) to be a winning outcome"
            );
            anyhow::ensure!(
                (yes_outcome.unwrap().final_price - 1.0).abs() < tolerance,
                "Yes outcome final_price should be 1.0"
            );
        } else if expected_outcome.abs() < tolerance {
            anyhow::ensure!(
                !resolution.winning_outcomes.is_empty(),
                "Market {} should have winning outcomes for consensus 0.0",
                &market_summary.market_id
            );
            let no_outcome = resolution
                .winning_outcomes
                .iter()
                .find(|o| o.outcome_index == 0);
            anyhow::ensure!(
                no_outcome.is_some(),
                "Expected No (index 0) to be a winning outcome"
            );
            anyhow::ensure!(
                (no_outcome.unwrap().final_price - 1.0).abs() < tolerance,
                "No outcome final_price should be 1.0"
            );
        } else {
            anyhow::ensure!(
                resolution.winning_outcomes.len() == 2,
                "Market with ABSTAIN consensus should have 2 winning outcomes (50/50 split)"
            );
            for winning in &resolution.winning_outcomes {
                anyhow::ensure!(
                    (winning.final_price - 0.5).abs() < tolerance,
                    "ABSTAIN outcome should have final_price ~0.5"
                );
            }
        }

        anyhow::ensure!(
            market_data.treasury == 0.0,
            "Market treasury should be 0 after payout distribution"
        );
    }

    // Verify market UTXOs consumed after payout distribution
    let post_ossification_utxos =
        truthcoin_nodes.issuer.rpc_client.list_utxos().await?;
    let remaining_treasury_utxos =
        utxo_verification::get_market_treasury_utxos(&post_ossification_utxos);
    let remaining_fee_utxos = utxo_verification::get_market_author_fee_utxos(
        &post_ossification_utxos,
    );

    for market_id in &market_ids {
        anyhow::ensure!(
            !remaining_treasury_utxos.contains_key(market_id),
            "Market {} treasury UTXO should be consumed after ossification",
            market_id
        );
        anyhow::ensure!(
            !remaining_fee_utxos.contains_key(market_id),
            "Market {} author fee UTXO should be consumed after ossification",
            market_id
        );
    }

    anyhow::ensure!(
        remaining_treasury_utxos.is_empty(),
        "No market treasury UTXOs should remain after ossification"
    );
    anyhow::ensure!(
        remaining_fee_utxos.is_empty(),
        "No market author fee UTXOs should remain after ossification"
    );

    // Verify VoteCoin conservation
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

    let voters_list = [
        (&truthcoin_nodes.voter_0, voter_addr_0),
        (&truthcoin_nodes.voter_1, voter_addr_1),
        (&truthcoin_nodes.voter_2, voter_addr_2),
        (&truthcoin_nodes.voter_3, voter_addr_3),
        (&truthcoin_nodes.voter_4, voter_addr_4),
        (&truthcoin_nodes.voter_5, voter_addr_5),
        (&truthcoin_nodes.voter_6, voter_addr_6),
    ];

    for (voter_node, _voter_addr) in voters_list.iter() {
        let wallet_utxos = voter_node.rpc_client.get_wallet_utxos().await?;
        let balance: u32 = wallet_utxos
            .iter()
            .filter_map(|utxo| utxo.output.content.votecoin())
            .sum();
        total_votecoin_snapshot += balance;
    }

    anyhow::ensure!(
        total_votecoin_snapshot == INITIAL_VOTECOIN_SUPPLY,
        "VoteCoin supply changed! Expected {}, got {}",
        INITIAL_VOTECOIN_SUPPLY,
        total_votecoin_snapshot
    );

    tracing::info!("✓ Phase 8: Period resolution completed");

    // Fund all voters with multiple UTXOs for comprehensive testing

    // Increased 10x to match 10x increase in market liquidity (INITIAL_LIQUIDITY/BETA)
    const PHASE9_DEPOSIT_AMOUNT: bitcoin::Amount =
        bitcoin::Amount::from_sat(50_000_000);
    const PHASE9_DEPOSIT_FEE: bitcoin::Amount =
        bitcoin::Amount::from_sat(500_000);
    const UTXOS_PER_NODE: usize = 5;

    let voter_nodes_mut = [
        &mut truthcoin_nodes.voter_0,
        &mut truthcoin_nodes.voter_1,
        &mut truthcoin_nodes.voter_2,
        &mut truthcoin_nodes.voter_3,
        &mut truthcoin_nodes.voter_4,
        &mut truthcoin_nodes.voter_5,
        &mut truthcoin_nodes.voter_6,
    ];

    for voter in voter_nodes_mut.into_iter() {
        for _ in 0..UTXOS_PER_NODE {
            let deposit_address = voter.get_deposit_address().await?;
            deposit(
                &mut enforcer_post_setup,
                voter,
                &deposit_address,
                PHASE9_DEPOSIT_AMOUNT,
                PHASE9_DEPOSIT_FEE,
            )
            .await?;
            sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    for _ in 0..UTXOS_PER_NODE {
        let issuer_deposit_address =
            truthcoin_nodes.issuer.get_deposit_address().await?;
        deposit(
            &mut enforcer_post_setup,
            &mut truthcoin_nodes.issuer,
            &issuer_deposit_address,
            PHASE9_DEPOSIT_AMOUNT,
            PHASE9_DEPOSIT_FEE,
        )
        .await?;
        sleep(std::time::Duration::from_millis(100)).await;
    }

    sleep(std::time::Duration::from_secs(1)).await;

    // ==========================================================================
    // Phase 9: Additional Slot Claims
    // Claims: scaled decisions, categorical slots, additional binary slots
    // ==========================================================================

    // Calculate dynamic test period based on current block height
    let phase9_height =
        truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    let current_period = phase9_height / 10;
    let test_period = current_period + 3;

    // Refresh all voter wallets
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

    // Claim scaled decision for Market A (BTC price prediction)
    truthcoin_nodes
        .voter_0
        .rpc_client
        .slot_claim(
            test_period,
            0,    // slot index
            true, // is_standard
            true, // is_scaled
            "What will BTC price be EOY 2025? (USD)".to_string(),
            Some(expected_phase2::scaled::BTC_PRICE_MIN),
            Some(expected_phase2::scaled::BTC_PRICE_MAX),
            None,
            None,
            1000,
        )
        .await?;

    // Claim categorical slots for Market B (3-way: Election)
    let cat_slot_10_hex = format!("{:06x}", (test_period << 14) | 10);
    let cat_slot_11_hex = format!("{:06x}", (test_period << 14) | 11);
    let cat_slot_12_hex = format!("{:06x}", (test_period << 14) | 12);

    let category_b_txid = truthcoin_nodes
        .voter_1
        .rpc_client
        .slot_claim_category(CategoryClaimRequest {
            slots: vec![
                CategorySlotItem {
                    slot_id_hex: cat_slot_10_hex.clone(),
                    question: "Will Candidate A win the 2028 election?"
                        .to_string(),
                },
                CategorySlotItem {
                    slot_id_hex: cat_slot_11_hex.clone(),
                    question: "Will Candidate B win the 2028 election?"
                        .to_string(),
                },
                CategorySlotItem {
                    slot_id_hex: cat_slot_12_hex.clone(),
                    question: "Will Candidate C win the 2028 election?"
                        .to_string(),
                },
            ],
            is_standard: true,
            fee_sats: 1000,
        })
        .await?;

    // Claim categorical slots for Market C (4-way: TVL)
    let cat_slot_20_hex = format!("{:06x}", (test_period << 14) | 20);
    let cat_slot_21_hex = format!("{:06x}", (test_period << 14) | 21);
    let cat_slot_22_hex = format!("{:06x}", (test_period << 14) | 22);
    let cat_slot_23_hex = format!("{:06x}", (test_period << 14) | 23);

    let category_c_txid = truthcoin_nodes
        .voter_2
        .rpc_client
        .slot_claim_category(CategoryClaimRequest {
            slots: vec![
                CategorySlotItem {
                    slot_id_hex: cat_slot_20_hex.clone(),
                    question: "Will Ethereum have highest TVL?".to_string(),
                },
                CategorySlotItem {
                    slot_id_hex: cat_slot_21_hex.clone(),
                    question: "Will Solana have highest TVL?".to_string(),
                },
                CategorySlotItem {
                    slot_id_hex: cat_slot_22_hex.clone(),
                    question: "Will Arbitrum have highest TVL?".to_string(),
                },
                CategorySlotItem {
                    slot_id_hex: cat_slot_23_hex.clone(),
                    question: "Will another chain have highest TVL?"
                        .to_string(),
                },
            ],
            is_standard: true,
            fee_sats: 1000,
        })
        .await?;

    // Claim binary slots for Markets D, E, and F
    truthcoin_nodes
        .voter_3
        .rpc_client
        .slot_claim(
            test_period,
            30,
            true,  // is_standard
            false, // is_scaled (binary)
            "Will inflation exceed 3% in 2025?".to_string(),
            None,
            None,
            None,
            None,
            1000,
        )
        .await?;

    truthcoin_nodes
        .voter_4
        .rpc_client
        .slot_claim(
            test_period,
            31,
            true,
            false,
            "Will the Fed cut rates in 2025?".to_string(),
            None,
            None,
            None,
            None,
            1000,
        )
        .await?;

    truthcoin_nodes
        .voter_5
        .rpc_client
        .slot_claim(
            test_period,
            32,
            true,
            false,
            "Will unemployment rise above 5%?".to_string(),
            None,
            None,
            None,
            None,
            1000,
        )
        .await?;

    truthcoin_nodes
        .voter_6
        .rpc_client
        .slot_claim(
            test_period,
            33,
            true,
            false,
            "Will GDP growth exceed 3% in 2025?".to_string(),
            None,
            None,
            None,
            None,
            1000,
        )
        .await?;

    truthcoin_nodes
        .voter_0
        .rpc_client
        .slot_claim(
            test_period,
            34,
            true,
            false,
            "Will housing prices rise in 2025?".to_string(),
            None,
            None,
            None,
            None,
            1000,
        )
        .await?;

    truthcoin_nodes
        .voter_1
        .rpc_client
        .slot_claim(
            test_period,
            35,
            true,
            false,
            "Will consumer confidence increase?".to_string(),
            None,
            None,
            None,
            None,
            1000,
        )
        .await?;

    truthcoin_nodes
        .voter_2
        .rpc_client
        .slot_claim(
            test_period,
            36,
            true,
            false,
            "Will retail sales exceed $7T?".to_string(),
            None,
            None,
            None,
            None,
            1000,
        )
        .await?;

    truthcoin_nodes
        .voter_3
        .rpc_client
        .slot_claim(
            test_period,
            37,
            true,
            false,
            "Will manufacturing output increase?".to_string(),
            None,
            None,
            None,
            None,
            1000,
        )
        .await?;

    // Claim second scaled slot for Market I (ETH/BTC ratio)
    truthcoin_nodes
        .voter_4
        .rpc_client
        .slot_claim(
            test_period,
            1,
            true,
            true,
            "ETH/BTC ratio at end of 2025".to_string(),
            Some(0),
            Some(100),
            None,
            None,
            1000,
        )
        .await?;

    sleep(std::time::Duration::from_secs(2)).await;
    truthcoin_nodes.issuer.rpc_client.refresh_wallet().await?;

    // Mine block to confirm claims
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    // Check slots and mine additional block if needed
    let claimed_slots = truthcoin_nodes
        .issuer
        .rpc_client
        .slot_list(Some(SlotFilter {
            period: Some(test_period),
            status: Some(SlotState::Claimed),
        }))
        .await?;

    if claimed_slots.len() < 4 {
        truthcoin_nodes
            .issuer
            .bmm_single(&mut enforcer_post_setup)
            .await?;
        sleep(std::time::Duration::from_secs(1)).await;
    }

    let claimed_slots = truthcoin_nodes
        .issuer
        .rpc_client
        .slot_list(Some(SlotFilter {
            period: Some(test_period),
            status: Some(SlotState::Claimed),
        }))
        .await?;

    anyhow::ensure!(
        claimed_slots.len() >= 11,
        "Expected at least 11 claimed slots in period {}, found {}",
        test_period,
        claimed_slots.len()
    );

    tracing::info!("✓ Phase 9: Claimed {} slots", claimed_slots.len());

    // Sync all nodes
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
    truthcoin_nodes.issuer.rpc_client.refresh_wallet().await?;

    // Collect slot IDs for market creation
    let scaled_slot_0 = format!("{:06x}", test_period << 14);
    let scaled_slot_1 = format!("{:06x}", (test_period << 14) | 1);
    let cat_slot_10 = format!("{:06x}", (test_period << 14) | 10);
    let cat_slot_11 = format!("{:06x}", (test_period << 14) | 11);
    let cat_slot_12 = format!("{:06x}", (test_period << 14) | 12);
    let cat_slot_20 = format!("{:06x}", (test_period << 14) | 20);
    let cat_slot_21 = format!("{:06x}", (test_period << 14) | 21);
    let cat_slot_22 = format!("{:06x}", (test_period << 14) | 22);
    let cat_slot_23 = format!("{:06x}", (test_period << 14) | 23);
    let bin_slot_30 = format!("{:06x}", (test_period << 14) | 30);
    let bin_slot_31 = format!("{:06x}", (test_period << 14) | 31);
    let bin_slot_32 = format!("{:06x}", (test_period << 14) | 32);
    let bin_slot_33 = format!("{:06x}", (test_period << 14) | 33);
    let bin_slot_34 = format!("{:06x}", (test_period << 14) | 34);
    let bin_slot_35 = format!("{:06x}", (test_period << 14) | 35);
    let bin_slot_36 = format!("{:06x}", (test_period << 14) | 36);
    let bin_slot_37 = format!("{:06x}", (test_period << 14) | 37);

    // Refresh wallets for market creation (including issuer for mempool sync)
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
    truthcoin_nodes.issuer.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_millis(500)).await;

    // Create Market A: Single scaled decision
    let market_a_id = truthcoin_nodes
        .voter_0
        .rpc_client
        .market_create(CreateMarketRequest {
            title: "BTC Price Prediction Market".to_string(),
            description: "Market based on BTC price prediction".to_string(),
            dimensions: format!("[{scaled_slot_0}]"),
            beta: Some(expected_costs::BETA),
            trading_fee: Some(0.005),
            tags: Some(vec!["scaled".to_string(), "phase2".to_string()]),
            initial_liquidity: None,
            category_txids: None,
            residual_names: None,
            fee_sats: 1000,
        })
        .await?;

    // Create Market B: 3-way categorical
    let market_b_id = truthcoin_nodes
        .voter_1
        .rpc_client
        .market_create(CreateMarketRequest {
            title: "2028 Presidential Election".to_string(),
            description: "Who will win the 2028 presidential election?"
                .to_string(),
            dimensions: format!(
                "[[{cat_slot_10},{cat_slot_11},{cat_slot_12}]]"
            ),
            beta: Some(expected_costs::BETA),
            trading_fee: Some(0.005),
            tags: Some(vec!["categorical".to_string(), "election".to_string()]),
            initial_liquidity: None,
            category_txids: Some(vec![format!("{}", category_b_txid)]),
            residual_names: None,
            fee_sats: 1000,
        })
        .await?;

    // Create Market C: 4-way categorical
    let market_c_id = truthcoin_nodes
        .voter_2
        .rpc_client
        .market_create(CreateMarketRequest {
            title: "2025 DeFi TVL Leader".to_string(),
            description:
                "Which chain will have the highest TVL at end of 2025?"
                    .to_string(),
            dimensions: format!(
                "[[{cat_slot_20},{cat_slot_21},{cat_slot_22},{cat_slot_23}]]"
            ),
            beta: Some(expected_costs::BETA),
            trading_fee: Some(0.005),
            tags: Some(vec!["categorical".to_string(), "defi".to_string()]),
            initial_liquidity: None,
            category_txids: Some(vec![format!("{}", category_c_txid)]),
            residual_names: None,
            fee_sats: 1000,
        })
        .await?;

    // Create Market D: 2×2 binary
    let market_d_id = truthcoin_nodes
        .voter_3
        .rpc_client
        .market_create(CreateMarketRequest {
            title: "Inflation & Fed Policy".to_string(),
            description: "Combined market on inflation and Fed rate decisions"
                .to_string(),
            dimensions: format!("[{bin_slot_30},{bin_slot_31}]"),
            beta: Some(expected_costs::BETA),
            trading_fee: Some(0.005),
            tags: Some(vec!["binary".to_string(), "macro".to_string()]),
            initial_liquidity: None,
            category_txids: None,
            residual_names: None,
            fee_sats: 1000,
        })
        .await?;

    // Create Market E: 2×2×2 binary
    let market_e_id = truthcoin_nodes
        .voter_4
        .rpc_client
        .market_create(CreateMarketRequest {
            title: "Macro Indicators 2025".to_string(),
            description:
                "Combined market on inflation, Fed policy, and unemployment"
                    .to_string(),
            dimensions: format!("[{bin_slot_30},{bin_slot_31},{bin_slot_32}]"),
            beta: Some(expected_costs::BETA),
            trading_fee: Some(0.005),
            tags: Some(vec!["binary".to_string(), "macro".to_string()]),
            initial_liquidity: None,
            category_txids: None,
            residual_names: None,
            fee_sats: 1000,
        })
        .await?;

    // Create Market F: 8-dimensional binary (256 outcomes)
    let market_f_id = truthcoin_nodes
        .voter_5
        .rpc_client
        .market_create(CreateMarketRequest {
            title: "2025 Macro Indicators Full".to_string(),
            description: "8-way binary market on macro indicators".to_string(),
            dimensions: format!("[{bin_slot_30},{bin_slot_31},{bin_slot_32},{bin_slot_33},{bin_slot_34},{bin_slot_35},{bin_slot_36},{bin_slot_37}]"),
            beta: Some(expected_costs::BETA),
            trading_fee: Some(0.005),
            tags: Some(vec!["binary".to_string(), "stress-test".to_string()]),
            initial_liquidity: None,
            category_txids: None,
            residual_names: None,
            fee_sats: 1000,
        })
        .await?;

    // Create Market G: Scaled × Binary
    let market_g_id = truthcoin_nodes
        .voter_6
        .rpc_client
        .market_create(CreateMarketRequest {
            title: "BTC Price vs Inflation".to_string(),
            description: "Combined scaled (BTC) and binary (inflation) market"
                .to_string(),
            dimensions: format!("[{scaled_slot_0},{bin_slot_30}]"),
            beta: Some(expected_costs::BETA),
            trading_fee: Some(0.005),
            tags: Some(vec!["mixed".to_string(), "scaled-binary".to_string()]),
            initial_liquidity: None,
            category_txids: None,
            residual_names: None,
            fee_sats: 1000,
        })
        .await?;

    // Create Market H: Scaled × Categorical
    let market_h_id = truthcoin_nodes
        .voter_0
        .rpc_client
        .market_create(CreateMarketRequest {
            title: "BTC Price vs Election".to_string(),
            description:
                "Combined scaled (BTC) and categorical (election) market"
                    .to_string(),
            dimensions: format!(
                "[{scaled_slot_0},[{cat_slot_10},{cat_slot_11},{cat_slot_12}]]"
            ),
            beta: Some(expected_costs::BETA),
            trading_fee: Some(0.005),
            tags: Some(vec![
                "mixed".to_string(),
                "scaled-categorical".to_string(),
            ]),
            initial_liquidity: None,
            category_txids: Some(vec![format!("{}", category_b_txid)]),
            residual_names: None,
            fee_sats: 1000,
        })
        .await?;

    // Create Market I: Scaled × Scaled
    let market_i_id = truthcoin_nodes
        .voter_1
        .rpc_client
        .market_create(CreateMarketRequest {
            title: "BTC Price vs ETH/BTC Ratio".to_string(),
            description: "Combined two-scaled market".to_string(),
            dimensions: format!("[{scaled_slot_0},{scaled_slot_1}]"),
            beta: Some(expected_costs::BETA),
            trading_fee: Some(0.005),
            tags: Some(vec!["mixed".to_string(), "scaled-scaled".to_string()]),
            initial_liquidity: None,
            category_txids: None,
            residual_names: None,
            fee_sats: 1000,
        })
        .await?;

    // Create Market J: Scaled × Binary × Binary × Categorical (32 outcomes)
    let market_j_id = truthcoin_nodes
        .voter_2
        .rpc_client
        .market_create(CreateMarketRequest {
            title: "Ultimate Macro Predictor".to_string(),
            description: "Combined scaled, binary, and categorical market"
                .to_string(),
            dimensions: format!("[{scaled_slot_0},{bin_slot_30},{bin_slot_31},[{cat_slot_10},{cat_slot_11},{cat_slot_12}]]"),
            beta: Some(expected_costs::BETA),
            trading_fee: Some(0.005),
            tags: Some(vec!["mixed".to_string(), "ultimate".to_string()]),
            initial_liquidity: None,
            category_txids: Some(vec![format!("{}", category_b_txid)]),
            residual_names: None,
            fee_sats: 1000,
        })
        .await?;

    // Wait for P2P propagation and mine block
    sleep(std::time::Duration::from_secs(2)).await;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(2)).await;

    // Refresh all wallets after mining
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
    truthcoin_nodes.issuer.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_millis(500)).await;

    // Verify markets created correctly
    let phase2_markets =
        truthcoin_nodes.issuer.rpc_client.market_list().await?;
    let phase2_market_ids = [
        &market_a_id,
        &market_b_id,
        &market_c_id,
        &market_d_id,
        &market_e_id,
        &market_f_id,
        &market_g_id,
        &market_h_id,
        &market_i_id,
        &market_j_id,
    ];
    // Verify LMSR invariant for all phase 2 markets
    for market in &phase2_markets {
        if phase2_market_ids.contains(&&market.market_id) {
            let market_detail = truthcoin_nodes
                .issuer
                .rpc_client
                .market_get(market.market_id.clone())
                .await?
                .ok_or_else(|| anyhow::anyhow!("Market not found"))?;

            let price_sum: f64 =
                market_detail.outcomes.iter().map(|o| o.current_price).sum();

            anyhow::ensure!(
                (price_sum - 1.0).abs() < expected::PRICE_SUM_TOLERANCE,
                "LMSR invariant violated for market {}: prices sum to {}",
                &market.market_id,
                price_sum
            );
        }
    }

    let market_a_detail = truthcoin_nodes
        .issuer
        .rpc_client
        .market_get(market_a_id.clone())
        .await?
        .ok_or_else(|| anyhow::anyhow!("Market A not found"))?;

    anyhow::ensure!(
        market_a_detail.outcomes.len() == 2,
        "Market A should have 2 outcomes, found {}",
        market_a_detail.outcomes.len()
    );

    tracing::info!("✓ Phase 10: Created 10 markets (A-J)");

    // Sync all voters to the issuer's tip
    let issuer_tip = truthcoin_nodes
        .issuer
        .rpc_client
        .get_best_sidechain_block_hash()
        .await?
        .expect("Issuer should have a tip");

    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
    ] {
        drop(voter.rpc_client.sync_to_tip(issuer_tip).await);
        voter.rpc_client.refresh_wallet().await?;
    }

    // Trade on Market A (scaled)
    truthcoin_nodes
        .voter_0
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_a_id.clone(),
            outcome_index: 0, // Yes
            shares_amount: 30000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;

    // Trade on Market B (3-way categorical)
    truthcoin_nodes
        .voter_1
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_b_id.clone(),
            outcome_index: 0, // Candidate A
            shares_amount: 40000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;
    truthcoin_nodes
        .voter_2
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_b_id.clone(),
            outcome_index: 1, // Candidate B
            shares_amount: 20000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;

    // Mine single block with all trades
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    // Sync and refresh wallets
    let issuer_tip = truthcoin_nodes
        .issuer
        .rpc_client
        .get_best_sidechain_block_hash()
        .await?
        .expect("Issuer should have a tip");
    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
    ] {
        let _sync_result = voter.rpc_client.sync_to_tip(issuer_tip).await;
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    // Test 1: Multiple buys on same market, different outcomes
    truthcoin_nodes
        .voter_3
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_a_id.clone(),
            outcome_index: 0,
            shares_amount: 25000,
            max_cost: Some(10_000_000),
            dry_run: None,
        })
        .await?;

    truthcoin_nodes
        .voter_4
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_a_id.clone(),
            outcome_index: 1,
            shares_amount: 20000,
            max_cost: Some(10_000_000),
            dry_run: None,
        })
        .await?;

    // voter_5 buys outcome 2 on Market B
    truthcoin_nodes
        .voter_5
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_b_id.clone(),
            outcome_index: 2,
            shares_amount: 30000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;

    sleep(std::time::Duration::from_secs(5)).await;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    // Sync and refresh wallets
    let issuer_tip = truthcoin_nodes
        .issuer
        .rpc_client
        .get_best_sidechain_block_hash()
        .await?
        .expect("Issuer should have a tip");
    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
    ] {
        let _sync_result = voter.rpc_client.sync_to_tip(issuer_tip).await;
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    // Test 2: Verify positions before selling
    let voter_3_addresses = truthcoin_nodes
        .voter_3
        .rpc_client
        .get_wallet_addresses()
        .await?;
    let mut voter_3_shares_outcome_0: i64 = 0;
    let mut voter_3_seller_address: Option<Address> = None;
    for addr in &voter_3_addresses {
        let positions = truthcoin_nodes
            .voter_3
            .rpc_client
            .market_positions(*addr, Some(market_a_id.clone()))
            .await?;
        for pos in &positions.positions {
            if pos.outcome_index == 0 && pos.shares > 0 {
                voter_3_shares_outcome_0 += pos.shares;
                if voter_3_seller_address.is_none() {
                    voter_3_seller_address = Some(*addr);
                }
            }
        }
    }
    let voter_3_seller_address =
        voter_3_seller_address.expect("voter_3 should have shares");
    anyhow::ensure!(
        voter_3_shares_outcome_0 >= 25000,
        "voter_3 should have at least 25000 shares from Test 1 buy"
    );

    let voter_4_addresses = truthcoin_nodes
        .voter_4
        .rpc_client
        .get_wallet_addresses()
        .await?;
    let mut voter_4_shares_outcome_1: i64 = 0;
    for addr in &voter_4_addresses {
        let positions = truthcoin_nodes
            .voter_4
            .rpc_client
            .market_positions(*addr, Some(market_a_id.clone()))
            .await?;
        for pos in &positions.positions {
            if pos.outcome_index == 1 && pos.shares > 0 {
                voter_4_shares_outcome_1 += pos.shares;
            }
        }
    }
    anyhow::ensure!(
        voter_4_shares_outcome_1 >= 20000,
        "voter_4 should have at least 20000 shares from Test 1 buy"
    );

    // Test 3: Partial sell with balance verification
    let pre_sell_shares = voter_3_shares_outcome_0;
    let shares_to_sell: i64 = 12000;

    let sell_response = truthcoin_nodes
        .voter_3
        .rpc_client
        .market_sell(MarketSellRequest {
            market_id: market_a_id.clone(),
            outcome_index: 0,
            shares_amount: shares_to_sell,
            seller_address: voter_3_seller_address,
            min_proceeds: Some(0), // No slippage protection for test
            dry_run: None,
        })
        .await?;

    anyhow::ensure!(
        sell_response.txid.is_some(),
        "Sell transaction should have a txid"
    );
    anyhow::ensure!(
        sell_response.net_proceeds_sats > 0,
        "Should receive proceeds from sell"
    );

    sleep(std::time::Duration::from_secs(2)).await;

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(2)).await;

    // Sync and refresh wallets
    let issuer_tip = truthcoin_nodes
        .issuer
        .rpc_client
        .get_best_sidechain_block_hash()
        .await?
        .expect("Issuer should have a tip");
    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
    ] {
        let _sync_result = voter.rpc_client.sync_to_tip(issuer_tip).await;
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_secs(1)).await;

    // Verify post-sell balance decreased correctly
    let voter_3_addresses_post = truthcoin_nodes
        .voter_3
        .rpc_client
        .get_wallet_addresses()
        .await?;
    let mut post_sell_shares: i64 = 0;
    for addr in &voter_3_addresses_post {
        let positions = truthcoin_nodes
            .voter_3
            .rpc_client
            .market_positions(*addr, Some(market_a_id.clone()))
            .await?;
        for pos in &positions.positions {
            if pos.outcome_index == 0 {
                post_sell_shares += pos.shares;
            }
        }
    }

    let expected_remaining = pre_sell_shares - shares_to_sell;
    anyhow::ensure!(
        (post_sell_shares - expected_remaining).abs() < 1,
        "Post-sell balance {} should be approximately {} (pre: {} - sold: {})",
        post_sell_shares,
        expected_remaining,
        pre_sell_shares,
        shares_to_sell
    );

    // Test 4: Buy and Sell in same block
    truthcoin_nodes
        .voter_3
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_a_id.clone(),
            outcome_index: 1,
            shares_amount: 15000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;

    // voter_1 sells some shares from Market B (they bought 40000 outcome 0 earlier in initial trades)
    let voter_1_addresses = truthcoin_nodes
        .voter_1
        .rpc_client
        .get_wallet_addresses()
        .await?;
    let mut voter_1_market_b_shares: i64 = 0;
    let mut voter_1_market_b_seller: Option<Address> = None;
    for addr in &voter_1_addresses {
        let positions = truthcoin_nodes
            .voter_1
            .rpc_client
            .market_positions(*addr, Some(market_b_id.clone()))
            .await?;
        for pos in &positions.positions {
            if pos.outcome_index == 0 && pos.shares > 0 {
                voter_1_market_b_shares += pos.shares;
                if voter_1_market_b_seller.is_none() {
                    voter_1_market_b_seller = Some(*addr);
                }
            }
        }
    }
    let voter_1_market_b_seller =
        voter_1_market_b_seller.expect("voter_1 should have Market B shares");
    anyhow::ensure!(
        voter_1_market_b_shares >= 40000,
        "voter_1 should own at least 40000 shares of Market B outcome 0"
    );

    let voter_1_sell_amount: i64 = 20000;
    truthcoin_nodes
        .voter_1
        .rpc_client
        .market_sell(MarketSellRequest {
            market_id: market_b_id.clone(),
            outcome_index: 0,
            shares_amount: voter_1_sell_amount,
            seller_address: voter_1_market_b_seller,
            min_proceeds: Some(0),
            dry_run: None,
        })
        .await?;

    // Mine both transactions in same block
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    // Sync and refresh wallets
    let issuer_tip = truthcoin_nodes
        .issuer
        .rpc_client
        .get_best_sidechain_block_hash()
        .await?
        .expect("Issuer should have a tip");
    for voter in [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
        &truthcoin_nodes.voter_4,
    ] {
        let _sync_result = voter.rpc_client.sync_to_tip(issuer_tip).await;
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    // Test 5: Multiple users buy and sell same outcome
    truthcoin_nodes
        .voter_4
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_a_id.clone(),
            outcome_index: 0,
            shares_amount: 100000,
            max_cost: Some(100_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;

    truthcoin_nodes
        .voter_5
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_a_id.clone(),
            outcome_index: 0,
            shares_amount: 120000,
            max_cost: Some(100_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;

    truthcoin_nodes
        .voter_6
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_a_id.clone(),
            outcome_index: 0,
            shares_amount: 80000,
            max_cost: Some(100_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    // Sync and refresh wallets
    let issuer_tip = truthcoin_nodes
        .issuer
        .rpc_client
        .get_best_sidechain_block_hash()
        .await?
        .expect("Issuer should have a tip");
    for voter in [&truthcoin_nodes.voter_4, &truthcoin_nodes.voter_5] {
        let _sync_result = voter.rpc_client.sync_to_tip(issuer_tip).await;
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    // Check voter_4's position before sell
    let voter_4_addrs = truthcoin_nodes
        .voter_4
        .rpc_client
        .get_wallet_addresses()
        .await?;
    let mut voter_4_pre_sell: i64 = 0;
    let mut voter_4_test5_seller: Option<Address> = None;
    for addr in &voter_4_addrs {
        let positions = truthcoin_nodes
            .voter_4
            .rpc_client
            .market_positions(*addr, Some(market_a_id.clone()))
            .await?;
        for pos in &positions.positions {
            if pos.outcome_index == 0 && pos.shares > 0 {
                voter_4_pre_sell += pos.shares;
                if voter_4_test5_seller.is_none() {
                    voter_4_test5_seller = Some(*addr);
                }
            }
        }
    }
    let voter_4_test5_seller =
        voter_4_test5_seller.expect("voter_4 should have outcome 0 shares");
    let voter_4_sell_amt: i64 = 50000;

    truthcoin_nodes
        .voter_4
        .rpc_client
        .market_sell(MarketSellRequest {
            market_id: market_a_id.clone(),
            outcome_index: 0,
            shares_amount: voter_4_sell_amt,
            seller_address: voter_4_test5_seller,
            min_proceeds: Some(0),
            dry_run: None,
        })
        .await?;

    sleep(std::time::Duration::from_secs(2)).await;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(2)).await;

    // Sync and refresh
    let issuer_tip = truthcoin_nodes
        .issuer
        .rpc_client
        .get_best_sidechain_block_hash()
        .await?
        .expect("Issuer should have a tip");
    let _sync_result = truthcoin_nodes
        .voter_4
        .rpc_client
        .sync_to_tip(issuer_tip)
        .await;
    truthcoin_nodes.voter_4.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_secs(1)).await;

    // Verify voter_4 post-sell balance
    let mut voter_4_post_sell: i64 = 0;
    for addr in &voter_4_addrs {
        let positions = truthcoin_nodes
            .voter_4
            .rpc_client
            .market_positions(*addr, Some(market_a_id.clone()))
            .await?;
        for pos in &positions.positions {
            if pos.outcome_index == 0 {
                voter_4_post_sell += pos.shares;
            }
        }
    }
    let voter_4_expected = voter_4_pre_sell - voter_4_sell_amt;
    anyhow::ensure!(
        (voter_4_post_sell - voter_4_expected).abs() < 1,
        "voter_4 post-sell balance {} should be ~{}",
        voter_4_post_sell,
        voter_4_expected
    );

    // Test 6: Dry run sell (no execution)
    let dry_run_result = truthcoin_nodes
        .voter_3
        .rpc_client
        .market_sell(MarketSellRequest {
            market_id: market_a_id.clone(),
            outcome_index: 0,
            shares_amount: 10000,
            seller_address: voter_3_seller_address,
            min_proceeds: Some(0),
            dry_run: Some(true),
        })
        .await?;

    anyhow::ensure!(
        dry_run_result.txid.is_none(),
        "Dry run should not produce a txid"
    );
    anyhow::ensure!(
        dry_run_result.net_proceeds_sats > 0,
        "Dry run should calculate proceeds"
    );

    // Refresh wallets
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

    // Test 8: Single-address sell
    truthcoin_nodes.voter_6.rpc_client.refresh_wallet().await?;
    truthcoin_nodes
        .voter_6
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_a_id.clone(),
            outcome_index: 1,
            shares_amount: 100000,
            max_cost: Some(100_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;

    // Wait for P2P propagation before mining
    sleep(std::time::Duration::from_secs(5)).await;

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;
    truthcoin_nodes.voter_6.rpc_client.refresh_wallet().await?;

    // Find the address that holds the shares
    let voter_6_addresses = truthcoin_nodes
        .voter_6
        .rpc_client
        .get_wallet_addresses()
        .await?;
    let mut seller_address: Option<Address> = None;
    let mut shares_at_address: i64 = 0;

    for addr in &voter_6_addresses {
        let positions = truthcoin_nodes
            .voter_6
            .rpc_client
            .market_positions(*addr, Some(market_a_id.clone()))
            .await?;
        for pos in &positions.positions {
            if pos.outcome_index == 1 && pos.shares > 0 {
                seller_address = Some(*addr);
                shares_at_address = pos.shares;
                break;
            }
        }
        if seller_address.is_some() {
            break;
        }
    }

    let seller_address = seller_address.expect("voter_6 should have shares");
    anyhow::ensure!(
        shares_at_address >= 15000,
        "voter_6 should have at least 15000 shares"
    );

    let sell_amount: i64 = 50000;
    let sell_response = truthcoin_nodes
        .voter_6
        .rpc_client
        .market_sell(MarketSellRequest {
            market_id: market_a_id.clone(),
            outcome_index: 1,
            shares_amount: sell_amount,
            seller_address,
            min_proceeds: Some(0),
            dry_run: None,
        })
        .await?;

    anyhow::ensure!(sell_response.txid.is_some(), "Sell should produce a txid");
    anyhow::ensure!(
        sell_response.net_proceeds_sats > 0,
        "Sell should have positive proceeds"
    );

    // Wait for P2P propagation before mining
    sleep(std::time::Duration::from_secs(2)).await;

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(2)).await;

    // Sync and refresh
    let issuer_tip = truthcoin_nodes
        .issuer
        .rpc_client
        .get_best_sidechain_block_hash()
        .await?
        .expect("Issuer should have a tip");
    let _sync_result = truthcoin_nodes
        .voter_6
        .rpc_client
        .sync_to_tip(issuer_tip)
        .await;
    truthcoin_nodes.voter_6.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_secs(1)).await;

    // Verify post-sell balance
    let post_positions = truthcoin_nodes
        .voter_6
        .rpc_client
        .market_positions(seller_address, Some(market_a_id.clone()))
        .await?;
    let voter_6_post_sell = post_positions
        .positions
        .iter()
        .find(|p| p.outcome_index == 1)
        .map(|p| p.shares)
        .unwrap_or(0);

    let expected_remaining = shares_at_address - sell_amount;
    anyhow::ensure!(
        (voter_6_post_sell - expected_remaining).abs() < 1,
        "voter_6 post-sell balance {} should be ~{} (had {} - sold {})",
        voter_6_post_sell,
        expected_remaining,
        shares_at_address,
        sell_amount
    );

    // Refresh all wallets before final verification
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

    // Trade on Market D (2x2)
    for voter in [&truthcoin_nodes.voter_3, &truthcoin_nodes.voter_4] {
        voter.rpc_client.refresh_wallet().await?;
    }
    truthcoin_nodes
        .voter_3
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_d_id.clone(),
            outcome_index: 0,
            shares_amount: 25000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;
    truthcoin_nodes
        .voter_4
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_d_id.clone(),
            outcome_index: 3,
            shares_amount: 20000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    // Trade on Market F (256 outcomes)
    for voter in [&truthcoin_nodes.voter_5, &truthcoin_nodes.voter_6] {
        voter.rpc_client.refresh_wallet().await?;
    }
    for (voter, outcome_idx) in [
        (&truthcoin_nodes.voter_5, 0usize),
        (&truthcoin_nodes.voter_6, 255usize),
    ] {
        voter
            .rpc_client
            .market_buy(MarketBuyRequest {
                market_id: market_f_id.clone(),
                outcome_index: outcome_idx,
                shares_amount: 5000,
                max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
                dry_run: None,
            })
            .await?;
    }
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    // Trade on Market G
    truthcoin_nodes.voter_3.rpc_client.refresh_wallet().await?;
    truthcoin_nodes
        .voter_3
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_g_id.clone(),
            outcome_index: 0,
            shares_amount: 20000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    // Trade on Market H
    truthcoin_nodes.voter_4.rpc_client.refresh_wallet().await?;
    truthcoin_nodes
        .voter_4
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_h_id.clone(),
            outcome_index: 3,
            shares_amount: 20000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    // Trade on Market I
    truthcoin_nodes.voter_5.rpc_client.refresh_wallet().await?;
    truthcoin_nodes
        .voter_5
        .rpc_client
        .market_buy(MarketBuyRequest {
            market_id: market_i_id.clone(),
            outcome_index: 3,
            shares_amount: 20000,
            max_cost: Some(10_000_000), // Large buffer to avoid slippage failures
            dry_run: None,
        })
        .await?;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(1)).await;

    // Verify LMSR invariants after trading
    for (market_id, market_name) in [
        (&market_a_id, "A"),
        (&market_b_id, "B"),
        (&market_d_id, "D"),
        (&market_f_id, "F"),
        (&market_g_id, "G"),
        (&market_h_id, "H"),
        (&market_i_id, "I"),
    ] {
        let market_data = truthcoin_nodes
            .issuer
            .rpc_client
            .market_get(market_id.clone())
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!("Market {} not found", market_name)
            })?;

        let price_sum: f64 =
            market_data.outcomes.iter().map(|o| o.current_price).sum();

        anyhow::ensure!(
            (price_sum - 1.0).abs() < expected::PRICE_SUM_TOLERANCE,
            "LMSR invariant violated for market {}: prices sum to {}",
            market_name,
            price_sum
        );
    }

    tracing::info!("✓ Phase 11: Trading completed");

    // Advance to voting period and submit votes

    // Advance to voting period for our test period (need to mine blocks)
    // Periods are 10 blocks each: Period N voting happens in blocks N*10 to N*10+9
    let current_height =
        truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    let voting_period_start = (test_period * 10) + 1;
    let blocks_needed = voting_period_start.saturating_sub(current_height);

    for _ in 0..blocks_needed {
        truthcoin_nodes
            .issuer
            .bmm_single(&mut enforcer_post_setup)
            .await?;
        sleep(std::time::Duration::from_millis(100)).await;
    }

    sleep(std::time::Duration::from_secs(2)).await;

    // Verify slots are in voting state
    let voting_slots = truthcoin_nodes
        .issuer
        .rpc_client
        .slot_list(Some(SlotFilter {
            period: Some(test_period),
            status: Some(SlotState::Voting),
        }))
        .await?;

    anyhow::ensure!(
        !voting_slots.is_empty(),
        "Expected slots in voting state for period {}",
        test_period
    );

    // Prepare vote values
    let btc_min = expected_phase2::scaled::BTC_PRICE_MIN as f64;
    let btc_max = expected_phase2::scaled::BTC_PRICE_MAX as f64;
    let btc_range = btc_max - btc_min;

    // ETH/BTC ratio range: 0 to 100 (representing 0.00 to 0.10)
    let eth_btc_min = 0.0_f64;
    let eth_btc_max = 100.0_f64;
    let eth_btc_range = eth_btc_max - eth_btc_min;

    let vote_matrix: Vec<Vec<(String, f64)>> = vec![
        // Voter 0: BTC $152,500, ETH/BTC 0.05, Candidate A, Ethereum, all binary YES except 34-37
        vec![
            (scaled_slot_0.clone(), btc_min + 0.75 * btc_range),
            (scaled_slot_1.clone(), eth_btc_min + 0.50 * eth_btc_range), // ETH/BTC = 0.05
            (cat_slot_10.clone(), 1.0), // Candidate A
            (cat_slot_11.clone(), 0.0),
            (cat_slot_12.clone(), 0.0),
            (cat_slot_20.clone(), 1.0), // Ethereum
            (cat_slot_21.clone(), 0.0),
            (cat_slot_22.clone(), 0.0),
            (cat_slot_23.clone(), 0.0),
            (bin_slot_30.clone(), 1.0), // Inflation > 3%: YES
            (bin_slot_31.clone(), 1.0), // Fed cuts: YES
            (bin_slot_32.clone(), 1.0), // Unemployment: YES
            (bin_slot_33.clone(), 1.0), // GDP growth: YES
            (bin_slot_34.clone(), 1.0), // Housing: YES
            (bin_slot_35.clone(), 1.0), // Consumer conf: YES
            (bin_slot_36.clone(), 1.0), // Retail: YES
            (bin_slot_37.clone(), 1.0), // Manufacturing: YES
        ],
        // Voter 1: BTC $162,000, ETH/BTC 0.06, Candidate A, Ethereum
        vec![
            (scaled_slot_0.clone(), btc_min + 0.80 * btc_range),
            (scaled_slot_1.clone(), eth_btc_min + 0.60 * eth_btc_range),
            (cat_slot_10.clone(), 1.0),
            (cat_slot_11.clone(), 0.0),
            (cat_slot_12.clone(), 0.0),
            (cat_slot_20.clone(), 1.0),
            (cat_slot_21.clone(), 0.0),
            (cat_slot_22.clone(), 0.0),
            (cat_slot_23.clone(), 0.0),
            (bin_slot_30.clone(), 1.0),
            (bin_slot_31.clone(), 1.0),
            (bin_slot_32.clone(), 1.0),
            (bin_slot_33.clone(), 1.0),
            (bin_slot_34.clone(), 1.0),
            (bin_slot_35.clone(), 1.0),
            (bin_slot_36.clone(), 1.0),
            (bin_slot_37.clone(), 1.0),
        ],
        // Voter 2: BTC $143,000, ETH/BTC 0.04, Candidate A, Ethereum
        vec![
            (scaled_slot_0.clone(), btc_min + 0.70 * btc_range),
            (scaled_slot_1.clone(), eth_btc_min + 0.40 * eth_btc_range),
            (cat_slot_10.clone(), 1.0),
            (cat_slot_11.clone(), 0.0),
            (cat_slot_12.clone(), 0.0),
            (cat_slot_20.clone(), 1.0),
            (cat_slot_21.clone(), 0.0),
            (cat_slot_22.clone(), 0.0),
            (cat_slot_23.clone(), 0.0),
            (bin_slot_30.clone(), 1.0),
            (bin_slot_31.clone(), 1.0),
            (bin_slot_32.clone(), 1.0),
            (bin_slot_33.clone(), 1.0),
            (bin_slot_34.clone(), 1.0),
            (bin_slot_35.clone(), 1.0),
            (bin_slot_36.clone(), 1.0),
            (bin_slot_37.clone(), 1.0),
        ],
        // Voter 3: BTC $152,500, ETH/BTC 0.05, Candidate B (minority), Solana (minority), mixed binary
        vec![
            (scaled_slot_0.clone(), btc_min + 0.75 * btc_range),
            (scaled_slot_1.clone(), eth_btc_min + 0.50 * eth_btc_range),
            (cat_slot_10.clone(), 0.0),
            (cat_slot_11.clone(), 1.0), // Candidate B (minority)
            (cat_slot_12.clone(), 0.0),
            (cat_slot_20.clone(), 0.0),
            (cat_slot_21.clone(), 1.0), // Solana (minority)
            (cat_slot_22.clone(), 0.0),
            (cat_slot_23.clone(), 0.0),
            (bin_slot_30.clone(), 1.0),
            (bin_slot_31.clone(), 1.0),
            (bin_slot_32.clone(), 0.0), // NO
            (bin_slot_33.clone(), 1.0),
            (bin_slot_34.clone(), 0.0), // NO
            (bin_slot_35.clone(), 0.0), // NO
            (bin_slot_36.clone(), 0.0), // NO
            (bin_slot_37.clone(), 0.0), // NO
        ],
        // Voter 4: BTC $158,200, ETH/BTC 0.055, Candidate A, Ethereum
        vec![
            (scaled_slot_0.clone(), btc_min + 0.78 * btc_range),
            (scaled_slot_1.clone(), eth_btc_min + 0.55 * eth_btc_range),
            (cat_slot_10.clone(), 1.0),
            (cat_slot_11.clone(), 0.0),
            (cat_slot_12.clone(), 0.0),
            (cat_slot_20.clone(), 1.0),
            (cat_slot_21.clone(), 0.0),
            (cat_slot_22.clone(), 0.0),
            (cat_slot_23.clone(), 0.0),
            (bin_slot_30.clone(), 1.0),
            (bin_slot_31.clone(), 1.0),
            (bin_slot_32.clone(), 0.0), // NO
            (bin_slot_33.clone(), 1.0),
            (bin_slot_34.clone(), 0.0), // NO
            (bin_slot_35.clone(), 0.0), // NO
            (bin_slot_36.clone(), 0.0), // NO
            (bin_slot_37.clone(), 0.0), // NO
        ],
        // Voter 5: BTC $154,400, ETH/BTC 0.052, Candidate A, Ethereum
        vec![
            (scaled_slot_0.clone(), btc_min + 0.76 * btc_range),
            (scaled_slot_1.clone(), eth_btc_min + 0.52 * eth_btc_range),
            (cat_slot_10.clone(), 1.0),
            (cat_slot_11.clone(), 0.0),
            (cat_slot_12.clone(), 0.0),
            (cat_slot_20.clone(), 1.0),
            (cat_slot_21.clone(), 0.0),
            (cat_slot_22.clone(), 0.0),
            (cat_slot_23.clone(), 0.0),
            (bin_slot_30.clone(), 1.0),
            (bin_slot_31.clone(), 1.0),
            (bin_slot_32.clone(), 0.0), // NO
            (bin_slot_33.clone(), 0.0), // NO (minority on this one)
            (bin_slot_34.clone(), 0.0), // NO
            (bin_slot_35.clone(), 0.0), // NO
            (bin_slot_36.clone(), 0.0), // NO
            (bin_slot_37.clone(), 0.0), // NO
        ],
        // Voter 6: BTC $150,600, ETH/BTC 0.048, Candidate A, Ethereum
        vec![
            (scaled_slot_0.clone(), btc_min + 0.74 * btc_range),
            (scaled_slot_1.clone(), eth_btc_min + 0.48 * eth_btc_range),
            (cat_slot_10.clone(), 1.0),
            (cat_slot_11.clone(), 0.0),
            (cat_slot_12.clone(), 0.0),
            (cat_slot_20.clone(), 1.0),
            (cat_slot_21.clone(), 0.0),
            (cat_slot_22.clone(), 0.0),
            (cat_slot_23.clone(), 0.0),
            (bin_slot_30.clone(), 1.0),
            (bin_slot_31.clone(), 1.0),
            (bin_slot_32.clone(), 0.0), // NO
            (bin_slot_33.clone(), 0.0), // NO
            (bin_slot_34.clone(), 0.0), // NO
            (bin_slot_35.clone(), 0.0), // NO
            (bin_slot_36.clone(), 0.0), // NO
            (bin_slot_37.clone(), 0.0), // NO
        ],
    ];

    // Submit votes
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

        let vote_items: Vec<VoteBatchItem> = votes
            .iter()
            .map(|(slot_id, value)| VoteBatchItem {
                decision_id: slot_id.clone(),
                vote_value: *value,
            })
            .collect();

        voter.rpc_client.vote_submit(vote_items, 1000).await?;
    }

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(2)).await;

    tracing::info!("✓ Phase 12: Voting completed");

    // Close voting period
    let blocks_to_close = 10;
    for _ in 0..blocks_to_close {
        truthcoin_nodes
            .issuer
            .bmm_single(&mut enforcer_post_setup)
            .await?;
        sleep(std::time::Duration::from_millis(100)).await;
    }

    // Trigger resolution
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    sleep(std::time::Duration::from_secs(2)).await;

    // Verify consensus results
    let test_period_id = test_period;
    let period_info = truthcoin_nodes
        .issuer
        .rpc_client
        .vote_period(Some(test_period_id))
        .await?;

    if let Some(info) = &period_info
        && let Some(consensus) = &info.consensus
    {
        // Verify scaled decisions
        if let Some(&outcome) = consensus.outcomes.get(&scaled_slot_0) {
            debug_helpers::assert_float_eq(
                outcome,
                expected_phase2::scaled::EXPECTED_BTC_CONSENSUS,
                expected_phase2::FLOAT_TOLERANCE,
                "Scaled BTC consensus",
            )?;
        }

        if let Some(&outcome) = consensus.outcomes.get(&scaled_slot_1) {
            debug_helpers::assert_float_eq(
                outcome,
                expected_phase2::scaled::EXPECTED_ETH_BTC_CONSENSUS,
                expected_phase2::FLOAT_TOLERANCE,
                "Scaled ETH/BTC consensus",
            )?;
        }

        // Verify categorical decisions
        let cat_expectations = [
            (
                &cat_slot_10,
                expected_phase2::categorical::EXPECTED_CAT_10,
                "Candidate A",
            ),
            (
                &cat_slot_11,
                expected_phase2::categorical::EXPECTED_CAT_11,
                "Candidate B",
            ),
            (
                &cat_slot_12,
                expected_phase2::categorical::EXPECTED_CAT_12,
                "Candidate C",
            ),
            (
                &cat_slot_20,
                expected_phase2::categorical::EXPECTED_CAT_20,
                "Ethereum",
            ),
            (
                &cat_slot_21,
                expected_phase2::categorical::EXPECTED_CAT_21,
                "Solana",
            ),
            (
                &cat_slot_22,
                expected_phase2::categorical::EXPECTED_CAT_22,
                "Arbitrum",
            ),
            (
                &cat_slot_23,
                expected_phase2::categorical::EXPECTED_CAT_23,
                "Other",
            ),
        ];

        for (slot, expected, name) in cat_expectations {
            if let Some(&outcome) = consensus.outcomes.get(slot) {
                debug_helpers::assert_float_eq(
                    outcome,
                    expected,
                    expected_phase2::FLOAT_TOLERANCE,
                    &format!("Categorical {name} consensus"),
                )?;
            }
        }

        // Verify binary decisions
        let bin_expectations = [
            (
                &bin_slot_30,
                expected_phase2::binary::EXPECTED_BIN_30,
                "Inflation>3%",
            ),
            (
                &bin_slot_31,
                expected_phase2::binary::EXPECTED_BIN_31,
                "Fed cuts",
            ),
            (
                &bin_slot_32,
                expected_phase2::binary::EXPECTED_BIN_32,
                "Unemployment",
            ),
            (
                &bin_slot_33,
                expected_phase2::binary::EXPECTED_BIN_33,
                "GDP growth",
            ),
            (
                &bin_slot_34,
                expected_phase2::binary::EXPECTED_BIN_34,
                "Housing",
            ),
            (
                &bin_slot_35,
                expected_phase2::binary::EXPECTED_BIN_35,
                "Consumer conf",
            ),
            (
                &bin_slot_36,
                expected_phase2::binary::EXPECTED_BIN_36,
                "Retail",
            ),
            (
                &bin_slot_37,
                expected_phase2::binary::EXPECTED_BIN_37,
                "Manufacturing",
            ),
        ];

        for (slot, expected, name) in bin_expectations {
            if let Some(&outcome) = consensus.outcomes.get(slot) {
                debug_helpers::assert_float_eq(
                    outcome,
                    expected,
                    expected_phase2::FLOAT_TOLERANCE,
                    &format!("Binary {name} consensus"),
                )?;
            }
        }
    }

    // Verify markets ossified
    let market_a_data = truthcoin_nodes
        .issuer
        .rpc_client
        .market_get(market_a_id.clone())
        .await?
        .ok_or_else(|| anyhow::anyhow!("Market A not found"))?;

    debug_helpers::assert_outcome_count(
        &market_a_data,
        expected_phase2::markets::MARKET_A_OUTCOMES,
        "Market A (Scaled)",
    )?;
    debug_helpers::assert_lmsr_invariant(&market_a_data, "Market A (Scaled)")?;

    // Market B: 3-way categorical (3 outcomes)
    let market_b_data = truthcoin_nodes
        .issuer
        .rpc_client
        .market_get(market_b_id.clone())
        .await?
        .ok_or_else(|| anyhow::anyhow!("Market B not found"))?;

    debug_helpers::assert_outcome_count(
        &market_b_data,
        expected_phase2::markets::MARKET_B_OUTCOMES,
        "Market B (3-way Cat)",
    )?;
    debug_helpers::assert_lmsr_invariant(
        &market_b_data,
        "Market B (3-way Cat)",
    )?;

    // Market C: 4-way categorical (5 outcomes: 4 options + residual)
    let market_c_data = truthcoin_nodes
        .issuer
        .rpc_client
        .market_get(market_c_id.clone())
        .await?
        .ok_or_else(|| anyhow::anyhow!("Market C not found"))?;

    debug_helpers::assert_outcome_count(
        &market_c_data,
        expected_phase2::markets::MARKET_C_OUTCOMES,
        "Market C (4-way Cat)",
    )?;
    debug_helpers::assert_lmsr_invariant(
        &market_c_data,
        "Market C (4-way Cat)",
    )?;

    // Market D: 2×2 binary (4 outcomes)
    let market_d_data = truthcoin_nodes
        .issuer
        .rpc_client
        .market_get(market_d_id.clone())
        .await?
        .ok_or_else(|| anyhow::anyhow!("Market D not found"))?;

    debug_helpers::assert_outcome_count(
        &market_d_data,
        expected_phase2::markets::MARKET_D_OUTCOMES,
        "Market D (2×2 Binary)",
    )?;
    debug_helpers::assert_lmsr_invariant(
        &market_d_data,
        "Market D (2×2 Binary)",
    )?;

    // Market E: 2×2×2 binary (8 outcomes)
    let market_e_data = truthcoin_nodes
        .issuer
        .rpc_client
        .market_get(market_e_id.clone())
        .await?
        .ok_or_else(|| anyhow::anyhow!("Market E not found"))?;

    debug_helpers::assert_outcome_count(
        &market_e_data,
        expected_phase2::markets::MARKET_E_OUTCOMES,
        "Market E (2×2×2 Binary)",
    )?;
    debug_helpers::assert_lmsr_invariant(
        &market_e_data,
        "Market E (2×2×2 Binary)",
    )?;

    // Market F: 8-dimensional binary (256 outcomes)
    let market_f_data = truthcoin_nodes
        .issuer
        .rpc_client
        .market_get(market_f_id.clone())
        .await?
        .ok_or_else(|| anyhow::anyhow!("Market F not found"))?;

    debug_helpers::assert_outcome_count(
        &market_f_data,
        expected_phase2::markets::MARKET_F_OUTCOMES,
        "Market F (8-dim Binary)",
    )?;
    debug_helpers::assert_lmsr_invariant(
        &market_f_data,
        "Market F (8-dim Binary)",
    )?;

    // Market G: Scaled × Binary (4 outcomes)
    let market_g_data = truthcoin_nodes
        .issuer
        .rpc_client
        .market_get(market_g_id.clone())
        .await?
        .ok_or_else(|| anyhow::anyhow!("Market G not found"))?;

    debug_helpers::assert_outcome_count(
        &market_g_data,
        expected_phase2::markets::MARKET_G_OUTCOMES,
        "Market G (Scaled×Binary)",
    )?;
    debug_helpers::assert_lmsr_invariant(
        &market_g_data,
        "Market G (Scaled×Binary)",
    )?;

    // Market H: Scaled × Categorical (8 outcomes)
    let market_h_data = truthcoin_nodes
        .issuer
        .rpc_client
        .market_get(market_h_id.clone())
        .await?
        .ok_or_else(|| anyhow::anyhow!("Market H not found"))?;

    debug_helpers::assert_outcome_count(
        &market_h_data,
        expected_phase2::markets::MARKET_H_OUTCOMES,
        "Market H (Scaled×Cat)",
    )?;
    debug_helpers::assert_lmsr_invariant(
        &market_h_data,
        "Market H (Scaled×Cat)",
    )?;

    // Market I: Scaled × Scaled (4 outcomes)
    let market_i_data = truthcoin_nodes
        .issuer
        .rpc_client
        .market_get(market_i_id.clone())
        .await?
        .ok_or_else(|| anyhow::anyhow!("Market I not found"))?;

    debug_helpers::assert_outcome_count(
        &market_i_data,
        expected_phase2::markets::MARKET_I_OUTCOMES,
        "Market I (Scaled×Scaled)",
    )?;
    debug_helpers::assert_lmsr_invariant(
        &market_i_data,
        "Market I (Scaled×Scaled)",
    )?;

    // Market J: Scaled × Binary × Binary × Categorical (32 outcomes)
    let market_j_data = truthcoin_nodes
        .issuer
        .rpc_client
        .market_get(market_j_id.clone())
        .await?
        .ok_or_else(|| anyhow::anyhow!("Market J not found"))?;

    debug_helpers::assert_outcome_count(
        &market_j_data,
        expected_phase2::markets::MARKET_J_OUTCOMES,
        "Market J (Ultimate Mixed)",
    )?;
    debug_helpers::assert_lmsr_invariant(
        &market_j_data,
        "Market J (Ultimate Mixed)",
    )?;

    // Verify market states and treasury
    for (market_id, market_data, _label) in [
        (&market_a_id, &market_a_data, "Market A"),
        (&market_b_id, &market_b_data, "Market B"),
        (&market_c_id, &market_c_data, "Market C"),
        (&market_d_id, &market_d_data, "Market D"),
        (&market_e_id, &market_e_data, "Market E"),
        (&market_f_id, &market_f_data, "Market F"),
        (&market_g_id, &market_g_data, "Market G"),
        (&market_h_id, &market_h_data, "Market H"),
        (&market_i_id, &market_i_data, "Market I"),
        (&market_j_id, &market_j_data, "Market J"),
    ] {
        debug_helpers::assert_treasury_zero(market_data, market_id)?;
    }

    tracing::info!("✓ Phase 13: Period resolution completed");

    // Final assertions - VoteCoin conservation
    let mut voter_balances: Vec<(String, u32)> = Vec::new();
    let mut final_votecoin_total = 0u32;

    let issuer_utxos =
        truthcoin_nodes.issuer.rpc_client.get_wallet_utxos().await?;
    let issuer_balance: u32 = issuer_utxos
        .iter()
        .filter_map(|u| u.output.content.votecoin())
        .sum();
    voter_balances.push(("Issuer".to_string(), issuer_balance));
    final_votecoin_total += issuer_balance;

    let voter_nodes = [
        ("Voter_0", &truthcoin_nodes.voter_0),
        ("Voter_1", &truthcoin_nodes.voter_1),
        ("Voter_2", &truthcoin_nodes.voter_2),
        ("Voter_3", &truthcoin_nodes.voter_3),
        ("Voter_4", &truthcoin_nodes.voter_4),
        ("Voter_5", &truthcoin_nodes.voter_5),
        ("Voter_6", &truthcoin_nodes.voter_6),
    ];

    for (name, voter) in voter_nodes {
        voter.rpc_client.refresh_wallet().await?;
        let utxos = voter.rpc_client.get_wallet_utxos().await?;
        let balance: u32 = utxos
            .iter()
            .filter_map(|u| u.output.content.votecoin())
            .sum();
        voter_balances.push((name.to_string(), balance));
        final_votecoin_total += balance;
    }

    debug_helpers::assert_votecoin_conservation(
        final_votecoin_total,
        INITIAL_VOTECOIN_SUPPLY,
        &voter_balances,
    )?;

    // Verify no orphaned market UTXOs
    let final_utxos = truthcoin_nodes.issuer.rpc_client.list_utxos().await?;
    let remaining_treasury =
        utxo_verification::get_market_treasury_utxos(&final_utxos);
    let remaining_fees =
        utxo_verification::get_market_author_fee_utxos(&final_utxos);

    // Check that ossified markets have no remaining UTXOs
    for market_id in [
        &market_a_id,
        &market_b_id,
        &market_c_id,
        &market_d_id,
        &market_e_id,
        &market_f_id,
        &market_g_id,
        &market_h_id,
        &market_i_id,
        &market_j_id,
    ] {
        let market_data = truthcoin_nodes
            .issuer
            .rpc_client
            .market_get(market_id.clone())
            .await?
            .ok_or_else(|| anyhow::anyhow!("Market not found"))?;

        if market_data.state == "Ossified" {
            anyhow::ensure!(
                !remaining_treasury.contains_key(market_id),
                "Ossified market {} should not have remaining treasury UTXO",
                market_id
            );
            anyhow::ensure!(
                !remaining_fees.contains_key(market_id),
                "Ossified market {} should not have remaining fee UTXO",
                market_id
            );
        }
    }

    tracing::info!("✓ All phases completed successfully");

    {
        drop(truthcoin_nodes);
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
