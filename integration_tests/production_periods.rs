//! Run the node as a downloaded binary does, without the testing flag, so
//! its periods are UTC calendar quarters. Bitcoind's mock clock moves the
//! mainchain across each quarter boundary, taking one decision from claim to
//! resolution.

use std::time::{SystemTime, UNIX_EPOCH};

use bip300301_enforcer_integration_tests::{
    integration_test::{
        activate_sidechain, deposit, fund_enforcer, propose_sidechain,
    },
    setup::{
        Mode, Network, PostSetup as EnforcerPostSetup,
        PreSetup as EnforcerPreSetup, SetupOpts as EnforcerSetupOpts,
        Sidechain as _,
    },
    util::{
        AbortOnDrop, AsyncTrial, BinPaths as EnforcerBinPaths,
        TestFailureCollector, TestFileRegistry,
    },
};
use bitcoin::Amount;
use futures::{
    FutureExt as _, StreamExt as _, channel::mpsc, future::BoxFuture,
};
use jsonrpsee::{core::client::ClientT as _, rpc_params};
use tokio::time::sleep;
use tracing::Instrument as _;
use truthcoin_dc_app_rpc_api::{
    markets::{
        BallotItem, DecisionClaimItem, DecisionClaimRequest, DecisionFilter,
        DecisionState,
    },
    node::{PrivateRpcClient as _, RpcClient as _},
    wallet::RpcClient as _,
};

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

const DEPOSIT_AMOUNT: Amount = Amount::from_sat(21_000_000);
const DEPOSIT_FEE: Amount = Amount::from_sat(1_000_000);

/// 00:00 UTC on the first day of each quarter, Q4 2026 to Q4 2029
const QUARTER_STARTS: [u64; 13] = [
    1_790_812_800,
    1_798_761_600,
    1_806_537_600,
    1_814_400_000,
    1_822_348_800,
    1_830_297_600,
    1_838_160_000,
    1_846_022_400,
    1_853_971_200,
    1_861_920_000,
    1_869_696_000,
    1_877_558_400,
    1_885_507_200,
];

const REVEAL_WINDOW_SECONDS: u64 = 7 * 24 * 3600;

/// Initial setup for the test
async fn setup(
    enforcer_bin_paths: &EnforcerBinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<EnforcerPostSetup> {
    let enforcer_pre_setup =
        EnforcerPreSetup::new(enforcer_bin_paths, Network::Regtest)?;
    let mut enforcer_post_setup = {
        let setup_opts: EnforcerSetupOpts = Default::default();
        enforcer_pre_setup
            .setup(Mode::Mempool, setup_opts, res_tx.clone())
            .await?
    };
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    Ok(enforcer_post_setup)
}

/// BMM one sidechain block, checking that it connected
async fn bmm_one(
    enforcer_post_setup: &mut EnforcerPostSetup,
    sidechain: &PostSetup,
) -> anyhow::Result<()> {
    let height = sidechain.rpc_client.getblockcount().await?;
    sidechain.bmm_single(enforcer_post_setup).await?;
    sleep(std::time::Duration::from_millis(500)).await;
    anyhow::ensure!(
        sidechain.rpc_client.getblockcount().await? > height,
        "BMM did not connect a sidechain block"
    );
    Ok(())
}

/// Mine a mainchain block at `timestamp`, give the enforcer time to make it
/// its tip, then BMM a sidechain block on it, which the enforcer stamps one
/// second after that tip
async fn bmm_at(
    enforcer_post_setup: &mut EnforcerPostSetup,
    sidechain: &PostSetup,
    timestamp: u64,
) -> anyhow::Result<()> {
    let bitcoind = &enforcer_post_setup.bitcoind_client;
    let () = bitcoind
        .request("setmocktime", rpc_params![timestamp])
        .await?;
    let _block_hashes: Vec<String> = bitcoind
        .request("generatetodescriptor", rpc_params![1, "raw(51)"])
        .await?;
    sleep(std::time::Duration::from_secs(2)).await;
    bmm_one(enforcer_post_setup, sidechain).await
}

async fn decision_state(
    sidechain: &PostSetup,
    claim_period: u32,
    state: DecisionState,
) -> anyhow::Result<bool> {
    Ok(!sidechain
        .rpc_client
        .decision_list(Some(DecisionFilter {
            period: Some(claim_period),
            status: Some(state),
        }))
        .await?
        .is_empty())
}

async fn production_periods_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let mut enforcer_post_setup =
        setup(&bin_paths.others, res_tx.clone()).await?;
    let mut sidechain = PostSetup::setup(
        Init {
            decision_config_testing: None,
            ..Init::new(bin_paths.truthcoin()?.clone())
        },
        &enforcer_post_setup,
        res_tx.clone(),
    )
    .await?;
    let deposit_address = sidechain.get_deposit_address().await?;
    let () = deposit(
        &mut enforcer_post_setup,
        &mut sidechain,
        &deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;

    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let quarter = QUARTER_STARTS
        .windows(2)
        .position(|bounds| (bounds[0]..bounds[1]).contains(&now))
        .filter(|quarter| quarter + 3 < QUARTER_STARTS.len())
        .ok_or_else(|| anyhow::anyhow!("extend QUARTER_STARTS past {now}"))?;
    let [voting_start, reveal_start] =
        [QUARTER_STARTS[quarter + 1], QUARTER_STARTS[quarter + 2]];
    let reveal_end = reveal_start + REVEAL_WINDOW_SECONDS;
    let quarter_name =
        format!("Q{} Y{}", (quarter + 3) % 4 + 1, 2026 + quarter.div_ceil(4));

    let status = sidechain.rpc_client.decision_status().await?;
    anyhow::ensure!(!status.is_testing_mode);
    anyhow::ensure!(status.current_period == 1);
    anyhow::ensure!(
        status.current_period_name == quarter_name,
        "period 1 is {}, expected {quarter_name}",
        status.current_period_name
    );
    let voting_period = sidechain
        .rpc_client
        .vote_period(Some(2))
        .await?
        .ok_or_else(|| anyhow::anyhow!("no voting period 2"))?;
    anyhow::ensure!(
        (voting_period.start_time, voting_period.end_time)
            == (voting_start, reveal_start)
    );

    let claim = sidechain
        .rpc_client
        .decision_claim(DecisionClaimRequest {
            decision_type: "binary".to_string(),
            decisions: vec![DecisionClaimItem {
                period_index: 1,
                header: "Production quarter decision".to_string(),
                description: None,
                option_0_label: None,
                option_1_label: None,
                option_labels: None,
                tags: None,
            }],
            min: None,
            max: None,
            increment: None,
            fee_rate: None,
            max_listing_fee_sats: None,
        })
        .await?;
    sleep(std::time::Duration::from_millis(500)).await;
    bmm_one(&mut enforcer_post_setup, &sidechain).await?;
    let decision_id = claim
        .decision_ids
        .first()
        .ok_or_else(|| anyhow::anyhow!("claim returned no decision"))?
        .clone();
    anyhow::ensure!(
        decision_state(&sidechain, 1, DecisionState::Claimed).await?
    );

    bmm_at(&mut enforcer_post_setup, &sidechain, voting_start - 60).await?;
    anyhow::ensure!(
        sidechain.rpc_client.decision_status().await?.current_period == 1
    );
    anyhow::ensure!(
        decision_state(&sidechain, 1, DecisionState::Claimed).await?
    );
    bmm_at(&mut enforcer_post_setup, &sidechain, voting_start + 5).await?;
    anyhow::ensure!(
        sidechain.rpc_client.decision_status().await?.current_period == 2
    );
    anyhow::ensure!(
        decision_state(&sidechain, 1, DecisionState::Voting).await?
    );

    sidechain
        .rpc_client
        .vote_submit(
            vec![BallotItem {
                decision_id: decision_id.clone(),
                vote_value: Some(1.0),
            }],
            None,
            None,
        )
        .await?;
    sleep(std::time::Duration::from_millis(500)).await;
    bmm_one(&mut enforcer_post_setup, &sidechain).await?;

    bmm_at(&mut enforcer_post_setup, &sidechain, reveal_start - 60).await?;
    anyhow::ensure!(
        decision_state(&sidechain, 1, DecisionState::Voting).await?
    );
    bmm_at(&mut enforcer_post_setup, &sidechain, reveal_start + 5).await?;
    anyhow::ensure!(
        decision_state(&sidechain, 1, DecisionState::Revealing).await?
    );

    sidechain.rpc_client.vote_reveal(2, None, None).await?;
    sleep(std::time::Duration::from_millis(500)).await;
    bmm_one(&mut enforcer_post_setup, &sidechain).await?;

    bmm_at(&mut enforcer_post_setup, &sidechain, reveal_end - 60).await?;
    anyhow::ensure!(
        decision_state(&sidechain, 1, DecisionState::Revealing).await?
    );
    bmm_at(&mut enforcer_post_setup, &sidechain, reveal_end + 5).await?;
    anyhow::ensure!(
        decision_state(&sidechain, 1, DecisionState::Resolved).await?
    );
    let outcome = sidechain
        .rpc_client
        .vote_period(Some(2))
        .await?
        .and_then(|period| period.consensus)
        .and_then(|consensus| consensus.outcomes.get(&decision_id).copied());
    anyhow::ensure!(outcome == Some(1.0), "outcome was {outcome:?}");

    let syncer = PostSetup::setup(
        Init {
            data_dir_suffix: Some("syncer".to_owned()),
            decision_config_testing: None,
            ..Init::new(bin_paths.truthcoin()?.clone())
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    syncer
        .rpc_client
        .connect_peer(sidechain.net_addr().into())
        .await?;
    let tip = sidechain.rpc_client.get_best_sidechain_block_hash().await?;
    let mut synced = false;
    for _ in 0..60 {
        if syncer.rpc_client.get_best_sidechain_block_hash().await? == tip {
            synced = true;
            break;
        }
        sleep(std::time::Duration::from_millis(500)).await;
    }
    anyhow::ensure!(synced, "the syncer never reached the tip {tip:?}");
    let syncer_status = syncer.rpc_client.decision_status().await?;
    anyhow::ensure!(!syncer_status.is_testing_mode);
    anyhow::ensure!(
        syncer_status.current_period
            == sidechain.rpc_client.decision_status().await?.current_period
    );
    anyhow::ensure!(decision_state(&syncer, 1, DecisionState::Resolved).await?);
    let syncer_outcome = syncer
        .rpc_client
        .vote_period(Some(2))
        .await?
        .and_then(|period| period.consensus)
        .and_then(|consensus| consensus.outcomes.get(&decision_id).copied());
    anyhow::ensure!(
        syncer_outcome == outcome,
        "the syncer resolved {syncer_outcome:?}, the miner {outcome:?}"
    );
    drop(syncer);

    drop(sidechain);
    tracing::info!(
        "Removing {}",
        enforcer_post_setup.directories.base_dir.path().display()
    );
    drop(enforcer_post_setup.tasks);
    sleep(std::time::Duration::from_secs(1)).await;
    enforcer_post_setup.directories.base_dir.cleanup()?;
    Ok(())
}

async fn production_periods(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = production_periods_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn production_periods_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "production_periods",
        production_periods(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
}
