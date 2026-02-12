//! Enhanced 5-node sync test to verify block propagation with transactions
//!
//! This test creates a mesh network of 5 nodes, funds multiple nodes,
//! and performs VoteCoin transfers between them to stress test syncing.

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
use truthcoin_dc::types::Address;
use truthcoin_dc_app_rpc_api::RpcClient as _;

use crate::{
    setup::{Init, PostSetup},
    util::{BinPaths, wait_for_network_sync},
};

#[derive(Debug)]
struct FiveNodes {
    node_0: PostSetup,
    node_1: PostSetup,
    node_2: PostSetup,
    node_3: PostSetup,
    node_4: PostSetup,
}

impl FiveNodes {
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
            node_0: setup_single("node_0").await?,
            node_1: setup_single("node_1").await?,
            node_2: setup_single("node_2").await?,
            node_3: setup_single("node_3").await?,
            node_4: setup_single("node_4").await?,
        };

        // Create mesh topology for robust sync
        // node_0 connects to all others (hub)
        let nodes = [&res.node_1, &res.node_2, &res.node_3, &res.node_4];
        for node in &nodes {
            res.node_0
                .rpc_client
                .connect_peer(node.net_addr().into())
                .await?;
        }

        // Additional cross-connections for mesh topology
        // node_1 <-> node_2, node_3
        res.node_1
            .rpc_client
            .connect_peer(res.node_2.net_addr().into())
            .await?;
        res.node_1
            .rpc_client
            .connect_peer(res.node_3.net_addr().into())
            .await?;

        // node_2 <-> node_4
        res.node_2
            .rpc_client
            .connect_peer(res.node_4.net_addr().into())
            .await?;

        // node_3 <-> node_4
        res.node_3
            .rpc_client
            .connect_peer(res.node_4.net_addr().into())
            .await?;

        // Allow connections to establish
        sleep(std::time::Duration::from_secs(2)).await;

        tracing::info!("Created 5-node mesh network:");
        tracing::info!("  - node_0 (hub) connected to nodes 1, 2, 3, 4");
        tracing::info!("  - node_1 connected to nodes 2, 3");
        tracing::info!("  - node_2 connected to node 4");
        tracing::info!("  - node_3 connected to node 4");
        Ok(res)
    }

    fn all_nodes(&self) -> Vec<&PostSetup> {
        vec![&self.node_0, &self.node_1, &self.node_2, &self.node_3, &self.node_4]
    }
}

const DEPOSIT_AMOUNT: bitcoin::Amount = bitcoin::Amount::from_sat(21_000_000);
const DEPOSIT_FEE: bitcoin::Amount = bitcoin::Amount::from_sat(1_000_000);
const SECONDARY_DEPOSIT_AMOUNT: bitcoin::Amount = bitcoin::Amount::from_sat(5_000_000);
const SECONDARY_DEPOSIT_FEE: bitcoin::Amount = bitcoin::Amount::from_sat(500_000);
/// VoteCoin allocation per node for transfers
const VOTECOIN_ALLOCATION: u32 = 50_000;
/// Minimum transaction fee
const TX_FEE: u64 = 1000;

async fn setup(
    bin_paths: &BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<(EnforcerPostSetup, FiveNodes)> {
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

    let mut nodes = FiveNodes::setup(bin_paths, res_tx, &enforcer_post_setup).await?;

    // Deposit to node_0 (primary node with VoteCoin)
    let deposit_address = nodes.node_0.get_deposit_address().await?;
    let () = deposit(
        &mut enforcer_post_setup,
        &mut nodes.node_0,
        &deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to node_0 (primary) successfully");

    // Deposit to node_1 and node_2 for additional funding
    let deposit_address_1 = nodes.node_1.get_deposit_address().await?;
    let () = deposit(
        &mut enforcer_post_setup,
        &mut nodes.node_1,
        &deposit_address_1,
        SECONDARY_DEPOSIT_AMOUNT,
        SECONDARY_DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to node_1 successfully");

    let deposit_address_2 = nodes.node_2.get_deposit_address().await?;
    let () = deposit(
        &mut enforcer_post_setup,
        &mut nodes.node_2,
        &deposit_address_2,
        SECONDARY_DEPOSIT_AMOUNT,
        SECONDARY_DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to node_2 successfully");

    Ok((enforcer_post_setup, nodes))
}

async fn sync_test_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let (mut enforcer_post_setup, nodes) =
        setup(&bin_paths, res_tx.clone()).await?;

    // Check initial state
    let initial_height = nodes.node_0.rpc_client.getblockcount().await?;
    tracing::info!("Initial height: {}", initial_height);

    // Wait for all nodes to sync to the initial state after deposits
    tracing::info!("Waiting for all nodes to sync to initial height {}...", initial_height);
    wait_for_network_sync(
        initial_height,
        &nodes.all_nodes(),
        200,
        "initial sync after deposits",
    )
    .await?;
    tracing::info!("All nodes synced to initial height!");

    // === Phase 1: Basic block mining and sync ===
    tracing::info!("=== Phase 1: Basic block mining ===");

    // Mine a block
    tracing::info!("Mining block 1...");
    nodes.node_0.bmm_single(&mut enforcer_post_setup).await?;

    let mut current_height = nodes.node_0.rpc_client.getblockcount().await?;
    tracing::info!("Node 0 height after block 1: {}", current_height);

    wait_for_network_sync(
        current_height,
        &nodes.all_nodes(),
        200,
        "block 1",
    )
    .await?;
    tracing::info!("All nodes synced to block 1!");

    // Mine another block
    tracing::info!("Mining block 2...");
    nodes.node_0.bmm_single(&mut enforcer_post_setup).await?;
    current_height = nodes.node_0.rpc_client.getblockcount().await?;

    wait_for_network_sync(
        current_height,
        &nodes.all_nodes(),
        200,
        "block 2",
    )
    .await?;
    tracing::info!("All nodes synced to block 2!");

    // === Phase 2: VoteCoin transfers ===
    tracing::info!("=== Phase 2: VoteCoin transfers ===");

    // First verify that VoteCoin exists in node_0's wallet (created in genesis block)
    let wallet_utxos = nodes.node_0.rpc_client.get_wallet_utxos().await?;
    let total_votecoin: u32 = wallet_utxos
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    tracing::info!("Node 0 wallet VoteCoin balance: {}", total_votecoin);
    anyhow::ensure!(
        total_votecoin > 0,
        "Node 0 has no VoteCoin in wallet - genesis block may not have been mined correctly. \
         Wallet UTXOs: {:?}",
        wallet_utxos.iter().map(|u| &u.output.content).collect::<Vec<_>>()
    );
    tracing::info!("Verified node_0 has {} VoteCoin available", total_votecoin);

    // Get addresses for each node to receive VoteCoin
    let node_1_addr: Address = nodes.node_1.rpc_client.get_new_address().await?;
    let node_2_addr: Address = nodes.node_2.rpc_client.get_new_address().await?;
    let node_3_addr: Address = nodes.node_3.rpc_client.get_new_address().await?;
    let node_4_addr: Address = nodes.node_4.rpc_client.get_new_address().await?;

    // Transfer VoteCoin from node_0 to each node, one at a time
    // Verify balance on receiving node immediately after sync to test state propagation
    let node_transfers: [(Address, &str, &PostSetup); 4] = [
        (node_1_addr, "node_1", &nodes.node_1),
        (node_2_addr, "node_2", &nodes.node_2),
        (node_3_addr, "node_3", &nodes.node_3),
        (node_4_addr, "node_4", &nodes.node_4),
    ];

    for (addr, name, receiving_node) in &node_transfers {
        tracing::info!("Transferring {} VoteCoin to {} at address {}...", VOTECOIN_ALLOCATION, name, addr);

        let transfer_txid = nodes.node_0.rpc_client
            .transfer_votecoin(*addr, VOTECOIN_ALLOCATION, TX_FEE, None)
            .await?;
        tracing::info!("Transfer txid: {} to address {}", transfer_txid, addr);

        let height_before = nodes.node_0.rpc_client.getblockcount().await?;
        nodes.node_0.bmm_single(&mut enforcer_post_setup).await?;
        current_height = nodes.node_0.rpc_client.getblockcount().await?;
        tracing::info!("Height before: {}, after: {} (increased: {})",
            height_before, current_height, current_height > height_before);

        wait_for_network_sync(
            current_height,
            &nodes.all_nodes(),
            200,
            &format!("VoteCoin transfer to {}", name),
        )
        .await?;

        // Verify balance immediately after sync - tests that state propagated correctly
        // Use node_0's view of chain UTXOs as ground truth (it mined the block)
        let utxos = nodes.node_0.rpc_client.list_utxos().await?;

        // Debug: show all VoteCoin UTXOs on chain
        let all_votecoin_utxos: Vec<_> = utxos
            .iter()
            .filter(|u| u.output.content.votecoin().is_some())
            .collect();
        tracing::info!("All VoteCoin UTXOs on chain: {} total", all_votecoin_utxos.len());
        for utxo in &all_votecoin_utxos {
            tracing::info!("  - addr={}, amount={:?}", utxo.output.address, utxo.output.content.votecoin());
        }

        let chain_balance: u32 = utxos
            .iter()
            .filter(|u| u.output.address == *addr)
            .filter_map(|u| u.output.content.votecoin())
            .sum();
        anyhow::ensure!(
            chain_balance == VOTECOIN_ALLOCATION,
            "{} VoteCoin balance on chain {} != expected {}. Target addr: {}",
            name, chain_balance, VOTECOIN_ALLOCATION, addr
        );

        // Also verify on receiving node's wallet view
        let wallet_balance = receiving_node.rpc_client.get_votecoin_balance(*addr).await?;
        anyhow::ensure!(
            wallet_balance == VOTECOIN_ALLOCATION,
            "{} VoteCoin wallet balance {} != expected {}",
            name, wallet_balance, VOTECOIN_ALLOCATION
        );
        tracing::info!("{} received {} VoteCoin (chain & wallet verified)", name, wallet_balance);
    }

    // === Phase 3: Chain of VoteCoin transfers (node_1 -> node_2 -> node_3) ===
    tracing::info!("=== Phase 3: Chain transfers ===");

    // node_1 transfers some VoteCoin to node_2
    let chain_amount = VOTECOIN_ALLOCATION / 2;
    tracing::info!("Node 1 transferring {} VoteCoin to node 2...", chain_amount);
    nodes.node_1.rpc_client
        .transfer_votecoin(node_2_addr, chain_amount, TX_FEE, None)
        .await?;

    nodes.node_0.bmm_single(&mut enforcer_post_setup).await?;
    current_height = nodes.node_0.rpc_client.getblockcount().await?;

    wait_for_network_sync(
        current_height,
        &nodes.all_nodes(),
        200,
        "chain transfer node_1 -> node_2",
    )
    .await?;

    // Verify updated balances
    let node_1_balance = nodes.node_1.rpc_client.get_votecoin_balance(node_1_addr).await?;
    let node_2_balance = nodes.node_2.rpc_client.get_votecoin_balance(node_2_addr).await?;
    tracing::info!("After chain transfer: node_1={}, node_2={}", node_1_balance, node_2_balance);

    // node_2 transfers some to node_3
    tracing::info!("Node 2 transferring {} VoteCoin to node 3...", chain_amount / 2);
    nodes.node_2.rpc_client
        .transfer_votecoin(node_3_addr, chain_amount / 2, TX_FEE, None)
        .await?;

    nodes.node_0.bmm_single(&mut enforcer_post_setup).await?;
    current_height = nodes.node_0.rpc_client.getblockcount().await?;

    wait_for_network_sync(
        current_height,
        &nodes.all_nodes(),
        200,
        "chain transfer node_2 -> node_3",
    )
    .await?;

    // === Phase 4: Multiple blocks in rapid succession ===
    tracing::info!("=== Phase 4: Rapid block mining ===");

    for i in 1..=5 {
        tracing::info!("Mining rapid block {}...", i);
        nodes.node_0.bmm_single(&mut enforcer_post_setup).await?;

        current_height = nodes.node_0.rpc_client.getblockcount().await?;

        wait_for_network_sync(
            current_height,
            &nodes.all_nodes(),
            200,
            &format!("rapid block {}", i),
        )
        .await?;
        tracing::info!("All nodes synced to rapid block {}!", i);
    }

    // === Phase 5: Cross-node transfers in single block ===
    tracing::info!("=== Phase 5: Multiple transfers in single block ===");

    // Multiple transfers happening at once
    let small_amount = 1000u32;

    // node_1 -> node_4
    nodes.node_1.rpc_client
        .transfer_votecoin(node_4_addr, small_amount, TX_FEE, None)
        .await?;

    // node_3 -> node_1 (circular)
    let node_1_new_addr: Address = nodes.node_1.rpc_client.get_new_address().await?;
    nodes.node_3.rpc_client
        .transfer_votecoin(node_1_new_addr, small_amount, TX_FEE, None)
        .await?;

    nodes.node_0.bmm_single(&mut enforcer_post_setup).await?;
    current_height = nodes.node_0.rpc_client.getblockcount().await?;

    wait_for_network_sync(
        current_height,
        &nodes.all_nodes(),
        200,
        "cross-node transfers",
    )
    .await?;
    tracing::info!("Cross-node transfers synced!");

    // === Final verification ===
    let final_height = nodes.node_0.rpc_client.getblockcount().await?;
    tracing::info!("\n=== Final Verification ===");

    for (i, node) in nodes.all_nodes().iter().enumerate() {
        let node_height = node.rpc_client.getblockcount().await?;
        tracing::info!("Node {} final height: {}", i, node_height);
        anyhow::ensure!(
            node_height == final_height,
            "Node {} height {} != expected {}",
            i, node_height, final_height
        );
    }

    tracing::info!("\n=== 5-Node Enhanced Sync Test PASSED ===");
    tracing::info!("All 5 nodes synced successfully to height {}", final_height);
    tracing::info!("Tested: basic sync, VoteCoin transfers, chain transfers, rapid mining, cross-node transfers");

    // Cleanup
    {
        drop(nodes);
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

async fn sync_test(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = sync_test_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn sync_test_trial(
    bin_paths: BinPaths,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new("sync_test", sync_test(bin_paths).boxed())
}
