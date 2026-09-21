use bip300301_enforcer_integration_tests::{
    setup::{
        Mode, Network, PreSetup as EnforcerPreSetup,
        SetupOpts as EnforcerSetupOpts,
    },
    util::{AsyncTrial, TestFailureCollector, TestFileRegistry},
};
use futures::{FutureExt, channel::mpsc, future::BoxFuture};

use crate::{
    block_template::block_template_trial,
    ibd::{ibd_trial, reorg_across_deposit_trial},
    list_mempool::list_mempool_trial,
    receive_address::receive_address_trial,
    roundtrip::roundtrip_trial,
    setup::{Init, PostSetup},
    transfer_many::transfer_many_trial,
    unknown_withdrawal::unknown_withdrawal_trial,
    util::BinPaths,
};

fn deposit_withdraw_roundtrip(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "deposit_withdraw_roundtrip",
        async move {
            let (res_tx, _) = mpsc::unbounded();
            let enforcer_pre_setup =
                EnforcerPreSetup::new(&bin_paths.others, Network::Regtest)?;
            let post_setup = {
                let setup_opts: EnforcerSetupOpts = Default::default();
                enforcer_pre_setup
                    .setup(Mode::Mempool, setup_opts, res_tx)
                    .await?
            };
            bip300301_enforcer_integration_tests::integration_test::deposit_withdraw_roundtrip::<PostSetup>(
                post_setup,
                Init {
                    truthcoin_app: bin_paths.truthcoin()?.clone(),
                    data_dir_suffix: None,
                },
            ).await
        }
        .boxed(),
        file_registry,
        failure_collector,
    )
}

pub fn tests(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> Vec<AsyncTrial<BoxFuture<'static, anyhow::Result<()>>>> {
    vec![
        block_template_trial(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        deposit_withdraw_roundtrip(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        ibd_trial(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        reorg_across_deposit_trial(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        list_mempool_trial(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        receive_address_trial(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        transfer_many_trial(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        unknown_withdrawal_trial(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        roundtrip_trial(bin_paths, file_registry, failure_collector),
    ]
}
