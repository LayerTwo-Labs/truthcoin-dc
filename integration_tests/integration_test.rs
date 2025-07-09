use bip300301_enforcer_integration_tests::{
    setup::{Mode, Network},
    util::{AsyncTrial, TestFailureCollector, TestFileRegistry},
};
use futures::{FutureExt, future::BoxFuture};

use crate::{
    ibd::ibd_trial,
    setup::{Init, PostSetup},
    unknown_withdrawal::unknown_withdrawal_trial,
    util::BinPaths,
    vote::vote_trial,
};

fn deposit_withdraw_roundtrip(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new("deposit_withdraw_roundtrip", async move {
        bip300301_enforcer_integration_tests::integration_test::deposit_withdraw_roundtrip::<PostSetup>(
            bin_paths.others, Network::Regtest, Mode::Mempool,
            Init {
                truthcoin_app: bin_paths.truthcoin,
                data_dir_suffix: None,
            },
        ).await
    }.boxed())
}

pub fn tests(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> Vec<AsyncTrial<BoxFuture<'static, anyhow::Result<()>>>> {
    vec![
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
        unknown_withdrawal_trial(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        vote_trial(bin_paths, file_registry, failure_collector),
    ]
}
