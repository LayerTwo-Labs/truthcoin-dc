use std::sync::{
    Arc,
    atomic::{self, AtomicBool},
};

use eframe::egui::{self, Button};
use truthcoin_dc::types::Network;

use crate::app::App;

#[derive(Debug)]
pub struct Miner {
    running: Arc<AtomicBool>,
}

impl Default for Miner {
    fn default() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Miner {
    pub fn show(
        &mut self,
        app: Option<&App>,
        network: Network,
        ui: &mut egui::Ui,
    ) {
        let tip = app.and_then(|app| app.node.try_get_tip().ok().flatten());
        let block_height =
            app.and_then(|app| app.node.try_get_tip_height().ok().flatten());

        match (tip, block_height) {
            (Some(hash), Some(h)) => {
                ui.label("Block height: ");
                ui.monospace(format!("{h}"));
                ui.label("Best hash: ");
                let best_hash = &format!("{hash}")[0..8];
                ui.monospace(format!("{best_hash}..."));
            }
            _ => {
                ui.label("No blocks mined yet");
            }
        }
        let running = self.running.load(atomic::Ordering::SeqCst);
        let button_label = if network == Network::Regtest {
            "Mine Block (Regtest)"
        } else {
            "Mine / Refresh Block"
        };

        if let Some(app) = app
            && ui
                .add_enabled(!running, Button::new(button_label))
                .clicked()
        {
            self.running.store(true, atomic::Ordering::SeqCst);
            app.local_pool.spawn_pinned({
                let app = app.clone();
                let running = self.running.clone();
                let is_regtest = network == Network::Regtest;
                move || async move {
                    tracing::debug!("Mining...");

                    // For regtest, first generate a mainchain block to include our BMM
                    if is_regtest
                        && let Some(miner) = app.miner.as_ref()
                    {
                        tracing::debug!("Regtest mode: generating mainchain block first...");
                        let mut miner_write = miner.write().await;
                        if let Err(err) = miner_write.generate().await {
                            tracing::error!("Failed to generate mainchain block: {:#}", anyhow::Error::new(err));
                        }
                        drop(miner_write);
                    }

                    let mining_result = app.mine(None).await;

                    // For regtest, generate another mainchain block to confirm our BMM
                    if is_regtest
                        && let Some(miner) = app.miner.as_ref()
                    {
                        tracing::debug!("Regtest mode: generating mainchain block to confirm BMM...");
                        let mut miner_write = miner.write().await;
                        if let Err(err) = miner_write.generate().await {
                            tracing::error!("Failed to generate confirmation block: {:#}", anyhow::Error::new(err));
                        }
                        drop(miner_write);
                    }

                    running.store(false, atomic::Ordering::SeqCst);
                    if let Err(err) = mining_result {
                        tracing::error!("{:#}", anyhow::Error::new(err))
                    }
                }
            });
        }
    }
}
