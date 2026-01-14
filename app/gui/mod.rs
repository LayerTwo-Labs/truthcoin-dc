use std::{pin::Pin, task::Poll};

use eframe::egui::{self, Color32, RichText};
use futures::Stream;
use strum::{EnumIter, IntoEnumIterator};
use truthcoin_dc::types::{GetBitcoinValue, Network};

use crate::{app::App, line_buffer::LineBuffer, util::PromiseStream};

mod activity;
mod coins;
mod console_logs;
mod fonts;
mod markets;
mod miner;
mod parent_chain;
mod seed;
mod util;
mod votecoin;

use activity::Activity;
use coins::Coins;
use console_logs::ConsoleLogs;
use fonts::FONT_DEFINITIONS;
use markets::Markets;
use miner::Miner;
use parent_chain::ParentChain;
use seed::SetSeed;
use util::{BITCOIN_LOGO_FA, BITCOIN_ORANGE, UiExt, show_btc_amount};
use votecoin::Votecoin;

/// Bottom panel, if initialized
struct BottomPanelInitialized {
    app: App,
    /// Single unified watch - fires on state or mempool changes
    node_updated: PromiseStream<Pin<Box<dyn Stream<Item = ()> + Send>>>,
}

impl BottomPanelInitialized {
    fn new(app: App) -> Self {
        let node_updated = {
            let rt_guard = app.runtime.enter();
            let node_updated = PromiseStream::from(app.node.watch());
            drop(rt_guard);
            node_updated
        };
        Self { app, node_updated }
    }
}

/// Balance information for display
struct BalanceInfo {
    /// Available balance (confirmed minus UTXOs spent in mempool)
    available: bitcoin::Amount,
    /// Amount locked in pending transactions (spent UTXOs waiting for confirmation)
    pending_spent: bitcoin::Amount,
}

struct BottomPanel {
    initialized: Option<BottomPanelInitialized>,
    /// None if uninitialized
    /// Some(None) if failed to initialize
    balance: Option<Option<BalanceInfo>>,
}

impl BottomPanel {
    /// MUST be run from within a tokio runtime
    fn new(app: Option<App>) -> Self {
        let initialized = app.map(BottomPanelInitialized::new);
        // Calculate initial balance including mempool state
        let balance = initialized
            .as_ref()
            .and_then(|init| Self::calculate_balance(&init.app));
        Self {
            initialized,
            balance,
        }
    }

    /// Updates balance values when node state or mempool changes
    fn update(&mut self) {
        let Some(initialized) = &mut self.initialized else {
            return;
        };
        let rt_guard = initialized.app.runtime.enter();

        let mut should_recalculate = false;
        while let Some(Poll::Ready(())) = initialized.node_updated.poll_next() {
            should_recalculate = true;
        }

        if should_recalculate {
            self.balance = Self::calculate_balance(&initialized.app);
        }

        drop(rt_guard);
    }

    /// Calculate available balance by subtracting pending-spent UTXOs from confirmed balance.
    /// Uses atomic read to ensure consistency between confirmed UTXOs and mempool state.
    fn calculate_balance(app: &App) -> Option<Option<BalanceInfo>> {
        // Get wallet addresses
        let addresses = match app.wallet.get_addresses() {
            Ok(addrs) => addrs,
            Err(err) => {
                let err = anyhow::Error::from(err);
                tracing::error!("Failed to get addresses: {err:#}");
                return Some(None);
            }
        };

        // Single atomic call - both reads use same LMDB transaction
        let (utxos, spent_in_mempool) =
            match app.node.get_utxos_with_mempool_status(&addresses) {
                Ok(result) => result,
                Err(err) => {
                    let err = anyhow::Error::from(err);
                    tracing::error!("Failed to get balance: {err:#}");
                    return Some(None);
                }
            };

        // Calculate total confirmed balance
        let confirmed_total: bitcoin::Amount = utxos
            .values()
            .map(|utxo| utxo.get_bitcoin_value())
            .sum();

        // Calculate amount locked in pending transactions
        let pending_spent: bitcoin::Amount = spent_in_mempool
            .iter()
            .filter_map(|(outpoint, _inpoint)| {
                utxos.get(outpoint).map(|utxo| utxo.get_bitcoin_value())
            })
            .sum();

        // Available = confirmed - pending_spent
        let available = confirmed_total
            .checked_sub(pending_spent)
            .unwrap_or(bitcoin::Amount::ZERO);

        Some(Some(BalanceInfo {
            available,
            pending_spent,
        }))
    }

    fn show_balance(&self, ui: &mut egui::Ui) {
        match &self.balance {
            Some(Some(balance_info)) => {
                ui.monospace(
                    RichText::new(BITCOIN_LOGO_FA.to_string())
                        .color(BITCOIN_ORANGE),
                );
                if balance_info.pending_spent > bitcoin::Amount::ZERO {
                    // Show available balance with pending indicator
                    ui.monospace_selectable_singleline(
                        false,
                        format!(
                            "Available: {} (pending: {})",
                            show_btc_amount(balance_info.available),
                            show_btc_amount(balance_info.pending_spent)
                        ),
                    );
                } else {
                    // No pending transactions, show simple balance
                    ui.monospace_selectable_singleline(
                        false,
                        format!("Balance: {}", show_btc_amount(balance_info.available)),
                    );
                }
            }
            Some(None) => {
                ui.monospace_selectable_singleline(
                    false,
                    "Balance error, check logs",
                );
            }
            None => {
                ui.monospace_selectable_singleline(false, "Loading balance");
            }
        }
    }

    fn show(&mut self, miner: &mut Miner, network: Network, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            self.update();
            self.show_balance(ui);
            // Fill center space,
            // see https://github.com/emilk/egui/discussions/3908#discussioncomment-8270353

            // this frame target width
            // == this frame initial max rect width - last frame others width
            let id_cal_target_size = egui::Id::new("cal_target_size");
            let this_init_max_width = ui.max_rect().width();
            let last_others_width = ui.data(|data| {
                data.get_temp(id_cal_target_size)
                    .unwrap_or(this_init_max_width)
            });
            // this is the total available space for expandable widgets, you can divide
            // it up if you have multiple widgets to expand, even with different ratios.
            let this_target_width = this_init_max_width - last_others_width;

            ui.add_space(this_target_width);
            ui.separator();
            miner.show(
                self.initialized
                    .as_ref()
                    .map(|initialized| &initialized.app),
                network,
                ui,
            );
            // this frame others width
            // == this frame final min rect width - this frame target width
            ui.data_mut(|data| {
                data.insert_temp(
                    id_cal_target_size,
                    ui.min_rect().width() - this_target_width,
                )
            });
        });
    }
}

pub struct EguiApp {
    activity: Activity,
    app: Option<App>,
    votecoin: Votecoin,
    bottom_panel: BottomPanel,
    coins: Coins,
    console_logs: ConsoleLogs,
    markets: Markets,
    miner: Miner,
    network: Network,
    parent_chain: ParentChain,
    set_seed: SetSeed,
    tab: Tab,
}

#[derive(Default, EnumIter, Eq, PartialEq, strum::Display)]
enum Tab {
    #[default]
    #[strum(to_string = "Parent Chain")]
    ParentChain,
    #[strum(to_string = "Coins")]
    Coins,
    #[strum(to_string = "Markets")]
    Markets,
    #[strum(to_string = "Votecoin")]
    Votecoin,
    #[strum(to_string = "Activity")]
    Activity,
    #[strum(to_string = "Console / Logs")]
    ConsoleLogs,
}

impl EguiApp {
    pub fn new(
        app: Option<App>,
        cc: &eframe::CreationContext<'_>,
        logs_capture: LineBuffer,
        rpc_host: url::Host,
        rpc_port: u16,
        network: Network,
    ) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        cc.egui_ctx.set_fonts(FONT_DEFINITIONS.clone());
        let mut style = (*cc.egui_ctx.style()).clone();
        // Palette found using https://coolors.co/005c80-a0a0a0-93032e-ff5400-ffbd00
        // Default blue, eg. selected buttons
        const _LAPIS_LAZULI: Color32 = Color32::from_rgb(0x0D, 0x5c, 0x80);
        // Default grey, eg. grid lines
        const _CADET_GREY: Color32 = Color32::from_rgb(0xa0, 0xa0, 0xa0);
        const _BURGUNDY: Color32 = Color32::from_rgb(0x93, 0x03, 0x2e);
        const ORANGE: Color32 = Color32::from_rgb(0xff, 0x54, 0x00);
        const _AMBER: Color32 = Color32::from_rgb(0xff, 0xbd, 0x00);
        // Accent color
        const ACCENT: Color32 = ORANGE;
        // Grid color / accent color
        style.visuals.widgets.noninteractive.bg_stroke.color = ACCENT;

        cc.egui_ctx.set_style(style);

        let activity = Activity::new(app.as_ref());
        let bottom_panel = BottomPanel::new(app.clone());
        let coins = Coins::new(app.as_ref());
        let console_logs = ConsoleLogs::new(logs_capture, rpc_host, rpc_port);
        let markets = Markets::new(app.as_ref());
        let parent_chain = ParentChain::new(app.as_ref());
        Self {
            activity,
            app,
            votecoin: Votecoin::default(),
            bottom_panel,
            coins,
            console_logs,
            markets,
            miner: Miner::default(),
            network,
            parent_chain,
            set_seed: SetSeed::default(),
            tab: Tab::default(),
        }
    }
}

impl eframe::App for EguiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        if let Some(app) = self.app.as_ref()
            && !app.wallet.has_seed().unwrap_or(false)
        {
            egui::CentralPanel::default().show(ctx, |_ui| {
                egui::Window::new("Set Seed").show(ctx, |ui| {
                    self.set_seed.show(app, ui);
                });
            });
        } else {
            egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    Tab::iter().for_each(|tab_variant| {
                        let tab_name = tab_variant.to_string();
                        ui.selectable_value(
                            &mut self.tab,
                            tab_variant,
                            tab_name,
                        );
                    })
                });
            });
            egui::TopBottomPanel::bottom("bottom_panel")
                .show(ctx, |ui| self.bottom_panel.show(&mut self.miner, self.network, ui));
            egui::CentralPanel::default().show(ctx, |ui| match self.tab {
                Tab::ParentChain => {
                    self.parent_chain.show(self.app.as_ref(), ui);
                }
                Tab::Coins => {
                    let () = self.coins.show(self.app.as_ref(), ui).unwrap();
                }
                Tab::Markets => {
                    self.markets.show(self.app.as_ref(), ui);
                }
                Tab::Votecoin => {
                    self.votecoin.show(self.app.as_ref(), ui);
                }
                Tab::Activity => {
                    self.activity.show(self.app.as_ref(), ui);
                }
                Tab::ConsoleLogs => {
                    self.console_logs.show(self.app.as_ref(), ui);
                }
            });
        }
    }
}
