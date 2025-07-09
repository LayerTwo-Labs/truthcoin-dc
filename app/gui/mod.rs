use std::task::Poll;

use eframe::egui::{self, Color32, RichText};
use truthcoin_dc::{util::Watchable, wallet::Wallet};
use strum::{EnumIter, IntoEnumIterator};

use crate::{app::App, line_buffer::LineBuffer, util::PromiseStream};

mod activity;
mod truthcoin;
mod coins;
mod console_logs;
mod fonts;
mod messaging;
mod miner;
mod parent_chain;
mod seed;
mod util;

use activity::Activity;
use truthcoin::Truthcoin;
use coins::Coins;
use console_logs::ConsoleLogs;
use fonts::FONT_DEFINITIONS;
use messaging::Messaging;
use miner::Miner;
use parent_chain::ParentChain;
use seed::SetSeed;
use util::{BITCOIN_LOGO_FA, BITCOIN_ORANGE, UiExt, show_btc_amount};

/// Bottom panel, if initialized
struct BottomPanelInitialized {
    app: App,
    wallet_updated: PromiseStream<<Wallet as Watchable<()>>::WatchStream>,
}

impl BottomPanelInitialized {
    fn new(app: App) -> Self {
        let wallet_updated = {
            let rt_guard = app.runtime.enter();
            let wallet_updated = PromiseStream::from(app.wallet.watch());
            drop(rt_guard);
            wallet_updated
        };
        Self {
            app,
            wallet_updated,
        }
    }
}

struct BottomPanel {
    initialized: Option<BottomPanelInitialized>,
    /// None if uninitialized
    /// Some(None) if failed to initialize
    balance: Option<Option<bitcoin::Amount>>,
}

impl BottomPanel {
    /// MUST be run from within a tokio runtime
    fn new(app: Option<App>) -> Self {
        let initialized = app.map(BottomPanelInitialized::new);
        Self {
            initialized,
            balance: None,
        }
    }

    /// Updates values if the wallet has been updated
    fn update(&mut self) {
        let Some(initialized) = &mut self.initialized else {
            return;
        };
        let rt_guard = initialized.app.runtime.enter();
        match initialized.wallet_updated.poll_next() {
            Some(Poll::Ready(())) => {
                self.balance =
                    match initialized.app.wallet.get_bitcoin_balance() {
                        Ok(balance) => Some(Some(balance.total)),
                        Err(err) => {
                            let err = anyhow::Error::from(err);
                            tracing::error!(
                                "Failed to update balance: {err:#}"
                            );
                            Some(None)
                        }
                    }
            }
            Some(Poll::Pending) | None => (),
        }
        drop(rt_guard);
    }

    fn show_balance(&self, ui: &mut egui::Ui) {
        match self.balance {
            Some(Some(balance)) => {
                ui.monospace(
                    RichText::new(BITCOIN_LOGO_FA.to_string())
                        .color(BITCOIN_ORANGE),
                );
                ui.monospace_selectable_singleline(
                    false,
                    format!("Balance: {}", show_btc_amount(balance)),
                );
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

    fn show(&mut self, miner: &mut Miner, ui: &mut egui::Ui) {
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
    truthcoin: Truthcoin,
    bottom_panel: BottomPanel,
    coins: Coins,
    console_logs: ConsoleLogs,
    messaging: Messaging,
    miner: Miner,
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
    #[strum(to_string = "Truthcoin")]
    Truthcoin,
    #[strum(to_string = "Messaging")]
    Messaging,
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
        let parent_chain = ParentChain::new(app.as_ref());
        Self {
            activity,
            app,
            truthcoin: Truthcoin::default(),
            bottom_panel,
            coins,
            console_logs,
            messaging: Messaging::new(),
            miner: Miner::default(),
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
                .show(ctx, |ui| self.bottom_panel.show(&mut self.miner, ui));
            egui::CentralPanel::default().show(ctx, |ui| match self.tab {
                Tab::ParentChain => {
                    self.parent_chain.show(self.app.as_ref(), ui);
                }
                Tab::Coins => {
                    let () = self.coins.show(self.app.as_ref(), ui).unwrap();
                }
                Tab::Truthcoin => {
                    self.truthcoin.show(self.app.as_ref(), ui);
                }
                Tab::Messaging => {
                    self.messaging.show(self.app.as_ref(), ui);
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
