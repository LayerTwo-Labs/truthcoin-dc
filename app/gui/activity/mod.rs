use eframe::egui;
use strum::{EnumIter, IntoEnumIterator};

use crate::app::App;

mod block_explorer;
mod mempool_explorer;
mod periods;

use block_explorer::BlockExplorer;
use mempool_explorer::MempoolExplorer;
use periods::Periods;

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Default, EnumIter, Eq, PartialEq, strum::Display)]
enum Tab {
    #[default]
    #[strum(to_string = "Block Explorer")]
    BlockExplorer,
    #[strum(to_string = "Mempool Explorer")]
    MemPoolExplorer,
    #[strum(to_string = "Periods")]
    Periods,
}

pub struct Activity {
    block_explorer: BlockExplorer,
    mempool_explorer: MempoolExplorer,
    periods: Periods,
    tab: Tab,
}

impl Activity {
    pub fn new(app: Option<&App>) -> Self {
        let height = app
            .as_ref()
            .and_then(|app| app.node.try_get_tip_height().ok().flatten())
            .unwrap_or(0);
        Self {
            block_explorer: BlockExplorer::new(height),
            mempool_explorer: Default::default(),
            periods: Default::default(),
            tab: Default::default(),
        }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::TopBottomPanel::top("activity_tabs").show(ui.ctx(), |ui| {
            ui.horizontal(|ui| {
                Tab::iter().for_each(|tab_variant| {
                    let tab_name = tab_variant.to_string();
                    ui.selectable_value(&mut self.tab, tab_variant, tab_name);
                })
            });
        });
        egui::CentralPanel::default().show(ui.ctx(), |ui| match self.tab {
            Tab::BlockExplorer => {
                self.block_explorer.show(app, ui);
            }
            Tab::MemPoolExplorer => {
                self.mempool_explorer.show(app, ui);
            }
            Tab::Periods => {
                self.periods.show(app, ui);
            }
        });
    }
}
