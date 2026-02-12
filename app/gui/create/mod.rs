use eframe::egui;
use strum::{EnumIter, IntoEnumIterator};

use crate::app::App;

mod decisions;
mod markets;
mod slots;

use decisions::Decisions;
use markets::CreateMarket;
use slots::Slots;

#[derive(Default, EnumIter, Eq, PartialEq, strum::Display)]
enum Tab {
    #[default]
    #[strum(to_string = "Slots")]
    Slots,
    #[strum(to_string = "Decisions")]
    Decisions,
    #[strum(to_string = "Markets")]
    Markets,
}

#[derive(Default)]
pub struct Create {
    tab: Tab,
    slots: Slots,
    decisions: Decisions,
    markets: CreateMarket,
}

impl Create {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::TopBottomPanel::top("create_tabs").show(ui.ctx(), |ui| {
            ui.horizontal(|ui| {
                Tab::iter().for_each(|tab_variant| {
                    let tab_name = tab_variant.to_string();
                    ui.selectable_value(&mut self.tab, tab_variant, tab_name);
                })
            });
        });
        egui::CentralPanel::default().show(ui.ctx(), |ui| match self.tab {
            Tab::Slots => {
                self.slots.show(app, ui);
            }
            Tab::Decisions => {
                self.decisions.show(app, ui);
            }
            Tab::Markets => {
                self.markets.show(app, ui);
            }
        });
    }
}
