use eframe::egui;
use strum::{EnumIter, IntoEnumIterator};

use crate::app::App;

mod all_truthcoin;
mod dutch_auction_explorer;
mod reserve_register;

use all_truthcoin::AllTruthcoin;
use dutch_auction_explorer::DutchAuctionExplorer;
use reserve_register::ReserveRegister;

#[derive(Default, EnumIter, Eq, PartialEq, strum::Display)]
enum Tab {
    #[default]
    #[strum(to_string = "All Truthcoin")]
    AllTruthcoin,
    #[strum(to_string = "Reserve & Register")]
    ReserveRegister,
    #[strum(to_string = "Dutch Auction Explorer")]
    DutchAuctionExplorer,
}

#[derive(Default)]
pub struct Truthcoin {
    all_truthcoin: AllTruthcoin,
    dutch_auction_explorer: DutchAuctionExplorer,
    reserve_register: ReserveRegister,
    tab: Tab,
}

impl Truthcoin {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::TopBottomPanel::top("truthcoin_tabs").show(ui.ctx(), |ui| {
            ui.horizontal(|ui| {
                Tab::iter().for_each(|tab_variant| {
                    let tab_name = tab_variant.to_string();
                    ui.selectable_value(&mut self.tab, tab_variant, tab_name);
                })
            });
        });
        egui::CentralPanel::default().show(ui.ctx(), |ui| match self.tab {
            Tab::AllTruthcoin => {
                let () = self.all_truthcoin.show(app, ui);
            }
            Tab::ReserveRegister => {
                let () = self.reserve_register.show(app, ui);
            }
            Tab::DutchAuctionExplorer => {
                let () = self.dutch_auction_explorer.show(app, ui);
            }
        });
    }
}
