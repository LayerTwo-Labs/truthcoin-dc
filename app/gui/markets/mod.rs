use eframe::egui;
use strum::{EnumIter, IntoEnumIterator};

use crate::app::App;

mod browse;
mod buy_shares;
mod create_market;
mod my_positions;
mod sell_shares;
mod slots;

use browse::Browse;
use create_market::CreateMarket;
use my_positions::MyPositions;
use slots::Slots;

#[derive(Default, EnumIter, Eq, PartialEq, strum::Display)]
enum Tab {
    #[default]
    #[strum(to_string = "Browse Markets")]
    Browse,
    #[strum(to_string = "My Positions")]
    MyPositions,
    #[strum(to_string = "Slots")]
    Slots,
    #[strum(to_string = "Create Market")]
    CreateMarket,
}

pub struct Markets {
    tab: Tab,
    browse: Browse,
    my_positions: MyPositions,
    slots: Slots,
    create_market: CreateMarket,
}

impl Markets {
    pub fn new(_app: Option<&App>) -> Self {
        Self {
            tab: Tab::default(),
            browse: Browse::default(),
            my_positions: MyPositions::default(),
            slots: Slots::default(),
            create_market: CreateMarket::default(),
        }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::TopBottomPanel::top("markets_tabs").show(ui.ctx(), |ui| {
            ui.horizontal(|ui| {
                Tab::iter().for_each(|tab_variant| {
                    let tab_name = tab_variant.to_string();
                    ui.selectable_value(&mut self.tab, tab_variant, tab_name);
                })
            });
        });
        egui::CentralPanel::default().show(ui.ctx(), |ui| match self.tab {
            Tab::Browse => {
                self.browse.show(app, ui);
            }
            Tab::MyPositions => {
                self.my_positions.show(app, ui);
            }
            Tab::Slots => {
                self.slots.show(app, ui);
            }
            Tab::CreateMarket => {
                self.create_market.show(app, ui);
            }
        });
    }
}
