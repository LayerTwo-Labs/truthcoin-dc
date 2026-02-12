use eframe::egui;
use strum::{EnumIter, IntoEnumIterator};

use crate::app::App;

mod browse;
mod list;
mod my_positions;
mod my_views;
mod sell_shares;

use browse::Browse;
use list::List;
use my_positions::MyPositions;
use my_views::MyViews;

#[derive(Default, EnumIter, Eq, PartialEq, strum::Display)]
enum Tab {
    #[default]
    #[strum(to_string = "Browse")]
    Browse,
    #[strum(to_string = "List")]
    List,
    #[strum(to_string = "My Trades")]
    MyTrades,
    #[strum(to_string = "My Views")]
    MyViews,
}

pub struct Markets {
    tab: Tab,
    browse: Browse,
    list: List,
    my_trades: MyPositions,
    my_views: MyViews,
}

impl Markets {
    pub fn new(_app: Option<&App>) -> Self {
        Self {
            tab: Tab::default(),
            browse: Browse::default(),
            list: List::default(),
            my_trades: MyPositions::default(),
            my_views: MyViews::default(),
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
            Tab::List => {
                self.list.show(app, ui);
            }
            Tab::MyTrades => {
                self.my_trades.show(app, ui);
            }
            Tab::MyViews => {
                self.my_views.show(app, ui);
            }
        });
    }
}
