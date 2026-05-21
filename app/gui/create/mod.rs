use eframe::egui;

use crate::app::App;

mod decisions;
mod markets;

use decisions::Decisions;
use markets::CreateMarket;

#[derive(Default, Eq, PartialEq)]
enum View {
    #[default]
    Market,
    Decision,
}

#[derive(Default)]
pub struct Create {
    view: View,
    decisions: Decisions,
    markets: CreateMarket,
}

impl Create {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show(ui.ctx(), |ui| {
            ui.horizontal(|ui| match self.view {
                View::Market => {
                    ui.heading("Create Prediction Market");
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if ui
                                .button(egui::RichText::new("Claim decision"))
                                .clicked()
                            {
                                self.view = View::Decision;
                            }
                        },
                    );
                }
                View::Decision => {
                    if ui.button("← Back").clicked() {
                        self.view = View::Market;
                    }
                    ui.heading("Claim decision");
                }
            });
            ui.separator();
            match self.view {
                View::Market => self.markets.show(app, ui),
                View::Decision => self.decisions.show(app, ui),
            }
        });
    }
}
