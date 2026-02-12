use eframe::egui;

use crate::app::App;

mod all_votecoin;

use all_votecoin::AllVotecoin;

#[derive(Default)]
pub struct Votecoin {
    all_votecoin: AllVotecoin,
}

impl Votecoin {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show(ui.ctx(), |ui| {
            self.all_votecoin.show(app, ui);
        });
    }
}
