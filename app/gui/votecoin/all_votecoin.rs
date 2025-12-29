use eframe::egui;

use crate::{app::App, gui::util::UiExt};

#[derive(Debug, Default)]
pub(super) struct AllVotecoin {}

impl AllVotecoin {
    fn show_votecoin(&mut self, ui: &mut egui::Ui, total_votecoin: u32) {
        ui.heading("Votecoin Balance");
        ui.separator();

        ui.horizontal(|ui| {
            ui.monospace("Total Votecoin: ");
            ui.monospace_selectable_singleline(
                false,
                format!("{total_votecoin}"),
            );
        });

        ui.separator();
        ui.label(
            "Votecoin is a fixed supply token with 1,000,000 total units.",
        );
        ui.label("Each unit represents voting power in the system.");
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let Some(app) = app else {
                ui.label("No app connection available");
                return;
            };

            // Calculate total votecoin from UTXOs
            let utxos_read = app.utxos.read();
            let total_votecoin: u32 = utxos_read
                .values()
                .filter_map(|output| output.votecoin())
                .sum();

            self.show_votecoin(ui, total_votecoin);
        });
    }
}
