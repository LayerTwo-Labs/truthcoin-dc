use eframe::egui;

use crate::{app::App, gui::util::UiExt};

#[derive(Debug, Default)]
pub(super) struct AllVotecoin {}

impl AllVotecoin {
    fn show_votecoin(&mut self, ui: &mut egui::Ui, total_reputation: f64) {
        ui.heading("Votecoin Balance");
        ui.separator();

        ui.horizontal(|ui| {
            ui.monospace("Total Reputation Weight: ");
            ui.monospace_selectable_singleline(
                false,
                format!("{total_reputation:.6}"),
            );
        });

        ui.separator();
        ui.label(
            "Votecoin reputation weight represents \
             voting power in the system.",
        );
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let Some(app) = app else {
                ui.label("No app connection available");
                return;
            };

            let addresses = match app.wallet.get_addresses() {
                Ok(addrs) => addrs,
                Err(err) => {
                    ui.label(format!("Error loading addresses: {err}"));
                    return;
                }
            };

            let rotxn = match app.node.read_txn() {
                Ok(txn) => txn,
                Err(err) => {
                    ui.label(format!("Error reading state: {err}"));
                    return;
                }
            };

            let total_reputation: f64 = addresses
                .iter()
                .map(|addr| {
                    app.node
                        .reputation()
                        .get_reputation(&rotxn, addr)
                        .unwrap_or(0.0)
                })
                .sum();

            self.show_votecoin(ui, total_reputation);
        });
    }
}
