use eframe::egui;

use crate::{app::App, gui::util::UiExt};

#[derive(Debug, Default)]
pub struct MyVotecoin;

impl MyVotecoin {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let Some(app) = app else {
                ui.label("No app connection available");
                return;
            };

            ui.heading("My Votecoin");
            ui.separator();

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

            let mut total_reputation: f64 = 0.0;
            let mut addr_weights: Vec<_> = Vec::new();

            for addr in &addresses {
                let weight = app
                    .node
                    .reputation()
                    .get_reputation(&rotxn, addr)
                    .unwrap_or(0.0);
                if weight > 0.0 {
                    addr_weights.push((*addr, weight));
                    total_reputation += weight;
                }
            }

            ui.horizontal(|ui| {
                ui.monospace("Total Reputation Weight: ");
                ui.monospace_selectable_singleline(
                    false,
                    format!("{total_reputation:.6}"),
                );
            });

            ui.separator();
            ui.heading("Reputation by Address");

            if addr_weights.is_empty() {
                ui.label("No reputation found for wallet addresses");
                return;
            }

            addr_weights.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

            egui::Grid::new("My Votecoin Reputation")
                .striped(true)
                .num_columns(2)
                .show(ui, |ui| {
                    ui.monospace_selectable_singleline(false, "Address");
                    ui.monospace_selectable_singleline(false, "Weight");
                    ui.end_row();

                    for (addr, weight) in &addr_weights {
                        ui.monospace_selectable_singleline(
                            true,
                            format!("{addr}"),
                        );
                        ui.monospace_selectable_singleline(
                            false,
                            format!("{weight:.6}"),
                        );
                        ui.end_row();
                    }
                });
        });
    }
}
