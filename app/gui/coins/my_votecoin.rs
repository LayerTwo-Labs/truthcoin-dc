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

            let utxos_read = app.utxos.read();
            let mut votecoin_utxos: Vec<_> = utxos_read
                .iter()
                .filter_map(|(outpoint, output)| {
                    output.votecoin().map(|amount| (outpoint, amount))
                })
                .collect();

            if votecoin_utxos.is_empty() {
                ui.label("No Votecoin UTXOs found");
                return;
            }

            // Sort by amount (highest first)
            votecoin_utxos.sort_by(|(_, a), (_, b)| b.cmp(a));

            let total_votecoin: u32 =
                votecoin_utxos.iter().map(|(_, amount)| amount).sum();

            ui.horizontal(|ui| {
                ui.monospace("Total Votecoin: ");
                ui.monospace_selectable_singleline(
                    false,
                    format!("{total_votecoin}"),
                );
            });

            ui.separator();
            ui.heading("Votecoin UTXOs");

            egui::Grid::new("My Votecoin UTXOs")
                .striped(true)
                .num_columns(3)
                .show(ui, |ui| {
                    ui.monospace_selectable_singleline(false, "Outpoint");
                    ui.monospace_selectable_singleline(false, "Amount");
                    ui.monospace_selectable_singleline(false, "Hash");
                    ui.end_row();

                    for (outpoint, amount) in votecoin_utxos {
                        ui.monospace_selectable_singleline(
                            false,
                            format!("{outpoint}"),
                        );
                        ui.monospace_selectable_singleline(
                            false,
                            format!("{amount}"),
                        );
                        let outpoint_str = format!("{outpoint}");
                        let hash_part = if outpoint_str.len() > 16 {
                            &outpoint_str[..16]
                        } else {
                            &outpoint_str
                        };
                        ui.monospace_selectable_singleline(
                            true,
                            format!("{hash_part}..."),
                        );
                        ui.end_row();
                    }
                });
        });
    }
}
