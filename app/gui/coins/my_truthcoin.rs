use eframe::egui;
use itertools::{Either, Itertools};

use truthcoin_dc::types::{TruthcoinId, FilledOutput, Hash, Txid};

use crate::{app::App, gui::util::UiExt};

type KnownNameReservation = (Txid, Hash, String);
type UnknownNameReservation = (Txid, Hash);

#[derive(Debug, Default)]
pub struct MyTruthcoin;

impl MyTruthcoin {
    /// Returns Truthcoin reservations with known and unknown names
    fn get_truthcoin_reservations(
        app: &App,
    ) -> (Vec<KnownNameReservation>, Vec<UnknownNameReservation>) {
        let utxos_read = app.utxos.read();
        // all truthcoin reservations
        let truthcoin_reservations = utxos_read
            .values()
            .filter_map(FilledOutput::reservation_data);
        // split into truthcoin reservations for which the names are known,
        // or unknown
        let (
            mut known_name_truthcoin_reservations,
            mut unknown_name_truthcoin_reservations,
        ): (Vec<_>, Vec<_>) =
            truthcoin_reservations.partition_map(|(txid, commitment)| {
                let plain_truthcoin = app
                    .wallet
                    .get_truthcoin_reservation_plaintext(commitment)
                    .expect("failed to retrieve truthcoin reservation data");
                match plain_truthcoin {
                    Some(plain_truthcoin) => {
                        Either::Left((*txid, *commitment, plain_truthcoin))
                    }
                    None => Either::Right((*txid, *commitment)),
                }
            });
        // sort name-known truthcoin reservations by plain name
        known_name_truthcoin_reservations.sort_by(
            |(_, _, plain_name_l), (_, _, plain_name_r)| {
                plain_name_l.cmp(plain_name_r)
            },
        );
        // sort name-unknown truthcoin reservations by txid
        unknown_name_truthcoin_reservations.sort_by_key(|(txid, _)| *txid);
        (
            known_name_truthcoin_reservations,
            unknown_name_truthcoin_reservations,
        )
    }

    pub fn show_reservations(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let (
            known_name_truthcoin_reservations,
            unknown_name_truthcoin_reservations,
        ) = app.map(Self::get_truthcoin_reservations).unwrap_or_default();
        let _response = egui::SidePanel::left("My Truthcoin Reservations")
            .exact_width(350.)
            .resizable(false)
            .show_inside(ui, move |ui| {
                ui.heading("Truthcoin Reservations");
                egui::Grid::new("My Truthcoin Reservations")
                    .num_columns(1)
                    .striped(true)
                    .show(ui, |ui| {
                        for (txid, commitment, plaintext_name) in
                            known_name_truthcoin_reservations
                        {
                            let txid = hex::encode(txid.0);
                            let commitment = hex::encode(commitment);
                            ui.vertical(|ui| {
                                ui.monospace_selectable_singleline(
                                    true,
                                    format!("plaintext name: {plaintext_name}"),
                                );
                                ui.monospace_selectable_singleline(
                                    false,
                                    format!("txid: {txid}"),
                                );
                                ui.monospace_selectable_singleline(
                                    false,
                                    format!("commitment: {commitment}"),
                                );
                            });
                            ui.end_row()
                        }
                        for (txid, commitment) in
                            unknown_name_truthcoin_reservations
                        {
                            let txid = hex::encode(txid.0);
                            let commitment = hex::encode(commitment);
                            ui.vertical(|ui| {
                                ui.monospace_selectable_singleline(
                                    false,
                                    format!("txid: {txid}"),
                                );
                                ui.monospace_selectable_singleline(
                                    false,
                                    format!("commitment: {commitment}"),
                                );
                            });
                            ui.end_row()
                        }
                    });
            });
    }

    /// Returns Truthcoin with known and unknown names
    fn get_truthcoin(
        app: &App,
    ) -> (Vec<(TruthcoinId, String)>, Vec<TruthcoinId>) {
        let utxos_read = app.utxos.read();
        // all owned truthcoin
        let truthcoin = utxos_read.values().filter_map(FilledOutput::truthcoin);
        // split into truthcoin for which the names are known or unknown
        let (mut known_name_truthcoin, mut unknown_name_truthcoin): (
            Vec<_>,
            Vec<_>,
        ) = truthcoin.partition_map(|truthcoin| {
            let plain_truthcoin = app
                .wallet
                .get_truthcoin_plaintext(truthcoin)
                .expect("failed to retrieve truthcoin data");
            match plain_truthcoin {
                Some(plain_truthcoin) => {
                    Either::Left((*truthcoin, plain_truthcoin))
                }
                None => Either::Right(*truthcoin),
            }
        });
        // sort name-known truthcoin by plain name
        known_name_truthcoin.sort_by(|(_, plain_name_l), (_, plain_name_r)| {
            plain_name_l.cmp(plain_name_r)
        });
        // sort name-unknown truthcoin by truthcoin value
        unknown_name_truthcoin.sort();
        (known_name_truthcoin, unknown_name_truthcoin)
    }

    pub fn show_truthcoin(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let (known_name_truthcoin, unknown_name_truthcoin) =
            app.map(Self::get_truthcoin).unwrap_or_default();
        egui::SidePanel::left("My Truthcoin")
            .exact_width(350.)
            .resizable(false)
            .show_inside(ui, |ui| {
                ui.heading("Truthcoin");
                egui::Grid::new("My Truthcoin")
                    .striped(true)
                    .num_columns(1)
                    .show(ui, |ui| {
                        for (truthcoin, plaintext_name) in known_name_truthcoin {
                            ui.vertical(|ui| {
                                ui.monospace_selectable_singleline(
                                    true,
                                    format!("plaintext name: {plaintext_name}"),
                                );
                                ui.monospace_selectable_singleline(
                                    false,
                                    format!(
                                        "truthcoin: {}",
                                        hex::encode(truthcoin.0)
                                    ),
                                );
                            });
                            ui.end_row()
                        }
                        for truthcoin in unknown_name_truthcoin {
                            ui.monospace_selectable_singleline(
                                false,
                                format!(
                                    "truthcoin: {}",
                                    hex::encode(truthcoin.0)
                                ),
                            );
                            ui.end_row()
                        }
                    });
            });
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let _reservations_response = self.show_reservations(app, ui);
        let _truthcoin_response = self.show_truthcoin(app, ui);
    }
}
