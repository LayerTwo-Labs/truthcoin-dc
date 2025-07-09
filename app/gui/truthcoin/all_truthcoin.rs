use std::collections::{BTreeMap, HashMap};

use eframe::egui;
use hex::FromHex;
use truthcoin_dc::{
    state::TruthcoinSeqId,
    types::{TruthcoinData, hashes::TruthcoinId},
};

use crate::{
    app::App,
    gui::util::{InnerResponseExt, UiExt},
};

#[derive(Debug, Default)]
pub(super) struct AllTruthcoin {
    query: String,
}

fn show_truthcoin_data(
    ui: &mut egui::Ui,
    truthcoin_data: &TruthcoinData,
) -> egui::Response {
    let TruthcoinData {
        commitment,
        socket_addr_v4,
        socket_addr_v6,
        encryption_pubkey,
        signing_pubkey,
    } = truthcoin_data;
    let commitment = commitment.map_or("Not set".to_owned(), hex::encode);
    let socket_addr_v4 = socket_addr_v4
        .map_or("Not set".to_owned(), |socket_addr_v4| {
            socket_addr_v4.to_string()
        });
    let socket_addr_v6 = socket_addr_v6
        .map_or("Not set".to_owned(), |socket_addr_v6| {
            socket_addr_v6.to_string()
        });
    let encryption_pubkey =
        encryption_pubkey.map_or("Not set".to_owned(), |epk| epk.to_string());
    let signing_pubkey =
        signing_pubkey.map_or("Not set".to_owned(), |svk| svk.to_string());
    ui.horizontal(|ui| {
        ui.monospace_selectable_singleline(
            true,
            format!("Commitment: {commitment}"),
        )
    })
    .join()
        | ui.horizontal(|ui| {
            ui.monospace_selectable_singleline(
                false,
                format!("IPv4 Address: {socket_addr_v4}"),
            )
        })
        .join()
        | ui.horizontal(|ui| {
            ui.monospace_selectable_singleline(
                false,
                format!("IPv6 Address: {socket_addr_v6}"),
            )
        })
        .join()
        | ui.horizontal(|ui| {
            ui.monospace_selectable_singleline(
                true,
                format!("Encryption Pubkey: {encryption_pubkey}"),
            )
        })
        .join()
        | ui.horizontal(|ui| {
            ui.monospace_selectable_singleline(
                true,
                format!("Signing Pubkey: {signing_pubkey}"),
            )
        })
        .join()
}

fn show_truthcoin_with_data(
    ui: &mut egui::Ui,
    truthcoin_id: &TruthcoinId,
    truthcoin_data: &TruthcoinData,
) -> egui::Response {
    ui.horizontal(|ui| {
        ui.monospace_selectable_singleline(
            true,
            format!("Truthcoin ID: {}", hex::encode(truthcoin_id.0)),
        )
    })
    .join()
        | show_truthcoin_data(ui, truthcoin_data)
}

impl AllTruthcoin {
    fn show_truthcoin(
        &mut self,
        ui: &mut egui::Ui,
        truthcoin: Vec<(TruthcoinSeqId, TruthcoinId, TruthcoinData)>,
    ) {
        let (seq_id_to_truthcoin_id, truthcoin): (
            HashMap<_, _>,
            BTreeMap<_, _>,
        ) = truthcoin
            .into_iter()
            .map(|(seq_id, truthcoin_id, truthcoin_data)| {
                ((seq_id, truthcoin_id), (truthcoin_id, truthcoin_data))
            })
            .unzip();
        ui.horizontal(|ui| {
            let query_edit = egui::TextEdit::singleline(&mut self.query)
                .hint_text("Search")
                .desired_width(150.);
            ui.add(query_edit);
        });
        if self.query.is_empty() {
            truthcoin
                .into_iter()
                .for_each(|(truthcoin_id, truthcoin_data)| {
                    show_truthcoin_with_data(ui, &truthcoin_id, &truthcoin_data);
                })
        } else {
            let name_hash = blake3::hash(self.query.as_bytes()).into();
            let name_hash_pattern = TruthcoinId(name_hash);
            if let Some(truthcoin_data) = truthcoin.get(&name_hash_pattern) {
                show_truthcoin_with_data(ui, &name_hash_pattern, truthcoin_data);
            };
            if let Ok(truthcoin_id_pattern) = TruthcoinId::from_hex(&self.query)
                && let Some(truthcoin_data) = truthcoin.get(&truthcoin_id_pattern)
            {
                show_truthcoin_with_data(
                    ui,
                    &truthcoin_id_pattern,
                    truthcoin_data,
                );
            };
            if let Ok(truthcoin_seq_id_pattern) =
                self.query.parse().map(TruthcoinSeqId)
                && let Some(truthcoin_id) =
                    seq_id_to_truthcoin_id.get(&truthcoin_seq_id_pattern)
            {
                let truthcoin_data = &truthcoin[truthcoin_id];
                show_truthcoin_with_data(ui, truthcoin_id, truthcoin_data);
            }
        }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let Some(app) = app else {
                return;
            };
            match app.node.truthcoin() {
                Err(node_err) => {
                    let err = anyhow::Error::from(node_err);
                    ui.monospace_selectable_multiline(format!("{err:#}"));
                }
                Ok(truthcoin) => self.show_truthcoin(ui, truthcoin),
            }
        });
    }
}
