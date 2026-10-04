use std::collections::HashSet;

use eframe::egui;
use truthcoin_dc::types::{
    GetValue, OutPoint, Output, PointedOutput, Transaction, hash,
};

use crate::app::App;

#[derive(Debug, Default)]
pub struct UtxoSelector;

impl UtxoSelector {
    pub fn show(
        &mut self,
        app: Option<&App>,
        ui: &mut egui::Ui,
        tx: &mut Transaction,
    ) {
        ui.heading("Spend UTXO");
        let selected: HashSet<_> =
            tx.inputs.iter().map(|(outpoint, _)| *outpoint).collect();
        let (total, utxos, unconfirmed_utxos): (
            bitcoin::Amount,
            Vec<_>,
            Vec<_>,
        ) = app
            .map(|app| {
                let utxos_read = app.utxos.read();
                let total: bitcoin::Amount = utxos_read
                    .iter()
                    .filter(|(outpoint, _)| !selected.contains(outpoint))
                    .map(|(_, output)| output.get_value())
                    .sum();
                let mut utxos: Vec<_> =
                    (*utxos_read).clone().into_iter().collect();
                drop(utxos_read);
                utxos.sort_by_key(|(outpoint, _)| {
                    truthcoin_dc::types::OutPointKey::from(outpoint)
                });
                let mut unconfirmed_utxos: Vec<_> =
                    app.unconfirmed_utxos.read().clone().into_iter().collect();
                unconfirmed_utxos.sort_by_key(|(outpoint, _)| {
                    truthcoin_dc::types::OutPointKey::from(outpoint)
                });
                (total, utxos, unconfirmed_utxos)
            })
            .unwrap_or_default();
        ui.separator();
        ui.monospace(format!("Total: {total}"));
        ui.separator();
        egui::Grid::new("utxos").striped(true).show(ui, |ui| {
            ui.monospace("kind");
            ui.monospace("outpoint");
            ui.monospace("value");
            ui.end_row();
            for (outpoint, output) in utxos {
                if selected.contains(&outpoint) {
                    continue;
                }
                show_utxo(ui, &outpoint, &output);

                if ui
                    .add_enabled(
                        !selected.contains(&outpoint),
                        egui::Button::new("spend"),
                    )
                    .clicked()
                {
                    let utxo_hash = hash(&PointedOutput {
                        outpoint,
                        output: output.clone(),
                    });
                    tx.inputs.push((outpoint, utxo_hash));
                }
                ui.end_row();
            }
            for (outpoint, output) in unconfirmed_utxos {
                show_utxo(ui, &outpoint, &output);
                ui.monospace("unconfirmed");
                ui.end_row();
            }
        });
    }
}

pub fn show_utxo(ui: &mut egui::Ui, outpoint: &OutPoint, output: &Output) {
    let (kind, hash, vout) = match outpoint {
        OutPoint::Regular { txid, vout } => {
            ("regular", format!("{txid}"), *vout)
        }
        OutPoint::Deposit(outpoint) => {
            ("deposit", format!("{}", outpoint.txid), outpoint.vout)
        }
        OutPoint::Coinbase { txid, vout } => {
            ("coinbase", format!("{txid}"), *vout)
        }
        OutPoint::MarketFunds {
            market_id,
            block_height,
            is_fee,
        } => (
            if *is_fee { "author_fee" } else { "market" },
            const_hex::encode(market_id),
            *block_height,
        ),
        OutPoint::Payout { hash, vout } => ("payout", format!("{hash}"), *vout),
    };
    let hash = &hash[0..8];
    let value = output.get_value();
    ui.monospace(kind.to_string());
    ui.monospace(format!("{hash}:{vout}",));
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Max), |ui| {
        ui.monospace(format!("{value}"));
    });
}
