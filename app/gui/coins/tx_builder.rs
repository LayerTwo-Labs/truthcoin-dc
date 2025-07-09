use std::collections::{BTreeMap, BTreeSet, HashSet};

use eframe::egui;

use truthcoin_dc::types::{
    AssetId, AssetOutputContent, TruthcoinId, BitcoinOutputContent,
    GetBitcoinValue, Transaction, WithdrawalOutputContent,
};

use super::{
    tx_creator::TxCreator,
    utxo_creator::UtxoCreator,
    utxo_selector::{UtxoSelector, show_utxo},
};
use crate::{app::App, gui::util::UiExt};

#[derive(Debug, Default)]
pub struct TxBuilder {
    // regular tx without extra data or special inputs/outputs
    base_tx: Transaction,
    tx_creator: TxCreator,
    utxo_creator: UtxoCreator,
    utxo_selector: UtxoSelector,
}

impl TxBuilder {
    pub fn show_value_in(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        ui.heading("Value In");
        let Some(app) = app else {
            return;
        };
        let selected: HashSet<_> =
            self.base_tx.inputs.iter().cloned().collect();
        let utxos_read = app.utxos.read();
        let mut spent_utxos: Vec<_> = utxos_read
            .iter()
            .filter(|(outpoint, _)| selected.contains(outpoint))
            .collect();
        let mut bitcoin_value_in = bitcoin::Amount::ZERO;
        let mut truthcoin_values_in = BTreeMap::<TruthcoinId, u64>::new();
        let mut truthcoin_controls_in = BTreeSet::<TruthcoinId>::new();
        spent_utxos
            .iter()
            .for_each(|(_, output)| match output.asset_value() {
                None => (),
                Some((AssetId::Bitcoin, value)) => {
                    bitcoin_value_in += bitcoin::Amount::from_sat(value);
                }
                Some((AssetId::Truthcoin(truthcoin_id), value)) => {
                    *truthcoin_values_in.entry(truthcoin_id).or_default() +=
                        value;
                }
                Some((AssetId::TruthcoinControl(truthcoin_id), value)) => {
                    assert_eq!(value, 1);
                    truthcoin_controls_in.insert(truthcoin_id);
                }
            });
        self.tx_creator.bitcoin_value_in = bitcoin_value_in;
        spent_utxos.sort_by_key(|(outpoint, _)| format!("{outpoint}"));
        ui.separator();
        egui::Grid::new("totals")
            .striped(true)
            .num_columns(2)
            .show(ui, |ui| {
                ui.monospace_selectable_singleline(false, "Asset");
                ui.monospace_selectable_singleline(false, "Amount");
                ui.end_row();

                ui.monospace_selectable_singleline(false, "Bitcoin");
                ui.monospace_selectable_singleline(
                    false,
                    format!("{bitcoin_value_in}"),
                );
                ui.end_row();

                for truthcoin_control_id in truthcoin_controls_in {
                    ui.monospace_selectable_singleline(
                        true,
                        format!(
                            "Truthcoin Control {}",
                            hex::encode(truthcoin_control_id.0)
                        ),
                    );
                    ui.monospace_selectable_singleline(false, "1");
                    ui.end_row();
                }

                for (truthcoin_id, value) in truthcoin_values_in {
                    ui.monospace_selectable_singleline(
                        true,
                        format!("Truthcoin {}", hex::encode(truthcoin_id.0)),
                    );
                    ui.monospace_selectable_singleline(
                        false,
                        format!("{value}"),
                    );
                    ui.end_row();
                }
            });
        ui.separator();
        egui::Grid::new("utxos")
            .striped(true)
            .num_columns(4)
            .show(ui, |ui| {
                ui.monospace_selectable_singleline(false, "Kind");
                ui.monospace_selectable_singleline(false, "Outpoint");
                ui.monospace_selectable_singleline(false, "Asset ID");
                ui.monospace_selectable_singleline(false, "Value");
                ui.end_row();
                let mut remove = None;
                for (vout, outpoint) in self.base_tx.inputs.iter().enumerate() {
                    let output = &utxos_read[outpoint];
                    if output.get_bitcoin_value() != bitcoin::Amount::ZERO {
                        show_utxo(ui, outpoint, output, true);
                        if ui.button("remove").clicked() {
                            remove = Some(vout);
                        }
                        ui.end_row();
                    }
                }
                if let Some(vout) = remove {
                    self.base_tx.inputs.remove(vout);
                }
            });
    }

    pub fn show_value_out(&mut self, ui: &mut egui::Ui) {
        ui.heading("Value Out");
        ui.separator();
        let bitcoin_value_out: bitcoin::Amount = self
            .base_tx
            .outputs
            .iter()
            .map(GetBitcoinValue::get_bitcoin_value)
            .sum();
        self.tx_creator.bitcoin_value_out = bitcoin_value_out;
        ui.monospace(format!("Total: {bitcoin_value_out}"));
        ui.separator();
        egui::Grid::new("outputs")
            .striped(true)
            .num_columns(4)
            .show(ui, |ui| {
                let mut remove = None;
                ui.monospace_selectable_singleline(false, "Kind");
                ui.monospace_selectable_singleline(false, "vout");
                ui.monospace_selectable_singleline(false, "Address");
                ui.monospace_selectable_singleline(false, "Value");
                ui.end_row();
                for (vout, output) in self.base_tx.indexed_asset_outputs() {
                    let address = &format!("{}", output.address)[0..8];
                    let (asset_kind, value) = match output.content {
                        AssetOutputContent::Bitcoin(BitcoinOutputContent(
                            value,
                        ))
                        | AssetOutputContent::Withdrawal(
                            WithdrawalOutputContent { value, .. },
                        ) => {
                            let bitcoin_value = format!("₿{value}");
                            ("Bitcoin", bitcoin_value)
                        }
                        AssetOutputContent::Truthcoin(value) => {
                            ("Truthcoin", format!("{value}"))
                        }
                        AssetOutputContent::TruthcoinControl => {
                            ("Truthcoin Control", "1".to_owned())
                        }
                    };
                    ui.monospace_selectable_singleline(false, asset_kind);
                    ui.monospace(format!("{vout}"));
                    ui.monospace(address.to_string());
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Max),
                        |ui| {
                            ui.monospace(value);
                        },
                    );
                    if ui.button("remove").clicked() {
                        remove = Some(vout);
                    }
                    ui.end_row();
                }
                if let Some(vout) = remove {
                    self.base_tx.outputs.remove(vout);
                }
            });
    }

    pub fn show(
        &mut self,
        app: Option<&App>,
        ui: &mut egui::Ui,
    ) -> anyhow::Result<()> {
        egui::ScrollArea::horizontal().show(ui, |ui| {
            egui::SidePanel::left("spend_utxo")
                .exact_width(250.)
                .resizable(false)
                .show_inside(ui, |ui| {
                    self.utxo_selector.show(app, ui, &mut self.base_tx);
                });
            egui::SidePanel::left("value_in")
                .exact_width(250.)
                .resizable(false)
                .show_inside(ui, |ui| {
                    let () = self.show_value_in(app, ui);
                });
            egui::SidePanel::left("value_out")
                .exact_width(250.)
                .resizable(false)
                .show_inside(ui, |ui| {
                    let () = self.show_value_out(ui);
                });
            egui::SidePanel::left("create_utxo")
                .exact_width(450.)
                .resizable(false)
                .show_separator_line(false)
                .show_inside(ui, |ui| {
                    self.utxo_creator.show(app, ui, &mut self.base_tx);
                    ui.separator();
                    self.tx_creator.show(app, ui, &mut self.base_tx).unwrap();
                });
        });
        Ok(())
    }
}
