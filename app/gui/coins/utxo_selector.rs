use std::collections::HashSet;

use eframe::egui;
use strum::{EnumIter, IntoEnumIterator, IntoStaticStr};
use truthcoin_dc::types::{
    AssetId, AssetOutputContent, BitcoinOutput, BitcoinOutputContent,
    FilledOutput, OutPoint, Output, Transaction, WithdrawalOutputContent,
};

use crate::{app::App, gui::util::UiExt};

#[derive(
    Clone, Copy, Debug, Default, EnumIter, Eq, IntoStaticStr, PartialEq,
)]
pub enum AssetKind {
    #[default]
    Bitcoin,
    Votecoin,
}

#[derive(Debug, Default)]
pub struct AssetInput {
    asset_kind: AssetKind,
}

impl PartialEq for AssetInput {
    fn eq(&self, other: &Self) -> bool {
        self.asset_kind == other.asset_kind
    }
}

impl Eq for AssetInput {}

impl AssetInput {
    pub fn asset_id(&self) -> anyhow::Result<AssetId> {
        match self.asset_kind {
            AssetKind::Bitcoin => Ok(AssetId::Bitcoin),
            AssetKind::Votecoin => Ok(AssetId::Votecoin),
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_id_salt("asset_kind")
            .selected_text(<&'static str>::from(self.asset_kind))
            .show_ui(ui, |ui| {
                for asset_kind in AssetKind::iter() {
                    ui.selectable_value(
                        &mut self.asset_kind,
                        asset_kind,
                        <&'static str>::from(asset_kind),
                    );
                }
            });
    }
}

#[derive(Debug, Default)]
pub struct UtxoSelector {
    asset_input: AssetInput,
}

impl UtxoSelector {
    fn show_utxos(
        app: Option<&App>,
        ui: &mut egui::Ui,
        tx: &mut Transaction,
        asset_id: AssetId,
    ) {
        let selected: HashSet<_> = tx.inputs.iter().cloned().collect();
        let (mut unconfirmed_utxos, mut utxos): (
            Vec<(OutPoint, BitcoinOutput)>,
            Vec<_>,
        ) = app
            .map(|app| {
                let unconfirmed_utxos = app
                    .unconfirmed_utxos
                    .read()
                    .iter()
                    .filter_map(|(outpoint, output)| {
                        if !selected.contains(outpoint)
                            && asset_id == AssetId::Bitcoin
                        {
                            let output =
                                Option::<BitcoinOutput>::from(output.clone())?;
                            Some((*outpoint, output))
                        } else {
                            None
                        }
                    })
                    .collect();
                let utxos = app
                    .utxos
                    .read()
                    .iter()
                    .filter_map(|(outpoint, output)| {
                        if !selected.contains(outpoint)
                            && output.asset_value().is_some_and(
                                |(output_asset_id, _)| {
                                    output_asset_id == asset_id
                                },
                            )
                        {
                            Some((*outpoint, output.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();
                (unconfirmed_utxos, utxos)
            })
            .unwrap_or_default();
        let total_confirmed_value: u64 = utxos
            .iter()
            .filter_map(|(_, output)| {
                output.asset_value().map(|(_, value)| value)
            })
            .sum();
        let total_unconfirmed_value: bitcoin::Amount = unconfirmed_utxos
            .iter()
            .map(|(_, output)| output.content.0)
            .sum();
        unconfirmed_utxos.sort_by_key(|(outpoint, _)| format!("{outpoint}"));
        utxos.sort_by_key(|(outpoint, _)| format!("{outpoint}"));
        ui.separator();
        if asset_id == AssetId::Bitcoin {
            if total_unconfirmed_value > bitcoin::Amount::ZERO {
                let total_value: bitcoin::Amount =
                    bitcoin::Amount::from_sat(total_confirmed_value)
                        + total_unconfirmed_value;
                ui.monospace(format!(
                    "Total: ₿{total_value} (₿{total_unconfirmed_value} unconfirmed)",
                ));
            } else {
                ui.monospace(format!(
                    "Total: ₿{}",
                    bitcoin::Amount::from_sat(total_confirmed_value),
                ));
            }
        } else if asset_id == AssetId::Votecoin {
            ui.monospace(format!("Total: {total_confirmed_value} VOT"));
        } else {
            ui.monospace(format!("Total: {total_confirmed_value}"));
        }
        ui.separator();
        egui::Grid::new("utxos")
            .striped(true)
            .num_columns(3)
            .show(ui, |ui| {
                ui.monospace_selectable_singleline(false, "Kind");
                ui.monospace_selectable_singleline(false, "Outpoint");
                ui.monospace_selectable_singleline(false, "Value");
                ui.end_row();
                for (outpoint, output) in utxos {
                    show_utxo(ui, &outpoint, &output, false);

                    if ui.button("spend").clicked() {
                        tx.inputs.push(outpoint);
                    }
                    ui.end_row();
                }
                for (outpoint, output) in unconfirmed_utxos {
                    show_unconfirmed_utxo(
                        ui,
                        &outpoint,
                        &output.map_content(BitcoinOutputContent::into),
                        false,
                    );

                    if ui.button("spend").clicked() {
                        tx.inputs.push(outpoint);
                    }
                    ui.end_row();
                }
            });
    }

    pub fn show(
        &mut self,
        app: Option<&App>,
        ui: &mut egui::Ui,
        tx: &mut Transaction,
    ) {
        ui.heading("Spend UTXO");
        self.asset_input.show(ui);
        match self.asset_input.asset_id() {
            Ok(asset_id) => Self::show_utxos(app, ui, tx, asset_id),
            Err(err) => {
                ui.monospace_selectable_multiline(format!("{err:#}"));
            }
        }
    }
}

pub fn show_utxo(
    ui: &mut egui::Ui,
    outpoint: &OutPoint,
    output: &FilledOutput,
    show_asset_id: bool,
) {
    let (kind, hash, vout) = match outpoint {
        OutPoint::Regular { txid, vout } => {
            ("regular", format!("{txid}"), *vout)
        }
        OutPoint::Deposit(outpoint) => {
            ("deposit", format!("{}", outpoint.txid), outpoint.vout)
        }
        OutPoint::Coinbase { merkle_root, vout } => {
            ("coinbase", format!("{merkle_root}"), *vout)
        }
        OutPoint::Market {
            market_id,
            block_height,
        } => ("market", hex::encode(market_id), *block_height),
        OutPoint::MarketAuthorFee {
            market_id,
            block_height,
        } => ("author_fee", hex::encode(market_id), *block_height),
    };
    let hash = &hash[0..8];
    ui.monospace_selectable_singleline(false, kind.to_string());
    ui.monospace_selectable_singleline(true, format!("{hash}:{vout}",));
    match output.asset_value() {
        None => (),
        Some((asset_id @ AssetId::Bitcoin, bitcoin_value)) => {
            ui.with_layout(
                egui::Layout::right_to_left(egui::Align::Max),
                |ui| {
                    let bitcoin_amount =
                        bitcoin::Amount::from_sat(bitcoin_value);
                    if show_asset_id {
                        ui.monospace_selectable_singleline(
                            true,
                            format!("{asset_id}"),
                        );
                    }
                    ui.monospace_selectable_singleline(
                        false,
                        format!("₿{bitcoin_amount}"),
                    );
                },
            );
        }
        Some((asset_id @ AssetId::Votecoin, value)) => {
            if show_asset_id {
                ui.monospace_selectable_singleline(true, format!("{asset_id}"));
            }
            ui.monospace_selectable_singleline(false, format!("{value} VOT"));
        }
    }
}

pub fn show_unconfirmed_utxo(
    ui: &mut egui::Ui,
    outpoint: &OutPoint,
    output: &Output,
    show_asset_id: bool,
) {
    let (kind, hash, vout) = match outpoint {
        OutPoint::Regular { txid, vout } => {
            ("regular", format!("{txid}"), *vout)
        }
        OutPoint::Deposit(outpoint) => {
            ("deposit", format!("{}", outpoint.txid), outpoint.vout)
        }
        OutPoint::Coinbase { merkle_root, vout } => {
            ("coinbase", format!("{merkle_root}"), *vout)
        }
        OutPoint::Market {
            market_id,
            block_height,
        } => ("market", hex::encode(market_id), *block_height),
        OutPoint::MarketAuthorFee {
            market_id,
            block_height,
        } => ("author_fee", hex::encode(market_id), *block_height),
    };
    let hash = &hash[0..8];
    ui.monospace_selectable_singleline(false, kind.to_string());
    ui.monospace_selectable_singleline(true, format!("{hash}:{vout}",));
    match output.content.clone().as_asset() {
        None => (),
        Some(
            AssetOutputContent::Bitcoin(BitcoinOutputContent(value))
            | AssetOutputContent::Withdrawal(WithdrawalOutputContent {
                value,
                ..
            }),
        ) => {
            ui.with_layout(
                egui::Layout::right_to_left(egui::Align::Max),
                |ui| {
                    if show_asset_id {
                        ui.monospace_selectable_singleline(
                            true,
                            format!("{}", AssetId::Bitcoin),
                        );
                    }
                    ui.monospace_selectable_singleline(
                        false,
                        format!("₿{value} (unconfirmed)"),
                    );
                },
            );
        }
        Some(AssetOutputContent::Votecoin(value)) => {
            if show_asset_id {
                ui.monospace_selectable_singleline(true, "Votecoin");
            }
            ui.monospace_selectable_singleline(
                false,
                format!("{value} VOT (unconfirmed)"),
            );
        }
    }
}
