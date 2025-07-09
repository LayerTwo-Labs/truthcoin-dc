use eframe::egui::{self, Button};

use truthcoin_dc::types::{
    self, AssetId, BitcoinOutputContent, Output, OutputContent, Transaction,
    WithdrawalOutputContent,
};

use super::utxo_selector::AssetInput;
use crate::{
    app::App,
    gui::util::{InnerResponseExt, UiExt as _},
};

#[derive(Debug, Eq, PartialEq)]
enum UtxoType {
    Regular { asset_input: AssetInput },
    Withdrawal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MemoEncoding {
    Base16,
    Plaintext,
}

// optional warning when decoding
type MemoEncoded = (Vec<u8>, Option<String>);

#[derive(Debug)]
pub struct UtxoCreator {
    utxo_type: UtxoType,
    value: String,
    address: String,
    main_address: String,
    main_fee: String,
    // None corresponds to no memo
    memo_encoding: Option<MemoEncoding>,
    // None corresponds to no memo
    memo_user_input: Option<String>,
    // None corresponds to no memo
    memo_encoded: Option<Result<MemoEncoded, hex::FromHexError>>,
}

impl std::fmt::Display for UtxoType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Regular { .. } => write!(f, "regular"),
            Self::Withdrawal => write!(f, "withdrawal"),
        }
    }
}

impl std::fmt::Display for MemoEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoEncoding::Base16 => write!(f, "hex"),
            MemoEncoding::Plaintext => write!(f, "plaintext"),
        }
    }
}

impl Default for UtxoCreator {
    fn default() -> Self {
        Self {
            value: "".into(),
            address: "".into(),
            main_address: "".into(),
            main_fee: "".into(),
            utxo_type: UtxoType::Regular {
                asset_input: Default::default(),
            },
            memo_encoding: None,
            memo_user_input: None,
            memo_encoded: None,
        }
    }
}

impl UtxoCreator {
    fn try_encode_memo(
        memo_encoding: MemoEncoding,
        memo_user_input: &str,
    ) -> Result<MemoEncoded, hex::FromHexError> {
        // try to decode as hex
        let decoded_hex = hex::decode(memo_user_input);
        if memo_user_input.is_empty() {
            return Ok((Vec::new(), None));
        }
        match (memo_encoding, decoded_hex) {
            (MemoEncoding::Base16, Ok(hex)) => Ok((hex, None)),
            (MemoEncoding::Base16, Err(err)) => Err(err),
            (MemoEncoding::Plaintext, Ok(hex)) => {
                let warning = "This looks like hex data. Are you sure that you want to decode it as ASCII?";
                Ok((hex, Some(warning.to_owned())))
            }
            (MemoEncoding::Plaintext, Err(_)) => {
                Ok((memo_user_input.as_bytes().to_owned(), None))
            }
        }
    }

    pub fn show(
        &mut self,
        app: Option<&App>,
        ui: &mut egui::Ui,
        tx: &mut Transaction,
    ) {
        ui.horizontal(|ui| {
            ui.heading("Create");
            egui::ComboBox::from_id_salt("utxo_type")
                .selected_text(format!("{}", self.utxo_type))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.utxo_type,
                        UtxoType::Regular {
                            asset_input: Default::default(),
                        },
                        "regular",
                    );
                    ui.selectable_value(
                        &mut self.utxo_type,
                        UtxoType::Withdrawal,
                        "withdrawal",
                    );
                });
            ui.heading("UTXO");
        });
        let asset_id = match &mut self.utxo_type {
            UtxoType::Regular { asset_input } => {
                ui.horizontal(|ui| asset_input.show(ui));
                match asset_input.asset_id() {
                    Ok(asset_id) => asset_id,
                    Err(err) => {
                        ui.monospace_selectable_multiline(format!("{err:#}"));
                        return;
                    }
                }
            }
            UtxoType::Withdrawal => AssetId::Bitcoin,
        };
        ui.separator();
        ui.horizontal(|ui| {
            if !matches!(asset_id, AssetId::TruthcoinControl(_)) {
                ui.monospace("Value:       ");
                ui.add(egui::TextEdit::singleline(&mut self.value));
            }
            if asset_id == AssetId::Bitcoin {
                ui.monospace("BTC");
            }
        });
        ui.horizontal(|ui| {
            ui.monospace("Address:     ");
            ui.add(egui::TextEdit::singleline(&mut self.address));
            if ui
                .add_enabled(app.is_some(), Button::new("generate"))
                .clicked()
            {
                self.address = app
                    .unwrap()
                    .wallet
                    .get_new_address()
                    .map(|address| format!("{address}"))
                    .unwrap_or("".into());
            }
        });
        let memo_encoding_changed = ui
            .horizontal(|ui| {
                ui.monospace("Memo:     ");
                egui::ComboBox::from_id_salt("memo_encoding")
                    .selected_text(match self.memo_encoding {
                        Some(MemoEncoding::Base16) => "hex",
                        Some(MemoEncoding::Plaintext) => "plaintext",
                        None => "no memo",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.memo_encoding,
                            Some(MemoEncoding::Base16),
                            "hex",
                        ) | ui.selectable_value(
                            &mut self.memo_encoding,
                            Some(MemoEncoding::Plaintext),
                            "plaintext",
                        ) | ui.selectable_value(
                            &mut self.memo_encoding,
                            None,
                            "no_memo",
                        )
                    })
                    .join()
            })
            .join()
            .changed();
        // (un)initialize memo
        if memo_encoding_changed {
            match self.memo_encoding {
                None => self.memo_user_input = None,
                Some(_) if self.memo_user_input.is_none() => {
                    self.memo_user_input = Some(String::new())
                }
                Some(_) => (),
            };
            self.memo_encoded = None;
        }
        if let Some(memo_user_input) = self.memo_user_input.as_mut() {
            let memo_user_input_changed = ui
                .horizontal(|ui| {
                    ui.add(egui::TextEdit::multiline(memo_user_input))
                })
                .join()
                .changed();
            if memo_encoding_changed || memo_user_input_changed {
                let memo_encoding = self
                    .memo_encoding
                    .as_ref()
                    .expect("impossible: memo encoding should be set");
                self.memo_encoded = Some(Self::try_encode_memo(
                    *memo_encoding,
                    memo_user_input,
                ));
            }
        }
        match &self.memo_encoded {
            Some(Err(err)) => {
                ui.horizontal(|ui| ui.monospace(format!("Error: {err}")));
            }
            Some(Ok((_, Some(warning)))) => {
                ui.horizontal(|ui| ui.monospace(format!("Warning: {warning}")));
            }
            _ => (),
        };
        if self.utxo_type == UtxoType::Withdrawal {
            ui.horizontal(|ui| {
                ui.monospace("Main Address:");
                ui.add(egui::TextEdit::singleline(&mut self.main_address));
                if ui
                    .add_enabled(app.is_some(), Button::new("generate"))
                    .clicked()
                {
                    match app.unwrap().get_new_main_address() {
                        Ok(main_address) => {
                            self.main_address = format!("{main_address}");
                        }
                        Err(err) => {
                            let err = anyhow::Error::new(err);
                            tracing::error!("{err:#}")
                        }
                    };
                }
            });
            ui.horizontal(|ui| {
                ui.monospace("Main Fee:    ");
                ui.add(egui::TextEdit::singleline(&mut self.main_fee));
                ui.monospace("BTC");
            });
        }
        ui.horizontal(|ui| {
            match self.utxo_type {
                UtxoType::Regular { .. } => {
                    let address: Option<types::Address> =
                        self.address.parse().ok();
                    let output_content: Option<OutputContent> = match asset_id {
                        AssetId::Bitcoin => bitcoin::Amount::from_str_in(
                            &self.value,
                            bitcoin::Denomination::Bitcoin,
                        )
                        .ok()
                        .map(|bitcoin_amount| {
                            OutputContent::Bitcoin(BitcoinOutputContent(
                                bitcoin_amount,
                            ))
                        }),
                        AssetId::Truthcoin(_) => {
                            self.value.parse().ok().map(OutputContent::Truthcoin)
                        }
                        AssetId::TruthcoinControl(_) => {
                            Some(OutputContent::TruthcoinControl)
                        }
                    };
                    if ui
                        .add_enabled(
                            address.is_some() && output_content.is_some(),
                            egui::Button::new("create"),
                        )
                        .clicked()
                    {
                        let memo = self
                            .memo_encoded
                            .clone()
                            .map(|memo_encoded| {
                                memo_encoded
                                    .expect("decoding error displayed elswhere")
                                    .0
                            })
                            .unwrap_or_default();
                        let utxo = Output {
                            address: address.expect("should not happen"),
                            content: output_content.expect("should not happen"),
                            memo,
                        };
                        tx.outputs.push(utxo);
                    }
                }
                UtxoType::Withdrawal => {
                    let value: Option<bitcoin::Amount> =
                        bitcoin::Amount::from_str_in(
                            &self.value,
                            bitcoin::Denomination::Bitcoin,
                        )
                        .ok();
                    let address: Option<types::Address> =
                        self.address.parse().ok();
                    let main_address: Option<
                        bitcoin::Address<bitcoin::address::NetworkUnchecked>,
                    > = self.main_address.parse().ok();
                    let main_fee: Option<bitcoin::Amount> =
                        bitcoin::Amount::from_str_in(
                            &self.main_fee,
                            bitcoin::Denomination::Bitcoin,
                        )
                        .ok();
                    if ui
                        .add_enabled(
                            value.is_some()
                                && address.is_some()
                                && main_address.is_some()
                                && main_fee.is_some(),
                            egui::Button::new("create"),
                        )
                        .clicked()
                    {
                        let utxo = Output {
                            address: address.expect("invalid address"),
                            content: OutputContent::Withdrawal(
                                WithdrawalOutputContent {
                                    value: value.expect("invalid value"),
                                    main_address: main_address
                                        .expect("invalid main_address"),
                                    main_fee: main_fee
                                        .expect("invalid main_fee"),
                                },
                            ),
                            memo: Vec::new(),
                        };
                        tx.outputs.push(utxo);
                    }
                }
            }
            if let Some(app) = app {
                let num_addresses = app.wallet.get_num_addresses().unwrap();
                ui.label(format!("{num_addresses} addresses generated"));
            }
        });
    }
}
