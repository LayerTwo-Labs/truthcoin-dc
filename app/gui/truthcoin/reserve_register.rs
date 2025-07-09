use std::borrow::Cow;

use eframe::egui;
use truthcoin_dc::types::TruthcoinData;

use crate::{
    app::App,
    gui::{coins::tx_creator, util::UiExt},
};

fn reserve_truthcoin(
    app: &App,
    plaintext_name: &str,
    fee: bitcoin::Amount,
) -> anyhow::Result<()> {
    let mut tx = app.wallet.create_regular_transaction(fee)?;
    let () = app.wallet.reserve_truthcoin(&mut tx, plaintext_name)?;
    app.sign_and_send(tx).map_err(anyhow::Error::from)
}

fn register_truthcoin(
    app: &App,
    plaintext_name: &str,
    initial_supply: u64,
    truthcoin_data: Cow<TruthcoinData>,
    fee: bitcoin::Amount,
) -> anyhow::Result<()> {
    let mut tx = app.wallet.create_regular_transaction(fee)?;
    let () = app.wallet.register_truthcoin(
        &mut tx,
        plaintext_name,
        truthcoin_data,
        initial_supply,
    )?;
    app.sign_and_send(tx).map_err(anyhow::Error::from)
}

#[derive(Debug, Default)]
struct Reserve {
    plaintext_name: String,
    fee: String,
}

impl Reserve {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        ui.add_sized((250., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let plaintext_name_edit =
                    egui::TextEdit::singleline(&mut self.plaintext_name)
                        .hint_text("Plaintext Name")
                        .desired_width(150.);
                ui.add(plaintext_name_edit);
            })
            .response
        });
        ui.add_sized((110., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let fee_edit = egui::TextEdit::singleline(&mut self.fee)
                    .hint_text("fee")
                    .desired_width(80.);
                ui.add(fee_edit);
                ui.label("BTC");
            })
            .response
        });
        let fee = bitcoin::Amount::from_str_in(
            &self.fee,
            bitcoin::Denomination::Bitcoin,
        );
        if ui
            .add_enabled(
                !self.plaintext_name.is_empty() && fee.is_ok() && app.is_some(),
                egui::Button::new("Reserve"),
            )
            .clicked()
        {
            if let Err(err) = reserve_truthcoin(
                app.unwrap(),
                &self.plaintext_name,
                fee.expect("should not happen"),
            ) {
                tracing::error!("{err:#}");
            } else {
                *self = Self::default();
            }
        }
    }
}

#[derive(Debug, Default)]
struct Register {
    plaintext_name: String,
    initial_supply: String,
    fee: String,
    truthcoin_data: tx_creator::TrySetTruthcoinData,
}

impl Register {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        ui.add_sized((250., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let plaintext_name_edit =
                    egui::TextEdit::singleline(&mut self.plaintext_name)
                        .hint_text("Plaintext Name")
                        .desired_width(150.);
                ui.add(plaintext_name_edit);
            })
            .response
        });
        ui.add_sized((110., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let initial_supply_edit =
                    egui::TextEdit::singleline(&mut self.initial_supply)
                        .hint_text("Initial Supply")
                        .desired_width(80.);
                ui.add(initial_supply_edit);
            })
            .response
        });
        let initial_supply = self.initial_supply.parse();
        ui.add_sized((110., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let fee_edit = egui::TextEdit::singleline(&mut self.fee)
                    .hint_text("Fee")
                    .desired_width(80.);
                ui.add(fee_edit);
                ui.label("BTC");
            })
            .response
        });
        let fee = bitcoin::Amount::from_str_in(
            &self.fee,
            bitcoin::Denomination::Bitcoin,
        );
        tx_creator::TxCreator::show_truthcoin_options(
            ui,
            &mut self.truthcoin_data,
        );
        let truthcoin_data: Result<TruthcoinData, _> =
            self.truthcoin_data.clone().try_into();
        if let Err(err) = &truthcoin_data {
            ui.monospace_selectable_multiline(err.clone());
        }
        if ui
            .add_enabled(
                !self.plaintext_name.is_empty()
                    && initial_supply.is_ok()
                    && fee.is_ok()
                    && truthcoin_data.is_ok()
                    && app.is_some(),
                egui::Button::new("Register"),
            )
            .clicked()
        {
            if let Err(err) = register_truthcoin(
                app.unwrap(),
                &self.plaintext_name,
                initial_supply.expect("should not happen"),
                Cow::Borrowed(&truthcoin_data.expect("should not happen")),
                fee.expect("should not happen"),
            ) {
                tracing::error!("{err:#}");
            } else {
                *self = Self::default();
            }
        }
    }
}

#[derive(Default)]
pub(super) struct ReserveRegister {
    reserve: Reserve,
    register: Register,
}

impl ReserveRegister {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::SidePanel::left("reserve")
            .exact_width(ui.available_width() / 2.)
            .resizable(false)
            .show_inside(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Reserve");
                    self.reserve.show(app, ui);
                })
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Register");
                self.register.show(app, ui);
            })
        });
    }
}
