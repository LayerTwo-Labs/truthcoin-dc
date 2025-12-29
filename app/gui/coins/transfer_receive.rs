use eframe::egui::{self, Button};
use strum::{EnumIter, IntoEnumIterator};
use truthcoin_dc::types::Address;

use crate::{app::App, gui::util::UiExt};

#[derive(
    Clone, Copy, Debug, Default, EnumIter, Eq, PartialEq, strum::Display,
)]
enum TransferType {
    #[default]
    Bitcoin,
    Votecoin,
}

#[derive(Debug, Default)]
struct Transfer {
    transfer_type: TransferType,
    dest: String,
    amount: String,
    fee: String,
}

fn create_bitcoin_transfer(
    app: &App,
    dest: Address,
    amount: bitcoin::Amount,
    fee: bitcoin::Amount,
) -> anyhow::Result<()> {
    let tx = app.wallet.create_transfer(dest, amount, fee, None)?;
    app.sign_and_send(tx)?;
    Ok(())
}

fn create_votecoin_transfer(
    app: &App,
    dest: Address,
    amount: u32,
    fee: bitcoin::Amount,
) -> anyhow::Result<()> {
    let tx = app
        .wallet
        .create_votecoin_transfer(dest, amount, fee, None)?;
    app.sign_and_send(tx)?;
    Ok(())
}

impl Transfer {
    fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        // Transfer type selector
        ui.horizontal(|ui| {
            ui.label("Transfer type:");
            egui::ComboBox::from_id_salt("transfer_type")
                .selected_text(self.transfer_type.to_string())
                .show_ui(ui, |ui| {
                    for transfer_type in TransferType::iter() {
                        ui.selectable_value(
                            &mut self.transfer_type,
                            transfer_type,
                            transfer_type.to_string(),
                        );
                    }
                });
        });

        ui.add_sized((250., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let dest_edit = egui::TextEdit::singleline(&mut self.dest)
                    .hint_text("destination address")
                    .desired_width(150.);
                ui.add(dest_edit);
            })
            .response
        });

        ui.add_sized((110., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let amount_edit = egui::TextEdit::singleline(&mut self.amount)
                    .hint_text("amount")
                    .desired_width(80.);
                ui.add(amount_edit);
                match self.transfer_type {
                    TransferType::Bitcoin => ui.label("BTC"),
                    TransferType::Votecoin => ui.label("VOT"),
                };
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

        let dest: Option<Address> = self.dest.parse().ok();
        let fee = bitcoin::Amount::from_str_in(
            &self.fee,
            bitcoin::Denomination::Bitcoin,
        );

        let transfer_enabled = match self.transfer_type {
            TransferType::Bitcoin => {
                let amount = bitcoin::Amount::from_str_in(
                    &self.amount,
                    bitcoin::Denomination::Bitcoin,
                );
                app.is_some() && dest.is_some() && amount.is_ok() && fee.is_ok()
            }
            TransferType::Votecoin => {
                let amount = self.amount.parse::<u32>();
                app.is_some() && dest.is_some() && amount.is_ok() && fee.is_ok()
            }
        };

        if ui
            .add_enabled(transfer_enabled, egui::Button::new("transfer"))
            .clicked()
        {
            let result = match self.transfer_type {
                TransferType::Bitcoin => {
                    let amount = bitcoin::Amount::from_str_in(
                        &self.amount,
                        bitcoin::Denomination::Bitcoin,
                    )
                    .expect("should not happen");
                    create_bitcoin_transfer(
                        app.unwrap(),
                        dest.expect("should not happen"),
                        amount,
                        fee.expect("should not happen"),
                    )
                }
                TransferType::Votecoin => {
                    let amount =
                        self.amount.parse::<u32>().expect("should not happen");
                    create_votecoin_transfer(
                        app.unwrap(),
                        dest.expect("should not happen"),
                        amount,
                        fee.expect("should not happen"),
                    )
                }
            };

            if let Err(err) = result {
                tracing::error!("{err:#}");
            } else {
                *self = Self::default();
            }
        }
    }
}

#[derive(Debug)]
struct Receive {
    address: Option<anyhow::Result<Address>>,
}

impl Receive {
    fn new(app: Option<&App>) -> Self {
        let Some(app) = app else {
            return Self { address: None };
        };
        let address = app
            .wallet
            .get_new_address()
            .map_err(anyhow::Error::from)
            .inspect_err(|err| tracing::error!("{err:#}"));
        Self {
            address: Some(address),
        }
    }

    fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        match &self.address {
            Some(Ok(address)) => {
                ui.monospace_selectable_singleline(false, address.to_string());
            }
            Some(Err(err)) => {
                ui.monospace_selectable_multiline(format!("{err:#}"));
            }
            None => (),
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(app.is_some(), Button::new("generate"))
                .clicked()
            {
                *self = Self::new(app)
            }

            let has_valid_address = matches!(&self.address, Some(Ok(_)));
            if ui
                .add_enabled(has_valid_address, Button::new("copy"))
                .clicked()
                && let Some(Ok(address)) = &self.address
            {
                ui.ctx().copy_text(address.to_string());
            }
        });
    }
}

#[derive(Debug)]
pub(super) struct TransferReceive {
    transfer: Transfer,
    receive: Receive,
}

impl TransferReceive {
    pub fn new(app: Option<&App>) -> Self {
        Self {
            transfer: Transfer::default(),
            receive: Receive::new(app),
        }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::SidePanel::left("transfer")
            .exact_width(ui.available_width() / 2.)
            .resizable(false)
            .show_inside(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Transfer");
                    self.transfer.show(app, ui);
                })
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Receive");
                self.receive.show(app, ui);
            })
        });
    }
}
