use eframe::egui::{self, Response};

use truthcoin_dc::types::{Transaction, Txid};

use crate::{app::App, gui::util::InnerResponseExt};

#[derive(
    Clone, Debug, Default, strum::Display, strum::EnumIter, Eq, PartialEq,
)]
pub enum TxType {
    #[default]
    Regular,
}

#[derive(Debug, Default)]
pub struct TxCreator {
    pub bitcoin_value_in: bitcoin::Amount,
    pub bitcoin_value_out: bitcoin::Amount,
    pub tx_type: TxType,

    base_txid: Txid,
    final_tx: Option<anyhow::Result<Transaction>>,
}

impl TxCreator {
    fn set_tx_data(
        &self,
        _app: &App,
        tx: Transaction,
    ) -> anyhow::Result<Transaction> {
        match &self.tx_type {
            TxType::Regular => Ok(tx),
        }
    }

    pub fn show(
        &mut self,
        app: Option<&App>,
        ui: &mut egui::Ui,
        base_tx: &mut Transaction,
    ) -> anyhow::Result<()> {
        let Some(app) = app else { return Ok(()) };
        let tx_type_dropdown = ui.horizontal(|ui| {
            let combobox = egui::ComboBox::from_id_salt("tx_type")
                .selected_text(format!("{}", self.tx_type))
                .show_ui(ui, |ui| {
                    use strum::IntoEnumIterator;
                    TxType::iter()
                        .map(|tx_type| {
                            let text = tx_type.to_string();
                            ui.selectable_value(
                                &mut self.tx_type,
                                tx_type,
                                text,
                            )
                        })
                        .reduce(|resp0, resp1| resp0 | resp1)
                        .unwrap()
                });
            combobox.join() | ui.heading("Transaction")
        });
        let tx_data_ui = match &mut self.tx_type {
            TxType::Regular => None,
        };
        let tx_data_changed =
            tx_data_ui.is_some_and(|resp: Response| resp.changed());
        let base_txid = base_tx.txid();
        let base_txid_changed = base_txid != self.base_txid;
        if base_txid_changed {
            self.base_txid = base_txid;
        }
        let refresh_final_tx = tx_type_dropdown.join().changed()
            || tx_data_changed
            || base_txid_changed
            || self.final_tx.is_none();
        if refresh_final_tx {
            self.final_tx = Some(self.set_tx_data(app, base_tx.clone()));
        }
        let final_tx = match &self.final_tx {
            None => panic!("impossible! final tx should have been set"),
            Some(Ok(final_tx)) => final_tx,
            Some(Err(wallet_err)) => {
                ui.monospace(format!("{wallet_err}"));
                return Ok(());
            }
        };
        let txid = &format!("{}", final_tx.txid())[0..8];
        ui.monospace(format!("txid: {txid}"));
        if self.bitcoin_value_in >= self.bitcoin_value_out {
            let fee = self.bitcoin_value_in - self.bitcoin_value_out;
            ui.monospace(format!("fee(sats):  {}", fee.to_sat()));
            if ui.button("sign and send").clicked() {
                let () = app.sign_and_send(final_tx.clone())?;
                *base_tx = Transaction::default();
                self.final_tx = None;
            }
        } else {
            ui.label("Not Enough Value In");
        }
        Ok(())
    }
}
