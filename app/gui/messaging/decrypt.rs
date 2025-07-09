use eframe::egui;

use truthcoin_dc::types::EncryptionPubKey;

use crate::{
    app::App,
    gui::util::{InnerResponseExt, UiExt, borsh_deserialize_hex},
};

#[derive(Debug)]
pub struct DecryptMessage {
    // Encryption pubkey or Truthcoin ID
    receiver_input: String,
    // none if not yet set, otherwise result of parsing/resolving receiver pubkey
    receiver_pubkey: Option<anyhow::Result<EncryptionPubKey>>,
    ciphertext: String,
    // none if not yet computed, otherwise result of attempting to decrypt
    plaintext_bytes: Option<anyhow::Result<Vec<u8>>>,
}

impl DecryptMessage {
    pub fn new() -> Self {
        Self {
            receiver_input: String::new(),
            receiver_pubkey: None,
            ciphertext: String::new(),
            plaintext_bytes: None,
        }
    }

    fn show_error(ui: &mut egui::Ui, error: &anyhow::Error) {
        ui.monospace_selectable_singleline(false, "Error: ");
        ui.horizontal_wrapped(|ui| {
            ui.monospace_selectable_multiline(format!("{error:#}"));
        });
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        ui.heading("Decrypt Message");
        let Some(app) = app else {
            return;
        };
        let receiver_input_response = ui
            .horizontal(|ui| {
                ui.monospace(
                    "Receiver's Truthcoin ID or Encryption Pubkey (Bech32m): ",
                ) | ui.add(egui::TextEdit::singleline(&mut self.receiver_input))
            })
            .join();
        if receiver_input_response.changed() {
            let receiver_pubkey: anyhow::Result<EncryptionPubKey> = {
                if let Ok(truthcoin_id) =
                    borsh_deserialize_hex(&self.receiver_input)
                {
                    app.node
                        .get_current_truthcoin_data(&truthcoin_id)
                        .map_err(anyhow::Error::from)
                        .and_then(|truthcoin_data| {
                            truthcoin_data.encryption_pubkey.ok_or(
                                anyhow::anyhow!(
                                "No encryption pubkey exists for this Truthcoin"
                        ),
                            )
                        })
                } else {
                    EncryptionPubKey::bech32m_decode(&self.receiver_input)
                        .map_err(|_| {
                            anyhow::anyhow!(
                                "Failed to parse Truthcoin ID or Encryption Pubkey"
                            )
                        })
                }
            };
            self.receiver_pubkey = Some(receiver_pubkey);
        }
        let ciphertext_response = ui
            .horizontal_wrapped(|ui| {
                ui.monospace("Ciphertext message (hex):\n")
                    | ui.add(egui::TextEdit::multiline(&mut self.ciphertext))
            })
            .join();
        let receiver_pubkey = match &self.receiver_pubkey {
            None => {
                return;
            }
            Some(Err(err)) => {
                self.plaintext_bytes = None;
                Self::show_error(ui, err);
                return;
            }
            Some(Ok(receiver_pubkey)) => receiver_pubkey,
        };
        // regenerate plaintext if possible
        if receiver_input_response.changed() || ciphertext_response.changed() {
            let ciphertext_bytes = match hex::decode(&self.ciphertext) {
                Ok(ciphertext_bytes) => ciphertext_bytes,
                Err(err) => {
                    Self::show_error(ui, &anyhow::Error::from(err));
                    return;
                }
            };
            self.plaintext_bytes = Some(
                app.wallet
                    .decrypt_msg(receiver_pubkey, &ciphertext_bytes)
                    .map_err(anyhow::Error::from),
            );
        }
        let plaintext_bytes = match &self.plaintext_bytes {
            None => {
                return;
            }
            Some(Err(err)) => {
                Self::show_error(ui, err);
                return;
            }
            Some(Ok(plaintext_bytes)) => plaintext_bytes,
        };
        // show plaintext if possible
        let _resp = ui.horizontal_wrapped(|ui| {
            let plaintext_hex = hex::encode(plaintext_bytes);
            ui.monospace_selectable_multiline(format!(
                "Decrypted message: \n{plaintext_hex}",
            ));
            if ui.button("📋").on_hover_text("Click to copy").clicked() {
                ui.output_mut(|po| {
                    po.commands.push(egui::OutputCommand::CopyText(
                        plaintext_hex.clone(),
                    ))
                });
            };
        });
        // show UTF8-decoded plaintext if possible
        if let Ok(plaintext) = std::str::from_utf8(plaintext_bytes) {
            let _resp = ui.horizontal_wrapped(|ui| {
                ui.monospace_selectable_multiline(format!(
                    "Decrypted message (UTF-8): \n{plaintext}",
                ));
                if ui.button("📋").on_hover_text("Click to copy").clicked() {
                    ui.output_mut(|po| {
                        po.commands.push(egui::OutputCommand::CopyText(
                            plaintext.to_owned(),
                        ))
                    });
                };
            });
        }
    }
}
