use eframe::egui;

use libes::key::conversion::PublicKeyFrom;
use truthcoin_dc::types::{EncryptionPubKey, keys::Ecies};

use crate::{
    app::App,
    gui::util::{InnerResponseExt, UiExt},
};

#[derive(Debug)]
pub struct EncryptMessage {
    receiver_pubkey_string: String,
    // none if not yet set, otherwise result of parsing receiver pubkey
    receiver_pubkey: Option<anyhow::Result<EncryptionPubKey>>,
    plaintext: String,
    // none if not yet computed, otherwise result of attempting to encrypt
    ciphertext: Option<anyhow::Result<String>>,
}

impl EncryptMessage {
    pub fn new() -> Self {
        Self {
            receiver_pubkey_string: String::new(),
            receiver_pubkey: None,
            plaintext: String::new(),
            ciphertext: None,
        }
    }

    fn show_error(ui: &mut egui::Ui, error: &anyhow::Error) {
        ui.horizontal_wrapped(|ui| {
            ui.monospace("Error: ");
            ui.code(format!("{error}"));
        });
    }

    pub fn show(&mut self, _app: Option<&App>, ui: &mut egui::Ui) {
        ui.heading("Encrypt Message");
        let receiver_pubkey_response = ui
            .horizontal(|ui| {
                ui.monospace("Receiver's Encryption Pubkey (Bech32m): ")
                    | ui.add(egui::TextEdit::singleline(
                        &mut self.receiver_pubkey_string,
                    ))
            })
            .join();
        if receiver_pubkey_response.changed() {
            self.receiver_pubkey = Some(
                EncryptionPubKey::bech32m_decode(&self.receiver_pubkey_string)
                    .map_err(anyhow::Error::new),
            );
        }
        let plaintext_response = ui
            .horizontal_wrapped(|ui| {
                ui.monospace("Plaintext message:\n")
                    | ui.add(egui::TextEdit::multiline(&mut self.plaintext))
            })
            .join();
        let receiver_pubkey = match &self.receiver_pubkey {
            None => {
                return;
            }
            Some(Err(err)) => {
                self.ciphertext = None;
                Self::show_error(ui, err);
                return;
            }
            Some(Ok(receiver_pubkey)) => receiver_pubkey,
        };
        // regenerate ciphertext if possible
        if receiver_pubkey_response.changed() || plaintext_response.changed() {
            let receiver_pubkey =
                libes::key::X25519::pk_from(receiver_pubkey.0);
            self.ciphertext = Some(
                // MUST instantiate a new ecies instance, even if only the
                // plaintext has changed. This is to prevent re-use of
                // ephemeral public keys & shared secrets.
                Ecies::new(receiver_pubkey)
                    .encrypt(self.plaintext.as_bytes())
                    .map(hex::encode)
                    .map_err(|err| anyhow::anyhow!("{err:?}")),
            );
        }
        let ciphertext = match &self.ciphertext {
            None => {
                return;
            }
            Some(Err(err)) => {
                Self::show_error(ui, err);
                return;
            }
            Some(Ok(ciphertext)) => ciphertext,
        };
        // show ciphertext if possible
        let _resp = ui.horizontal_wrapped(|ui| {
            ui.monospace_selectable_multiline(format!(
                "Encrypted message: \n{ciphertext}"
            ));
            if ui.button("📋").on_hover_text("Click to copy").clicked() {
                ui.output_mut(|po| {
                    po.commands
                        .push(egui::OutputCommand::CopyText(ciphertext.clone()))
                });
            };
        });
    }
}
