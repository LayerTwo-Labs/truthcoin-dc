use eframe::egui::{self, Button, RichText};
use truthcoin_dc::math::trading;
use truthcoin_dc::state::MarketId;
use truthcoin_dc::types::Address;

use crate::app::App;

pub struct SellShares {
    market_id: MarketId,
    outcome_index: u32,
    outcome_label: String,
    current_price: f64,
    trading_fee_pct: f64,
    max_shares: i64,
    seller_address: Address,
    shares_input: String,
    preview: Option<ProceedsPreview>,
    preview_error: Option<String>,
    is_processing: bool,
    tx_error: Option<String>,
}

struct ProceedsPreview {
    shares: i64,
    gross_proceeds_sats: u64,
    trading_fee_sats: u64,
    net_proceeds_sats: u64,
    new_price: f64,
}

#[derive(PartialEq)]
pub enum SellSharesResult {
    Pending,
    Cancelled,
    Completed,
}

impl SellShares {
    pub fn new(
        market_id: MarketId,
        outcome_index: u32,
        outcome_label: String,
        current_price: f64,
        trading_fee_pct: f64,
        max_shares: i64,
        seller_address: Address,
    ) -> Self {
        Self {
            market_id,
            outcome_index,
            outcome_label,
            current_price,
            trading_fee_pct,
            max_shares,
            seller_address,
            shares_input: String::new(),
            preview: None,
            preview_error: None,
            is_processing: false,
            tx_error: None,
        }
    }

    fn calculate_preview(&mut self, app: &App) {
        let shares: i64 = match self.shares_input.parse() {
            Ok(s) if s > 0 => s,
            Ok(_) => {
                self.preview = None;
                self.preview_error =
                    Some("Shares must be positive".to_string());
                return;
            }
            Err(_) => {
                self.preview = None;
                self.preview_error = None;
                return;
            }
        };

        if shares > self.max_shares {
            self.preview = None;
            self.preview_error = Some(format!(
                "Cannot sell more than {} shares",
                self.max_shares
            ));
            return;
        }

        let market = match app.node.get_market_by_id(&self.market_id) {
            Ok(Some(m)) => m,
            Ok(None) => {
                self.preview_error = Some("Market not found".to_string());
                return;
            }
            Err(e) => {
                self.preview_error = Some(format!("Error: {e:#}"));
                return;
            }
        };

        // outcome_index is already the actual state index (not a display index)
        let actual_idx = self.outcome_index as usize;

        // Validate the index is within bounds
        if actual_idx >= market.shares.len() {
            self.preview_error = Some("Invalid outcome index".to_string());
            return;
        }

        // Calculate proceeds: C(current_shares) - C(new_shares)
        let mut new_shares = market.shares.clone();
        new_shares[actual_idx] -= shares;

        // Check for negative shares
        if new_shares[actual_idx] < 0 {
            self.preview_error =
                Some("Would result in negative market shares".to_string());
            return;
        }

        let old_cost =
            match trading::calculate_treasury(&market.shares, market.b()) {
                Ok(c) => c,
                Err(e) => {
                    self.preview_error =
                        Some(format!("Cost calculation error: {e:?}"));
                    return;
                }
            };

        let new_cost =
            match trading::calculate_treasury(&new_shares, market.b()) {
                Ok(c) => c,
                Err(e) => {
                    self.preview_error =
                        Some(format!("Cost calculation error: {e:?}"));
                    return;
                }
            };

        let gross_proceeds_f64 = old_cost - new_cost;
        if gross_proceeds_f64 < 0.0 {
            self.preview_error =
                Some("Invalid proceeds calculation".to_string());
            return;
        }

        // Use shared trading calculation for consistency
        let sell_proceeds = match trading::calculate_sell_proceeds(
            gross_proceeds_f64,
            self.trading_fee_pct,
        ) {
            Ok(proceeds) => proceeds,
            Err(e) => {
                self.preview_error =
                    Some(format!("Proceeds calculation error: {e}"));
                return;
            }
        };

        // Check for fee >= proceeds (would result in 0 or negative net proceeds)
        if sell_proceeds.trading_fee_sats >= sell_proceeds.gross_proceeds_sats {
            self.preview_error =
                Some("Trade too small: fee would exceed proceeds".to_string());
            return;
        }

        let gross_proceeds_sats = sell_proceeds.gross_proceeds_sats;
        let trading_fee_sats = sell_proceeds.trading_fee_sats;
        let net_proceeds_sats = sell_proceeds.net_proceeds_sats;

        // Use shared post-trade price calculation
        let valid_indices: Vec<usize> = market
            .get_valid_state_combos()
            .iter()
            .map(|(idx, _)| *idx)
            .collect();
        let new_price = trading::calculate_post_trade_price(
            &new_shares,
            market.b(),
            actual_idx,
            &valid_indices,
        );
        // Fallback to current price if calculation returns 0
        let new_price = if new_price > 0.0 {
            new_price
        } else {
            self.current_price
        };

        self.preview = Some(ProceedsPreview {
            shares,
            gross_proceeds_sats,
            trading_fee_sats,
            net_proceeds_sats,
            new_price,
        });
        self.preview_error = None;
    }

    fn execute_sell(&mut self, app: &App) {
        let Some(preview) = &self.preview else {
            return;
        };

        // outcome_index is already the actual state index (not a display index)
        let actual_idx = self.outcome_index as usize;

        // Use exact preview value for slippage protection (consistent with buy)
        // Transaction will fail if market conditions change and proceeds fall below this
        let min_proceeds = preview.net_proceeds_sats;

        match app.wallet.trade(
            self.market_id.clone(),
            actual_idx,
            -preview.shares, // Negative for sell
            self.seller_address,
            min_proceeds,
        ) {
            Ok(tx) => {
                if let Err(e) = app.sign_and_send(tx) {
                    self.tx_error = Some(format!("Failed to send: {e:#}"));
                    tracing::error!("Sell shares failed: {e:#}");
                } else {
                    tracing::info!(
                        "Sold {} shares of outcome {} in market {:?}",
                        preview.shares,
                        self.outcome_label,
                        self.market_id
                    );
                    self.is_processing = false;
                }
            }
            Err(e) => {
                self.tx_error = Some(format!("Failed to create tx: {e:#}"));
                tracing::error!("Sell shares tx creation failed: {e:#}");
            }
        }
    }

    pub fn show(&mut self, app: &App, ui: &mut egui::Ui) -> SellSharesResult {
        let mut result = SellSharesResult::Pending;

        ui.group(|ui| {
            ui.heading(format!("Sell Shares: {}", self.outcome_label));
            ui.add_space(5.0);

            ui.horizontal(|ui| {
                ui.label("Current price:");
                ui.label(
                    RichText::new(format!(
                        "{:.1}%",
                        self.current_price * 100.0
                    ))
                    .strong(),
                );
            });

            ui.horizontal(|ui| {
                ui.label("Your shares:");
                ui.label(
                    RichText::new(format!("{}", self.max_shares)).strong(),
                );
            });

            ui.add_space(5.0);

            ui.horizontal(|ui| {
                ui.label("Shares to sell:");
                let old_input = self.shares_input.clone();
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.shares_input)
                        .hint_text("e.g., 10")
                        .desired_width(100.0),
                );

                if self.shares_input != old_input || response.changed() {
                    self.calculate_preview(app);
                }

                if ui.button("Max").clicked() {
                    self.shares_input = format!("{}", self.max_shares);
                    self.calculate_preview(app);
                }
            });

            if let Some(err) = &self.preview_error {
                ui.colored_label(egui::Color32::RED, err);
            }

            if let Some(preview) = &self.preview {
                ui.add_space(5.0);
                ui.label(RichText::new("Proceeds Preview:").strong());

                let gross_proceeds_btc =
                    preview.gross_proceeds_sats as f64 / 100_000_000.0;
                let trading_fee_btc =
                    preview.trading_fee_sats as f64 / 100_000_000.0;
                let net_proceeds_btc =
                    preview.net_proceeds_sats as f64 / 100_000_000.0;

                egui::Grid::new("proceeds_preview")
                    .num_columns(2)
                    .spacing([10.0, 2.0])
                    .show(ui, |ui| {
                        ui.label("Gross proceeds:");
                        ui.label(format!("{gross_proceeds_btc:.8} BTC"));
                        ui.end_row();

                        ui.label(format!(
                            "Fee ({:.1}%):",
                            self.trading_fee_pct * 100.0
                        ));
                        ui.label(format!("-{trading_fee_btc:.8} BTC"));
                        ui.end_row();

                        ui.label(RichText::new("Net proceeds:").strong());
                        ui.label(
                            RichText::new(format!("{net_proceeds_btc:.8} BTC"))
                                .strong()
                                .color(egui::Color32::GREEN),
                        );
                        ui.end_row();

                        ui.label("New price:");
                        ui.label(format!("{:.1}%", preview.new_price * 100.0));
                        ui.end_row();
                    });
            }

            if let Some(err) = &self.tx_error {
                ui.add_space(5.0);
                ui.colored_label(egui::Color32::RED, err);
            }

            ui.add_space(10.0);

            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    result = SellSharesResult::Cancelled;
                }

                let can_sell = self.preview.is_some()
                    && !self.is_processing
                    && self.tx_error.is_none();

                if ui
                    .add_enabled(can_sell, Button::new("Confirm Sale"))
                    .clicked()
                {
                    self.is_processing = true;
                    self.execute_sell(app);

                    if self.tx_error.is_none() {
                        result = SellSharesResult::Completed;
                    } else {
                        self.is_processing = false;
                    }
                }
            });
        });

        result
    }
}
