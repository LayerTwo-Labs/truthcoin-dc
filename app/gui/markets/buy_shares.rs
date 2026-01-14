use eframe::egui::{self, Button, RichText};
use truthcoin_dc::math::safe_math::{Rounding, to_sats};
use truthcoin_dc::state::MarketId;

use crate::app::App;

use super::browse::BuySharesResult;

pub struct BuyShares {
    market_id: MarketId,
    outcome_index: u32,
    outcome_label: String,
    current_price: f64,
    trading_fee_pct: f64,
    shares_input: String,
    preview: Option<CostPreview>,
    preview_error: Option<String>,
    is_processing: bool,
    tx_error: Option<String>,
}

struct CostPreview {
    shares: f64,
    base_cost_sats: u64,
    trading_fee_sats: u64,
    total_cost_sats: u64,
    new_price: f64,
}

impl BuyShares {
    pub fn new(
        market_id: MarketId,
        outcome_index: u32,
        outcome_label: String,
        current_price: f64,
        trading_fee_pct: f64,
    ) -> Self {
        Self {
            market_id,
            outcome_index,
            outcome_label,
            current_price,
            trading_fee_pct,
            shares_input: String::new(),
            preview: None,
            preview_error: None,
            is_processing: false,
            tx_error: None,
        }
    }

    fn calculate_preview(&mut self, app: &App) {
        let shares: f64 = match self.shares_input.parse() {
            Ok(s) if s > 0.0 => s,
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

        let mut new_shares = market.shares.clone();

        let valid_combos = market.get_valid_state_combos();
        let actual_idx = match valid_combos.get(self.outcome_index as usize) {
            Some((idx, _)) => *idx,
            None => {
                self.preview_error = Some("Invalid outcome index".to_string());
                return;
            }
        };

        new_shares[actual_idx] += shares;

        let base_cost_f64 = match market.query_update_cost(new_shares.clone()) {
            Ok(cost) => cost,
            Err(e) => {
                self.preview_error =
                    Some(format!("Cost calculation error: {e}"));
                return;
            }
        };

        let base_cost_sats = match to_sats(base_cost_f64, Rounding::Up) {
            Ok(sats) => sats,
            Err(e) => {
                self.preview_error =
                    Some(format!("Cost conversion error: {e}"));
                return;
            }
        };

        let trading_fee_f64 = base_cost_f64 * self.trading_fee_pct;
        let trading_fee_sats = match to_sats(trading_fee_f64, Rounding::Up) {
            Ok(sats) => sats,
            Err(e) => {
                self.preview_error = Some(format!("Fee conversion error: {e}"));
                return;
            }
        };

        let total_cost_sats = base_cost_sats + trading_fee_sats;

        let new_prices = market.calculate_prices(&new_shares);

        let valid_prices: Vec<f64> = valid_combos
            .iter()
            .map(|(idx, _)| new_prices.get(*idx).copied().unwrap_or(0.0))
            .collect();
        let valid_sum: f64 = valid_prices.iter().sum();

        let new_price = if valid_sum > 0.0 {
            valid_prices
                .get(self.outcome_index as usize)
                .copied()
                .unwrap_or(0.0)
                / valid_sum
        } else {
            self.current_price
        };

        self.preview = Some(CostPreview {
            shares,
            base_cost_sats,
            trading_fee_sats,
            total_cost_sats,
            new_price,
        });
        self.preview_error = None;
    }

    fn execute_buy(&mut self, app: &App) {
        let Some(preview) = &self.preview else {
            return;
        };

        let market = match app.node.get_market_by_id(&self.market_id) {
            Ok(Some(m)) => m,
            Ok(None) => {
                self.tx_error = Some("Market not found".to_string());
                return;
            }
            Err(e) => {
                self.tx_error = Some(format!("Error: {e:#}"));
                return;
            }
        };

        let valid_combos = market.get_valid_state_combos();
        let actual_idx = match valid_combos.get(self.outcome_index as usize) {
            Some((idx, _)) => *idx,
            None => {
                self.tx_error = Some("Invalid outcome index".to_string());
                return;
            }
        };

        let tx_fee = bitcoin::Amount::from_sat(1000);

        match app.wallet.buy_shares(
            self.market_id.clone(),
            actual_idx,
            preview.shares,
            preview.base_cost_sats,
            preview.trading_fee_sats,
            tx_fee,
        ) {
            Ok(tx) => {
                if let Err(e) = app.sign_and_send(tx) {
                    self.tx_error = Some(format!("Failed to send: {e:#}"));
                    tracing::error!("Buy shares failed: {e:#}");
                } else {
                    tracing::info!(
                        "Bought {} shares of outcome {} in market {:?}",
                        preview.shares,
                        self.outcome_label,
                        self.market_id
                    );
                    self.is_processing = false;
                }
            }
            Err(e) => {
                self.tx_error = Some(format!("Failed to create tx: {e:#}"));
                tracing::error!("Buy shares tx creation failed: {e:#}");
            }
        }
    }

    pub fn show(&mut self, app: &App, ui: &mut egui::Ui) -> BuySharesResult {
        let mut result = BuySharesResult::Pending;

        ui.group(|ui| {
            ui.heading(format!("Buy Shares: {}", self.outcome_label));
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

            ui.add_space(5.0);

            ui.horizontal(|ui| {
                ui.label("Shares to buy:");
                let old_input = self.shares_input.clone();
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.shares_input)
                        .hint_text("e.g., 10")
                        .desired_width(100.0),
                );

                if self.shares_input != old_input || response.changed() {
                    self.calculate_preview(app);
                }
            });

            if let Some(err) = &self.preview_error {
                ui.colored_label(egui::Color32::RED, err);
            }

            if let Some(preview) = &self.preview {
                ui.add_space(5.0);
                ui.label(RichText::new("Cost Preview:").strong());

                let base_cost_btc =
                    preview.base_cost_sats as f64 / 100_000_000.0;
                let trading_fee_btc =
                    preview.trading_fee_sats as f64 / 100_000_000.0;
                let total_cost_btc =
                    preview.total_cost_sats as f64 / 100_000_000.0;

                egui::Grid::new("cost_preview")
                    .num_columns(2)
                    .spacing([10.0, 2.0])
                    .show(ui, |ui| {
                        ui.label("Base cost:");
                        ui.label(format!("{base_cost_btc:.8} BTC"));
                        ui.end_row();

                        ui.label(format!(
                            "Fee ({:.1}%):",
                            self.trading_fee_pct * 100.0
                        ));
                        ui.label(format!("{trading_fee_btc:.8} BTC"));
                        ui.end_row();

                        ui.label(RichText::new("Total:").strong());
                        ui.label(
                            RichText::new(format!("{total_cost_btc:.8} BTC"))
                                .strong(),
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
                    result = BuySharesResult::Cancelled;
                }

                let can_buy = self.preview.is_some()
                    && !self.is_processing
                    && self.tx_error.is_none();

                if ui
                    .add_enabled(can_buy, Button::new("Confirm Purchase"))
                    .clicked()
                {
                    self.is_processing = true;
                    self.execute_buy(app);

                    if self.tx_error.is_none() {
                        result = BuySharesResult::Completed;
                    } else {
                        self.is_processing = false;
                    }
                }
            });
        });

        result
    }
}
