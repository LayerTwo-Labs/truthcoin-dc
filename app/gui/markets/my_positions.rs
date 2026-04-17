use eframe::egui::{self, RichText, ScrollArea};
use truthcoin_dc::state::{Market, MarketId, MarketState};
use truthcoin_dc::types::Address;

use crate::app::App;

use super::sell_shares::{SellShares, SellSharesResult};

#[derive(Default)]
pub struct MyPositions {
    positions: Vec<PositionInfo>,
    error: Option<String>,
    loaded: bool,
    sell_shares: Option<SellShares>,
}

struct PositionInfo {
    market_id: MarketId,
    market_title: String,
    market_state: MarketState,
    outcome_index: u32,
    outcome_label: String,
    shares: i64,
    current_price: f64,
    current_value_btc: f64,
    trading_fee_pct: f64,
    seller_address: Address,
    tx_pow_config: truthcoin_dc::types::tx_pow::TxPowConfig,
}

impl MyPositions {
    fn refresh_positions(&mut self, app: &App) {
        self.positions.clear();
        self.error = None;

        let addresses = match app.wallet.get_addresses() {
            Ok(addrs) => addrs,
            Err(e) => {
                self.error = Some(format!("Failed to get addresses: {e:#}"));
                self.loaded = true;
                return;
            }
        };

        let markets: std::collections::HashMap<
            MarketId,
            (Market, MarketState),
        > = match app.node.get_all_markets_with_states() {
            Ok(m) => m.into_iter().map(|(m, s)| (m.id, (m, s))).collect(),
            Err(e) => {
                self.error = Some(format!("Failed to get markets: {e:#}"));
                self.loaded = true;
                return;
            }
        };

        for address in addresses {
            match app.node.get_user_share_positions(&address) {
                Ok(user_positions) => {
                    for (market_id, outcome_idx, shares) in user_positions {
                        if shares <= 0 {
                            continue;
                        }

                        let (title, state, price, label, fee_pct, pow_config) =
                            if let Some((market, mstate)) =
                                markets.get(&market_id)
                            {
                                let beta = app
                                    .node
                                    .get_market_beta(&market_id, market)
                                    .unwrap_or(0.0);
                                let prices = market.current_prices(beta);
                                let valid_combos =
                                    market.get_valid_state_combos();

                                let tradeable_idx = outcome_idx as usize;
                                let (p, lbl) = if let Some((_, combo)) =
                                    valid_combos.get(tradeable_idx)
                                {
                                    let pr = prices
                                        .get(tradeable_idx)
                                        .copied()
                                        .unwrap_or(0.0);
                                    let l = self.format_outcome_label(
                                        app, market, combo,
                                    );
                                    (pr, l)
                                } else {
                                    (0.0, format!("Outcome {outcome_idx}"))
                                };

                                (
                                    market.title.clone(),
                                    *mstate,
                                    p,
                                    lbl,
                                    market.trading_fee,
                                    market.tx_pow_config(),
                                )
                            } else {
                                (
                                    "Unknown Market".to_string(),
                                    MarketState::Invalid,
                                    0.0,
                                    format!("Outcome {outcome_idx}"),
                                    0.0,
                                    truthcoin_dc::types::tx_pow::TxPowConfig::default(),
                                )
                            };

                        self.positions.push(PositionInfo {
                            market_id,
                            market_title: title,
                            market_state: state,
                            outcome_index: outcome_idx,
                            outcome_label: label,
                            shares,
                            current_price: price,
                            current_value_btc: shares as f64 * price
                                / 100_000_000.0,
                            trading_fee_pct: fee_pct,
                            seller_address: address,
                            tx_pow_config: pow_config,
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to get positions for {address}: {e:#}"
                    );
                }
            }
        }

        self.loaded = true;
    }

    fn format_outcome_label(
        &self,
        app: &App,
        market: &Market,
        combo: &Vec<usize>,
    ) -> String {
        if market.decision_ids.len() > 1 {
            combo
                .iter()
                .enumerate()
                .map(|(dim, &val)| {
                    let val_str = match val {
                        0 => "No",
                        1 => "Yes",
                        _ => "?",
                    };
                    let question = market
                        .decision_ids
                        .get(dim)
                        .and_then(|decision_id| {
                            app.node
                                .get_decision_entry(*decision_id)
                                .ok()
                                .flatten()
                                .and_then(|entry| {
                                    entry.decision.as_ref().map(|d| {
                                        if d.header.len() > 15 {
                                            format!("{}...", &d.header[..15])
                                        } else {
                                            d.header.clone()
                                        }
                                    })
                                })
                        })
                        .unwrap_or_else(|| format!("D{}", dim + 1));
                    format!("{question}: {val_str}")
                })
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            let decision_info =
                market.decision_ids.first().and_then(|decision_id| {
                    app.node
                        .get_decision_entry(*decision_id)
                        .ok()
                        .flatten()
                        .and_then(|entry| {
                            entry
                                .decision
                                .as_ref()
                                .map(|d| (d.is_scaled(), d.header.clone()))
                        })
                });

            let (is_scaled, question) =
                decision_info.unwrap_or((false, String::new()));
            let question_prefix = if !question.is_empty() {
                let q = if question.len() > 20 {
                    format!("{}...", &question[..20])
                } else {
                    question
                };
                format!("{q}: ")
            } else {
                String::new()
            };

            match combo.first() {
                Some(0) => format!(
                    "{}{}",
                    question_prefix,
                    if is_scaled { "Min" } else { "No" }
                ),
                Some(1) => format!(
                    "{}{}",
                    question_prefix,
                    if is_scaled { "Max" } else { "Yes" }
                ),
                Some(2) => format!("{question_prefix}Abstain"),
                _ => format!("Outcome {combo:?}"),
            }
        }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let Some(app) = app else {
            ui.label("No app connection available");
            return;
        };

        // If we're in sell mode, show the sell dialog
        if let Some(sell_shares) = &mut self.sell_shares {
            let result = sell_shares.show(app, ui);
            match result {
                SellSharesResult::Pending => {}
                SellSharesResult::Cancelled => {
                    self.sell_shares = None;
                }
                SellSharesResult::Completed => {
                    self.sell_shares = None;
                    self.loaded = false; // Force refresh after sell
                }
            }
            return;
        }

        ui.horizontal(|ui| {
            ui.heading("My Positions");
            if ui.button("Refresh").clicked() {
                self.loaded = false; // Force refresh on button click
            }
        });
        ui.separator();

        // Only refresh once on first load or when explicitly requested
        if !self.loaded {
            self.refresh_positions(app);
        }

        if let Some(err) = &self.error {
            ui.colored_label(egui::Color32::RED, err);
            return;
        }

        if self.positions.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("No positions found. Buy shares in a market to see them here.");
            });
            return;
        }

        let total_value: f64 =
            self.positions.iter().map(|p| p.current_value_btc).sum();

        ui.horizontal(|ui| {
            ui.label("Total Value:");
            ui.label(RichText::new(format!("{total_value:.8} BTC")).strong());
        });

        ui.add_space(10.0);

        // Collect positions that user wants to sell
        let mut position_to_sell: Option<usize> = None;

        ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("positions_grid")
                .num_columns(6)
                .striped(true)
                .spacing([10.0, 4.0])
                .show(ui, |ui| {
                    ui.label(RichText::new("Market").strong());
                    ui.label(RichText::new("Outcome").strong());
                    ui.label(RichText::new("Shares").strong());
                    ui.label(RichText::new("Price").strong());
                    ui.label(RichText::new("Value").strong());
                    ui.label(RichText::new("Action").strong());
                    ui.end_row();

                    for (idx, pos) in self.positions.iter().enumerate() {
                        let state_str = match pos.market_state {
                            MarketState::Trading => "",
                            MarketState::Ossified => " [Resolved]",
                            MarketState::Cancelled => " [Cancelled]",
                            MarketState::Invalid => " [Invalid]",
                        };
                        ui.label(format!("{}{}", pos.market_title, state_str));

                        ui.label(&pos.outcome_label);
                        ui.label(format!("{:.2}", pos.shares));
                        ui.label(format!("{:.1}%", pos.current_price * 100.0));
                        ui.label(format!("{:.8} BTC", pos.current_value_btc));

                        // Only show sell button if market is trading
                        if pos.market_state == MarketState::Trading {
                            if ui.button("Sell").clicked() {
                                position_to_sell = Some(idx);
                            }
                        } else {
                            ui.label("-");
                        }
                        ui.end_row();
                    }
                });
        });

        // Handle sell button clicks (outside the grid to avoid borrow issues)
        if let Some(idx) = position_to_sell
            && let Some(pos) = self.positions.get(idx)
        {
            self.sell_shares = Some(SellShares::new(
                pos.market_id,
                pos.outcome_index,
                pos.outcome_label.clone(),
                pos.current_price,
                pos.trading_fee_pct,
                pos.shares,
                pos.seller_address,
                pos.tx_pow_config,
            ));
        }
    }
}
