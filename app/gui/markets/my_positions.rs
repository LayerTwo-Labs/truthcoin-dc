use eframe::egui::{self, RichText, ScrollArea};
use truthcoin_dc::state::{Market, MarketId, MarketState};

use crate::app::App;

#[derive(Default)]
pub struct MyPositions {
    positions: Vec<PositionInfo>,
    error: Option<String>,
    loaded: bool,
}

#[allow(dead_code)]
struct PositionInfo {
    market_id: MarketId,
    market_title: String,
    market_state: MarketState,
    outcome_index: u32,
    outcome_label: String,
    shares: f64,
    current_price: f64,
    current_value_btc: f64,
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
            Ok(m) => {
                m.into_iter().map(|(m, s)| (m.id.clone(), (m, s))).collect()
            }
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
                        if shares <= 0.0 {
                            continue;
                        }

                        let (title, state, price, label) =
                            if let Some((market, mstate)) =
                                markets.get(&market_id)
                            {
                                let prices =
                                    market.calculate_prices_for_display();
                                let valid_combos =
                                    market.get_valid_state_combos();

                                // Find the display index for this outcome
                                let display_idx =
                                    valid_combos.iter().position(|(idx, _)| {
                                        *idx == outcome_idx as usize
                                    });

                                let (p, lbl) = if let Some(di) = display_idx {
                                    let pr =
                                        prices.get(di).copied().unwrap_or(0.0);
                                    let combo = &valid_combos[di].1;
                                    let l = self.format_outcome_label(
                                        app, market, combo,
                                    );
                                    (pr, l)
                                } else {
                                    (0.0, format!("Outcome {outcome_idx}"))
                                };

                                (market.title.clone(), *mstate, p, lbl)
                            } else {
                                (
                                    "Unknown Market".to_string(),
                                    MarketState::Invalid,
                                    0.0,
                                    format!("Outcome {outcome_idx}"),
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
                            current_value_btc: shares * price / 100_000_000.0,
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
        if market.decision_slots.len() > 1 {
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
                        .decision_slots
                        .get(dim)
                        .and_then(|slot_id| {
                            app.node.get_slot(*slot_id).ok().flatten().and_then(
                                |slot| {
                                    slot.decision.as_ref().map(|d| {
                                        if d.question.len() > 15 {
                                            format!("{}...", &d.question[..15])
                                        } else {
                                            d.question.clone()
                                        }
                                    })
                                },
                            )
                        })
                        .unwrap_or_else(|| format!("D{}", dim + 1));
                    format!("{question}: {val_str}")
                })
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            let slot_info = market.decision_slots.first().and_then(|slot_id| {
                app.node.get_slot(*slot_id).ok().flatten().and_then(|slot| {
                    slot.decision
                        .as_ref()
                        .map(|d| (d.is_scaled, d.question.clone()))
                })
            });

            let (is_scaled, question) =
                slot_info.unwrap_or((false, String::new()));
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

        ui.horizontal(|ui| {
            ui.heading("My Positions");
            if ui.button("Refresh").clicked() {
                self.refresh_positions(app);
            }
        });
        ui.separator();

        // Always refresh from node to show latest state after blocks
        self.refresh_positions(app);

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

        ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("positions_grid")
                .num_columns(5)
                .striped(true)
                .spacing([10.0, 4.0])
                .show(ui, |ui| {
                    ui.label(RichText::new("Market").strong());
                    ui.label(RichText::new("Outcome").strong());
                    ui.label(RichText::new("Shares").strong());
                    ui.label(RichText::new("Price").strong());
                    ui.label(RichText::new("Value").strong());
                    ui.end_row();

                    for pos in &self.positions {
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
                        ui.end_row();
                    }
                });
        });
    }
}
