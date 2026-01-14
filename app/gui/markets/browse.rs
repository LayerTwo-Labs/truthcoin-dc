use eframe::egui::{self, Button, RichText, ScrollArea};
use truthcoin_dc::state::slots::SlotId;
use truthcoin_dc::state::{Market, MarketId, MarketState};

use crate::app::App;

use super::buy_shares::BuyShares;

#[derive(Default)]
pub struct Browse {
    markets: Vec<(Market, MarketState)>,
    selected_market: Option<MarketId>,
    selected_outcome: Option<usize>,
    buy_shares: Option<BuyShares>,
    error: Option<String>,
    slot_questions: Vec<(SlotId, String)>,
    cached_decision_info: Option<CachedDecisionInfo>,
}

#[derive(Clone)]
struct CachedDecisionInfo {
    is_scaled: bool,
    min: i64,
    max: i64,
    #[allow(dead_code)]
    question: String,
}

impl Browse {
    fn refresh_markets(&mut self, app: &App) {
        match app.node.get_all_markets_with_states() {
            Ok(markets) => {
                self.markets = markets;
                self.error = None;
            }
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                tracing::error!("Failed to fetch markets: {e:#}");
            }
        }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let Some(app) = app else {
            ui.label("No app connection available");
            return;
        };

        self.refresh_markets(app);

        egui::SidePanel::left("market_list_panel")
            .default_width(250.0)
            .resizable(true)
            .show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Markets");
                    if ui.button("Refresh").clicked() {
                        self.refresh_markets(app);
                    }
                });
                ui.separator();

                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::RED, err);
                }

                ScrollArea::vertical().show(ui, |ui| {
                    if self.markets.is_empty() {
                        ui.label("No markets found");
                    } else {
                        for (market, state) in &self.markets {
                            let is_selected = self.selected_market.as_ref()
                                == Some(&market.id);

                            let state_str = match state {
                                MarketState::Trading => "Trading",
                                MarketState::Ossified => "Resolved",
                                MarketState::Cancelled => "Cancelled",
                                MarketState::Invalid => "Invalid",
                            };

                            let label = format!(
                                "{}\n[{}] {} outcomes",
                                market.title,
                                state_str,
                                market.get_valid_state_combos().len()
                            );

                            if ui.selectable_label(is_selected, label).clicked()
                            {
                                self.selected_market = Some(market.id.clone());
                                self.buy_shares = None;
                                self.selected_outcome = None;
                                self.slot_questions.clear();
                                self.cached_decision_info = None;
                            }
                        }
                    }
                });
            });

        let selected_data =
            self.selected_market.as_ref().and_then(|selected_id| {
                match app.node.get_market_by_id(selected_id) {
                    Ok(Some(market)) => {
                        let state = market.state();
                        Some((market, state))
                    }
                    Ok(None) => None,
                    Err(e) => {
                        tracing::error!(
                            "Failed to fetch market {selected_id:?}: {e:#}"
                        );
                        None
                    }
                }
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            if let Some((market, state)) = selected_data {
                self.show_market_detail(app, &market, &state, ui);
            } else if self.selected_market.is_some() {
                ui.label("Market not found");
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Select a market from the list");
                });
            }
        });
    }

    fn show_market_detail(
        &mut self,
        app: &App,
        market: &Market,
        state: &MarketState,
        ui: &mut egui::Ui,
    ) {
        ui.heading(&market.title);
        ui.separator();

        egui::Grid::new("market_info_grid")
            .num_columns(2)
            .spacing([20.0, 4.0])
            .show(ui, |ui| {
                ui.label("State:");
                let state_text = match state {
                    MarketState::Trading => RichText::new("Trading").color(egui::Color32::GREEN),
                    MarketState::Ossified => RichText::new("Resolved").color(egui::Color32::BLUE),
                    MarketState::Cancelled => RichText::new("Cancelled").color(egui::Color32::RED),
                    MarketState::Invalid => RichText::new("Invalid").color(egui::Color32::GRAY),
                };
                ui.label(state_text);
                ui.end_row();

                let n_outcomes = market.get_valid_state_combos().len() as f64;
                let liquidity_sats = market.b * n_outcomes.ln();
                let liquidity_btc = liquidity_sats / 100_000_000.0;

                ui.label("Liquidity Pool:");
                ui.label(format!("{liquidity_btc:.8} BTC"))
                    .on_hover_text("Initial liquidity that subsidizes the market. Higher liquidity = harder to move prices.");
                ui.end_row();

                ui.label("Trading Volume:");
                let volume_btc = market.total_volume_sats as f64 / 100_000_000.0;
                ui.label(format!("{volume_btc:.8} BTC"));
                ui.end_row();

                ui.label("Trading Fee:");
                ui.label(format!("{:.1}%", market.trading_fee * 100.0));
                ui.end_row();

                ui.label("Decisions:");
                ui.label(format!("{} slots", market.decision_slots.len()));
                ui.end_row();
            });

        ui.add_space(10.0);

        if !market.description.is_empty() {
            ui.collapsing("Description", |ui| {
                ui.label(&market.description);
            });
        }

        ui.add_space(10.0);
        ui.separator();

        ui.heading("Outcomes");

        let valid_combos = market.get_valid_state_combos();

        let prices = if let Ok(Some(mempool_shares)) =
            app.node.get_mempool_shares(&market.id)
        {
            let all_prices = market.calculate_prices(&mempool_shares);
            let valid_prices: Vec<f64> = valid_combos
                .iter()
                .map(|(idx, _)| all_prices.get(*idx).copied().unwrap_or(0.0))
                .collect();
            let sum: f64 = valid_prices.iter().sum();
            if sum > 0.0 {
                valid_prices.iter().map(|p| p / sum).collect()
            } else {
                valid_prices
            }
        } else {
            market.calculate_prices_for_display()
        };
        let is_trading = matches!(state, MarketState::Trading);

        if self.cached_decision_info.is_none()
            && market.decision_slots.len() == 1
            && let Some(slot_id) = market.decision_slots.first()
            && let Ok(Some(slot)) = app.node.get_slot(*slot_id)
            && let Some(decision) = &slot.decision
        {
            self.cached_decision_info = Some(CachedDecisionInfo {
                is_scaled: decision.is_scaled,
                min: decision.min.unwrap_or(0),
                max: decision.max.unwrap_or(100),
                question: decision.question.clone(),
            });
        }

        let is_scaled = self
            .cached_decision_info
            .as_ref()
            .map(|info| info.is_scaled)
            .unwrap_or(false);

        let is_categorical = !is_scaled
            && market.decision_slots.len() > 1
            && valid_combos.iter().all(|(_, combo)| {
                combo.iter().filter(|&&v| v == 1).count() <= 1
            });

        if market.decision_slots.len() > 1 && self.slot_questions.is_empty() {
            for slot_id in &market.decision_slots {
                if let Ok(Some(slot)) = app.node.get_slot(*slot_id) {
                    let question = slot
                        .decision
                        .as_ref()
                        .map(|d| d.question.clone())
                        .unwrap_or_else(|| format!("Slot {slot_id:?}"));
                    self.slot_questions.push((*slot_id, question));
                }
            }
        }

        ui.add_space(5.0);

        if is_scaled {
            self.show_scaled_market(app, market, is_trading, &prices, ui);
        } else if is_categorical {
            ui.label(
                RichText::new("Select one outcome:")
                    .italics()
                    .color(egui::Color32::GRAY),
            );
            ui.add_space(5.0);

            ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                for (i, (_outcome_idx, combo)) in
                    valid_combos.iter().enumerate()
                {
                    let price = prices.get(i).copied().unwrap_or(0.0);

                    let yes_slot_idx = combo.iter().position(|&v| v == 1);

                    let outcome_label = if let Some(slot_idx) = yes_slot_idx {
                        self.slot_questions
                            .get(slot_idx)
                            .map(|(_, q)| q.clone())
                            .unwrap_or_else(|| {
                                format!("Option {}", slot_idx + 1)
                            })
                    } else {
                        market
                            .residual_names
                            .as_ref()
                            .and_then(|names| names.first().cloned())
                            .unwrap_or_else(|| {
                                "Other / None of the above".to_string()
                            })
                    };

                    let is_selected = self.selected_outcome == Some(i);

                    let frame = egui::Frame::new()
                        .fill(if is_selected {
                            egui::Color32::from_rgb(60, 100, 60)
                        } else {
                            egui::Color32::TRANSPARENT
                        })
                        .inner_margin(8.0)
                        .corner_radius(4.0);

                    frame.show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let indicator =
                                if is_selected { "◉" } else { "○" };
                            ui.label(RichText::new(indicator).size(16.0));

                            ui.vertical(|ui| {
                                ui.label(
                                    RichText::new(&outcome_label).strong(),
                                );
                                ui.horizontal(|ui| {
                                    ui.label(format!("{:.1}%", price * 100.0));
                                    ui.label(
                                        RichText::new(format!(
                                            "(Price: {price:.4})"
                                        ))
                                        .small()
                                        .color(egui::Color32::GRAY),
                                    );
                                });
                            });

                            if is_trading && self.buy_shares.is_none() {
                                let response = ui.interact(
                                    ui.min_rect(),
                                    ui.id().with(i),
                                    egui::Sense::click(),
                                );
                                if response.clicked() {
                                    self.selected_outcome = Some(i);
                                }
                            }
                        });
                    });

                    if is_selected && is_trading && self.buy_shares.is_none() {
                        ui.horizontal(|ui| {
                            ui.add_space(24.0);
                            if ui.button("Buy Selected Outcome").clicked() {
                                self.buy_shares = Some(BuyShares::new(
                                    market.id.clone(),
                                    i as u32,
                                    outcome_label.clone(),
                                    price,
                                    market.trading_fee,
                                ));
                            }
                        });
                    }

                    ui.add_space(4.0);
                }
            });
        } else {
            let is_dimensional = market.decision_slots.len() > 1;

            if is_dimensional {
                ui.label(
                    RichText::new(format!(
                        "Dimensional Market: {} linked decisions",
                        market.decision_slots.len()
                    ))
                    .italics(),
                );
            }

            ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                egui::Grid::new("outcomes_grid")
                    .num_columns(if is_trading { 4 } else { 3 })
                    .striped(true)
                    .spacing([10.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(RichText::new("Outcome").strong());
                        ui.label(RichText::new("Probability").strong());
                        ui.label(RichText::new("Price").strong());
                        if is_trading {
                            ui.label(RichText::new("Action").strong());
                        }
                        ui.end_row();

                        for (i, (outcome_idx, combo)) in
                            valid_combos.iter().enumerate()
                        {
                            let price = prices.get(i).copied().unwrap_or(0.0);

                            let outcome_label = if is_dimensional {
                                combo
                                    .iter()
                                    .enumerate()
                                    .map(|(dim, &val)| {
                                        let val_str = match val {
                                            0 => "No",
                                            1 => "Yes",
                                            _ => "?",
                                        };
                                        let question = self
                                            .slot_questions
                                            .get(dim)
                                            .map(|(_, q)| {
                                                if q.len() > 20 {
                                                    format!("{}...", &q[..20])
                                                } else {
                                                    q.clone()
                                                }
                                            })
                                            .unwrap_or_else(|| {
                                                format!("D{}", dim + 1)
                                            });
                                        format!("{question}: {val_str}")
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            } else {
                                match combo.first() {
                                    Some(0) => "No".to_string(),
                                    Some(1) => "Yes".to_string(),
                                    _ => format!("Outcome {outcome_idx}"),
                                }
                            };

                            ui.label(&outcome_label);
                            ui.label(format!("{:.1}%", price * 100.0));
                            ui.label(format!("{price:.4}"));

                            if is_trading
                                && ui
                                    .add_enabled(
                                        self.buy_shares.is_none(),
                                        Button::new("Buy"),
                                    )
                                    .clicked()
                            {
                                self.buy_shares = Some(BuyShares::new(
                                    market.id.clone(),
                                    i as u32,
                                    outcome_label.clone(),
                                    price,
                                    market.trading_fee,
                                ));
                            }
                            ui.end_row();
                        }
                    });
            });
        }

        if let Some(buy_shares) = &mut self.buy_shares {
            ui.add_space(10.0);
            ui.separator();

            let result = buy_shares.show(app, ui);

            match result {
                BuySharesResult::Pending => {}
                BuySharesResult::Cancelled => {
                    self.buy_shares = None;
                }
                BuySharesResult::Completed => {
                    self.buy_shares = None;
                    self.refresh_markets(app);
                }
            }
        }
    }

    fn show_scaled_market(
        &mut self,
        _app: &App,
        market: &Market,
        is_trading: bool,
        prices: &[f64],
        ui: &mut egui::Ui,
    ) {
        let Some(info) = &self.cached_decision_info else {
            ui.label("Unable to load decision info");
            return;
        };

        let p_min = prices.first().copied().unwrap_or(0.5);
        let p_max = prices.get(1).copied().unwrap_or(0.5);
        let p_abstain = prices.get(2).copied().unwrap_or(0.0);

        let sum = p_min + p_max;
        let normalized_value = if sum > 0.0 { p_max / sum } else { 0.5 };

        let implied_value =
            info.min as f64 + normalized_value * (info.max - info.min) as f64;

        ui.label(
            RichText::new("Scaled Decision Market")
                .italics()
                .color(egui::Color32::GRAY),
        );
        ui.add_space(5.0);

        // Show prominent implied value
        ui.group(|ui| {
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("Market Estimate").strong());
                ui.label(
                    RichText::new(format!("{implied_value:.2}"))
                        .size(32.0)
                        .strong()
                        .color(egui::Color32::from_rgb(100, 200, 100)),
                );
                ui.label(format!("Range: {} - {}", info.min, info.max));
                ui.label(
                    RichText::new(format!(
                        "({:.1}% normalized)",
                        normalized_value * 100.0
                    ))
                    .small()
                    .color(egui::Color32::GRAY),
                );
            });
        });

        ui.add_space(10.0);

        ui.collapsing("Price Details", |ui| {
            egui::Grid::new("scaled_prices_grid")
                .num_columns(2)
                .spacing([20.0, 4.0])
                .show(ui, |ui| {
                    ui.label(format!("Min ({}):", info.min));
                    ui.label(format!("{:.1}%", p_min * 100.0));
                    ui.end_row();

                    ui.label(format!("Max ({}):", info.max));
                    ui.label(format!("{:.1}%", p_max * 100.0));
                    ui.end_row();

                    ui.label("Abstain:");
                    ui.label(format!("{:.1}%", p_abstain * 100.0));
                    ui.end_row();
                });
        });

        ui.add_space(10.0);

        if is_trading && self.buy_shares.is_none() {
            ui.horizontal(|ui| {
                ui.label("Trade: ");

                if ui
                    .button(format!("Bet Lower (toward {})", info.min))
                    .clicked()
                {
                    self.buy_shares = Some(BuyShares::new(
                        market.id.clone(),
                        0,
                        format!("Lower ({})", info.min),
                        p_min,
                        market.trading_fee,
                    ));
                }

                if ui
                    .button(format!("Bet Higher (toward {})", info.max))
                    .clicked()
                {
                    self.buy_shares = Some(BuyShares::new(
                        market.id.clone(),
                        1,
                        format!("Higher ({})", info.max),
                        p_max,
                        market.trading_fee,
                    ));
                }
            });

            ui.add_space(5.0);
            ui.label(
                RichText::new(
                    "Betting lower increases the weight of the minimum bound.\n\
                     Betting higher increases the weight of the maximum bound.",
                )
                .small()
                .color(egui::Color32::GRAY),
            );
        }
    }
}

pub enum BuySharesResult {
    Pending,
    Cancelled,
    Completed,
}
