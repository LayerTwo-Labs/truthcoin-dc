use std::collections::{HashMap, HashSet};

use eframe::egui::{self, Button, Color32, RichText, ScrollArea, Vec2};
use truthcoin_dc::math::trading;
use truthcoin_dc::state::slots::SlotId;
use truthcoin_dc::state::{Market, MarketId, MarketState};
use truthcoin_dc::types::Address;

use crate::app::App;

#[derive(Clone, Default)]
struct MarketVotingStatus {
    slots_in_voting: usize,
    total_slots: usize,
    all_in_voting: bool,
}

const GREEN: Color32 = Color32::from_rgb(16, 185, 129);
const RED: Color32 = Color32::from_rgb(239, 68, 68);

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum TradeMode {
    #[default]
    Buy,
    Sell,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    #[default]
    Sats,
    Shares,
}

#[derive(Clone)]
pub struct TradePreview {
    pub shares: u64,
    pub total_cost_sats: u64,
    pub fee_sats: u64,
    pub base_amount_sats: u64,
    pub new_price: f64,
}

#[derive(Default)]
pub struct TradingPanelState {
    pub mode: TradeMode,
    pub input_mode: InputMode,
    pub amount_input: String,
    pub preview: Option<TradePreview>,
    pub is_processing: bool,
    pub tx_error: Option<String>,
    pub preview_error: Option<String>,
}

#[derive(Clone)]
struct UserPosition {
    outcome_index: usize,
    shares: i64,
    seller_address: Address,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum SortBy {
    #[default]
    Popular,
    Newest,
    MostActive,
    HighestLiquidity,
}

impl SortBy {
    fn label(&self) -> &'static str {
        match self {
            SortBy::Popular => "Popular",
            SortBy::Newest => "Newest",
            SortBy::MostActive => "Most Active",
            SortBy::HighestLiquidity => "Liquidity",
        }
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    #[default]
    Grid,
    Detail,
}

#[derive(Default)]
pub struct Browse {
    markets: Vec<(Market, MarketState)>,
    filtered_markets: Vec<usize>,
    selected_market: Option<MarketId>,
    selected_outcome: Option<usize>,
    error: Option<String>,
    slot_questions: Vec<(SlotId, String)>,
    cached_decision_info: Option<CachedDecisionInfo>,
    sort_by: SortBy,
    selected_tag: Option<String>,
    all_tags: Vec<String>,
    search_query: String,
    show_resolved: bool,
    view_mode: ViewMode,
    trading_panel: TradingPanelState,
    user_positions: Vec<UserPosition>,
    cached_balance_sats: Option<u64>,
    market_voting_status: HashMap<MarketId, MarketVotingStatus>,
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
                self.collect_tags();
                self.refresh_voting_status(app);
                self.apply_filters_and_sort();
            }
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                tracing::error!("Failed to fetch markets: {e:#}");
            }
        }
    }

    fn refresh_voting_status(&mut self, app: &App) {
        self.market_voting_status.clear();

        for (market, _state) in &self.markets {
            let total_slots = market.decision_slots.len();
            if total_slots == 0 {
                continue;
            }

            let mut slots_in_voting = 0;
            for &slot_id in &market.decision_slots {
                if app.node.is_slot_in_voting(slot_id).unwrap_or(false) {
                    slots_in_voting += 1;
                }
            }

            if slots_in_voting > 0 {
                let all_in_voting = slots_in_voting == total_slots;
                self.market_voting_status.insert(
                    market.id.clone(),
                    MarketVotingStatus {
                        slots_in_voting,
                        total_slots,
                        all_in_voting,
                    },
                );
            }
        }
    }

    fn collect_tags(&mut self) {
        let mut tags_set: HashSet<String> = HashSet::new();
        for (market, _) in &self.markets {
            for tag in &market.tags {
                tags_set.insert(tag.clone());
            }
        }
        self.all_tags = tags_set.into_iter().collect();
        self.all_tags.sort();
    }

    fn apply_filters_and_sort(&mut self) {
        let mut indices: Vec<usize> = self
            .markets
            .iter()
            .enumerate()
            .filter(|(_, (market, state))| {
                if !self.show_resolved && *state != MarketState::Trading {
                    return false;
                }
                if let Some(tag) = &self.selected_tag
                    && !market.tags.contains(tag)
                {
                    return false;
                }
                if !self.search_query.is_empty() {
                    let query = self.search_query.to_lowercase();
                    if !market.title.to_lowercase().contains(&query)
                        && !market.description.to_lowercase().contains(&query)
                    {
                        return false;
                    }
                }
                true
            })
            .map(|(i, _)| i)
            .collect();

        indices.sort_by(|&a, &b| {
            let (market_a, _) = &self.markets[a];
            let (market_b, _) = &self.markets[b];
            match self.sort_by {
                SortBy::Popular => {
                    market_b.total_volume_sats.cmp(&market_a.total_volume_sats)
                }
                SortBy::Newest => {
                    market_b.created_at_height.cmp(&market_a.created_at_height)
                }
                SortBy::MostActive => market_b
                    .last_updated_height
                    .cmp(&market_a.last_updated_height),
                SortBy::HighestLiquidity => market_b
                    .b
                    .partial_cmp(&market_a.b)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });

        self.filtered_markets = indices;
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let Some(app) = app else {
            ui.label("No app connection available");
            return;
        };

        self.refresh_markets(app);

        match self.view_mode {
            ViewMode::Grid => self.show_grid_view(app, ui),
            ViewMode::Detail => self.show_detail_view(app, ui),
        }
    }

    fn show_grid_view(&mut self, app: &App, ui: &mut egui::Ui) {
        egui::SidePanel::left("browse_filters")
            .default_width(180.0)
            .resizable(true)
            .show_inside(ui, |ui| {
                self.show_filter_panel(ui);
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.search_query)
                        .hint_text("Search markets...")
                        .desired_width(200.0),
                );
                if ui.button("Clear").clicked() && !self.search_query.is_empty()
                {
                    self.search_query.clear();
                    self.apply_filters_and_sort();
                }

                ui.add_space(20.0);

                ui.label("Sort:");
                for sort_option in [
                    SortBy::Popular,
                    SortBy::Newest,
                    SortBy::MostActive,
                    SortBy::HighestLiquidity,
                ] {
                    let is_selected = self.sort_by == sort_option;
                    if ui
                        .selectable_label(is_selected, sort_option.label())
                        .clicked()
                    {
                        self.sort_by = sort_option;
                        self.apply_filters_and_sort();
                    }
                }

                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        ui.checkbox(&mut self.show_resolved, "Show Resolved");
                        if ui.button("Refresh").clicked() {
                            self.refresh_markets(app);
                        }
                    },
                );
            });

            ui.separator();

            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!(
                        "{} markets",
                        self.filtered_markets.len()
                    ))
                    .weak(),
                );
                if (self.selected_tag.is_some()
                    || !self.search_query.is_empty())
                    && ui.small_button("Clear filters").clicked()
                {
                    self.selected_tag = None;
                    self.search_query.clear();
                    self.apply_filters_and_sort();
                }
            });

            ui.add_space(10.0);

            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, err);
            }

            ScrollArea::vertical().show(ui, |ui| {
                if self.filtered_markets.is_empty() {
                    ui.centered_and_justified(|ui| {
                        ui.label("No markets found");
                    });
                } else {
                    self.show_market_cards(app, ui);
                }
            });
        });
    }

    fn show_filter_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Filters");
        ui.separator();

        ui.add_space(5.0);
        ui.label(RichText::new("Tags").strong());
        ui.add_space(5.0);

        if ui
            .selectable_label(self.selected_tag.is_none(), "All Markets")
            .clicked()
        {
            self.selected_tag = None;
            self.apply_filters_and_sort();
        }

        ui.add_space(5.0);

        ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
            for tag in self.all_tags.clone() {
                let is_selected = self.selected_tag.as_ref() == Some(&tag);
                let count = self
                    .markets
                    .iter()
                    .filter(|(m, _)| m.tags.contains(&tag))
                    .count();
                let label = format!("{tag} ({count})");

                if ui.selectable_label(is_selected, label).clicked() {
                    if is_selected {
                        self.selected_tag = None;
                    } else {
                        self.selected_tag = Some(tag);
                    }
                    self.apply_filters_and_sort();
                }
            }
        });

        if self.all_tags.is_empty() {
            ui.label(RichText::new("No tags available").weak().italics());
        }
    }

    fn show_market_cards(&mut self, app: &App, ui: &mut egui::Ui) {
        let card_width = 300.0;
        let card_height = 180.0;
        let spacing = 12.0;

        let mut clicked_market: Option<MarketId> = None;

        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(spacing, spacing);

            for &market_idx in &self.filtered_markets.clone() {
                let (market, state) = &self.markets[market_idx];

                if let Some(id) = self.show_market_card(
                    app,
                    market,
                    state,
                    ui,
                    card_width,
                    card_height,
                ) {
                    clicked_market = Some(id);
                }
            }
        });

        if let Some(market_id) = clicked_market {
            self.selected_market = Some(market_id);
            self.selected_outcome = None;
            self.trading_panel = TradingPanelState::default();
            self.user_positions.clear();
            self.cached_balance_sats = None;
            self.slot_questions.clear();
            self.cached_decision_info = None;
            self.view_mode = ViewMode::Detail;
        }
    }

    fn show_market_card(
        &self,
        app: &App,
        market: &Market,
        state: &MarketState,
        ui: &mut egui::Ui,
        card_width: f32,
        card_height: f32,
    ) -> Option<MarketId> {
        let mut clicked = false;

        let (card_rect, response) = ui.allocate_exact_size(
            Vec2::new(card_width, card_height),
            egui::Sense::click(),
        );

        let bg_color = if response.hovered() {
            ui.visuals().widgets.hovered.bg_fill
        } else {
            ui.visuals().widgets.noninteractive.bg_fill
        };

        ui.painter().rect(
            card_rect,
            8.0,
            bg_color,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Outside,
        );

        let padding = 10.0;
        let content_rect = card_rect.shrink(padding);

        let valid_combos = market.get_valid_state_combos();
        let prices = market.calculate_prices_for_display();
        let n_outcomes = valid_combos.len();
        let liquidity_sats =
            trading::calculate_lmsr_liquidity(market.b, n_outcomes);
        let liquidity_btc = liquidity_sats / 100_000_000.0;
        let volume_btc = market.total_volume_sats as f64 / 100_000_000.0;

        let voting_status = self.market_voting_status.get(&market.id);
        let has_voting_slots = voting_status
            .map(|vs| vs.slots_in_voting > 0)
            .unwrap_or(false);
        let all_slots_voting =
            voting_status.map(|vs| vs.all_in_voting).unwrap_or(false);

        let (state_text, state_color) = match state {
            MarketState::Trading => {
                if all_slots_voting {
                    ("VOTING", egui::Color32::from_rgb(255, 165, 0))
                } else {
                    ("LIVE", egui::Color32::from_rgb(0, 200, 83))
                }
            }
            MarketState::Ossified => {
                ("RESOLVED", egui::Color32::from_rgb(100, 149, 237))
            }
            MarketState::Cancelled => {
                ("CANCELLED", egui::Color32::from_rgb(220, 20, 60))
            }
            MarketState::Invalid => ("INVALID", egui::Color32::GRAY),
        };

        let max_outcomes = 2.min(n_outcomes);
        let mut outcome_data: Vec<(f64, String)> = Vec::new();
        for i in 0..max_outcomes {
            let price = prices.get(i).copied().unwrap_or(0.0);
            let label = self.get_outcome_label(app, market, i, &valid_combos);
            let truncated = if label.len() > 30 {
                format!("{}...", &label[..27])
            } else {
                label
            };
            outcome_data.push((price, truncated));
        }

        let painter = ui.painter_at(card_rect);
        let font_title = egui::FontId::proportional(13.0);
        let font_small = egui::FontId::proportional(11.0);
        let text_color = ui.visuals().text_color();
        let weak_color = ui.visuals().weak_text_color();

        let mut y = content_rect.top();
        let x = content_rect.left();
        let line_height = 16.0;

        painter.text(
            egui::pos2(x, y),
            egui::Align2::LEFT_TOP,
            &market.title,
            font_title.clone(),
            text_color,
        );
        y += line_height + 4.0;

        let state_galley = painter.layout_no_wrap(
            state_text.to_string(),
            font_small.clone(),
            state_color,
        );
        painter.galley(egui::pos2(x, y), state_galley.clone(), text_color);

        let mut state_offset = state_galley.rect.width() + 5.0;

        if has_voting_slots && !all_slots_voting {
            let voting_text = if let Some(vs) = voting_status {
                format!("({}/{} VOTING)", vs.slots_in_voting, vs.total_slots)
            } else {
                "IN VOTING".to_string()
            };
            let voting_color = egui::Color32::from_rgb(255, 165, 0);
            painter.text(
                egui::pos2(x + state_offset, y),
                egui::Align2::LEFT_TOP,
                &voting_text,
                font_small.clone(),
                voting_color,
            );
            state_offset += 80.0; // Account for voting text width
        }

        painter.text(
            egui::pos2(x + state_offset, y),
            egui::Align2::LEFT_TOP,
            format!("| {volume_btc:.4} BTC vol"),
            font_small.clone(),
            weak_color,
        );
        y += line_height + 6.0;

        for (price, label) in &outcome_data {
            let price_color = if *price > 0.5 {
                egui::Color32::from_rgb(0, 180, 80)
            } else {
                egui::Color32::from_rgb(120, 120, 200)
            };

            painter.text(
                egui::pos2(x, y),
                egui::Align2::LEFT_TOP,
                format!("{:>3.0}%", price * 100.0),
                font_small.clone(),
                price_color,
            );

            painter.text(
                egui::pos2(x + 35.0, y),
                egui::Align2::LEFT_TOP,
                label,
                font_small.clone(),
                text_color,
            );
            y += line_height;
        }

        if n_outcomes > max_outcomes {
            painter.text(
                egui::pos2(x, y),
                egui::Align2::LEFT_TOP,
                format!("+{} more", n_outcomes - max_outcomes),
                font_small.clone(),
                weak_color,
            );
            y += line_height;
        }

        y += 4.0;

        if !market.tags.is_empty() {
            let tags_text: String = market
                .tags
                .iter()
                .take(2)
                .map(|t| format!("#{t}"))
                .collect::<Vec<_>>()
                .join(" ");
            painter.text(
                egui::pos2(x, y),
                egui::Align2::LEFT_TOP,
                tags_text,
                font_small.clone(),
                egui::Color32::from_rgb(100, 149, 237),
            );
        }

        let bottom_y = content_rect.bottom() - line_height;
        painter.text(
            egui::pos2(x, bottom_y),
            egui::Align2::LEFT_TOP,
            format!("{liquidity_btc:.4} BTC liq | {n_outcomes} outcomes"),
            font_small.clone(),
            weak_color,
        );

        let market_id_hex = market.id.to_string();
        let market_id_short = if market_id_hex.len() > 8 {
            format!("{}...", &market_id_hex[..8])
        } else {
            market_id_hex
        };
        painter.text(
            egui::pos2(content_rect.right(), bottom_y),
            egui::Align2::RIGHT_TOP,
            market_id_short,
            font_small,
            egui::Color32::from_rgb(80, 80, 80),
        );

        if response.clicked() {
            clicked = true;
        }

        if response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }

        if clicked {
            Some(market.id.clone())
        } else {
            None
        }
    }

    fn get_outcome_label(
        &self,
        app: &App,
        market: &Market,
        outcome_display_idx: usize,
        valid_combos: &[(usize, &Vec<usize>)],
    ) -> String {
        let Some((_, combo)) = valid_combos.get(outcome_display_idx) else {
            return format!("Outcome {outcome_display_idx}");
        };

        // Categorical market: multiple slots, one "yes" per combo
        let is_categorical = market.decision_slots.len() > 1
            && valid_combos
                .iter()
                .all(|(_, c)| c.iter().filter(|&&v| v == 1).count() <= 1);

        if is_categorical {
            if let Some(slot_idx) = combo.iter().position(|&v| v == 1) {
                if let Some(slot_id) = market.decision_slots.get(slot_idx)
                    && let Ok(Some(slot)) = app.node.get_slot(*slot_id)
                    && let Some(decision) = &slot.decision
                {
                    return decision.question.clone();
                }
                return format!("Option {}", slot_idx + 1);
            } else {
                return market
                    .residual_names
                    .as_ref()
                    .and_then(|names| names.first().cloned())
                    .unwrap_or_else(|| "Other".to_string());
            }
        }

        if market.decision_slots.len() == 1 {
            match combo.first() {
                Some(0) => return "No".to_string(),
                Some(1) => return "Yes".to_string(),
                _ => {}
            }
        }

        combo
            .iter()
            .enumerate()
            .map(|(dim, &val)| {
                let val_str = match val {
                    0 => "No",
                    1 => "Yes",
                    _ => "?",
                };
                format!("D{}: {}", dim + 1, val_str)
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn show_detail_view(&mut self, app: &App, ui: &mut egui::Ui) {
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

        let Some((market, state)) = selected_data else {
            ui.centered_and_justified(|ui| {
                ui.label("Market not found");
            });
            return;
        };

        self.load_user_positions(app, &market);
        self.load_balance(app);

        // Clone to avoid borrow issues in closures
        let voting_status = self.market_voting_status.get(&market.id).cloned();
        let has_voting_slots = voting_status
            .as_ref()
            .map(|vs| vs.slots_in_voting > 0)
            .unwrap_or(false);
        let all_slots_voting = voting_status
            .as_ref()
            .map(|vs| vs.all_in_voting)
            .unwrap_or(false);

        let is_trading = matches!(state, MarketState::Trading);
        let trading_enabled = is_trading && !all_slots_voting;

        egui::SidePanel::right("trading_panel")
            .default_width(340.0)
            .min_width(280.0)
            .resizable(true)
            .show_inside(ui, |ui| {
                if trading_enabled {
                    if has_voting_slots && !all_slots_voting
                        && let Some(ref vs) = voting_status
                    {
                        ui.horizontal(|ui| {
                            let warning_color = Color32::from_rgb(255, 165, 0);
                            ui.colored_label(
                                warning_color,
                                format!(
                                    "{} of {} slots in voting - trading still available",
                                    vs.slots_in_voting, vs.total_slots
                                ),
                            );
                        });
                        ui.add_space(10.0);
                    }
                    self.show_trading_panel(app, &market, ui);
                } else {
                    ui.vertical_centered(|ui| {
                        ui.add_space(20.0);
                        if all_slots_voting {
                            ui.label(
                                RichText::new("VOTING")
                                    .size(18.0)
                                    .color(Color32::from_rgb(255, 165, 0)),
                            );
                            ui.add_space(10.0);
                            ui.label("Trading disabled - all decision slots are in voting.");
                        } else {
                            let (state_text, state_color) = match state {
                                MarketState::Ossified => {
                                    ("RESOLVED", Color32::from_rgb(100, 149, 237))
                                }
                                MarketState::Cancelled => ("CANCELLED", RED),
                                MarketState::Invalid => ("INVALID", Color32::GRAY),
                                MarketState::Trading => ("", Color32::GRAY),
                            };
                            ui.label(
                                RichText::new(state_text)
                                    .size(18.0)
                                    .color(state_color),
                            );
                            ui.add_space(10.0);
                            ui.label("Trading is not available for this market.");
                        }
                    });
                }
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("← Back to Markets").clicked() {
                    self.view_mode = ViewMode::Grid;
                    self.selected_market = None;
                    self.selected_outcome = None;
                    self.trading_panel = TradingPanelState::default();
                    self.user_positions.clear();
                    self.cached_balance_sats = None;
                }
            });
            ui.add_space(5.0);

            ScrollArea::vertical().show(ui, |ui| {
                self.show_market_info(app, &market, &state, ui);
            });
        });
    }

    fn load_user_positions(&mut self, app: &App, market: &Market) {
        if !self.user_positions.is_empty() {
            return;
        }

        let addresses = match app.wallet.get_addresses() {
            Ok(addrs) => addrs,
            Err(_) => return,
        };

        for address in addresses {
            if let Ok(user_positions) =
                app.node.get_user_share_positions(&address)
            {
                for (market_id, outcome_idx, shares) in user_positions {
                    if market_id == market.id && shares > 0 {
                        self.user_positions.push(UserPosition {
                            outcome_index: outcome_idx as usize,
                            shares,
                            seller_address: address,
                        });
                    }
                }
            }
        }
    }

    fn load_balance(&mut self, app: &App) {
        if self.cached_balance_sats.is_some() {
            return;
        }

        let addresses = match app.wallet.get_addresses() {
            Ok(addrs) => addrs,
            Err(_) => return,
        };

        let (utxos, spent_in_mempool) =
            match app.node.get_utxos_with_mempool_status(&addresses) {
                Ok(result) => result,
                Err(_) => return,
            };

        use truthcoin_dc::types::GetBitcoinValue;
        let confirmed_total: bitcoin::Amount =
            utxos.values().map(|utxo| utxo.get_bitcoin_value()).sum();

        let pending_spent: bitcoin::Amount = spent_in_mempool
            .iter()
            .filter_map(|(outpoint, _)| {
                utxos.get(outpoint).map(|utxo| utxo.get_bitcoin_value())
            })
            .sum();

        let available = confirmed_total
            .checked_sub(pending_spent)
            .unwrap_or(bitcoin::Amount::ZERO);

        self.cached_balance_sats = Some(available.to_sat());
    }

    fn show_market_info(
        &mut self,
        app: &App,
        market: &Market,
        state: &MarketState,
        ui: &mut egui::Ui,
    ) {
        ui.heading(&market.title);
        ui.add_space(5.0);

        ui.horizontal(|ui| {
            if !market.tags.is_empty() {
                for tag in &market.tags {
                    ui.label(
                        RichText::new(format!("#{tag}"))
                            .color(Color32::from_rgb(100, 149, 237)),
                    );
                }
            } else {
                ui.label(RichText::new("No tags").weak().italics());
            }
        });

        ui.add_space(10.0);

        ui.horizontal(|ui| {
            let voting_status = self.market_voting_status.get(&market.id);
            let has_voting_slots = voting_status
                .map(|vs| vs.slots_in_voting > 0)
                .unwrap_or(false);
            let all_slots_voting =
                voting_status.map(|vs| vs.all_in_voting).unwrap_or(false);

            let (state_text, state_color) = match state {
                MarketState::Trading => {
                    if all_slots_voting {
                        ("VOTING", Color32::from_rgb(255, 165, 0))
                    } else {
                        ("LIVE", GREEN)
                    }
                }
                MarketState::Ossified => {
                    ("RESOLVED", Color32::from_rgb(100, 149, 237))
                }
                MarketState::Cancelled => ("CANCELLED", RED),
                MarketState::Invalid => ("INVALID", Color32::GRAY),
            };
            egui::Frame::new()
                .fill(state_color.gamma_multiply(0.2))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(8, 4))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(state_text).strong().color(state_color),
                    );
                });

            if has_voting_slots && !all_slots_voting {
                ui.add_space(5.0);
                let voting_color = Color32::from_rgb(255, 165, 0);
                egui::Frame::new()
                    .fill(voting_color.gamma_multiply(0.2))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::symmetric(8, 4))
                    .show(ui, |ui| {
                        if let Some(vs) = voting_status {
                            ui.label(
                                RichText::new(format!(
                                    "{}/{} VOTING",
                                    vs.slots_in_voting, vs.total_slots
                                ))
                                .strong()
                                .color(voting_color),
                            );
                        }
                    });
            }

            ui.add_space(15.0);

            let volume_btc = market.total_volume_sats as f64 / 100_000_000.0;
            ui.vertical(|ui| {
                ui.label(RichText::new("Volume").small().weak());
                ui.label(
                    RichText::new(format!("{volume_btc:.6} BTC")).strong(),
                );
            });

            ui.add_space(15.0);

            let n_outcomes = market.get_valid_state_combos().len();
            let liquidity_sats =
                trading::calculate_lmsr_liquidity(market.b, n_outcomes);
            let liquidity_btc = liquidity_sats / 100_000_000.0;
            ui.vertical(|ui| {
                ui.label(RichText::new("Liquidity").small().weak());
                ui.label(
                    RichText::new(format!("{liquidity_btc:.6} BTC")).strong(),
                );
            });

            ui.add_space(15.0);

            ui.vertical(|ui| {
                ui.label(RichText::new("Fee").small().weak());
                ui.label(
                    RichText::new(format!(
                        "{:.1}%",
                        market.trading_fee * 100.0
                    ))
                    .strong(),
                );
            });
        });

        ui.add_space(10.0);

        ui.horizontal(|ui| {
            let market_id_hex = market.id.to_string();
            ui.label(
                RichText::new(format!("ID: {market_id_hex}"))
                    .small()
                    .weak()
                    .monospace(),
            );
            if ui.small_button("Copy").clicked() {
                ui.ctx().copy_text(market_id_hex);
            }
        });

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(10.0);

        if !market.description.is_empty() {
            ui.collapsing("Description", |ui| {
                ui.label(&market.description);
            });
            ui.add_space(10.0);
        }

        ui.label(RichText::new("Outcomes").strong().size(16.0));
        ui.add_space(8.0);

        let valid_combos = market.get_valid_state_combos();
        let valid_state_indices: Vec<usize> =
            valid_combos.iter().map(|(idx, _)| *idx).collect();
        let prices = if let Ok(Some(mempool_shares)) =
            app.node.get_mempool_shares(&market.id)
        {
            trading::calculate_display_prices(
                &mempool_shares,
                market.b(),
                &valid_state_indices,
            )
        } else {
            market.calculate_prices_for_display()
        };

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

        if is_scaled {
            self.show_scaled_market_info(&prices, ui);
        } else {
            self.show_outcomes_with_probability_bars(
                app,
                market,
                &prices,
                &valid_combos,
                ui,
            );
        }
    }

    fn show_outcomes_with_probability_bars(
        &mut self,
        app: &App,
        market: &Market,
        prices: &[f64],
        valid_combos: &[(usize, &Vec<usize>)],
        ui: &mut egui::Ui,
    ) {
        for (i, (_, combo)) in valid_combos.iter().enumerate() {
            let price = prices.get(i).copied().unwrap_or(0.0);
            let outcome_label =
                self.get_outcome_label(app, market, i, valid_combos);

            let is_selected = self.selected_outcome == Some(i);

            let is_positive = outcome_label.to_lowercase().contains("yes")
                || outcome_label.to_lowercase().contains("higher")
                || i == 1; // Second outcome is typically "Yes" in binary markets
            let bar_color = if is_positive { GREEN } else { RED };

            let frame = egui::Frame::new()
                .fill(if is_selected {
                    bar_color.gamma_multiply(0.15)
                } else {
                    ui.visuals().widgets.noninteractive.bg_fill
                })
                .stroke(if is_selected {
                    egui::Stroke::new(2.0, bar_color)
                } else {
                    ui.visuals().widgets.noninteractive.bg_stroke
                })
                .corner_radius(8.0)
                .inner_margin(egui::Margin::symmetric(12, 10));

            let response = frame.show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new(&outcome_label).strong());
                        if combo.len() > 1 {
                            let combo_str = combo
                                .iter()
                                .enumerate()
                                .map(|(d, &v)| {
                                    let slot_name = self
                                        .slot_questions
                                        .get(d)
                                        .map(|(_, q)| {
                                            if q.len() > 15 {
                                                format!("{}...", &q[..12])
                                            } else {
                                                q.clone()
                                            }
                                        })
                                        .unwrap_or_else(|| {
                                            format!("D{}", d + 1)
                                        });
                                    format!(
                                        "{}: {}",
                                        slot_name,
                                        if v == 1 { "Yes" } else { "No" }
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join(" | ");
                            ui.label(RichText::new(combo_str).small().weak());
                        }
                    });

                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            let price_pct = price * 100.0;

                            let bar_width = 60.0;
                            let bar_height = 8.0;
                            let (bar_rect, _) = ui.allocate_exact_size(
                                Vec2::new(bar_width, bar_height),
                                egui::Sense::hover(),
                            );

                            ui.painter().rect_filled(
                                bar_rect,
                                4.0,
                                Color32::from_gray(60),
                            );

                            let filled_width = bar_width * price as f32;
                            let filled_rect = egui::Rect::from_min_size(
                                bar_rect.min,
                                Vec2::new(filled_width, bar_height),
                            );
                            ui.painter().rect_filled(
                                filled_rect,
                                4.0,
                                bar_color,
                            );

                            ui.add_space(8.0);

                            ui.label(
                                RichText::new(format!("{price_pct:.1}%"))
                                    .size(16.0)
                                    .strong()
                                    .color(bar_color),
                            );
                        },
                    );
                });
            });

            if response.response.interact(egui::Sense::click()).clicked() {
                self.selected_outcome = Some(i);
                self.trading_panel.amount_input.clear();
                self.trading_panel.preview = None;
                self.trading_panel.tx_error = None;
                self.trading_panel.preview_error = None;
            }

            if response.response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }

            ui.add_space(4.0);
        }
    }

    fn show_scaled_market_info(&mut self, prices: &[f64], ui: &mut egui::Ui) {
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

        ui.label(RichText::new("Scaled Decision Market").italics().weak());
        ui.add_space(10.0);

        egui::Frame::new()
            .fill(ui.visuals().widgets.noninteractive.bg_fill)
            .corner_radius(8.0)
            .inner_margin(16.0)
            .show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new("Market Estimate").strong());
                    ui.label(
                        RichText::new(format!("{implied_value:.2}"))
                            .size(36.0)
                            .strong()
                            .color(GREEN),
                    );

                    ui.add_space(10.0);

                    let slider_width = ui.available_width().min(300.0);
                    let slider_height = 20.0;
                    let (slider_rect, _) = ui.allocate_exact_size(
                        Vec2::new(slider_width, slider_height),
                        egui::Sense::hover(),
                    );

                    ui.painter().rect_filled(
                        slider_rect,
                        4.0,
                        Color32::from_gray(50),
                    );

                    let segments = 20;
                    for i in 0..segments {
                        let t = i as f32 / segments as f32;
                        let color = Color32::from_rgb(
                            (239.0 * (1.0 - t) + 16.0 * t) as u8,
                            (68.0 * (1.0 - t) + 185.0 * t) as u8,
                            (68.0 * (1.0 - t) + 129.0 * t) as u8,
                        );
                        let seg_width = slider_width / segments as f32;
                        let seg_rect = egui::Rect::from_min_size(
                            egui::pos2(
                                slider_rect.min.x + i as f32 * seg_width,
                                slider_rect.min.y,
                            ),
                            Vec2::new(seg_width + 1.0, slider_height),
                        );
                        let corner_radius = if i == 0 || i == segments - 1 {
                            4.0
                        } else {
                            0.0
                        };
                        ui.painter().rect_filled(
                            seg_rect,
                            corner_radius,
                            color.gamma_multiply(0.6),
                        );
                    }

                    let marker_x = slider_rect.min.x
                        + (normalized_value as f32) * slider_width;
                    let marker_rect = egui::Rect::from_center_size(
                        egui::pos2(marker_x, slider_rect.center().y),
                        Vec2::new(4.0, slider_height + 4.0),
                    );
                    ui.painter().rect_filled(marker_rect, 2.0, Color32::WHITE);

                    ui.add_space(8.0);

                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!("{}", info.min)).color(RED),
                        );
                        ui.add_space(slider_width - 60.0);
                        ui.label(
                            RichText::new(format!("{}", info.max)).color(GREEN),
                        );
                    });
                });
            });

        ui.add_space(10.0);

        ui.collapsing("Price Details", |ui| {
            egui::Grid::new("scaled_prices")
                .num_columns(2)
                .spacing([20.0, 4.0])
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(format!("Lower ({}):", info.min))
                            .color(RED),
                    );
                    ui.label(format!("{:.1}%", p_min * 100.0));
                    ui.end_row();

                    ui.label(
                        RichText::new(format!("Higher ({}):", info.max))
                            .color(GREEN),
                    );
                    ui.label(format!("{:.1}%", p_max * 100.0));
                    ui.end_row();

                    ui.label("Abstain:");
                    ui.label(format!("{:.1}%", p_abstain * 100.0));
                    ui.end_row();
                });
        });

        ui.add_space(10.0);

        ui.label(RichText::new("Select Position").strong());
        ui.add_space(5.0);

        let lower_selected = self.selected_outcome == Some(0);
        let lower_frame = egui::Frame::new()
            .fill(if lower_selected {
                RED.gamma_multiply(0.15)
            } else {
                ui.visuals().widgets.noninteractive.bg_fill
            })
            .stroke(if lower_selected {
                egui::Stroke::new(2.0, RED)
            } else {
                ui.visuals().widgets.noninteractive.bg_stroke
            })
            .corner_radius(8.0)
            .inner_margin(egui::Margin::symmetric(12, 8));

        let lower_response = lower_frame.show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("Lower (toward {})", info.min))
                        .strong(),
                );
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        ui.label(
                            RichText::new(format!("{:.1}%", p_min * 100.0))
                                .strong()
                                .color(RED),
                        );
                    },
                );
            });
        });

        if lower_response
            .response
            .interact(egui::Sense::click())
            .clicked()
        {
            self.selected_outcome = Some(0);
            self.trading_panel.amount_input.clear();
            self.trading_panel.preview = None;
        }
        if lower_response.response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }

        ui.add_space(4.0);

        let higher_selected = self.selected_outcome == Some(1);
        let higher_frame = egui::Frame::new()
            .fill(if higher_selected {
                GREEN.gamma_multiply(0.15)
            } else {
                ui.visuals().widgets.noninteractive.bg_fill
            })
            .stroke(if higher_selected {
                egui::Stroke::new(2.0, GREEN)
            } else {
                ui.visuals().widgets.noninteractive.bg_stroke
            })
            .corner_radius(8.0)
            .inner_margin(egui::Margin::symmetric(12, 8));

        let higher_response = higher_frame.show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("Higher (toward {})", info.max))
                        .strong(),
                );
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        ui.label(
                            RichText::new(format!("{:.1}%", p_max * 100.0))
                                .strong()
                                .color(GREEN),
                        );
                    },
                );
            });
        });

        if higher_response
            .response
            .interact(egui::Sense::click())
            .clicked()
        {
            self.selected_outcome = Some(1);
            self.trading_panel.amount_input.clear();
            self.trading_panel.preview = None;
        }
        if higher_response.response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
    }

    fn show_trading_panel(
        &mut self,
        app: &App,
        market: &Market,
        ui: &mut egui::Ui,
    ) {
        ui.vertical(|ui| {
            ui.add_space(5.0);

            ui.horizontal(|ui| {
                let buy_selected = self.trading_panel.mode == TradeMode::Buy;
                let sell_selected = self.trading_panel.mode == TradeMode::Sell;

                let buy_btn = Button::new(RichText::new("Buy").strong())
                    .fill(if buy_selected {
                        GREEN.gamma_multiply(0.3)
                    } else {
                        Color32::TRANSPARENT
                    })
                    .stroke(if buy_selected {
                        egui::Stroke::new(2.0, GREEN)
                    } else {
                        egui::Stroke::NONE
                    })
                    .min_size(Vec2::new(
                        ui.available_width() / 2.0 - 5.0,
                        32.0,
                    ));

                if ui.add(buy_btn).clicked() {
                    self.trading_panel.mode = TradeMode::Buy;
                    self.trading_panel.amount_input.clear();
                    self.trading_panel.preview = None;
                    self.trading_panel.tx_error = None;
                }

                let sell_btn = Button::new(RichText::new("Sell").strong())
                    .fill(if sell_selected {
                        RED.gamma_multiply(0.3)
                    } else {
                        Color32::TRANSPARENT
                    })
                    .stroke(if sell_selected {
                        egui::Stroke::new(2.0, RED)
                    } else {
                        egui::Stroke::NONE
                    })
                    .min_size(Vec2::new(ui.available_width() - 5.0, 32.0));

                if ui.add(sell_btn).clicked() {
                    self.trading_panel.mode = TradeMode::Sell;
                    self.trading_panel.amount_input.clear();
                    self.trading_panel.preview = None;
                    self.trading_panel.tx_error = None;
                }
            });

            ui.add_space(15.0);

            let valid_combos = market.get_valid_state_combos();
            if let Some(outcome_idx) = self.selected_outcome {
                let outcome_label = self.get_outcome_label(
                    app,
                    market,
                    outcome_idx,
                    &valid_combos,
                );
                let prices = market.calculate_prices_for_display();
                let current_price =
                    prices.get(outcome_idx).copied().unwrap_or(0.0);

                ui.label(RichText::new("Selected Outcome").small().weak());
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&outcome_label).strong());
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            ui.label(
                                RichText::new(format!(
                                    "{:.1}%",
                                    current_price * 100.0
                                ))
                                .strong(),
                            );
                        },
                    );
                });
            } else {
                ui.label(
                    RichText::new("Select an outcome to trade")
                        .weak()
                        .italics(),
                );
                return;
            }

            ui.add_space(15.0);

            let mode_color = if self.trading_panel.mode == TradeMode::Buy {
                GREEN
            } else {
                RED
            };

            if self.trading_panel.mode == TradeMode::Buy {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Input:").small().weak());
                    if ui
                        .selectable_label(
                            self.trading_panel.input_mode == InputMode::Sats,
                            "Sats",
                        )
                        .clicked()
                    {
                        self.trading_panel.input_mode = InputMode::Sats;
                        self.trading_panel.amount_input.clear();
                        self.trading_panel.preview = None;
                    }
                    if ui
                        .selectable_label(
                            self.trading_panel.input_mode == InputMode::Shares,
                            "Shares",
                        )
                        .clicked()
                    {
                        self.trading_panel.input_mode = InputMode::Shares;
                        self.trading_panel.amount_input.clear();
                        self.trading_panel.preview = None;
                    }
                });
                ui.add_space(5.0);
            }

            let (amount_label, hint_text) = if self.trading_panel.mode
                == TradeMode::Buy
            {
                match self.trading_panel.input_mode {
                    InputMode::Sats => ("Amount (sats)", "Enter sats to spend"),
                    InputMode::Shares => {
                        ("Shares to buy", "Enter whole shares")
                    }
                }
            } else {
                ("Shares to sell", "Enter whole shares")
            };

            ui.label(RichText::new(amount_label).small().weak());
            let old_input = self.trading_panel.amount_input.clone();
            ui.add(
                egui::TextEdit::singleline(
                    &mut self.trading_panel.amount_input,
                )
                .hint_text(hint_text)
                .desired_width(ui.available_width()),
            );

            if self.trading_panel.amount_input != old_input {
                self.calculate_trade_preview(app, market);
            }

            ui.add_space(8.0);

            if self.trading_panel.mode == TradeMode::Buy {
                match self.trading_panel.input_mode {
                    InputMode::Sats => {
                        ui.horizontal_wrapped(|ui| {
                            for (label, amount) in [
                                ("1m", 1_000_000u64),
                                ("5m", 5_000_000),
                                ("10m", 10_000_000),
                                ("25m", 25_000_000),
                                ("50m", 50_000_000),
                                ("100m", 100_000_000),
                            ] {
                                if ui.small_button(label).clicked() {
                                    self.trading_panel.amount_input =
                                        amount.to_string();
                                    self.calculate_trade_preview(app, market);
                                }
                            }
                        });
                    }
                    InputMode::Shares => {
                        ui.horizontal_wrapped(|ui| {
                            for shares in [1u64, 5, 10, 25, 50, 100] {
                                if ui.small_button(shares.to_string()).clicked()
                                {
                                    self.trading_panel.amount_input =
                                        shares.to_string();
                                    self.calculate_trade_preview(app, market);
                                }
                            }
                        });
                    }
                }
            } else if let Some(outcome_idx) = self.selected_outcome {
                let actual_idx =
                    valid_combos.get(outcome_idx).map(|(idx, _)| *idx);
                if let Some(actual_idx) = actual_idx {
                    let position_shares = self
                        .user_positions
                        .iter()
                        .find(|p| p.outcome_index == actual_idx)
                        .map(|p| p.shares as u64);

                    if let Some(shares) = position_shares {
                        let mut clicked_max = false;
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("Your shares: {shares}"))
                                    .small(),
                            );
                            if ui.small_button("Max").clicked() {
                                clicked_max = true;
                            }
                        });
                        if clicked_max {
                            self.trading_panel.amount_input =
                                shares.to_string();
                            self.calculate_trade_preview(app, market);
                        }
                    } else {
                        ui.label(
                            RichText::new("You have no shares in this outcome")
                                .small()
                                .weak(),
                        );
                    }
                }
            }

            ui.add_space(10.0);

            if let Some(err) = &self.trading_panel.preview_error {
                ui.colored_label(RED, err);
            }

            if let Some(preview) = &self.trading_panel.preview {
                egui::Frame::new()
                    .fill(ui.visuals().widgets.noninteractive.bg_fill)
                    .corner_radius(6.0)
                    .inner_margin(10.0)
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new("Trade Preview").small().strong(),
                        );
                        ui.add_space(5.0);

                        egui::Grid::new("preview_grid")
                            .num_columns(2)
                            .spacing([10.0, 4.0])
                            .show(ui, |ui| {
                                ui.label("Shares:");
                                ui.label(format!("{}", preview.shares));
                                ui.end_row();

                                if self.trading_panel.mode == TradeMode::Buy {
                                    ui.label("Cost:");
                                    ui.label(format!(
                                        "{} sats",
                                        preview.base_amount_sats
                                    ));
                                    ui.end_row();
                                } else {
                                    ui.label("Proceeds:");
                                    ui.label(format!(
                                        "{} sats",
                                        preview.base_amount_sats
                                    ));
                                    ui.end_row();
                                }

                                ui.label("Fee:");
                                ui.label(format!("{} sats", preview.fee_sats));
                                ui.end_row();

                                ui.label(RichText::new("Total:").strong());
                                ui.label(
                                    RichText::new(format!(
                                        "{} sats",
                                        preview.total_cost_sats
                                    ))
                                    .strong(),
                                );
                                ui.end_row();

                                ui.label("New price:");
                                ui.label(format!(
                                    "{:.1}%",
                                    preview.new_price * 100.0
                                ));
                                ui.end_row();
                            });
                    });
            }

            ui.add_space(10.0);

            if let Some(err) = &self.trading_panel.tx_error {
                ui.colored_label(RED, err);
                ui.add_space(5.0);
            }

            let can_trade = self.trading_panel.preview.is_some()
                && !self.trading_panel.is_processing
                && self.trading_panel.tx_error.is_none();

            let button_text = if self.trading_panel.is_processing {
                "Processing..."
            } else if self.trading_panel.mode == TradeMode::Buy {
                "Place Buy Order"
            } else {
                "Place Sell Order"
            };

            let order_btn = Button::new(RichText::new(button_text).strong())
                .fill(if can_trade {
                    mode_color
                } else {
                    Color32::from_gray(60)
                })
                .min_size(Vec2::new(ui.available_width(), 40.0));

            if ui.add_enabled(can_trade, order_btn).clicked() {
                self.execute_trade(app, market);
            }

            ui.add_space(15.0);

            ui.separator();
            ui.add_space(5.0);

            if let Some(balance_sats) = self.cached_balance_sats {
                let balance_btc = balance_sats as f64 / 100_000_000.0;
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Available:").small().weak());
                    ui.label(
                        RichText::new(format!("{balance_btc:.8} BTC")).small(),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("").small()); // spacer
                    ui.label(
                        RichText::new(format!("({balance_sats} sats)"))
                            .small()
                            .weak(),
                    );
                });
            }
        });
    }

    fn calculate_trade_preview(&mut self, app: &App, market: &Market) {
        self.trading_panel.preview = None;
        self.trading_panel.preview_error = None;

        let Some(outcome_idx) = self.selected_outcome else {
            return;
        };

        let valid_combos = market.get_valid_state_combos();
        let actual_idx = match valid_combos.get(outcome_idx) {
            Some((idx, _)) => *idx,
            None => {
                self.trading_panel.preview_error =
                    Some("Invalid outcome".to_string());
                return;
            }
        };

        let prices = market.calculate_prices_for_display();
        let current_price = prices.get(outcome_idx).copied().unwrap_or(0.0);

        if self.trading_panel.mode == TradeMode::Buy {
            self.calculate_buy_preview(app, market, actual_idx, current_price);
        } else {
            self.calculate_sell_preview(app, market, actual_idx, current_price);
        }
    }

    fn calculate_buy_preview(
        &mut self,
        _app: &App,
        market: &Market,
        actual_idx: usize,
        current_price: f64,
    ) {
        let calc_cost = |shares: u64| -> Option<trading::BuyCost> {
            let mut new_shares = market.shares.clone();
            new_shares[actual_idx] += shares as i64;
            let base_cost = market.query_update_cost(new_shares).ok()?;
            trading::calculate_buy_cost(base_cost, market.trading_fee).ok()
        };

        let (shares, buy_cost) = match self.trading_panel.input_mode {
            InputMode::Sats => {
                let budget_sats: u64 =
                    match self.trading_panel.amount_input.parse() {
                        Ok(s) if s > 0 => s,
                        Ok(_) => {
                            self.trading_panel.preview_error =
                                Some("Amount must be positive".to_string());
                            return;
                        }
                        Err(_) => return,
                    };

                // Binary search for max whole shares affordable within budget
                let mut low: u64 = 0;
                let mut high: u64 = budget_sats;
                let mut last_valid_cost: Option<trading::BuyCost> = None;

                while low < high {
                    let mid = (low + high).div_ceil(2);

                    if let Some(cost) = calc_cost(mid) {
                        if cost.total_cost_sats <= budget_sats {
                            low = mid;
                            last_valid_cost = Some(cost);
                        } else {
                            high = mid - 1;
                        }
                    } else {
                        high = mid - 1;
                    }
                }

                if low == 0 {
                    self.trading_panel.preview_error =
                        Some("Cannot afford any whole shares".to_string());
                    return;
                }

                let cost =
                    last_valid_cost.unwrap_or_else(|| calc_cost(low).unwrap());
                (low, cost)
            }
            InputMode::Shares => {
                let shares: u64 = match self.trading_panel.amount_input.parse()
                {
                    Ok(s) if s > 0 => s,
                    Ok(_) => {
                        self.trading_panel.preview_error =
                            Some("Shares must be positive".to_string());
                        return;
                    }
                    Err(_) => return,
                };

                let cost = match calc_cost(shares) {
                    Some(c) => c,
                    None => {
                        self.trading_panel.preview_error =
                            Some("Cost calculation error".to_string());
                        return;
                    }
                };
                (shares, cost)
            }
        };

        let mut new_shares = market.shares.clone();
        new_shares[actual_idx] += shares as i64;
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
        let new_price = if new_price > 0.0 {
            new_price
        } else {
            current_price
        };

        self.trading_panel.preview = Some(TradePreview {
            shares,
            total_cost_sats: buy_cost.total_cost_sats,
            fee_sats: buy_cost.trading_fee_sats,
            base_amount_sats: buy_cost.base_cost_sats,
            new_price,
        });
    }

    fn calculate_sell_preview(
        &mut self,
        _app: &App,
        market: &Market,
        actual_idx: usize,
        current_price: f64,
    ) {
        let shares: u64 = match self.trading_panel.amount_input.parse() {
            Ok(s) if s > 0 => s,
            Ok(_) => {
                self.trading_panel.preview_error =
                    Some("Shares must be positive".to_string());
                return;
            }
            Err(_) => return,
        };

        let user_shares = self
            .user_positions
            .iter()
            .find(|p| p.outcome_index == actual_idx)
            .map(|p| p.shares as u64)
            .unwrap_or(0);

        if shares > user_shares {
            self.trading_panel.preview_error =
                Some(format!("Max shares: {user_shares}"));
            return;
        }

        let mut new_shares = market.shares.clone();
        new_shares[actual_idx] -= shares as i64;

        if new_shares[actual_idx] < 0 {
            self.trading_panel.preview_error =
                Some("Would result in negative shares".to_string());
            return;
        }

        let old_cost =
            match trading::calculate_treasury(&market.shares, market.b()) {
                Ok(c) => c,
                Err(e) => {
                    self.trading_panel.preview_error =
                        Some(format!("Cost error: {e:?}"));
                    return;
                }
            };

        let new_cost =
            match trading::calculate_treasury(&new_shares, market.b()) {
                Ok(c) => c,
                Err(e) => {
                    self.trading_panel.preview_error =
                        Some(format!("Cost error: {e:?}"));
                    return;
                }
            };

        let gross_proceeds_f64 = old_cost - new_cost;
        if gross_proceeds_f64 < 0.0 {
            self.trading_panel.preview_error =
                Some("Invalid proceeds".to_string());
            return;
        }

        let sell_proceeds = match trading::calculate_sell_proceeds(
            gross_proceeds_f64,
            market.trading_fee,
        ) {
            Ok(p) => p,
            Err(e) => {
                self.trading_panel.preview_error =
                    Some(format!("Proceeds error: {e}"));
                return;
            }
        };

        if sell_proceeds.trading_fee_sats >= sell_proceeds.gross_proceeds_sats {
            self.trading_panel.preview_error =
                Some("Trade too small: fee exceeds proceeds".to_string());
            return;
        }

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
        let new_price = if new_price > 0.0 {
            new_price
        } else {
            current_price
        };

        self.trading_panel.preview = Some(TradePreview {
            shares,
            total_cost_sats: sell_proceeds.net_proceeds_sats,
            fee_sats: sell_proceeds.trading_fee_sats,
            base_amount_sats: sell_proceeds.gross_proceeds_sats,
            new_price,
        });
    }

    fn execute_trade(&mut self, app: &App, market: &Market) {
        let Some(preview) = self.trading_panel.preview.clone() else {
            return;
        };
        let Some(outcome_idx) = self.selected_outcome else {
            return;
        };

        let valid_combos = market.get_valid_state_combos();
        let actual_idx = match valid_combos.get(outcome_idx) {
            Some((idx, _)) => *idx,
            None => {
                self.trading_panel.tx_error =
                    Some("Invalid outcome".to_string());
                return;
            }
        };

        self.trading_panel.is_processing = true;

        if self.trading_panel.mode == TradeMode::Buy {
            self.execute_buy(app, market, actual_idx, &preview);
        } else {
            self.execute_sell(app, market, actual_idx, &preview);
        }
    }

    fn execute_buy(
        &mut self,
        app: &App,
        market: &Market,
        actual_idx: usize,
        preview: &TradePreview,
    ) {
        let trader = match app.wallet.get_addresses() {
            Ok(addrs) => match addrs.into_iter().next() {
                Some(addr) => addr,
                None => {
                    self.trading_panel.tx_error =
                        Some("Wallet has no addresses".to_string());
                    self.trading_panel.is_processing = false;
                    return;
                }
            },
            Err(e) => {
                self.trading_panel.tx_error =
                    Some(format!("Wallet error: {e:#}"));
                self.trading_panel.is_processing = false;
                return;
            }
        };

        // 2% slippage buffer for price movement between preview and execution
        let slippage_buffer = (preview.total_cost_sats / 50).max(1);
        let limit_sats = preview.total_cost_sats
            + trading::TRADE_MINER_FEE_SATS
            + slippage_buffer;
        match app.wallet.trade(
            market.id.clone(),
            actual_idx,
            preview.shares as i64,
            trader,
            limit_sats,
        ) {
            Ok(tx) => {
                if let Err(e) = app.sign_and_send(tx) {
                    self.trading_panel.tx_error =
                        Some(format!("Failed to send: {e:#}"));
                    self.trading_panel.is_processing = false;
                } else {
                    tracing::info!(
                        "Bought {} shares in market {:?}",
                        preview.shares,
                        market.id
                    );
                    self.trading_panel.is_processing = false;
                    self.trading_panel.amount_input.clear();
                    self.trading_panel.preview = None;
                    self.user_positions.clear(); // Force reload
                    self.cached_balance_sats = None;
                }
            }
            Err(e) => {
                self.trading_panel.tx_error =
                    Some(format!("Failed to create tx: {e:#}"));
                self.trading_panel.is_processing = false;
            }
        }
    }

    fn execute_sell(
        &mut self,
        app: &App,
        market: &Market,
        actual_idx: usize,
        preview: &TradePreview,
    ) {
        let seller_address = match self
            .user_positions
            .iter()
            .find(|p| p.outcome_index == actual_idx)
        {
            Some(pos) => pos.seller_address,
            None => {
                self.trading_panel.tx_error =
                    Some("No position found".to_string());
                self.trading_panel.is_processing = false;
                return;
            }
        };

        match app.wallet.trade(
            market.id.clone(),
            actual_idx,
            -(preview.shares as i64),
            seller_address,
            preview.total_cost_sats,
        ) {
            Ok(tx) => {
                if let Err(e) = app.sign_and_send(tx) {
                    self.trading_panel.tx_error =
                        Some(format!("Failed to send: {e:#}"));
                    self.trading_panel.is_processing = false;
                } else {
                    tracing::info!(
                        "Sold {} shares in market {:?}",
                        preview.shares,
                        market.id
                    );
                    self.trading_panel.is_processing = false;
                    self.trading_panel.amount_input.clear();
                    self.trading_panel.preview = None;
                    self.user_positions.clear(); // Force reload
                    self.cached_balance_sats = None;
                }
            }
            Err(e) => {
                self.trading_panel.tx_error =
                    Some(format!("Failed to create tx: {e:#}"));
                self.trading_panel.is_processing = false;
            }
        }
    }
}
