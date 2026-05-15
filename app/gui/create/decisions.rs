use std::collections::BTreeSet;
use std::pin::Pin;
use std::task::Poll;
use std::time::Duration;

use eframe::egui::{self, Button, Color32, RichText, ScrollArea};
use futures::Stream;
use truthcoin_dc::math::decisions::{
    SLOTS_PER_TIER_PER_MINT, TIER_COUNT, fee_for_index, slot_price,
    slot_unlocked,
};
use truthcoin_dc::state::decisions::{
    DecisionEntry, DecisionId, DecisionType, PeriodPricing,
};
use truthcoin_dc::types::DecisionClaimEntry;
use truthcoin_dc::wallet::DecisionClaimInput;

use crate::app::App;
use crate::util::PromiseStream;

type NodeUpdates = PromiseStream<Pin<Box<dyn Stream<Item = ()> + Send>>>;

pub struct Decisions {
    current_period: u32,
    periods: Vec<(u32, u64)>,
    selected_period: Option<u32>,
    available_decisions: Vec<DecisionId>,
    claimed_decisions: Vec<DecisionEntry>,
    pending_claim_ids: BTreeSet<DecisionId>,
    pricing: Option<PeriodPricing>,
    claim: Claim,
    error: Option<String>,
    success: Option<String>,
    is_blocks_mode: bool,
    node_updated: Option<NodeUpdates>,
    needs_initial_load: bool,
}

impl Default for Decisions {
    fn default() -> Self {
        Self {
            current_period: 0,
            periods: Vec::new(),
            selected_period: None,
            available_decisions: Vec::new(),
            claimed_decisions: Vec::new(),
            pending_claim_ids: BTreeSet::new(),
            pricing: None,
            claim: Claim::default(),
            error: None,
            success: None,
            is_blocks_mode: true,
            node_updated: None,
            needs_initial_load: true,
        }
    }
}

struct Claim {
    header: String,
    description: String,
    r#type: DecisionType,
    min_input: String,
    max_input: String,
    increment_input: String,
    option_0_label: String,
    option_1_label: String,
    outcomes: Vec<String>,
    tags_input: String,
    tx_fee_sats_input: String,
    max_listing_fee_sats_input: String,
    is_processing: bool,
}

impl Default for Claim {
    fn default() -> Self {
        Self {
            header: String::new(),
            description: String::new(),
            r#type: DecisionType::Binary,
            min_input: String::new(),
            max_input: String::new(),
            increment_input: String::new(),
            option_0_label: String::new(),
            option_1_label: String::new(),
            outcomes: vec![String::new(), String::new()],
            tags_input: String::new(),
            tx_fee_sats_input: "1000".to_string(),
            max_listing_fee_sats_input: String::new(),
            is_processing: false,
        }
    }
}

impl Decisions {
    fn ensure_subscribed(&mut self, app: &App) {
        if self.node_updated.is_none() {
            let stream: Pin<Box<dyn Stream<Item = ()> + Send>> =
                app.node.watch();
            self.node_updated =
                Some(PromiseStream::new(stream, app.runtime.handle().clone()));
        }
    }

    fn tip_changed(&mut self) -> bool {
        let Some(stream) = self.node_updated.as_mut() else {
            return false;
        };
        let mut changed = false;
        while let Some(Poll::Ready(())) = stream.poll_next() {
            changed = true;
        }
        changed
    }

    fn refresh_data(&mut self, app: &App) {
        self.ensure_subscribed(app);
        if !(self.needs_initial_load || self.tip_changed()) {
            return;
        }
        self.needs_initial_load = false;

        let rotxn = match app.node.read_txn() {
            Ok(t) => t,
            Err(e) => {
                self.error = Some(format!("Failed to read state: {e:#}"));
                return;
            }
        };
        let state = app.node.state();
        let config = state.decisions().get_config();
        self.is_blocks_mode = config.is_blocks_mode();

        let mainchain_ts = state
            .try_get_mainchain_timestamp(&rotxn)
            .unwrap_or(None)
            .unwrap_or(0);
        let genesis_ts = state
            .try_get_genesis_timestamp(&rotxn)
            .unwrap_or(None)
            .unwrap_or(0);
        let height = state.try_get_height(&rotxn).unwrap_or(None);
        let current_period = match state.decisions().get_current_period(
            mainchain_ts,
            height,
            genesis_ts,
        ) {
            Ok(p) => p,
            Err(e) => {
                self.error =
                    Some(format!("Failed to get current period: {e:#}"));
                return;
            }
        };
        let period_changed = self.current_period != current_period;
        self.current_period = current_period;
        if period_changed || self.selected_period.is_none() {
            self.selected_period = Some(current_period);
        }

        match state.get_all_decision_periods(&rotxn) {
            Ok(periods) => {
                let current = self.current_period;
                self.periods = periods
                    .into_iter()
                    .filter(|(p, _)| *p >= current)
                    .collect();
                self.error = None;

                if let Some(selected) = self.selected_period {
                    let is_valid =
                        self.periods.iter().any(|(p, _)| *p == selected);
                    if !is_valid {
                        self.selected_period = Some(self.current_period);
                    }
                }
            }
            Err(e) => {
                self.error = Some(format!("Failed to load periods: {e:#}"));
                tracing::error!("Failed to load decision periods: {e:#}");
            }
        }

        if let Some(period) = self.selected_period {
            self.load_period(app, period);
        }
    }

    fn load_period(&mut self, app: &App, period: u32) {
        let rotxn = match app.node.read_txn() {
            Ok(t) => t,
            Err(e) => {
                self.error = Some(format!("Failed to read state: {e:#}"));
                return;
            }
        };
        let state = app.node.state();
        match state.get_available_decisions_in_period(&rotxn, period) {
            Ok(available) => self.available_decisions = available,
            Err(e) => {
                self.error =
                    Some(format!("Failed to load available decisions: {e:#}"));
                tracing::error!("Failed to load available decisions: {e:#}");
            }
        }
        match state
            .decisions()
            .get_claimed_decisions_in_period(&rotxn, period)
        {
            Ok(claimed) => self.claimed_decisions = claimed,
            Err(e) => {
                self.error =
                    Some(format!("Failed to load claimed decisions: {e:#}"));
                tracing::error!("Failed to load claimed decisions: {e:#}");
            }
        }
        self.pricing = state
            .decisions()
            .get_listing_fee_info(&rotxn, period)
            .ok()
            .flatten();
        match app.node.get_pending_decision_claim_ids() {
            Ok(ids) => {
                self.pending_claim_ids = ids
                    .into_iter()
                    .filter_map(|bytes| DecisionId::from_bytes(bytes).ok())
                    .filter(|id| {
                        id.is_standard() && id.period_index() == period
                    })
                    .collect();
            }
            Err(e) => {
                tracing::warn!("Failed to load pending claims: {e:#}");
            }
        }
    }

    fn next_target(&self) -> Option<(DecisionId, u64)> {
        let pricing = self.pricing.as_ref()?;
        let id = self.available_decisions.iter().copied().find(|id| {
            slot_unlocked(id.decision_index(), pricing.mints)
                && !self.pending_claim_ids.contains(id)
        })?;
        let fee =
            fee_for_index(pricing.p_period, pricing.mints, id.decision_index())
                .ok()?;
        Some((id, fee))
    }

    fn claimed_index_set(&self) -> BTreeSet<u32> {
        self.claimed_decisions
            .iter()
            .filter(|e| e.decision_id.is_standard())
            .map(|e| e.decision_id.decision_index())
            .collect()
    }

    fn pending_index_set(&self) -> BTreeSet<u32> {
        self.pending_claim_ids
            .iter()
            .map(|id| id.decision_index())
            .collect()
    }

    fn claim_decision(&mut self, app: &App) {
        self.error = None;
        self.success = None;

        let Some((decision_id, listing_fee)) = self.next_target() else {
            self.error = Some("No available slots in this period".to_string());
            return;
        };

        if self.claim.header.trim().is_empty() {
            self.error = Some("Question is required".to_string());
            return;
        }

        let (decision_type, option_0_label, option_1_label, option_labels) =
            match &self.claim.r#type {
                DecisionType::Binary => {
                    let opt0 = self.claim.option_0_label.trim().to_string();
                    let opt1 = self.claim.option_1_label.trim().to_string();
                    (
                        DecisionType::Binary,
                        (!opt0.is_empty()).then_some(opt0),
                        (!opt1.is_empty()).then_some(opt1),
                        None,
                    )
                }
                DecisionType::Scaled { .. } => {
                    let min = match self.claim.min_input.parse::<f64>() {
                        Ok(v) => v,
                        Err(_) => {
                            self.error =
                                Some("Min must be a number".to_string());
                            return;
                        }
                    };
                    let max = match self.claim.max_input.parse::<f64>() {
                        Ok(v) => v,
                        Err(_) => {
                            self.error =
                                Some("Max must be a number".to_string());
                            return;
                        }
                    };
                    let increment = if self.claim.increment_input.is_empty() {
                        1.0
                    } else {
                        match self.claim.increment_input.parse::<f64>() {
                            Ok(v) => v,
                            Err(_) => {
                                self.error = Some(
                                    "Increment must be a number".to_string(),
                                );
                                return;
                            }
                        }
                    };
                    (
                        DecisionType::Scaled {
                            min,
                            max,
                            increment,
                        },
                        None,
                        None,
                        None,
                    )
                }
                DecisionType::Category { .. } => {
                    let trimmed: Vec<String> = self
                        .claim
                        .outcomes
                        .iter()
                        .map(|o| o.trim().to_string())
                        .collect();
                    if trimmed.len() < 2 {
                        self.error = Some(
                            "Categorical decisions require at least 2 outcomes"
                                .to_string(),
                        );
                        return;
                    }
                    if let Some((i, _)) =
                        trimmed.iter().enumerate().find(|(_, s)| s.is_empty())
                    {
                        self.error =
                            Some(format!("Outcome {} label is empty", i + 1));
                        return;
                    }
                    (
                        DecisionType::Category {
                            options: trimmed.clone(),
                        },
                        None,
                        None,
                        Some(trimmed),
                    )
                }
            };

        let tags: Vec<String> = self
            .claim
            .tags_input
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();

        if tags.len() > 10 {
            self.error = Some("Maximum 10 tags allowed".to_string());
            return;
        }
        if let Some(bad) = tags.iter().find(|t| t.len() > 50) {
            self.error = Some(format!("Tag '{bad}' exceeds 50 byte limit"));
            return;
        }

        let tx_fee_sats =
            match self.claim.tx_fee_sats_input.trim().parse::<u64>() {
                Ok(v) => v,
                Err(_) => {
                    self.error =
                        Some("Invalid miner fee: must be a number".to_string());
                    return;
                }
            };

        let trimmed_max = self.claim.max_listing_fee_sats_input.trim();
        let max_listing_fee_sats: Option<u64> = if trimmed_max.is_empty() {
            None
        } else {
            match trimmed_max.parse::<u64>() {
                Ok(v) => Some(v),
                Err(_) => {
                    self.error = Some(
                        "Invalid max listing fee: must be a number".to_string(),
                    );
                    return;
                }
            }
        };

        if let Some(cap) = max_listing_fee_sats
            && listing_fee > cap
        {
            self.error =
                Some(format!("Listing fee {listing_fee} exceeds max ({cap})"));
            return;
        }

        let entry = DecisionClaimEntry {
            decision_id_bytes: decision_id.as_bytes(),
            header: self.claim.header.clone(),
            description: self.claim.description.clone(),
            option_0_label,
            option_1_label,
            option_labels,
            tags: if tags.is_empty() { None } else { Some(tags) },
        };

        let input = DecisionClaimInput {
            decision_type,
            decisions: vec![entry],
        };

        let total_fee =
            bitcoin::Amount::from_sat(listing_fee.saturating_add(tx_fee_sats));

        match app.wallet.claim_decision(input, total_fee) {
            Ok(tx) => {
                if let Err(e) = app.sign_and_send(tx) {
                    self.error = Some(format!("Failed to send: {e:#}"));
                    tracing::error!("Decision claim failed: {e:#}");
                } else {
                    self.success = Some(format!(
                        "Decision claimed in period {} \
                         (listing fee paid: {listing_fee} sats)",
                        decision_id.period_index()
                    ));
                    tracing::info!(
                        "Decision claimed: {} (listing_fee={listing_fee})",
                        decision_id.to_hex()
                    );
                    self.claim = Claim::default();
                    if let Some(period) = self.selected_period {
                        self.load_period(app, period);
                    }
                }
            }
            Err(e) => {
                self.error = Some(format!("Failed to create claim tx: {e:#}"));
                tracing::error!("Decision claim tx creation failed: {e:#}");
            }
        }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let Some(app) = app else {
            ui.label("No app connection available");
            return;
        };

        self.refresh_data(app);

        if !self.is_blocks_mode {
            ui.ctx().request_repaint_after(Duration::from_secs(1));
        }

        ui.horizontal(|ui| {
            ui.heading("Create Decision");
            ui.label(
                RichText::new(format!(
                    "(Current: Period {})",
                    self.current_period
                ))
                .weak()
                .italics(),
            );
            if ui.button("Refresh").clicked() {
                self.needs_initial_load = true;
                self.refresh_data(app);
            }
        });
        ui.separator();

        if let Some(msg) = &self.success {
            ui.colored_label(Color32::GREEN, msg);
        }
        if let Some(err) = &self.error {
            ui.colored_label(Color32::RED, err);
        }

        ui.add_space(5.0);

        ui.horizontal(|ui| {
            ui.label("Period:");
            let prev_selection = self.selected_period;
            egui::ComboBox::from_id_salt("period_selector")
                .selected_text(
                    self.selected_period
                        .map(|p| {
                            if p == self.current_period {
                                format!("Period {p} (current)")
                            } else {
                                format!("Period {p}")
                            }
                        })
                        .unwrap_or_else(|| "Select a period".to_string()),
                )
                .show_ui(ui, |ui| {
                    for (period, available) in &self.periods {
                        let is_current = *period == self.current_period;
                        let label = if is_current {
                            format!(
                                "Period {period} ({available} available) \u{2190} current"
                            )
                        } else {
                            format!("Period {period} ({available} available)")
                        };
                        ui.selectable_value(
                            &mut self.selected_period,
                            Some(*period),
                            label,
                        );
                    }
                });
            if self.selected_period != prev_selection
                && let Some(period) = self.selected_period
            {
                self.load_period(app, period);
            }
        });

        ui.add_space(10.0);

        if self.selected_period.is_none() {
            ui.centered_and_justified(|ui| {
                ui.label("Select a period to continue");
            });
            return;
        }

        ScrollArea::vertical()
            .id_salt("decisions_body")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.show_period_grid(ui);
                ui.add_space(15.0);
                ui.separator();
                self.show_claim(app, ui);
                ui.add_space(15.0);
                ui.separator();
                self.show_existing(ui);
            });
    }

    fn show_period_grid(&self, ui: &mut egui::Ui) {
        let Some(period) = self.selected_period else {
            return;
        };
        let Some(pricing) = self.pricing.as_ref() else {
            ui.label("No pricing data for this period");
            return;
        };

        let cells_per_tier = (pricing.mints * SLOTS_PER_TIER_PER_MINT) as u32;
        let claimed = self.claimed_index_set();
        let pending = self.pending_index_set();
        let next_idx = self.next_target().map(|(id, _)| id.decision_index());

        let max_label_width = (0..TIER_COUNT)
            .map(|t| format_with_commas(slot_price(pricing.p_period, t)).len())
            .max()
            .unwrap_or(0);

        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("Period {period}")).strong());
                ui.label(
                    RichText::new(format!(
                        "p_period {} sats   mints {}/20",
                        format_with_commas(pricing.p_period),
                        pricing.mints,
                    ))
                    .weak()
                    .small(),
                );
            });
            ui.add_space(4.0);

            ScrollArea::horizontal()
                .id_salt("tier_grid_scroll")
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for tier in 0..TIER_COUNT as u32 {
                        let price = slot_price(pricing.p_period, tier as usize);
                        let mut claimed_in_tier = 0u64;
                        ui.horizontal(|ui| {
                            let price_text = format_with_commas(price);
                            let pad = max_label_width
                                .saturating_sub(price_text.len());
                            ui.monospace(format!(
                                "{}{} sats",
                                " ".repeat(pad),
                                price_text,
                            ));
                            ui.add_space(8.0);
                            for pos in 0..cells_per_tier {
                                let idx = tier * 100 + pos;
                                let is_claimed = claimed.contains(&idx);
                                let is_pending = pending.contains(&idx);
                                let is_next = next_idx == Some(idx);
                                if is_claimed || is_pending {
                                    claimed_in_tier += 1;
                                }
                                let (ch, color) = if is_claimed {
                                    ('\u{2588}', Color32::from_rgb(220, 80, 80))
                                } else if is_pending {
                                    (
                                        '\u{2588}',
                                        Color32::from_rgb(230, 200, 80),
                                    )
                                } else if is_next {
                                    (
                                        '\u{2588}',
                                        Color32::from_rgb(110, 200, 130),
                                    )
                                } else {
                                    ('\u{2591}', Color32::from_gray(80))
                                };
                                ui.label(
                                    RichText::new(ch).monospace().color(color),
                                );
                            }
                            ui.add_space(8.0);
                            ui.label(
                                RichText::new(format!(
                                    "{claimed_in_tier}/{cells_per_tier}"
                                ))
                                .weak()
                                .small(),
                            );
                        });
                    }
                });

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!(
                        "Period total: {} / {}",
                        pricing.claimed,
                        pricing.period_capacity(),
                    ))
                    .weak(),
                );
            });

            ui.add_space(4.0);
            match self.next_target() {
                Some((_, fee)) => {
                    ui.label(
                        RichText::new(format!(
                            "Current Slot \u{2014} {} sats",
                            format_with_commas(fee),
                        ))
                        .color(Color32::from_rgb(110, 200, 130)),
                    );
                }
                None => {
                    ui.colored_label(
                        Color32::YELLOW,
                        "No slots available in this period",
                    );
                }
            }
        });
    }

    fn show_claim(&mut self, app: &App, ui: &mut egui::Ui) {
        egui::Grid::new("claim")
            .num_columns(2)
            .spacing([10.0, 8.0])
            .show(ui, |ui| {
                ui.label("Question:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.claim.header)
                        .hint_text("Short title (max 100 bytes)")
                        .desired_width(360.0),
                );
                ui.end_row();

                ui.label("Description:");
                ui.add(
                    egui::TextEdit::multiline(&mut self.claim.description)
                        .hint_text("Detailed description (max 2000 bytes)")
                        .desired_width(360.0)
                        .desired_rows(3),
                );
                ui.end_row();

                ui.label("Type:");
                ui.horizontal(|ui| {
                    let prev_type = self.claim.r#type.clone();
                    if ui
                        .selectable_label(
                            matches!(self.claim.r#type, DecisionType::Binary),
                            "Binary",
                        )
                        .clicked()
                    {
                        self.claim.r#type = DecisionType::Binary;
                    }
                    if ui
                        .selectable_label(
                            matches!(
                                self.claim.r#type,
                                DecisionType::Scaled { .. }
                            ),
                            "Scaled",
                        )
                        .clicked()
                    {
                        self.claim.r#type = DecisionType::Scaled {
                            min: 0.0,
                            max: 0.0,
                            increment: 1.0,
                        };
                    }
                    if ui
                        .selectable_label(
                            matches!(
                                self.claim.r#type,
                                DecisionType::Category { .. }
                            ),
                            "Categorical",
                        )
                        .clicked()
                    {
                        self.claim.r#type = DecisionType::Category {
                            options: Vec::new(),
                        };
                    }
                    if std::mem::discriminant(&prev_type)
                        != std::mem::discriminant(&self.claim.r#type)
                    {
                        match prev_type {
                            DecisionType::Binary => {
                                self.claim.option_0_label.clear();
                                self.claim.option_1_label.clear();
                            }
                            DecisionType::Scaled { .. } => {
                                self.claim.min_input.clear();
                                self.claim.max_input.clear();
                                self.claim.increment_input.clear();
                            }
                            DecisionType::Category { .. } => {
                                self.claim.outcomes =
                                    vec![String::new(), String::new()];
                            }
                        }
                    }
                });
                ui.end_row();

                match &self.claim.r#type {
                    DecisionType::Binary => {
                        ui.label("Option 0 label:");
                        ui.add(
                            egui::TextEdit::singleline(
                                &mut self.claim.option_0_label,
                            )
                            .hint_text("e.g., False (default: No)")
                            .desired_width(220.0),
                        );
                        ui.end_row();

                        ui.label("Option 1 label:");
                        ui.add(
                            egui::TextEdit::singleline(
                                &mut self.claim.option_1_label,
                            )
                            .hint_text("e.g., True (default: Yes)")
                            .desired_width(220.0),
                        );
                        ui.end_row();
                    }
                    DecisionType::Scaled { .. } => {
                        ui.label("Scale Range:");
                        ui.horizontal(|ui| {
                            ui.label("Min:");
                            ui.add(
                                egui::TextEdit::singleline(
                                    &mut self.claim.min_input,
                                )
                                .hint_text("e.g., 0")
                                .desired_width(80.0),
                            );
                            ui.label("Max:");
                            ui.add(
                                egui::TextEdit::singleline(
                                    &mut self.claim.max_input,
                                )
                                .hint_text("e.g., 100")
                                .desired_width(80.0),
                            );
                        });
                        ui.end_row();

                        ui.label("Increment:");
                        ui.add(
                            egui::TextEdit::singleline(
                                &mut self.claim.increment_input,
                            )
                            .hint_text("e.g., 0.5 (default: 1)")
                            .desired_width(80.0),
                        );
                        ui.end_row();
                    }
                    DecisionType::Category { .. } => {
                        let mut remove_idx: Option<usize> = None;
                        let can_remove = self.claim.outcomes.len() > 2;
                        for (idx, outcome) in
                            self.claim.outcomes.iter_mut().enumerate()
                        {
                            ui.label(format!("Outcome {}:", idx + 1));
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(outcome)
                                        .id_salt(format!("claim_outcome_{idx}"))
                                        .hint_text(format!(
                                            "e.g., Option {}",
                                            idx + 1
                                        ))
                                        .desired_width(220.0),
                                );
                                if ui
                                    .add_enabled(
                                        can_remove,
                                        Button::new("\u{274C}").small(),
                                    )
                                    .clicked()
                                {
                                    remove_idx = Some(idx);
                                }
                            });
                            ui.end_row();
                        }
                        if let Some(idx) = remove_idx {
                            self.claim.outcomes.remove(idx);
                        }
                        ui.label("");
                        if ui.button("\u{2795} Add outcome").clicked() {
                            self.claim.outcomes.push(String::new());
                        }
                        ui.end_row();
                    }
                }

                ui.label("Tags:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.claim.tags_input)
                        .hint_text(
                            "Comma-separated (max 10 tags, 50 bytes each)",
                        )
                        .desired_width(360.0),
                );
                ui.end_row();

                ui.label("Miner fee (sats):");
                ui.add(
                    egui::TextEdit::singleline(
                        &mut self.claim.tx_fee_sats_input,
                    )
                    .hint_text("e.g. 1000")
                    .desired_width(120.0),
                );
                ui.end_row();
            });

        ui.collapsing("Advanced", |ui| {
            ui.horizontal(|ui| {
                ui.label("Max listing fee (sats):");
                ui.add(
                    egui::TextEdit::singleline(
                        &mut self.claim.max_listing_fee_sats_input,
                    )
                    .hint_text("blank = accept current")
                    .desired_width(160.0),
                );
            });
        });

        ui.add_space(10.0);

        let categorical_ok =
            !matches!(self.claim.r#type, DecisionType::Category { .. })
                || (self.claim.outcomes.len() >= 2
                    && self
                        .claim
                        .outcomes
                        .iter()
                        .all(|o| !o.trim().is_empty()));

        let has_slot = self.next_target().is_some();
        let can_claim = !self.claim.header.trim().is_empty()
            && categorical_ok
            && has_slot
            && !self.claim.is_processing;

        if ui
            .add_enabled(can_claim, Button::new("Create Decision"))
            .clicked()
        {
            self.claim.is_processing = true;
            self.claim_decision(app);
            self.claim.is_processing = false;
        }
    }

    fn show_existing(&self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new(format!(
            "Existing decisions in this period ({})",
            self.claimed_decisions.len()
        ))
        .id_salt("existing_decisions")
        .show(ui, |ui| {
            if self.claimed_decisions.is_empty() {
                ui.label(
                    RichText::new("None yet \u{2014} be the first")
                        .weak()
                        .italics(),
                );
                return;
            }
            ScrollArea::vertical()
                .id_salt("existing_decisions_scroll")
                .max_height(240.0)
                .show(ui, |ui| {
                    for entry in &self.claimed_decisions {
                        let entry_hex =
                            hex::encode(entry.decision_id.as_bytes());
                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                ui.monospace(&entry_hex);
                                if ui.small_button("Copy").clicked() {
                                    ui.ctx().copy_text(entry_hex.clone());
                                }
                            });
                            if let Some(decision) = &entry.decision {
                                ui.label(&decision.header);
                                ui.horizontal(|ui| {
                                    if decision.is_categorical() {
                                        let labels = decision
                                            .get_category_labels()
                                            .unwrap_or(&[]);
                                        let shown = if labels.len() > 4 {
                                            format!(
                                                "{}/\u{2026}",
                                                labels[..4].join("/")
                                            )
                                        } else {
                                            labels.join("/")
                                        };
                                        ui.label(
                                            RichText::new(format!(
                                                "Categorical [{shown}]"
                                            ))
                                            .small()
                                            .weak(),
                                        );
                                    } else if decision.is_scaled() {
                                        let range = format!(
                                            "Scaled [{}-{}]",
                                            decision.scale_min().unwrap_or(0.0),
                                            decision
                                                .scale_max()
                                                .unwrap_or(100.0),
                                        );
                                        ui.label(
                                            RichText::new(range).small().weak(),
                                        );
                                    } else {
                                        let (label0, label1) =
                                            decision.get_binary_labels();
                                        ui.label(
                                            RichText::new(format!(
                                                "Binary [{label0}/{label1}]"
                                            ))
                                            .small()
                                            .weak(),
                                        );
                                    }
                                });
                            }
                        });
                    }
                });
        });
    }
}

fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in bytes.iter().enumerate() {
        let pos_from_end = bytes.len() - i;
        if i > 0 && pos_from_end.is_multiple_of(3) {
            out.push(',');
        }
        out.push(*c as char);
    }
    out
}
