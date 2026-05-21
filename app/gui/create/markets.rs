use std::collections::{HashMap, HashSet};

use eframe::egui::{self, Button, RichText, ScrollArea};
use truthcoin_dc::state::decisions::DecisionType;
use truthcoin_dc::state::markets::DEFAULT_TRADING_FEE;
use truthcoin_dc::state::voting::types::VotingPeriodId;
use truthcoin_dc::types::ClaimDecisionPayload;
use truthcoin_dc::wallet::CreateMarketInput;
use truthcoin_dc_app_rpc_api::PeriodPricingSummary;

use crate::app::App;
use crate::rpc_server::{SlotRequest, allocate_decision_slots};

const DEFAULT_LIQUIDITY_SATS: u64 = 10_000;

const PLACEHOLDER_PREFIX: &str = "__new_";

struct PreviewRow {
    source: String,
    header: String,
    type_label: &'static str,
    cardinality: usize,
}

#[derive(Clone)]
struct ClaimedDecisionInfo {
    decision_id_hex: String,
    header: String,
    decision_type: DecisionType,
    is_categorical: bool,
    option_count: usize,
}

#[derive(Clone)]
struct PendingNewDecision {
    period_index: u32,
    r#type: Option<DecisionType>,
    header: String,
    description: String,
    option_0_label_input: String,
    option_1_label_input: String,
    option_labels_input: String,
    min_input: String,
    max_input: String,
    increment_input: String,
    tags_input: String,
}

impl Default for PendingNewDecision {
    fn default() -> Self {
        Self {
            period_index: 0,
            r#type: None,
            header: String::new(),
            description: String::new(),
            option_0_label_input: String::new(),
            option_1_label_input: String::new(),
            option_labels_input: String::new(),
            min_input: String::from("0"),
            max_input: String::from("1"),
            increment_input: String::from("1"),
            tags_input: String::new(),
        }
    }
}

impl PendingNewDecision {
    fn is_complete(&self) -> bool {
        match &self.r#type {
            None => false,
            Some(DecisionType::Binary) => true,
            Some(DecisionType::Scaled { .. }) => {
                self.min_input.trim().parse::<f64>().is_ok()
                    && self.max_input.trim().parse::<f64>().is_ok()
                    && (self.increment_input.trim().is_empty()
                        || self.increment_input.trim().parse::<f64>().is_ok())
            }
            Some(DecisionType::Category { .. }) => {
                self.option_labels_input
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .count()
                    >= 2
            }
        }
    }
}

pub struct CreateMarket {
    title: String,
    description: String,
    trading_fee_input: String,
    initial_liquidity_input: String,
    existing_dimension_ids: Vec<String>,
    claimed_decisions: Vec<ClaimedDecisionInfo>,
    decisions_loaded: bool,
    pending_new_decisions: Vec<PendingNewDecision>,
    open_periods: Vec<PeriodPricingSummary>,
    open_periods_loaded: bool,
    is_processing: bool,
    error: Option<String>,
    success_message: Option<String>,
    tx_pow_enabled: bool,
    tx_pow_hash_selector: u8,
    tx_pow_ordering_input: String,
    tx_pow_difficulty_input: String,
}

impl Default for CreateMarket {
    fn default() -> Self {
        Self {
            title: String::new(),
            description: String::new(),
            trading_fee_input: String::new(),
            initial_liquidity_input: DEFAULT_LIQUIDITY_SATS.to_string(),
            existing_dimension_ids: Vec::new(),
            claimed_decisions: Vec::new(),
            decisions_loaded: false,
            pending_new_decisions: vec![PendingNewDecision::default()],
            open_periods: Vec::new(),
            open_periods_loaded: false,
            is_processing: false,
            error: None,
            success_message: None,
            tx_pow_enabled: false,
            tx_pow_hash_selector: 0b0000_0001,
            tx_pow_ordering_input: String::from("0"),
            tx_pow_difficulty_input: String::from("8"),
        }
    }
}

impl CreateMarket {
    fn reset(&mut self) {
        self.title.clear();
        self.description.clear();
        self.trading_fee_input.clear();
        self.initial_liquidity_input = DEFAULT_LIQUIDITY_SATS.to_string();
        self.existing_dimension_ids.clear();
        self.pending_new_decisions = vec![PendingNewDecision::default()];
        self.is_processing = false;
        self.error = None;
        self.tx_pow_enabled = false;
        self.tx_pow_hash_selector = 0b0000_0001;
        self.tx_pow_ordering_input = String::from("0");
        self.tx_pow_difficulty_input = String::from("8");
    }

    fn load_open_periods(&mut self, app: &App) {
        self.open_periods.clear();

        let current_period = match app.node.get_current_period() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    "Failed to load current period for dropdown: {e:#}"
                );
                self.open_periods_loaded = true;
                return;
            }
        };

        use truthcoin_dc::math::decisions as math_decisions;

        for period in current_period..=current_period.saturating_add(19) {
            let pricing = match app.node.get_listing_fee_info(period) {
                Ok(Some(p)) => p,
                Ok(None) => continue,
                Err(e) => {
                    tracing::warn!(
                        "Failed to load pricing for period {period}: {e:#}"
                    );
                    continue;
                }
            };

            let available = match app
                .node
                .get_available_decisions_in_period(VotingPeriodId::new(period))
            {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        "Failed to load available decisions for period {period}: {e:#}"
                    );
                    continue;
                }
            };

            let mut slots_by_tier = [0u32; 5];
            let mut cheapest_tier: Option<u8> = None;
            let mut cheapest_fee: Option<u64> = None;
            for id in &available {
                if !math_decisions::slot_unlocked(
                    id.decision_index(),
                    pricing.mints,
                ) {
                    continue;
                }
                let tier = (id.decision_index() / 100) as usize;
                if tier >= 5 {
                    continue;
                }
                slots_by_tier[tier] = slots_by_tier[tier].saturating_add(1);
                if cheapest_tier.is_none_or(|t| (tier as u8) < t) {
                    cheapest_tier = Some(tier as u8);
                    cheapest_fee = math_decisions::fee_for_index(
                        pricing.p_period,
                        pricing.mints,
                        id.decision_index(),
                    )
                    .ok();
                }
            }

            if let (Some(tier), Some(fee)) = (cheapest_tier, cheapest_fee) {
                self.open_periods.push(PeriodPricingSummary {
                    period_index: period,
                    cheapest_available_slot_sats: fee,
                    cheapest_available_tier: tier,
                    slots_available_by_tier: slots_by_tier.to_vec(),
                });
            }
        }

        if let Some(first) = self.open_periods.first() {
            for pending in &mut self.pending_new_decisions {
                if pending.period_index == 0 {
                    pending.period_index = first.period_index;
                }
            }
        }

        self.open_periods_loaded = true;
    }

    fn load_claimed_decisions(&mut self, app: &App) {
        self.claimed_decisions.clear();

        let periods = match app.node.get_all_decision_periods() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    "Failed to load periods for decision dropdown: {e:#}"
                );
                self.decisions_loaded = true;
                return;
            }
        };

        for (period, _) in periods {
            let period_id = VotingPeriodId::new(period);
            match app.node.get_claimed_decisions_in_period(period_id) {
                Ok(entries) => {
                    for entry in entries {
                        if let Some(decision) = &entry.decision {
                            self.claimed_decisions.push(ClaimedDecisionInfo {
                                decision_id_hex: hex::encode(
                                    entry.decision_id.as_bytes(),
                                ),
                                header: decision.header.clone(),
                                decision_type: decision.decision_type.clone(),
                                is_categorical: decision.is_categorical(),
                                option_count: decision
                                    .option_count()
                                    .unwrap_or(2),
                            });
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load decisions for period {period}: {e:#}"
                    );
                }
            }
        }

        self.decisions_loaded = true;
    }

    fn build_dimensions_string(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        for (i, _) in self.pending_new_decisions.iter().enumerate() {
            let placeholder = format!("{PLACEHOLDER_PREFIX}{i}");
            if self.is_categorical_decision(&placeholder) {
                parts.push(format!("[{placeholder}]"));
            } else {
                parts.push(placeholder);
            }
        }
        for id in &self.existing_dimension_ids {
            if self.is_categorical_decision(id) {
                parts.push(format!("[{id}]"));
            } else {
                parts.push(id.clone());
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(format!("[{}]", parts.join(",")))
        }
    }

    fn placeholder_index(decision_id_hex: &str) -> Option<usize> {
        decision_id_hex
            .strip_prefix(PLACEHOLDER_PREFIX)
            .and_then(|s| s.parse::<usize>().ok())
    }

    fn pending_options_count(pending: &PendingNewDecision) -> usize {
        pending
            .option_labels_input
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .count()
    }

    fn is_categorical_decision(&self, decision_id_hex: &str) -> bool {
        if let Some(idx) = Self::placeholder_index(decision_id_hex) {
            return self.pending_new_decisions.get(idx).is_some_and(|p| {
                matches!(p.r#type, Some(DecisionType::Category { .. }))
            });
        }
        self.claimed_decisions
            .iter()
            .find(|d| d.decision_id_hex == decision_id_hex)
            .map(|d| d.is_categorical)
            .unwrap_or(false)
    }

    fn preview_rows(&self) -> Vec<PreviewRow> {
        let mut rows = Vec::with_capacity(
            self.pending_new_decisions.len()
                + self.existing_dimension_ids.len(),
        );
        for (i, pending) in self.pending_new_decisions.iter().enumerate() {
            let (type_label, cardinality) = match &pending.r#type {
                None => ("Untyped", 2),
                Some(DecisionType::Binary) => ("Binary", 2),
                Some(DecisionType::Scaled { .. }) => ("Scaled", 2),
                Some(DecisionType::Category { .. }) => {
                    let n = Self::pending_options_count(pending).max(2);
                    ("Categorical", n)
                }
            };
            let header = if !pending.header.is_empty() {
                pending.header.clone()
            } else if !self.title.is_empty() {
                self.title.clone()
            } else {
                String::new()
            };
            rows.push(PreviewRow {
                source: format!("[NEW #{}]", i + 1),
                header,
                type_label,
                cardinality,
            });
        }
        for id in &self.existing_dimension_ids {
            let info = self
                .claimed_decisions
                .iter()
                .find(|d| &d.decision_id_hex == id);
            let (header, type_label, cardinality) = match info {
                Some(info) => {
                    let (tl, c) = if matches!(
                        info.decision_type,
                        DecisionType::Scaled { .. }
                    ) {
                        ("Scaled", 2)
                    } else if info.is_categorical {
                        ("Categorical", info.option_count.max(2))
                    } else {
                        ("Binary", 2)
                    };
                    (info.header.clone(), tl, c)
                }
                None => (String::new(), "Unknown", 2),
            };
            let short_id = if id.len() > 8 {
                format!("{}…", &id[..8])
            } else {
                id.clone()
            };
            rows.push(PreviewRow {
                source: short_id,
                header,
                type_label,
                cardinality,
            });
        }
        rows
    }

    fn build_slot_request(
        pending: &PendingNewDecision,
        index: usize,
    ) -> Result<SlotRequest, String> {
        if pending.header.trim().is_empty() {
            return Err(format!(
                "New decision #{}: market title required (used as \
                 decision header)",
                index + 1
            ));
        }
        let decision_type = match pending.r#type.as_ref().ok_or_else(|| {
            format!("New decision #{}: choose a type", index + 1)
        })? {
            DecisionType::Binary => DecisionType::Binary,
            DecisionType::Scaled { .. } => {
                let min = pending.min_input.parse::<f64>().map_err(|_| {
                    format!("New decision #{}: invalid min", index + 1)
                })?;
                let max = pending.max_input.parse::<f64>().map_err(|_| {
                    format!("New decision #{}: invalid max", index + 1)
                })?;
                let increment = if pending.increment_input.trim().is_empty() {
                    1.0
                } else {
                    pending.increment_input.parse::<f64>().map_err(|_| {
                        format!(
                            "New decision #{}: invalid increment",
                            index + 1
                        )
                    })?
                };
                DecisionType::Scaled {
                    min,
                    max,
                    increment,
                }
            }
            DecisionType::Category { .. } => {
                let options: Vec<String> = pending
                    .option_labels_input
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if options.len() < 2 {
                    return Err(format!(
                        "New decision #{}: category requires \
                         at least 2 option labels",
                        index + 1
                    ));
                }
                DecisionType::Category { options }
            }
        };
        let option_labels =
            if matches!(decision_type, DecisionType::Category { .. }) {
                let labels: Vec<String> = pending
                    .option_labels_input
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                Some(labels)
            } else {
                None
            };
        let tags: Vec<String> = pending
            .tags_input
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let tags = if tags.is_empty() { None } else { Some(tags) };
        let (option_0_label, option_1_label) =
            if matches!(decision_type, DecisionType::Binary) {
                let opt0 = pending.option_0_label_input.trim();
                let opt1 = pending.option_1_label_input.trim();
                (
                    (!opt0.is_empty()).then(|| opt0.to_string()),
                    (!opt1.is_empty()).then(|| opt1.to_string()),
                )
            } else {
                (None, None)
            };
        Ok(SlotRequest {
            period_index: pending.period_index,
            decision_type,
            header: pending.header.clone(),
            description: pending.description.clone(),
            option_0_label,
            option_1_label,
            option_labels,
            tags,
        })
    }

    fn create_market(&mut self, app: &App) {
        self.error = None;
        self.success_message = None;

        let dimensions = self.build_dimensions_string();

        if dimensions.is_none() {
            self.error = Some(
                "At least one dimension with a decision is required"
                    .to_string(),
            );
            return;
        }

        let trading_fee = if self.trading_fee_input.is_empty() {
            Some(DEFAULT_TRADING_FEE)
        } else {
            match self.trading_fee_input.parse::<f64>() {
                Ok(f) if (0.0..=1.0).contains(&f) => Some(f),
                Ok(_) => {
                    self.error =
                        Some("Trading fee must be between 0 and 1".to_string());
                    return;
                }
                Err(_) => {
                    self.error = Some("Invalid trading fee value".to_string());
                    return;
                }
            }
        };

        let initial_liquidity = if self.initial_liquidity_input.is_empty() {
            None
        } else {
            match self.initial_liquidity_input.parse::<u64>() {
                Ok(l) => Some(l),
                Err(_) => {
                    self.error =
                        Some("Invalid initial liquidity value".to_string());
                    return;
                }
            }
        };

        let dimensions_str = dimensions.unwrap();
        tracing::debug!("Creating market with dimensions: {}", dimensions_str);

        let referenced_placeholders: Vec<usize> =
            (0..self.pending_new_decisions.len()).collect();

        let derived_header: String = self.title.chars().take(100).collect();
        let derived_description = self.description.clone();
        for pending in self.pending_new_decisions.iter_mut() {
            if pending.header.trim().is_empty() {
                pending.header = derived_header.clone();
            } else {
                pending.header = pending.header.chars().take(100).collect();
            }
            if pending.description.trim().is_empty() {
                pending.description = derived_description.clone();
            }
        }

        let mut slot_requests: Vec<SlotRequest> =
            Vec::with_capacity(referenced_placeholders.len());
        for idx in &referenced_placeholders {
            let Some(pending) = self.pending_new_decisions.get(*idx) else {
                self.error = Some(format!(
                    "Internal: pending new decision #{} missing",
                    idx + 1
                ));
                return;
            };
            match Self::build_slot_request(pending, *idx) {
                Ok(req) => slot_requests.push(req),
                Err(e) => {
                    self.error = Some(e);
                    return;
                }
            }
        }

        let allocated = match allocate_decision_slots(&app.node, &slot_requests)
        {
            Ok(v) => v,
            Err(e) => {
                self.error =
                    Some(format!("Failed to allocate decision slots: {e}"));
                return;
            }
        };

        let mut placeholder_to_id: HashMap<String, String> = HashMap::new();
        for (slot, placeholder_idx) in
            allocated.iter().zip(referenced_placeholders.iter())
        {
            placeholder_to_id.insert(
                format!("{PLACEHOLDER_PREFIX}{placeholder_idx}"),
                slot.decision_id.to_hex(),
            );
        }

        let resolved_dimensions_str = {
            let mut s = dimensions_str.clone();
            for (placeholder, real) in &placeholder_to_id {
                s = s.replace(placeholder.as_str(), real.as_str());
            }
            s
        };

        let new_claims: Vec<ClaimDecisionPayload> = allocated
            .iter()
            .map(|slot| ClaimDecisionPayload {
                decision_type: slot.decision_type.clone(),
                decisions: vec![slot.entry.clone()],
            })
            .collect();

        let total_listing_fee: u64 = allocated
            .iter()
            .try_fold(0u64, |acc, slot| acc.checked_add(slot.listing_fee_sats))
            .unwrap_or(u64::MAX);

        let (pow_selector, pow_ordering, pow_difficulty) = if self
            .tx_pow_enabled
        {
            let diff: u8 = self.tx_pow_difficulty_input.parse().unwrap_or(0);
            let ord: u8 = self.tx_pow_ordering_input.parse().unwrap_or(0);
            (Some(self.tx_pow_hash_selector), Some(ord), Some(diff))
        } else {
            (None, None, None)
        };

        let category_option_counts = {
            use truthcoin_dc::state::markets::{
                DimensionSpec, parse_dimensions,
            };
            let mut counts = Vec::new();
            if let Ok(specs) = parse_dimensions(&resolved_dimensions_str) {
                for spec in &specs {
                    if let DimensionSpec::Categorical(id) = spec {
                        let from_new_claims =
                            new_claims.iter().find_map(|payload| {
                                payload
                                    .decisions
                                    .iter()
                                    .find(|e| {
                                        e.decision_id_bytes == id.as_bytes()
                                    })
                                    .map(|_| match &payload.decision_type {
                                        DecisionType::Category { options } => {
                                            options.len()
                                        }
                                        _ => 2,
                                    })
                            });
                        let n = if let Some(n) = from_new_claims {
                            n
                        } else {
                            app.node
                                .get_decision_entry(*id)
                                .ok()
                                .flatten()
                                .and_then(|e| e.decision)
                                .and_then(|d| d.option_count())
                                .unwrap_or(2)
                        };
                        counts.push(n);
                    }
                }
            }
            if counts.is_empty() {
                None
            } else {
                Some(counts)
            }
        };

        let input = CreateMarketInput {
            title: self.title.clone(),
            description: self.description.clone(),
            dimensions: resolved_dimensions_str,
            beta: None,
            trading_fee,
            initial_liquidity,
            category_option_counts,
            tx_pow_hash_selector: pow_selector,
            tx_pow_ordering: pow_ordering,
            tx_pow_difficulty: pow_difficulty,
            new_claims,
        };

        let tx_fee = bitcoin::Amount::from_sat(
            1000u64.saturating_add(total_listing_fee),
        );

        match app.wallet.create_market(input, tx_fee) {
            Ok((tx, _market_id)) => {
                if let Err(e) = app.sign_and_send(tx) {
                    self.error = Some(format!("Failed to send: {e:#}"));
                    tracing::error!("Create market failed: {e:#}");
                } else {
                    tracing::info!("Market created: {}", self.title);
                    let mut msg = format!(
                        "Market '{}' created successfully!",
                        self.title
                    );
                    if !allocated.is_empty() {
                        msg.push_str("\nClaimed decisions:");
                        for slot in &allocated {
                            msg.push_str(&format!(
                                "\n  - {} (period {}, fee {} sats)",
                                slot.decision_id.to_hex(),
                                slot.decision_id.period_index(),
                                slot.listing_fee_sats
                            ));
                        }
                    }
                    self.success_message = Some(msg);
                    self.reset();
                }
            }
            Err(e) => {
                self.error = Some(format!("Failed to create market: {e:#}"));
                tracing::error!("Create market tx creation failed: {e:#}");
            }
        }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let Some(app) = app else {
            ui.label("No app connection available");
            return;
        };

        self.load_claimed_decisions(app);
        if !self.open_periods_loaded {
            self.load_open_periods(app);
        }

        if let Some(msg) = &self.success_message {
            ui.colored_label(egui::Color32::GREEN, msg);
            ui.add_space(10.0);
        }

        let multi_new_decisions = self.pending_new_decisions.len() >= 2;

        ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("create_market_basic")
                .num_columns(2)
                .spacing([10.0, 8.0])
                .show(ui, |ui| {
                    let title_label = if multi_new_decisions {
                        "Market Title:"
                    } else {
                        "Title:"
                    };
                    ui.label(title_label);
                    ui.add(
                        egui::TextEdit::singleline(&mut self.title)
                            .hint_text(if multi_new_decisions {
                                "Short market name"
                            } else {
                                "e.g., Crypto Predictions 2025"
                            })
                            .desired_width(400.0),
                    );
                    ui.end_row();

                    ui.label("Description:");
                    ui.add(
                        egui::TextEdit::multiline(&mut self.description)
                            .hint_text("Description...")
                            .desired_width(400.0)
                            .desired_rows(2),
                    );
                    ui.end_row();

                    ui.label("Initial Liquidity:");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.initial_liquidity_input)
                                .desired_width(120.0),
                        );
                        ui.label(RichText::new("sats").weak());
                    });
                    ui.end_row();
                });

            ui.add_space(15.0);
            ui.separator();

            ui.heading("New Decisions");
            ui.add_space(8.0);

            let mut pending_to_remove: Option<usize> = None;
            for idx in 0..self.pending_new_decisions.len() {
                let title_hint = format!("New decision #{}", idx + 1);
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(title_hint).strong());
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui.small_button("Remove").clicked() {
                                    pending_to_remove = Some(idx);
                                }
                            },
                        );
                    });
                    ui.add_space(4.0);

                    egui::Grid::new(format!("pending_new_{idx}"))
                        .num_columns(2)
                        .spacing([10.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("Period:");
                            let pending_period =
                                self.pending_new_decisions[idx].period_index;
                            let selected_period_label = self
                                .open_periods
                                .iter()
                                .find(|p| {
                                    p.period_index == pending_period
                                })
                                .map(|p| {
                                    format!(
                                        "Period {} — {} sats",
                                        p.period_index,
                                        p.cheapest_available_slot_sats,
                                    )
                                })
                                .unwrap_or_else(|| {
                                    format!("Period {pending_period}")
                                });
                            egui::ComboBox::from_id_salt(format!(
                                "pending_period_{idx}"
                            ))
                            .selected_text(selected_period_label)
                            .width(400.0)
                            .show_ui(ui, |ui| {
                                ScrollArea::vertical()
                                    .max_height(240.0)
                                    .show(ui, |ui| {
                                        for p in &self.open_periods {
                                            let label = format!(
                                                "Period {} — {} sats",
                                                p.period_index,
                                                p.cheapest_available_slot_sats,
                                            );
                                            ui.selectable_value(
                                                &mut self
                                                    .pending_new_decisions[idx]
                                                    .period_index,
                                                p.period_index,
                                                label,
                                            );
                                        }
                                    });
                            });
                            ui.end_row();

                            ui.label("Type:");
                            ui.horizontal(|ui| {
                                let current_disc = self.pending_new_decisions
                                    [idx]
                                    .r#type
                                    .as_ref()
                                    .map(std::mem::discriminant);
                                let options = [
                                    DecisionType::Binary,
                                    DecisionType::Scaled {
                                        min: 0.0,
                                        max: 1.0,
                                        increment: 1.0,
                                    },
                                    DecisionType::Category {
                                        options: Vec::new(),
                                    },
                                ];
                                for tag in options {
                                    let label = match tag {
                                        DecisionType::Binary => "Binary",
                                        DecisionType::Scaled { .. } => "Scaled",
                                        DecisionType::Category { .. } => {
                                            "Category"
                                        }
                                    };
                                    let selected = current_disc
                                        == Some(std::mem::discriminant(&tag));
                                    if ui
                                        .selectable_label(selected, label)
                                        .clicked()
                                    {
                                        self.pending_new_decisions[idx].r#type =
                                            Some(tag);
                                    }
                                }
                            });
                            ui.end_row();

                            let decision_type =
                                self.pending_new_decisions[idx].r#type.clone();
                            if let Some(decision_type) = decision_type {
                                if multi_new_decisions {
                                    ui.label("Header:");
                                    ui.add(
                                        egui::TextEdit::singleline(
                                            &mut self.pending_new_decisions
                                                [idx]
                                                .header,
                                        )
                                        .hint_text(
                                            "Decision-specific question",
                                        )
                                        .desired_width(400.0),
                                    );
                                    ui.end_row();
                                }

                                match decision_type {
                                    DecisionType::Scaled { .. } => {
                                        ui.label("Min:");
                                        ui.add(
                                            egui::TextEdit::singleline(
                                                &mut self
                                                    .pending_new_decisions[idx]
                                                    .min_input,
                                            )
                                            .desired_width(80.0),
                                        );
                                        ui.end_row();

                                        ui.label("Max:");
                                        ui.add(
                                            egui::TextEdit::singleline(
                                                &mut self
                                                    .pending_new_decisions[idx]
                                                    .max_input,
                                            )
                                            .desired_width(80.0),
                                        );
                                        ui.end_row();

                                        ui.label("Increment:");
                                        ui.add(
                                            egui::TextEdit::singleline(
                                                &mut self
                                                    .pending_new_decisions[idx]
                                                    .increment_input,
                                            )
                                            .desired_width(80.0),
                                        );
                                        ui.end_row();
                                    }
                                    DecisionType::Category { .. } => {
                                        ui.label("Option labels:");
                                        ui.add(
                                            egui::TextEdit::singleline(
                                                &mut self
                                                    .pending_new_decisions[idx]
                                                    .option_labels_input,
                                            )
                                            .hint_text(
                                                "comma,separated,labels",
                                            )
                                            .desired_width(400.0),
                                        );
                                        ui.end_row();
                                    }
                                    DecisionType::Binary => {
                                        ui.label("Option 0 label:");
                                        ui.add(
                                            egui::TextEdit::singleline(
                                                &mut self
                                                    .pending_new_decisions[idx]
                                                    .option_0_label_input,
                                            )
                                            .hint_text("e.g., No / False")
                                            .desired_width(400.0),
                                        );
                                        ui.end_row();

                                        ui.label("Option 1 label:");
                                        ui.add(
                                            egui::TextEdit::singleline(
                                                &mut self
                                                    .pending_new_decisions[idx]
                                                    .option_1_label_input,
                                            )
                                            .hint_text("e.g., Yes / True")
                                            .desired_width(400.0),
                                        );
                                        ui.end_row();
                                    }
                                }

                                ui.label("Tags:");
                                ui.add(
                                    egui::TextEdit::singleline(
                                        &mut self.pending_new_decisions[idx]
                                            .tags_input,
                                    )
                                    .hint_text("optional, comma-separated")
                                    .desired_width(400.0),
                                );
                                ui.end_row();
                            }
                        });
                });
                ui.add_space(4.0);
            }

            if let Some(idx) = pending_to_remove {
                self.pending_new_decisions.remove(idx);
            }

            ui.horizontal(|ui| {
                if ui.button("+ Add new decision").clicked() {
                    let mut new_pending = PendingNewDecision::default();
                    if let Some(first) = self.open_periods.first() {
                        new_pending.period_index = first.period_index;
                    }
                    self.pending_new_decisions.push(new_pending);
                }
                if ui.small_button("Refresh periods").clicked() {
                    self.open_periods_loaded = false;
                    self.load_open_periods(app);
                }
            });

            ui.add_space(8.0);

            let used: HashSet<String> =
                self.existing_dimension_ids.iter().cloned().collect();
            let mut to_add: Option<String> = None;
            egui::ComboBox::from_id_salt("add_existing_decision")
                .selected_text("Add existing decision…")
                .width(400.0)
                .show_ui(ui, |ui| {
                    ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
                        let mut any = false;
                        for claimed in &self.claimed_decisions {
                            if used.contains(&claimed.decision_id_hex) {
                                continue;
                            }
                            any = true;
                            let suffix = if matches!(
                                claimed.decision_type,
                                DecisionType::Scaled { .. }
                            ) {
                                " [Scaled]".to_string()
                            } else if claimed.is_categorical {
                                format!(
                                    " [Categorical: {}]",
                                    claimed.option_count
                                )
                            } else {
                                String::new()
                            };
                            let label = format!(
                                "{} - {}{}",
                                claimed.decision_id_hex,
                                if claimed.header.len() > 40 {
                                    format!("{}...", &claimed.header[..40])
                                } else {
                                    claimed.header.clone()
                                },
                                suffix
                            );
                            if ui.selectable_label(false, label).clicked() {
                                to_add = Some(claimed.decision_id_hex.clone());
                            }
                        }
                        if !any {
                            ui.label(
                                RichText::new("(no available decisions)")
                                    .weak(),
                            );
                        }
                    });
                });
            if let Some(id) = to_add {
                self.existing_dimension_ids.push(id);
            }

            let mut existing_to_remove: Option<usize> = None;
            for (i, id) in self.existing_dimension_ids.iter().enumerate() {
                let info = self
                    .claimed_decisions
                    .iter()
                    .find(|d| &d.decision_id_hex == id);
                let label = if let Some(info) = info {
                    let suffix = if matches!(
                        info.decision_type,
                        DecisionType::Scaled { .. }
                    ) {
                        " — Scaled"
                    } else if info.is_categorical {
                        " — Categorical"
                    } else {
                        " — Binary"
                    };
                    format!("{} - {}{}", id, info.header, suffix)
                } else {
                    id.clone()
                };
                ui.horizontal(|ui| {
                    ui.label(format!("  {label}"));
                    if ui.small_button("Remove").clicked() {
                        existing_to_remove = Some(i);
                    }
                });
            }
            if let Some(i) = existing_to_remove {
                self.existing_dimension_ids.remove(i);
            }

            ui.add_space(15.0);
            ui.separator();

            let preview_rows = self.preview_rows();
            ui.group(|ui| {
                ui.label(RichText::new("Preview").strong());
                ui.add_space(4.0);
                if preview_rows.is_empty() {
                    ui.label(
                        RichText::new("(no dimensions yet)").weak().small(),
                    );
                } else {
                    egui::Grid::new("preview_grid")
                        .num_columns(5)
                        .spacing([12.0, 4.0])
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label(RichText::new("#").strong());
                            ui.label(RichText::new("Source").strong());
                            ui.label(RichText::new("Decision").strong());
                            ui.label(RichText::new("Type").strong());
                            ui.label(RichText::new("Outcomes").strong());
                            ui.end_row();

                            for (i, row) in preview_rows.iter().enumerate() {
                                ui.label(format!("{}", i + 1));
                                ui.monospace(&row.source);
                                let header = if row.header.is_empty() {
                                    "(unset)".to_string()
                                } else if row.header.chars().count() > 40 {
                                    let truncated: String =
                                        row.header.chars().take(37).collect();
                                    format!("{truncated}...")
                                } else {
                                    row.header.clone()
                                };
                                ui.label(header);
                                ui.label(row.type_label);
                                ui.label(format!("{}", row.cardinality));
                                ui.end_row();
                            }
                        });
                    ui.add_space(6.0);
                    let factors: Vec<String> = preview_rows
                        .iter()
                        .map(|r| r.cardinality.to_string())
                        .collect();
                    let total: usize = preview_rows
                        .iter()
                        .map(|r| r.cardinality)
                        .product();
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Total outcomes:").strong());
                        if preview_rows.len() == 1 {
                            ui.label(format!("{total}"));
                        } else {
                            ui.monospace(format!(
                                "{} = {}",
                                factors.join(" × "),
                                total
                            ));
                        }
                    });
                    if preview_rows.len() >= 8 {
                        ui.add_space(2.0);
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            "Note: 8 decisions is the protocol \
                             cap for one market.",
                        );
                    }
                }
            });

            ui.add_space(15.0);
            ui.separator();

            ui.collapsing("Advanced Settings", |ui| {
                egui::Grid::new("advanced_settings")
                    .num_columns(2)
                    .spacing([10.0, 8.0])
                    .show(ui, |ui| {
                        ui.label("Trading Fee:");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.trading_fee_input)
                                    .hint_text("0.02")
                                    .desired_width(80.0),
                            );
                            ui.label(RichText::new("(0.02 = 2%)").weak());
                        });
                        ui.end_row();

                        ui.label("TX-PoW:");
                        ui.checkbox(
                            &mut self.tx_pow_enabled,
                            "Require proof-of-work for trades",
                        );
                        ui.end_row();
                    });

                if self.tx_pow_enabled {
                    ui.add_space(5.0);
                    ui.group(|ui| {
                        ui.label(
                            RichText::new(
                                "TX-PoW: anti-front-running \
                                 proof-of-work",
                            )
                            .strong(),
                        );
                        ui.add_space(5.0);

                        ui.label("Hash Functions:");
                        let hash_names = [
                            "SHA-256",
                            "SHA-512",
                            "SHA3-256",
                            "SHA3-512",
                            "SHA-512/256",
                            "SHA-384",
                            "BLAKE3",
                            "SHAKE256",
                        ];
                        ui.horizontal_wrapped(|ui| {
                            for (bit, name) in
                                hash_names.iter().enumerate()
                            {
                                let mut selected =
                                    self.tx_pow_hash_selector
                                        & (1 << bit)
                                        != 0;
                                if ui.checkbox(&mut selected, *name)
                                    .changed()
                                {
                                    if selected {
                                        self.tx_pow_hash_selector |=
                                            1 << bit;
                                    } else {
                                        self.tx_pow_hash_selector &=
                                            !(1 << bit);
                                    }
                                }
                            }
                        });

                        egui::Grid::new("tx_pow_params")
                            .num_columns(2)
                            .spacing([10.0, 8.0])
                            .show(ui, |ui| {
                                ui.label("Ordering:");
                                ui.add(
                                    egui::TextEdit::singleline(
                                        &mut self
                                            .tx_pow_ordering_input,
                                    )
                                    .desired_width(60.0),
                                );
                                ui.end_row();

                                ui.label("Difficulty:");
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::TextEdit::singleline(
                                            &mut self
                                                .tx_pow_difficulty_input,
                                        )
                                        .desired_width(60.0),
                                    );
                                    ui.label(
                                        RichText::new(
                                            "leading zero bits",
                                        )
                                        .weak(),
                                    );
                                });
                                ui.end_row();
                            });
                    });
                }
            });

            ui.add_space(15.0);

            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, err);
                ui.add_space(10.0);
            }

            ui.horizontal(|ui| {
                let has_valid_dims =
                    self.build_dimensions_string().is_some();
                let pending_ok = self
                    .pending_new_decisions
                    .iter()
                    .all(|p| p.is_complete());
                let can_create = !self.title.is_empty()
                    && has_valid_dims
                    && pending_ok
                    && !self.is_processing;

                if ui
                    .add_enabled(can_create, Button::new("Create Market"))
                    .clicked()
                {
                    self.is_processing = true;
                    self.create_market(app);
                    self.is_processing = false;
                }

                if ui.button("Clear").clicked() {
                    self.reset();
                }

                if ui.button("Refresh Decisions").clicked() {
                    self.decisions_loaded = false;
                    self.load_claimed_decisions(app);
                }
            });

            if self.claimed_decisions.is_empty() {
                ui.add_space(10.0);
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "No claimed decisions found. Go to the Decisions tab to claim decisions first.",
                );
            }
        });
    }
}
