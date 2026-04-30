use std::pin::Pin;
use std::task::Poll;
use std::time::Duration;

use eframe::egui::{self, Button, RichText, ScrollArea};
use futures::Stream;
use truthcoin_dc::state::decisions::{DecisionEntry, DecisionId, DecisionType};
use truthcoin_dc::types::DecisionClaimEntry;
use truthcoin_dc::wallet::DecisionClaimInput;

use crate::app::App;
use crate::util::PromiseStream;

type NodeUpdates = PromiseStream<Pin<Box<dyn Stream<Item = ()> + Send>>>;

const DEFAULT_FEE_SATS: u64 = 1000;

pub struct Decisions {
    current_period: u32,
    periods: Vec<(u32, u64)>,
    selected_period: Option<u32>,
    available_decisions: Vec<DecisionId>,
    claimed_decisions: Vec<DecisionEntry>,
    claim_form: ClaimForm,
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
            claim_form: ClaimForm::default(),
            error: None,
            success: None,
            is_blocks_mode: true,
            node_updated: None,
            needs_initial_load: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecisionKind {
    Binary,
    Scaled,
    Categorical,
}

struct ClaimForm {
    decision_id_input: String,
    header: String,
    description: String,
    kind: DecisionKind,
    min_input: String,
    max_input: String,
    option_0_label: String,
    option_1_label: String,
    outcomes: Vec<String>,
    tags_input: String,
    fee_input: String,
    expanded: bool,
    is_processing: bool,
}

impl Default for ClaimForm {
    fn default() -> Self {
        Self {
            decision_id_input: String::new(),
            header: String::new(),
            description: String::new(),
            kind: DecisionKind::Binary,
            min_input: String::new(),
            max_input: String::new(),
            option_0_label: String::new(),
            option_1_label: String::new(),
            outcomes: vec![String::new(), String::new()],
            tags_input: String::new(),
            fee_input: DEFAULT_FEE_SATS.to_string(),
            expanded: false,
            is_processing: false,
        }
    }
}

impl Decisions {
    fn ensure_subscribed(&mut self, app: &App) {
        if self.node_updated.is_none() {
            let stream: Pin<Box<dyn Stream<Item = ()> + Send>> =
                Box::pin(app.node.watch_state());
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
            match state.get_available_decisions_in_period(&rotxn, period) {
                Ok(available) => self.available_decisions = available,
                Err(e) => {
                    self.error = Some(format!(
                        "Failed to load available decisions: {e:#}"
                    ));
                    tracing::error!(
                        "Failed to load available decisions: {e:#}"
                    );
                }
            }
            match state
                .decisions()
                .get_claimed_decisions_in_period(&rotxn, period)
            {
                Ok(claimed) => self.claimed_decisions = claimed,
                Err(e) => {
                    self.error = Some(format!(
                        "Failed to load claimed decisions: {e:#}"
                    ));
                    tracing::error!("Failed to load claimed decisions: {e:#}");
                }
            }
        }
    }

    fn refresh_decisions_for_period(&mut self, app: &App, period: u32) {
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
    }

    fn claim_decision(&mut self, app: &App) {
        self.error = None;
        self.success = None;

        let decision_id_bytes =
            match DecisionId::from_hex(&self.claim_form.decision_id_input) {
                Ok(decision_id) => decision_id.as_bytes(),
                Err(e) => {
                    self.error = Some(format!("Invalid decision ID: {e}"));
                    return;
                }
            };

        if self.claim_form.header.trim().is_empty() {
            self.error = Some("Header is required".to_string());
            return;
        }

        let fee_sats = match self.claim_form.fee_input.parse::<u64>() {
            Ok(v) if v > 0 => v,
            Ok(_) => {
                self.error = Some("Fee must be greater than 0".to_string());
                return;
            }
            Err(_) => {
                self.error =
                    Some("Invalid fee: must be a positive number".to_string());
                return;
            }
        };

        let (decision_type, option_0_label, option_1_label, option_labels) =
            match self.claim_form.kind {
                DecisionKind::Binary => {
                    let opt0 =
                        self.claim_form.option_0_label.trim().to_string();
                    let opt1 =
                        self.claim_form.option_1_label.trim().to_string();
                    (
                        DecisionType::Binary,
                        (!opt0.is_empty()).then_some(opt0),
                        (!opt1.is_empty()).then_some(opt1),
                        None,
                    )
                }
                DecisionKind::Scaled => {
                    let min = match self.claim_form.min_input.parse::<i64>() {
                        Ok(v) => v,
                        Err(_) if self.claim_form.min_input.is_empty() => {
                            self.error = Some(
                                "Min value is required for scaled decisions"
                                    .to_string(),
                            );
                            return;
                        }
                        Err(_) => {
                            self.error = Some(
                                "Invalid min value: must be a number"
                                    .to_string(),
                            );
                            return;
                        }
                    };
                    let max = match self.claim_form.max_input.parse::<i64>() {
                        Ok(v) => v,
                        Err(_) if self.claim_form.max_input.is_empty() => {
                            self.error = Some(
                                "Max value is required for scaled decisions"
                                    .to_string(),
                            );
                            return;
                        }
                        Err(_) => {
                            self.error = Some(
                                "Invalid max value: must be a number"
                                    .to_string(),
                            );
                            return;
                        }
                    };
                    if min >= max {
                        self.error = Some(format!(
                            "Min ({min}) must be less than max ({max})"
                        ));
                        return;
                    }
                    (DecisionType::Scaled { min, max }, None, None, None)
                }
                DecisionKind::Categorical => {
                    let trimmed: Vec<String> = self
                        .claim_form
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
            .claim_form
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

        let entry = DecisionClaimEntry {
            decision_id_bytes,
            header: self.claim_form.header.clone(),
            description: self.claim_form.description.clone(),
            option_0_label,
            option_1_label,
            option_labels,
            tags: if tags.is_empty() { None } else { Some(tags) },
        };

        let input = DecisionClaimInput {
            decision_type,
            decisions: vec![entry],
        };

        let tx_fee = bitcoin::Amount::from_sat(fee_sats);

        match app.wallet.claim_decision(input, tx_fee) {
            Ok(tx) => {
                if let Err(e) = app.sign_and_send(tx) {
                    self.error = Some(format!("Failed to send: {e:#}"));
                    tracing::error!("Decision claim failed: {e:#}");
                } else {
                    self.success = Some(format!(
                        "Decision {} claimed successfully!",
                        self.claim_form.decision_id_input
                    ));
                    tracing::info!(
                        "Decision claimed: {}",
                        self.claim_form.decision_id_input
                    );
                    self.claim_form = ClaimForm::default();
                    if let Some(period) = self.selected_period {
                        self.refresh_decisions_for_period(app, period);
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
            ui.heading("Decisions");
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
            ui.colored_label(egui::Color32::GREEN, msg);
        }
        if let Some(err) = &self.error {
            ui.colored_label(egui::Color32::RED, err);
        }

        ui.add_space(5.0);

        ui.horizontal(|ui| {
            ui.label("Period:");
            let prev_selection = self.selected_period;
            egui::ComboBox::from_id_salt("period_selector")
                .selected_text(
                    self.selected_period
                        .map(|p| {
                            let is_current = p == self.current_period;
                            if is_current {
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
                            format!("Period {period} ({available} available) \u{2190} current")
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
                self.refresh_decisions_for_period(app, period);
            }
        });

        ui.add_space(10.0);

        if self.selected_period.is_some() {
            ScrollArea::vertical()
                .id_salt("decisions_body")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    self.show_period_body(app, ui);
                });
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("Select a period to view decisions");
            });
        }
    }

    fn show_period_body(&mut self, app: &App, ui: &mut egui::Ui) {
        ui.columns(2, |cols| {
            cols[0].heading("Available Decisions");
            cols[0].label(
                RichText::new(format!(
                    "{} unclaimed",
                    self.available_decisions.len()
                ))
                .weak(),
            );

            ScrollArea::vertical()
                .id_salt("available_decisions")
                .max_height(200.0)
                .show(&mut cols[0], |ui| {
                    if self.available_decisions.is_empty() {
                        ui.label("No available decisions in this period");
                    } else {
                        for decision_id in &self.available_decisions {
                            let entry_hex = hex::encode(decision_id.as_bytes());
                            let idx = decision_id.decision_index();
                            let decision_type =
                                if idx < 500 { "std" } else { "non-std" };

                            ui.horizontal(|ui| {
                                ui.monospace(&entry_hex);
                                ui.label(format!("[{decision_type}]"));
                                if ui.small_button("Claim").clicked() {
                                    self.claim_form.decision_id_input =
                                        entry_hex.clone();
                                    self.claim_form.expanded = true;
                                }
                            });
                        }
                    }
                });

            cols[1].heading("Claimed Decisions");
            cols[1].label(
                RichText::new(format!(
                    "{} claimed",
                    self.claimed_decisions.len()
                ))
                .weak(),
            );

            ScrollArea::vertical()
                .id_salt("claimed_decisions")
                .max_height(200.0)
                .show(&mut cols[1], |ui| {
                    if self.claimed_decisions.is_empty() {
                        ui.label("No claimed decisions in this period");
                    } else {
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
                                        if entry.decision_id.is_standard() {
                                            ui.label(
                                                RichText::new("Standard")
                                                    .small()
                                                    .weak(),
                                            );
                                        }
                                        if decision.is_categorical() {
                                            let labels = decision
                                                .get_category_labels()
                                                .unwrap_or(&[]);
                                            let shown = if labels.len() > 4 {
                                                format!(
                                                    "{}/…",
                                                    labels[..4].join("/")
                                                )
                                            } else {
                                                labels.join("/")
                                            };
                                            ui.label(
                                                RichText::new(format!(
                                                    "Category [{shown}]"
                                                ))
                                                .small()
                                                .weak(),
                                            );
                                        } else if decision.is_scaled() {
                                            let range = format!(
                                                "Scaled [{}-{}]",
                                                decision
                                                    .scale_min()
                                                    .unwrap_or(0),
                                                decision
                                                    .scale_max()
                                                    .unwrap_or(100)
                                            );
                                            ui.label(
                                                RichText::new(range)
                                                    .small()
                                                    .weak(),
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
                    }
                });
        });

        ui.add_space(15.0);
        ui.separator();

        let open_override = if self.claim_form.expanded {
            Some(true)
        } else {
            None
        };
        egui::CollapsingHeader::new(RichText::new("Claim Decision").heading())
            .id_salt("claim_decision")
            .open(open_override)
            .show(ui, |ui| {
                self.show_claim_form(app, ui);
            });
        self.claim_form.expanded = false;
    }

    fn show_claim_form(&mut self, app: &App, ui: &mut egui::Ui) {
        egui::Grid::new("claim_form")
            .num_columns(2)
            .spacing([10.0, 8.0])
            .show(ui, |ui| {
                ui.label("Decision ID (hex):");
                ui.add(
                    egui::TextEdit::singleline(
                        &mut self.claim_form.decision_id_input,
                    )
                    .hint_text("e.g., 0a1b2c")
                    .desired_width(150.0)
                    .font(egui::TextStyle::Monospace),
                );
                ui.end_row();

                ui.label("Header:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.claim_form.header)
                        .hint_text("Short title (max 100 bytes)")
                        .desired_width(300.0),
                );
                ui.end_row();

                ui.label("Description:");
                ui.add(
                    egui::TextEdit::multiline(&mut self.claim_form.description)
                        .hint_text("Detailed description (max 2000 bytes)")
                        .desired_width(300.0)
                        .desired_rows(3),
                );
                ui.end_row();

                ui.label("Type:");
                ui.horizontal(|ui| {
                    let prev_kind = self.claim_form.kind;
                    ui.selectable_value(
                        &mut self.claim_form.kind,
                        DecisionKind::Binary,
                        "Binary",
                    );
                    ui.selectable_value(
                        &mut self.claim_form.kind,
                        DecisionKind::Scaled,
                        "Scaled",
                    );
                    ui.selectable_value(
                        &mut self.claim_form.kind,
                        DecisionKind::Categorical,
                        "Categorical",
                    );
                    if prev_kind != self.claim_form.kind {
                        match prev_kind {
                            DecisionKind::Binary => {
                                self.claim_form.option_0_label.clear();
                                self.claim_form.option_1_label.clear();
                            }
                            DecisionKind::Scaled => {
                                self.claim_form.min_input.clear();
                                self.claim_form.max_input.clear();
                            }
                            DecisionKind::Categorical => {
                                self.claim_form.outcomes =
                                    vec![String::new(), String::new()];
                            }
                        }
                    }
                });
                ui.end_row();

                match self.claim_form.kind {
                    DecisionKind::Binary => {
                        ui.label("Option 0 label:");
                        ui.add(
                            egui::TextEdit::singleline(
                                &mut self.claim_form.option_0_label,
                            )
                            .hint_text("e.g., False (default: No)")
                            .desired_width(200.0),
                        );
                        ui.end_row();

                        ui.label("Option 1 label:");
                        ui.add(
                            egui::TextEdit::singleline(
                                &mut self.claim_form.option_1_label,
                            )
                            .hint_text("e.g., True (default: Yes)")
                            .desired_width(200.0),
                        );
                        ui.end_row();
                    }
                    DecisionKind::Scaled => {
                        ui.label("Scale Range:");
                        ui.horizontal(|ui| {
                            ui.label("Min:");
                            ui.add(
                                egui::TextEdit::singleline(
                                    &mut self.claim_form.min_input,
                                )
                                .hint_text("e.g., 0")
                                .desired_width(80.0),
                            );
                            ui.label("Max:");
                            ui.add(
                                egui::TextEdit::singleline(
                                    &mut self.claim_form.max_input,
                                )
                                .hint_text("e.g., 100")
                                .desired_width(80.0),
                            );
                        });
                        ui.end_row();
                    }
                    DecisionKind::Categorical => {
                        let mut remove_idx: Option<usize> = None;
                        let can_remove = self.claim_form.outcomes.len() > 2;
                        for (idx, outcome) in
                            self.claim_form.outcomes.iter_mut().enumerate()
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
                            self.claim_form.outcomes.remove(idx);
                        }
                        ui.label("");
                        if ui.button("\u{2795} Add outcome").clicked() {
                            self.claim_form.outcomes.push(String::new());
                        }
                        ui.end_row();
                    }
                }

                ui.label("Tags:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.claim_form.tags_input)
                        .hint_text(
                            "Comma-separated (max 10 tags, 50 bytes each)",
                        )
                        .desired_width(300.0),
                );
                ui.end_row();

                ui.label("Fee (sats):");
                ui.add(
                    egui::TextEdit::singleline(&mut self.claim_form.fee_input)
                        .desired_width(100.0),
                );
                ui.end_row();
            });

        ui.add_space(10.0);

        let categorical_ok = self.claim_form.kind != DecisionKind::Categorical
            || (self.claim_form.outcomes.len() >= 2
                && self
                    .claim_form
                    .outcomes
                    .iter()
                    .all(|o| !o.trim().is_empty()));

        let can_claim = !self.claim_form.decision_id_input.is_empty()
            && !self.claim_form.header.trim().is_empty()
            && categorical_ok
            && !self.claim_form.is_processing;

        if ui
            .add_enabled(can_claim, Button::new("Claim Decision"))
            .clicked()
        {
            self.claim_form.is_processing = true;
            self.claim_decision(app);
            self.claim_form.is_processing = false;
        }
    }
}
