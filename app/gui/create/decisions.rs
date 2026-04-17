use std::time::{Duration, Instant};

use eframe::egui::{self, Button, RichText, ScrollArea};
use truthcoin_dc::state::decisions::{DecisionEntry, DecisionId, DecisionType};
use truthcoin_dc::state::voting::types::VotingPeriodId;
use truthcoin_dc::types::DecisionClaimEntry;
use truthcoin_dc::wallet::DecisionClaimInput;

use crate::app::App;

const DEFAULT_FEE_SATS: u64 = 1000;

pub struct Decisions {
    current_period: u32,
    periods: Vec<(u32, u64)>,
    selected_period: Option<u32>,
    available_decisions: Vec<DecisionId>,
    claimed_decisions: Vec<DecisionEntry>,
    claim_form: ClaimForm,
    category_form: CategoryDecisionClaimForm,
    error: Option<String>,
    success: Option<String>,
    last_refresh: Instant,
    is_blocks_mode: bool,
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
            category_form: CategoryDecisionClaimForm::default(),
            error: None,
            success: None,
            last_refresh: Instant::now() - Duration::from_secs(10), // Force initial refresh
            is_blocks_mode: true,
        }
    }
}

struct ClaimForm {
    decision_id_input: String,
    header: String,
    description: String,
    is_scaled: bool,
    min_input: String,
    max_input: String,
    option_0_label: String,
    option_1_label: String,
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
            is_scaled: false,
            min_input: String::new(),
            max_input: String::new(),
            option_0_label: String::new(),
            option_1_label: String::new(),
            fee_input: DEFAULT_FEE_SATS.to_string(),
            expanded: false,
            is_processing: false,
        }
    }
}

#[derive(Clone, Default)]
struct CategoryDecisionEntry {
    decision_id_input: String,
    header: String,
}

struct CategoryDecisionClaimForm {
    decisions: Vec<CategoryDecisionEntry>,
    category_name: String,
    fee_input: String,
    is_processing: bool,
}

impl Default for CategoryDecisionClaimForm {
    fn default() -> Self {
        Self {
            decisions: Vec::new(),
            category_name: String::new(),
            fee_input: DEFAULT_FEE_SATS.to_string(),
            is_processing: false,
        }
    }
}

impl Decisions {
    fn refresh_data(&mut self, app: &App) {
        if self.last_refresh.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last_refresh = Instant::now();

        let config = app.node.get_decision_config();
        self.is_blocks_mode = config.is_blocks_mode();

        match app.node.get_current_period() {
            Ok(period) => {
                let period_changed = self.current_period != period;
                self.current_period = period;

                if period_changed || self.selected_period.is_none() {
                    self.selected_period = Some(period);
                }
            }
            Err(e) => {
                self.error =
                    Some(format!("Failed to get current period: {e:#}"));
                return;
            }
        }

        match app.node.get_all_decision_periods() {
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
            self.refresh_decisions_for_period(app, period);
        }
    }

    fn refresh_decisions_for_period(&mut self, app: &App, period: u32) {
        let period_id = VotingPeriodId::new(period);

        match app.node.get_available_decisions_in_period(period_id) {
            Ok(available) => {
                self.available_decisions = available;
            }
            Err(e) => {
                self.error =
                    Some(format!("Failed to load available decisions: {e:#}"));
                tracing::error!("Failed to load available decisions: {e:#}");
            }
        }

        match app.node.get_claimed_decisions_in_period(period_id) {
            Ok(claimed) => {
                self.claimed_decisions = claimed;
            }
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

        let (min, max) = if self.claim_form.is_scaled {
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
                    self.error =
                        Some("Invalid min value: must be a number".to_string());
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
                    self.error =
                        Some("Invalid max value: must be a number".to_string());
                    return;
                }
            };
            if min >= max {
                self.error =
                    Some(format!("Min ({min}) must be less than max ({max})"));
                return;
            }
            (Some(min), Some(max))
        } else {
            (None, None)
        };

        if self.claim_form.header.is_empty() {
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

        let option_0_label = if self.claim_form.option_0_label.trim().is_empty()
        {
            None
        } else {
            Some(self.claim_form.option_0_label.trim().to_string())
        };
        let option_1_label = if self.claim_form.option_1_label.trim().is_empty()
        {
            None
        } else {
            Some(self.claim_form.option_1_label.trim().to_string())
        };

        let decision_type = if self.claim_form.is_scaled {
            DecisionType::Scaled {
                min: min.unwrap_or(0),
                max: max.unwrap_or(100),
            }
        } else {
            DecisionType::Binary
        };

        let entry = DecisionClaimEntry {
            decision_id_bytes,
            header: self.claim_form.header.clone(),
            description: self.claim_form.description.clone(),
            option_0_label,
            option_1_label,
            option_labels: None,
            tags: None,
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

    fn claim_category(&mut self, app: &App) {
        self.error = None;
        self.success = None;

        if self.category_form.decisions.len() < 2 {
            self.error =
                Some("Category requires at least 2 options".to_string());
            return;
        }

        let first_entry = &self.category_form.decisions[0];
        let decision_id_bytes: [u8; 3] =
            match hex::decode(&first_entry.decision_id_input) {
                Ok(bytes) if bytes.len() == 3 => {
                    let mut arr = [0u8; 3];
                    arr.copy_from_slice(&bytes);
                    arr
                }
                Ok(bytes) => {
                    self.error = Some(format!(
                        "Decision ID {} must be 3 bytes \
                         (6 hex chars), got {} bytes",
                        first_entry.decision_id_input,
                        bytes.len()
                    ));
                    return;
                }
                Err(e) => {
                    self.error = Some(format!(
                        "Invalid hex for decision ID {}: {e}",
                        first_entry.decision_id_input
                    ));
                    return;
                }
            };

        let option_labels: Vec<String> = self
            .category_form
            .decisions
            .iter()
            .map(|d| d.header.clone())
            .collect();

        for (i, label) in option_labels.iter().enumerate() {
            if label.is_empty() {
                self.error = Some(format!("Label is required for option {i}"));
                return;
            }
        }

        let entry = DecisionClaimEntry {
            decision_id_bytes,
            header: first_entry.header.clone(),
            description: String::new(),
            option_0_label: None,
            option_1_label: None,
            option_labels: Some(option_labels.clone()),
            tags: None,
        };

        let fee_sats = match self.category_form.fee_input.parse::<u64>() {
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

        let input = DecisionClaimInput {
            decision_type: DecisionType::Category {
                options: option_labels,
            },
            decisions: vec![entry],
        };

        let tx_fee = bitcoin::Amount::from_sat(fee_sats);

        match app.wallet.claim_decision(input, tx_fee) {
            Ok(tx) => {
                let txid = tx.txid();
                if let Err(e) = app.sign_and_send(tx) {
                    self.error = Some(format!("Failed to send: {e:#}"));
                    tracing::error!("Category claim failed: {e:#}");
                } else {
                    self.success = Some(format!(
                        "Category claimed! Txid (category_id): {}",
                        hex::encode(txid.0)
                    ));
                    tracing::info!(
                        "Category claimed with {} decisions, txid: {}",
                        self.category_form.decisions.len(),
                        hex::encode(txid.0)
                    );
                    self.category_form = CategoryDecisionClaimForm::default();
                    if let Some(period) = self.selected_period {
                        self.refresh_decisions_for_period(app, period);
                    }
                }
            }
            Err(e) => {
                self.error =
                    Some(format!("Failed to create category claim tx: {e:#}"));
                tracing::error!("Category claim tx creation failed: {e:#}");
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
                self.last_refresh = Instant::now() - Duration::from_secs(10);
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
                                            ui.ctx()
                                                .copy_text(entry_hex.clone());
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
                                            if decision.is_scaled() {
                                                let range = format!(
                                                    "Scaled [{}-{}]",
                                                    decision.scale_min().unwrap_or(0),
                                                    decision.scale_max().unwrap_or(100)
                                                );
                                                ui.label(
                                                    RichText::new(range)
                                                        .small()
                                                        .weak(),
                                                );
                                            } else {
                                                let (label0, label1) = decision
                                                    .get_binary_labels();
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

            ui.columns(2, |cols| {
                let open_override = if self.claim_form.expanded {
                    Some(true)
                } else {
                    None
                };
                egui::CollapsingHeader::new(
                    RichText::new("Claim Single Decision").heading(),
                )
                .id_salt("claim_single_decision")
                .open(open_override)
                .show(&mut cols[0], |ui| {
                    self.show_claim_form(app, ui);
                });

                egui::CollapsingHeader::new(
                    RichText::new("Claim Category").heading(),
                )
                .id_salt("claim_category")
                .show(&mut cols[1], |ui| {
                    self.show_category_claim_form(app, ui);
                });
            });
            self.claim_form.expanded = false;
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("Select a period to view decisions");
            });
        }
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
                    ui.selectable_value(
                        &mut self.claim_form.is_scaled,
                        false,
                        "Binary",
                    );
                    ui.selectable_value(
                        &mut self.claim_form.is_scaled,
                        true,
                        "Scaled",
                    );
                });
                ui.end_row();

                if self.claim_form.is_scaled {
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
                } else {
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

                ui.label("Fee (sats):");
                ui.add(
                    egui::TextEdit::singleline(&mut self.claim_form.fee_input)
                        .desired_width(100.0),
                );
                ui.end_row();
            });

        ui.add_space(10.0);

        let can_claim = !self.claim_form.decision_id_input.is_empty()
            && !self.claim_form.header.is_empty()
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

    fn show_category_claim_form(&mut self, app: &App, ui: &mut egui::Ui) {
        ui.label(
            RichText::new(
                "Claim multiple decisions atomically as a category. \
                 The transaction ID becomes the category identifier.",
            )
            .weak()
            .italics(),
        );
        ui.add_space(5.0);

        ui.horizontal(|ui| {
            ui.label("Category Name:");
            ui.add(
                egui::TextEdit::singleline(
                    &mut self.category_form.category_name,
                )
                .hint_text("e.g., AFC North Winner")
                .desired_width(250.0),
            );
        });

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(5.0);

        ui.horizontal(|ui| {
            ui.heading("Category Decisions");
            ui.label(
                RichText::new(format!(
                    "({} decisions)",
                    self.category_form.decisions.len()
                ))
                .weak(),
            );
        });

        let mut remove_idx: Option<usize> = None;
        ScrollArea::vertical()
            .id_salt("category_decisions_list")
            .max_height(200.0)
            .show(ui, |ui| {
                for (idx, entry) in
                    self.category_form.decisions.iter_mut().enumerate()
                {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("Decision {}:", idx + 1));
                            ui.add(
                                egui::TextEdit::singleline(
                                    &mut entry.decision_id_input,
                                )
                                .hint_text("e.g., 004008")
                                .desired_width(100.0)
                                .font(egui::TextStyle::Monospace),
                            );
                            if ui.small_button("❌").clicked() {
                                remove_idx = Some(idx);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Header:");
                            ui.add(
                                egui::TextEdit::singleline(&mut entry.header)
                                    .hint_text("e.g., Will the Steelers win?")
                                    .desired_width(300.0),
                            );
                        });
                    });
                }
            });

        if let Some(idx) = remove_idx {
            self.category_form.decisions.remove(idx);
        }

        ui.add_space(5.0);

        ui.horizontal(|ui| {
            if ui.button("➕ Add Decision").clicked() {
                self.category_form
                    .decisions
                    .push(CategoryDecisionEntry::default());
            }

            if !self.available_decisions.is_empty() {
                egui::ComboBox::from_label("Quick Add")
                    .selected_text("Select available decision...")
                    .show_ui(ui, |ui| {
                        for decision_id in &self.available_decisions {
                            let entry_hex = hex::encode(decision_id.as_bytes());
                            if ui.selectable_label(false, &entry_hex).clicked()
                            {
                                self.category_form.decisions.push(
                                    CategoryDecisionEntry {
                                        decision_id_input: entry_hex,
                                        header: String::new(),
                                    },
                                );
                            }
                        }
                    });
            }
        });

        ui.add_space(10.0);

        ui.horizontal(|ui| {
            ui.label("Fee (sats):");
            ui.add(
                egui::TextEdit::singleline(&mut self.category_form.fee_input)
                    .desired_width(100.0),
            );
        });

        ui.add_space(10.0);

        let can_claim = self.category_form.decisions.len() >= 2
            && self.category_form.decisions.iter().all(|s| {
                !s.decision_id_input.is_empty() && !s.header.is_empty()
            })
            && !self.category_form.is_processing;

        let button_text = if self.category_form.decisions.len() < 2 {
            format!(
                "Claim Category (need {} more decisions)",
                2 - self.category_form.decisions.len()
            )
        } else {
            format!(
                "Claim Category ({} decisions)",
                self.category_form.decisions.len()
            )
        };

        if ui
            .add_enabled(can_claim, Button::new(button_text))
            .clicked()
        {
            self.category_form.is_processing = true;
            self.claim_category(app);
            self.category_form.is_processing = false;
        }

        if !can_claim && self.category_form.decisions.len() >= 2 {
            ui.label(
                RichText::new("Fill in all decision IDs and headers to claim")
                    .weak()
                    .italics(),
            );
        }
    }
}
