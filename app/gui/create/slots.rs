use std::time::{Duration, Instant};

use eframe::egui::{self, Button, RichText, ScrollArea};
use truthcoin_dc::state::slots::{Slot, SlotId};
use truthcoin_dc::state::voting::types::VotingPeriodId;
use truthcoin_dc::wallet::{CategoryClaimInput, SlotClaimInput};

use crate::app::App;

const DEFAULT_FEE_SATS: u64 = 1000;

pub struct Slots {
    current_period: u32,
    periods: Vec<(u32, u64)>,
    selected_period: Option<u32>,
    available_slots: Vec<SlotId>,
    claimed_slots: Vec<Slot>,
    claim_form: ClaimForm,
    category_form: CategoryClaimForm,
    error: Option<String>,
    success: Option<String>,
    last_refresh: Instant,
    is_blocks_mode: bool,
}

impl Default for Slots {
    fn default() -> Self {
        Self {
            current_period: 0,
            periods: Vec::new(),
            selected_period: None,
            available_slots: Vec::new(),
            claimed_slots: Vec::new(),
            claim_form: ClaimForm::default(),
            category_form: CategoryClaimForm::default(),
            error: None,
            success: None,
            last_refresh: Instant::now() - Duration::from_secs(10), // Force initial refresh
            is_blocks_mode: true,
        }
    }
}

struct ClaimForm {
    slot_id_input: String,
    question: String,
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
            slot_id_input: String::new(),
            question: String::new(),
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
struct CategorySlotEntry {
    slot_id_input: String,
    question: String,
}

struct CategoryClaimForm {
    slots: Vec<CategorySlotEntry>,
    is_standard: bool,
    category_name: String,
    fee_input: String,
    is_processing: bool,
}

impl Default for CategoryClaimForm {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            is_standard: true,
            category_name: String::new(),
            fee_input: DEFAULT_FEE_SATS.to_string(),
            is_processing: false,
        }
    }
}

impl Slots {
    fn refresh_data(&mut self, app: &App) {
        if self.last_refresh.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last_refresh = Instant::now();

        let config = app.node.get_slot_config();
        self.is_blocks_mode = config.is_blocks_mode();

        let mainchain_ts = app.node.get_mainchain_timestamp().unwrap_or(0);

        match app.node.get_current_period(mainchain_ts) {
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

        match app.node.get_all_slot_periods() {
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
                tracing::error!("Failed to load slot periods: {e:#}");
            }
        }

        if let Some(period) = self.selected_period {
            self.refresh_slots_for_period(app, period);
        }
    }

    fn refresh_slots_for_period(&mut self, app: &App, period: u32) {
        let period_id = VotingPeriodId::new(period);

        match app.node.get_available_slots_in_period(period_id) {
            Ok(slots) => {
                self.available_slots = slots;
            }
            Err(e) => {
                self.error =
                    Some(format!("Failed to load available slots: {e:#}"));
                tracing::error!("Failed to load available slots: {e:#}");
            }
        }

        match app.node.get_claimed_slots_in_period(period_id) {
            Ok(slots) => {
                self.claimed_slots = slots;
            }
            Err(e) => {
                self.error =
                    Some(format!("Failed to load claimed slots: {e:#}"));
                tracing::error!("Failed to load claimed slots: {e:#}");
            }
        }
    }

    fn claim_slot(&mut self, app: &App) {
        self.error = None;
        self.success = None;

        let (slot_id_bytes, is_standard) =
            match SlotId::from_hex(&self.claim_form.slot_id_input) {
                Ok(slot_id) => {
                    let is_standard = slot_id.slot_index() < 500;
                    (slot_id.as_bytes(), is_standard)
                }
                Err(e) => {
                    self.error = Some(format!("Invalid slot ID: {e}"));
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

        if self.claim_form.question.is_empty() {
            self.error = Some("Question is required".to_string());
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

        let input = SlotClaimInput {
            slot_id_bytes,
            is_standard,
            is_scaled: self.claim_form.is_scaled,
            question: self.claim_form.question.clone(),
            min,
            max,
            option_0_label,
            option_1_label,
        };

        let tx_fee = bitcoin::Amount::from_sat(fee_sats);

        match app.wallet.claim_decision_slot(input, tx_fee) {
            Ok(tx) => {
                if let Err(e) = app.sign_and_send(tx) {
                    self.error = Some(format!("Failed to send: {e:#}"));
                    tracing::error!("Slot claim failed: {e:#}");
                } else {
                    self.success = Some(format!(
                        "Slot {} claimed successfully!",
                        self.claim_form.slot_id_input
                    ));
                    tracing::info!(
                        "Slot claimed: {}",
                        self.claim_form.slot_id_input
                    );
                    self.claim_form = ClaimForm::default();
                    if let Some(period) = self.selected_period {
                        self.refresh_slots_for_period(app, period);
                    }
                }
            }
            Err(e) => {
                self.error = Some(format!("Failed to create claim tx: {e:#}"));
                tracing::error!("Slot claim tx creation failed: {e:#}");
            }
        }
    }

    fn claim_category(&mut self, app: &App) {
        self.error = None;
        self.success = None;

        if self.category_form.slots.len() < 2 {
            self.error = Some("Category requires at least 2 slots".to_string());
            return;
        }

        let mut slots: Vec<([u8; 3], String)> = Vec::new();
        for entry in &self.category_form.slots {
            let slot_id_bytes: [u8; 3] = match hex::decode(&entry.slot_id_input)
            {
                Ok(bytes) if bytes.len() == 3 => {
                    let mut arr = [0u8; 3];
                    arr.copy_from_slice(&bytes);
                    arr
                }
                Ok(bytes) => {
                    self.error = Some(format!(
                        "Slot ID {} must be 3 bytes (6 hex chars), got {} bytes",
                        entry.slot_id_input,
                        bytes.len()
                    ));
                    return;
                }
                Err(e) => {
                    self.error = Some(format!(
                        "Invalid hex for slot ID {}: {e}",
                        entry.slot_id_input
                    ));
                    return;
                }
            };

            if entry.question.is_empty() {
                self.error = Some(format!(
                    "Question is required for slot {}",
                    entry.slot_id_input
                ));
                return;
            }

            slots.push((slot_id_bytes, entry.question.clone()));
        }

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

        let input = CategoryClaimInput {
            slots,
            is_standard: self.category_form.is_standard,
        };

        let tx_fee = bitcoin::Amount::from_sat(fee_sats);

        match app.wallet.claim_category_slots(input, tx_fee) {
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
                        "Category claimed with {} slots, txid: {}",
                        self.category_form.slots.len(),
                        hex::encode(txid.0)
                    );
                    self.category_form = CategoryClaimForm::default();
                    if let Some(period) = self.selected_period {
                        self.refresh_slots_for_period(app, period);
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
            ui.heading("Decision Slots");
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
                self.refresh_slots_for_period(app, period);
            }
        });

        ui.add_space(10.0);

        if self.selected_period.is_some() {
            ui.columns(2, |cols| {
                cols[0].heading("Available Slots");
                cols[0].label(
                    RichText::new(format!(
                        "{} unclaimed",
                        self.available_slots.len()
                    ))
                    .weak(),
                );

                ScrollArea::vertical()
                    .id_salt("available_slots")
                    .max_height(200.0)
                    .show(&mut cols[0], |ui| {
                        if self.available_slots.is_empty() {
                            ui.label("No available slots in this period");
                        } else {
                            for slot_id in &self.available_slots {
                                let slot_hex = hex::encode(slot_id.as_bytes());
                                let idx = slot_id.slot_index();
                                let slot_type =
                                    if idx < 500 { "std" } else { "non-std" };

                                ui.horizontal(|ui| {
                                    ui.monospace(&slot_hex);
                                    ui.label(format!("[{slot_type}]"));
                                    if ui.small_button("Claim").clicked() {
                                        self.claim_form.slot_id_input =
                                            slot_hex.clone();
                                        self.claim_form.expanded = true;
                                    }
                                });
                            }
                        }
                    });

                cols[1].heading("Claimed Slots");
                cols[1].label(
                    RichText::new(format!(
                        "{} claimed",
                        self.claimed_slots.len()
                    ))
                    .weak(),
                );

                ScrollArea::vertical()
                    .id_salt("claimed_slots")
                    .max_height(200.0)
                    .show(&mut cols[1], |ui| {
                        if self.claimed_slots.is_empty() {
                            ui.label("No claimed slots in this period");
                        } else {
                            for slot in &self.claimed_slots {
                                let slot_hex =
                                    hex::encode(slot.slot_id.as_bytes());

                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.monospace(&slot_hex);
                                        if ui.small_button("Copy").clicked() {
                                            ui.ctx()
                                                .copy_text(slot_hex.clone());
                                        }
                                    });

                                    if let Some(decision) = &slot.decision {
                                        ui.label(&decision.question);
                                        ui.horizontal(|ui| {
                                            if decision.is_standard {
                                                ui.label(
                                                    RichText::new("Standard")
                                                        .small()
                                                        .weak(),
                                                );
                                            }
                                            if decision.is_scaled {
                                                let range = format!(
                                                    "Scaled [{}-{}]",
                                                    decision.min.unwrap_or(0),
                                                    decision.max.unwrap_or(100)
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
                    RichText::new("Claim Single Slot").heading(),
                )
                .id_salt("claim_single_slot")
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
                ui.label("Select a period to view slots");
            });
        }
    }

    fn show_claim_form(&mut self, app: &App, ui: &mut egui::Ui) {
        egui::Grid::new("claim_form")
            .num_columns(2)
            .spacing([10.0, 8.0])
            .show(ui, |ui| {
                ui.label("Slot ID (hex):");
                ui.add(
                    egui::TextEdit::singleline(
                        &mut self.claim_form.slot_id_input,
                    )
                    .hint_text("e.g., 0a1b2c")
                    .desired_width(150.0)
                    .font(egui::TextStyle::Monospace),
                );
                ui.end_row();

                ui.label("Question:");
                ui.add(
                    egui::TextEdit::multiline(&mut self.claim_form.question)
                        .hint_text("What decision does this slot represent?")
                        .desired_width(300.0)
                        .desired_rows(2),
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

        let can_claim = !self.claim_form.slot_id_input.is_empty()
            && !self.claim_form.question.is_empty()
            && !self.claim_form.is_processing;

        if ui
            .add_enabled(can_claim, Button::new("Claim Slot"))
            .clicked()
        {
            self.claim_form.is_processing = true;
            self.claim_slot(app);
            self.claim_form.is_processing = false;
        }
    }

    #[allow(dead_code)]
    pub fn get_user_claimed_slots(&self) -> Vec<(String, String)> {
        self.claimed_slots
            .iter()
            .filter_map(|slot| {
                slot.decision.as_ref().map(|d| {
                    (hex::encode(slot.slot_id.as_bytes()), d.question.clone())
                })
            })
            .collect()
    }

    fn show_category_claim_form(&mut self, app: &App, ui: &mut egui::Ui) {
        ui.label(
            RichText::new(
                "Claim multiple slots atomically as a category. \
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

        ui.checkbox(
            &mut self.category_form.is_standard,
            "Standard slots (index 0-499)",
        );

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(5.0);

        ui.horizontal(|ui| {
            ui.heading("Category Slots");
            ui.label(
                RichText::new(format!(
                    "({} slots)",
                    self.category_form.slots.len()
                ))
                .weak(),
            );
        });

        let mut remove_idx: Option<usize> = None;
        ScrollArea::vertical()
            .id_salt("category_slots_list")
            .max_height(200.0)
            .show(ui, |ui| {
                for (idx, entry) in
                    self.category_form.slots.iter_mut().enumerate()
                {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("Slot {}:", idx + 1));
                            ui.add(
                                egui::TextEdit::singleline(
                                    &mut entry.slot_id_input,
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
                            ui.label("Question:");
                            ui.add(
                                egui::TextEdit::singleline(&mut entry.question)
                                    .hint_text("e.g., Will the Steelers win?")
                                    .desired_width(300.0),
                            );
                        });
                    });
                }
            });

        if let Some(idx) = remove_idx {
            self.category_form.slots.remove(idx);
        }

        ui.add_space(5.0);

        ui.horizontal(|ui| {
            if ui.button("➕ Add Slot").clicked() {
                self.category_form.slots.push(CategorySlotEntry::default());
            }

            if !self.available_slots.is_empty() {
                egui::ComboBox::from_label("Quick Add")
                    .selected_text("Select available slot...")
                    .show_ui(ui, |ui| {
                        for slot_id in &self.available_slots {
                            let slot_hex = hex::encode(slot_id.as_bytes());
                            if ui.selectable_label(false, &slot_hex).clicked() {
                                self.category_form.slots.push(
                                    CategorySlotEntry {
                                        slot_id_input: slot_hex,
                                        question: String::new(),
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

        let can_claim =
            self.category_form.slots.len() >= 2
                && self.category_form.slots.iter().all(|s| {
                    !s.slot_id_input.is_empty() && !s.question.is_empty()
                })
                && !self.category_form.is_processing;

        let button_text = if self.category_form.slots.len() < 2 {
            format!(
                "Claim Category (need {} more slots)",
                2 - self.category_form.slots.len()
            )
        } else {
            format!("Claim Category ({} slots)", self.category_form.slots.len())
        };

        if ui
            .add_enabled(can_claim, Button::new(button_text))
            .clicked()
        {
            self.category_form.is_processing = true;
            self.claim_category(app);
            self.category_form.is_processing = false;
        }

        if !can_claim && self.category_form.slots.len() >= 2 {
            ui.label(
                RichText::new("Fill in all slot IDs and questions to claim")
                    .weak()
                    .italics(),
            );
        }
    }
}
