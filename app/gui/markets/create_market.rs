use std::collections::{HashMap, HashSet};

use eframe::egui::{self, Button, RichText, ScrollArea};
use truthcoin_dc::state::markets::DEFAULT_TRADING_FEE;
use truthcoin_dc::state::voting::types::VotingPeriodId;
use truthcoin_dc::wallet::CreateMarketInput;

use crate::app::App;

const DEFAULT_LIQUIDITY_SATS: u64 = 10_000;

#[derive(Clone)]
struct ClaimedSlotInfo {
    slot_id_hex: String,
    question: String,
    is_scaled: bool,
    #[allow(dead_code)]
    is_standard: bool,
    claiming_txid: [u8; 32],
}

#[derive(Clone, Default)]
struct Dimension {
    is_categorical: bool,
    slot_ids: Vec<String>,
    category_txid: String,
    residual_name: String,
}

pub struct CreateMarket {
    title: String,
    description: String,
    trading_fee_input: String,
    initial_liquidity_input: String,
    tags_input: String,
    dimensions: Vec<Dimension>,
    claimed_slots: Vec<ClaimedSlotInfo>,
    slots_loaded: bool,
    category_slots: HashMap<String, Vec<String>>,
    is_processing: bool,
    error: Option<String>,
    success_message: Option<String>,
}

impl Default for CreateMarket {
    fn default() -> Self {
        Self {
            title: String::new(),
            description: String::new(),
            trading_fee_input: String::new(),
            initial_liquidity_input: DEFAULT_LIQUIDITY_SATS.to_string(),
            tags_input: String::new(),
            dimensions: vec![Dimension::default()], // Start with one dimension
            claimed_slots: Vec::new(),
            slots_loaded: false,
            category_slots: HashMap::new(),
            is_processing: false,
            error: None,
            success_message: None,
        }
    }
}

impl CreateMarket {
    fn reset(&mut self) {
        self.title.clear();
        self.description.clear();
        self.trading_fee_input.clear();
        self.initial_liquidity_input = DEFAULT_LIQUIDITY_SATS.to_string();
        self.tags_input.clear();
        self.dimensions = vec![Dimension::default()];
        self.is_processing = false;
        self.error = None;
    }

    fn load_claimed_slots(&mut self, app: &App) {
        self.claimed_slots.clear();
        self.category_slots.clear();

        let periods = match app.node.get_all_slot_quarters() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    "Failed to load periods for slot dropdown: {e:#}"
                );
                self.slots_loaded = true;
                return;
            }
        };

        for (period, _) in periods {
            let period_id = VotingPeriodId::new(period);
            match app.node.get_claimed_slots_in_period(period_id) {
                Ok(slots) => {
                    for slot in slots {
                        if let Some(decision) = &slot.decision {
                            let slot_id_hex =
                                hex::encode(slot.slot_id.as_bytes());
                            let claiming_txid_bytes: [u8; 32] =
                                slot.claiming_txid.into();

                            self.claimed_slots.push(ClaimedSlotInfo {
                                slot_id_hex: slot_id_hex.clone(),
                                question: decision.question.clone(),
                                is_scaled: decision.is_scaled,
                                is_standard: decision.is_standard,
                                claiming_txid: claiming_txid_bytes,
                            });

                            let txid_hex = hex::encode(claiming_txid_bytes);
                            self.category_slots
                                .entry(txid_hex)
                                .or_default()
                                .push(slot_id_hex);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load slots for period {period}: {e:#}"
                    );
                }
            }
        }

        self.category_slots.retain(|_, slots| slots.len() >= 2);

        self.slots_loaded = true;
    }

    fn build_dimensions_string(&self) -> Option<String> {
        if self.dimensions.is_empty() {
            return None;
        }

        let parts: Vec<String> = self
            .dimensions
            .iter()
            .filter_map(|dim| {
                if dim.slot_ids.is_empty()
                    || dim.slot_ids.iter().all(|s| s.is_empty())
                {
                    return None;
                }

                let non_empty: Vec<&String> =
                    dim.slot_ids.iter().filter(|s| !s.is_empty()).collect();

                if non_empty.is_empty() {
                    return None;
                }

                if dim.is_categorical && non_empty.len() >= 2 {
                    // Categorical: [slot1,slot2,slot3]
                    Some(format!(
                        "[{}]",
                        non_empty
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(",")
                    ))
                } else if !non_empty.is_empty() {
                    // Single: just the slot id
                    Some(non_empty[0].clone())
                } else {
                    None
                }
            })
            .collect();

        if parts.is_empty() {
            None
        } else {
            Some(format!("[{}]", parts.join(",")))
        }
    }

    fn calculate_outcome_count(&self) -> usize {
        let mut count = 1usize;

        for dim in &self.dimensions {
            let non_empty_slots =
                dim.slot_ids.iter().filter(|s| !s.is_empty()).count();
            if non_empty_slots == 0 {
                continue;
            }

            if dim.is_categorical && non_empty_slots >= 2 {
                count *= non_empty_slots;
            } else {
                count *= 2;
            }
        }

        count
    }

    fn create_market(&mut self, app: &App) {
        self.error = None;
        self.success_message = None;

        let dimensions = self.build_dimensions_string();

        if dimensions.is_none() {
            self.error = Some(
                "At least one dimension with a slot is required".to_string(),
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

        let tags = if self.tags_input.is_empty() {
            None
        } else {
            Some(
                self.tags_input
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            )
        };

        let category_txids: Option<Vec<[u8; 32]>> = {
            let txids: Vec<[u8; 32]> = self
                .dimensions
                .iter()
                .filter(|d| {
                    d.is_categorical
                        && d.slot_ids.len() >= 2
                        && !d.category_txid.is_empty()
                })
                .filter_map(|d| {
                    hex::decode(&d.category_txid).ok().and_then(|bytes| {
                        if bytes.len() == 32 {
                            let mut arr = [0u8; 32];
                            arr.copy_from_slice(&bytes);
                            Some(arr)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            if txids.is_empty() { None } else { Some(txids) }
        };

        let residual_names: Option<Vec<String>> = {
            let names: Vec<String> = self
                .dimensions
                .iter()
                .filter(|d| d.is_categorical && d.slot_ids.len() >= 2)
                .map(|d| d.residual_name.clone())
                .filter(|s| !s.is_empty())
                .collect();
            if names.is_empty() { None } else { Some(names) }
        };

        let dimensions_str = dimensions.unwrap();
        tracing::debug!("Creating market with dimensions: {}", dimensions_str);

        let input = CreateMarketInput {
            title: self.title.clone(),
            description: self.description.clone(),
            dimensions: dimensions_str,
            beta: None, // Derived from initial_liquidity in wallet
            trading_fee,
            tags,
            initial_liquidity,
            category_txids,
            residual_names,
        };

        let tx_fee = bitcoin::Amount::from_sat(1000);

        match app.wallet.create_market(input, tx_fee) {
            Ok(tx) => {
                if let Err(e) = app.sign_and_send(tx) {
                    self.error = Some(format!("Failed to send: {e:#}"));
                    tracing::error!("Create market failed: {e:#}");
                } else {
                    tracing::info!("Market created: {}", self.title);
                    self.success_message = Some(format!(
                        "Market '{}' created successfully!",
                        self.title
                    ));
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

        // Always load fresh slots from node to catch new blocks
        self.load_claimed_slots(app);

        ui.heading("Create Prediction Market");
        ui.separator();

        // Success message
        if let Some(msg) = &self.success_message {
            ui.colored_label(egui::Color32::GREEN, msg);
            ui.add_space(10.0);
        }

        ScrollArea::vertical().show(ui, |ui| {
            // Basic info
            egui::Grid::new("create_market_basic")
                .num_columns(2)
                .spacing([10.0, 8.0])
                .show(ui, |ui| {
                    ui.label("Title:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.title)
                            .hint_text("e.g., Crypto Predictions 2025")
                            .desired_width(400.0),
                    );
                    ui.end_row();

                    ui.label("Description:");
                    ui.add(
                        egui::TextEdit::multiline(&mut self.description)
                            .hint_text("Detailed description...")
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

            ui.heading("Market Dimensions");
            ui.label(
                RichText::new("Build your market by adding dimensions. Each dimension can be a single decision or a categorical choice between multiple options.")
                    .weak()
                    .small(),
            );
            ui.add_space(10.0);

            let mut dim_to_remove: Option<usize> = None;
            let mut slot_to_remove: Option<(usize, usize)> = None;
            let mut add_slot_to_dim: Option<usize> = None;
            let mut switch_to_single: Option<usize> = None;
            let mut use_category: Option<(usize, String, Vec<String>)> = None;

            let dims_count = self.dimensions.len();

            let used_slots: HashSet<String> = self
                .dimensions
                .iter()
                .flat_map(|d| d.slot_ids.iter())
                .filter(|s| !s.is_empty())
                .cloned()
                .collect();

            for dim_idx in 0..dims_count {
                let dim = &self.dimensions[dim_idx];
                let is_categorical = dim.is_categorical;
                let slots_count = dim.slot_ids.len();

                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("Dimension {}", dim_idx + 1)).strong());
                        if is_categorical {
                            ui.label(RichText::new("(Categorical)").weak().small());
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Remove").clicked() && dims_count > 1 {
                                dim_to_remove = Some(dim_idx);
                            }
                            if is_categorical && ui.small_button("Make Single").clicked() {
                                switch_to_single = Some(dim_idx);
                            }
                        });
                    });

                    if is_categorical {
                        ui.label(
                            RichText::new("Mutually exclusive options (one will be true)")
                                .weak()
                                .small(),
                        );
                    }

                    ui.add_space(5.0);

                    let actual_slots_count = slots_count.max(1);
                    for slot_idx in 0..actual_slots_count {
                        let slot_id_value = self.dimensions[dim_idx]
                            .slot_ids
                            .get(slot_idx)
                            .cloned()
                            .unwrap_or_default();

                        ui.horizontal(|ui| {
                            if is_categorical {
                                ui.label(format!("  Option {}:", slot_idx + 1));
                            } else {
                                ui.label("  Slot:");
                            }

                            let selected_text = if slot_id_value.is_empty() {
                                "Select a slot...".to_string()
                            } else {
                                self.claimed_slots
                                    .iter()
                                    .find(|s| s.slot_id_hex == slot_id_value)
                                    .map(|s| {
                                        let q = if s.question.len() > 30 {
                                            format!("{}...", &s.question[..30])
                                        } else {
                                            s.question.clone()
                                        };
                                        format!("{slot_id_value}: {q}")
                                    })
                                    .unwrap_or_else(|| slot_id_value.clone())
                            };

                            let mut new_value = slot_id_value.clone();
                            egui::ComboBox::from_id_salt(format!("slot_{dim_idx}_{slot_idx}"))
                                .selected_text(selected_text)
                                .width(350.0)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut new_value, String::new(), "-- None --");
                                    for claimed in &self.claimed_slots {
                                        if used_slots.contains(&claimed.slot_id_hex)
                                            && claimed.slot_id_hex != slot_id_value
                                        {
                                            continue;
                                        }
                                        let label = format!(
                                            "{} - {}{}",
                                            claimed.slot_id_hex,
                                            if claimed.question.len() > 40 {
                                                format!("{}...", &claimed.question[..40])
                                            } else {
                                                claimed.question.clone()
                                            },
                                            if claimed.is_scaled { " [Scaled]" } else { "" }
                                        );
                                        ui.selectable_value(
                                            &mut new_value,
                                            claimed.slot_id_hex.clone(),
                                            label,
                                        );
                                    }
                                });

                            if new_value != slot_id_value {
                                while self.dimensions[dim_idx].slot_ids.len() <= slot_idx {
                                    self.dimensions[dim_idx].slot_ids.push(String::new());
                                }
                                self.dimensions[dim_idx].slot_ids[slot_idx] = new_value.clone();
                            }

                            if !new_value.is_empty()
                                && !is_categorical
                                && let Some(slot_info) = self.claimed_slots.iter().find(|s| s.slot_id_hex == new_value)
                            {
                                let txid_hex = hex::encode(slot_info.claiming_txid);
                                if let Some(category_slot_ids) = self.category_slots.get(&txid_hex)
                                    && ui.small_button("Populate category").clicked()
                                {
                                    use_category = Some((dim_idx, txid_hex, category_slot_ids.clone()));
                                }
                            }

                            if is_categorical
                                && slots_count > 2
                                && ui.small_button("×").clicked()
                            {
                                slot_to_remove = Some((dim_idx, slot_idx));
                            }
                        });
                    }

                    if is_categorical && ui.small_button("+ Add Option").clicked() {
                        add_slot_to_dim = Some(dim_idx);
                    }

                    let category_txid = &self.dimensions[dim_idx].category_txid;
                    if !category_txid.is_empty() {
                        let slot_count = self.dimensions[dim_idx].slot_ids.len();
                        ui.horizontal(|ui| {
                            ui.label("✓");
                            ui.label(
                                RichText::new(format!(
                                    "Using category: {}... ({} slots)",
                                    &category_txid[..12.min(category_txid.len())],
                                    slot_count
                                ))
                                .weak(),
                            );
                        });
                    }

                    if is_categorical && slots_count >= 2 {
                        ui.add_space(5.0);
                        ui.horizontal(|ui| {
                            ui.label("Residual Name:");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.dimensions[dim_idx].residual_name)
                                    .hint_text("e.g., 'Other' or specific option")
                                    .desired_width(200.0),
                            );
                        });
                        ui.label(
                            RichText::new("Optional: name for 'none of the above' outcome")
                                .weak()
                                .small(),
                        );
                    }
                });
                ui.add_space(5.0);
            }

            if let Some(dim_idx) = switch_to_single {
                self.dimensions[dim_idx].is_categorical = false;
                self.dimensions[dim_idx].slot_ids.truncate(1);
                self.dimensions[dim_idx].category_txid.clear();
            }
            if let Some(dim_idx) = add_slot_to_dim {
                self.dimensions[dim_idx].slot_ids.push(String::new());
            }
            if let Some((dim_idx, slot_idx)) = slot_to_remove {
                self.dimensions[dim_idx].slot_ids.remove(slot_idx);
            }
            if let Some(dim_idx) = dim_to_remove {
                self.dimensions.remove(dim_idx);
            }
            if let Some((dim_idx, txid, slot_ids)) = use_category {
                self.dimensions[dim_idx].is_categorical = true;
                self.dimensions[dim_idx].slot_ids = slot_ids;
                self.dimensions[dim_idx].category_txid = txid;
            }

            if ui.button("+ Add Dimension").clicked() {
                self.dimensions.push(Dimension::default());
            }

            ui.add_space(10.0);

            let dims_str = self.build_dimensions_string();
            let outcome_count = self.calculate_outcome_count();

            ui.group(|ui| {
                ui.label(RichText::new("Preview").strong());
                ui.horizontal(|ui| {
                    ui.label("Dimensions:");
                    ui.monospace(dims_str.as_deref().unwrap_or("(none)"));
                });
                ui.horizontal(|ui| {
                    ui.label("Valid Outcomes:");
                    ui.label(format!("{outcome_count}"));
                });
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

                        ui.label("Tags:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.tags_input)
                                .hint_text("crypto, prediction, 2025")
                                .desired_width(300.0),
                        );
                        ui.end_row();
                    });
            });

            ui.add_space(15.0);

            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, err);
                ui.add_space(10.0);
            }

            ui.horizontal(|ui| {
                let has_valid_dims = dims_str.is_some();
                let can_create = !self.title.is_empty() && has_valid_dims && !self.is_processing;

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

                if ui.button("Refresh Slots").clicked() {
                    self.slots_loaded = false;
                    self.load_claimed_slots(app);
                }
            });

            if self.claimed_slots.is_empty() {
                ui.add_space(10.0);
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "No claimed slots found. Go to the Slots tab to claim decision slots first.",
                );
            }
        });
    }
}
