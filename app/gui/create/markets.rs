use std::collections::{HashMap, HashSet};

use eframe::egui::{self, Button, RichText, ScrollArea};
use truthcoin_dc::state::markets::DEFAULT_TRADING_FEE;
use truthcoin_dc::state::voting::types::VotingPeriodId;
use truthcoin_dc::wallet::CreateMarketInput;

use crate::app::App;

const DEFAULT_LIQUIDITY_SATS: u64 = 10_000;

#[derive(Clone)]
struct ClaimedDecisionInfo {
    decision_id_hex: String,
    header: String,
    is_scaled: bool,
    is_categorical: bool,
    option_count: usize,
    claiming_txid: [u8; 32],
}

#[derive(Clone, Default)]
struct Dimension {
    is_categorical: bool,
    decision_ids: Vec<String>,
    category_txid: String,
    residual_name: String,
}

pub struct CreateMarket {
    title: String,
    description: String,
    trading_fee_input: String,
    initial_liquidity_input: String,
    dimensions: Vec<Dimension>,
    claimed_decisions: Vec<ClaimedDecisionInfo>,
    decisions_loaded: bool,
    category_decisions: HashMap<String, Vec<String>>,
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
            dimensions: vec![Dimension::default()],
            claimed_decisions: Vec::new(),
            decisions_loaded: false,
            category_decisions: HashMap::new(),
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
        self.dimensions = vec![Dimension::default()];
        self.is_processing = false;
        self.error = None;
        self.tx_pow_enabled = false;
        self.tx_pow_hash_selector = 0b0000_0001;
        self.tx_pow_ordering_input = String::from("0");
        self.tx_pow_difficulty_input = String::from("8");
    }

    fn load_claimed_decisions(&mut self, app: &App) {
        self.claimed_decisions.clear();
        self.category_decisions.clear();

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
                            let decision_id_hex =
                                hex::encode(entry.decision_id.as_bytes());
                            let claiming_txid_bytes: [u8; 32] =
                                entry.claiming_txid.into();

                            self.claimed_decisions.push(ClaimedDecisionInfo {
                                decision_id_hex: decision_id_hex.clone(),
                                header: decision.header.clone(),
                                is_scaled: decision.is_scaled(),
                                is_categorical: decision.is_categorical(),
                                option_count: decision
                                    .option_count()
                                    .unwrap_or(2),
                                claiming_txid: claiming_txid_bytes,
                            });

                            let txid_hex = hex::encode(claiming_txid_bytes);
                            self.category_decisions
                                .entry(txid_hex)
                                .or_default()
                                .push(decision_id_hex);
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

        self.category_decisions
            .retain(|_, decisions| decisions.len() >= 2);

        self.decisions_loaded = true;
    }

    fn build_dimensions_string(&self) -> Option<String> {
        if self.dimensions.is_empty() {
            return None;
        }

        let parts: Vec<String> = self
            .dimensions
            .iter()
            .filter_map(|dim| {
                if dim.decision_ids.is_empty()
                    || dim.decision_ids.iter().all(|s| s.is_empty())
                {
                    return None;
                }

                let non_empty: Vec<&String> =
                    dim.decision_ids.iter().filter(|s| !s.is_empty()).collect();

                if non_empty.is_empty() {
                    return None;
                }

                if dim.is_categorical && non_empty.len() >= 2 {
                    Some(format!(
                        "[{}]",
                        non_empty
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(",")
                    ))
                } else if non_empty.len() == 1
                    && self.is_categorical_decision(non_empty[0])
                {
                    Some(format!("[{}]", non_empty[0]))
                } else {
                    Some(non_empty[0].clone())
                }
            })
            .collect();

        if parts.is_empty() {
            None
        } else {
            Some(format!("[{}]", parts.join(",")))
        }
    }

    fn is_categorical_decision(&self, decision_id_hex: &str) -> bool {
        self.claimed_decisions
            .iter()
            .find(|d| d.decision_id_hex == decision_id_hex)
            .map(|d| d.is_categorical)
            .unwrap_or(false)
    }

    fn option_count_for(&self, decision_id_hex: &str) -> Option<usize> {
        self.claimed_decisions
            .iter()
            .find(|d| d.decision_id_hex == decision_id_hex)
            .filter(|d| d.is_categorical)
            .map(|d| d.option_count)
    }

    fn calculate_outcome_count(&self) -> usize {
        let mut count = 1usize;

        for dim in &self.dimensions {
            let non_empty: Vec<&String> =
                dim.decision_ids.iter().filter(|s| !s.is_empty()).collect();
            if non_empty.is_empty() {
                continue;
            }

            if dim.is_categorical && non_empty.len() >= 2 {
                count *= non_empty.len();
            } else if non_empty.len() == 1
                && let Some(n) = self.option_count_for(non_empty[0])
            {
                count *= n;
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
            if let Ok(specs) = parse_dimensions(&dimensions_str) {
                for spec in &specs {
                    if let DimensionSpec::Categorical(id) = spec {
                        let n = app
                            .node
                            .get_decision_entry(*id)
                            .ok()
                            .flatten()
                            .and_then(|e| e.decision)
                            .and_then(|d| d.option_count())
                            .unwrap_or(2);
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
            dimensions: dimensions_str,
            beta: None,
            trading_fee,
            initial_liquidity,
            category_option_counts,
            tx_pow_hash_selector: pow_selector,
            tx_pow_ordering: pow_ordering,
            tx_pow_difficulty: pow_difficulty,
        };

        let tx_fee = bitcoin::Amount::from_sat(1000);

        match app.wallet.create_market(input, tx_fee) {
            Ok((tx, _market_id)) => {
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

        self.load_claimed_decisions(app);

        ui.heading("Create Prediction Market");
        ui.separator();

        if let Some(msg) = &self.success_message {
            ui.colored_label(egui::Color32::GREEN, msg);
            ui.add_space(10.0);
        }

        ScrollArea::vertical().show(ui, |ui| {
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
            let mut decision_to_remove: Option<(usize, usize)> = None;
            let mut add_decision_to_dim: Option<usize> = None;
            let mut switch_to_single: Option<usize> = None;
            let mut use_category: Option<(usize, String, Vec<String>)> = None;

            let dims_count = self.dimensions.len();

            let used_decisions: HashSet<String> = self
                .dimensions
                .iter()
                .flat_map(|d| d.decision_ids.iter())
                .filter(|s| !s.is_empty())
                .cloned()
                .collect();

            for dim_idx in 0..dims_count {
                let dim = &self.dimensions[dim_idx];
                let is_categorical = dim.is_categorical;
                let decisions_count = dim.decision_ids.len();

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

                    let actual_decisions_count = decisions_count.max(1);
                    for decision_idx in 0..actual_decisions_count {
                        let decision_id_value = self.dimensions[dim_idx]
                            .decision_ids
                            .get(decision_idx)
                            .cloned()
                            .unwrap_or_default();

                        ui.horizontal(|ui| {
                            if is_categorical {
                                ui.label(format!("  Option {}:", decision_idx + 1));
                            } else {
                                ui.label("  Decision:");
                            }

                            let selected_text = if decision_id_value.is_empty() {
                                "Select a decision...".to_string()
                            } else {
                                self.claimed_decisions
                                    .iter()
                                    .find(|s| s.decision_id_hex == decision_id_value)
                                    .map(|s| {
                                        let q = if s.header.len() > 30 {
                                            format!("{}...", &s.header[..30])
                                        } else {
                                            s.header.clone()
                                        };
                                        format!("{decision_id_value}: {q}")
                                    })
                                    .unwrap_or_else(|| decision_id_value.clone())
                            };

                            let mut new_value = decision_id_value.clone();
                            egui::ComboBox::from_id_salt(format!("decision_{dim_idx}_{decision_idx}"))
                                .selected_text(selected_text)
                                .width(350.0)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut new_value, String::new(), "-- None --");
                                    for claimed in &self.claimed_decisions {
                                        if used_decisions.contains(&claimed.decision_id_hex)
                                            && claimed.decision_id_hex != decision_id_value
                                        {
                                            continue;
                                        }
                                        let kind_suffix = if claimed.is_scaled {
                                            " [Scaled]".to_string()
                                        } else if claimed.is_categorical {
                                            format!(" [Categorical: {}]", claimed.option_count)
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
                                            kind_suffix
                                        );
                                        ui.selectable_value(
                                            &mut new_value,
                                            claimed.decision_id_hex.clone(),
                                            label,
                                        );
                                    }
                                });

                            if new_value != decision_id_value {
                                while self.dimensions[dim_idx].decision_ids.len() <= decision_idx {
                                    self.dimensions[dim_idx].decision_ids.push(String::new());
                                }
                                self.dimensions[dim_idx].decision_ids[decision_idx] = new_value.clone();
                            }

                            if !new_value.is_empty()
                                && !is_categorical
                                && let Some(decision_info) = self.claimed_decisions.iter().find(|s| s.decision_id_hex == new_value)
                            {
                                let txid_hex = hex::encode(decision_info.claiming_txid);
                                if let Some(category_decision_ids) = self.category_decisions.get(&txid_hex)
                                    && ui.small_button("Populate category").clicked()
                                {
                                    use_category = Some((dim_idx, txid_hex, category_decision_ids.clone()));
                                }
                            }

                            if is_categorical
                                && decisions_count > 2
                                && ui.small_button("×").clicked()
                            {
                                decision_to_remove = Some((dim_idx, decision_idx));
                            }
                        });
                    }

                    if is_categorical && ui.small_button("+ Add Option").clicked() {
                        add_decision_to_dim = Some(dim_idx);
                    }

                    let category_txid = &self.dimensions[dim_idx].category_txid;
                    if !category_txid.is_empty() {
                        let decision_count = self.dimensions[dim_idx].decision_ids.len();
                        ui.horizontal(|ui| {
                            ui.label("✓");
                            ui.label(
                                RichText::new(format!(
                                    "Using category: {}... ({} decisions)",
                                    &category_txid[..12.min(category_txid.len())],
                                    decision_count
                                ))
                                .weak(),
                            );
                        });
                    }

                    if is_categorical && decisions_count >= 2 {
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
                self.dimensions[dim_idx].decision_ids.truncate(1);
                self.dimensions[dim_idx].category_txid.clear();
            }
            if let Some(dim_idx) = add_decision_to_dim {
                self.dimensions[dim_idx].decision_ids.push(String::new());
            }
            if let Some((dim_idx, decision_idx)) = decision_to_remove {
                self.dimensions[dim_idx].decision_ids.remove(decision_idx);
            }
            if let Some(dim_idx) = dim_to_remove {
                self.dimensions.remove(dim_idx);
            }
            if let Some((dim_idx, txid, decision_ids)) = use_category {
                self.dimensions[dim_idx].is_categorical = true;
                self.dimensions[dim_idx].decision_ids = decision_ids;
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
