use std::pin::Pin;
use std::task::Poll;

use eframe::egui::{self, Color32, RichText, ScrollArea};
use futures::Stream;
use truthcoin_dc::state::decisions::{Decision, DecisionId, DecisionType};
use truthcoin_dc::types::BallotItem;

use crate::util::PromiseStream;
use crate::{app::App, gui::util::UiExt};

type NodeUpdates = PromiseStream<Pin<Box<dyn Stream<Item = ()> + Send>>>;

pub(super) struct Ballot {
    rows: Vec<BallotRow>,
    fee_sats: String,
    error: Option<String>,
    last_submit: Option<SubmitResult>,
    current_period: u32,
    node_updated: Option<NodeUpdates>,
    needs_initial_load: bool,
}

struct BallotRow {
    decision_id: DecisionId,
    decision: Decision,
    value: RowValue,
    abstain: bool,
}

enum RowValue {
    Binary(Option<bool>),
    Scaled { text: String },
    Categorical(u16),
}

struct SubmitResult {
    txid: String,
    period: u32,
    vote_count: usize,
}

impl Default for Ballot {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            fee_sats: String::new(),
            error: None,
            last_submit: None,
            current_period: 0,
            node_updated: None,
            needs_initial_load: true,
        }
    }
}

impl Ballot {
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

    fn refresh(&mut self, app: &App, current_period: u32) {
        self.ensure_subscribed(app);
        let period_changed = self.current_period != current_period;
        if !period_changed && !self.needs_initial_load && !self.tip_changed() {
            return;
        }
        self.needs_initial_load = false;

        let rotxn = match app.node.read_txn() {
            Ok(t) => t,
            Err(err) => {
                self.error = Some(format!("Failed to read state: {err:#}"));
                return;
            }
        };
        let claim_period = current_period.saturating_sub(1);
        let entries = match app
            .node
            .state()
            .decisions()
            .get_claimed_decisions_in_period(&rotxn, claim_period)
        {
            Ok(entries) => entries,
            Err(err) => {
                self.error = Some(format!("Failed to load decisions: {err:#}"));
                return;
            }
        };

        if period_changed {
            self.rows.clear();
            self.current_period = current_period;
        }

        let existing_ids: std::collections::HashSet<DecisionId> =
            self.rows.iter().map(|r| r.decision_id).collect();
        let current_ids: std::collections::HashSet<DecisionId> =
            entries.iter().map(|e| e.decision_id).collect();

        self.rows.retain(|r| current_ids.contains(&r.decision_id));

        for entry in entries {
            if existing_ids.contains(&entry.decision_id) {
                continue;
            }
            let Some(decision) = entry.decision else {
                continue;
            };
            let value = match &decision.decision_type {
                DecisionType::Binary => RowValue::Binary(None),
                DecisionType::Scaled { .. } => RowValue::Scaled {
                    text: String::new(),
                },
                DecisionType::Category { .. } => RowValue::Categorical(0),
            };
            self.rows.push(BallotRow {
                decision_id: entry.decision_id,
                decision,
                value,
                abstain: false,
            });
        }

        self.error = None;
    }

    pub fn show(
        &mut self,
        app: &App,
        ui: &mut egui::Ui,
        current_period: u32,
        total_reputation: f64,
    ) {
        self.refresh(app, current_period);

        ScrollArea::vertical()
            .id_salt("voter_ballot_scroll")
            .show(ui, |ui| {
                ui.heading(format!("Ballot for Period {current_period}"));
                ui.add_space(8.0);

                if let Some(err) = &self.error {
                    ui.colored_label(Color32::RED, err);
                    ui.add_space(6.0);
                }

                if let Some(result) = &self.last_submit {
                    egui::Frame::new()
                        .fill(Color32::from_rgb(40, 80, 40))
                        .corner_radius(6.0)
                        .inner_margin(10.0)
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(format!(
                                    "Submitted {} votes for period {}",
                                    result.vote_count, result.period
                                ))
                                .strong()
                                .color(Color32::WHITE),
                            );
                            ui.horizontal(|ui| {
                                ui.label("txid:");
                                ui.monospace_selectable_singleline(
                                    false,
                                    result.txid.clone(),
                                );
                            });
                        });
                    ui.add_space(8.0);
                }

                let has_reputation = total_reputation > 0.0;
                if !has_reputation {
                    egui::Frame::new()
                        .fill(ui.visuals().widgets.noninteractive.bg_fill)
                        .corner_radius(6.0)
                        .inner_margin(12.0)
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(
                                    "You need Votecoin reputation in a \
                                     wallet address to cast a ballot.",
                                )
                                .italics()
                                .color(Color32::from_rgb(200, 120, 120)),
                            );
                        });
                    ui.add_space(10.0);
                }

                if self.rows.is_empty() {
                    egui::Frame::new()
                        .fill(ui.visuals().widgets.noninteractive.bg_fill)
                        .corner_radius(6.0)
                        .inner_margin(12.0)
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(
                                    "No claimed decisions are open for \
                                     voting in this period.",
                                )
                                .italics()
                                .weak(),
                            );
                        });
                    return;
                }

                ui.add_enabled_ui(has_reputation, |ui| {
                    for row_idx in 0..self.rows.len() {
                        show_row(ui, &mut self.rows[row_idx], row_idx);
                        ui.add_space(6.0);
                    }

                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(8.0);

                    ui.horizontal(|ui| {
                        ui.label("Fee (sats):");
                        let fee_edit =
                            egui::TextEdit::singleline(&mut self.fee_sats)
                                .hint_text("e.g. 1000")
                                .desired_width(100.0);
                        ui.add(fee_edit);
                    });

                    let (enabled, disabled_reason) =
                        self.submit_gate(total_reputation);

                    let submit_button = egui::Button::new(
                        RichText::new("Submit Ballot").strong(),
                    );
                    let response = ui.add_enabled(enabled, submit_button);

                    if !enabled && let Some(reason) = &disabled_reason {
                        ui.label(
                            RichText::new(reason.as_str())
                                .small()
                                .weak()
                                .italics(),
                        );
                    }

                    if response.clicked() && enabled {
                        self.submit(app);
                    }
                });
            });
    }

    fn submit_gate(&self, total_reputation: f64) -> (bool, Option<String>) {
        if total_reputation <= 0.0 {
            return (false, Some("No voting reputation".to_string()));
        }
        if self.fee_sats.parse::<u64>().is_err() {
            return (false, Some("Fee must be an integer in sats".to_string()));
        }
        let non_abstain: Vec<&BallotRow> =
            self.rows.iter().filter(|r| !r.abstain).collect();
        if non_abstain.is_empty() {
            return (
                false,
                Some(
                    "At least one row must be voted (not Abstain)".to_string(),
                ),
            );
        }
        for row in &non_abstain {
            if let Err(reason) = validate_row(row) {
                return (false, Some(reason));
            }
        }
        (true, None)
    }

    fn submit(&mut self, app: &App) {
        self.last_submit = None;
        self.error = None;

        let fee_sats = match self.fee_sats.parse::<u64>() {
            Ok(f) => f,
            Err(_) => {
                self.error = Some("Invalid fee".to_string());
                return;
            }
        };

        let mut items: Vec<BallotItem> = Vec::new();
        for row in self.rows.iter().filter(|r| !r.abstain) {
            let user_value = match row_user_value(row) {
                Ok(v) => v,
                Err(reason) => {
                    self.error = Some(reason);
                    return;
                }
            };
            let normalized =
                match row.decision.validate_and_normalize(user_value) {
                    Ok(v) => v,
                    Err(err) => {
                        self.error = Some(format!("Invalid vote: {err}"));
                        return;
                    }
                };
            items.push(BallotItem {
                decision_id_bytes: row.decision_id.as_bytes(),
                vote_value: normalized,
            });
        }

        if items.is_empty() {
            self.error = Some("No non-abstain votes to submit".to_string());
            return;
        }

        let vote_count = items.len();
        let fee = bitcoin::Amount::from_sat(fee_sats);
        let period = self.current_period;

        let tx = match app.wallet.submit_ballot(items, period, fee) {
            Ok(tx) => tx,
            Err(err) => {
                self.error = Some(format!("Failed to build ballot: {err:#}"));
                return;
            }
        };
        let txid = format!("{}", tx.txid());
        if let Err(err) = app.sign_and_send(tx) {
            self.error = Some(format!("Failed to submit ballot: {err:#}"));
            return;
        }

        self.last_submit = Some(SubmitResult {
            txid,
            period,
            vote_count,
        });
        self.rows.clear();
        self.fee_sats.clear();
        self.current_period = 0;
    }
}

fn show_row(ui: &mut egui::Ui, row: &mut BallotRow, row_idx: usize) {
    egui::Frame::new()
        .fill(ui.visuals().widgets.noninteractive.bg_fill)
        .corner_radius(6.0)
        .inner_margin(12.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.monospace(
                    RichText::new(format!("[{}]", row.decision_id.to_hex()))
                        .small()
                        .color(Color32::GRAY),
                );
                ui.add_space(6.0);
                let type_text = match &row.decision.decision_type {
                    DecisionType::Binary => "Binary",
                    DecisionType::Scaled { .. } => "Scaled",
                    DecisionType::Category { .. } => "Categorical",
                };
                let type_color = match &row.decision.decision_type {
                    DecisionType::Binary => Color32::from_rgb(100, 200, 100),
                    DecisionType::Scaled { .. } => {
                        Color32::from_rgb(100, 149, 237)
                    }
                    DecisionType::Category { .. } => {
                        Color32::from_rgb(200, 170, 100)
                    }
                };
                ui.label(
                    RichText::new(type_text).small().strong().color(type_color),
                );
                ui.add_space(6.0);
                ui.checkbox(&mut row.abstain, "Abstain");
            });

            ui.add_space(4.0);
            let header_label = if row.decision.header.len() > 120 {
                format!("{}…", &row.decision.header[..117])
            } else {
                row.decision.header.clone()
            };
            ui.label(RichText::new(header_label).strong());

            ui.add_space(6.0);
            ui.add_enabled_ui(!row.abstain, |ui| match &mut row.value {
                RowValue::Binary(choice) => {
                    let (label0, label1) = row.decision.get_binary_labels();
                    ui.horizontal(|ui| {
                        if ui
                            .radio(matches!(choice, Some(true)), label1)
                            .clicked()
                        {
                            *choice = Some(true);
                        }
                        if ui
                            .radio(matches!(choice, Some(false)), label0)
                            .clicked()
                        {
                            *choice = Some(false);
                        }
                    });
                }
                RowValue::Scaled { text } => {
                    let (min_i, max_i) = match &row.decision.decision_type {
                        DecisionType::Scaled { min, max } => (*min, *max),
                        _ => (0, 1),
                    };
                    let min_f = min_i as f64;
                    let max_f = max_i as f64;
                    ui.horizontal(|ui| {
                        let edit = egui::TextEdit::singleline(text)
                            .hint_text(format!("[{min_i} – {max_i}]"))
                            .desired_width(120.0);
                        ui.add(edit);
                        let mut slider_val =
                            text.parse::<f64>().unwrap_or(min_f);
                        let response = ui.add(
                            egui::Slider::new(&mut slider_val, min_f..=max_f)
                                .show_value(false),
                        );
                        if response.changed() {
                            *text = format!("{slider_val:.4}");
                        }
                    });
                }
                RowValue::Categorical(idx) => {
                    let Some(options) = row.decision.get_category_labels()
                    else {
                        ui.label(
                            RichText::new("Decision has no options")
                                .color(Color32::RED),
                        );
                        return;
                    };
                    let inconclusive_idx = options.len() as u16;
                    let selected_label: String = if *idx == inconclusive_idx {
                        "Inconclusive".to_string()
                    } else {
                        options
                            .get(*idx as usize)
                            .cloned()
                            .unwrap_or_else(|| format!("Option {idx}"))
                    };
                    egui::ComboBox::from_id_salt(format!(
                        "voter_cat_{row_idx}"
                    ))
                    .selected_text(selected_label)
                    .show_ui(ui, |ui| {
                        for (i, label) in options.iter().enumerate() {
                            let i16 = i as u16;
                            ui.selectable_value(idx, i16, label);
                        }
                        ui.selectable_value(
                            idx,
                            inconclusive_idx,
                            "Inconclusive",
                        );
                    });
                }
            });
        });
}

fn row_user_value(row: &BallotRow) -> Result<f64, String> {
    match &row.value {
        RowValue::Binary(Some(true)) => Ok(1.0),
        RowValue::Binary(Some(false)) => Ok(0.0),
        RowValue::Binary(None) => {
            Err(format!("Row {} has no choice", row.decision_id.to_hex()))
        }
        RowValue::Scaled { text } => text.parse::<f64>().map_err(|_| {
            format!(
                "Row {} has invalid numeric value",
                row.decision_id.to_hex()
            )
        }),
        RowValue::Categorical(idx) => Ok(*idx as f64),
    }
}

fn validate_row(row: &BallotRow) -> Result<(), String> {
    let value = row_user_value(row)?;
    row.decision
        .validate_and_normalize(value)
        .map(|_| ())
        .map_err(|e| format!("Row {}: {e}", row.decision_id.to_hex()))
}
