use std::pin::Pin;
use std::task::Poll;

use eframe::egui::{self, Color32, RichText, ScrollArea};
use futures::Stream;
use truthcoin_dc::state::decisions::{Decision, DecisionId, DecisionType};
use truthcoin_dc::state::voting::types::{VoteValue, VotingPeriodId};

use crate::app::App;
use crate::util::PromiseStream;

type NodeUpdates = PromiseStream<Pin<Box<dyn Stream<Item = ()> + Send>>>;

pub(super) struct History {
    entries: Vec<HistoryEntry>,
    error: Option<String>,
    last_period: u32,
    node_updated: Option<NodeUpdates>,
    needs_initial_load: bool,
}

struct HistoryEntry {
    period: VotingPeriodId,
    decision_id: DecisionId,
    header: String,
    type_label: String,
    type_color: Color32,
    display_value: String,
    value_color: Color32,
    block_height: u32,
}

impl Default for History {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            error: None,
            last_period: u32::MAX,
            node_updated: None,
            needs_initial_load: true,
        }
    }
}

impl History {
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
        let period_changed = self.last_period != current_period;
        if !period_changed && !self.needs_initial_load && !self.tip_changed() {
            return;
        }
        self.needs_initial_load = false;
        self.last_period = current_period;

        let addresses = match app.wallet.get_addresses() {
            Ok(addrs) => addrs,
            Err(err) => {
                self.error = Some(format!("Failed to load addresses: {err:#}"));
                return;
            }
        };

        let rotxn = match app.node.read_txn() {
            Ok(txn) => txn,
            Err(err) => {
                self.error = Some(format!("Failed to read state: {err:#}"));
                return;
            }
        };
        let state = app.node.state();

        let voting = state.voting().databases();
        let mut entries: Vec<HistoryEntry> = Vec::new();

        for addr in addresses {
            let votes = match voting.get_votes_by_voter(&rotxn, addr) {
                Ok(v) => v,
                Err(err) => {
                    self.error = Some(format!("Failed to read votes: {err:#}"));
                    return;
                }
            };
            for (key, entry) in votes {
                let decision_entry = match state
                    .decisions()
                    .get_decision_entry(&rotxn, key.decision_id)
                {
                    Ok(Some(e)) => Some(e),
                    Ok(None) => None,
                    Err(err) => {
                        tracing::warn!(
                            "Failed to load decision {}: {err:#}",
                            key.decision_id.to_hex()
                        );
                        None
                    }
                };

                let (
                    header,
                    type_label,
                    type_color,
                    display_value,
                    value_color,
                ) = match decision_entry.and_then(|e| e.decision) {
                    Some(decision) => {
                        let header = truncate(&decision.header, 80);
                        let (type_label, type_color) =
                            type_of(&decision.decision_type);
                        let (display_value, value_color) =
                            format_vote(&entry.value, &decision);
                        (
                            header,
                            type_label.to_string(),
                            type_color,
                            display_value,
                            value_color,
                        )
                    }
                    None => {
                        let (display_value, value_color) =
                            format_vote_unknown(&entry.value);
                        (
                            "(decision unavailable)".to_string(),
                            "?".to_string(),
                            Color32::GRAY,
                            display_value,
                            value_color,
                        )
                    }
                };

                entries.push(HistoryEntry {
                    period: key.period_id,
                    decision_id: key.decision_id,
                    header,
                    type_label,
                    type_color,
                    display_value,
                    value_color,
                    block_height: entry.block_height,
                });
            }
        }

        entries.sort_by(|a, b| {
            b.period
                .as_u32()
                .cmp(&a.period.as_u32())
                .then_with(|| b.block_height.cmp(&a.block_height))
        });
        self.entries = entries;
        self.error = None;
    }

    pub fn show(&mut self, app: &App, ui: &mut egui::Ui, current_period: u32) {
        self.refresh(app, current_period);

        ScrollArea::vertical()
            .id_salt("voter_history_scroll")
            .show(ui, |ui| {
                ui.heading("My Voting History");
                ui.add_space(8.0);

                if let Some(err) = &self.error {
                    ui.colored_label(Color32::RED, err);
                    ui.add_space(6.0);
                }

                if self.entries.is_empty() {
                    egui::Frame::new()
                        .fill(ui.visuals().widgets.noninteractive.bg_fill)
                        .corner_radius(6.0)
                        .inner_margin(12.0)
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(
                                    "No votes recorded for your wallet addresses.",
                                )
                                .italics()
                                .weak(),
                            );
                        });
                    return;
                }

                for entry in &self.entries {
                    egui::Frame::new()
                        .fill(ui.visuals().widgets.noninteractive.bg_fill)
                        .corner_radius(6.0)
                        .inner_margin(10.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(format!(
                                        "Period {}",
                                        entry.period.as_u32()
                                    ))
                                    .strong(),
                                );
                                ui.add_space(8.0);
                                ui.monospace(
                                    RichText::new(format!(
                                        "[{}]",
                                        entry.decision_id.to_hex()
                                    ))
                                    .small()
                                    .color(Color32::GRAY),
                                );
                                ui.add_space(6.0);
                                ui.label(
                                    RichText::new(&entry.type_label)
                                        .small()
                                        .strong()
                                        .color(entry.type_color),
                                );
                                ui.add_space(6.0);
                                ui.label(
                                    RichText::new(format!(
                                        "block {}",
                                        entry.block_height
                                    ))
                                    .small()
                                    .weak(),
                                );
                            });
                            ui.add_space(4.0);
                            ui.label(&entry.header);
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Vote:").weak());
                                ui.label(
                                    RichText::new(&entry.display_value)
                                        .strong()
                                        .color(entry.value_color),
                                );
                            });
                        });
                    ui.add_space(4.0);
                }
            });
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut cut = max.saturating_sub(1);
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &s[..cut])
}

fn type_of(t: &DecisionType) -> (&'static str, Color32) {
    match t {
        DecisionType::Binary => ("Binary", Color32::from_rgb(100, 200, 100)),
        DecisionType::Scaled { .. } => {
            ("Scaled", Color32::from_rgb(100, 149, 237))
        }
        DecisionType::Category { .. } => {
            ("Categorical", Color32::from_rgb(200, 170, 100))
        }
    }
}

fn format_vote(value: &VoteValue, decision: &Decision) -> (String, Color32) {
    match value {
        VoteValue::Binary(b) => {
            let (label0, label1) = decision.get_binary_labels();
            let text = if *b { label1 } else { label0 };
            let color = if *b {
                Color32::from_rgb(100, 200, 100)
            } else {
                Color32::from_rgb(200, 120, 120)
            };
            (text, color)
        }
        VoteValue::Scalar(f) => {
            let real = decision.denormalize_value(*f);
            (format!("{real:.4}"), Color32::from_rgb(100, 149, 237))
        }
        VoteValue::Categorical(idx) => {
            let label = match decision.get_category_labels() {
                Some(options) => {
                    let i = *idx as usize;
                    if i == options.len() {
                        "Inconclusive".to_string()
                    } else {
                        options
                            .get(i)
                            .cloned()
                            .unwrap_or_else(|| format!("Option {idx}"))
                    }
                }
                None => format!("{idx}"),
            };
            (label, Color32::from_rgb(200, 170, 100))
        }
        VoteValue::Abstain => {
            ("Abstain".to_string(), Color32::from_rgb(150, 150, 150))
        }
    }
}

fn format_vote_unknown(value: &VoteValue) -> (String, Color32) {
    match value {
        VoteValue::Binary(b) => {
            (if *b { "Yes" } else { "No" }.to_string(), Color32::GRAY)
        }
        VoteValue::Scalar(f) => (format!("{f:.4}"), Color32::GRAY),
        VoteValue::Categorical(idx) => (format!("{idx}"), Color32::GRAY),
        VoteValue::Abstain => ("Abstain".to_string(), Color32::GRAY),
    }
}
