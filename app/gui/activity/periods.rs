use std::collections::HashMap;
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, RichText, ScrollArea};
use truthcoin_dc::state::{
    decisions::{DecisionConfig, DecisionEntry},
    voting::period_calculator,
};

use crate::app::App;

#[allow(clippy::type_complexity)]
pub struct Periods {
    current_period: u32,
    period_summary: Option<(Vec<(u32, u64)>, Vec<(u32, u64)>)>, // (active_periods, voting_periods)
    voting_decisions: HashMap<u32, Vec<DecisionEntry>>, // claim_period -> claimed decisions in voting
    last_refresh: Instant,
    genesis_timestamp: Option<u64>,
    mainchain_timestamp: u64,
    timestamp_fetched_at: Instant,
    decision_config: Option<DecisionConfig>,
    error: Option<String>,
}

impl Default for Periods {
    fn default() -> Self {
        Self {
            current_period: 0,
            period_summary: None,
            voting_decisions: HashMap::new(),
            last_refresh: Instant::now(),
            genesis_timestamp: None,
            mainchain_timestamp: 0,
            timestamp_fetched_at: Instant::now(),
            decision_config: None,
            error: None,
        }
    }
}

impl Periods {
    fn estimated_timestamp(&self) -> u64 {
        let wall_elapsed = self.timestamp_fetched_at.elapsed().as_secs();
        self.mainchain_timestamp.saturating_add(wall_elapsed)
    }

    fn refresh_data(&mut self, app: &App) {
        if self.last_refresh.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last_refresh = Instant::now();

        let new_timestamp = app.node.get_mainchain_timestamp().unwrap_or(0);
        if new_timestamp != self.mainchain_timestamp {
            self.timestamp_fetched_at = Instant::now();
        }
        self.mainchain_timestamp = new_timestamp;

        match app.node.get_current_period() {
            Ok(period) => {
                self.current_period = period;
                self.error = None;
            }
            Err(e) => {
                self.error =
                    Some(format!("Failed to get current period: {e:#}"));
                return;
            }
        }

        match app.node.get_genesis_timestamp() {
            Ok(ts) => self.genesis_timestamp = ts,
            Err(e) => {
                tracing::warn!("Failed to get genesis timestamp: {e:#}");
            }
        }

        self.decision_config = Some(app.node.get_decision_config().clone());

        match app.node.get_period_summary() {
            Ok(summary) => {
                self.period_summary = Some(summary);
            }
            Err(e) => {
                tracing::warn!("Failed to get period summary: {e:#}");
            }
        }

        match app.node.get_voting_periods() {
            Ok(voting_periods) => {
                self.voting_decisions.clear();
                for (claim_period, _voting_count, _total) in voting_periods {
                    if let Ok(entries) = app
                        .node
                        .get_claimed_decisions_in_period(
                        truthcoin_dc::state::voting::types::VotingPeriodId::new(
                            claim_period,
                        ),
                    ) && !entries.is_empty()
                    {
                        self.voting_decisions.insert(claim_period, entries);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to get voting periods: {e:#}");
            }
        }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let Some(app) = app else {
            ui.label("No app connection available");
            return;
        };

        self.refresh_data(app);

        if let Some(ref config) = self.decision_config
            && !config.is_blocks_mode()
        {
            ui.ctx().request_repaint_after(Duration::from_secs(1));
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ScrollArea::vertical().show(ui, |ui| {
                if let Some(err) = &self.error {
                    ui.colored_label(Color32::RED, err);
                    ui.add_space(10.0);
                }

                self.show_current_period_info(ui);
                ui.add_space(20.0);

                self.show_timer_section(ui);
                ui.add_space(20.0);

                self.show_voting_periods_section(app, ui);
            });
        });
    }

    fn show_current_period_info(&self, ui: &mut egui::Ui) {
        ui.heading("Current Period Info");
        ui.add_space(10.0);

        egui::Frame::new()
            .fill(ui.visuals().widgets.noninteractive.bg_fill)
            .corner_radius(8.0)
            .inner_margin(16.0)
            .show(ui, |ui| {
                egui::Grid::new("period_info_grid")
                    .num_columns(2)
                    .spacing([20.0, 8.0])
                    .show(ui, |ui| {
                        ui.label(RichText::new("Current Period:").strong());

                        let display_period = match (
                            &self.decision_config,
                            self.genesis_timestamp,
                        ) {
                            (Some(cfg), Some(genesis_ts)) => {
                                period_calculator::get_current_period(
                                    self.estimated_timestamp(),
                                    None,
                                    genesis_ts,
                                    cfg,
                                )
                                .unwrap_or(self.current_period)
                            }
                            _ => self.current_period,
                        };

                        ui.label(
                            RichText::new(format!("{display_period}"))
                                .size(18.0)
                                .color(Color32::from_rgb(100, 200, 100)),
                        );
                        ui.end_row();

                        if let Some(ref config) = self.decision_config {
                            ui.label(RichText::new("Mode:").strong());
                            let mode_text = if config.is_blocks_mode() {
                                "Blocks"
                            } else {
                                "Seconds (Time-based)"
                            };
                            ui.label(mode_text);
                            ui.end_row();

                            ui.label(RichText::new("Period Length:").strong());
                            if let Some(blocks) = config.blocks_per_period() {
                                ui.label(format!("{blocks} blocks"));
                            } else if let Some(secs) =
                                config.seconds_per_period()
                            {
                                ui.label(format_duration(secs));
                            }
                            ui.end_row();
                        }

                        if let Some(genesis_ts) = self.genesis_timestamp {
                            ui.label(
                                RichText::new("Genesis Timestamp:").strong(),
                            );
                            ui.label(format_timestamp(genesis_ts));
                            ui.end_row();
                        }
                    });
            });
    }

    fn show_timer_section(&self, ui: &mut egui::Ui) {
        let Some(ref config) = self.decision_config else {
            return;
        };

        if config.is_blocks_mode() {
            ui.heading("Period Progress");
            ui.add_space(5.0);
            ui.label(
                RichText::new(
                    "In blocks mode, periods advance with each block.",
                )
                .weak()
                .italics(),
            );
            return;
        }

        let Some(genesis_ts) = self.genesis_timestamp else {
            ui.heading("Timer Until Next Period");
            ui.add_space(5.0);
            ui.label(
                RichText::new(
                    "Genesis timestamp not yet set (no blocks mined)",
                )
                .weak()
                .italics(),
            );
            return;
        };

        let Some(seconds_per_period) = config.seconds_per_period() else {
            return;
        };

        let estimated_now = self.estimated_timestamp();
        let elapsed_since_genesis = estimated_now.saturating_sub(genesis_ts);
        let estimated_period = period_calculator::get_current_period(
            estimated_now,
            None,
            genesis_ts,
            config,
        )
        .unwrap_or(self.current_period);
        let time_in_period = elapsed_since_genesis % seconds_per_period;
        let time_remaining = seconds_per_period.saturating_sub(time_in_period);

        ui.heading("Time Until Next Period");
        ui.add_space(10.0);

        egui::Frame::new()
            .fill(ui.visuals().widgets.noninteractive.bg_fill)
            .corner_radius(8.0)
            .inner_margin(16.0)
            .show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(
                        RichText::new(format_duration(time_remaining))
                            .size(32.0)
                            .strong()
                            .color(Color32::from_rgb(100, 149, 237)),
                    );
                    ui.add_space(5.0);
                    ui.label(
                        RichText::new(format!(
                            "remaining in period {estimated_period} (estimated)"
                        ))
                        .weak(),
                    );

                    ui.add_space(15.0);

                    let progress =
                        time_in_period as f32 / seconds_per_period as f32;
                    let progress_bar = egui::ProgressBar::new(progress)
                        .show_percentage()
                        .animate(false);
                    ui.add_sized([300.0, 20.0], progress_bar);

                    ui.add_space(10.0);

                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "Elapsed: {} / {}",
                            format_duration(time_in_period),
                            format_duration(seconds_per_period)
                        ));
                    });
                });
            });
    }

    fn show_voting_periods_section(&self, app: &App, ui: &mut egui::Ui) {
        ui.heading("Voting Periods");
        ui.add_space(10.0);

        if self.voting_decisions.is_empty() {
            egui::Frame::new()
                .fill(ui.visuals().widgets.noninteractive.bg_fill)
                .corner_radius(8.0)
                .inner_margin(16.0)
                .show(ui, |ui| {
                    ui.label(
                        RichText::new("No periods currently in voting")
                            .weak()
                            .italics(),
                    );
                });
            return;
        }

        let mut sorted_periods: Vec<_> =
            self.voting_decisions.keys().copied().collect();
        sorted_periods.sort();

        for period_id in sorted_periods {
            let decisions = match self.voting_decisions.get(&period_id) {
                Some(s) => s,
                None => continue,
            };

            egui::Frame::new()
                .fill(ui.visuals().widgets.noninteractive.bg_fill)
                .corner_radius(8.0)
                .inner_margin(12.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        egui::Frame::new()
                            .fill(
                                Color32::from_rgb(255, 165, 0)
                                    .gamma_multiply(0.3),
                            )
                            .corner_radius(4.0)
                            .inner_margin(egui::Margin::symmetric(8, 4))
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new("VOTING")
                                        .strong()
                                        .color(Color32::from_rgb(255, 165, 0)),
                                );
                            });

                        ui.add_space(10.0);

                        ui.label(
                            RichText::new(format!("Period {period_id}"))
                                .strong()
                                .size(16.0),
                        );

                        ui.add_space(10.0);

                        ui.label(
                            RichText::new(format!(
                                "{} decisions",
                                decisions.len()
                            ))
                            .weak(),
                        );
                    });

                    ui.add_space(10.0);

                    egui::CollapsingHeader::new(format!(
                        "Decisions in Period {} ({} decisions)",
                        period_id,
                        decisions.len()
                    ))
                    .default_open(true)
                    .show(ui, |ui| {
                        for entry in decisions {
                            self.show_decision_row(app, entry, ui);
                        }
                    });
                });

            ui.add_space(10.0);
        }
    }

    fn show_decision_row(
        &self,
        _app: &App,
        entry: &DecisionEntry,
        ui: &mut egui::Ui,
    ) {
        ui.horizontal(|ui| {
            let entry_hex = entry.decision_id.to_hex();
            ui.monospace(
                RichText::new(format!("[{entry_hex}]"))
                    .small()
                    .color(Color32::GRAY),
            );

            ui.add_space(10.0);

            if let Some(ref decision) = entry.decision {
                let header = if decision.header.len() > 60 {
                    format!("{}...", &decision.header[..57])
                } else {
                    decision.header.clone()
                };
                ui.label(&header);

                ui.add_space(10.0);

                let type_text = if decision.is_scaled() {
                    "Scaled"
                } else {
                    "Binary"
                };
                let type_color = if decision.is_scaled() {
                    Color32::from_rgb(100, 149, 237)
                } else {
                    Color32::from_rgb(100, 200, 100)
                };
                ui.label(RichText::new(type_text).small().color(type_color));

                if decision.is_scaled()
                    && let (Some(min), Some(max)) =
                        (decision.scale_min(), decision.scale_max())
                {
                    ui.label(
                        RichText::new(format!("[{min} - {max}]"))
                            .small()
                            .weak(),
                    );
                }
            } else {
                ui.label(RichText::new("No decision").weak().italics());
            }
        });
    }
}

fn format_timestamp(ts: u64) -> String {
    use std::time::{Duration, UNIX_EPOCH};

    let datetime = UNIX_EPOCH + Duration::from_secs(ts);

    if let Ok(dur) = datetime.duration_since(UNIX_EPOCH) {
        let secs = dur.as_secs();
        let days_since_epoch = secs / 86400;
        let remaining_secs = secs % 86400;
        let hours = remaining_secs / 3600;
        let minutes = (remaining_secs % 3600) / 60;
        let seconds = remaining_secs % 60;

        let years = 1970 + (days_since_epoch / 365);
        let day_of_year = days_since_epoch % 365;
        let month = day_of_year / 30 + 1;
        let day = day_of_year % 30 + 1;

        format!(
            "{years:04}-{month:02}-{day:02} {hours:02}:{minutes:02}:{seconds:02} UTC"
        )
    } else {
        format!("Timestamp: {ts}")
    }
}

fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        let minutes = secs / 60;
        let remaining_secs = secs % 60;
        format!("{minutes}m {remaining_secs}s")
    } else if secs < 86400 {
        let hours = secs / 3600;
        let minutes = (secs % 3600) / 60;
        let remaining_secs = secs % 60;
        format!("{hours}h {minutes}m {remaining_secs}s")
    } else {
        let days = secs / 86400;
        let hours = (secs % 86400) / 3600;
        let minutes = (secs % 3600) / 60;
        format!("{days}d {hours}h {minutes}m")
    }
}
