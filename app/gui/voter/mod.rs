use std::pin::Pin;
use std::task::Poll;

use eframe::egui::{self, Color32, RichText};
use futures::Stream;
use strum::{EnumIter, IntoEnumIterator};
use truthcoin_dc::state::voting::types::VotingPeriodId;

use crate::app::App;
use crate::util::PromiseStream;

type NodeUpdates = PromiseStream<Pin<Box<dyn Stream<Item = ()> + Send>>>;

mod ballot;
mod history;

use ballot::Ballot;
use history::History;

#[derive(Default, EnumIter, Eq, PartialEq, strum::Display)]
enum Tab {
    #[default]
    #[strum(to_string = "Ballot")]
    Ballot,
    #[strum(to_string = "History")]
    History,
}

pub struct Voter {
    tab: Tab,
    header: VoterHeader,
    ballot: Ballot,
    history: History,
}

impl Voter {
    pub fn new(_app: Option<&App>) -> Self {
        Self {
            tab: Tab::default(),
            header: VoterHeader::default(),
            ballot: Ballot::default(),
            history: History::default(),
        }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let Some(app) = app else {
            ui.label("No app connection available");
            return;
        };

        self.header.refresh(app);
        self.header.show(ui);
        ui.add_space(8.0);

        egui::TopBottomPanel::top("voter_tabs").show(ui.ctx(), |ui| {
            ui.horizontal(|ui| {
                Tab::iter().for_each(|tab_variant| {
                    let tab_name = tab_variant.to_string();
                    ui.selectable_value(&mut self.tab, tab_variant, tab_name);
                })
            });
        });

        let current_period = self.header.current_period;
        let total_reputation = self.header.total_reputation;

        egui::CentralPanel::default().show(ui.ctx(), |ui| match self.tab {
            Tab::Ballot => {
                self.ballot.show(app, ui, current_period, total_reputation);
            }
            Tab::History => {
                self.history.show(app, ui, current_period);
            }
        });
    }
}

pub(super) struct VoterHeader {
    pub current_period: u32,
    pub total_reputation: f64,
    decisions_available: u32,
    votes_cast_this_period: u32,
    error: Option<String>,
    node_updated: Option<NodeUpdates>,
    needs_initial_load: bool,
}

impl Default for VoterHeader {
    fn default() -> Self {
        Self {
            current_period: 0,
            total_reputation: 0.0,
            decisions_available: 0,
            votes_cast_this_period: 0,
            error: None,
            node_updated: None,
            needs_initial_load: true,
        }
    }
}

impl VoterHeader {
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

    fn refresh(&mut self, app: &App) {
        self.ensure_subscribed(app);
        if !(self.needs_initial_load || self.tip_changed()) {
            return;
        }
        self.needs_initial_load = false;

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

        self.total_reputation = addresses
            .iter()
            .map(|addr| {
                state
                    .reputation()
                    .get_reputation(&rotxn, addr)
                    .unwrap_or(0.0)
            })
            .sum();

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
            Err(err) => {
                self.error =
                    Some(format!("Failed to get current period: {err:#}"));
                return;
            }
        };
        self.current_period = current_period;

        let period_id = VotingPeriodId::new(current_period);

        let claim_period = current_period.saturating_sub(1);
        self.decisions_available = match state
            .decisions()
            .get_claimed_decisions_in_period(&rotxn, claim_period)
        {
            Ok(entries) => entries.len() as u32,
            Err(err) => {
                self.error =
                    Some(format!("Failed to get claimed decisions: {err:#}"));
                return;
            }
        };

        let voting = state.voting().databases();
        let mut votes_cast = 0u32;
        for addr in &addresses {
            match voting.get_votes_by_voter(&rotxn, *addr) {
                Ok(votes) => {
                    votes_cast += votes
                        .keys()
                        .filter(|key| key.period_id == period_id)
                        .count() as u32;
                }
                Err(err) => {
                    self.error = Some(format!("Failed to read votes: {err:#}"));
                    return;
                }
            }
        }
        self.votes_cast_this_period = votes_cast;

        self.error = None;
    }

    fn show(&self, ui: &mut egui::Ui) {
        if let Some(err) = &self.error {
            ui.colored_label(Color32::RED, err);
            ui.add_space(6.0);
        }

        egui::Frame::new()
            .fill(ui.visuals().widgets.noninteractive.bg_fill)
            .corner_radius(8.0)
            .inner_margin(16.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new("Reputation").small().weak());
                        ui.label(
                            RichText::new(format!(
                                "{:.6}",
                                self.total_reputation
                            ))
                            .size(20.0)
                            .strong()
                            .color(Color32::from_rgb(100, 200, 100)),
                        );
                    });

                    ui.separator();

                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new("Current Period").small().weak(),
                        );
                        ui.label(
                            RichText::new(format!("{}", self.current_period))
                                .size(20.0)
                                .strong()
                                .color(Color32::from_rgb(100, 149, 237)),
                        );
                    });

                    ui.separator();

                    ui.vertical(|ui| {
                        ui.label(RichText::new("Participation").small().weak());
                        let label = format!(
                            "{} / {} decisions",
                            self.votes_cast_this_period,
                            self.decisions_available
                        );
                        ui.label(RichText::new(label).size(16.0).strong());
                    });

                    ui.separator();

                    ui.vertical(|ui| {
                        ui.label(RichText::new("Status").small().weak());
                        let (text, color) = if self.total_reputation > 0.0 {
                            ("Active voter", Color32::from_rgb(100, 200, 100))
                        } else {
                            ("No reputation", Color32::from_rgb(200, 120, 120))
                        };
                        ui.label(
                            RichText::new(text)
                                .size(16.0)
                                .strong()
                                .color(color),
                        );
                    });
                });
            });
    }
}
