use std::{collections::BTreeMap, fmt::Display};

use eframe::egui::{self, InnerResponse, Response};

use hex::FromHex;
use truthcoin_dc::{state::DutchAuctionState, types::DutchAuctionId};

use crate::{
    app::App,
    gui::util::{InnerResponseExt, UiExt},
};

#[derive(Debug, Default)]
pub(super) struct DutchAuctionExplorer {
    query: String,
}

fn show_dutch_auction_state(
    ui: &mut egui::Ui,
    auction_state: &DutchAuctionState,
) -> Response {
    fn show_line(
        ui: &mut egui::Ui,
        value: &dyn Display,
        descriptor: &str,
    ) -> InnerResponse<Response> {
        ui.horizontal(|ui| {
            ui.monospace_selectable_singleline(
                false,
                format!("{descriptor}: {value}"),
            )
        })
    }
    let DutchAuctionState {
        start_block,
        most_recent_bid_block,
        duration,
        base_asset,
        initial_base_amount,
        base_amount_remaining,
        quote_asset,
        quote_amount,
        initial_price,
        price_after_most_recent_bid,
        initial_end_price,
        end_price_after_most_recent_bid,
    } = auction_state;
    [
        (start_block as &dyn Display, "Start Block"),
        (
            &most_recent_bid_block.latest().data,
            "Most Recent Bid Block",
        ),
        (duration, "Duration"),
        (base_asset, "Base Asset"),
        (initial_base_amount, "Initial Base Amount"),
        (
            &base_amount_remaining.latest().data,
            "Base Amount Remaining",
        ),
        (quote_asset, "Quote Asset"),
        (&quote_amount.latest().data, "Quote Amount"),
        (initial_price, "Initial Price"),
        (
            &price_after_most_recent_bid.latest().data,
            "Price after most recent bid",
        ),
        (initial_end_price, "Initial End Price"),
        (
            &end_price_after_most_recent_bid.latest().data,
            "End Price after most recent bid",
        ),
    ]
    .into_iter()
    .map(|(value, descriptor)| show_line(ui, value, descriptor).join())
    .reduce(|resp0, resp1| resp0 | resp1)
    .unwrap()
}

fn show_dutch_auction_with_state(
    ui: &mut egui::Ui,
    auction_id: DutchAuctionId,
    auction_state: &DutchAuctionState,
) -> Response {
    ui.monospace_selectable_singleline(
        false,
        format!("Auction ID: {}", auction_id.0),
    ) | show_dutch_auction_state(ui, auction_state)
}

impl DutchAuctionExplorer {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let Some(app) = app else {
                return;
            };
            ui.heading("Dutch Auction Explorer");
            let auctions = match app.node.dutch_auctions() {
                Ok(auctions) => BTreeMap::from_iter(auctions),
                Err(err) => {
                    ui.monospace_selectable_multiline(format!("{err}"));
                    return;
                }
            };
            ui.horizontal(|ui| {
                let query_edit = egui::TextEdit::singleline(&mut self.query)
                    .hint_text("Search by Auction ID")
                    .desired_width(150.);
                ui.add(query_edit);
            });
            if self.query.is_empty() {
                auctions
                    .into_iter()
                    .for_each(|(auction_id, auction_state)| {
                        show_dutch_auction_with_state(
                            ui,
                            auction_id,
                            &auction_state,
                        );
                    })
            } else if let Ok(auction_id) = DutchAuctionId::from_hex(&self.query)
                && let Some(auction_state) = auctions.get(&auction_id)
            {
                show_dutch_auction_with_state(ui, auction_id, auction_state);
            }
        });
    }
}
