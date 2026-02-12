use eframe::egui;

use crate::app::App;

#[derive(Default)]
pub struct List {}

impl List {
    pub fn show(&mut self, _app: Option<&App>, ui: &mut egui::Ui) {
        ui.heading("Market List");
        ui.separator();
        ui.label("View and filter all markets.");
        // TODO: Add market listing functionality
    }
}
