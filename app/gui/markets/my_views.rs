use eframe::egui;

use crate::app::App;

#[derive(Default)]
pub struct MyViews {}

impl MyViews {
    pub fn show(&mut self, _app: Option<&App>, ui: &mut egui::Ui) {
        ui.heading("My Views");
        ui.separator();
        ui.label("Track your market predictions and views.");
        // TODO: Add views tracking functionality
    }
}
