use eframe::egui;

use crate::app::App;

#[derive(Default)]
pub struct Decisions {}

impl Decisions {
    pub fn show(&mut self, _app: Option<&App>, ui: &mut egui::Ui) {
        ui.heading("Decisions");
        ui.separator();
        ui.label("Manage and create decisions here.");
        // TODO: Add decision management functionality
    }
}
