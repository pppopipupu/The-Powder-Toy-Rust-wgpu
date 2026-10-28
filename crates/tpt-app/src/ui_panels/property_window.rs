//! Floating property window (`PropertyWindow`).
//!
//! Ports `PropertyTool.cpp`'s window: a dropdown with every editable particle
//! property, a value box and an OK button that stores the configuration in the
//! `Prop.Type` / `Prop.Value` preferences. The window is opened by the `PROP`
//! tool (palette button or `Ctrl+P`) and by shift-clicking with `SMPL`, which
//! seeds the value from the particle under the cursor (`TakePropertyFrom`).
//!
//! Two rows do not exist in the reference window:
//!
//! * the `DrawFill` checkbox selects `PropertyTool::DrawFill`: while it is on, a
//!   click floods the connected particles instead of stamping the brush
//!   (`Simulation::flood_prop`; see `app_state::property`'s module docs);
//! * a dim line above the dropdown reports the wall that a sample landed on,
//!   because a wall has no editable particle property to seed the value box
//!   with (`SimulationSample::WallType`).

use egui::{Color32, RichText};

use crate::app_state::{AppState, PropertyKind};
use crate::i18n::{t, t_fmt};

use super::shared::{ACCENT, TEXT_DIM};
use super::widgets::option_checkbox;

/// Colour of an invalid value (`style::Colour::ErrorTitle`).
const ERROR: Color32 = Color32::from_rgb(255, 96, 96);

/// Draws the property window when it is open.
pub(super) fn draw_property_window(ctx: &egui::Context, state: &mut AppState) {
    if !state.property_window.open {
        return;
    }

    let mut open = true;
    let mut commit = false;
    let mut close = false;

    egui::Window::new(t("property.title"))
        .open(&mut open)
        .resizable(false)
        .default_width(250.0)
        .show(ctx, |ui| {
            ui.label(RichText::new(t("property.heading")).color(ACCENT));
            ui.add_space(2.0);

            let window = &mut state.property_window;
            let elements = &state.engine.registry;
            let scale = state.prefs.temperature_scale;

            // A sample that landed on a wall has no value to seed; report the
            // wall instead (`SimulationSample::WallType`).
            if let Some(wall) = &window.seeded_wall {
                ui.label(RichText::new(wall.summary(scale)).small().color(TEXT_DIM));
                ui.add_space(2.0);
            }

            ui.horizontal(|ui| {
                ui.label(t("property.property"));
                let mut selected = None;
                egui::ComboBox::from_id_salt("tpt_property_field")
                    .selected_text(window.field.display())
                    .show_ui(ui, |ui| {
                        for candidate in PropertyKind::ALL {
                            if ui
                                .selectable_label(window.field == candidate, candidate.display())
                                .clicked()
                            {
                                selected = Some(candidate);
                            }
                        }
                    });
                if let Some(candidate) = selected {
                    // `HandlePropertyChange`: re-validate the current text
                    // against the newly selected property.
                    window.select_field(candidate, elements, scale);
                }
            });

            let edit = ui.add(
                egui::TextEdit::singleline(&mut window.value_text)
                    .hint_text(t("property.hint_value"))
                    .desired_width(f32::INFINITY),
            );
            if edit.changed() {
                window.revalidate(elements, scale);
            }
            // Enter activates OK, like the reference window's okay button.
            if edit.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                commit = true;
            }

            match window.error {
                Some(message) => {
                    ui.label(RichText::new(message).small().color(ERROR));
                }
                None => {
                    ui.label(
                        RichText::new(hint(window.field, scale))
                            .small()
                            .color(TEXT_DIM),
                    );
                }
            }

            ui.add_space(4.0);
            // `PropertyTool::DrawFill`: the reference selects the fill draw
            // mode with ctrl+shift (`GameView::UpdateDrawMode`). The port binds
            // the mode to this checkbox instead, so it stays on across strokes
            // and is visible while the window is open.
            option_checkbox(
                ui,
                &mut window.fill,
                t("property.draw_fill"),
                "",
                t("property.draw_fill_desc"),
            );

            let okay = ui.add_enabled(
                window.value.is_some(),
                egui::Button::new(t("property.ok"))
                    .min_size(egui::vec2(ui.available_width(), 22.0)),
            );
            if okay.clicked() {
                commit = true;
            }

            // The reference window's up/down keys cycle the property and Esc
            // closes it (`OnKeyPress` / `OnTryExit`).
            let active = edit.has_focus() || ui.rect_contains_pointer(ui.max_rect());
            if active {
                if ui.input(|input| input.key_pressed(egui::Key::ArrowUp)) {
                    window.cycle_field(-1, elements, scale);
                }
                if ui.input(|input| input.key_pressed(egui::Key::ArrowDown)) {
                    window.cycle_field(1, elements, scale);
                }
                if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
                    close = true;
                }
            }
        });

    if !open || close {
        // Closing without OK keeps the tool's previous configuration.
        state.property_window.close();
    } else if commit {
        let scale = state.prefs.temperature_scale;
        if state
            .property_window
            .commit(&mut state.prefs, &state.engine.registry, scale)
        {
            // `PropertyTool::SetConfiguration` writes the preference file.
            state.prefs.save();
        }
    }
}

/// One-line help shown under a valid value box.
fn hint(kind: PropertyKind, scale: u32) -> String {
    match kind {
        PropertyKind::Temp => {
            let letter = match scale {
                tpt_core::constants::TEMPSCALE_KELVIN => "K",
                tpt_core::constants::TEMPSCALE_FAHRENHEIT => "F",
                _ => "C",
            };
            let sample = format!(
                "{:.0}{letter}",
                crate::app_state::temperature_to_scale(300.0, scale),
            );
            t_fmt("property.hint_temp", &[&sample])
        }
        PropertyKind::Type | PropertyKind::Ctype => t("property.hint_element").to_string(),
        PropertyKind::Dcolour => t("property.hint_dcolour").to_string(),
        _ => t("property.hint_integer").to_string(),
    }
}
