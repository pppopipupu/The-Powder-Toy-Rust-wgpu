//! Reusable settings-window controls (checkbox rows, dropdowns, drag fields).

use egui::{RichText, Ui};
use tpt_core::constants::{DISPLAY_AIRC, DISPLAY_AIRH, DISPLAY_AIRP, DISPLAY_AIRV, DISPLAY_AIRW};

use crate::i18n::t;
use crate::prefs::{FpsLimit, MAX_SANE_DRAW_FPS, MAX_SANE_FPS, MIN_SANE_DRAW_FPS, MIN_SANE_FPS};

use super::shared::TEXT_DIM;

/// Mutually exclusive air display modes (TPT's `DISPLAY_AIR`).
const DISPLAY_AIR_GROUP: u32 =
    DISPLAY_AIRC | DISPLAY_AIRP | DISPLAY_AIRV | DISPLAY_AIRH | DISPLAY_AIRW;

pub(super) fn option_checkbox(
    ui: &mut Ui,
    value: &mut bool,
    label: &str,
    note: &str,
    info: &str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed = ui.checkbox(value, label).changed();
        if !note.is_empty() {
            ui.label(RichText::new(note).small().color(TEXT_DIM));
        }
    });
    option_info(ui, info);
    ui.add_space(4.0);
    changed
}

/// Dim helper text under a control.
pub(super) fn option_info(ui: &mut Ui, text: &str) {
    if !text.is_empty() {
        ui.label(RichText::new(text).small().color(TEXT_DIM));
    }
}

/// Labelled combo box over a `u32` value.
pub(super) fn option_dropdown(
    ui: &mut Ui,
    id: &str,
    label: &str,
    value: &mut u32,
    options: &[(u32, &str)],
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        let selected = options
            .iter()
            .find(|entry| entry.0 == *value)
            .map(|entry| entry.1)
            .unwrap_or("?");
        egui::ComboBox::from_id_salt(id)
            .selected_text(selected)
            .width(150.0)
            .show_ui(ui, |ui| {
                for (option, text) in options {
                    if ui.selectable_value(value, *option, *text).changed() {
                        changed = true;
                    }
                }
            });
    });
    changed
}

/// Labelled floating-point drag field.
pub(super) fn option_drag_f32(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    speed: f64,
    range: std::ops::RangeInclusive<f32>,
    decimals: usize,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed = ui
            .add(
                egui::DragValue::new(value)
                    .speed(speed)
                    .range(range)
                    .max_decimals(decimals),
            )
            .changed();
    });
    changed
}

/// Labelled integer drag field.
pub(super) fn option_drag_u32(
    ui: &mut Ui,
    label: &str,
    value: &mut u32,
    range: std::ops::RangeInclusive<u32>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed = ui
            .add(egui::DragValue::new(value).speed(1.0).range(range))
            .changed();
    });
    changed
}

/// Simulation framerate cap row (Exact/Uncapped plus the value).
pub(super) fn fps_limit_control(ui: &mut Ui, limit: &mut FpsLimit) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(t("options.limit_sim_cap"));
        let uncapped = matches!(limit, FpsLimit::Uncapped);
        egui::ComboBox::from_id_salt("tpt_fps_limit_mode")
            .selected_text(t(limit.label_key()))
            .width(110.0)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(!uncapped, t("options.limit_exact"))
                    .clicked()
                    && uncapped
                {
                    *limit = FpsLimit::Exact(60.0);
                    changed = true;
                }
                if ui
                    .selectable_label(uncapped, t("options.limit_uncapped"))
                    .clicked()
                    && !uncapped
                {
                    *limit = FpsLimit::Uncapped;
                    changed = true;
                }
            });
        if let FpsLimit::Exact(fps) = limit {
            changed |= ui
                .add(
                    egui::DragValue::new(fps)
                        .speed(1.0)
                        .range(MIN_SANE_FPS..=MAX_SANE_FPS)
                        .max_decimals(1)
                        .suffix(" fps"),
                )
                .changed();
        }
    });
    changed
}

/// Rendering framerate cap row: a plain frames-per-second value.
pub(super) fn draw_fps_control(ui: &mut Ui, fps: &mut u32) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(t("options.limit_draw_cap"));
        changed |= ui
            .add(
                egui::DragValue::new(fps)
                    .speed(1.0)
                    .range(MIN_SANE_DRAW_FPS..=MAX_SANE_DRAW_FPS)
                    .suffix(" fps"),
            )
            .changed();
    });
    changed
}

/// Display-mode checkbox. The air display modes are mutually exclusive, as in
/// TPT's Render window.
pub(super) fn display_mode_checkbox(
    ui: &mut Ui,
    display_mode: &mut u32,
    mode: u32,
    label: &str,
    info: &str,
) -> bool {
    let mut checked = (*display_mode & mode) != 0;
    if !ui
        .checkbox(&mut checked, label)
        .on_hover_text(info)
        .changed()
    {
        return false;
    }
    if checked {
        if mode & DISPLAY_AIR_GROUP != 0 {
            *display_mode &= !DISPLAY_AIR_GROUP;
        }
        *display_mode |= mode;
    } else {
        *display_mode &= !mode;
    }
    true
}

/// Render-mode checkbox, a plain bit toggle.
pub(super) fn render_mode_checkbox(
    ui: &mut Ui,
    render_mode: &mut u32,
    mode: u32,
    label: &str,
    info: &str,
) -> bool {
    let mut checked = (*render_mode & mode) != 0;
    if !ui
        .checkbox(&mut checked, label)
        .on_hover_text(info)
        .changed()
    {
        return false;
    }
    if checked {
        *render_mode |= mode;
    } else {
        *render_mode &= !mode;
    }
    true
}
