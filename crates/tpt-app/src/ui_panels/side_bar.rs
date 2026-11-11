//! Right sidebar: category tabs and the bound tools, plus the floating stamp
//! browser.

use std::path::PathBuf;

use egui::{Color32, RichText, Ui};
use tpt_core::elements::MenuCategory;

use crate::app_state::{AppState, STAMPS_DIR, SelectedTool, format_save_time};
use crate::i18n::{t, t_fmt};

use super::shared::{ACCENT, TEXT_DIM, element_def, element_description, tool_label};

/// TPT menu sections in `msections` order (`SimulationData.cpp:79-95`), plus
/// the port-only rigid-body list appended at the end.
const CATEGORY_TABS: [(MenuCategory, &str); 15] = [
    (MenuCategory::Walls, "side_bar.cat_walls"),
    (MenuCategory::Electronics, "side_bar.cat_electronics"),
    (MenuCategory::Powered, "side_bar.cat_powered"),
    (MenuCategory::Sensors, "side_bar.cat_sensors"),
    (MenuCategory::Force, "side_bar.cat_force"),
    (MenuCategory::Explosives, "side_bar.cat_explosives"),
    (MenuCategory::Gases, "side_bar.cat_gases"),
    (MenuCategory::Liquids, "side_bar.cat_liquids"),
    (MenuCategory::Powders, "side_bar.cat_powders"),
    (MenuCategory::Solids, "side_bar.cat_solids"),
    (MenuCategory::Radioactive, "side_bar.cat_radioactive"),
    (MenuCategory::Special, "side_bar.cat_special"),
    (MenuCategory::Life, "side_bar.cat_life"),
    (MenuCategory::Tools, "side_bar.cat_tools"),
    // Port-only: TPT has no rigid-body section in `msections`.
    (MenuCategory::RigidBodies, "side_bar.cat_rigid_bodies"),
];

/// Localized label of a palette category tab.
///
/// Used by the element search, which matches the query against the menu
/// section description just like `ElementSearchActivity::searchTools`
/// (`ElementSearchActivity.cpp:115-131`).
pub(super) fn category_label(category: MenuCategory) -> &'static str {
    CATEGORY_TABS
        .iter()
        .find(|(candidate, _)| *candidate == category)
        .map(|(_, label)| t(label))
        .unwrap_or(t("side_bar.cat_tools"))
}

pub(super) fn draw_side_bar(ui: &mut Ui, state: &mut AppState) {
    ui.label(
        RichText::new(t("side_bar.categories"))
            .small()
            .strong()
            .color(TEXT_DIM),
    );
    ui.add_space(4.0);

    for (category, label) in CATEGORY_TABS {
        let width = ui.available_width();
        let selected = state.active_category == category;
        let response = ui.add_sized([width, 24.0], egui::Button::selectable(selected, t(label)));
        // `GameView.cpp:428-435`: a click always selects the category; with
        // `MouseClickRequired` off, entering a category button selects it as
        // well, but only while no mouse button is held (the same guard as
        // `!GetMouseDown()`), so a drag that ends over the sidebar does not
        // switch sections.
        if response.clicked()
            || (!state.prefs.mouse_click_required
                && response.hovered()
                && !ui.input(|input| input.pointer.any_down()))
        {
            state.active_category = category;
        }
    }

    ui.separator();
    ui.label(
        RichText::new(t("side_bar.bound_tools"))
            .small()
            .strong()
            .color(TEXT_DIM),
    );
    ui.label(
        RichText::new(tool_label(state, state.primary_tool))
            .small()
            .color(TEXT_DIM),
    );
    ui.label(
        RichText::new(tool_label(state, state.secondary_tool))
            .small()
            .color(TEXT_DIM),
    );
    ui.label(
        RichText::new(t("side_bar.selected_element_hint"))
            .small()
            .color(TEXT_DIM),
    );

    // `GameModel.cpp:1851-1852` puts `Element::Description` on the tool and
    // `GameView.cpp:1304-1331` renders it as the palette tooltip while the
    // pointer is over an element chip. Port addition: keep the same
    // introduction visible here after the pick, so a selected element is
    // described without hovering a small palette button.
    if let SelectedTool::Element(ty) = state.primary_tool
        && let Some(definition) = element_def(state, ty)
    {
        ui.add_space(4.0);
        ui.label(RichText::new(&definition.name).strong().color(ACCENT));
        ui.add(
            egui::Label::new(
                RichText::new(element_description(state, ty))
                    .small()
                    .color(TEXT_DIM),
            )
            .wrap(),
        );
    }

    // The element search is a floating HUD window (`GameView.cpp:325-328`
    // opens it from the simulation view); it is drawn here so the whole HUD
    // keeps a single entry point per frame.
    super::element_search::draw_element_search(ui.ctx(), state);
}

/// Draws the floating stamp browser while [`AppState::stamps`] is open.
///
/// Ports `GameController::OpenStamps` (`GameController.cpp:1399-1410`): the
/// reference opens a `LocalBrowserController` window over `./stamps`; picking
/// a row calls `Client::GetStamp` and feeds `GameController::LoadStamp` into
/// the placement path, and the delete button removes the file through
/// `Client::DeleteStamp` (`Client.cpp:319-328`). The port keeps the two-step
/// confirmation used by the `.tptrs` card (`menu::draw_load_card`).
pub(super) fn draw_stamp_browser(ctx: &egui::Context, state: &mut AppState) {
    if !state.stamps.open {
        return;
    }
    let mut open = true;
    let mut place: Option<PathBuf> = None;
    let mut delete: Option<PathBuf> = None;

    egui::Window::new(t("side_bar.stamps_title"))
        .id(egui::Id::unique("tpt_stamp_browser_window"))
        .open(&mut open)
        .resizable(true)
        .collapsible(false)
        .default_width(380.0)
        .default_pos(ctx.content_rect().center() - egui::vec2(190.0, 200.0))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(t_fmt("side_bar.stamps_in", &[STAMPS_DIR]))
                        .small()
                        .color(TEXT_DIM),
                );
                if ui.button(t("menu.load_refresh")).clicked() {
                    state.stamps.refresh();
                }
            });
            ui.label(
                RichText::new(t("side_bar.stamps_help"))
                    .small()
                    .color(TEXT_DIM),
            );

            if let Some(error) = &state.stamps.error {
                ui.label(RichText::new(error).color(Color32::from_rgb(255, 140, 120)));
            }
            if state.stamps.entries.is_empty() {
                ui.label(
                    RichText::new(t("side_bar.stamps_empty"))
                        .italics()
                        .color(TEXT_DIM),
                );
                return;
            }

            let selected = state.stamps.selected.clone();
            let mut clicked: Option<PathBuf> = None;
            egui::ScrollArea::vertical()
                .id_salt("tpt_stamp_browser")
                .max_height(280.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for entry in &state.stamps.entries {
                        ui.horizontal(|ui| {
                            if ui.button(t("side_bar.stamps_place")).clicked() {
                                place = Some(entry.path.clone());
                            }
                            let is_selected = selected.as_deref() == Some(entry.path.as_path());
                            if ui
                                .selectable_label(
                                    is_selected,
                                    RichText::new(&entry.name).monospace(),
                                )
                                .clicked()
                            {
                                clicked = Some(entry.path.clone());
                            }
                            if let Some(stamp) = format_save_time(entry.modified) {
                                ui.label(
                                    RichText::new(t_fmt("menu.load_modified", &[&stamp]))
                                        .small()
                                        .color(TEXT_DIM),
                                );
                            }
                        });
                    }
                });
            if let Some(path) = clicked {
                state.stamps.select(&path);
            }

            let Some(selected) = state.stamps.selected().cloned() else {
                return;
            };
            ui.separator();
            if state.stamps.delete_confirm {
                ui.label(
                    RichText::new(t_fmt("side_bar.stamps_delete_prompt", &[&selected.name]))
                        .color(Color32::from_rgb(255, 180, 120)),
                );
                ui.horizontal(|ui| {
                    if ui.button(t("side_bar.stamps_delete_confirm")).clicked() {
                        delete = Some(selected.path.clone());
                    }
                    if ui.button(t("menu.load_cancel")).clicked() {
                        state.stamps.delete_confirm = false;
                    }
                });
            } else if ui.button(t("side_bar.stamps_delete")).clicked() {
                state.stamps.delete_confirm = true;
            }
        });

    state.stamps.open = open;
    if let Some(path) = place {
        // `GameController::OpenStamps` callback: a picked stamp is loaded into
        // the placement path; the window stays open for repeated stamping.
        let _ = state.load_stamp(&path);
    }
    if let Some(path) = delete {
        let name = state
            .stamps
            .entries
            .iter()
            .find(|entry| entry.path == path)
            .map(|entry| entry.name.clone())
            .unwrap_or_default();
        state.stamps.selected = Some(path);
        match state.stamps.delete_selected() {
            Ok(()) => state.set_status(t_fmt("app.status_stamp_deleted", &[&name]), false),
            Err(error) => state.set_status(error, true),
        }
    }
}
