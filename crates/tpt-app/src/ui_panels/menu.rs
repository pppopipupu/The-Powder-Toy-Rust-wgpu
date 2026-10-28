//! Title-screen cards: new canvas and the save browser.

use std::path::PathBuf;
use std::time::Duration;

use egui::{Align2, Color32, FontId, Frame, RichText, Ui, vec2};
use tpt_core::constants::CELL;

use crate::app_state::{AppState, CanvasSizePreset, NewCanvasDialog, SAVES_DIR, format_save_time};
use crate::i18n::{t, t_fmt};

use super::shared::{ACCENT, TEXT_DIM, card, preset_button_label, thousands};
use super::wip_badge;

/// Page background behind the main-menu cards.
const BG_DEEP: Color32 = Color32::from_rgb(15, 16, 20);
/// Smallest custom canvas dimension offered by the new-canvas dialog.
const MIN_CANVAS: u32 = 128;
/// Largest custom canvas dimension offered by the new-canvas dialog.
const MAX_CANVAS: u32 = 2048;

/// Splash text colour and drop shadow, matching Minecraft's `0xFFFF00` /
/// `0x3F3F00` pair.
const SPLASH_COLOR: Color32 = Color32::from_rgb(255, 255, 60);
const SPLASH_SHADOW: Color32 = Color32::from_rgb(63, 63, 0);
/// Splash tilt in radians; negative lifts the right end like MC's splash.
const SPLASH_ANGLE: f32 = -0.34;
/// Splash size in points before the pulse, relative to the 36 pt title.
const SPLASH_SIZE: f32 = 18.0;
/// Seconds per pulse, matching Minecraft's one-second `sin` loop.
const SPLASH_PERIOD: f32 = 1.0;

pub(super) fn draw_main_menu(ui: &mut Ui, state: &mut AppState) -> Option<(u32, u32)> {
    let mut resize = None;

    egui::CentralPanel::default()
        .frame(Frame::new().fill(BG_DEEP))
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("tpt_main_menu_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add_space(26.0);
                    ui.vertical_centered(|ui| {
                        ui.set_max_width(1040.0);
                        // The Options window is reachable from the title screen
                        // as well as with Ctrl+O in game, like TPT's Settings.
                        //
                        // `allocate_ui_with_layout` caps the row height: a bare
                        // `with_layout(right_to_left, ...)` inside the scroll
                        // area inherits the full remaining height, vertically
                        // centres the button and pushes every later widget
                        // below the window, which showed up as a black title
                        // screen with a single floating Settings button.
                        ui.allocate_ui_with_layout(
                            vec2(ui.available_width(), 24.0),
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui
                                    .button(t("menu.settings"))
                                    .on_hover_text(t("menu.settings_tooltip"))
                                    .clicked()
                                {
                                    state.open_options();
                                }
                            },
                        );
                        let title = ui.label(
                            RichText::new(t("menu.title"))
                                .size(36.0)
                                .strong()
                                .color(ACCENT),
                        );
                        draw_splash(ui, title.rect);
                    });
                    ui.add_space(22.0);

                    ui.vertical_centered(|ui| {
                        ui.set_max_width(1040.0);

                        let start = card(ui, t("menu.new_simulation"), |ui| {
                            draw_new_canvas_card(ui, &mut state.new_canvas)
                        });
                        if start {
                            resize = state.start_new_simulation().or(resize);
                        }

                        ui.add_space(12.0);
                        let action = card(ui, t("menu.load_saved_simulation"), |ui| {
                            draw_load_card(ui, &mut state.new_canvas)
                        });
                        match action {
                            Some(LoadCardAction::Refresh) => state.refresh_saved_files(),
                            Some(LoadCardAction::Load(path)) => {
                                resize = state.load_save_file(&path).or(resize);
                            }
                            None => {}
                        }

                        ui.add_space(12.0);
                        card(ui, t("voxel3d.title"), |ui| draw_voxel3d_card(ui, state));

                        if !state.status_message.is_empty() {
                            ui.add_space(12.0);
                            let colour = if state.status_is_error {
                                Color32::from_rgb(255, 120, 110)
                            } else {
                                Color32::from_rgb(150, 220, 150)
                            };
                            ui.label(RichText::new(&state.status_message).color(colour));
                        }
                    });
                });
        });

    resize
}

/// Draws the Minecraft-style splash next to the title.
///
/// The splash pulses between 0.9x and 1.0x size once per [`SPLASH_PERIOD`] and
/// leans counter-clockwise, shadowed one point down and to the right. The size
/// is quantised to half points so egui's galley and font-atlas caches only see
/// a handful of entries instead of one per frame.
fn draw_splash(ui: &mut Ui, title: egui::Rect) {
    let time = ui.input(|input| input.time) as f32;
    // `MainMenu::renderSplashText`: `1 - (sin(...) + 1) * 0.05`.
    let pulse = 1.0 - ((time * std::f32::consts::TAU / SPLASH_PERIOD).sin() + 1.0) * 0.05;
    let size = (SPLASH_SIZE * pulse * 2.0).round() * 0.5;
    let painter = ui.painter();
    let galley = painter.layout(
        t("menu.splash").to_owned(),
        FontId::proportional(size),
        SPLASH_COLOR,
        f32::INFINITY,
    );
    // Anchor the left edge just past the title so the tilt cannot walk the
    // splash into the logo as it pulses.
    let origin = title.right_center() + vec2(8.0, 2.0);
    // Minecraft draws the shadow one pixel down and to the right in screen
    // space, so the offset is applied before the rotation anchor.
    painter.add(
        egui::epaint::TextShape::new(origin + vec2(1.5, 1.5), galley.clone(), SPLASH_SHADOW)
            .with_override_text_color(SPLASH_SHADOW)
            .with_angle_and_anchor(SPLASH_ANGLE, Align2::LEFT_CENTER),
    );
    painter.add(
        egui::epaint::TextShape::new(origin, galley, SPLASH_COLOR)
            .with_angle_and_anchor(SPLASH_ANGLE, Align2::LEFT_CENTER),
    );
    // The title screen is otherwise static; the splash animates on its own.
    ui.ctx().request_repaint_after(Duration::from_millis(16));
}

/// "New Simulation" card: presets, custom size, world settings and the start
/// button. Returns `true` when the user asked to start the simulation.
fn draw_new_canvas_card(ui: &mut Ui, dialog: &mut NewCanvasDialog) -> bool {
    ui.horizontal_wrapped(|ui| {
        for preset in CanvasSizePreset::ALL {
            let selected = dialog.preset == preset;
            if ui
                .selectable_label(selected, preset_button_label(preset))
                .clicked()
            {
                dialog.preset = preset;
                if let Some((width, height)) = preset.dimensions() {
                    dialog.custom_width = width;
                    dialog.custom_height = height;
                }
            }
        }
    });

    ui.add_space(8.0);

    // Both fields stay editable; typing a size switches the preset to Custom.
    let mut width = dialog.width();
    let mut height = dialog.height();
    egui::Grid::new("tpt_new_canvas_size")
        .num_columns(3)
        .spacing(vec2(10.0, 6.0))
        .show(ui, |ui| {
            ui.label(t("menu.canvas_width"));
            ui.add(
                egui::DragValue::new(&mut width)
                    .range(MIN_CANVAS..=MAX_CANVAS)
                    .speed(4.0)
                    .suffix(t("menu.canvas_px_suffix")),
            );
            ui.add(
                egui::Slider::new(&mut width, MIN_CANVAS..=MAX_CANVAS)
                    .step_by(4.0)
                    .show_value(false),
            );
            ui.end_row();

            ui.label(t("menu.canvas_height"));
            ui.add(
                egui::DragValue::new(&mut height)
                    .range(MIN_CANVAS..=MAX_CANVAS)
                    .speed(4.0)
                    .suffix(t("menu.canvas_px_suffix")),
            );
            ui.add(
                egui::Slider::new(&mut height, MIN_CANVAS..=MAX_CANVAS)
                    .step_by(4.0)
                    .show_value(false),
            );
            ui.end_row();
        });
    if width != dialog.width() || height != dialog.height() {
        dialog.preset = CanvasSizePreset::Custom;
        dialog.custom_width = align_to_cell(width);
        dialog.custom_height = align_to_cell(height);
    }

    let (resolved_w, resolved_h) = dialog.resolved_size();
    let pixels = u64::from(resolved_w) * u64::from(resolved_h);
    let cells = u64::from(resolved_w / CELL) * u64::from(resolved_h / CELL);
    ui.label(
        RichText::new(t_fmt(
            "menu.canvas_summary",
            &[
                &resolved_w.to_string(),
                &resolved_h.to_string(),
                &thousands(pixels),
                &thousands(cells),
            ],
        ))
        .color(TEXT_DIM),
    );

    ui.add_space(10.0);
    let button = egui::Button::new(RichText::new(t("menu.canvas_start")).size(16.0).strong())
        .fill(Color32::from_rgb(46, 125, 50))
        .min_size(vec2(240.0, 36.0));
    ui.add(button).clicked()
}

enum LoadCardAction {
    Refresh,
    Load(PathBuf),
}

/// "3D Sandbox" card: enter/resume, reset, save-name input and the `.tpt3`
/// browser.
fn draw_voxel3d_card(ui: &mut Ui, state: &mut AppState) {
    ui.vertical_centered(|ui| {
        wip_badge::draw_wip_badge(ui, vec2(280.0, 88.0));
    });
    ui.add_space(8.0);

    ui.horizontal_wrapped(|ui| {
        let enter_label = if state.voxel3d.is_some() {
            t("voxel3d.menu_resume")
        } else {
            t("voxel3d.menu_enter")
        };
        let button = egui::Button::new(RichText::new(enter_label).size(16.0).strong())
            .fill(Color32::from_rgb(46, 104, 152))
            .min_size(vec2(220.0, 34.0));
        if ui.add(button).clicked() {
            state.enter_voxel_mode();
        }
        if ui
            .button(t("voxel3d.menu_reset"))
            .on_hover_text(t("voxel3d.reset_tooltip"))
            .clicked()
        {
            state.reset_voxel_world();
            state.enter_voxel_mode();
        }
    });

    ui.add_space(6.0);
    ui.horizontal_wrapped(|ui| {
        ui.label(t("voxel3d.save_as"));
        ui.add(
            egui::TextEdit::singleline(&mut state.voxel_save_name_input)
                .desired_width(150.0)
                .hint_text(t("voxel3d.save_name_hint")),
        );
        if ui
            .button(t("voxel3d.save"))
            .on_hover_text(t("voxel3d.save_tooltip"))
            .clicked()
        {
            let _ = state.save_voxel_from_input();
        }
        if ui.button(t("voxel3d.load_refresh")).clicked() {
            state.refresh_voxel_saves();
        }
    });

    let saves = state.voxel_saves.clone();
    if saves.is_empty() {
        ui.label(
            RichText::new(t("voxel3d.load_empty"))
                .italics()
                .color(TEXT_DIM),
        );
        return;
    }

    let mut load = None;
    egui::ScrollArea::vertical()
        .id_salt("tpt_voxel_save_browser")
        .max_height(150.0)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for path in saves {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_else(|| t("voxel3d.load_invalid_name"));
                ui.horizontal(|ui| {
                    if ui.button(t("menu.load_button")).clicked() {
                        load = Some(path.clone());
                    }
                    ui.label(RichText::new(name).monospace());
                });
            }
        });

    if let Some(path) = load {
        state.load_voxel_save_file(&path);
    }
}

/// "Load Saved Simulation" card listing every `./saves/*.tptrs` file.
///
/// Reproduces the reference local browser (`LocalBrowserView.cpp:19-36,55-61`,
/// `LocalBrowserController.cpp:42-98`): a text filter over the save names, 20
/// rows per page with prev/next, single-row selection, an in-card delete
/// confirmation and an inline rename field. Rows are newest first and carry
/// their modification date, next to the name the reference draws under each
/// thumbnail (`SaveButton.cpp:101`, `SaveButton.cpp:219-223`).
///
/// Thumbnails are skipped: the only preview path the port exposes is
/// `SimulationEngine::render_rgba`, which needs a live engine per file, so
/// rendering one per row would mean spinning up a simulation for every save in
/// the list. The reference queues a `ThumbnailRendererTask` for exactly this
/// (`SaveButton.cpp:123-150`); the port has no cheap equivalent yet.
fn draw_load_card(ui: &mut Ui, dialog: &mut NewCanvasDialog) -> Option<LoadCardAction> {
    let mut action = None;

    ui.horizontal(|ui| {
        ui.label(
            RichText::new(t_fmt("menu.load_saves_in", &[SAVES_DIR]))
                .small()
                .color(TEXT_DIM),
        );
        if ui.button(t("menu.load_refresh")).clicked() {
            action = Some(LoadCardAction::Refresh);
        }
    });

    // Search box above the grid (`LocalBrowserView.cpp:23`); changing the
    // query restarts at page 0 (`LocalBrowserView::searchChanged`,
    // `LocalBrowserView.cpp:67-73`) and clears the selection the way
    // `LocalBrowserModel::UpdateSavesList` does (`LocalBrowserModel.cpp:76-79`).
    ui.horizontal(|ui| {
        let response = ui.add(
            egui::TextEdit::singleline(&mut dialog.save_filter)
                .desired_width(200.0)
                .hint_text(t("menu.load_filter_hint")),
        );
        if response.changed() {
            dialog.clear_save_selection();
            dialog.save_page = 0;
        }
        if !dialog.save_filter.is_empty() && ui.button(t("menu.load_filter_clear")).clicked() {
            dialog.save_filter.clear();
            dialog.clear_save_selection();
            dialog.save_page = 0;
        }
    });

    if let Some(error) = dialog.save_load_error() {
        ui.label(RichText::new(error).color(Color32::from_rgb(255, 140, 120)));
    }

    if dialog.save_entries.is_empty() {
        ui.label(
            RichText::new(t("menu.load_empty"))
                .italics()
                .color(TEXT_DIM),
        );
        return action;
    }

    let page = dialog.sync_visible_saves();
    if page.total == 0 {
        ui.label(
            RichText::new(t("menu.load_no_matches"))
                .italics()
                .color(TEXT_DIM),
        );
    } else {
        let selected_path = dialog.save_selected.clone();
        let mut clicked = None;
        egui::ScrollArea::vertical()
            .id_salt("tpt_save_browser")
            .max_height(240.0)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                for &index in &page.indices {
                    let entry = &dialog.save_entries[index];
                    ui.horizontal(|ui| {
                        if ui.button(t("menu.load_button")).clicked() {
                            action = Some(LoadCardAction::Load(entry.path.clone()));
                        }
                        let selected = selected_path.as_deref() == Some(entry.path.as_path());
                        if ui
                            .selectable_label(selected, RichText::new(&entry.name).monospace())
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
            dialog.select_save(&path);
        }

        // Prev/next only when the filtered list spills past one page, like
        // `LocalBrowserView`'s bottom bar (`LocalBrowserView.cpp:20-21,48-53`).
        if page.page_count > 1 {
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(page.page > 0, egui::Button::new(t("menu.load_prev")))
                    .clicked()
                {
                    dialog.clear_save_selection();
                    dialog.save_page = page.page - 1;
                }
                ui.label(t_fmt(
                    "menu.load_page",
                    &[&(page.page + 1).to_string(), &page.page_count.to_string()],
                ));
                if ui
                    .add_enabled(
                        page.page + 1 < page.page_count,
                        egui::Button::new(t("menu.load_next")),
                    )
                    .clicked()
                {
                    dialog.clear_save_selection();
                    dialog.save_page = page.page + 1;
                }
            });
        }
    }

    // Delete / rename act on the selected row, exactly like the reference's
    // buttons that stay hidden until something is selected
    // (`LocalBrowserView.cpp:50-60`, `LocalBrowserController.cpp:42-98`).
    let Some(selected_name) = dialog.selected_save().map(|entry| entry.name.clone()) else {
        return action;
    };

    ui.separator();
    if dialog.save_delete_confirm {
        ui.label(
            RichText::new(t_fmt("menu.load_delete_prompt", &[&selected_name]))
                .color(Color32::from_rgb(255, 180, 120)),
        );
        ui.horizontal(|ui| {
            if ui.button(t("menu.load_delete_confirm")).clicked() {
                let _ = dialog.delete_selected_save();
            }
            if ui.button(t("menu.load_cancel")).clicked() {
                dialog.save_delete_confirm = false;
            }
        });
    } else if dialog.save_rename_input.is_some() {
        let mut input = dialog.save_rename_input.take().unwrap_or_default();
        let mut submit = false;
        let mut cancel = false;
        ui.horizontal(|ui| {
            ui.label(t("menu.load_rename_prompt"));
            let response = ui.add(egui::TextEdit::singleline(&mut input).desired_width(180.0));
            submit = ui.button(t("menu.load_rename_apply")).clicked()
                || (response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)));
            cancel = ui.button(t("menu.load_cancel")).clicked();
        });
        if submit {
            // Keep the field open on failure so the user can fix the name.
            if dialog.rename_selected_save(&input).is_err() {
                dialog.save_rename_input = Some(input);
            }
        } else if cancel {
            dialog.save_rename_input = None;
        } else {
            dialog.save_rename_input = Some(input);
        }
    } else {
        ui.horizontal(|ui| {
            if ui.button(t("menu.load_rename")).clicked() {
                dialog.save_rename_input = Some(selected_name.clone());
                dialog.save_delete_confirm = false;
            }
            if ui.button(t("menu.load_delete")).clicked() {
                dialog.save_delete_confirm = true;
                dialog.save_rename_input = None;
            }
        });
    }

    action
}

/// Rounds a canvas dimension to a multiple of [`CELL`] inside the allowed range.
fn align_to_cell(value: u32) -> u32 {
    let clamped = value.clamp(MIN_CANVAS, MAX_CANVAS);
    ((clamped + CELL / 2) / CELL * CELL).clamp(MIN_CANVAS, MAX_CANVAS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_to_cell_rounds_and_clamps() {
        assert_eq!(align_to_cell(128), 128);
        assert_eq!(align_to_cell(129), 128);
        assert_eq!(align_to_cell(130), 132);
        assert_eq!(align_to_cell(0), MIN_CANVAS);
        assert_eq!(align_to_cell(99_999), MAX_CANVAS);
        assert_eq!(align_to_cell(613), 612);
    }

    /// The Settings row must stay a thin strip at the top of the title screen.
    ///
    /// Regression guard: a `with_layout(right_to_left, ...)` nested directly in
    /// the scroll area inherits its whole height, so the button was drawn
    /// vertically centred and every later menu widget was pushed below the
    /// window, leaving a black title screen with one floating button.
    #[test]
    fn settings_row_keeps_the_menu_header_at_the_top() {
        let ctx = egui::Context::default();
        let mut button_rect = egui::Rect::NOTHING;
        let mut title_rect = egui::Rect::NOTHING;

        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                vec2(1280.0, 800.0),
            )),
            ..Default::default()
        };
        let mut output = ctx.run_ui(input, |root| {
            egui::CentralPanel::default().show(root, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("menu_layout_probe")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.add_space(26.0);
                        ui.vertical_centered(|ui| {
                            ui.set_max_width(1040.0);
                            button_rect = ui
                                .allocate_ui_with_layout(
                                    vec2(ui.available_width(), 24.0),
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| ui.button(t("menu.settings")),
                                )
                                .inner
                                .rect;
                            title_rect = ui
                                .label(RichText::new(t("menu.title")).size(36.0).color(ACCENT))
                                .rect;
                        });
                    });
            });
        });
        output.textures_delta.clear();

        assert!(
            button_rect.min.y >= 20.0 && button_rect.max.y <= 60.0,
            "Settings row must stay in the header strip, got {button_rect:?}"
        );
        assert!(
            button_rect.width() > 40.0 && button_rect.width() < 200.0,
            "Settings button must keep its natural width, got {button_rect:?}"
        );
        assert!(
            button_rect.max.x <= 1280.0,
            "Settings button must stay inside the window, got {button_rect:?}"
        );
        assert!(
            title_rect.min.y < 120.0,
            "title must follow directly below the header, got {title_rect:?}"
        );
    }

    /// The splash must reach the painter tilted, shadowed and outside the title.
    #[test]
    fn splash_is_painted_rotated_beside_the_title() {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                vec2(1280.0, 800.0),
            )),
            ..Default::default()
        };
        let mut title_rect = egui::Rect::NOTHING;
        let mut output = ctx.run_ui(input, |root| {
            egui::CentralPanel::default().show(root, |ui| {
                ui.vertical_centered(|ui| {
                    title_rect = ui
                        .label(RichText::new(t("menu.title")).size(36.0).color(ACCENT))
                        .rect;
                    draw_splash(ui, title_rect);
                });
            });
        });

        let splash = t("menu.splash");
        let mut drawn = Vec::new();
        for clipped in &output.shapes {
            if let egui::Shape::Text(text) = &clipped.shape
                && text.galley.job.text == splash
            {
                drawn.push(text.clone());
            }
        }
        // One shadow plus one lit copy.
        assert_eq!(drawn.len(), 2, "splash must be drawn as text plus shadow");
        for text in &drawn {
            assert!(
                (text.angle - SPLASH_ANGLE).abs() < 1e-6,
                "splash must carry the Minecraft tilt, got {}",
                text.angle
            );
            assert!(
                text.pos.x >= title_rect.max.x,
                "splash must start beside the title, got {:?} for {title_rect:?}",
                text.pos
            );
            let bounds = text.visual_bounding_rect();
            assert!(
                bounds.min.x >= 0.0 && bounds.max.x <= 1280.0 && bounds.max.y <= 800.0,
                "the tilted splash must stay inside the title screen, got {bounds:?}"
            );
        }
        assert_eq!(
            drawn[0].override_text_color,
            Some(SPLASH_SHADOW),
            "the first copy is the drop shadow"
        );
        output.textures_delta.clear();
    }
}
