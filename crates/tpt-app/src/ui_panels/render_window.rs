//! Floating Render window (display, render and colour modes, presets) and the
//! floating decoration colour picker it opens.

use egui::{Color32, CornerRadius, RichText, Sense, Stroke, StrokeKind};
use tpt_core::constants::{
    COLOUR_BASC, COLOUR_DEFAULT, COLOUR_DEST, COLOUR_GRAD, COLOUR_HEAT, COLOUR_LIFE, DISPLAY_AIRC,
    DISPLAY_AIRH, DISPLAY_AIRP, DISPLAY_AIRV, DISPLAY_AIRW, DISPLAY_EFFE, DISPLAY_PERS,
    DISPLAY_WARP, MAX_TEMP, MIN_TEMP, RENDER_BASC, RENDER_BLOB, RENDER_BLUR, RENDER_EFFE,
    RENDER_FIRE, RENDER_GLOW, RENDER_SPRK,
};

use crate::app_state::AppState;
use crate::i18n::{t, t_fmt};

use super::widgets::{display_mode_checkbox, option_info, render_mode_checkbox};

/// One entry of the render preset list, mirroring TPT's
/// `Renderer::renderModePresets`.
struct RenderPreset {
    /// Button label translation key.
    label_key: &'static str,
    /// Tooltip translation key shown on hover.
    tooltip_key: &'static str,
    /// `RENDER_*` bitmask applied by the preset.
    render_mode: u32,
    /// `DISPLAY_*` bitmask applied by the preset.
    display_mode: u32,
    /// `COLOUR_*` mode applied by the preset.
    colour_mode: u32,
    /// `true` when the preset wants `HdispLimitAuto` (TPT's Dynamic Heat).
    heat_auto: bool,
}

/// Render presets exposed by the Render window, in TPT order
/// (`Renderer::renderModePresets`, 14 entries).
const RENDER_PRESETS: [RenderPreset; 14] = [
    RenderPreset {
        label_key: "render.preset_alt_velocity",
        tooltip_key: "render.preset_alt_velocity_desc",
        render_mode: RENDER_EFFE | RENDER_BASC,
        display_mode: DISPLAY_AIRC,
        heat_auto: false,
        colour_mode: 0,
    },
    RenderPreset {
        label_key: "render.preset_velocity",
        tooltip_key: "render.preset_velocity_desc",
        render_mode: RENDER_EFFE | RENDER_BASC,
        display_mode: DISPLAY_AIRV,
        heat_auto: false,
        colour_mode: 0,
    },
    RenderPreset {
        label_key: "render.preset_pressure",
        tooltip_key: "render.preset_pressure_desc",
        render_mode: RENDER_EFFE | RENDER_BASC,
        display_mode: DISPLAY_AIRP,
        heat_auto: false,
        colour_mode: 0,
    },
    RenderPreset {
        label_key: "render.preset_persistent",
        tooltip_key: "render.preset_persistent_desc",
        render_mode: RENDER_EFFE | RENDER_BASC,
        display_mode: DISPLAY_PERS,
        heat_auto: false,
        colour_mode: 0,
    },
    RenderPreset {
        label_key: "render.preset_fire",
        tooltip_key: "render.preset_fire_desc",
        render_mode: RENDER_FIRE | RENDER_SPRK | RENDER_EFFE | RENDER_BASC,
        display_mode: 0,
        heat_auto: false,
        colour_mode: 0,
    },
    RenderPreset {
        label_key: "render.preset_blob",
        tooltip_key: "render.preset_blob_desc",
        render_mode: RENDER_FIRE | RENDER_SPRK | RENDER_EFFE | RENDER_BLOB,
        display_mode: 0,
        heat_auto: false,
        colour_mode: 0,
    },
    RenderPreset {
        label_key: "render.preset_heat",
        tooltip_key: "render.preset_heat_desc",
        render_mode: RENDER_BASC,
        display_mode: DISPLAY_AIRH,
        heat_auto: false,
        colour_mode: COLOUR_HEAT,
    },
    RenderPreset {
        label_key: "render.preset_fancy",
        tooltip_key: "render.preset_fancy_desc",
        render_mode: RENDER_FIRE
            | RENDER_SPRK
            | RENDER_GLOW
            | RENDER_BLUR
            | RENDER_EFFE
            | RENDER_BASC,
        display_mode: DISPLAY_WARP,
        heat_auto: false,
        colour_mode: 0,
    },
    RenderPreset {
        label_key: "render.preset_nothing",
        tooltip_key: "render.preset_nothing_desc",
        render_mode: RENDER_BASC,
        display_mode: 0,
        heat_auto: false,
        colour_mode: 0,
    },
    RenderPreset {
        label_key: "render.preset_heat_gradient",
        tooltip_key: "render.preset_heat_gradient_desc",
        render_mode: RENDER_BASC,
        display_mode: 0,
        heat_auto: false,
        colour_mode: COLOUR_GRAD,
    },
    RenderPreset {
        label_key: "render.preset_life_gradient",
        tooltip_key: "render.preset_life_gradient_desc",
        render_mode: RENDER_BASC,
        display_mode: 0,
        heat_auto: false,
        colour_mode: COLOUR_LIFE,
    },
    RenderPreset {
        label_key: "render.preset_dynamic_heat",
        tooltip_key: "render.preset_dynamic_heat_desc",
        render_mode: RENDER_BASC,
        display_mode: DISPLAY_AIRH,
        heat_auto: true,
        colour_mode: COLOUR_HEAT,
    },
    RenderPreset {
        label_key: "render.preset_vorticity",
        tooltip_key: "render.preset_vorticity_desc",
        render_mode: RENDER_EFFE | RENDER_BASC,
        display_mode: DISPLAY_AIRW,
        heat_auto: false,
        colour_mode: 0,
    },
    RenderPreset {
        label_key: "render.preset_destructibility",
        tooltip_key: "render.preset_destructibility_desc",
        render_mode: RENDER_BASC,
        display_mode: 0,
        heat_auto: false,
        colour_mode: COLOUR_DEST,
    },
];

/// Draws the floating Render window when [`AppState::render_window_open`] is
/// set. Mirrors `RenderView.cpp` (display, render and colour modes, presets and
/// the fire intensity slider).
pub(super) fn draw_render_window(ctx: &egui::Context, state: &mut AppState) {
    // The colour picker is its own window (`ColourPickerActivity`), so it
    // survives closing the Render window that opened it.
    draw_colour_picker(ctx, state);

    if !state.render_window_open {
        return;
    }

    let mut open = true;
    let mut changed = false;

    egui::Window::new(t("render.title"))
        .open(&mut open)
        .resizable(true)
        .default_width(460.0)
        .default_height(540.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("tpt_render_scroll")
                .max_height(560.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.label(RichText::new(t("render.display_modes")).strong());
                    changed |= display_mode_checkbox(
                        ui,
                        &mut state.prefs.display_mode,
                        DISPLAY_AIRC,
                        t("render.disp_alt_velocity"),
                        t("render.disp_alt_velocity_desc"),
                    );
                    changed |= display_mode_checkbox(
                        ui,
                        &mut state.prefs.display_mode,
                        DISPLAY_AIRP,
                        t("render.disp_pressure"),
                        t("render.disp_pressure_desc"),
                    );
                    changed |= display_mode_checkbox(
                        ui,
                        &mut state.prefs.display_mode,
                        DISPLAY_AIRV,
                        t("render.disp_velocity"),
                        t("render.disp_velocity_desc"),
                    );
                    changed |= display_mode_checkbox(
                        ui,
                        &mut state.prefs.display_mode,
                        DISPLAY_AIRH,
                        t("render.disp_ambient_heat"),
                        t("render.disp_ambient_heat_desc"),
                    );
                    changed |= display_mode_checkbox(
                        ui,
                        &mut state.prefs.display_mode,
                        DISPLAY_AIRW,
                        t("render.disp_vorticity"),
                        t("render.disp_vorticity_desc"),
                    );
                    changed |= display_mode_checkbox(
                        ui,
                        &mut state.prefs.display_mode,
                        DISPLAY_WARP,
                        t("render.disp_gravity_lensing"),
                        t("render.disp_gravity_lensing_desc"),
                    );
                    changed |= display_mode_checkbox(
                        ui,
                        &mut state.prefs.display_mode,
                        DISPLAY_EFFE,
                        t("render.disp_effects"),
                        t("render.disp_effects_desc"),
                    );
                    changed |= display_mode_checkbox(
                        ui,
                        &mut state.prefs.display_mode,
                        DISPLAY_PERS,
                        t("render.disp_persistent"),
                        t("render.disp_persistent_desc"),
                    );
                    changed |= ui
                        .checkbox(
                            &mut state.prefs.gravity_field,
                            t("render.disp_gravity_grid"),
                        )
                        .on_hover_text(t("render.disp_gravity_grid_desc"))
                        .changed();

                    ui.separator();
                    ui.label(RichText::new(t("render.render_modes")).strong());
                    changed |= render_mode_checkbox(
                        ui,
                        &mut state.prefs.render_mode,
                        RENDER_EFFE,
                        t("render.ren_effects"),
                        t("render.ren_effects_desc"),
                    );
                    changed |= render_mode_checkbox(
                        ui,
                        &mut state.prefs.render_mode,
                        RENDER_FIRE,
                        t("render.ren_fire"),
                        t("render.ren_fire_desc"),
                    );
                    changed |= render_mode_checkbox(
                        ui,
                        &mut state.prefs.render_mode,
                        RENDER_GLOW,
                        t("render.ren_glow"),
                        t("render.ren_glow_desc"),
                    );
                    changed |= render_mode_checkbox(
                        ui,
                        &mut state.prefs.render_mode,
                        RENDER_BLUR,
                        t("render.ren_blur"),
                        t("render.ren_blur_desc"),
                    );
                    changed |= render_mode_checkbox(
                        ui,
                        &mut state.prefs.render_mode,
                        RENDER_SPRK,
                        t("render.ren_spark"),
                        t("render.ren_spark_desc"),
                    );
                    changed |= render_mode_checkbox(
                        ui,
                        &mut state.prefs.render_mode,
                        RENDER_BLOB,
                        t("render.ren_blob"),
                        t("render.ren_blob_desc"),
                    );
                    changed |= render_mode_checkbox(
                        ui,
                        &mut state.prefs.render_mode,
                        RENDER_BASC,
                        t("render.ren_basic"),
                        t("render.ren_basic_desc"),
                    );

                    ui.separator();
                    ui.label(RichText::new(t("render.colour_modes")).strong());
                    ui.horizontal_wrapped(|ui| {
                        for (mode, label_key, tooltip_key) in [
                            (
                                COLOUR_DEFAULT,
                                "render.colour_default",
                                "render.colour_default_desc",
                            ),
                            (COLOUR_HEAT, "render.colour_heat", "render.colour_heat_desc"),
                            (COLOUR_LIFE, "render.colour_life", "render.colour_life_desc"),
                            (
                                COLOUR_GRAD,
                                "render.colour_gradient",
                                "render.colour_gradient_desc",
                            ),
                            (
                                COLOUR_BASC,
                                "render.colour_basic",
                                "render.colour_basic_desc",
                            ),
                            (
                                COLOUR_DEST,
                                "render.colour_destructibility",
                                "render.colour_destructibility_desc",
                            ),
                        ] {
                            if ui
                                .selectable_label(state.prefs.colour_mode == mode, t(label_key))
                                .on_hover_text(t(tooltip_key))
                                .clicked()
                                && state.prefs.colour_mode != mode
                            {
                                state.prefs.colour_mode = mode;
                                changed = true;
                            }
                        }
                    });

                    draw_heat_range(ui, state, &mut changed);
                    draw_decoration_colour(ui, state);

                    ui.separator();
                    ui.label(RichText::new(t("render.presets")).strong());
                    ui.horizontal_wrapped(|ui| {
                        for preset in &RENDER_PRESETS {
                            if ui
                                .button(t(preset.label_key))
                                .on_hover_text(t(preset.tooltip_key))
                                .clicked()
                            {
                                state.prefs.render_mode = preset.render_mode;
                                state.prefs.display_mode = preset.display_mode;
                                state.prefs.colour_mode = preset.colour_mode;
                                state.engine.bufs.config.heat_display_limit_min =
                                    if preset.heat_auto {
                                        None
                                    } else {
                                        Some(tpt_core::constants::MIN_TEMP)
                                    };
                                state.engine.bufs.config.heat_display_limit_max =
                                    if preset.heat_auto {
                                        None
                                    } else {
                                        Some(tpt_core::constants::MAX_TEMP)
                                    };
                                changed = true;
                            }
                        }
                    });

                    ui.separator();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut state.prefs.fire_intensity, 0.0..=4.0)
                                .text(t("render.fire_intensity"))
                                .max_decimals(2),
                        )
                        .changed();
                    option_info(ui, t("render.fire_intensity_desc"));
                });
        });

    state.render_window_open = open;
    if changed {
        state.apply_prefs();
    }
}

/// Heat-display range controls.
///
/// TPT stores each end of the scale independently as `HdispLimitExplicit` or
/// `HdispLimitAuto` (`RendererSettings.h:10-19`), and `Renderer::AdjustHdispLimit`
/// (`Renderer.cpp:1444-1512`) resolves the pair every frame. The two checkboxes
/// therefore drive the configuration's `Option<f32>` fields directly: `None`
/// is the auto variant, `Some` the manual one. Dragging a value while auto is
/// on switches that end to manual at the resolved value, which is exactly the
/// "user edited a bound by hand" transition.
fn draw_heat_range(ui: &mut egui::Ui, state: &mut AppState, changed: &mut bool) {
    ui.separator();
    ui.label(RichText::new(t("render.heat_range")).strong());

    // `GameView.cpp:2563-2569` prints the same resolved pair in the FPS info.
    let (auto_min, auto_max, valid) = tpt_sim::render_kernel::current_hdisp_limits();
    let cfg = &mut state.engine.bufs.config;

    ui.horizontal(|ui| {
        let mut auto = cfg.heat_display_limit_min.is_none();
        if ui.checkbox(&mut auto, t("render.heat_auto_min")).changed() {
            cfg.heat_display_limit_min = if auto { None } else { Some(auto_min) };
            *changed = true;
        }
        let mut min = cfg.heat_display_limit_min.unwrap_or(auto_min);
        if ui
            .add(
                egui::DragValue::new(&mut min)
                    .speed(1.0)
                    .range(MIN_TEMP..=MAX_TEMP)
                    .prefix(t("render.heat_min")),
            )
            .changed()
        {
            cfg.heat_display_limit_min = Some(min);
            *changed = true;
        }
    });

    ui.horizontal(|ui| {
        let mut auto = cfg.heat_display_limit_max.is_none();
        if ui.checkbox(&mut auto, t("render.heat_auto_max")).changed() {
            cfg.heat_display_limit_max = if auto { None } else { Some(auto_max) };
            *changed = true;
        }
        let mut max = cfg.heat_display_limit_max.unwrap_or(auto_max);
        if ui
            .add(
                egui::DragValue::new(&mut max)
                    .speed(1.0)
                    .range(MIN_TEMP..=MAX_TEMP)
                    .prefix(t("render.heat_max")),
            )
            .changed()
        {
            cfg.heat_display_limit_max = Some(max);
            *changed = true;
        }
    });

    if cfg.heat_display_limit_min.is_none() || cfg.heat_display_limit_max.is_none() {
        if valid {
            ui.label(t_fmt(
                "render.heat_range_auto",
                &[&format!("{auto_min:.1}"), &format!("{auto_max:.1}")],
            ));
        } else {
            ui.label(t("render.heat_range_no_samples"));
        }
    }
}

/// The eight decoration presets and the button that opens the colour picker.
///
/// `GameView::NotifyColourPresetsChanged` (`GameView.cpp:679-706`) lays the
/// swatches out in a row and applies one on click; `OpenColourPicker`
/// (`GameController.cpp:1378-1382`) opens the free-form window. The picked
/// colour is the decoration colour every `DECO` tool blends over the element
/// display (`GameModel::SetColourSelectorColour`, `GameModel.cpp:1098-1107`).
fn draw_decoration_colour(ui: &mut egui::Ui, state: &mut AppState) {
    ui.separator();
    ui.label(RichText::new(t("render.decoration_colour")).strong());
    ui.horizontal_wrapped(|ui| {
        let presets = state.colour_picker.presets;
        let active = state.colour_picker.active_preset;
        for (index, preset) in presets.into_iter().enumerate() {
            if colour_swatch(ui, preset, active == Some(index)).clicked() {
                state.colour_picker.active_preset = Some(index);
                state.set_decoration_colour(preset);
            }
        }
        if ui
            .button(t("render.open_colour_picker"))
            .on_hover_text(t("render.open_colour_picker_desc"))
            .clicked()
        {
            state.open_colour_picker();
        }
    });
    ui.label(t("render.decoration_colour_desc"));
}

/// One preset / preview swatch, sized like TPT's 30x18 `ToolButton`
/// (`GameView.cpp:690-702`).
fn colour_swatch(
    ui: &mut egui::Ui,
    [red, green, blue, alpha]: [u8; 4],
    selected: bool,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(22.0, 18.0), Sense::click());
    if ui.is_rect_visible(rect) {
        ui.painter().rect_filled(
            rect,
            CornerRadius::same(3),
            Color32::from_rgba_unmultiplied(red, green, blue, alpha),
        );
        if selected {
            ui.painter().rect_stroke(
                rect,
                CornerRadius::same(3),
                Stroke::new(1.5, Color32::WHITE),
                StrokeKind::Inside,
            );
            ui.painter().rect_stroke(
                rect,
                CornerRadius::same(3),
                Stroke::new(1.0, Color32::BLACK),
                StrokeKind::Outside,
            );
        }
    }
    response.on_hover_text(format!("0x{red:02X}{green:02X}{blue:02X}"))
}

/// Draws the floating decoration colour picker while it is open.
///
/// Ports `ColourPickerActivity.cpp`: the eight editable presets of
/// `GameModel::colourPresets` on top, the HSV sliders of the original window
/// (`ColourPickerActivity.cpp:63-75`) and the RGBA numeric entry with the
/// `0xAARRGGBB` label (`ColourPickerActivity.cpp:86-117`). `Done` applies the
/// colour and closes the window; the title-bar close discards the draft, like
/// `ColourPickerActivity::OnTryExit`.
fn draw_colour_picker(ctx: &egui::Context, state: &mut AppState) {
    if !state.colour_picker.open {
        return;
    }

    let mut picker = state.colour_picker.clone();
    let mut open = true;
    let mut done = false;

    egui::Window::new(t("render.picker_title"))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_width(320.0)
        .show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(t("render.picker_presets"));
                for index in 0..picker.presets.len() {
                    let colour = picker.presets[index];
                    if colour_swatch(ui, colour, picker.active_preset == Some(index)).clicked() {
                        picker.active_preset = Some(index);
                        picker.set_rgba(colour);
                    }
                }
            });
            ui.separator();

            ui.horizontal(|ui| {
                colour_swatch(ui, picker.rgba(), false);
                ui.label(RichText::new(picker.hex()).monospace());
            });

            // `currentHue/Saturation/Value` sliders: the slider callback owns
            // the HSV triple, the RGB entry below derives from it.
            let mut hue = picker.hue;
            let mut saturation = picker.saturation;
            let mut value = picker.value;
            let mut hsv_changed = ui
                .add(egui::Slider::new(&mut hue, 0..=359).text(t("render.picker_hue")))
                .changed();
            hsv_changed |= ui
                .add(
                    egui::Slider::new(&mut saturation, 0..=255).text(t("render.picker_saturation")),
                )
                .changed();
            hsv_changed |= ui
                .add(egui::Slider::new(&mut value, 0..=255).text(t("render.picker_value")))
                .changed();
            if hsv_changed {
                picker.set_hsv(hue, saturation, value);
            }

            let [mut red, mut green, mut blue, mut alpha] = picker.rgba();
            let mut rgb_changed = false;
            ui.horizontal(|ui| {
                rgb_changed |= ui
                    .add(
                        egui::DragValue::new(&mut red)
                            .range(0..=255)
                            .prefix(t("render.picker_red")),
                    )
                    .changed();
                rgb_changed |= ui
                    .add(
                        egui::DragValue::new(&mut green)
                            .range(0..=255)
                            .prefix(t("render.picker_green")),
                    )
                    .changed();
                rgb_changed |= ui
                    .add(
                        egui::DragValue::new(&mut blue)
                            .range(0..=255)
                            .prefix(t("render.picker_blue")),
                    )
                    .changed();
            });
            let alpha_changed = ui
                .add(
                    egui::DragValue::new(&mut alpha)
                        .range(0..=255)
                        .prefix(t("render.picker_alpha")),
                )
                .changed();
            if rgb_changed || alpha_changed {
                picker.set_rgba([red, green, blue, alpha]);
            }

            ui.separator();
            if ui.button(t("render.picker_done")).clicked() {
                done = true;
            }
        });

    if done {
        state.colour_picker.active_preset = picker.active_preset;
        state.set_decoration_colour(picker.rgba());
        state.colour_picker.open = false;
    } else if !open {
        // Title-bar close: `OnTryExit` exits without invoking `onPicked`.
        state.colour_picker.open = false;
    } else {
        // Keep the draft across frames; `open_from` reseeds it every time the
        // window is opened, exactly like the reference recreating the window.
        state.colour_picker = picker;
        state.colour_picker.open = true;
    }
}
