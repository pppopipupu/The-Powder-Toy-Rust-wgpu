//! Top toolbar: transport, save, settings buttons and telemetry.
//!
//! The bar is deliberately compact: every simulation, render and display
//! setting lives in the Options (TPT `OptionsView.cpp`) or Render (TPT
//! `RenderView.cpp`) window, so the HUD only carries play/pause/step/clear,
//! undo/redo, save/load, the console toggle and the two settings buttons.

use egui::{Color32, RichText, Ui};

use crate::app_state::{AppScreen, AppState, GameMode, temperature_suffix, temperature_to_scale};
use crate::i18n::{t, t_fmt};

use tpt_core::particle::GpuParticle;

use super::shared::{
    ACCENT, TEXT_DIM, cell_pressure, element_color, element_def, element_name, element_rgb,
    rigid_id_at, thousands,
};

pub(super) fn draw_top_bar(ui: &mut Ui, state: &mut AppState) {
    if state.mode == GameMode::Sandbox3D {
        draw_voxel_transport_row(ui, state);
        telemetry_row(
            ui,
            &t_fmt("top_bar.tps", &[&format!("{:5.1}", state.tps)]),
            |ui| draw_voxel_telemetry(ui, state),
        );
        draw_status_row(ui, state);
        return;
    }

    draw_transport_row(ui, state);

    telemetry_row(
        ui,
        &t_fmt("top_bar.tps", &[&format!("{:5.1}", state.tps)]),
        |ui| draw_telemetry(ui, state),
    );

    draw_status_row(ui, state);
}

/// Short status line shared by the 2D and 3D toolbars.
fn draw_status_row(ui: &mut Ui, state: &AppState) {
    if state.status_message.is_empty() {
        return;
    }
    let colour = if state.status_is_error {
        Color32::from_rgb(255, 120, 110)
    } else {
        Color32::from_rgb(150, 220, 150)
    };
    ui.label(RichText::new(&state.status_message).small().color(colour));
}

/// 3D telemetry: FPS, tick, brick count, rigid-body count and the mode.
fn draw_voxel_telemetry(ui: &mut Ui, state: &AppState) {
    ui.label(
        RichText::new(t_fmt("top_bar.fps", &[&format!("{:5.1}", state.fps)]))
            .monospace()
            .color(TEXT_DIM),
    );
    ui.separator();
    let tick = state
        .voxel3d
        .as_ref()
        .map(|engine| engine.tick)
        .unwrap_or(0);
    ui.label(RichText::new(t_fmt("voxel3d.tick", &[&tick.to_string()])).monospace());
    ui.separator();
    let bricks = state
        .voxel_active_bricks
        .map(|count| thousands(u64::from(count)))
        .unwrap_or_else(|| String::from("-"));
    ui.label(
        RichText::new(t_fmt("voxel3d.active_bricks", &[&bricks]))
            .monospace()
            .color(ACCENT),
    );
    ui.separator();
    let bodies = state
        .voxel3d
        .as_ref()
        .map(|engine| engine.rigid.body_count)
        .unwrap_or(0);
    ui.label(RichText::new(t_fmt("top_bar.rigid_bodies", &[&bodies.to_string()])).monospace());
    ui.separator();
    ui.label(
        RichText::new(t("voxel3d.world_size"))
            .monospace()
            .color(TEXT_DIM),
    );
}

/// Transport, save and settings buttons of the 3D sandbox.
fn draw_voxel_transport_row(ui: &mut Ui, state: &mut AppState) {
    ui.horizontal_wrapped(|ui| {
        if ui
            .button(t("top_bar.main_menu"))
            .on_hover_text(t("top_bar.main_menu_tooltip"))
            .clicked()
        {
            state.refresh_saved_files();
            state.screen = AppScreen::MainMenu;
        }

        ui.separator();
        let paused = state.is_paused();
        if ui
            .button(if paused {
                t("top_bar.resume")
            } else {
                t("top_bar.pause")
            })
            .clicked()
        {
            state.toggle_pause();
        }
        if ui
            .button(t("top_bar.step"))
            .on_hover_text(t("top_bar.step_tooltip"))
            .clicked()
        {
            state.step_frame();
        }
        if ui
            .button(t("voxel3d.reset"))
            .on_hover_text(t("voxel3d.reset_tooltip"))
            .clicked()
        {
            state.reset_voxel_world();
        }

        ui.separator();
        if ui
            .button(t("top_bar.undo"))
            .on_hover_text(t("top_bar.undo_tooltip"))
            .clicked()
        {
            state.do_undo();
        }
        if ui
            .button(t("top_bar.redo"))
            .on_hover_text(t("top_bar.redo_tooltip"))
            .clicked()
        {
            state.do_redo();
        }

        ui.separator();
        ui.label(t("voxel3d.save_as"));
        ui.add(
            egui::TextEdit::singleline(&mut state.voxel_save_name_input)
                .desired_width(140.0)
                .hint_text(t("voxel3d.save_name_hint")),
        );
        if ui
            .button(t("voxel3d.save"))
            .on_hover_text(t("voxel3d.save_tooltip"))
            .clicked()
        {
            let _ = state.save_voxel_from_input();
        }

        ui.separator();
        if ui
            .button(t("top_bar.settings"))
            .on_hover_text(t("top_bar.settings_tooltip"))
            .clicked()
        {
            if state.options_open {
                state.options_open = false;
            } else {
                state.open_options();
            }
        }
    });
}

/// Lays out the right-aligned TPS counter next to the telemetry row.
///
/// The right-to-left child claims the whole row, draws the TPS first and hands
/// the remaining width to `add`. That row is allocated with a fixed height and
/// never wraps: the inspector text changes width on every cursor move and the
/// counters gain and lose digits, so a wrapping row would add or drop a line
/// whenever it crossed the wrap boundary, the taller bar would shrink the
/// viewport below it, and the canvas visibly twitched.
fn telemetry_row(ui: &mut Ui, tps: &str, add: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(tps).monospace().color(TEXT_DIM));
            let height = ui.spacing().interact_size.y;
            let width = ui.available_width();
            ui.allocate_ui_with_layout(
                egui::vec2(width, height),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.set_clip_rect(ui.max_rect());
                    add(ui);
                },
            );
        });
    });
}

/// FPS, logical TPS, canvas size, particle counts and the under-cursor
/// inspector.
fn draw_telemetry(ui: &mut Ui, state: &AppState) {
    let (xres, yres) = state.canvas_size();

    ui.label(
        RichText::new(t_fmt("top_bar.fps", &[&format!("{:5.1}", state.fps)]))
            .monospace()
            .color(TEXT_DIM),
    );
    ui.separator();
    ui.label(
        RichText::new(t_fmt(
            "top_bar.canvas",
            &[&xres.to_string(), &yres.to_string()],
        ))
        .monospace(),
    );
    ui.separator();
    ui.label(
        RichText::new(t_fmt(
            "top_bar.active_particles",
            &[&thousands(state.active_particles() as u64)],
        ))
        .monospace(),
    );
    ui.separator();
    ui.label(
        RichText::new(t_fmt(
            "top_bar.rigid_bodies",
            &[&state.rigid_body_count().to_string()],
        ))
        .monospace(),
    );
    ui.separator();
    if (state.engine.bufs.config.colour_mode & tpt_core::constants::COLOUR_HEAT) != 0 {
        let (hmin, hmax) = state.resolve_heat_display_limits();
        ui.label(
            RichText::new(t_fmt(
                "top_bar.temp_range",
                &[&format!("{hmin:.1}"), &format!("{hmax:.1}")],
            ))
            .monospace()
            .color(ACCENT),
        );
        ui.separator();
    }

    match state.hover_pos {
        Some((x, y)) => {
            ui.label(
                RichText::new(format!("({x}, {y})"))
                    .monospace()
                    .color(ACCENT),
            );
            draw_inspector(ui, state, x, y);
        }
        None => {
            ui.label(RichText::new("(-, -)").monospace().color(TEXT_DIM));
            ui.label(
                RichText::new(t("top_bar.hint_cursor"))
                    .small()
                    .color(TEXT_DIM),
            );
        }
    }
}

/// Text for the `Ctype` field of the inspector.
///
/// Ported from the reference HUD (`GameView.cpp:2419-2432`). Most elements keep
/// an element id in `ctype` and the HUD names it, but several keep something
/// else entirely: `LITH` stores its absorbed energy there (`LITH.cpp:71`
/// `storedEnergy`, accumulating `+= 1` / `-= 1` with thresholds at 24, 90 and
/// 100), the wavelength elements store a wavelength, and `GLOW`/`WIRE`/`SOAP`
/// store a bit pattern. Naming those as elements is what printed `Ctype GAS`
/// on a lithium pixel holding 10 units of energy, so the reference prints the
/// raw number for them instead.
fn ctype_text(state: &AppState, particle: &GpuParticle) -> String {
    match ctype_field(particle.ty, particle.ctype, particle.tmp, |id| {
        element_def(state, id).is_some()
    }) {
        CtypeField::None => t("top_bar.inspector_none").to_string(),
        CtypeField::Number(value) => value.to_string(),
        CtypeField::Element(element, rule) => {
            if element == tpt_core::constants::PT_LIFE {
                tpt_sim::gol_kernel::serialise_gol_rule(tpt_sim::gol_kernel::resolve_gol_rule(rule))
            } else {
                element_name(state, element)
            }
        }
    }
}

/// How the inspector renders `ctype` for one particle.
#[derive(Debug, PartialEq, Eq)]
enum CtypeField {
    /// `ctype == 0`: the reference prints no value at all.
    None,
    /// The raw number: `ctype` holds a charge, a wavelength or a bit pattern.
    Number(u32),
    /// An element name, with the `life`/rule argument `ElementResolve` takes.
    Element(u32, u32),
}

/// Classifies `ctype` exactly as `GameView.cpp:2419-2432` does.
fn ctype_field(ty: u32, ctype: u32, tmp: i32, is_element: impl Fn(u32) -> bool) -> CtypeField {
    use tpt_core::constants::{
        PT_BCLN, PT_BIZR, PT_BIZRG, PT_BIZRS, PT_BRAY, PT_C5, PT_CLNE, PT_CONV, PT_CRAY, PT_DRAY,
        PT_DTEC, PT_FILT, PT_GLOW, PT_LDTC, PT_LITH, PT_PBCN, PT_PCLN, PT_PHOT, PT_SOAP, PT_WIRE,
    };

    if ctype == 0 {
        return CtypeField::None;
    }
    // `GameView.cpp:2420-2424`: `ctype` is a wavelength for these, so the
    // reference prints no parenthetical at all. The inspector has a dedicated
    // row, and the raw value is the wavelength, so it prints that number.
    if matches!(
        ty,
        PT_PHOT | PT_BIZR | PT_BIZRG | PT_BIZRS | PT_FILT | PT_BRAY | PT_C5
    ) {
        return CtypeField::Number(ctype);
    }
    // `GameView.cpp:2426-2427`: the low nine bits name the element, the rest is
    // the stored `life` (`TYP`/`ID`).
    if matches!(ty, PT_CRAY | PT_DRAY | PT_CONV | PT_LDTC) {
        return CtypeField::Element(ctype & 0x1FF, ctype >> 9);
    }
    // `GameView.cpp:2428-2429`: clones carry the element to spawn in `ctype`
    // and the GoL rule in `tmp`.
    if matches!(ty, PT_CLNE | PT_BCLN | PT_PCLN | PT_PBCN | PT_DTEC) {
        return CtypeField::Element(ctype, tmp.max(0) as u32);
    }
    // `GameView.cpp:2430-2432`: an element id only when it names one and the
    // element does not overload `ctype`; otherwise the plain number.
    if is_element(ctype) && !matches!(ty, PT_GLOW | PT_WIRE | PT_SOAP | PT_LITH) {
        return CtypeField::Element(ctype, 0);
    }
    CtypeField::Number(ctype)
}

/// Element / wall details for the pixel under the cursor.
fn draw_inspector(ui: &mut Ui, state: &AppState, x: i32, y: i32) {
    if let Some(particle) = state.particle_at(x, y) {
        let colour = element_color(element_rgb(state, particle.ty));
        ui.label(
            RichText::new(element_name(state, particle.ty))
                .strong()
                .color(colour),
        );
        ui.separator();
        ui.label(
            RichText::new(t_fmt(
                "top_bar.inspector_temp",
                &[&format!(
                    "{:.1}{}",
                    temperature_to_scale(particle.temp, state.prefs.temperature_scale),
                    temperature_suffix(state.prefs.temperature_scale),
                )],
            ))
            .monospace(),
        );
        ui.separator();
        ui.label(
            RichText::new(t_fmt(
                "top_bar.inspector_pressure",
                &[&format!("{:+.2}", cell_pressure(state, x, y))],
            ))
            .monospace(),
        );
        ui.separator();
        ui.label(
            RichText::new(t_fmt(
                "top_bar.inspector_life",
                &[&particle.life.to_string()],
            ))
            .monospace(),
        );
        ui.separator();
        let ctype = ctype_text(state, &particle);
        ui.label(RichText::new(t_fmt("top_bar.inspector_ctype", &[&ctype])).monospace());
        if let Some(body_id) = rigid_id_at(state, x, y) {
            ui.separator();
            ui.label(
                RichText::new(t_fmt("top_bar.inspector_rigid_id", &[&body_id.to_string()]))
                    .monospace()
                    .color(Color32::from_rgb(140, 200, 255)),
            );
        }
        return;
    }

    if let Some(w) = state.wall_sample_at(x, y) {
        ui.label(RichText::new(w.summary(state.prefs.temperature_scale)).color(TEXT_DIM));
        return;
    }

    // The panel reports the air pressure under the cursor even where no
    // particle or wall sits, so the ambient field is always readable.
    ui.label(RichText::new(t("top_bar.inspector_empty")).color(TEXT_DIM));
    ui.separator();
    ui.label(
        RichText::new(t_fmt(
            "top_bar.inspector_pressure",
            &[&format!("{:+.2}", cell_pressure(state, x, y))],
        ))
        .monospace(),
    );
}

/// Transport, save, undo and settings buttons.
fn draw_transport_row(ui: &mut Ui, state: &mut AppState) {
    ui.horizontal_wrapped(|ui| {
        if ui
            .button(t("top_bar.main_menu"))
            .on_hover_text(t("top_bar.main_menu_tooltip"))
            .clicked()
        {
            state.refresh_saved_files();
            state.screen = AppScreen::MainMenu;
            state.property_window.close();
            state.close_sign_editor();
        }

        ui.separator();
        let paused = state.is_paused();
        if ui
            .button(if paused {
                t("top_bar.resume")
            } else {
                t("top_bar.pause")
            })
            .clicked()
        {
            state.toggle_pause();
        }
        if ui
            .button(t("top_bar.step"))
            .on_hover_text(t("top_bar.step_tooltip"))
            .clicked()
        {
            state.step_frame();
        }
        if ui
            .button(t("top_bar.clear"))
            .on_hover_text(t("top_bar.clear_tooltip"))
            .clicked()
        {
            state.clear_sim();
        }

        ui.separator();
        if ui
            .selectable_label(state.move_view, t("top_bar.view_mode"))
            .on_hover_text(t("top_bar.view_mode_tooltip"))
            .clicked()
        {
            state.move_view = !state.move_view;
        }

        ui.separator();
        if ui
            .button(t("top_bar.undo"))
            .on_hover_text(t("top_bar.undo_tooltip"))
            .clicked()
        {
            state.do_undo();
        }
        if ui
            .button(t("top_bar.redo"))
            .on_hover_text(t("top_bar.redo_tooltip"))
            .clicked()
        {
            state.do_redo();
        }

        ui.separator();
        ui.label(t("top_bar.save_as"));
        ui.add(
            egui::TextEdit::singleline(&mut state.save_name_input)
                .desired_width(150.0)
                .hint_text(t("top_bar.save_name_hint")),
        );
        if ui
            .button(t("top_bar.save"))
            .on_hover_text(t("top_bar.save_tooltip"))
            .clicked()
        {
            let _ = state.save_current_from_input();
        }

        ui.separator();
        if ui
            .selectable_label(state.show_lua_console, t("top_bar.lua_console"))
            .on_hover_text(t("top_bar.lua_console_tooltip"))
            .clicked()
        {
            state.show_lua_console = !state.show_lua_console;
        }

        // The two settings windows are the only homes for simulation, render
        // and display options; the HUD does not duplicate any of them.
        ui.separator();
        if ui
            .button(t("top_bar.settings"))
            .on_hover_text(t("top_bar.settings_tooltip"))
            .clicked()
        {
            if state.options_open {
                state.options_open = false;
            } else {
                state.open_options();
            }
        }
        if ui
            .button(t("top_bar.render_options"))
            .on_hover_text(t("top_bar.render_options_tooltip"))
            .clicked()
        {
            if state.render_window_open {
                state.render_window_open = false;
            } else {
                state.open_render_window();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Rect, pos2, vec2};

    /// Height of a top bar built from [`telemetry_row`] with `text` inside it.
    fn bar_height(text: &str) -> f32 {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 240.0))),
            ..Default::default()
        };
        let mut height = 0.0;
        let mut output = ctx.run_ui(input, |root| {
            let response = egui::Panel::top("probe_top_bar").show(root, |ui| {
                telemetry_row(ui, "TPS  60.0", |ui| {
                    ui.label(RichText::new(text).monospace());
                });
            });
            height = response.response.rect.height();
        });
        output.textures_delta.clear();
        height
    }

    /// Regression guard: the telemetry row used to wrap, so a wider inspector
    /// string made the top bar a line taller, which shrank the viewport and
    /// rescaled the simulation canvas on every cursor move.
    #[test]
    fn telemetry_row_height_does_not_follow_its_content() {
        let short = bar_height("(12, 34)");
        let long = bar_height(&"WATR  293.1 K   +0.42   life 128   ctype NONE ".repeat(6));
        assert_eq!(short, long, "the top bar must not grow with its content");
    }

    /// `GameView.cpp:2419-2432`: a lithium pixel prints its stored energy
    /// (`LITH.cpp:71`), not the element that shares the numeric value.
    #[test]
    fn ctype_prints_the_stored_energy_for_lithium() {
        let all_elements = |id: u32| id < 195;
        assert_eq!(
            ctype_field(tpt_core::constants::PT_LITH, 3, 0, all_elements),
            CtypeField::Number(3)
        );
        assert_eq!(
            ctype_field(tpt_core::constants::PT_LITH, 0, 0, all_elements),
            CtypeField::None
        );
    }

    /// The elements that overload `ctype` with a wavelength or a bit pattern
    /// print the raw value even when it names a real element.
    #[test]
    fn ctype_prints_raw_values_for_overloaded_elements() {
        let all_elements = |id: u32| id < 195;
        for ty in [
            tpt_core::constants::PT_PHOT,
            tpt_core::constants::PT_FILT,
            tpt_core::constants::PT_GLOW,
            tpt_core::constants::PT_WIRE,
            tpt_core::constants::PT_SOAP,
        ] {
            assert_eq!(
                ctype_field(ty, tpt_core::constants::PT_GAS, 0, all_elements),
                CtypeField::Number(tpt_core::constants::PT_GAS),
                "element {ty} must print the raw ctype"
            );
        }
    }

    /// Clone and ray elements resolve `ctype` to an element name, and the ray
    /// family splits the stored `life` out of the high bits.
    #[test]
    fn ctype_resolves_element_ids_for_clones_and_rays() {
        let all_elements = |id: u32| id < 195;
        assert_eq!(
            ctype_field(
                tpt_core::constants::PT_CLNE,
                tpt_core::constants::PT_WATR,
                0,
                all_elements
            ),
            CtypeField::Element(tpt_core::constants::PT_WATR, 0)
        );
        // `TYP`/`ID` (`ElementDefs.h:66-74`): low nine bits type, high bits life.
        let packed = (7u32 << 9) | tpt_core::constants::PT_SAND;
        assert_eq!(
            ctype_field(tpt_core::constants::PT_CRAY, packed, 0, all_elements),
            CtypeField::Element(tpt_core::constants::PT_SAND, 7)
        );
        // A non-element value falls back to the number.
        assert_eq!(
            ctype_field(tpt_core::constants::PT_WATR, 400, 0, all_elements),
            CtypeField::Number(400)
        );
    }
}
