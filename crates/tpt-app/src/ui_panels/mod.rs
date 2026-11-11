//! egui panels for The Powder Toy (Rust + wgpu + CubeCL).
//!
//! [`draw_app_ui`] is the single per-frame entry point used by `main.rs`. It
//! builds the main menu or the in-game HUD on the root [`egui::Ui`] and returns
//! the new canvas dimensions whenever an action replaced the simulation with a
//! canvas of a different size (new canvas or loaded save), so the
//! renderer can resize the `sim_texture` registered with `egui-wgpu`.
//!
//! Layout of the in-game screen:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │ top panel: transport, save, display mode, toggles, telemetry │
//! ├───────────────────────────────────────────┬──────────────────┤
//! │ central panel: simulation viewport        │ right panel:     │
//! │  - sim texture (zoom / pan)               │  category tabs   │
//! │  - brush, line, rect, rigid-body previews │  bound tools     │
//! ├───────────────────────────────────────────┤  viewport help   │
//! │ bottom panel: brush controls + tool grid  │                  │
//! └───────────────────────────────────────────┴──────────────────┘
//! ```
//!
//! The floating Lua console, the Settings window, the Render window and the
//! stamp browser are [`egui::Window`]s drawn on top of everything; the shared
//! modal layer (`modal`) paints the newest confirm/text/error dialog above
//! them and owns the keyboard.
//!
//! Submodules, one per panel: `menu` (title screen), `top_bar`, `side_bar`,
//! `bottom_bar` (brush controls and palette grids), `viewport` (image, input
//! and previews), `console`, `options_window`, `render_window`,
//! `property_window`, `sign_window`, `modal` (confirm/text/error dialogs),
//! plus `widgets` (settings controls) and `shared` (palette, naming and button
//! helpers).
//! `hotkeys` owns the in-game key map; this file only draws the panels.

mod bottom_bar;
mod console;
mod element_search;
mod hotkeys;
mod menu;
mod modal;
mod options_window;
mod property_window;
mod render_window;
mod shared;
mod side_bar;
mod sign_window;
mod top_bar;
mod viewport;
mod voxel_options;
mod voxel_palette;
mod voxel_viewport;
mod widgets;
mod wip_badge;

use egui::{Frame, Margin, TextureId, Ui};

use crate::app_state::{AppScreen, AppState, GameMode};

use self::hotkeys::{draw_finding_highlight, handle_hotkeys};
use self::shared::{BG_PANEL, BG_VIEWPORT};

/// Draws the complete application UI for one frame.
///
/// Returns `Some((width, height))` when the simulation canvas changed size, so
/// the caller can resize the GPU simulation texture and its bind group.
/// Returns `Some((width, height))` when the simulation canvas changed size, so
/// the caller can resize the GPU simulation texture and its bind group.
pub fn draw_app_ui(
    ctx: &egui::Context,
    state: &mut AppState,
    sim_tex_id: egui::TextureId,
    view3d_tex_id: egui::TextureId,
) -> Option<(u32, u32)> {
    let mut resize = None;

    // Capture results (HUD status) and `tpt.record` requests are drained once
    // per UI frame, before any panel reads `status_message`.
    state.poll_screenshot_events();

    // Copy the screen out first: matching on `&state.screen` would keep a
    // shared borrow of `state` alive across the mutable calls in the arms.
    let screen = state.screen;

    // egui 0.37 panels are added to a `Ui`, not to the `Context`; `root_ui`
    // hands out the full-window root Ui for the current pass.
    ctx.root_ui(|ui| match screen {
        AppScreen::MainMenu => resize = menu::draw_main_menu(ui, state),
        AppScreen::InGame => resize = draw_in_game(ui, state, sim_tex_id, view3d_tex_id),
    });

    options_window::draw_options_window(ctx, state);
    if state.mode == GameMode::Powder2D {
        render_window::draw_render_window(ctx, state);
    }
    // Newest modal paints last so its backdrop blocks the panels below and it
    // owns the keyboard (`ui::Window::MakeActiveWindow`).
    modal::draw_modals(ctx, state);

    // The simulation advances every frame, so keep the event loop awake.
    ctx.request_repaint();
    resize
}

/// Draws the in-game HUD: panels around the simulation viewport plus the
/// floating Lua console.
fn draw_in_game(
    ui: &mut Ui,
    state: &mut AppState,
    sim_tex_id: TextureId,
    view3d_tex_id: TextureId,
) -> Option<(u32, u32)> {
    // Loading a save snapshots the previous canvas, so Ctrl+Z can restore a
    // simulation with a different canvas size; the caller only recreates the
    // simulation texture when this function reports the new size.
    let canvas_before = state.canvas_size();
    handle_hotkeys(ui.ctx(), state);
    let sandbox = state.mode == GameMode::Sandbox3D;
    if sandbox || state.engine.bufs.config.hud {
        egui::Panel::top("tpt_top_bar")
            .frame(
                Frame::new()
                    .fill(BG_PANEL)
                    .inner_margin(Margin::symmetric(8, 5)),
            )
            .show(ui, |ui| top_bar::draw_top_bar(ui, state));

        egui::Panel::right("tpt_side_bar")
            .default_size(214.0)
            .min_size(170.0)
            .max_size(340.0)
            .resizable(true)
            .frame(
                Frame::new()
                    .fill(BG_PANEL)
                    .inner_margin(Margin::symmetric(8, 6)),
            )
            .show(ui, |ui| {
                if sandbox {
                    voxel_palette::draw_voxel_palette(ui, state);
                } else {
                    side_bar::draw_side_bar(ui, state);
                }
            });

        // The 2D and 3D bottom bars both hold brush controls plus the active
        // category's spawn grid (3D `Tools` holds the tool buttons), so they
        // share the default size; the panel id and limits stay shared.
        let bottom_default = 184.0;
        egui::Panel::bottom("tpt_bottom_bar")
            .default_size(bottom_default)
            .min_size(110.0)
            .max_size(420.0)
            .resizable(true)
            .frame(
                Frame::new()
                    .fill(BG_PANEL)
                    .inner_margin(Margin::symmetric(8, 6)),
            )
            .show(ui, |ui| {
                if sandbox {
                    voxel_palette::draw_voxel_bottom_bar(ui, state);
                } else {
                    bottom_bar::draw_bottom_bar(ui, state);
                }
            });
    }
    let mut viewport_rect = None;
    egui::CentralPanel::default()
        .frame(Frame::new().fill(BG_VIEWPORT))
        .show(ui, |ui| {
            if sandbox {
                voxel_viewport::draw_voxel_viewport(ui, state, view3d_tex_id);
            } else {
                viewport_rect = Some(ui.max_rect());
                viewport::draw_viewport(ui, state, sim_tex_id);
            }
        });

    if let Some(rect) = viewport_rect {
        // `GameView.cpp:1464-1476` / `Renderer.cpp:460-520`: while Ctrl+F is
        // armed, the matching particles are tinted over the live image. The
        // viewport renders from the device, so refresh the host shadows first.
        draw_finding_highlight(ui.ctx(), state, rect);
    }

    if !sandbox {
        // The Lua console, property, sign editors and stamp browser all edit
        // 2D simulation data, so they stay hidden while the 3D sandbox is
        // active.
        console::draw_lua_console(ui.ctx(), state);
        property_window::draw_property_window(ui.ctx(), state);
        sign_window::draw_sign_window(ui.ctx(), state);
        // `GameController::OpenStamps` (`GameController.cpp:1399-1410`): a
        // floating window, so it stays reachable with the HUD hidden.
        side_bar::draw_stamp_browser(ui.ctx(), state);
    }
    let canvas = state.canvas_size();
    (canvas != canvas_before).then_some(canvas)
}

#[cfg(test)]
mod tests {
    use super::hotkeys::apply_stickman_key;
    use super::hotkeys::pressed_once;
    use egui::{Context, Event, Key, Modifiers, RawInput};

    /// `GameController.cpp:622-662`: presses set bits, and the placement
    /// check only covers the arrow-key player.
    #[test]
    fn stickman_press_sets_command_bits() {
        let mut c = [0u32; 4];
        apply_stickman_key(&mut c, Key::ArrowLeft, true, false);
        apply_stickman_key(&mut c, Key::ArrowRight, true, false);
        apply_stickman_key(&mut c, Key::ArrowUp, true, false);
        apply_stickman_key(&mut c, Key::ArrowDown, true, false);
        assert_eq!(c[0], 0x0F);
        assert_eq!(c[2], 0, "WASD must not touch the first player");

        let mut placing = [0u32; 4];
        apply_stickman_key(&mut placing, Key::ArrowLeft, true, true);
        assert_eq!(placing[0], 0, "arrows move the save, not the stickman");
        apply_stickman_key(&mut placing, Key::A, true, true);
        assert_eq!(placing[2], 0x01, "WASD is outside the placement check");
    }

    /// `GameController.cpp:702-728`: releases clear bits and save `pcomm`.
    #[test]
    fn stickman_release_clears_bits_and_saves_direction() {
        let mut c = [0u32; 4];
        apply_stickman_key(&mut c, Key::ArrowLeft, true, false);
        apply_stickman_key(&mut c, Key::ArrowUp, true, false);
        apply_stickman_key(&mut c, Key::ArrowDown, true, false);
        assert_eq!(c[0], 0x0D);
        apply_stickman_key(&mut c, Key::ArrowLeft, false, false);
        assert_eq!(c[0], 0x0C, "a horizontal release keeps jump/use");
        assert_eq!(c[1], 0x0D, "pcomm snapshots the direction word");
        apply_stickman_key(&mut c, Key::ArrowUp, false, false);
        assert_eq!(c[0], 0x08, "the jump release clears only jump");
        apply_stickman_key(&mut c, Key::ArrowDown, false, false);
        assert_eq!(c[0], 0x00, "the use release clears only use");
        assert_eq!(c[1], 0x0D, "pcomm survives the other releases");
    }

    /// `comm &= 12` drops both horizontal bits on any left/right release.
    #[test]
    fn stickman_release_clears_both_horizontal_bits() {
        let mut c = [0u32; 4];
        apply_stickman_key(&mut c, Key::W, true, false);
        apply_stickman_key(&mut c, Key::A, true, false);
        apply_stickman_key(&mut c, Key::D, true, false);
        assert_eq!(c[2], 0x07);
        apply_stickman_key(&mut c, Key::A, false, false);
        assert_eq!(c[2], 0x04, "left and right are cleared together");
        assert_eq!(c[3], 0x07);
    }

    /// Builds one key event, optionally marked as an auto-repeat.
    fn key_event(key: Key, repeat: bool) -> Event {
        Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat,
            modifiers: Modifiers::NONE,
        }
    }

    /// `GameView.cpp:1408-1409` drops repeated key presses, so a held key must
    /// not re-fire its shortcut. Holding Z used to clear the pinned zoom scope
    /// on every repeat.
    /// Runs one pass with `events`; returns (`key_pressed`, `pressed_once`).
    fn pass_with(ctx: &Context, events: Vec<Event>) -> (bool, bool) {
        ctx.begin_pass(RawInput {
            events,
            ..Default::default()
        });
        let seen = ctx.input(|input| {
            (
                input.key_pressed(Key::Z),
                pressed_once(input, Key::Z),
            )
        });
        // epaint asserts when a `TexturesDelta` is dropped unapplied; the test
        // has no GPU texture to hand it to, so drop it deliberately.
        let mut output = ctx.end_pass();
        output.textures_delta.clear();
        seen
    }

    #[test]
    fn pressed_once_sees_a_fresh_press() {
        let ctx = Context::default();
        assert_eq!(pass_with(&ctx, vec![key_event(Key::Z, false)]), (true, true));
    }

    /// `GameView.cpp:1419-1434,1676-1684` plus `OnMouseUp`: hold Z, click to
    /// pin the scope, release Z and it stays put; a later Z press unfixes it
    /// and the following release closes it.
    #[test]
    fn zoom_scope_pins_on_click_and_survives_the_key_release() {
        use super::hotkeys::zoom_key_state;

        // Z goes down: the scope opens and is not pinned.
        let (enabled, fixed) = zoom_key_state(true, true, false, false);
        assert!(enabled && !fixed);

        // Held: still open, still following the cursor.
        let (enabled, fixed) = zoom_key_state(false, true, fixed, enabled);
        assert!(enabled && !fixed);

        // The click pins it (done by the viewport on mouse release).
        let fixed = true;

        // Z goes up: a pinned scope survives.
        let (enabled, pinned) = zoom_key_state(false, false, fixed, enabled);
        assert!(enabled && pinned);

        // Z goes down again: unfixed and open.
        let (enabled, pinned) = zoom_key_state(true, true, pinned, enabled);
        assert!(enabled && !pinned);

        // Releasing without a click closes it.
        let (enabled, _) = zoom_key_state(false, false, pinned, enabled);
        assert!(!enabled);
    }

    /// egui marks the second press of a held key as a repeat and still counts
    /// it as a press, which is what made a pinned zoom scope start following
    /// the cursor again.
    #[test]
    fn pressed_once_ignores_auto_repeat() {
        let ctx = Context::default();
        assert_eq!(pass_with(&ctx, vec![key_event(Key::Z, false)]), (true, true));
        assert_eq!(pass_with(&ctx, vec![key_event(Key::Z, false)]), (true, false));
    }
}
