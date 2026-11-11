//! In-game keyboard shortcuts.
//!
//! [`handle_hotkeys`] ports `GameView::OnKeyPress` / `OnKeyRelease` and the
//! `GameController` key handlers. This module owns only key handling; the panel
//! layout stays in `mod.rs`.

use std::path::Path;

use egui::{Color32, Id, Key, LayerId, Order, Rect, Vec2, vec2};

use tpt_core::commands::SimCommand;
use tpt_core::constants::PT_NONE;
use tpt_core::elements::MenuCategory;
use tpt_core::particle::{AirMode, EdgeMode};

use crate::app_state::{
    AppState, GameMode, PropertyKind, PropertyValue, SAVE_EXTENSION, SAVES_DIR, SelectedTool,
};
use crate::brushes::adjust_brush_size;
use crate::i18n::{t, t_fmt};

use super::element_search;
use super::shared::{DisplayPreset, RenderPreset, element_name};
use super::viewport;

/// `GameView::OnKeyPress`/`OnKeyRelease` SDL_SCANCODE_Z: the next
/// `(zoom_enabled, zoomCursorFixed)` pair.
///
/// A fresh press unfixes the scope and opens it, holding the key keeps it
/// open, and the release closes it only when nothing pinned it. The pin
/// itself happens in the viewport (`GameView::OnMouseUp`), which runs after
/// this in the same frame.
pub(super) fn zoom_key_state(down: bool, held: bool, fixed: bool, enabled: bool) -> (bool, bool) {
    if down {
        // `GameView.cpp:1430-1434`.
        return (true, false);
    }
    if !held && !fixed {
        // `GameView.cpp:1680-1684`: an unpinned scope dies with the key.
        return (false, false);
    }
    (enabled, fixed)
}

/// True only on the frame the key goes down, never on auto-repeat.
///
/// egui's [`egui::InputState::key_pressed`] counts repeat events as presses
/// (`InputState::num_presses`), while the reference drops every repeated key
/// press outright (`GameView.cpp:1408-1409`). Without this, holding a key
/// re-fires its shortcut: holding Z cleared `zoomCursorFixed` on every repeat,
/// so a pinned zoom scope started following the cursor again.
pub(super) fn pressed_once(input: &egui::InputState, key: egui::Key) -> bool {
    input.events.iter().any(|event| {
        matches!(
            event,
            egui::Event::Key {
                key: pressed,
                pressed: true,
                repeat: false,
                ..
            } if *pressed == key
        )
    })
}

/// Keyboard shortcuts. Ignored while a text edit has keyboard focus so typing
/// a save name or a Lua script never triggers simulation hotkeys.
pub(super) fn handle_hotkeys(ctx: &egui::Context, state: &mut AppState) {
    // The newest modal is the active window and owns the keyboard, so no
    // simulation or panel shortcut fires while one is open.
    if ctx.text_edit_focused() || !state.modals.is_empty() {
        clear_stickman_controls(state);
        return;
    }
    let sandbox = state.mode == GameMode::Sandbox3D;

    // Lua gets every key press first; a truthy `evt.KEYPRESS` handler suppresses
    // the built-in shortcut for that key. The 3D sandbox never ticks Lua, so
    // the forwarding path is 2D-only.
    let suppressed = if sandbox {
        Vec::new()
    } else {
        fire_lua_key_presses(ctx, state)
    };
    let free = |key: Key| !suppressed.contains(&key);

    // `GameController::KeyPress` / `KeyRelease`
    // (`GameController.cpp:611-730`): arrow keys drive `STKM`, WASD drives
    // `STKM2`. View mode, the 3D sandbox and the property window own those
    // keys instead; the element search is covered by `text_edit_focused`.
    if sandbox || state.move_view || state.property_window.open {
        clear_stickman_controls(state);
    } else {
        update_stickman_controls(ctx, state, &free);
    }

    let (pause, step, undo, redo, shrink, grow, zoom_down, zoom_held) = ctx.input(|input| {
        let ctrl = input.modifiers.ctrl || input.modifiers.command;
        let shift = input.modifiers.shift;
        (
            pressed_once(input, Key::Space) && free(Key::Space),
            pressed_once(input, Key::F) && free(Key::F),
            // `GameView::OnKeyDown` SDL_SCANCODE_Z: Ctrl+Z restores the
            // history and Ctrl+Shift+Z steps forward again; a bare Z opens
            // the zoom scope instead.
            pressed_once(input, Key::Z) && ctrl && !shift && free(Key::Z),
            (pressed_once(input, Key::Z) && ctrl && shift && free(Key::Z))
                || (pressed_once(input, Key::Y) && free(Key::Y)),
            pressed_once(input, Key::OpenBracket) && free(Key::OpenBracket),
            pressed_once(input, Key::CloseBracket) && free(Key::CloseBracket),
            pressed_once(input, Key::Z) && !ctrl && free(Key::Z),
            input.key_down(Key::Z),
        )
    });

    if pause {
        state.toggle_pause();
    }
    if step {
        state.step_frame();
    }
    if undo {
        state.do_undo();
    }
    if redo {
        state.do_redo();
    }
    // `GameView::OnKeyDown`/`OnKeyRelease` SDL_SCANCODE_Z: the scope follows
    // the cursor while the key is held, and releasing it hides the scope again
    // unless a click inside the magnification window pinned it
    // (`zoomCursorFixed`).
    if !sandbox {
        let (enabled, fixed) = zoom_key_state(
            zoom_down,
            zoom_held,
            state.zoom_cursor_fixed,
            state.engine.bufs.config.zoom_enabled,
        );
        state.engine.bufs.config.zoom_enabled = enabled;
        state.zoom_cursor_fixed = fixed;
    }
    // `GameView::OnKeyDown` SDL_SCANCODE_C/X/V (`GameView.cpp:1548-1585`):
    // the region clipboard. Escape and the right button abandon a pending
    // selection.
    if !sandbox {
        let (copy, cut, paste, escape) = ctx.input(|input| {
            let ctrl = input.modifiers.ctrl || input.modifiers.command;
            (
                ctrl && pressed_once(input, Key::C),
                ctrl && pressed_once(input, Key::X),
                ctrl && pressed_once(input, Key::V),
                pressed_once(input, Key::Escape),
            )
        });
        if copy {
            state.begin_selection(crate::app_state::selection::SelectMode::Copy);
        }
        if cut {
            state.begin_selection(crate::app_state::selection::SelectMode::Cut);
        }
        if paste {
            // `OptionsView.cpp:413-415`: with the platform clipboard on, a
            // region copied by another instance can be pasted here. Unrelated
            // text is ignored, leaving the in-process clipboard untouched.
            if state.prefs.native_clipboard
                && let Some(text) = ctx.input(|input| {
                    input.events.iter().rev().find_map(|event| match event {
                        egui::Event::Paste(text) => Some(text.clone()),
                        _ => None,
                    })
                })
                && let Some(region) =
                    crate::app_state::selection::Region::from_clipboard_text(&text)
            {
                state.clipboard = Some(region);
            }
            state.begin_paste();
        }
        if escape {
            state.cancel_selection();
            state.placement = None;
        }
        // `GameView.cpp:1458-1462`: a bare `E` opens the element search;
        // Ctrl+E stays the edge-mode toggle.
        let search = ctx.input(|input| {
            pressed_once(input, Key::E) && !(input.modifiers.ctrl || input.modifiers.command)
        });
        if search {
            element_search::open_element_search(state);
        }

        // `GameView.cpp:1449-1462`: bare `P` / `F2` set `doScreenshot`, which
        // `GameView::Render` turns into `TakeScreenshot(0, 0)`. Ctrl+P and
        // Ctrl+Shift+P keep binding the property tool to the mouse buttons
        // (`bottom_bar::handle_tool_shortcuts`).
        let (screenshot_p, screenshot_f2) = ctx.input(|input| {
            let ctrl = input.modifiers.ctrl || input.modifiers.command;
            (
                !ctrl && pressed_once(input, Key::P) && free(Key::P),
                pressed_once(input, Key::F2) && free(Key::F2),
            )
        });
        if screenshot_p || screenshot_f2 {
            state.take_screenshot();
        }

        // `GameView::OnKeyPress` SDL_SCANCODE_L/K (`GameView.cpp:1583-1601`):
        // a bare `L` places the newest stamp (`Client::GetStamps()[0]`), a
        // bare `K` opens the stamp browser, and Ctrl+S arms the stamp rubber
        // band (`GameController.cpp:674-675`,
        // `GameView::BeginStampSelection`, `GameView.cpp:1351-1360`).
        let (stamp_capture, stamp_browse, stamp_place) = ctx.input(|input| {
            let ctrl = input.modifiers.ctrl || input.modifiers.command;
            (
                ctrl && pressed_once(input, Key::S) && free(Key::S),
                !ctrl && pressed_once(input, Key::K) && free(Key::K),
                !ctrl && pressed_once(input, Key::L) && free(Key::L),
            )
        });
        if stamp_capture {
            state.begin_stamp_selection();
        }
        if stamp_browse {
            state.open_stamp_browser();
        }
        if stamp_place && !state.place_newest_stamp() {
            // `GameView.cpp:1583-1601`: the `L` case has no `break` when the
            // stamp list is empty, so it falls through to the `K` case and
            // opens the browser.
            state.open_stamp_browser();
        }

        // `GameView.cpp:1539-1542` SDL_SCANCODE_U: Ctrl+U resets the ambient
        // heat field to the configured ambient temperature (a bare U toggles
        // ambient heat simulation in the reference; the toggle lives in the
        // options window here). `GameView.cpp:1607-1612` SDL_SCANCODE_I: a bare
        // I runs `InvertAirSim`, negating pressure and both air velocity
        // components; Ctrl+I runs `GameController::Install`, which on this
        // portable build reports that no installation is necessary
        // (`GameController.cpp:266-285`, `CAN_INSTALL == false` branch).
        let (invert_air, reset_ambient_heat, install) = ctx.input(|input| {
            let ctrl = input.modifiers.ctrl || input.modifiers.command;
            (
                pressed_once(input, Key::I) && !ctrl && free(Key::I),
                pressed_once(input, Key::U) && ctrl && free(Key::U),
                pressed_once(input, Key::I) && ctrl && free(Key::I),
            )
        });
        if invert_air {
            state.engine.apply_command(SimCommand::InvertAir);
        }
        if reset_ambient_heat {
            state.engine.apply_command(SimCommand::ResetAmbientHeat);
        }
        if install {
            state
                .modals
                .open_error(t("app.install_title"), t("app.install_message"));
        }

        // `GameView.cpp:1380-1404` (`OnKeyPress`, `SelectMode::PlaceSave`):
        // while a stamp / clipboard placement is armed, the arrow keys
        // translate it and R rotates it (Shift+R mirrors horizontally,
        // Ctrl+Shift+R vertically). This runs before the general map so
        // Ctrl+R transforms the placement instead of reloading the save.
        let (placement_keys, placement_r, placement_ctrl, placement_shift) = ctx.input(|input| {
            (
                (
                    pressed_once(input, Key::ArrowLeft),
                    pressed_once(input, Key::ArrowRight),
                    pressed_once(input, Key::ArrowUp),
                    pressed_once(input, Key::ArrowDown),
                ),
                pressed_once(input, Key::R),
                input.modifiers.ctrl || input.modifiers.command,
                input.modifiers.shift,
            )
        });
        let mut placement_transformed = false;
        if let Some(placement) = state.placement.as_mut() {
            let (left, right, up, down) = placement_keys;
            if left {
                placement.translate(-1, 0);
            }
            if right {
                placement.translate(1, 0);
            }
            if up {
                placement.translate(0, -1);
            }
            if down {
                placement.translate(0, 1);
            }
            if placement_r {
                if placement_ctrl && placement_shift {
                    // Vertical flip (`Mat2::MirrorY`).
                    placement.mirror_vertical();
                } else if placement_shift {
                    // Horizontal flip (`Mat2::MirrorX`).
                    placement.mirror_horizontal();
                } else {
                    // Rotate 90 degrees counter-clockwise.
                    placement.rotate_ccw();
                }
                placement_transformed = true;
            }
        }

        // `GameView.cpp:1452-1535`: the rest of the reference keyboard map.
        let (
            debug_hud,
            reload_f5,
            reload_ctrl_r,
            edge_cycle,
            finding_toggle,
            grid_next,
            grid_previous,
            gravity_grid,
            hud_toggle,
            fullscreen_toggle,
            decoration_toggle,
            decoration_tools,
            air_cycle,
            newtonian_toggle,
            reset_air,
            reset_sparks,
            replace_toggle,
            specific_delete_toggle,
            insert_key,
            delete_key,
            brush_cycle,
        ) = ctx.input(|input| {
            let ctrl = input.modifiers.ctrl || input.modifiers.command;
            let shift = input.modifiers.shift;
            (
                pressed_once(input, Key::F3) && free(Key::F3),
                pressed_once(input, Key::F5) && free(Key::F5),
                ctrl && pressed_once(input, Key::R) && free(Key::R) && !placement_transformed,
                ctrl && pressed_once(input, Key::E) && free(Key::E),
                ctrl && pressed_once(input, Key::F) && free(Key::F),
                !ctrl && !shift && pressed_once(input, Key::G) && free(Key::G),
                !ctrl && shift && pressed_once(input, Key::G) && free(Key::G),
                ctrl && pressed_once(input, Key::G) && free(Key::G),
                !ctrl && pressed_once(input, Key::H) && free(Key::H),
                pressed_once(input, Key::F11),
                ctrl && pressed_once(input, Key::B) && free(Key::B),
                !ctrl && pressed_once(input, Key::B) && free(Key::B),
                !ctrl && pressed_once(input, Key::Y) && free(Key::Y),
                pressed_once(input, Key::N) && free(Key::N),
                !ctrl && pressed_once(input, Key::Equals) && free(Key::Equals),
                ctrl && pressed_once(input, Key::Equals) && free(Key::Equals),
                !ctrl && pressed_once(input, Key::Semicolon) && free(Key::Semicolon),
                ctrl && pressed_once(input, Key::Semicolon) && free(Key::Semicolon),
                pressed_once(input, Key::Insert) && free(Key::Insert),
                pressed_once(input, Key::Delete) && free(Key::Delete),
                !ctrl && pressed_once(input, Key::Tab) && free(Key::Tab),
            )
        });

        // `GameView.cpp:1452-1453`: F3 toggles the debug telemetry overlay.
        if debug_hud {
            let visible = !state.engine.bufs.config.debug_hud;
            state.engine.bufs.config.debug_hud = visible;
            state.set_status(
                t(if visible {
                    "app.status_debug_hud_on"
                } else {
                    "app.status_debug_hud_off"
                }),
                false,
            );
        }
        // `GameView.cpp:1455-1466`: F5 / Ctrl+R reload the current save.
        if reload_f5 || reload_ctrl_r {
            reload_simulation(state);
        }
        // `GameView.cpp:1460`: Ctrl+E cycles the edge mode through
        // `SimulationEngine::set_edge_mode`.
        if edge_cycle {
            cycle_edge_mode(state);
        }
        // `GameView.cpp:1468-1478`: Ctrl+F arms the element highlight.
        if finding_toggle {
            toggle_finding(ctx, state);
        }
        // `GameView.cpp:1481-1491`: G / Shift+G cycle the grid size, Ctrl+G
        // toggles the gravity field overlay.
        if grid_next {
            adjust_grid_size(state, 1);
        }
        if grid_previous {
            adjust_grid_size(state, -1);
        }
        if gravity_grid {
            let visible = !state.engine.bufs.config.gravity_field_enabled;
            state.engine.bufs.config.gravity_field_enabled = visible;
            state.engine.gravity_grid = visible as u32;
            state.prefs.gravity_field = visible;
            state.set_status(
                t(if visible {
                    "app.status_gravity_grid_on"
                } else {
                    "app.status_gravity_grid_off"
                }),
                false,
            );
        }
        // `GameView.cpp:1508-1510`: H hides or shows the HUD.
        if hud_toggle {
            let visible = !state.engine.bufs.config.hud;
            state.engine.bufs.config.hud = visible;
            state.set_status(
                t(if visible {
                    "app.status_hud_on"
                } else {
                    "app.status_hud_off"
                }),
                false,
            );
        }
        // `GameView.cpp:1495-1496`: F11 toggles fullscreen at runtime.
        if fullscreen_toggle {
            let fullscreen = ctx.input(|input| input.viewport().fullscreen.unwrap_or(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(!fullscreen));
            state.set_status(
                t(if fullscreen {
                    "app.status_fullscreen_off"
                } else {
                    "app.status_fullscreen_on"
                }),
                false,
            );
        }
        // `GameView.cpp:1513-1521`: Ctrl+B toggles decoration rendering, bare
        // B opens the decoration tool page (`SC_DECO`).
        if decoration_toggle {
            let visible = !state.engine.bufs.config.decorations;
            state.engine.bufs.config.decorations = visible;
            state.prefs.decorations = visible;
            state.set_status(
                t(if visible {
                    "app.status_decoration_on"
                } else {
                    "app.status_decoration_off"
                }),
                false,
            );
        }
        if decoration_tools {
            // The port merges the reference `SC_DECO` page into the `Tools`
            // category, in the same place as the `SimTool` / GUI tool rows.
            // `GameView.cpp:1513-1521` also switches decoration rendering on
            // when the page opens (`SetDecoration(true)`), which is what makes
            // the decoration tools' effect visible immediately.
            state.active_category = MenuCategory::Tools;
            state.engine.bufs.config.decorations = true;
            state.prefs.decorations = true;
            state.set_status(t("app.status_decoration_tools"), false);
        }
        // `GameView.cpp:1522-1530`: Y cycles the air mode.
        if air_cycle {
            cycle_air_mode(state);
        }
        // `GameView.cpp:1534-1535`: N toggles Newtonian gravity.
        if newtonian_toggle {
            let enabled = !state.engine.bufs.config.newton_grav;
            state.engine.bufs.config.newton_grav = enabled;
            state.prefs.newtonian_gravity = enabled;
            if !enabled {
                // `Simulation::EnableNewtonianGravity(false)` clears the
                // cached force field immediately, not only on the next tick.
                state.engine.bufs.zero_gravity_forces();
            }
            state.set_status(
                t(if enabled {
                    "app.status_newtonian_on"
                } else {
                    "app.status_newtonian_off"
                }),
                false,
            );
        }
        // `GameView.cpp:1538-1542`: `=` resets the air, Ctrl+`=` the sparks.
        if reset_air {
            let (xcells, ycells) = {
                let config = &state.engine.bufs.config;
                (config.xcells() as i32, config.ycells() as i32)
            };
            state.engine.apply_command(SimCommand::ResetPressure {
                x: 0,
                y: 0,
                w: xcells,
                h: ycells,
            });
            state.engine.apply_command(SimCommand::ResetVelocity {
                x: 0,
                y: 0,
                w: xcells,
                h: ycells,
            });
            state.set_status(t("app.status_air_reset"), false);
        }
        if reset_sparks {
            state.engine.apply_command(SimCommand::ResetSpark);
            state.set_status(t("app.status_sparks_reset"), false);
        }
        // `GameView.cpp:1625-1650`: `;`, Insert and Delete toggle replace mode
        // and the separate `SPECIFIC_DELETE` flag.
        if replace_toggle {
            toggle_replace_flags(state, false);
        }
        if specific_delete_toggle {
            toggle_replace_flags(state, true);
        }
        let insert_specific =
            insert_key && ctx.input(|input| input.modifiers.ctrl || input.modifiers.command);
        if insert_key && !insert_specific {
            toggle_replace_flags(state, false);
        }
        if insert_specific || delete_key {
            toggle_replace_flags(state, true);
        }
        // `GameView.cpp:1640-1641`: Tab cycles the brush shape.
        if brush_cycle {
            let shape = match state.brush_shape {
                tpt_core::commands::BrushShape::Circle => tpt_core::commands::BrushShape::Square,
                tpt_core::commands::BrushShape::Square => tpt_core::commands::BrushShape::Triangle,
                tpt_core::commands::BrushShape::Triangle => tpt_core::commands::BrushShape::Circle,
            };
            state.brush_shape = shape;
            let label = match shape {
                tpt_core::commands::BrushShape::Circle => t("bottom_bar.brush_circle"),
                tpt_core::commands::BrushShape::Square => t("bottom_bar.brush_square"),
                tpt_core::commands::BrushShape::Triangle => t("bottom_bar.brush_triangle"),
            };
            state.set_status(t_fmt("app.status_brush", &[label]), false);
        }
    }
    let logarithmic = ctx.input(|input| !input.modifiers.alt);
    // `GameView::OnKeyDown` SDL_SCANCODE_LEFTBRACKET/RIGHTBRACKET: the brackets
    // resize the zoom scope while it is open, and the brush otherwise.
    let zooming = !sandbox && state.engine.bufs.config.zoom_enabled && !state.zoom_cursor_fixed;
    if shrink {
        if sandbox {
            state.adjust_voxel_brush(-1, logarithmic);
        } else if zooming {
            state.adjust_zoom_size(-1, logarithmic);
        } else {
            adjust_brush_size(&mut state.brush_rx, &mut state.brush_ry, -1, false, false);
        }
    }
    if grow {
        if sandbox {
            state.adjust_voxel_brush(1, logarithmic);
        } else if zooming {
            state.adjust_zoom_size(1, logarithmic);
        } else {
            adjust_brush_size(&mut state.brush_rx, &mut state.brush_ry, 1, false, false);
        }
    }

    // The 1-5 display presets only exist in the 2D renderer. `Shift+1/6/0/7`
    // are the reference's four extra render presets 10..13
    // (`GameView.cpp:1655-1669`), so the plain digits must ignore Shift.
    if sandbox {
        return;
    }
    let shift = ctx.input(|input| input.modifiers.shift);
    for (key, preset) in [
        (Key::Num1, DisplayPreset::Normal),
        (Key::Num2, DisplayPreset::FireGlow),
        (Key::Num3, DisplayPreset::Thermal),
        (Key::Num4, DisplayPreset::Pressure),
        (Key::Num5, DisplayPreset::Velocity),
    ] {
        if !shift && free(key) && ctx.input(|input| pressed_once(input, key)) {
            preset.apply(&mut state.engine.bufs.config);
        }
    }
    if shift {
        for key in [Key::Num1, Key::Num6, Key::Num0, Key::Num7] {
            if !free(key) || !ctx.input(|input| pressed_once(input, key)) {
                continue;
            }
            if let Some(preset) = RenderPreset::for_shift_digit(key) {
                preset.apply(&mut state.engine.bufs.config);
                state.set_status(t_fmt("app.status_render_preset", &[preset.name()]), false);
            }
        }
    }
}

/// Applies one `GameController::KeyPress`/`KeyRelease` transition for a
/// stickman control key to the `[comm1, pcomm1, comm2, pcomm2]` word array.
///
/// `GameController.cpp:622-662` ORs the bit in on a press; `:702-728`
/// clears it on a release and snapshots `comm` into `pcomm` when a
/// left/right key goes up (`STKM.cpp:451, 501-529` uses that direction for
/// the head and for spawned elements).
pub(super) fn apply_stickman_key(
    control: &mut [u32; 4],
    key: Key,
    pressed: bool,
    placing_save: bool,
) {
    const LEFT: u32 = 0x01;
    const RIGHT: u32 = 0x02;
    const JUMP: u32 = 0x04;
    const USE: u32 = 0x08;

    let (player, bit, arrow) = match key {
        Key::ArrowLeft => (0, LEFT, true),
        Key::ArrowRight => (0, RIGHT, true),
        Key::ArrowUp => (0, JUMP, true),
        Key::ArrowDown => (0, USE, true),
        Key::A => (1, LEFT, false),
        Key::D => (1, RIGHT, false),
        Key::W => (1, JUMP, false),
        Key::S => (1, USE, false),
        _ => return,
    };
    let i = player * 2;
    if pressed {
        // `GameController.cpp:618-641`: while a save is being placed the
        // arrow keys translate it instead of driving `sim->player`. WASD is
        // handled outside that check (`:643-662`).
        if arrow && placing_save {
            return;
        }
        // The reference guards jump/use with "only if the bit is clear"
        // (`:632, 637, 654, 659`); a bitwise OR is idempotent, so the guard
        // adds nothing once auto-repeat is filtered by the caller.
        control[i] |= bit;
    } else {
        let comm = control[i];
        if bit == LEFT || bit == RIGHT {
            // `pcomm = comm` is taken *before* the clear in the reference
            // (`:704-705, 718-719`), so it keeps the last horizontal bit.
            control[i + 1] = comm;
            control[i] = comm & 12; // keep jump/use, drop left and right
        } else if bit == JUMP {
            control[i] = comm & 11; // 0b1011
        } else {
            control[i] = comm & 7; // 0b0111
        }
    }
}

/// Reads this frame's key transitions and pushes the resulting command
/// words into the engine (`GameController.cpp:611-730`).
fn update_stickman_controls(
    ctx: &egui::Context,
    state: &mut AppState,
    free: &impl Fn(Key) -> bool,
) {
    let mut control = state.engine.bufs.stickman_control;
    let before = control;
    let placing_save = state.placement.is_some();
    ctx.input(|input| {
        for event in &input.events {
            let egui::Event::Key {
                key,
                pressed,
                repeat,
                ..
            } = event
            else {
                continue;
            };
            // `GameController.cpp:614-615, 697-698`: auto-repeat is dropped.
            // A Lua `KEYPRESS` handler can suppress a press; a release must
            // always clear its bit or the rig keeps walking forever.
            if *repeat || (*pressed && !free(*key)) {
                continue;
            }
            apply_stickman_key(&mut control, *key, *pressed, placing_save);
        }
    });
    if control == before {
        return;
    }
    for player in 0..2u8 {
        state.engine.apply_command(SimCommand::SetStickmanControl {
            player,
            comm: control[player as usize * 2],
            pcomm: control[player as usize * 2 + 1],
        });
    }
}

/// Treats every control key as released when a text field, a modal, view
/// mode or the 3D sandbox takes the keyboard, so a rig cannot keep walking
/// after its key-up event was swallowed. `pcomm` survives like the
/// reference's player struct.
fn clear_stickman_controls(state: &mut AppState) {
    let control = state.engine.bufs.stickman_control;
    if control[0] == 0 && control[2] == 0 {
        return;
    }
    for player in 0..2u8 {
        state.engine.apply_command(SimCommand::SetStickmanControl {
            player,
            comm: 0,
            pcomm: control[player as usize * 2 + 1],
        });
    }
}

/// Reloads the simulation from the save file it was loaded from
/// (`GameView.cpp:1455-1466`, `GameController::ReloadSim`).
///
/// The port tracks the current save by name in [`AppState::save_name_input`]
/// (set by both `save_current` and `load_from_file`). A simulation that has
/// never been saved has no file to reload, so the HUD reports that instead of
/// touching the canvas.
fn reload_simulation(state: &mut AppState) {
    let stem = state.save_name_input.trim();
    if stem.is_empty() {
        state.set_status(t("app.status_reload_no_save"), true);
        return;
    }
    let path = Path::new(SAVES_DIR).join(format!("{stem}.{SAVE_EXTENSION}"));
    if !path.is_file() {
        state.set_status(t("app.status_reload_no_save"), true);
        return;
    }
    if let Err(err) = state.load_from_file(&path) {
        state.set_status(err, true);
    }
}

/// `GameView.cpp:1460` / `GameController::SetEdgeMode`: advances the live edge
/// mode and restamps the border through [`SimulationEngine::set_edge_mode`].
fn cycle_edge_mode(state: &mut AppState) {
    let next = match state.engine.bufs.config.edge_mode {
        EdgeMode::Void => EdgeMode::Solid,
        EdgeMode::Solid => EdgeMode::Loop,
        EdgeMode::Loop => EdgeMode::Void,
    };
    state.engine.set_edge_mode(next);
    // Keep the Options window's persisted mirror in step with the live config.
    state.prefs.edge_mode = next as u32;
    state.new_canvas.edge_mode = next;
    let label = match next {
        EdgeMode::Void => t("options.edge_mode_void"),
        EdgeMode::Solid => t("options.edge_mode_solid"),
        EdgeMode::Loop => t("options.edge_mode_loop"),
    };
    state.set_status(t_fmt("app.status_edge_mode", &[label]), false);
}

/// `GameView.cpp:1522-1530` / `GameController::SwitchAir`: advances the air
/// mode through the five `AIR_*` states.
fn cycle_air_mode(state: &mut AppState) {
    let next = (state.engine.bufs.config.air_mode as u32 + 1) % 5;
    state.engine.bufs.config.air_mode = match next {
        1 => AirMode::PressureOff,
        2 => AirMode::VelocityOff,
        3 => AirMode::Off,
        4 => AirMode::NoUpdate,
        _ => AirMode::On,
    };
    state.prefs.air_mode = next;
    let label = match next {
        1 => t("options.air_mode_pressure_off"),
        2 => t("options.air_mode_velocity_off"),
        3 => t("options.air_mode_off"),
        4 => t("options.air_mode_no_update"),
        _ => t("options.air_mode_on"),
    };
    state.set_status(t_fmt("app.status_air_mode", &[label]), false);
}

/// `GameView.cpp:1481-1491` / `GameController::AdjustGridSize`: cycles the
/// element-grid size through `0..=9`; `0` disables the overlay.
fn adjust_grid_size(state: &mut AppState, delta: i32) {
    let next = (state.engine.bufs.config.grid_size + delta).rem_euclid(10);
    state.engine.bufs.config.grid_size = next;
    state.engine.bufs.config.grid = next != 0;
    let size = next.to_string();
    state.set_status(t_fmt("app.status_grid_size", &[&size]), false);
}

/// `REPLACE_MODE` bit of [`tpt_sim::SimulationEngine::replace_mode_flags`]
/// (`SimulationData.h:124`).
const REPLACE_MODE: u32 = 0x0000_0001;
/// `SPECIFIC_DELETE` bit of the same word (`SimulationData.h:125`): delete
/// matching particles instead of refusing to overwrite them.
const SPECIFIC_DELETE: u32 = 0x0000_0002;

/// `GameController::SetReplaceModeFlags` (`GameController.cpp:1195-1213`):
/// toggles `REPLACE_MODE` or `SPECIFIC_DELETE` and keeps the two flags
/// mutually exclusive. The UI checkbox mirrors bit 0; the engine keeps the
/// combined word Lua's `sim.replaceModeFlags` reads.
fn toggle_replace_flags(state: &mut AppState, specific: bool) {
    let old = (state.replace_mode as u32 & REPLACE_MODE)
        | (state.engine.replace_mode_flags & SPECIFIC_DELETE);
    let mut flags = if specific {
        old ^ SPECIFIC_DELETE
    } else {
        old ^ REPLACE_MODE
    };
    if old & REPLACE_MODE == 0 && flags & REPLACE_MODE != 0 {
        flags &= !SPECIFIC_DELETE;
    }
    if old & SPECIFIC_DELETE == 0 && flags & SPECIFIC_DELETE != 0 {
        flags &= !REPLACE_MODE;
    }
    if flags & (REPLACE_MODE | SPECIFIC_DELETE) == REPLACE_MODE | SPECIFIC_DELETE {
        flags &= !SPECIFIC_DELETE;
    }
    state.engine.replace_mode_flags = flags;
    state.replace_mode = flags & REPLACE_MODE != 0;
    let bit = if specific {
        SPECIFIC_DELETE
    } else {
        REPLACE_MODE
    };
    let enabled = flags & bit != 0;
    let key = match (specific, enabled) {
        (false, true) => "app.status_replace_on",
        (false, false) => "app.status_replace_off",
        (true, true) => "app.status_specific_delete_on",
        (true, false) => "app.status_specific_delete_off",
    };
    state.set_status(t(key), false);
}

/// egui memory key of the armed `Ctrl+F` finding filter.
const FINDING_KEY: &str = "tpt_finding_element";

/// What `Ctrl+F` highlights, mirroring `GameView::FindingElementCandidate`
/// (`GameView.cpp:2708-2725`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum FindingFilter {
    /// The element carried by the primary tool.
    Element(u32),
    /// The field and value configured in the property tool.
    Property(PropertyKind, PropertyValue),
}

/// Selects the filter `Ctrl+F` would arm for the current primary tool.
fn finding_candidate(state: &AppState) -> Option<FindingFilter> {
    match state.primary_tool {
        SelectedTool::Element(ty) if ty != PT_NONE => Some(FindingFilter::Element(ty)),
        SelectedTool::Property => {
            let value = state.property_window.value?;
            Some(FindingFilter::Property(state.property_window.field, value))
        }
        _ => None,
    }
}

/// `GameView.cpp:1464-1476`: Ctrl+F arms the highlight for the primary tool's
/// element / property, and pressing it again with the same candidate disarms
/// it.
fn toggle_finding(ctx: &egui::Context, state: &mut AppState) {
    let candidate = finding_candidate(state);
    let current: Option<FindingFilter> = ctx
        .data(|data| data.get_temp::<Option<FindingFilter>>(Id::unique(FINDING_KEY)))
        .flatten();
    let next = if current == candidate {
        None
    } else {
        candidate
    };
    ctx.data_mut(|data| data.insert_temp(Id::unique(FINDING_KEY), next));
    let message = match next {
        Some(FindingFilter::Element(ty)) => {
            t_fmt("app.status_finding_on", &[&element_name(state, ty)])
        }
        Some(FindingFilter::Property(kind, _)) => t_fmt("app.status_finding_on", &[kind.display()]),
        None => t("app.status_finding_off").to_string(),
    };
    state.set_status(message, false);
}

/// Tints every particle matching the armed filter over the live viewport.
///
/// `Renderer::RenderSimulation` (`Renderer.cpp:460-620`) paints matches bright
/// and dims everything else; the port keeps the GPU image untouched and adds a
/// translucent yellow overlay on the matching pixels. The viewport renders
/// from device buffers, so the host shadow is refreshed first (only while the
/// highlight is armed).
pub(super) fn draw_finding_highlight(ctx: &egui::Context, state: &mut AppState, view_rect: Rect) {
    if state.mode != GameMode::Powder2D {
        return;
    }
    let Some(filter) = ctx
        .data(|data| data.get_temp::<Option<FindingFilter>>(Id::unique(FINDING_KEY)))
        .flatten()
    else {
        return;
    };
    let (xres, yres) = state.canvas_size();
    if xres == 0 || yres == 0 {
        return;
    }
    // Only the used prefix changes per tick; a paused frame or a frame after
    // an explicit read-back already has a current host shadow.
    if state.engine.bufs.gpu_particles_dirty {
        state.engine.bufs.sync_particles_gpu_to_host();
    }

    let fit = (view_rect.width() / xres as f32).min(view_rect.height() / yres as f32);
    let scale = state.prefs.viewport_blit_scale(fit) * state.zoom.clamp(0.10, 24.0);
    let pan: Vec2 = ctx
        .data(|data| data.get_temp::<Vec2>(viewport::pan_key()))
        .unwrap_or(Vec2::ZERO);
    let img_rect = Rect::from_center_size(
        view_rect.center() + pan,
        vec2(xres as f32 * scale, yres as f32 * scale),
    );
    let painter = ctx
        .layer_painter(LayerId::new(
            Order::Middle,
            Id::unique("tpt_finding_highlight"),
        ))
        .with_clip_rect(view_rect);
    let bufs = &state.engine.bufs;
    let slots = bufs
        .part_ty
        .len()
        .min(bufs.part_x.len())
        .min(bufs.part_y.len());
    let mut mesh = egui::Mesh::default();
    let tint = Color32::from_rgba_unmultiplied(255, 255, 64, 120);
    for slot in 0..slots {
        if bufs.part_ty[slot] == PT_NONE {
            continue;
        }
        let matches = match filter {
            FindingFilter::Element(ty) => bufs.part_ty[slot] == ty,
            FindingFilter::Property(kind, value) => match kind {
                PropertyKind::Type => bufs.part_ty[slot] as i64 == value.as_i64(),
                PropertyKind::Life => bufs.part_life[slot] as i64 == value.as_i64(),
                PropertyKind::Ctype => bufs.part_ctype[slot] as i64 == value.as_i64(),
                PropertyKind::Vx => bufs.part_vx[slot] == value.as_f32(),
                PropertyKind::Vy => bufs.part_vy[slot] == value.as_f32(),
                PropertyKind::Temp => bufs.part_temp[slot] == value.as_f32(),
                PropertyKind::Tmp => bufs.part_tmp[slot] as i64 == value.as_i64(),
                PropertyKind::Tmp2 => bufs.part_tmp2[slot] as i64 == value.as_i64(),
                PropertyKind::Tmp3 => bufs.part_tmp3[slot] as i64 == value.as_i64(),
                PropertyKind::Tmp4 => bufs.part_tmp4[slot] as i64 == value.as_i64(),
                PropertyKind::Dcolour => bufs.part_dcolour[slot] as i64 == value.as_i64(),
            },
        };
        if !matches {
            continue;
        }
        let (x, y) = (bufs.part_x[slot], bufs.part_y[slot]);
        if x < 0.0 || y < 0.0 || x >= xres as f32 || y >= yres as f32 {
            continue;
        }
        let rect = Rect::from_min_size(
            img_rect.min + vec2(x * scale, y * scale),
            Vec2::splat(scale.max(1.0)),
        );
        mesh.add_colored_rect(rect, tint);
    }
    if !mesh.is_empty() {
        painter.add(egui::Shape::mesh(mesh));
    }
}

/// Forwards this frame's key presses to Lua's `evt.KEYPRESS` handlers.
///
/// Returns the keys whose default shortcut a handler claimed.
fn fire_lua_key_presses(ctx: &egui::Context, state: &mut AppState) -> Vec<Key> {
    let pressed: Vec<Key> = ctx.input(|input| {
        input
            .events
            .iter()
            .filter_map(|event| match event {
                egui::Event::Key {
                    key, pressed: true, ..
                } => Some(*key),
                _ => None,
            })
            .collect()
    });
    pressed
        .into_iter()
        .filter(|key| state.lua.fire_key_press(key.name(), &mut state.engine))
        .collect()
}
