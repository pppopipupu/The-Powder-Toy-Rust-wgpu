//! Mouse-button stroke state machine for the canvas.

use tpt_core::constants::PT_LIGH;

use crate::app_state::{AppState, SelectedTool};
use crate::brushes::{apply_tool_stroke, snap_line_coords};

use super::{PointerFrame, StrokeState, stroke_key};

/// `true` when the held simulation tool must be dropped.
///
/// [`handle_paint_button`] runs once per button every frame. A one-button
/// stroke leaves the other button up, so a plain "button is up" test would clear
/// the stroke on every frame and turn a held `HEAT`/`AIR` into a single stamp.
/// The held tool therefore records which button owns it and only that button can
/// end it.
fn held_tool_ends(
    down: bool,
    pointer_in_view: bool,
    held: Option<(bool, (i32, i32))>,
    is_secondary: bool,
) -> bool {
    (!down || !pointer_in_view)
        && held.is_some_and(|(held_secondary, _)| held_secondary == is_secondary)
}

/// `true` when a held free-hand stroke needs another dispatch this frame.
///
/// Most tools only paint when the cursor moved. `LIGH` is the exception:
/// `GameView::OnTick` (`GameView.cpp:1739-1756`) calls `DrawPoints` every
/// frame the button is held even at a fixed position, and
/// `Element_LIGH_Tool::DrawLine` (`ElementTool.cpp:20-24`) creates a bolt on
/// each call; `Editing.cpp:782-793` throttles it to `power / 4` ticks. A
/// stationary hold therefore has to keep dispatching to keep striking.
fn hold_dispatches(previous: (i32, i32), position: (i32, i32), tool: SelectedTool) -> bool {
    previous != position || tool == SelectedTool::Element(PT_LIGH)
}

/// Applies one stroke with the selected tool.
///
/// The property tool has its own dispatch ([`AppState::apply_property_stroke`])
/// because the window can select float properties (`temp`, `vx`, `vy`, `type`,
/// `dcolour`) that have no `ParticleField` variant; every other tool goes
/// through [`apply_tool_stroke`].
fn dispatch_stroke(
    state: &mut AppState,
    p0: (i32, i32),
    p1: (i32, i32),
    is_secondary: bool,
    is_line: bool,
    is_rect: bool,
    is_fill: bool,
) {
    if state.selected_tool(is_secondary) == SelectedTool::Property {
        if is_fill {
            let _ = state.apply_property_fill(p0);
        } else {
            state.apply_property_stroke(p0, p1, is_line, is_rect);
        }
    } else {
        apply_tool_stroke(state, p0, p1, is_secondary, is_line, is_rect, is_fill);
    }
}

/// Runs one mouse button's stroke state machine.
///
/// * press  — stamp immediately (or arm a line/rect/fill gesture)
/// * drag   — interpolate from the previous sample for free-hand strokes
/// * release— commit armed line/rect gestures
///
/// Rigid-body tools spawn a single body on press and ignore the drag, matching
/// [`apply_tool_stroke`](crate::brushes::apply_tool_stroke).
pub(super) fn handle_paint_button(
    ctx: &egui::Context,
    state: &mut AppState,
    frame: &PointerFrame,
    is_secondary: bool,
) {
    let key = stroke_key(is_secondary);
    let mut stroke = ctx
        .data_mut(|data| data.get_temp::<StrokeState>(key))
        .unwrap_or_default();

    let pressed = if is_secondary {
        frame.secondary_pressed
    } else {
        frame.primary_pressed
    };
    let down = if is_secondary {
        frame.secondary_down
    } else {
        frame.primary_down
    };
    let released = if is_secondary {
        frame.secondary_released
    } else {
        frame.primary_released
    };

    if held_tool_ends(
        down,
        frame.pointer_in_view,
        state.held_sim_tool,
        is_secondary,
    ) {
        // A release or a drag off the viewport never reaches the branches
        // below, so a held tool is cleared whenever the button that owns it is
        // up or the pointer left the canvas (`GameView::OnBlur` ends the stroke
        // too).
        state.held_sim_tool = None;
    }

    let is_rigid = matches!(
        state.selected_tool(is_secondary),
        SelectedTool::RigidBody { .. }
    );
    if pressed && frame.pointer_in_view {
        if let Some(position) = frame.sim_pos {
            // Lua sees the click before the brush. TPT's button codes are
            // 1 = left and 2 = right; a truthy handler suppresses the default
            // paint so a script can implement its own tool.
            let button = if is_secondary { 2 } else { 1 };
            if state
                .lua
                .fire_mouse_down(position.0, position.1, button, &mut state.engine)
            {
                return;
            }

            // `SampleTool`'s `shiftBehaviour`: shift-click adopts the property
            // tool and opens its window seeded from the particle under the
            // cursor instead of sampling the element. Empty pixels do nothing.
            if frame.mods.shift && state.selected_tool(is_secondary) == SelectedTool::Sample {
                state.sample_property_window(position.0, position.1);
                return;
            }

            let is_line = frame.mods.shift && !(frame.mods.ctrl || frame.mods.command);
            let is_rect = (frame.mods.ctrl || frame.mods.command) && !frame.mods.shift;
            let is_fill = frame.mods.alt && !is_line && !is_rect;

            stroke = StrokeState {
                start: position,
                is_line,
                is_rect,
                is_fill,
                active: true,
            };
            state.drag_start = Some(position);
            state.last_cursor_sim = Some(position);
            // A held simulation tool keeps acting on every tick (see
            // `AppState::held_sim_tool`); a line/rect gesture only runs on
            // release.
            state.held_sim_tool = if is_line || is_rect {
                None
            } else {
                matches!(state.selected_tool(is_secondary), SelectedTool::SimTool(_))
                    .then_some((is_secondary, position))
            };

            // One undo snapshot per stroke, captured before the first mutation
            // so Ctrl+Z restores the canvas to its pre-stroke state.
            state.push_undo_snapshot();

            if is_fill {
                // Flood fill is a single-shot action seeded at the press point.
                dispatch_stroke(state, position, position, is_secondary, false, false, true);
                stroke.active = false;
                state.drag_start = None;
                state.last_cursor_sim = None;
            } else if is_rigid {
                // One rigid body per press; the drag would spawn a body per frame.
                dispatch_stroke(state, position, position, is_secondary, false, false, false);
                stroke.active = false;
                state.drag_start = None;
                state.last_cursor_sim = None;
            } else if !is_line && !is_rect {
                dispatch_stroke(state, position, position, is_secondary, false, false, false);
            }
            if !stroke.active {
                // A fill/rigid press is a single shot, never a held tool.
                state.held_sim_tool = None;
            }

            ctx.data_mut(|data| data.insert_temp(key, stroke));
        }
        return;
    }
    if down && stroke.active {
        if !(stroke.is_line || stroke.is_rect || stroke.is_fill)
            && let Some(position) = frame.sim_pos
        {
            let previous = state.last_cursor_sim.unwrap_or(stroke.start);
            if hold_dispatches(previous, position, state.selected_tool(is_secondary)) {
                dispatch_stroke(state, previous, position, is_secondary, false, false, false);
                state.last_cursor_sim = Some(position);
                state.drag_start = Some(stroke.start);
            }
            if let Some(held) = state.held_sim_tool.as_mut() {
                held.1 = position;
            }
        }
        ctx.data_mut(|data| data.insert_temp(key, stroke));
        return;
    }
    if released && stroke.active {
        state.held_sim_tool = None;
        let end = frame
            .sim_pos
            .or(state.last_cursor_sim)
            .unwrap_or(stroke.start);

        if stroke.is_line || stroke.is_rect {
            let mut end = end;
            if stroke.is_line && frame.mods.alt {
                let (sx, sy) = snap_line_coords(stroke.start.0, stroke.start.1, end.0, end.1);
                end = (sx, sy);
            }
            dispatch_stroke(
                state,
                stroke.start,
                end,
                is_secondary,
                stroke.is_line,
                stroke.is_rect,
                false,
            );
        } else if frame.sim_pos.is_some() {
            // Flush the last free-hand segment: the press already stamped, so
            // this only paints when the cursor moved since the previous sample.
            let previous = state.last_cursor_sim.unwrap_or(stroke.start);
            if previous != end {
                dispatch_stroke(state, previous, end, is_secondary, false, false, false);
            }
        }

        stroke.active = false;
        state.drag_start = None;
        state.last_cursor_sim = None;
        ctx.data_mut(|data| data.insert_temp(key, stroke));
    }
}

#[cfg(test)]
mod tests {
    use super::{hold_dispatches, held_tool_ends};
    use tpt_core::constants::{PT_DUST, PT_LIGH};

    use crate::app_state::SelectedTool;

    /// A primary hold must survive the secondary button's pass, which sees its
    /// own button up on every frame of a one-button stroke.
    #[test]
    fn held_tool_is_only_ended_by_the_button_that_owns_it() {
        let held_primary = Some((false, (4, 5)));
        assert!(!held_tool_ends(true, true, held_primary, false));
        assert!(!held_tool_ends(false, true, held_primary, true));
        assert!(held_tool_ends(false, true, held_primary, false));

        let held_secondary = Some((true, (4, 5)));
        assert!(!held_tool_ends(true, true, held_secondary, true));
        assert!(!held_tool_ends(false, true, held_secondary, false));
        assert!(held_tool_ends(false, true, held_secondary, true));
    }

    /// `LIGH` keeps striking while the button is held at a fixed position;
    /// other tools only paint when the cursor moves.
    #[test]
    fn lightning_repeats_a_stationary_hold() {
        let here = (12, 34);
        assert!(hold_dispatches(here, here, SelectedTool::Element(PT_LIGH)));
        assert!(!hold_dispatches(here, here, SelectedTool::Element(PT_DUST)));
        assert!(!hold_dispatches(here, here, SelectedTool::Eraser));
        assert!(hold_dispatches((12, 33), here, SelectedTool::Element(PT_DUST)));
    }

    /// Leaving the canvas ends the stroke, and an idle tool is never touched.
    #[test]
    fn leaving_the_viewport_ends_the_owned_tool() {
        assert!(held_tool_ends(true, false, Some((false, (1, 1))), false));
        assert!(!held_tool_ends(true, false, Some((true, (1, 1))), false));
        assert!(!held_tool_ends(false, false, None, false));
    }
}
