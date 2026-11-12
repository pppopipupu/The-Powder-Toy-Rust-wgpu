//! Simulation viewport: image, zoom/pan, canvas input and preview overlays.

mod paint;
mod preview;
mod signs;

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, Id, Key, PointerButton, Pos2, Rect, Sense,
    Stroke, StrokeKind, TextureId, Ui, Vec2, pos2, vec2,
};

use tpt_core::constants::WL_ERASE;

use crate::app_state::{AppState, SelectedTool};
use crate::brushes::{adjust_brush_size, snap_line_coords};
use crate::i18n::{t, t_fmt};

use super::shared::{BG_VIEWPORT, TEXT_DIM};
use paint::handle_paint_button;
use preview::{draw_brush_outline, draw_line_preview, draw_rect_preview, draw_rigid_preview};
use signs::{box_hit, draw_signs, handle_special_sign_click, press_pending};

/// Lowest zoom factor the viewport accepts.
const MIN_ZOOM: f32 = 0.10;
/// Highest zoom factor the viewport accepts.
const MAX_ZOOM: f32 = 24.0;

/// Pan speed of the navigation mode, as a fraction of the view size per second.
const PAN_SPEED: f32 = 1.2;

/// Wheel zoom exponent for Ctrl+wheel over the canvas.
const WHEEL_ZOOM: f32 = 0.0025;
/// Wheel zoom exponent for the free navigation mode, which is the only way to
/// zoom while WASD is panning, so it needs a coarser step than Ctrl+wheel.
const MOVE_VIEW_WHEEL_ZOOM: f32 = 0.012;

/// Pan for the navigation keys, in image-translation space.
///
/// `pan` translates the image, not the camera, so the sign is the opposite of
/// the key that drives it: pressing W looks further up, which drags the image
/// down. Every axis is inverted for the same reason.
fn navigation_pan(up: bool, down: bool, left: bool, right: bool) -> Vec2 {
    let mut dir = Vec2::ZERO;
    if up {
        dir.y += 1.0;
    }
    if down {
        dir.y -= 1.0;
    }
    if left {
        dir.x += 1.0;
    }
    if right {
        dir.x -= 1.0;
    }
    dir
}

/// Maps between simulation pixels and screen coordinates of the canvas image.
#[derive(Clone, Copy)]
struct ViewTransform {
    /// Screen rect covered by the simulation texture.
    img_rect: Rect,
    /// Screen pixels per simulation pixel.
    scale: f32,
}

impl ViewTransform {
    /// Converts a simulation coordinate to the top-left corner of its pixel.
    fn to_screen(self, x: f32, y: f32) -> Pos2 {
        self.img_rect.min + vec2(x * self.scale, y * self.scale)
    }

    /// Screen rect covered by simulation pixel `(x, y)`.
    fn pixel_rect(self, x: i32, y: i32) -> Rect {
        Rect::from_min_size(
            self.to_screen(x as f32, y as f32),
            Vec2::splat(self.scale.max(1.0)),
        )
    }

    /// Converts a screen position to a simulation pixel, clamped to the canvas.
    fn to_sim(self, position: Pos2, xres: u32, yres: u32) -> Option<(i32, i32)> {
        if !self.img_rect.contains(position)
            || self.img_rect.width() <= 0.0
            || self.img_rect.height() <= 0.0
        {
            return None;
        }
        let fx = (position.x - self.img_rect.min.x) / self.img_rect.width();
        let fy = (position.y - self.img_rect.min.y) / self.img_rect.height();
        let x = (fx * xres as f32).floor() as i32;
        let y = (fy * yres as f32).floor() as i32;
        Some((x.clamp(0, xres as i32 - 1), y.clamp(0, yres as i32 - 1)))
    }
}

/// Raw pointer state sampled once per frame and shared by both mouse buttons.
#[derive(Clone, Copy)]
struct PointerFrame {
    sim_pos: Option<(i32, i32)>,
    pointer_in_view: bool,
    primary_pressed: bool,
    primary_down: bool,
    primary_released: bool,
    secondary_pressed: bool,
    secondary_down: bool,
    secondary_released: bool,
    mods: egui::Modifiers,
}

/// Per-button stroke bookkeeping kept in egui's temporary memory so the
/// gesture survives modifier changes between press and release.
#[derive(Clone, Copy, Debug, Default)]
struct StrokeState {
    start: (i32, i32),
    is_line: bool,
    is_rect: bool,
    is_fill: bool,
    active: bool,
}

/// egui memory key holding the viewport pan offset.
pub(super) fn pan_key() -> Id {
    Id::unique("tpt_viewport_pan")
}

/// egui memory key holding one mouse button's stroke state.
fn stroke_key(is_secondary: bool) -> Id {
    Id::unique(("tpt_stroke_state", is_secondary))
}

/// Raw wheel delta for this frame, summed over the wheel events.
///
/// Unlike [`egui::InputState::smooth_scroll_delta`] this is not smoothed over
/// several frames, so one wheel notch produces exactly one discrete step.
fn wheel_delta(ctx: &egui::Context) -> f32 {
    ctx.input(|input| {
        input
            .events
            .iter()
            .filter_map(|event| match event {
                egui::Event::MouseWheel { delta, .. } => Some(delta.y),
                _ => None,
            })
            .sum()
    })
}

/// Draws the simulation texture, the overlays and handles all canvas input.
pub(super) fn draw_viewport(ui: &mut Ui, state: &mut AppState, sim_tex_id: TextureId) {
    let available = ui.available_size();
    if available.x < 4.0 || available.y < 4.0 {
        return;
    }
    let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
    let view_rect = response.rect;
    painter.rect_filled(view_rect, CornerRadius::ZERO, BG_VIEWPORT);

    let (xres, yres) = state.canvas_size();
    if xres == 0 || yres == 0 {
        return;
    }
    let mut pan: Vec2 = ui
        .ctx()
        .data_mut(|data| data.get_temp::<Vec2>(pan_key()))
        .unwrap_or(Vec2::ZERO);

    let fit = (view_rect.width() / xres as f32).min(view_rect.height() / yres as f32);
    // `OptionsView.cpp:340-362`: the saved window scale multiplies the fit
    // factor, so a small canvas still fills the viewport.
    let base_scale = state.prefs.viewport_blit_scale(fit);

    let pointer = response
        .interact_pointer_pos()
        .or_else(|| response.hover_pos());
    let mods = ui.ctx().input(|input| input.modifiers);

    if response.hovered() {
        let wheel = wheel_delta(ui.ctx());
        if wheel.abs() > f32::EPSILON {
            let step = if wheel > 0.0 { 1 } else { -1 };
            // `GameView::OnMouseWheel`: the wheel resizes the zoom scope while
            // it is open and the brush otherwise. The view zoom on Ctrl+wheel
            // is the port's own addition, since it has a scrolling viewport.
            if state.engine.bufs.config.zoom_enabled && !state.zoom_cursor_fixed {
                state.adjust_zoom_size(step, false);
            } else if mods.ctrl || mods.command || state.move_view {
                // Zoom about the cursor so the pixel under it stays put.
                let old_scale = base_scale * state.zoom.clamp(MIN_ZOOM, MAX_ZOOM);
                let step = if state.move_view {
                    MOVE_VIEW_WHEEL_ZOOM
                } else {
                    WHEEL_ZOOM
                };
                let new_zoom = (state.zoom * (wheel * step).exp()).clamp(MIN_ZOOM, MAX_ZOOM);
                let new_scale = base_scale * new_zoom;
                if let Some(cursor) = pointer
                    && old_scale > f32::EPSILON
                {
                    let centre = view_rect.center() + pan;
                    let offset = (cursor - centre) / old_scale;
                    pan = (cursor - offset * new_scale) - view_rect.center();
                }
                state.zoom = new_zoom;
            } else {
                // One wheel notch = one radius step.
                adjust_brush_size(&mut state.brush_rx, &mut state.brush_ry, step, false, false);
            }
        }
    }

    if response.dragged_by(PointerButton::Middle) {
        pan += response.drag_delta();
    }

    if state.move_view {
        // Held keys pan at a fixed fraction of the view per second, so the
        // speed does not depend on the zoom factor or on the frame rate. Text
        // fields keep their keyboard input.
        if !ui.ctx().egui_wants_keyboard_input() {
            let (dir, reset) = ui.ctx().input(|input| {
                (
                    navigation_pan(
                        input.key_down(Key::W) || input.key_down(Key::ArrowUp),
                        input.key_down(Key::S) || input.key_down(Key::ArrowDown),
                        input.key_down(Key::A) || input.key_down(Key::ArrowLeft),
                        input.key_down(Key::D) || input.key_down(Key::ArrowRight),
                    ),
                    super::hotkeys::pressed_once(input, Key::Home),
                )
            });
            if dir != Vec2::ZERO {
                let dt = ui.ctx().input(|input| input.stable_dt).min(0.1);
                pan += dir.normalized() * view_rect.size().max_elem() * PAN_SPEED * dt;
            }
            if reset {
                state.zoom = 1.0;
                pan = Vec2::ZERO;
            }
        }
    }

    let scale = base_scale * state.zoom.clamp(MIN_ZOOM, MAX_ZOOM);
    let img_size = vec2(xres as f32 * scale, yres as f32 * scale);

    // Keep at least a sliver of the image on screen.
    let limit_x = (view_rect.width() * 0.5 + img_size.x * 0.5 - 32.0).max(0.0);
    let limit_y = (view_rect.height() * 0.5 + img_size.y * 0.5 - 32.0).max(0.0);
    pan.x = pan.x.clamp(-limit_x, limit_x);
    pan.y = pan.y.clamp(-limit_y, limit_y);
    ui.ctx().data_mut(|data| data.insert_temp(pan_key(), pan));

    let view = ViewTransform {
        img_rect: Rect::from_center_size(view_rect.center() + pan, img_size),
        scale,
    };
    painter.image(
        sim_tex_id,
        view.img_rect,
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
        Color32::WHITE,
    );
    painter.rect_stroke(
        view.img_rect,
        CornerRadius::ZERO,
        Stroke::new(1.0, Color32::from_rgb(70, 76, 88)),
        StrokeKind::Inside,
    );
    if state.engine.bufs.config.grid {
        let gs4 = (state.engine.bufs.config.grid_size.max(1) * 4) as f32;
        if gs4 > 0.0 {
            if state.engine.bufs.config.grid_checkerboard {
                let cols = (xres as f32 / gs4).ceil() as i32;
                let rows = (yres as f32 / gs4).ceil() as i32;
                let checker_color = Color32::from_rgba_unmultiplied(100, 100, 100, 80);
                let mut mesh = egui::Mesh::default();
                for r in 0..rows {
                    for c in 0..cols {
                        if (c + r) % 2 != 0 {
                            let min = view.to_screen(c as f32 * gs4, r as f32 * gs4);
                            let max = view.to_screen(
                                ((c + 1) as f32 * gs4).min(xres as f32),
                                ((r + 1) as f32 * gs4).min(yres as f32),
                            );
                            mesh.add_colored_rect(Rect::from_min_max(min, max), checker_color);
                        }
                    }
                }
                painter.add(egui::Shape::mesh(mesh));
            } else {
                let grid_color = Color32::from_rgba_unmultiplied(100, 100, 100, 80);
                let stroke = Stroke::new(1.0, grid_color);
                let mut x = 0.0;
                while x <= xres as f32 {
                    painter.line_segment(
                        [view.to_screen(x, 0.0), view.to_screen(x, yres as f32)],
                        stroke,
                    );
                    x += gs4;
                }
                let mut y = 0.0;
                while y <= yres as f32 {
                    painter.line_segment(
                        [view.to_screen(0.0, y), view.to_screen(xres as f32, y)],
                        stroke,
                    );
                    y += gs4;
                }
            }
        }
    }

    // `GameView::OnTick`: while the scope is open and unpinned it follows the
    // raw cursor, not the pixel under a magnification window.
    if state.engine.bufs.config.zoom_enabled
        && !state.zoom_cursor_fixed
        && let Some((zx, zy)) = pointer.and_then(|position| view.to_sim(position, xres, yres))
    {
        state.set_zoom_position(zx, zy);
    }

    // The magnification window, shared by the pointer remap below and by the
    // renderer at the end of the function.
    let zoom_win_rect = {
        let cfg = &state.engine.bufs.config;
        if cfg.zoom_enabled {
            let win_min = view.to_screen(cfg.zoom_window_x as f32, cfg.zoom_window_y as f32);
            let (win_size, cell) = zoom_window_metrics(
                cfg.zoom_scope_size,
                cfg.zoom_factor,
                view.scale,
                view_rect.width().min(view_rect.height()),
            );
            Some((Rect::from_min_size(win_min, Vec2::splat(win_size)), cell))
        } else {
            None
        }
    };

    let sim_pos = pointer.and_then(|position| {
        let cfg = &state.engine.bufs.config;
        if let Some((win_rect, cell)) = zoom_win_rect
            && let Some((sx, sy)) =
                zoom_source_cell(position, win_rect, cell, cfg.zoom_scope_x, cfg.zoom_scope_y)
        {
            return Some((sx.clamp(0, xres as i32 - 1), sy.clamp(0, yres as i32 - 1)));
        }
        view.to_sim(position, xres, yres)
    });
    state.hover_pos = sim_pos;
    // `GameView::OnMouseDown` skips every tool branch while the scope is open
    // and unpinned, so the click that pins it must not also paint. Sampling the
    // flag before the pin below keeps that frame's stroke swallowed.
    let zoom_was_unfixed = state.engine.bufs.config.zoom_enabled && !state.zoom_cursor_fixed;
    // `GameView::OnMouseUp`: any button release while the scope is open pins
    // it (`zoomCursorFixed = true`) and swallows the stroke, so the window
    // stops following the cursor and the tools reach the magnified cells from
    // the next click on. Releasing Z then leaves the window in place; tapping Z
    // again unfixes it and the next release closes it.
    if zoom_win_rect.is_some()
        && !state.zoom_cursor_fixed
        && ui
            .ctx()
            .input(|input| input.pointer.primary_released() || input.pointer.secondary_released())
    {
        state.zoom_cursor_fixed = true;
    }
    // `Simulation::mousePos`: the portal debug lines and the sign hit test read
    // the pointer pixel from the simulation state.
    let (mx, my) = sim_pos.unwrap_or((-1, -1));
    state.engine.bufs.config.mouse_x = mx;
    state.engine.bufs.config.mouse_y = my;

    if response.hovered() {
        ui.ctx().set_cursor_icon(if state.move_view {
            if response.dragged() {
                CursorIcon::Grabbing
            } else {
                CursorIcon::Grab
            }
        } else {
            CursorIcon::Crosshair
        });
    }
    let frame = PointerFrame {
        sim_pos,
        pointer_in_view: response.contains_pointer(),
        primary_pressed: ui.ctx().input(|input| input.pointer.primary_pressed()),
        primary_down: ui.ctx().input(|input| input.pointer.primary_down()),
        primary_released: ui.ctx().input(|input| input.pointer.primary_released()),
        secondary_pressed: ui.ctx().input(|input| input.pointer.secondary_pressed()),
        secondary_down: ui.ctx().input(|input| input.pointer.secondary_down()),
        secondary_released: ui.ctx().input(|input| input.pointer.secondary_released()),
        mods,
    };
    // Region clipboard (`GameView.cpp:1234-1266`): while a copy/cut is armed
    // the drag draws the rubber band instead of a stroke, and the release
    // captures the rectangle. The right button cancels.
    if state.select_mode != crate::app_state::selection::SelectMode::None {
        let (pressed, down, released, secondary) = ui.ctx().input(|input| {
            (
                input.pointer.primary_pressed(),
                input.pointer.primary_down(),
                input.pointer.primary_released(),
                input.pointer.secondary_pressed(),
            )
        });
        if let Some(pos) = sim_pos {
            if pressed {
                state.select_start = Some(pos);
                state.select_end = Some(pos);
            } else if down && state.select_start.is_some() {
                state.select_end = Some(pos);
            }
        }
        if released && state.select_start.is_some() {
            let mode = state.select_mode;
            let stored = state.finish_selection();
            // `OptionsView.cpp:413-415`: with the platform clipboard enabled the
            // captured region is mirrored to the system clipboard as the same
            // serialized payload, so another instance can paste it.
            if stored
                && state.prefs.native_clipboard
                && matches!(
                    mode,
                    crate::app_state::selection::SelectMode::Copy
                        | crate::app_state::selection::SelectMode::Cut
                )
                && let Some(region) = &state.clipboard
            {
                ui.ctx().copy_text(region.to_clipboard_text());
            }
        }
        if secondary {
            state.cancel_selection();
        }
    }

    // A clipboard region follows the cursor until a click drops it
    // (`GameController::LoadClipboard` + `PlaceSave`).
    if state.placement.is_some() {
        if let Some(pos) = sim_pos {
            state.move_placement(pos);
        }
        let clicked = ui.ctx().input(|input| input.pointer.primary_pressed());
        let cancelled = ui.ctx().input(|input| input.pointer.secondary_pressed());
        if clicked && response.contains_pointer() {
            state.commit_placement();
        } else if cancelled {
            state.placement = None;
        }
    }

    // Sampling is read-only, so the middle click keeps working in the
    // navigation mode; only the editing tools are switched off there.
    // `GameView::GetToolActive`: the scope swallows paint input until a click
    // pins it.
    let zoom_captures_input = zoom_was_unfixed;
    let painting = !state.move_view
        && !zoom_captures_input
        && state.select_mode == crate::app_state::selection::SelectMode::None
        && state.placement.is_none();
    if response.middle_clicked()
        && let Some((x, y)) = sim_pos
    {
        // A click samples the pixel the device holds now, not the HUD's cached
        // neighbour from up to 40 ms ago.
        if let Some(particle) = state.particle_at_now(x, y) {
            state.primary_tool = SelectedTool::Element(particle.ty);
        } else if let Some(wall) = state.wall_at(x, y)
            && wall != WL_ERASE
        {
            state.primary_tool = SelectedTool::Wall(wall);
        }
    }

    if painting {
        // The sign tool is a click-to-edit UI action, not a brush stroke: it
        // is handled below once the sign boxes have been laid out.
        for is_secondary in [false, true] {
            let sign_tool = state.selected_tool(is_secondary) == SelectedTool::SignTool;
            // `GameController::MouseDown`/`MouseUp` (`GameController.cpp:531-591`):
            // a click on a special `{c:..}`/`{t:..}`/`{b|..}`/`{s:..}` sign
            // runs its action and swallows the stroke. The sign tool's left
            // button is the exception and opens the editor below.
            if handle_special_sign_click(
                ui.ctx(),
                &painter,
                view,
                state,
                &frame,
                pointer,
                is_secondary,
                sign_tool,
            ) {
                continue;
            }
            if !sign_tool {
                handle_paint_button(ui.ctx(), state, &frame, is_secondary);
            }
        }
    }
    if painting
        && state.engine.bufs.config.show_brush
        && let Some((cx, cy)) = sim_pos
    {
        draw_brush_outline(
            &painter,
            view,
            cx,
            cy,
            state.brush_rx,
            state.brush_ry,
            state.brush_shape,
        );
    }

    if let Some((win_rect, step)) = zoom_win_rect {
        let cfg = &state.engine.bufs.config;
        let scope_min = view.to_screen(cfg.zoom_scope_x as f32, cfg.zoom_scope_y as f32);
        let scope_size = cfg.zoom_scope_size as f32 * view.scale;
        let scope_rect = Rect::from_min_size(scope_min, Vec2::splat(scope_size));
        painter.rect_stroke(
            scope_rect,
            CornerRadius::ZERO,
            Stroke::new(1.0, Color32::WHITE),
            StrokeKind::Outside,
        );
        painter.rect_stroke(
            scope_rect.expand(1.0),
            CornerRadius::ZERO,
            Stroke::new(1.0, Color32::BLACK),
            StrokeKind::Outside,
        );

        let u0 = cfg.zoom_scope_x as f32 / xres as f32;
        let v0 = cfg.zoom_scope_y as f32 / yres as f32;
        let u1 = (cfg.zoom_scope_x + cfg.zoom_scope_size) as f32 / xres as f32;
        let v1 = (cfg.zoom_scope_y + cfg.zoom_scope_size) as f32 / yres as f32;
        // `Graphics::RenderZoom` fills the window black first, then copies
        // each source pixel into a `ZFACTOR - 1` block, so every magnified
        // pixel keeps a one-pixel black grid line.
        painter.rect_filled(win_rect, CornerRadius::ZERO, Color32::BLACK);
        painter.image(
            sim_tex_id,
            win_rect,
            Rect::from_min_max(pos2(u0, v0), pos2(u1, v1)),
            Color32::WHITE,
        );
        if step > 1.0 {
            let gap = Stroke::new(1.0, Color32::BLACK);
            let mut gx = win_rect.left() + step;
            while gx < win_rect.right() {
                painter.line_segment([pos2(gx, win_rect.top()), pos2(gx, win_rect.bottom())], gap);
                gx += step;
            }
            let mut gy = win_rect.top() + step;
            while gy < win_rect.bottom() {
                painter.line_segment([pos2(win_rect.left(), gy), pos2(win_rect.right(), gy)], gap);
                gy += step;
            }
        }
        painter.rect_stroke(
            win_rect,
            CornerRadius::ZERO,
            Stroke::new(1.5, Color32::from_rgb(0xC0, 0xC0, 0xC0)),
            StrokeKind::Outside,
        );
        painter.rect_stroke(
            win_rect.expand(1.5),
            CornerRadius::ZERO,
            Stroke::new(1.0, Color32::BLACK),
            StrokeKind::Outside,
        );
    }

    // Rubber band (`GameView::Render` selection rectangle) and the clipboard
    // ghost that follows the cursor before it is dropped.
    if state.select_mode != crate::app_state::selection::SelectMode::None
        && let Some((x0, y0, x1, y1)) = state.selection_rect()
    {
        let min = view.to_screen(x0 as f32, y0 as f32);
        let max = view.to_screen((x1 + 1) as f32, (y1 + 1) as f32);
        painter.rect_stroke(
            Rect::from_min_max(min, max),
            CornerRadius::ZERO,
            Stroke::new(1.0, Color32::WHITE),
            StrokeKind::Outside,
        );
    }
    if let Some(placement) = &state.placement {
        let (ax, ay) = placement.anchor;
        let w = placement.region.w as f32;
        let h = placement.region.h as f32;
        let min = view.to_screen(ax as f32, ay as f32);
        let max = view.to_screen(ax as f32 + w, ay as f32 + h);
        painter.rect_stroke(
            Rect::from_min_max(min, max),
            CornerRadius::ZERO,
            Stroke::new(1.0, Color32::from_rgb(0xC0, 0xC0, 0xC0)),
            StrokeKind::Outside,
        );
        // The ghost draws at most a few thousand cells; a huge region still
        // shows its outline, which is what the placement needs.
        for particle in placement.region.particles.iter().take(4096) {
            let colour = state
                .engine
                .registry
                .elements
                .get(particle.ty as usize)
                .map(|e| e.colour)
                .unwrap_or(0x808080);
            let px = ax as f32 + particle.x;
            let py = ay as f32 + particle.y;
            let cell =
                Rect::from_min_max(view.to_screen(px, py), view.to_screen(px + 1.0, py + 1.0));
            let rgb = Color32::from_rgb(
                ((colour >> 16) & 0xFF) as u8,
                ((colour >> 8) & 0xFF) as u8,
                (colour & 0xFF) as u8,
            );
            painter.rect_filled(cell, CornerRadius::ZERO, rgb.gamma_multiply(0.75));
        }
    }

    if let (Some(start), Some(end)) = (state.drag_start, sim_pos) {
        if mods.shift && !(mods.ctrl || mods.command) {
            let end = if mods.alt {
                snap_line_coords(start.0, start.1, end.0, end.1)
            } else {
                end
            };
            draw_line_preview(
                &painter,
                view,
                start,
                end,
                state.brush_rx,
                state.brush_ry,
                state.brush_shape,
            );
        } else if mods.ctrl || mods.command {
            draw_rect_preview(&painter, view, start, end);
        }
    }

    // Rigid bodies spawn once per press with the brush footprint, so show that
    // footprint continuously for the tool bound to either button.
    let rigid_tool = match (state.primary_tool, state.secondary_tool) {
        (SelectedTool::RigidBody { shape, .. }, _) => Some(shape),
        (_, SelectedTool::RigidBody { shape, .. }) => Some(shape),
        _ => None,
    };
    if let (Some(shape), Some((cx, cy))) = (rigid_tool, sim_pos) {
        draw_rigid_preview(
            &painter,
            view,
            cx,
            cy,
            state.brush_rx,
            state.brush_ry,
            shape,
        );
    }

    // Signs are drawn after the simulation and the tool previews, and before
    // the HUD, matching `Renderer::DrawSigns`'s place on top of the scene.
    // `SignWindow::DoMouseMove`: a sign being moved follows the pointer until
    // the next click places it.
    if state.sign_editor.moving
        && let Some(index) = state.sign_editor.index
        && let Some((x, y)) = state.hover_pos
    {
        state.engine.move_sign(index, x, y);
        state.sign_editor.x = x;
        state.sign_editor.y = y;
    }
    let sign_boxes = draw_signs(&painter, view, state);
    if painting {
        for is_secondary in [false, true] {
            if state.selected_tool(is_secondary) != SelectedTool::SignTool {
                continue;
            }
            let pressed = if is_secondary {
                frame.secondary_pressed
            } else {
                frame.primary_pressed
            };
            if !pressed || !frame.pointer_in_view {
                continue;
            }
            // A right click on a special sign is routing its action, not
            // opening the editor (`GameController.cpp:531-533`).
            if is_secondary && press_pending(ui.ctx(), is_secondary) {
                continue;
            }
            let Some(position) = pointer else {
                continue;
            };
            // An open editor owns the click: opening a second one would
            // discard the text the user is typing.
            if state.sign_editor.open {
                continue;
            }
            // `SignWindow::DoMouseDown`: the click places a sign that is
            // following the pointer.
            if state.sign_editor.moving {
                state.close_sign_editor();
                continue;
            }
            if let Some(index) = box_hit(&sign_boxes, position) {
                state.open_sign_editor_for(index);
            } else if let Some((x, y)) = sim_pos {
                state.open_sign_editor_at(x, y);
            }
        }
    }

    if state.engine.bufs.config.hud {
        if state.is_paused() {
            painter.text(
                view_rect.center_top() + vec2(0.0, 20.0),
                Align2::CENTER_CENTER,
                t("viewport.paused_hint"),
                FontId::proportional(18.0),
                Color32::from_rgba_premultiplied(255, 205, 90, 235),
            );
        }

        painter.text(
            view_rect.left_bottom() + vec2(8.0, -8.0),
            Align2::LEFT_BOTTOM,
            format!("{:.0}%", state.zoom * 100.0),
            FontId::monospace(12.0),
            TEXT_DIM,
        );
        if let Some((x, y)) = sim_pos {
            painter.text(
                view_rect.right_bottom() + vec2(-8.0, -8.0),
                Align2::RIGHT_BOTTOM,
                format!("{x}, {y}"),
                FontId::monospace(12.0),
                TEXT_DIM,
            );
        }

        if state.engine.bufs.config.debug_hud {
            draw_debug_hud(&painter, view_rect, state, sim_pos);
        }
    }
}

/// Draws the TPT-style debug telemetry HUD overlay over the viewport.
fn draw_debug_hud(
    painter: &egui::Painter,
    view_rect: Rect,
    state: &AppState,
    sim_pos: Option<(i32, i32)>,
) {
    let (xres, yres) = state.canvas_size();
    let cfg = &state.engine.bufs.config;
    let fps_str = format!("{:5.1}", state.fps);
    let canvas_dim = format!("{xres}x{yres}");
    let parts_str = state.active_particles().to_string();
    let bodies_str = state.rigid_body_count().to_string();
    let r_hex = format!("{:04X}", cfg.render_mode);
    let d_hex = format!("{:04X}", cfg.display_mode);
    let c_hex = format!("{:04X}", cfg.colour_mode);
    let deco_str = cfg.decorations.to_string();
    let grid_str = cfg.grid.to_string();
    let checker_str = cfg.grid_checkerboard.to_string();
    let brush_str = cfg.show_brush.to_string();
    let zoom_en = cfg.zoom_enabled.to_string();
    let zoom_factor = cfg.zoom_factor.to_string();
    let zoom_scope = cfg.zoom_scope_size.to_string();
    let zoom_win_x = cfg.zoom_window_x.to_string();
    let zoom_win_y = cfg.zoom_window_y.to_string();

    let mut lines = vec![
        t_fmt("viewport.hud_fps", &[&fps_str]),
        t_fmt("viewport.hud_canvas_parts", &[&canvas_dim, &parts_str]),
        t_fmt("viewport.hud_rigid_bodies", &[&bodies_str]),
        t_fmt("viewport.hud_modes", &[&r_hex, &d_hex, &c_hex]),
        t_fmt(
            "viewport.hud_toggles",
            &[&deco_str, &grid_str, &checker_str, &brush_str],
        ),
        t_fmt(
            "viewport.hud_zoom",
            &[
                &zoom_en,
                &zoom_factor,
                &zoom_scope,
                &zoom_win_x,
                &zoom_win_y,
            ],
        ),
    ];
    if let Some((x, y)) = sim_pos {
        let x_str = x.to_string();
        let y_str = y.to_string();
        if let Some(p) = state.particle_at(x, y) {
            let temp_str = format!("{:.1}", p.temp);
            let life_str = p.life.to_string();
            let ctype_str = p.ctype.to_string();
            let tmp_str = p.tmp.to_string();
            let tmp2_str = p.tmp2.to_string();
            let vx_str = format!("{:+.2}", p.vx);
            let vy_str = format!("{:+.2}", p.vy);
            let flags_str = format!("{:04X}", p.flags);
            lines.push(t_fmt(
                "viewport.hud_particle",
                &[&super::shared::element_name(state, p.ty), &x_str, &y_str],
            ));
            lines.push(t_fmt(
                "viewport.hud_particle_props",
                &[&temp_str, &life_str, &ctype_str, &tmp_str, &tmp2_str],
            ));
            lines.push(t_fmt(
                "viewport.hud_vel_flags",
                &[&vx_str, &vy_str, &flags_str],
            ));
        } else {
            lines.push(t_fmt("viewport.hud_cursor_empty", &[&x_str, &y_str]));
        }
    }
    let pad = 8.0;
    let line_height = 14.0;
    let box_height = lines.len() as f32 * line_height + pad * 2.0;
    let box_width = 380.0;
    let box_rect = Rect::from_min_size(
        view_rect.left_top() + vec2(10.0, 10.0),
        vec2(box_width, box_height),
    );
    painter.rect_filled(
        box_rect,
        CornerRadius::same(4),
        Color32::from_black_alpha(200),
    );
    painter.rect_stroke(
        box_rect,
        CornerRadius::same(4),
        Stroke::new(1.0, Color32::from_rgb(80, 120, 180)),
        StrokeKind::Outside,
    );
    for (i, line) in lines.iter().enumerate() {
        painter.text(
            box_rect.min + vec2(pad, pad + i as f32 * line_height),
            Align2::LEFT_TOP,
            line,
            FontId::monospace(11.0),
            if i == 0 {
                Color32::from_rgb(140, 200, 255)
            } else {
                Color32::from_rgb(220, 225, 230)
            },
        );
    }
}

/// Size in screen pixels of the magnified window and of one source pixel.
///
/// `Graphics::RenderZoom` draws `zoomScopeSize * ZFACTOR` framebuffer pixels.
/// The port scales that by the view zoom, then caps it to the viewport so a
/// small canvas cannot produce a window bigger than the screen; the cell size
/// follows the capped size, which keeps the pointer remap exact.
fn zoom_window_metrics(
    scope_size: i32,
    factor: i32,
    view_scale: f32,
    view_side: f32,
) -> (f32, f32) {
    let scope = scope_size.max(1) as f32;
    let factor = factor.max(1) as f32;
    let scale = if view_scale.is_finite() && view_scale > 0.0 {
        view_scale
    } else {
        1.0
    };
    let side = if view_side.is_finite() && view_side > 0.0 {
        view_side
    } else {
        f32::MAX
    };
    let win_size = (scope * factor * scale).min(side).max(scope);
    (win_size, win_size / scope)
}

/// Source pixel under `pointer` when it sits inside the magnification window
/// (`GameModel::AdjustZoomCoords`).
fn zoom_source_cell(
    pointer: Pos2,
    win_rect: Rect,
    cell: f32,
    scope_x: i32,
    scope_y: i32,
) -> Option<(i32, i32)> {
    if !win_rect.contains(pointer) || cell <= 0.0 {
        return None;
    }
    let rel_x = ((pointer.x - win_rect.min.x) / cell).floor() as i32;
    let rel_y = ((pointer.y - win_rect.min.y) / cell).floor() as i32;
    Some((scope_x + rel_x, scope_y + rel_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The navigation keys move the camera, so the image travels the other
    /// way: W drags the image down, A drags it right. Getting this backwards
    /// made every WASD key look inverted.
    #[test]
    fn navigation_keys_move_the_camera_not_the_image() {
        assert_eq!(navigation_pan(true, false, false, false), vec2(0.0, 1.0));
        assert_eq!(navigation_pan(false, true, false, false), vec2(0.0, -1.0));
        assert_eq!(navigation_pan(false, false, true, false), vec2(1.0, 0.0));
        assert_eq!(navigation_pan(false, false, false, true), vec2(-1.0, 0.0));
        // Opposite keys cancel instead of doubling.
        assert_eq!(navigation_pan(true, true, false, false), Vec2::ZERO);
        assert_eq!(navigation_pan(false, false, true, true), Vec2::ZERO);
    }

    #[test]
    fn zoom_window_keeps_the_reference_magnification() {
        // 32 source pixels at ZFACTOR 8 in an unscaled view: 256 screen pixels,
        // one screen pixel per source pixel times eight (`Graphics.h:51`).
        let (size, cell) = zoom_window_metrics(32, 8, 1.0, 10_000.0);
        assert_eq!(size, 256.0);
        assert_eq!(cell, 8.0);
        // The view zoom multiplies both, so the window still shows the same
        // number of source pixels.
        let (size, cell) = zoom_window_metrics(32, 8, 2.0, 10_000.0);
        assert_eq!(size, 512.0);
        assert_eq!(cell, 16.0);
    }

    #[test]
    fn zoom_window_is_capped_to_the_viewport() {
        // A 128 pixel canvas fitted to a short viewport scales up five times;
        // without the cap the window would be five times the screen.
        let (size, cell) = zoom_window_metrics(32, 8, 5.0, 700.0);
        assert_eq!(size, 700.0);
        assert!((cell - 700.0 / 32.0).abs() < 1e-3);
    }

    #[test]
    fn zoom_window_hit_test_maps_back_to_the_source_pixel() {
        let win = Rect::from_min_size(pos2(100.0, 40.0), Vec2::splat(256.0));
        // Centre of the window is the centre of the scope: 16 of 32 across.
        assert_eq!(
            zoom_source_cell(pos2(228.0, 168.0), win, 8.0, 400, 200),
            Some((416, 216))
        );
        // Outside the window the pointer keeps its canvas coordinates.
        assert_eq!(zoom_source_cell(pos2(99.0, 40.0), win, 8.0, 400, 200), None);
        assert_eq!(
            zoom_source_cell(pos2(357.0, 40.0), win, 8.0, 400, 200),
            None
        );
    }
}
