//! 3D sandbox application state: mode, tool selection, undo log, save/load.
//!
//! The simulation and renderer live in `tpt_sim::voxel3d` and
//! [`crate::render3d`]; this module only wires them to [`AppState`]: the game
//! mode, the lazily created engine/renderer, the orbit camera, the palette
//! selection, the bounded brick undo log and the `.tpt3` save browser.
//!
//! [`AppState`]: super::AppState

use std::collections::HashSet;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tpt_core::constants::PT_STNE;
use tpt_core::particle::EdgeMode;
use tpt_core::voxel3d::{Brush3d, BrushShape3d, MAX_BODIES, VOXEL_DIM, Voxel3dConfig, brick_index};
use tpt_save::voxel3d::{Voxel3dBodySave, Voxel3dSaveData};
use tpt_sim::voxel3d::Voxel3dEngine;
use tpt_sim::voxel3d::pick::PickHit;

use crate::i18n::{t, t_fmt};
use crate::render3d::{Camera, Renderer3d};

use super::{AppScreen, AppState, SAVES_DIR};

/// File extension used by the 3D save format.
pub const VOXEL_SAVE_EXTENSION: &str = "tpt3";
/// Maximum number of brick snapshots the 3D undo log retains.
pub const VOXEL_UNDO_LIMIT: usize = 32;

/// Voxel radius of the "Stone Floor" button's slab sphere.
///
/// A sphere this large, centred far below the world, has an almost planar top
/// surface: every column of the 256 x 256 footprint is cut between `y = 2.0`
/// and `y = 2.9`, which paints exactly three layers (`y = 0, 1, 2`).
const FLOOR_SPHERE_RADIUS: f32 = 18_202.0;
/// Vertical centre of the "Stone Floor" sphere; see [`FLOOR_SPHERE_RADIUS`].
const FLOOR_SPHERE_Y: f32 = -18_199.1;

/// Largest voxel count a single flood fill may paint.
///
/// The connected-set gather and the undo snapshot both scale with the region,
/// so an unbounded fill of the empty sky would freeze the UI and multiply the
/// 64 MiB world read-back into a multi-hundred-MiB undo entry.
const MAX_FILL_VOXELS: usize = 1_000_000;
/// Largest number of bricks one fill may snapshot for undo (32 MiB).
const MAX_FILL_UNDO_BRICKS: usize = 16_384;

/// Top-level simulation flavour.
///
/// [`AppScreen`] stays unchanged: the 3D sandbox is still the in-game screen,
/// only the simulation behind it differs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GameMode {
    /// Classic powder simulation.
    #[default]
    Powder2D,
    /// Volumetric voxel sandbox.
    Sandbox3D,
}

impl GameMode {
    /// Persisted identifier, also used by `prefs.json`.
    pub fn key(self) -> &'static str {
        match self {
            Self::Powder2D => "2d",
            Self::Sandbox3D => "3d",
        }
    }

    /// Parses [`Self::key`]; anything unknown is the classic mode.
    pub fn from_key(key: &str) -> Self {
        match key {
            "3d" => Self::Sandbox3D,
            _ => Self::Powder2D,
        }
    }
}

/// Tool bound to the left mouse button in the 3D viewport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelTool {
    /// Paint the brush volume with the given element id.
    Element(u32),
    /// Clear the brush volume.
    Eraser,
    /// Spawn a rigid box at the pick point.
    StoneBox,
    /// Add heat to the non-empty voxels under the brush.
    Heat,
    /// Remove heat from the non-empty voxels under the brush.
    Cool,
}

impl VoxelTool {
    /// Whether the tool edits existing voxels instead of creating/removing them.
    pub fn is_temperature(self) -> bool {
        matches!(self, Self::Heat | Self::Cool)
    }
}

impl Default for VoxelTool {
    fn default() -> Self {
        Self::Element(PT_STNE)
    }
}

/// Gesture used by the 3D brush, mirroring TPT's `DrawMode` plus a volume box.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VoxelDrawMode {
    /// Freehand brush along the pointer path.
    #[default]
    Brush,
    /// Straight line between the press and release points.
    Line,
    /// Axis-aligned filled cuboid between press and release.
    Rect,
    /// Flood-fills the connected empty region under the cursor.
    Fill,
}

impl VoxelDrawMode {
    /// Stable locale key suffix.
    pub fn key(self) -> &'static str {
        match self {
            Self::Brush => "brush",
            Self::Line => "line",
            Self::Rect => "rect",
            Self::Fill => "fill",
        }
    }
}

/// One undo entry: the voxel and auxiliary words of every brick a brush
/// touched.
pub type Voxel3dBrickSnapshot = tpt_sim::voxel3d::VoxelBrickSnapshot;

/// Bounded undo/redo log of brick snapshots.
///
/// A brush touches at most a handful of bricks, so a snapshot is a few KiB;
/// pushing a full-world snapshot would be 64 MiB and is deliberately avoided.
#[derive(Default)]
pub struct Voxel3dUndoHistory {
    undo: Vec<Voxel3dBrickSnapshot>,
    redo: Vec<Voxel3dBrickSnapshot>,
}

impl Voxel3dUndoHistory {
    /// Records a new edit and drops the redo branch.
    pub fn push_undo(&mut self, entry: Voxel3dBrickSnapshot) {
        self.redo.clear();
        push_bounded(&mut self.undo, entry);
    }

    /// Puts a snapshot back on the undo stack after a redo, keeping the redo
    /// stack intact.
    pub fn restore_undo(&mut self, entry: Voxel3dBrickSnapshot) {
        push_bounded(&mut self.undo, entry);
    }

    /// Records the state an undo replaced, for the redo branch.
    pub fn push_redo(&mut self, entry: Voxel3dBrickSnapshot) {
        push_bounded(&mut self.redo, entry);
    }

    /// Removes and returns the newest undo snapshot.
    pub fn take_undo(&mut self) -> Option<Voxel3dBrickSnapshot> {
        self.undo.pop()
    }

    /// Removes and returns the newest redo snapshot.
    pub fn take_redo(&mut self) -> Option<Voxel3dBrickSnapshot> {
        self.redo.pop()
    }

    /// Whether an undo snapshot exists.
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    /// Whether a redo snapshot exists.
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Drops both stacks.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}

/// Appends `entry`, dropping the oldest snapshots past [`VOXEL_UNDO_LIMIT`].
fn push_bounded(stack: &mut Vec<Voxel3dBrickSnapshot>, entry: Voxel3dBrickSnapshot) {
    stack.push(entry);
    if stack.len() > VOXEL_UNDO_LIMIT {
        stack.remove(0);
    }
}

/// Brick indices covered by a brush's inclusive voxel AABB.
///
/// Used to capture the undo payload before an edit; the list is deduplicated
/// and sorted, so it is also the exact index list handed to
/// [`Voxel3dEngine::read_bricks`].
pub(crate) fn brush_bricks(brush: Brush3d) -> Vec<u32> {
    let (min, max) = brush.bounds();
    if min[0] > max[0] || min[1] > max[1] || min[2] > max[2] {
        return Vec::new();
    }
    let mut bricks = Vec::new();
    for y in (min[1] >> 3)..=(max[1] >> 3) {
        for z in (min[2] >> 3)..=(max[2] >> 3) {
            for x in (min[0] >> 3)..=(max[0] >> 3) {
                bricks.push(brick_index(x << 3, y << 3, z << 3) as u32);
            }
        }
    }
    bricks
}

/// Default camera for a fresh 3D world.
pub(crate) fn default_voxel_camera() -> Camera {
    Camera::default()
}

/// Strips a `.tpt3` suffix before deferring to the shared stem validator.
fn normalize_voxel_save_stem(name: &str) -> Result<String, String> {
    let trimmed = name.trim();
    let suffix = format!(".{VOXEL_SAVE_EXTENSION}");
    let stem = if trimmed.len() >= suffix.len()
        && trimmed
            .get(trimmed.len() - suffix.len()..)
            .is_some_and(|tail| tail.eq_ignore_ascii_case(&suffix))
    {
        &trimmed[..trimmed.len() - suffix.len()]
    } else {
        trimmed
    };
    super::commands::normalize_save_stem(stem)
}

impl AppState {
    /// Switching to a mode also makes it the persisted mode.
    pub fn set_game_mode(&mut self, mode: GameMode) {
        if self.mode == mode {
            if mode == GameMode::Sandbox3D {
                self.ensure_voxel_engine();
                self.screen = AppScreen::InGame;
                self.last_sim_step_instant = std::time::Instant::now();
                self.sim_step_debt = 0.0;
            }
            return;
        }
        self.mode = mode;
        self.screen = AppScreen::InGame;
        self.last_sim_step_instant = std::time::Instant::now();
        self.sim_step_debt = 0.0;
        self.prefs.game_mode = mode;
        self.prefs.save();
        match mode {
            GameMode::Powder2D => {
                self.set_status(t("voxel3d.status_back_2d"), false);
            }
            GameMode::Sandbox3D => {
                self.ensure_voxel_engine();
                self.refresh_voxel_saves();
                self.screen = AppScreen::InGame;
                self.set_status(t("voxel3d.status_entered"), false);
            }
        }
    }

    /// Enters the 3D sandbox, creating the world on first use.
    pub fn enter_voxel_mode(&mut self) {
        self.set_game_mode(GameMode::Sandbox3D);
    }

    /// Returns to the 2D powder mode; the 3D world stays resident.
    pub fn leave_voxel_mode(&mut self) {
        self.set_game_mode(GameMode::Powder2D);
    }

    /// Creates [`Voxel3dEngine`] on the shared CubeCL client if it does not
    /// exist yet and returns it.
    pub fn ensure_voxel_engine(&mut self) -> &mut Voxel3dEngine {
        if self.voxel3d.is_none() {
            let client = self.engine.bufs.client.clone();
            let config = self.voxel_config();
            self.voxel3d = Some(Box::new(Voxel3dEngine::new(client, config)));
        }
        self.voxel3d
            .as_mut()
            .expect("the 3D engine was created above")
    }

    /// Creates [`Renderer3d`] on the application device if needed.
    ///
    /// The renderer settings are copied from the live engine config rather than
    /// from the persisted preferences: a loaded `.tpt3` carries its own 3D
    /// settings and must not be overwritten by the local defaults.
    pub fn ensure_voxel_renderer(&mut self, device: &wgpu::Device) {
        if self.voxel_renderer.is_none() {
            self.voxel_renderer = Some(Renderer3d::new(device, wgpu::TextureFormat::Rgba8Unorm));
        }
        self.sync_voxel_render_settings();
    }

    /// Copies the live engine's 3D settings into the renderer.
    pub fn sync_voxel_render_settings(&mut self) {
        let Some(engine) = self.voxel3d.as_ref() else {
            return;
        };
        let (shadows, ao_rays) = (engine.config.shadows, engine.config.ao_rays);
        if let Some(renderer) = self.voxel_renderer.as_mut() {
            let settings = renderer.settings_mut();
            settings.shadows = shadows;
            settings.ambient_occlusion = ao_rays > 0;
        }
    }

    /// Render scale the 3D renderer must use.
    pub fn voxel_render_scale(&self) -> f32 {
        self.voxel3d
            .as_ref()
            .map(|engine| engine.config.render_scale)
            .unwrap_or(self.prefs.voxel_render_scale)
    }

    /// Config used when the 3D engine is first created.
    fn voxel_config(&self) -> Voxel3dConfig {
        Voxel3dConfig {
            render_scale: self.prefs.voxel_render_scale,
            ao_rays: self.prefs.voxel_ao_rays,
            shadows: self.prefs.voxel_shadows,
            air_iterations: self.prefs.voxel_air_iterations,
            edge_mode: self.voxel_edge_mode,
            ..Voxel3dConfig::default()
        }
    }

    /// Pushes the persisted 3D settings into the live engine and renderer.
    pub fn apply_voxel_prefs(&mut self) {
        if let Some(engine) = self.voxel3d.as_mut() {
            engine.config.render_scale = self.prefs.voxel_render_scale;
            engine.config.ao_rays = self.prefs.voxel_ao_rays;
            engine.config.shadows = self.prefs.voxel_shadows;
            engine.config.air_iterations = self.prefs.voxel_air_iterations;
        }
        if let Some(renderer) = self.voxel_renderer.as_mut() {
            let settings = renderer.settings_mut();
            settings.shadows = self.prefs.voxel_shadows;
            settings.ambient_occlusion = self.prefs.voxel_ao_rays > 0;
        }
    }

    /// Refreshes the throttled 3D telemetry (active brick count).
    pub fn refresh_voxel_stats(&mut self) {
        if self.mode != GameMode::Sandbox3D {
            return;
        }
        if self.voxel_stats_at.elapsed() < std::time::Duration::from_millis(250) {
            return;
        }
        self.voxel_stats_at = std::time::Instant::now();
        if let Some(engine) = self.voxel3d.as_ref() {
            self.voxel_active_bricks = engine.read_active_bricks();
        }
    }

    /// Drops pointer, hover and stroke caches after the world changed.
    fn clear_voxel_pointer_state(&mut self) {
        self.voxel_last_pick = None;
        self.voxel_hover_pick = None;
        self.voxel_hover_pointer = None;
        self.voxel_stroke_start = None;
        self.voxel_stroke_last = None;
        self.voxel_stroke_mode = None;
        self.voxel_stroke_snapshot.clear();
        self.voxel_stroke_bricks.clear();
        self.voxel_stroke_active = false;
    }

    /// Resets the 3D world, camera and undo log.
    pub fn reset_voxel_world(&mut self) {
        let engine = self.ensure_voxel_engine();
        engine.reset();
        engine.config.edge_mode = EdgeMode::Void;
        self.voxel_edge_mode = EdgeMode::Void;
        self.voxel_undo.clear();
        self.voxel_camera = default_voxel_camera();
        self.clear_voxel_pointer_state();
        self.voxel_active_bricks = Some(0);
        self.set_status(t("voxel3d.status_reset"), false);
    }

    /// Casts a pick ray into the 3D world.
    pub fn pick_voxel(&self, origin: [f32; 3], dir: [f32; 3]) -> Option<PickHit> {
        self.voxel3d.as_ref()?.pick(origin, dir)
    }

    /// Starts a paint stroke and captures the bricks under `brush`.
    ///
    /// The snapshot accumulates as the stroke reaches new bricks, so a freehand
    /// drag remains one undo entry without ever reading the whole 64 MiB world.
    pub fn begin_voxel_stroke(&mut self, brush: Brush3d) {
        if !self.voxel_stroke_active {
            self.voxel_stroke_active = true;
            self.voxel_stroke_snapshot.clear();
            self.voxel_stroke_bricks.clear();
        }
        self.capture_voxel_bricks(brush);
    }

    /// Captures the not-yet-recorded bricks of one brush stamp.
    fn capture_voxel_bricks(&mut self, brush: Brush3d) {
        let bricks: Vec<u32> = brush_bricks(brush)
            .into_iter()
            .filter(|brick| self.voxel_stroke_bricks.insert(*brick))
            .collect();
        if bricks.is_empty() {
            return;
        }
        let Some(old) = self
            .voxel3d
            .as_ref()
            .and_then(|engine| engine.read_brick_snapshot(&bricks))
        else {
            for brick in bricks {
                self.voxel_stroke_bricks.remove(&brick);
            }
            return;
        };
        self.voxel_stroke_snapshot.extend(old);
    }

    /// Captures the bricks covered by a swept segment.
    pub fn capture_voxel_segment(&mut self, from: [f32; 3], to: [f32; 3]) {
        let mut half = [0.0f32; 3];
        let mut center = [0.0f32; 3];
        for axis in 0..3 {
            let low = from[axis].min(to[axis]);
            let high = from[axis].max(to[axis]);
            half[axis] = (high - low) * 0.5 + self.voxel_brush_radii[axis].max(1) as f32;
            center[axis] = (low + high) * 0.5;
        }
        let brush = Brush3d::with_shape(center, half, BrushShape3d::Cuboid);
        self.capture_voxel_bricks(brush);
    }

    /// Ends the current stroke and publishes its undo entry.
    pub fn end_voxel_stroke(&mut self) {
        if !self.voxel_stroke_active {
            return;
        }
        self.voxel_stroke_active = false;
        if !self.voxel_stroke_snapshot.is_empty() {
            let snapshot = std::mem::take(&mut self.voxel_stroke_snapshot);
            self.voxel_undo.push_undo(snapshot);
        }
        self.voxel_stroke_bricks.clear();
    }

    /// Brush placed at `center` with the current three radii and shape.
    pub fn voxel_brush_at(&self, center: [f32; 3]) -> Brush3d {
        let radii = self
            .voxel_brush_radii
            .map(|radius| radius.clamp(1, 64) as f32);
        Brush3d::with_shape(center, radii, self.voxel_brush_shape)
    }

    /// Resizes the unlocked brush axes.
    ///
    /// Mirrors TPT `Brush::AdjustSize`: a logarithmic change moves each axis by
    /// a fifth of its current size (minimum one), a linear change by one. The
    /// per-axis locks correspond to TPT's Ctrl/Shift modifiers plus the Z lock
    /// the 3D brush needs.
    pub fn adjust_voxel_brush(&mut self, delta: i32, logarithmic: bool) {
        for axis in 0..3 {
            if self.voxel_brush_locks[axis] {
                continue;
            }
            let current = self.voxel_brush_radii[axis].clamp(1, 64) as i32;
            let step = if logarithmic { (current / 5).max(1) } else { 1 };
            self.voxel_brush_radii[axis] = (current + delta.signum() * step).clamp(1, 64) as u32;
        }
    }

    /// Largest current brush radius, used for empty-world placement.
    pub fn voxel_brush_radius(&self) -> f32 {
        self.voxel_brush_radii
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
            .clamp(1, 64) as f32
    }

    /// Applies one brush stamp without starting an undo snapshot.
    ///
    /// The freehand gesture calls this once per press and then once per
    /// dragged segment; the press owns the undo snapshot.
    pub fn stamp_voxel_brush(&mut self, center: [f32; 3], erase: bool) -> u32 {
        let brush = self.voxel_brush_at(center);
        let tool = self.voxel_tool;
        let erasing = erase || matches!(tool, VoxelTool::Eraser);
        let engine = self.ensure_voxel_engine();
        match tool {
            VoxelTool::Heat => engine.adjust_brush_temp(brush, 200.0),
            VoxelTool::Cool => engine.adjust_brush_temp(brush, -200.0),
            VoxelTool::Element(ty) if !erasing => engine.place_brush(ty, brush),
            VoxelTool::StoneBox if !erasing => engine.place_brush(PT_STNE, brush),
            _ => engine.erase_brush(brush),
        }
    }

    /// Paints or erases one brush stamp and captures its undo snapshot.
    pub fn paint_voxel_brush(&mut self, center: [f32; 3], erase: bool) -> u32 {
        let brush = self.voxel_brush_at(center);
        self.begin_voxel_stroke(brush);
        self.stamp_voxel_brush(center, erase)
    }

    /// Stamps the brush along a segment between two drag samples.
    pub fn stamp_voxel_segment(&mut self, from: [f32; 3], to: [f32; 3], erase: bool) -> u32 {
        let radii = self
            .voxel_brush_radii
            .map(|radius| radius.clamp(1, 64) as f32);
        let shape = self.voxel_brush_shape;
        let tool = self.voxel_tool;
        let erasing = erase || matches!(tool, VoxelTool::Eraser);
        let end_brush = self.voxel_brush_at(to);
        let engine = self.ensure_voxel_engine();
        match tool {
            // Temperature tools use the brush only; a swept heat line is not
            // part of TPT's sim-tool gesture.
            VoxelTool::Heat => engine.adjust_brush_temp(end_brush, 200.0),
            VoxelTool::Cool => engine.adjust_brush_temp(end_brush, -200.0),
            VoxelTool::Element(ty) if !erasing => {
                engine.place_brush_segment(ty, from, to, radii, shape)
            }
            VoxelTool::StoneBox if !erasing => {
                engine.place_brush_segment(PT_STNE, from, to, radii, shape)
            }
            _ => engine.erase_brush_segment(from, to, radii, shape),
        }
    }

    /// Flood-fills the connected empty region that contains `seed`.
    ///
    /// TPT's FLOD tool is a connected-empty fill, so this is a host breadth
    /// search over the packed world words (the only voxel read-back in the 3D
    /// edit path), followed by one indexed device dispatch. The region is
    /// bounded by [`MAX_FILL_VOXELS`] and its undo snapshot by
    /// [`MAX_FILL_UNDO_BRICKS`], because an unbounded sky fill would touch the
    /// entire 64 MiB world. Returns the number of painted voxels.
    pub fn fill_voxel_at(&mut self, seed: [u32; 3]) -> u32 {
        let ty = match self.voxel_tool {
            VoxelTool::Element(ty) => ty,
            VoxelTool::StoneBox => PT_STNE,
            VoxelTool::Eraser => {
                self.set_status(t("voxel3d.status_fill_needs_element"), true);
                return 0;
            }
            VoxelTool::Heat | VoxelTool::Cool => {
                self.set_status(t("voxel3d.status_fill_needs_element"), true);
                return 0;
            }
        };

        let engine = self.ensure_voxel_engine();
        let dim = engine.config.dim.min(VOXEL_DIM);
        if seed.iter().any(|&coord| coord >= dim) {
            self.set_status(t("voxel3d.status_no_target"), true);
            return 0;
        }
        let Some(world) = engine.read_world_words() else {
            self.set_status(t("voxel3d.error_read_world"), true);
            return 0;
        };
        let indices = match collect_fill_region(&world, dim, seed) {
            Ok(indices) => indices,
            Err(FillRegionError::Solid) => {
                self.set_status(t("voxel3d.status_fill_solid"), true);
                return 0;
            }
            Err(FillRegionError::OutOfBounds) => {
                self.set_status(t("voxel3d.status_no_target"), true);
                return 0;
            }
            Err(FillRegionError::TooLarge) => {
                self.set_status(t("voxel3d.status_fill_too_large"), true);
                return 0;
            }
        };

        let mut bricks: HashSet<usize> = HashSet::new();
        for &index in &indices {
            let x = index & 0xFF;
            let y = (index >> 16) & 0xFF;
            let z = (index >> 8) & 0xFF;
            bricks.insert(brick_index(x, y, z));
        }
        if bricks.len() > MAX_FILL_UNDO_BRICKS {
            self.set_status(t("voxel3d.status_fill_too_large"), true);
            return 0;
        }

        let brick_list: Vec<u32> = bricks.into_iter().map(|brick| brick as u32).collect();
        let Some(snapshot) = engine.read_brick_snapshot(&brick_list) else {
            self.set_status(t("voxel3d.error_read_world"), true);
            return 0;
        };
        let painted = engine.paint_indices(ty, &indices);
        if painted == 0 {
            self.set_status(t("voxel3d.status_no_target"), true);
            return 0;
        }
        self.voxel_undo.push_undo(snapshot);
        self.set_status(
            t_fmt("voxel3d.status_filled", &[&painted.to_string()]),
            false,
        );
        painted
    }

    /// Applies the TPT rectangle gesture: a sharp filled cuboid between the two
    /// drag samples. The brush radius is deliberately ignored, like
    /// `ElementTool::DrawRect` calling `Simulation::CreateBox`.
    pub fn paint_voxel_box(&mut self, from: [f32; 3], to: [f32; 3], erase: bool) -> u32 {
        let mut half = [0.5f32; 3];
        let mut center = [0.0f32; 3];
        for axis in 0..3 {
            let low = from[axis].min(to[axis]);
            let high = from[axis].max(to[axis]);
            half[axis] = (high - low) * 0.5 + 0.5;
            center[axis] = (low + high) * 0.5;
        }
        let brush = Brush3d::with_shape(center, half, BrushShape3d::Cuboid);
        self.begin_voxel_stroke(brush);
        let tool = self.voxel_tool;
        let erasing = erase || matches!(tool, VoxelTool::Eraser);
        let engine = self.ensure_voxel_engine();
        match tool {
            VoxelTool::Heat => engine.adjust_brush_temp(brush, 200.0),
            VoxelTool::Cool => engine.adjust_brush_temp(brush, -200.0),
            VoxelTool::Element(ty) if !erasing => engine.place_brush(ty, brush),
            VoxelTool::StoneBox if !erasing => engine.place_brush(PT_STNE, brush),
            _ => engine.erase_brush(brush),
        }
    }

    /// Spawns the rigid box tool at `hit` (or the world centre as fallback).
    pub fn spawn_voxel_box(&mut self, hit: Option<PickHit>) -> Result<usize, String> {
        let half = self.voxel_box_half.map(|value| value.clamp(1, 16));
        let element = self.voxel_box_element;
        let engine = self.ensure_voxel_engine();
        if engine.rigid.body_count as usize >= MAX_BODIES {
            let message = t_fmt("voxel3d.error_max_bodies", &[&MAX_BODIES.to_string()]);
            self.set_status(message.clone(), true);
            return Err(message);
        }

        let center = match hit {
            Some(hit) => {
                let mut center = [
                    hit.voxel[0] as f32 + 0.5,
                    hit.voxel[1] as f32 + 0.5,
                    hit.voxel[2] as f32 + 0.5,
                ];
                for axis in 0..3 {
                    center[axis] += hit.normal[axis] as f32 * (half[axis] as f32 + 0.5);
                }
                center
            }
            None => [128.0, 96.0, 128.0],
        };
        let slot = engine.spawn_box(center, half, element);
        self.set_status(t("voxel3d.status_box_spawned"), false);
        Ok(slot)
    }

    /// Paints the 2..3 voxel-thick stone floor offered by the sidebar.
    ///
    /// One sphere centred below the world paints the slab: the huge radius makes
    /// the surface flat across the whole footprint, and the world clip at
    /// `y = 0` turns the lower half into a floor.
    pub fn paint_voxel_floor(&mut self) {
        let brush = Brush3d::new([128.0, FLOOR_SPHERE_Y, 128.0], FLOOR_SPHERE_RADIUS);
        let (min, max) = brush.bounds();
        let mut bricks = Vec::new();
        for by in (min[1] / 8)..=(max[1] / 8) {
            for bz in (min[2] / 8)..=(max[2] / 8) {
                for bx in (min[0] / 8)..=(max[0] / 8) {
                    bricks.push(brick_index(bx * 8, by * 8, bz * 8) as u32);
                }
            }
        }
        let engine = self.ensure_voxel_engine();
        let Some(snapshot) = engine.read_brick_snapshot(&bricks) else {
            self.set_status(t("voxel3d.error_read_world"), true);
            return;
        };
        let painted = engine.place_brush(PT_STNE, brush);
        if painted == 0 {
            return;
        }
        self.voxel_undo.push_undo(snapshot);
        self.set_status(
            t_fmt("voxel3d.status_floor", &[&painted.to_string()]),
            false,
        );
    }

    /// Sets the boundary behaviour of loose voxels (`void`, `solid`, `loop`).
    ///
    /// The live engine reads `config.edge_mode` every tick, so changing it
    /// needs no reset; a later `.tpt3` load restores the saved mode.
    pub fn set_voxel_edge_mode(&mut self, mode: EdgeMode) {
        if self.voxel_edge_mode == mode {
            return;
        }
        self.voxel_edge_mode = mode;
        if let Some(engine) = self.voxel3d.as_mut() {
            engine.config.edge_mode = mode;
        }
        let key = match mode {
            EdgeMode::Void => "voxel3d.edge_void",
            EdgeMode::Solid => "voxel3d.edge_solid",
            EdgeMode::Loop => "voxel3d.edge_loop",
        };
        self.set_status(t_fmt("voxel3d.status_edge_mode", &[t(key)]), false);
    }

    /// Restores the newest 3D undo snapshot. Returns whether anything changed.
    pub fn undo_voxel(&mut self) -> bool {
        if self.voxel3d.is_none() {
            self.set_status(t("voxel3d.status_nothing_to_undo"), false);
            return false;
        }
        let Some(entry) = self.voxel_undo.take_undo() else {
            self.set_status(t("voxel3d.status_nothing_to_undo"), false);
            return false;
        };
        let bricks: Vec<u32> = entry.voxels.iter().map(|(brick, _)| *brick).collect();
        let current = self
            .voxel3d
            .as_ref()
            .and_then(|engine| engine.read_brick_snapshot(&bricks));
        if let Some(engine) = self.voxel3d.as_mut() {
            engine.restore_brick_snapshot(&entry);
        }
        if let Some(current) = current {
            self.voxel_undo.push_redo(current);
        }
        self.clear_voxel_pointer_state();
        self.set_status(t("voxel3d.status_undo"), false);
        true
    }

    /// Reapplies the newest undone 3D snapshot. Returns whether anything
    /// changed.
    pub fn redo_voxel(&mut self) -> bool {
        if self.voxel3d.is_none() {
            self.set_status(t("voxel3d.status_nothing_to_redo"), false);
            return false;
        }
        let Some(entry) = self.voxel_undo.take_redo() else {
            self.set_status(t("voxel3d.status_nothing_to_redo"), false);
            return false;
        };
        let bricks: Vec<u32> = entry.voxels.iter().map(|(brick, _)| *brick).collect();
        let current = self
            .voxel3d
            .as_ref()
            .and_then(|engine| engine.read_brick_snapshot(&bricks));
        if let Some(engine) = self.voxel3d.as_mut() {
            engine.restore_brick_snapshot(&entry);
        }
        if let Some(current) = current {
            self.voxel_undo.restore_undo(current);
        }
        self.clear_voxel_pointer_state();
        self.set_status(t("voxel3d.status_redo"), false);
        true
    }

    /// Saves the 3D world as `<name>.tpt3` under [`SAVES_DIR`].
    pub fn save_voxel(&mut self, name: &str) -> Result<String, String> {
        let stem = normalize_voxel_save_stem(name)?;
        let saves_dir = Path::new(SAVES_DIR);
        std::fs::create_dir_all(saves_dir)
            .map_err(|err| t_fmt("app.error_create_saves_dir", &[SAVES_DIR, &err.to_string()]))?;

        let engine = self
            .voxel3d
            .as_ref()
            .ok_or_else(|| t("voxel3d.error_no_engine").to_string())?;
        let voxels = engine
            .read_world_words()
            .ok_or_else(|| t("voxel3d.error_read_world").to_string())?;
        let air = engine.read_air_fields().unwrap_or_default();
        let (aux_ctype, aux_tmp, aux_tmp2, aux_tmp3) = engine
            .read_aux_words()
            .ok_or_else(|| t("voxel3d.error_read_world").to_string())?;
        let mut bodies = Vec::new();
        for index in 0..engine.rigid.body_count as usize {
            let Some(body) = engine.rigid.body(index) else {
                continue;
            };
            bodies.push(Voxel3dBodySave {
                pos: body.pos,
                quat: body.quat,
                lin_vel: body.lin_vel,
                ang_vel: body.ang_vel,
                half_extents: body.half_extents.map(|value| value as f32),
                ty: body.ty,
                local_voxels: body.local.clone(),
            });
        }

        let data = Voxel3dSaveData::with_aux(
            engine.config,
            voxels,
            aux_ctype,
            aux_tmp,
            aux_tmp2,
            aux_tmp3,
            air,
            bodies,
        );
        let path = saves_dir.join(format!("{stem}.{VOXEL_SAVE_EXTENSION}"));
        data.save_to_path(&path)?;

        let target = format!("{stem}.{VOXEL_SAVE_EXTENSION}");
        let message = t_fmt("voxel3d.status_saved", &[&target]);
        self.voxel_save_name_input = stem;
        self.refresh_voxel_saves();
        Ok(message)
    }

    /// Saves using [`AppState::voxel_save_name_input`], reporting through the
    /// HUD status line.
    pub fn save_voxel_from_input(&mut self) -> Result<String, String> {
        let name = self.voxel_save_name_input.clone();
        match self.save_voxel(&name) {
            Ok(message) => {
                self.set_status(message.clone(), false);
                Ok(message)
            }
            Err(err) => {
                self.set_status(err.clone(), true);
                Err(err)
            }
        }
    }

    /// Loads a `.tpt3` save, restores its bodies and enters the 3D mode.
    pub fn load_voxel_file(&mut self, path: &Path) -> Result<(), String> {
        let data = Voxel3dSaveData::load_from_path(path)?;
        // Keep the Options page in sync with the world that was just loaded;
        // otherwise changing an unrelated preference would push the local
        // defaults back into the engine.
        self.prefs.voxel_render_scale =
            crate::prefs::snap_voxel_render_scale(data.config.render_scale);
        self.prefs.voxel_ao_rays = crate::prefs::snap_voxel_ao_rays(data.config.ao_rays);
        self.prefs.voxel_shadows = data.config.shadows;
        self.prefs.voxel_air_iterations = data.config.air_iterations.clamp(
            crate::prefs::MIN_VOXEL_AIR_ITERATIONS,
            crate::prefs::MAX_VOXEL_AIR_ITERATIONS,
        );
        let engine = self.ensure_voxel_engine();
        engine.config = data.config;
        engine.load_world_with_aux(
            &data.voxels,
            &data.aux_ctype,
            &data.aux_tmp,
            &data.aux_tmp2,
            &data.aux_tmp3,
            &data.air,
        )?;

        for body in &data.rigid_bodies {
            let half = [
                (body.half_extents[0].round() as i64).clamp(1, 16) as u32,
                (body.half_extents[1].round() as i64).clamp(1, 16) as u32,
                (body.half_extents[2].round() as i64).clamp(1, 16) as u32,
            ];
            if engine.rigid.body_count as usize >= MAX_BODIES {
                break;
            }
            engine.spawn_box(body.pos, half, body.ty);
        }

        self.sync_voxel_render_settings();

        self.mode = GameMode::Sandbox3D;
        self.prefs.game_mode = GameMode::Sandbox3D;
        self.screen = AppScreen::InGame;
        self.last_sim_step_instant = std::time::Instant::now();
        self.sim_step_debt = 0.0;
        self.voxel_camera = default_voxel_camera();
        self.voxel_edge_mode = data.config.edge_mode;
        self.voxel_undo.clear();
        self.clear_voxel_pointer_state();
        self.voxel_active_bricks = None;
        self.voxel_stats_at = std::time::Instant::now();
        self.voxel_save_name_input = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("sandbox3d")
            .to_string();
        self.prefs.save();

        let message = t_fmt("voxel3d.status_loaded", &[&path.display().to_string()]);
        self.set_status(message, false);
        Ok(())
    }

    /// Loads a save for the main-menu browser, reporting failures through the
    /// HUD instead of propagating them.
    pub fn load_voxel_save_file(&mut self, path: &Path) -> bool {
        match self.load_voxel_file(path) {
            Ok(()) => true,
            Err(err) => {
                self.set_status(err, true);
                false
            }
        }
    }

    /// Rescans `./saves` for `.tpt3` files.
    pub fn refresh_voxel_saves(&mut self) {
        let mut saves = Vec::new();
        if let Ok(entries) = std::fs::read_dir(SAVES_DIR) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file()
                    && path.extension().and_then(|ext| ext.to_str()) == Some(VOXEL_SAVE_EXTENSION)
                {
                    saves.push(path);
                }
            }
        }
        saves.sort();
        self.voxel_saves = saves;
    }
}

/// Why [`collect_fill_region`] refused a seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FillRegionError {
    /// The seed voxel is not empty.
    Solid,
    /// The seed lies outside the simulation cube.
    OutOfBounds,
    /// The connected region exceeded [`MAX_FILL_VOXELS`].
    TooLarge,
}

/// Collects the empty 6-connected region that contains `seed`.
///
/// `world` is the packed `voxel_a` word array (`0xFF` is the element id), and
/// `dim` is the active cube edge. The search is bounded, so a click in the
/// empty sky refuses the fill instead of walking all 16.7 M voxels; every
/// coordinate is checked against `dim` before it is visited.
fn collect_fill_region(
    world: &[u32],
    dim: u32,
    seed: [u32; 3],
) -> Result<Vec<u32>, FillRegionError> {
    if seed.iter().any(|&coord| coord >= dim) {
        return Err(FillRegionError::OutOfBounds);
    }
    let seed_index = (seed[1] << 16) | (seed[2] << 8) | seed[0];
    if world
        .get(seed_index as usize)
        .is_none_or(|word| word & 0xFF != 0)
    {
        return Err(FillRegionError::Solid);
    }

    let mut visited = vec![0u64; world.len().div_ceil(64)];
    let mut stack = vec![seed_index];
    visited[(seed_index / 64) as usize] |= 1u64 << (seed_index % 64);
    let mut indices = Vec::new();

    while let Some(index) = stack.pop() {
        if indices.len() >= MAX_FILL_VOXELS {
            return Err(FillRegionError::TooLarge);
        }
        let x = index & 0xFF;
        let y = (index >> 16) & 0xFF;
        let z = (index >> 8) & 0xFF;
        indices.push(index);

        let mut try_push = |neighbour: u32| {
            let word = (neighbour / 64) as usize;
            let mask = 1u64 << (neighbour % 64);
            if visited[word] & mask == 0
                && world
                    .get(neighbour as usize)
                    .is_some_and(|word| word & 0xFF == 0)
            {
                visited[word] |= mask;
                stack.push(neighbour);
            }
        };
        if x > 0 {
            try_push(index - 1);
        }
        if x + 1 < dim {
            try_push(index + 1);
        }
        if z > 0 {
            try_push(index - 256);
        }
        if z + 1 < dim {
            try_push(index + 256);
        }
        if y > 0 {
            try_push(index - 65_536);
        }
        if y + 1 < dim {
            try_push(index + 65_536);
        }
    }

    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_core::voxel3d::VOXEL_COUNT;

    #[test]
    fn flood_region_stops_at_walls_and_world_edges() {
        let mut world = vec![0u32; VOXEL_COUNT];
        let index = |x: u32, y: u32, z: u32| ((y << 16) | (z << 8) | x) as usize;
        for y in 0..8 {
            for z in 0..8 {
                world[index(3, y, z)] = PT_STNE;
            }
        }

        let region = collect_fill_region(&world, 8, [1, 1, 1]).expect("open region");
        assert_eq!(region.len(), 3 * 8 * 8, "the wall must stop the fill");
        assert!(
            region.iter().all(|&voxel| voxel & 0xFF < 3
                && (voxel >> 16) & 0xFF < 8
                && (voxel >> 8) & 0xFF < 8),
            "region escaped the wall or the 8^3 cube"
        );

        assert_eq!(
            collect_fill_region(&world, 8, [3, 1, 1]),
            Err(FillRegionError::Solid)
        );
        assert_eq!(
            collect_fill_region(&world, 8, [8, 0, 0]),
            Err(FillRegionError::OutOfBounds)
        );
    }

    #[test]
    fn flood_region_rejects_a_solid_seed() {
        let mut world = vec![0u32; VOXEL_COUNT];
        world[0] = PT_STNE;
        assert_eq!(
            collect_fill_region(&world, 4, [0, 0, 0]),
            Err(FillRegionError::Solid)
        );
    }

    #[test]
    fn game_mode_keys_round_trip() {
        for mode in [GameMode::Powder2D, GameMode::Sandbox3D] {
            assert_eq!(GameMode::from_key(mode.key()), mode);
        }
        assert_eq!(GameMode::from_key("unknown"), GameMode::Powder2D);
    }

    #[test]
    fn floor_sphere_covers_exactly_three_voxel_layers() {
        let brush = Brush3d::new([128.0, FLOOR_SPHERE_Y, 128.0], FLOOR_SPHERE_RADIUS);
        let (min, max) = brush.bounds();
        assert_eq!(min, [0, 0, 0], "the floor must span the whole footprint");
        assert_eq!(max, [255, 2, 255], "the clipped sphere must end at layer 2");
        assert!(brush.contains(128, 2, 128));
        assert!(!brush.contains(128, 3, 128));
    }

    #[test]
    fn undo_history_is_bounded_and_clears_redo_on_new_edits() {
        let mut history = Voxel3dUndoHistory::default();
        let snapshot = |brick: u32| Voxel3dBrickSnapshot {
            voxels: vec![(brick, vec![0; 512])],
            ..Default::default()
        };
        for value in 0..(VOXEL_UNDO_LIMIT + 5) {
            history.push_undo(snapshot(value as u32));
        }
        assert_eq!(history.undo.len(), VOXEL_UNDO_LIMIT);
        let entry = history.take_undo().expect("newest entry");
        assert_eq!(entry.voxels[0].0, (VOXEL_UNDO_LIMIT + 4) as u32);
        history.push_redo(snapshot(1));
        assert!(history.can_redo());
        history.push_undo(snapshot(2));
        assert!(!history.can_redo(), "a new edit drops the redo branch");
    }
}
