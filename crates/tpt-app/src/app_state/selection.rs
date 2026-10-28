//! Region selection, copy, cut, paste and stamps.
//!
//! Ports `GameView::OnKeyDown` (`SDL_SCANCODE_C`/`X`/`V`,
//! `GameView.cpp:1548-1585`), the drag that ends in `CopyRegion`/`CutRegion`/
//! `StampRegion` (`GameView.cpp:1234-1266`) and `GameController::LoadClipboard`
//! (`GameController.cpp:421-434`). TPT keeps the clipboard as a `GameSave`; the
//! port stores the same data in [`Region`], which is also what a Ctrl+S stamp
//! is captured from (`GameController::StampRegion`,
//! `GameController.cpp:476-493`).
//!
//! Every pasting action snapshots the canvas first (`GameController::PlaceSave`,
//! `GameController.cpp:253-263`) and a cut snapshots before it erases
//! (`GameController::CutRegion`, `GameController.cpp:514-516`), so Ctrl+Z
//! always returns to the canvas as it was before the action.

use tpt_core::commands::{RegionAir, SimCommand};

use tpt_core::constants::{CELL, PT_NONE, WL_ERASE};
use tpt_core::particle::{GpuParticle, SimulationConfig};
use tpt_core::sign::{MAX_SIGNS, Sign};
use tpt_save::{SAVE_VERSION, SaveData};

/// Magic header of a region written to the platform clipboard.
pub const CLIPBOARD_MAGIC: &str = "TPTRS-REGION-1\n";

/// What the next drag on the canvas is for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelectMode {
    /// Normal drawing.
    #[default]
    None,
    /// Rubber band, then copy the region (`Ctrl+C`).
    Copy,
    /// Rubber band, then copy and erase the region (`Ctrl+X`).
    Cut,
    /// Rubber band, then save the region as a stamp (`Ctrl+S`).
    Stamp,
}

/// A rectangle of the canvas captured by copy, cut or a Ctrl+S stamp.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Region {
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
    /// Live particles with region-relative coordinates.
    pub particles: Vec<GpuParticle>,
    /// Wall ids, row-major, `0` for none.
    pub walls: Vec<u32>,
    /// Air grids of the cells covering the rectangle, region-relative.
    ///
    /// Empty when the region was captured with `Simulation.IncludePressure`
    /// off (`Simulation.cpp:435-444`); the payload then carries no air at all.
    pub air: RegionAir,
    /// Signs anchored inside the rectangle, with region-relative anchors
    /// (`Simulation.cpp:414-424`).
    pub signs: Vec<Sign>,
}

impl Region {
    /// Whether the capture holds nothing to place.
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
            && self.walls.iter().all(|&w| w == 0)
            && self.air.is_empty()
            && self.signs.is_empty()
    }

    /// Serialized payload written to the platform clipboard.
    ///
    /// The same `Region` that the in-process clipboard keeps is written as a
    /// short header plus JSON, so another running instance can paste it
    /// (`OptionsView.cpp:413-415`, `Clipboard::SetClipboardData`). JSON matches
    /// the rest of the front-end's persistence and keeps the payload inspectable.
    pub fn to_clipboard_text(&self) -> String {
        match serde_json::to_string(self) {
            Ok(json) => format!("{CLIPBOARD_MAGIC}{json}"),
            Err(_) => String::new(),
        }
    }

    /// Parses a platform-clipboard payload produced by [`Self::to_clipboard_text`].
    ///
    /// Returns `None` for unrelated clipboard text, so a user pasting a URL or
    /// code snippet into the canvas falls back to the in-process clipboard
    /// instead of clearing it.
    pub fn from_clipboard_text(text: &str) -> Option<Self> {
        let json = text.strip_prefix(CLIPBOARD_MAGIC)?;
        serde_json::from_str(json).ok()
    }

    /// Rotates the region 90 degrees counter-clockwise.
    ///
    /// Ports `GameSave::Transform(Mat2<int>::CCW)` (`GameSave.cpp:280-370`,
    /// driven by `R` in `GameView::OnKeyPress`, `GameView.cpp:1393-1404`).
    /// Pixel `(x, y)` becomes `(y, w - 1 - x)`, the wall and air grids rotate
    /// with it, and particle / sign velocities follow the same matrix.
    pub fn rotate_ccw(&self) -> Self {
        if self.w == 0 || self.h == 0 {
            return Self::default();
        }
        let w = self.w;
        let h = self.h;
        let mut particles = Vec::with_capacity(self.particles.len());
        for particle in &self.particles {
            let mut particle = *particle;
            let (x, y) = (particle.x, particle.y);
            particle.x = y;
            particle.y = (w as f32 - 1.0) - x;
            std::mem::swap(&mut particle.vx, &mut particle.vy);
            particle.vy = -particle.vy;
            particles.push(particle);
        }
        // The rotated grid is `h` wide and `w` tall.
        let mut walls = vec![0u32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let wall = self.walls[(y * w + x) as usize];
                if wall != 0 {
                    let nx = y;
                    let ny = w - 1 - x;
                    walls[(ny * h + nx) as usize] = wall;
                }
            }
        }
        let mut signs = Vec::with_capacity(self.signs.len());
        for sign in &self.signs {
            let mut sign = sign.clone();
            let (x, y) = (sign.x, sign.y);
            sign.x = y;
            sign.y = w as i32 - 1 - x;
            if sign.x >= 0 && sign.y >= 0 && sign.x < h as i32 && sign.y < w as i32 {
                signs.push(sign);
            }
        }
        Self {
            w: h,
            h: w,
            particles,
            walls,
            air: transform_air_cells(&self.air, true, false, false),
            signs,
        }
    }

    /// Mirrors the region horizontally (`GameSave::Transform(MirrorX)`,
    /// `GameView.cpp:1395-1398`): pixel `x` becomes `w - 1 - x`.
    pub fn mirror_horizontal(&self) -> Self {
        if self.w == 0 || self.h == 0 {
            return Self::default();
        }
        let w = self.w;
        let mut particles = self.particles.clone();
        for particle in &mut particles {
            particle.x = (w as f32 - 1.0) - particle.x;
            particle.vx = -particle.vx;
        }
        let mut walls = vec![0u32; self.walls.len()];
        for y in 0..self.h {
            for x in 0..w {
                let wall = self.walls[(y * w + x) as usize];
                if wall != 0 {
                    walls[(y * w + (w - 1 - x)) as usize] = wall;
                }
            }
        }
        let mut signs = Vec::with_capacity(self.signs.len());
        for sign in &self.signs {
            let mut sign = sign.clone();
            sign.x = w as i32 - 1 - sign.x;
            if sign.x >= 0 && sign.x < w as i32 {
                signs.push(sign);
            }
        }
        Self {
            w: self.w,
            h: self.h,
            particles,
            walls,
            air: transform_air_cells(&self.air, false, true, false),
            signs,
        }
    }

    /// Mirrors the region vertically (`GameSave::Transform(MirrorY)`,
    /// `GameView.cpp:1389-1392`): pixel `y` becomes `h - 1 - y`.
    pub fn mirror_vertical(&self) -> Self {
        if self.w == 0 || self.h == 0 {
            return Self::default();
        }
        let w = self.w;
        let h = self.h;
        let mut particles = self.particles.clone();
        for particle in &mut particles {
            particle.y = (h as f32 - 1.0) - particle.y;
            particle.vy = -particle.vy;
        }
        let mut walls = vec![0u32; self.walls.len()];
        for y in 0..h {
            for x in 0..w {
                let wall = self.walls[(y * w + x) as usize];
                if wall != 0 {
                    walls[((h - 1 - y) * w + x) as usize] = wall;
                }
            }
        }
        let mut signs = Vec::with_capacity(self.signs.len());
        for sign in &self.signs {
            let mut sign = sign.clone();
            sign.y = h as i32 - 1 - sign.y;
            if sign.y >= 0 && sign.y < h as i32 {
                signs.push(sign);
            }
        }
        Self {
            w: self.w,
            h: self.h,
            particles,
            walls,
            air: transform_air_cells(&self.air, false, false, true),
            signs,
        }
    }

    /// Crops `snapshot` to the inclusive pixel rectangle `x0..=x1, y0..=y1`
    /// and offsets every particle and sign into region-relative coordinates.
    ///
    /// `Simulation::Save` (`Simulation.cpp:341-445`) copies the particles,
    /// signs, wall cells and (with `includePressure`) air grids of the
    /// requested rectangle into a `GameSave`; the port keeps pixel addressing
    /// because [`Placement`] places pixels. Wall cells are looked up on the
    /// `CELL`-sized grid of the snapshot, which is the grid
    /// `SimBuffers::wall_bmap` uses. The air crop follows the reference's
    /// `RectBetween(partR.TopLeft()/CELL, partR.BottomRight()/CELL)`
    /// (`Simulation.cpp:343`), so it covers every cell the rectangle
    /// touches, up to three pixels of border on each side.
    pub fn capture(snapshot: &SaveData, x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        let w = (x1 - x0 + 1).max(0) as u32;
        let h = (y1 - y0 + 1).max(0) as u32;
        let xres = snapshot.config.xres as i32;
        let yres = snapshot.config.yres as i32;
        let xcells = snapshot.config.xcells().max(1);
        let mut region = Region {
            w,
            h,
            particles: Vec::new(),
            walls: vec![0u32; (w * h) as usize],
            air: capture_air(snapshot, x0, y0, x1, y1),
            signs: Vec::new(),
        };
        for particle in &snapshot.particles {
            let mut particle = *particle;
            // `Simulation.cpp:362-363`: `int(parts[i].x + 0.5f)` decides
            // membership; the fractional position itself is kept, as the
            // reference subtracts the block offset from the live float.
            let px = (particle.x + 0.5).floor() as i32;
            let py = (particle.y + 0.5).floor() as i32;
            if px < x0 || px > x1 || py < y0 || py > y1 {
                continue;
            }
            particle.x -= x0 as f32;
            particle.y -= y0 as f32;
            region.particles.push(particle);
        }
        for ry in 0..h as i32 {
            for rx in 0..w as i32 {
                let px = (x0 + rx).clamp(0, xres.max(1) - 1) as u32;
                let py = (y0 + ry).clamp(0, yres.max(1) - 1) as u32;
                let cell = py.div_euclid(CELL) * xcells + px.div_euclid(CELL);
                if let Some(&wall) = snapshot.wall_bmap.get(cell as usize) {
                    region.walls[(ry as u32 * w + rx as u32) as usize] = wall;
                }
            }
        }
        // `Simulation.cpp:414-424`: only signs with text whose anchor is inside
        // the rectangle travel, offset into region coordinates.
        for sign in snapshot.signs.iter().take(MAX_SIGNS) {
            if sign.text.is_empty() || sign.x < x0 || sign.x > x1 || sign.y < y0 || sign.y > y1 {
                continue;
            }
            let mut sign = sign.clone();
            sign.x -= x0;
            sign.y -= y0;
            region.signs.push(sign);
        }
        region
    }

    /// Builds the standalone `SaveData` payload a stamp file stores.
    ///
    /// The payload's canvas is the region itself, so its air grids are fitted
    /// to the cells of `w x h` (`Simulation::Save` sizes the `GameSave` to the
    /// saved rectangle, `GameSave.cpp:242-260`). Regions whose capture was not
    /// `CELL`-aligned lose at most the extra border cell, the same
    /// quantisation the reference's block-offset save has.
    pub fn to_save_data(&self) -> SaveData {
        let config = SimulationConfig {
            xres: self.w.max(1),
            yres: self.h.max(1),
            ..SimulationConfig::default()
        };
        let xcells = config.xcells();
        let ycells = config.ycells();
        let (air_pv, air_vx, air_vy, air_hv) = self.canonical_air(xcells, ycells);
        SaveData {
            version: SAVE_VERSION,
            config,
            particles: self.particles.clone(),
            air_vx,
            air_vy,
            air_pv,
            air_hv,
            fan_vx: Vec::new(),
            fan_vy: Vec::new(),
            // The region keeps one wall id per pixel; the stamp payload keeps
            // that pixel grid verbatim so a decode reproduces every cell.
            wall_bmap: self.walls.clone(),
            wall_emap: Vec::new(),
            rigid_bodies: Vec::new(),
            signs: self.signs.clone(),
        }
    }

    /// Rebuilds a region from a standalone payload (a `.stm` stamp file).
    ///
    /// The wall grid is resized to `w * h` so a hand-written file can never
    /// make [`Placement`] index past the end of its wall vector.
    pub fn from_save_data(save: &SaveData) -> Self {
        let w = save.config.xres;
        let h = save.config.yres;
        let mut walls = save.wall_bmap.clone();
        walls.resize((w as usize) * (h as usize), 0);
        let xcells = save.config.xcells();
        let ycells = save.config.ycells();
        let cells = save.config.num_cells();
        // A stamp written before region air existed (or one captured with
        // `IncludePressure` off) has empty grids; the region then carries no
        // air and placement leaves the destination air alone.
        let air = if cells > 0
            && save.air_pv.len() == cells
            && save.air_vx.len() == cells
            && save.air_vy.len() == cells
            && save.air_hv.len() == cells
        {
            RegionAir {
                wcells: xcells,
                hcells: ycells,
                pv: save.air_pv.clone(),
                vx: save.air_vx.clone(),
                vy: save.air_vy.clone(),
                hv: save.air_hv.clone(),
            }
        } else {
            RegionAir::default()
        };
        Region {
            w,
            h,
            particles: save.particles.clone(),
            walls,
            air,
            signs: save.signs.iter().take(MAX_SIGNS).cloned().collect(),
        }
    }

    /// Copies the whole region air grid into a `wcells x hcells` grid,
    /// dropping the extra border cell of an unaligned capture and zero-filling
    /// the cells a smaller capture does not reach.
    fn canonical_air(&self, xcells: u32, ycells: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let cells = (xcells as usize) * (ycells as usize);
        if cells == 0 || self.air.is_empty() || !self.air.is_complete() {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }
        let mut pv = vec![0.0; cells];
        let mut vx = vec![0.0; cells];
        let mut vy = vec![0.0; cells];
        let mut hv = vec![0.0; cells];
        let cols = self.air.wcells.min(xcells);
        let rows = self.air.hcells.min(ycells);
        for cy in 0..rows {
            for cx in 0..cols {
                let src = (cy * self.air.wcells + cx) as usize;
                let dst = (cy * xcells + cx) as usize;
                pv[dst] = self.air.pv[src];
                vx[dst] = self.air.vx[src];
                vy[dst] = self.air.vy[src];
                hv[dst] = self.air.hv[src];
            }
        }
        (pv, vx, vy, hv)
    }
}

/// Applies a quarter-turn or a mirror to a region air grid.
///
/// `swap` rotates the cells 90 degrees counter-clockwise (`(cx, cy)` becomes
/// `(cy, wcells - 1 - cx)`), `flip_x` / `flip_y` mirror one axis. The velocity
/// components rotate with the grid, exactly like the particle velocities in
/// `GameSave::Transform` (`GameSave.cpp:355-365`). Incomplete grids are
/// dropped, mirroring [`capture_air`]'s "no pressure data" case.
fn transform_air_cells(air: &RegionAir, swap: bool, flip_x: bool, flip_y: bool) -> RegionAir {
    if air.is_empty() || !air.is_complete() {
        return RegionAir::default();
    }
    let wcells = if swap { air.hcells } else { air.wcells };
    let hcells = if swap { air.wcells } else { air.hcells };
    let len = (wcells as usize) * (hcells as usize);
    let mut out = RegionAir {
        wcells,
        hcells,
        pv: vec![0.0; len],
        vx: vec![0.0; len],
        vy: vec![0.0; len],
        hv: vec![0.0; len],
    };
    for cy in 0..air.hcells {
        for cx in 0..air.wcells {
            let (nx, ny) = if swap {
                (cy, air.wcells - 1 - cx)
            } else if flip_x {
                (air.wcells - 1 - cx, cy)
            } else {
                (cx, air.hcells - 1 - cy)
            };
            let src = (cy * air.wcells + cx) as usize;
            let dst = (ny * wcells + nx) as usize;
            out.pv[dst] = air.pv[src];
            out.hv[dst] = air.hv[src];
            let (mut vx, mut vy) = (air.vx[src], air.vy[src]);
            if swap {
                std::mem::swap(&mut vx, &mut vy);
                vy = -vy;
            } else if flip_x {
                vx = -vx;
            } else if flip_y {
                vy = -vy;
            }
            out.vx[dst] = vx;
            out.vy[dst] = vy;
        }
    }
    out
}

/// Crops the four air grids of `snapshot` to the cells covering the inclusive
/// pixel rectangle.
///
/// Returns an empty [`RegionAir`] when the snapshot carries no full air grids,
/// which is exactly a capture taken with `Simulation.IncludePressure` off
/// (`Simulation.cpp:435-444`, `OptionsView.cpp:401`).
fn capture_air(snapshot: &SaveData, x0: i32, y0: i32, x1: i32, y1: i32) -> RegionAir {
    let cells = snapshot.config.num_cells();
    if x1 < x0
        || y1 < y0
        || cells == 0
        || snapshot.air_pv.len() != cells
        || snapshot.air_vx.len() != cells
        || snapshot.air_vy.len() != cells
        || snapshot.air_hv.len() != cells
    {
        return RegionAir::default();
    }
    let xcells = snapshot.config.xcells().max(1) as i32;
    let ycells = snapshot.config.ycells().max(1) as i32;
    let cx0 = x0.div_euclid(CELL as i32).clamp(0, xcells - 1);
    let cx1 = x1.div_euclid(CELL as i32).clamp(0, xcells - 1);
    let cy0 = y0.div_euclid(CELL as i32).clamp(0, ycells - 1);
    let cy1 = y1.div_euclid(CELL as i32).clamp(0, ycells - 1);
    let wcells = (cx1 - cx0 + 1) as u32;
    let hcells = (cy1 - cy0 + 1) as u32;
    let mut air = RegionAir {
        wcells,
        hcells,
        pv: Vec::with_capacity((wcells * hcells) as usize),
        vx: Vec::with_capacity((wcells * hcells) as usize),
        vy: Vec::with_capacity((wcells * hcells) as usize),
        hv: Vec::with_capacity((wcells * hcells) as usize),
    };
    for cy in cy0..=cy1 {
        for cx in cx0..=cx1 {
            let idx = cy as usize * xcells as usize + cx as usize;
            air.pv.push(snapshot.air_pv[idx]);
            air.vx.push(snapshot.air_vx[idx]);
            air.vy.push(snapshot.air_vy[idx]);
            air.hv.push(snapshot.air_hv[idx]);
        }
    }
    air
}

/// A clipboard region being placed under the cursor.
#[derive(Clone, Debug)]
pub struct Placement {
    /// Captured contents.
    pub region: Region,
    /// Canvas pixel the region's top-left corner follows.
    pub anchor: (i32, i32),
}

impl Placement {
    /// Moves the pending region by `(dx, dy)` simulation pixels
    /// (`GameView::TranslateSave`, `GameView.cpp:1975-1979`).
    pub fn translate(&mut self, dx: i32, dy: i32) {
        self.anchor = (self.anchor.0 + dx, self.anchor.1 + dy);
    }

    /// Rotates the pending region 90 degrees counter-clockwise
    /// (`GameView::TransformSave(Mat2::CCW)`).
    pub fn rotate_ccw(&mut self) {
        self.region = self.region.rotate_ccw();
    }

    /// Mirrors the pending region horizontally (`Mat2::MirrorX`).
    pub fn mirror_horizontal(&mut self) {
        self.region = self.region.mirror_horizontal();
    }

    /// Mirrors the pending region vertically (`Mat2::MirrorY`).
    pub fn mirror_vertical(&mut self) {
        self.region = self.region.mirror_vertical();
    }
}

impl super::AppState {
    /// `Ctrl+C` / `Ctrl+X`: arm the rubber band and drop any placement.
    pub fn begin_selection(&mut self, mode: SelectMode) {
        self.select_mode = mode;
        self.select_start = None;
        self.select_end = None;
        self.placement = None;
    }

    /// Abandons the pending selection (`right click = cancel`).
    pub fn cancel_selection(&mut self) {
        self.select_mode = SelectMode::None;
        self.select_start = None;
        self.select_end = None;
    }

    /// Normalised `(x0, y0, x1, y1)` pixel rectangle clamped to the canvas.
    pub fn selection_rect(&self) -> Option<(i32, i32, i32, i32)> {
        let (ax, ay) = self.select_start?;
        let (bx, by) = self.select_end?;
        let (xres, yres) = self.canvas_size();
        let x0 = ax.min(bx).clamp(0, xres as i32 - 1);
        let y0 = ay.min(by).clamp(0, yres as i32 - 1);
        let x1 = ax.max(bx).clamp(0, xres as i32 - 1);
        let y1 = ay.max(by).clamp(0, yres as i32 - 1);
        Some((x0, y0, x1, y1))
    }

    /// Captures the rectangle into the clipboard, erases it for a cut, or
    /// writes it as a stamp (`Ctrl+S`).
    ///
    /// The capture goes through [`tpt_save::SaveData::capture_from_with_pressure`],
    /// the same bounded read-back the save and undo paths use. It honours
    /// `Simulation.IncludePressure` like `GameController::CopyRegion`
    /// (`GameController.cpp:496`): with the option off the clipboard carries no
    /// air grids.
    ///
    /// Returns whether the capture replaced the in-process clipboard, so the
    /// caller can mirror it to the platform clipboard when
    /// `NativeClipboard.Enabled` is on.
    pub fn finish_selection(&mut self) -> bool {
        let Some((x0, y0, x1, y1)) = self.selection_rect() else {
            self.cancel_selection();
            return false;
        };
        let mode = self.select_mode;
        let snapshot =
            SaveData::capture_from_with_pressure(&mut self.engine, self.prefs.include_pressure);
        let region = Region::capture(&snapshot, x0, y0, x1, y1);
        let mut stored = false;

        match mode {
            SelectMode::Cut => {
                // `GameController::CutRegion` (`GameController.cpp:514-516`)
                // copies the region, pushes a history snapshot and then clears
                // the area, so Ctrl+Z restores the cut region.
                self.push_undo_snapshot();
                let mut points = Vec::with_capacity((region.w * region.h) as usize);
                for py in y0..=y1 {
                    for px in x0..=x1 {
                        points.push((px, py));
                    }
                }
                self.erase_region(&points);
                if !region.is_empty() {
                    self.clipboard = Some(region);
                    stored = true;
                }
            }
            SelectMode::Copy => {
                if !region.is_empty() {
                    self.clipboard = Some(region);
                    stored = true;
                }
            }
            SelectMode::Stamp => {
                // `GameController::StampRegion` (`GameController.cpp:476-493`):
                // the rectangle becomes a `.stm` file; the canvas is untouched.
                let _ = self.save_stamp_region(&region);
            }
            SelectMode::None => {}
        }

        self.cancel_selection();
        stored
    }

    /// `Ctrl+V`: start placing the clipboard at the cursor.
    pub fn begin_paste(&mut self) -> bool {
        let Some(region) = self.clipboard.clone() else {
            return false;
        };
        let anchor = self.hover_pos.unwrap_or((0, 0));
        self.placement = Some(Placement { region, anchor });
        true
    }

    /// Moves the pending placement to `pos`.
    pub fn move_placement(&mut self, pos: (i32, i32)) {
        if let Some(placement) = &mut self.placement {
            placement.anchor = pos;
        }
    }

    /// Places the clipboard contents and clears the pending placement.
    ///
    /// Mirrors `GameController::PlaceSave` (`GameController.cpp:253-263`): the
    /// canvas is snapshotted first, then [`SimCommand::PlaceRegion`] clears the
    /// destination rectangle and stamps the particles, walls, air grids and
    /// signs (`Simulation::Load`, `Simulation.cpp:49-340`).
    pub fn commit_placement(&mut self) {
        let Some(placement) = self.placement.take() else {
            return;
        };
        // `GameController::PlaceSave` pushes a `HistorySnapshot` before the
        // save / stamp is loaded into the simulation, so Ctrl+Z removes the
        // whole placement instead of the canvas losing it for good
        // (`GameController.cpp:258`).
        self.push_undo_snapshot();
        let region = placement.region;
        let (ax, ay) = placement.anchor;
        // `OptionsView.cpp:401` / `Simulation::Load`: the air cells are only
        // written back when the preference is enabled at placement time. The
        // capture already dropped them when it was off, so this also covers
        // turning the option off between copying and pasting.
        let air = if self.prefs.include_pressure {
            region.air
        } else {
            RegionAir::default()
        };
        self.engine.apply_command(SimCommand::PlaceRegion {
            x: ax,
            y: ay,
            w: region.w,
            h: region.h,
            particles: region.particles,
            walls: region.walls,
            air,
            signs: region.signs,
        });
    }

    /// Erases particles, walls and sign anchors under `points`, like the
    /// `PT_NONE` brush (`GameController::EraseRegion`).
    fn erase_region(&mut self, points: &[(i32, i32)]) {
        if points.is_empty() {
            return;
        }
        self.erase_signs(points);
        self.engine.apply_command(SimCommand::PlaceParticles {
            points: points.to_vec(),
            elem_ty: PT_NONE,
            replace_mode: true,
            dcolour: 0,
        });
        let mut cells: Vec<(i32, i32)> = Vec::with_capacity(points.len());
        for &(x, y) in points {
            let cell = (x.div_euclid(CELL as i32), y.div_euclid(CELL as i32));
            if !cells.contains(&cell) {
                cells.push(cell);
            }
        }
        self.engine.apply_command(SimCommand::PlaceWalls {
            cells,
            wall_id: WL_ERASE,
            fan_vel: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full-canvas snapshot carrying the four air grids, one sign inside the
    /// capture rectangle and one outside it.
    fn sample_snapshot() -> SaveData {
        let config = SimulationConfig::new(128, 96);
        let cells = config.num_cells();
        SaveData {
            version: SAVE_VERSION,
            config,
            particles: vec![GpuParticle {
                ty: 3,
                x: 5.0,
                y: 6.0,
                ..GpuParticle::default()
            }],
            air_vx: (0..cells).map(|i| -(i as f32) * 0.5).collect(),
            air_vy: (0..cells).map(|i| (i as f32) * 0.25).collect(),
            air_pv: (0..cells).map(|i| 1.0 + (i as f32) * 0.125).collect(),
            air_hv: (0..cells).map(|i| 293.0 + (i % 17) as f32).collect(),
            fan_vx: vec![0.0; cells],
            fan_vy: vec![0.0; cells],
            wall_bmap: vec![0; cells],
            wall_emap: vec![0; cells],
            rigid_bodies: Vec::new(),
            signs: vec![Sign::new(20, 30, "Hot zone"), Sign::new(100, 30, "Outside")],
        }
    }

    /// Copying a rectangle and rebuilding its stamp payload keeps the pressure,
    /// both velocity components, the ambient heat and the anchored sign
    /// (`Simulation::Save` / `Simulation::Load`, `Simulation.cpp:341-445` and
    /// `Simulation.cpp:49-340`), while a sign whose anchor is outside the
    /// rectangle stays behind.
    #[test]
    fn region_round_trip_preserves_air_and_signs() {
        let snapshot = sample_snapshot();
        let region = Region::capture(&snapshot, 0, 0, 63, 47);

        assert_eq!((region.w, region.h), (64, 48));
        assert!(region.air.is_complete());
        assert_eq!(region.air.pv.len(), 16 * 12);
        // The crop really copies the source cells, not zeros.
        assert_eq!(region.air.pv[0], 1.0);
        assert_eq!(region.air.pv[1], 1.125);
        assert_eq!(region.air.vx[1], -0.5);
        assert_eq!(region.signs.len(), 1);
        assert_eq!(region.signs[0].text, "Hot zone");

        let stamp = region.to_save_data();
        assert_eq!((stamp.config.xres, stamp.config.yres), (64, 48));
        // The stamp file path: serialize, decode, rebuild the region.
        let bytes = stamp.to_bytes().expect("serialize stamp payload");
        let decoded = SaveData::from_bytes(&bytes).expect("decode stamp payload");
        let back = Region::from_save_data(&decoded);

        assert_eq!(back.air.pv, region.air.pv);
        assert_eq!(back.air.vx, region.air.vx);
        assert_eq!(back.air.vy, region.air.vy);
        assert_eq!(back.air.hv, region.air.hv);
        assert_eq!(back.signs.len(), 1);
        assert_eq!(back.signs[0].text, "Hot zone");
        assert_eq!((back.signs[0].x, back.signs[0].y), (20, 30));
        assert_eq!(back.particles, region.particles);
    }

    /// `Simulation.IncludePressure` off (`OptionsView.cpp:401`) drops every air
    /// grid from the capture, and the stamp payload must not resurrect them.
    #[test]
    fn capture_without_pressure_drops_every_air_grid() {
        let snapshot = sample_snapshot().retain_air_pressure(false);
        let region = Region::capture(&snapshot, 0, 0, 63, 47);

        assert!(region.air.is_empty());
        assert!(region.air.pv.is_empty());
        assert!(region.air.vx.is_empty());
        assert!(region.air.vy.is_empty());
        assert!(region.air.hv.is_empty());

        let stamp = region.to_save_data();
        assert!(stamp.air_pv.is_empty());
        assert!(stamp.air_vx.is_empty());
        assert!(stamp.air_vy.is_empty());
        assert!(stamp.air_hv.is_empty());
        // Only the air grids are gated; signs and particles still travel
        // (`Simulation.cpp:435-444`).
        assert_eq!(region.signs.len(), 1);
        assert_eq!(region.particles.len(), 1);
    }

    /// A 4x3 region with one particle, a sign, two wall cells and a complete
    /// 2x2 air grid, enough to pin every coordinate of the transforms.
    fn sample_region() -> Region {
        let mut walls = vec![0u32; 4 * 3];
        walls[1] = 7; // (1, 0)
        walls[11] = 9; // (3, 2)
        Region {
            w: 4,
            h: 3,
            particles: vec![GpuParticle {
                ty: 3,
                x: 1.0,
                y: 0.0,
                vx: 2.0,
                vy: 3.0,
                ..GpuParticle::default()
            }],
            walls,
            air: RegionAir {
                wcells: 2,
                hcells: 2,
                pv: vec![10.0, 20.0, 30.0, 40.0],
                vx: vec![1.0, 2.0, 3.0, 4.0],
                vy: vec![5.0, 6.0, 7.0, 8.0],
                hv: vec![100.0, 200.0, 300.0, 400.0],
            },
            signs: vec![Sign::new(1, 0, "mark")],
        }
    }

    /// `GameView.cpp:1393-1404` / `GameSave::Transform` (`GameSave.cpp:280-370`):
    /// rotating the pending stamp swaps the region dimensions, moves the
    /// particle, wall, sign and air-cell anchors through the counter-clockwise
    /// matrix and rotates the velocity vectors with them.
    #[test]
    fn placement_rotate_ccw_transforms_every_layer() {
        let region = sample_region();
        let rotated = region.rotate_ccw();

        assert_eq!((rotated.w, rotated.h), (3, 4));
        // Pixel (x, y) -> (y, w - 1 - x).
        let particle = rotated.particles[0];
        assert_eq!((particle.x, particle.y), (0.0, 2.0));
        // Velocity (vx, vy) -> (vy, -vx).
        assert_eq!((particle.vx, particle.vy), (3.0, -2.0));
        // New wall grid is 3 wide, 4 tall.
        assert_eq!(rotated.walls[6], 7); // (1, 0) -> (0, 2)
        assert_eq!(rotated.walls[2], 9); // (3, 2) -> (2, 0)
        assert_eq!(rotated.walls.iter().filter(|&&wall| wall != 0).count(), 2);
        // Sign anchors rotate too, and the text survives.
        assert_eq!((rotated.signs[0].x, rotated.signs[0].y), (0, 2));
        assert_eq!(rotated.signs[0].text, "mark");
        // The air grid rotates cell-by-cell with the velocity fields.
        assert_eq!(rotated.air.pv, vec![20.0, 40.0, 10.0, 30.0]);
        assert_eq!(rotated.air.vx, vec![6.0, 8.0, 5.0, 7.0]);
        assert_eq!(rotated.air.vy, vec![-2.0, -4.0, -1.0, -3.0]);
        assert_eq!(rotated.air.hv, vec![200.0, 400.0, 100.0, 300.0]);
    }

    /// `Shift+R` / `Ctrl+Shift+R` (`GameView.cpp:1389-1400`): the two mirrors
    /// keep the region size and flip exactly one axis.
    #[test]
    fn placement_mirrors_flip_one_axis() {
        let region = sample_region();

        let horizontal = region.mirror_horizontal();
        assert_eq!((horizontal.w, horizontal.h), (4, 3));
        let particle = horizontal.particles[0];
        assert_eq!((particle.x, particle.y), (2.0, 0.0));
        assert_eq!((particle.vx, particle.vy), (-2.0, 3.0));
        assert_eq!(horizontal.walls[2], 7); // (1, 0) -> (2, 0)
        assert_eq!(horizontal.walls[8], 9); // (3, 2) -> (0, 2)
        assert_eq!((horizontal.signs[0].x, horizontal.signs[0].y), (2, 0));
        assert_eq!(horizontal.air.pv, vec![20.0, 10.0, 40.0, 30.0]);
        assert_eq!(horizontal.air.vx, vec![-2.0, -1.0, -4.0, -3.0]);
        assert_eq!(horizontal.air.vy, vec![6.0, 5.0, 8.0, 7.0]);

        let vertical = region.mirror_vertical();
        assert_eq!((vertical.w, vertical.h), (4, 3));
        let particle = vertical.particles[0];
        assert_eq!((particle.x, particle.y), (1.0, 2.0));
        assert_eq!((particle.vx, particle.vy), (2.0, -3.0));
        assert_eq!(vertical.walls[9], 7); // (1, 0) -> (1, 2)
        assert_eq!(vertical.walls[3], 9); // (3, 2) -> (3, 0)
        assert_eq!((vertical.signs[0].x, vertical.signs[0].y), (1, 2));
        assert_eq!(vertical.air.pv, vec![30.0, 40.0, 10.0, 20.0]);
        assert_eq!(vertical.air.vx, vec![3.0, 4.0, 1.0, 2.0]);
        assert_eq!(vertical.air.vy, vec![-7.0, -8.0, -5.0, -6.0]);
    }

    /// `GameView::TranslateSave` (`GameView.cpp:1975-1986`): the arrow keys
    /// move the placement anchor without touching the captured region.
    #[test]
    fn placement_translate_moves_only_the_anchor() {
        let region = sample_region();
        let mut placement = Placement {
            region: region.clone(),
            anchor: (10, 20),
        };
        placement.translate(-3, 4);
        assert_eq!(placement.anchor, (7, 24));
        assert_eq!(placement.region.particles, region.particles);

        // Transforming the placement replaces the region with the rotated
        // copy, so a following commit places the rotated dimensions.
        placement.rotate_ccw();
        assert_eq!((placement.region.w, placement.region.h), (3, 4));
        assert_eq!(placement.anchor, (7, 24));
    }

    /// The platform clipboard payload round-trips the exact same `Region` the
    /// in-process clipboard keeps, and unrelated text is rejected.
    #[test]
    fn platform_clipboard_payload_round_trips() {
        let snapshot = sample_snapshot();
        let region = Region::capture(&snapshot, 0, 0, 63, 47);
        let text = region.to_clipboard_text();
        assert!(text.starts_with(CLIPBOARD_MAGIC));

        let back = Region::from_clipboard_text(&text).expect("region payload parses");
        assert_eq!(back.w, region.w);
        assert_eq!(back.h, region.h);
        assert_eq!(back.particles, region.particles);
        assert_eq!(back.walls, region.walls);
        assert_eq!(back.air.pv, region.air.pv);
        assert_eq!(back.signs, region.signs);

        assert!(Region::from_clipboard_text("https://example.com").is_none());
        assert!(Region::from_clipboard_text("").is_none());
    }
}
