//! High-level simulation commands issued by the UI, Lua scripting and save
//! loading.
//!
//! Commands are plain data so they can be queued, logged, replayed and sent
//! across threads without borrowing the simulation. `SimulationEngine` in
//! `tpt-sim` consumes them through `apply_command`.

use serde::{Deserialize, Serialize};

use crate::particle::GpuParticle;
use crate::sign::Sign;

/// Brush rasterization shape used by the drawing tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrushShape {
    /// Elliptical/circular brush.
    Circle = 0,
    /// Axis-aligned rectangular brush.
    Square = 1,
    /// Upward pointing triangular brush.
    Triangle = 2,
}

/// Non-drawing simulation tools that act directly on the air, pressure and
/// temperature fields (the reference Powder Toy "sim tools").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimToolKind {
    /// Heat the particle under the cursor by `strength * 2.0` K (`* 0.1` for
    /// pumps), never the air heat field (`HEAT.cpp:16-28`).
    Heat,
    /// Cool the particle under the cursor by `strength * 2.0` K (`* 0.1` for
    /// pumps) (`COOL.cpp:16-28`).
    Cool,
    /// Inject air pressure (`+strength * 0.05` per pixel) into the touched
    /// cells (`AIR.cpp:17-23`).
    Air,
    /// Remove air pressure (`-strength * 0.05` per pixel) from the touched
    /// cells (`VAC.cpp:17-23`).
    Vac,
    /// Push air along the stroke step by `delta * 0.002 * strength`
    /// (`WIND.cpp:20-29`).
    Wind,
    /// Swap the particle under the cursor with a same-state particle within
    /// `sqrt(strength) * 10` pixels, on a 1-in-100 roll (`MIX.cpp:21-57`).
    Mix,
    /// Rotate air velocity about the brush centre (cyclone, `CYCL.cpp:24-51`).
    Cycl,
    /// Increase the ambient heat field.
    AmbHeatPlus,
    /// Decrease the ambient heat field.
    AmbHeatMinus,
    /// `DEFAULT_TOOL_PGRV`: push the touched particles towards the brush
    /// centre, emulating a short-lived positive gravity well.
    PosGrav,
    /// `DEFAULT_TOOL_NGRV`: push the touched particles away from the brush
    /// centre, emulating a short-lived negative gravity well.
    NegGrav,
}

impl SimToolKind {
    /// Every simulation tool, in the order the tool palette lists them.
    pub const ALL: [Self; 11] = [
        // Keep this list sorted like `SIM_TOOL_BUTTONS` in the app crate.
        Self::Heat,
        Self::Cool,
        Self::Air,
        Self::Vac,
        Self::Wind,
        Self::Mix,
        Self::Cycl,
        Self::AmbHeatPlus,
        Self::AmbHeatMinus,
        Self::PosGrav,
        Self::NegGrav,
    ];

    /// Looks up a tool by its index in [`Self::ALL`].
    pub fn from_index(index: u32) -> Option<Self> {
        Self::ALL.get(index as usize).copied()
    }
}

/// Colour-blend mode of the decoration tool (`DECO_*` in `SimulationData.h`).
///
/// Every variant is one reference decoration tool (`DEFAULT_DECOR_SET`,
/// `DEFAULT_DECOR_CLR`, `DEFAULT_DECOR_ADD`, ...): the palette binds one tool
/// per mode and [`SimCommand::DecorateAt`] carries the selected mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecoMode {
    /// `DECO_DRAW`: overwrite the decoration with the tool colour.
    Draw = 0,
    /// `DECO_CLEAR`: remove the decoration.
    Clear = 1,
    /// `DECO_ADD`: add the tool colour.
    Add = 2,
    /// `DECO_SUBTRACT`: subtract the tool colour.
    Subtract = 3,
    /// `DECO_MULTIPLY`: scale the colour by `1 + channel * 0.01`.
    Multiply = 4,
    /// `DECO_DIVIDE`: scale the colour by `1 / (1 + channel * 0.01)`.
    Divide = 5,
    /// `DECO_SMUDGE`: blend in the surrounding decorations.
    Smudge = 6,
}

impl DecoMode {
    /// Every mode, in the reference `DECO_*` order.
    pub const ALL: [Self; 7] = [
        Self::Draw,
        Self::Clear,
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Divide,
        Self::Smudge,
    ];

    /// `DECO_*` index, used by the `Decoration.Mode` preference.
    pub fn index(self) -> u32 {
        self as u32
    }

    /// Mode selected by a `Decoration.Mode` preference value; out-of-range
    /// values fall back to [`Self::Draw`].
    pub fn from_index(index: u32) -> Self {
        Self::ALL.get(index as usize).copied().unwrap_or(Self::Draw)
    }

    /// Human-readable name for combo boxes.
    pub fn label(self) -> &'static str {
        match self {
            Self::Draw => "Draw",
            Self::Clear => "Clear",
            Self::Add => "Add",
            Self::Subtract => "Subtract",
            Self::Multiply => "Multiply",
            Self::Divide => "Divide",
            Self::Smudge => "Smudge",
        }
    }
}

/// Integer addressed particle property written by
/// [`SimCommand::SetParticleFieldAt`].
///
/// The property tool (`DEFAULT_UI_PROPERTY`) edits a single scalar field of
/// every particle under the brush; naming the field explicitly keeps the UI
/// from addressing the SoA directly. Decoration colours use
/// [`SimCommand::DecorateAt`] instead, because they blend rather than replace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticleField {
    /// Element-specific scratch value (`tmp`).
    Tmp,
    /// Element-specific scratch value (`tmp2`).
    Tmp2,
    /// Element-specific scratch value (`tmp3`).
    Tmp3,
    /// Element-specific scratch value (`tmp4`).
    Tmp4,
    /// Remaining life (`life`).
    Life,
    /// Element spawned on transition / clone target (`ctype`).
    Ctype,
}

/// Shape rasterized when a new rigid body is spawned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RigidSpawnShape {
    /// Rectangular block of pixels.
    Box,
    /// Filled disc of pixels.
    Circle,
}

/// Cropped Eulerian air grids of a captured canvas region.
///
/// `Simulation::Save` (`Simulation.cpp:435-444`) copies the pressure, both
/// velocity components and the ambient-heat grid of every `CELL x CELL` block
/// touched by the saved rectangle into the `GameSave`; this is the same crop,
/// taken only when `includePressure` is on. The four grids are all-or-nothing:
/// a payload captured with the option off carries none of them, exactly like a
/// save written without `hasPressure`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RegionAir {
    /// Cells spanned horizontally.
    pub wcells: u32,
    /// Cells spanned vertically.
    pub hcells: u32,
    /// Air pressure per cell, row-major (`wcells * hcells`).
    pub pv: Vec<f32>,
    /// Air x-velocity per cell.
    pub vx: Vec<f32>,
    /// Air y-velocity per cell.
    pub vy: Vec<f32>,
    /// Ambient heat per cell.
    pub hv: Vec<f32>,
}

impl RegionAir {
    /// Whether the payload carries no air at all.
    ///
    /// `Simulation::Save`'s `includePressure` block is the only source of these
    /// grids, so an empty payload is exactly a capture taken with the option
    /// off (`OptionsView.cpp:401`).
    pub fn is_empty(&self) -> bool {
        self.wcells == 0 || self.hcells == 0 || self.pv.is_empty()
    }

    /// Number of cells in one full grid.
    pub fn len(&self) -> usize {
        (self.wcells as usize).saturating_mul(self.hcells as usize)
    }

    /// Whether every grid holds a full `wcells * hcells` array.
    pub fn is_complete(&self) -> bool {
        let len = self.len();
        len > 0
            && self.pv.len() == len
            && self.vx.len() == len
            && self.vy.len() == len
            && self.hv.len() == len
    }
}

/// A single editing/state mutation for the simulation.
///
/// Coordinates are in simulation pixels, with `(0, 0)` at the top-left corner.
/// `elem_ty` is a `PT_*` element id, `wall_id` a `WL_*` wall id and `dcolour`
/// the packed `0x00RRGGBB` decoration colour (`0` = none).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SimCommand {
    /// Remove every particle, wall and rigid body.
    ClearAll,
    /// Reallocate the canvas; the simulation is reset to the new size.
    ResizeCanvas {
        /// New width in simulation pixels.
        xres: u32,
        /// New height in simulation pixels.
        yres: u32,
    },
    /// Place particles at the given pixels, typically one brush stamp.
    PlaceParticles {
        /// Pixels to fill, in simulation coordinates.
        points: Vec<(i32, i32)>,
        /// Element id (`PT_*`) to create.
        elem_ty: u32,
        /// Replace existing particles instead of only filling empty pixels.
        replace_mode: bool,
        /// Decoration colour (`0` = none).
        dcolour: u32,
    },
    /// Fire one lightning bolt, the `PT_LIGH` special case of
    /// `Simulation::CreateParts` (`Editing.cpp:750-760`).
    ///
    /// The brush radius sum is the bolt power (clamped to 55) and the engine
    /// throttles the next bolt to `power / 4` ticks.
    CreateLightning {
        /// Bolt pixel x coordinate.
        x: i32,
        /// Bolt pixel y coordinate.
        y: i32,
        /// Brush radius x, packed into the bolt power.
        radius_x: i32,
        /// Brush radius y, packed into the bolt power.
        radius_y: i32,
    },
    /// Flood-fill a contiguous region of the same element starting at `(x, y)`.
    FloodFillParticles {
        /// Seed pixel x coordinate.
        x: i32,
        /// Seed pixel y coordinate.
        y: i32,
        /// Element id (`PT_*`) to fill with.
        elem_ty: u32,
    },
    /// Place or erase walls on the coarse `CELL x CELL` block grid.
    PlaceWalls {
        /// Block coordinates, in `CELL`-sized wall cells.
        cells: Vec<(i32, i32)>,
        /// Wall id (`WL_*`); `WL_ERASE` removes walls.
        wall_id: u8,
        /// Fan/streamline velocity set on the touched blocks, if any.
        fan_vel: Option<(f32, f32)>,
    },
    /// Apply a non-drawing simulation tool over a brush stroke.
    ApplySimTool {
        /// Pixels covered by the stroke.
        points: Vec<(i32, i32)>,
        /// Tool to apply.
        tool: SimToolKind,
        /// Tool strength multiplier, normally `0.0 ..= 1.0`.
        strength: f32,
        /// Stroke step in simulation pixels (`position2 - position1` in
        /// `WIND.cpp:28-29`), not a unit vector: the wind tool scales it.
        dir: (f32, f32),
        /// Brush centre in simulation pixels (TPT's `brushX`/`brushY`,
        /// `SimTool.cpp:16`); the cyclone tool rotates around it.
        center: (i32, i32),
        /// `true` for the drag/tick path (`GameController::DrawPoints` with
        /// `held`), `false` for the click stamp and the line/rect gestures
        /// (`PerformDrag`). Only `WIND` uses it (`WIND.cpp:20-22`).
        dragging: bool,
    },
    /// Spawn a new pixel-cluster rigid body.
    SpawnRigidBody {
        /// Centre x coordinate in simulation pixels.
        x: f32,
        /// Centre y coordinate in simulation pixels.
        y: f32,
        /// Half width in pixels.
        half_w: i32,
        /// Half height in pixels.
        half_h: i32,
        /// Shape to rasterize.
        shape: RigidSpawnShape,
        /// Element id (`PT_*`) of the constituent pixels.
        elem_ty: u32,
    },
    /// Write one property of an existing particle.
    ///
    /// `field` uses the lower-case Powder Toy property names understood by the
    /// Lua API (`"type"`, `"life"`, `"ctype"`, `"temp"`, `"vx"`, `"vy"`,
    /// `"x"`, `"y"`, `"dcolour"`, ...). Unknown fields are ignored.
    SetParticleProperty {
        /// Target particle index.
        part_id: u32,
        /// Property name.
        field: String,
        /// New value; integer properties are rounded.
        value: f32,
    },
    /// Write one integer property of every particle under a brush stroke.
    ///
    /// This is the property tool (`DEFAULT_UI_PROPERTY`): it walks the brushed
    /// pixels and rewrites a single scalar field, leaving the element itself
    /// untouched.
    SetParticleFieldAt {
        /// Pixels covered by the stroke, in simulation coordinates.
        points: Vec<(i32, i32)>,
        /// Field to rewrite.
        field: ParticleField,
        /// New value; integer properties are rounded.
        value: i64,
    },
    /// Blend the decoration colour of every particle under a brush stroke.
    ///
    /// Mirrors `Simulation::ApplyDecorationPoint` behind the reference
    /// `DecorationTool`: each touched particle's `dcolour` is read and blended
    /// towards `colour` according to `mode`. Only the decoration is touched;
    /// the element, its temperature and its velocities are left alone.
    DecorateAt {
        /// Pixels covered by the stroke, in simulation coordinates.
        points: Vec<(i32, i32)>,
        /// Tool colour as `0xAARRGGBB`; the alpha byte is the blend weight.
        colour: u32,
        /// Blend mode.
        mode: DecoMode,
    },
    /// Create exactly one particle, optionally with an initial velocity.
    ///
    /// The `PARTICLE` tool and Lua's `sim.partCreate` use this instead of
    /// [`Self::PlaceParticles`] so a single spawn is one particle, not a brush
    /// stamp.
    SpawnParticle {
        /// Pixel x coordinate.
        x: i32,
        /// Pixel y coordinate.
        y: i32,
        /// Element id (`PT_*`) to create.
        elem_ty: u32,
        /// Initial horizontal velocity in pixels per step.
        vx: f32,
        /// Initial vertical velocity in pixels per step.
        vy: f32,
    },
    /// Kill the particle occupying the given pixel, if any.
    KillAt {
        /// Pixel x coordinate.
        x: i32,
        /// Pixel y coordinate.
        y: i32,
    },
    /// Flood-fill walls starting at the given block or pixel.
    FloodFillWalls {
        /// Seed pixel x coordinate.
        x: i32,
        /// Seed pixel y coordinate.
        y: i32,
        /// Wall id (`WL_*`) to fill with.
        wall_id: u8,
        /// Wall id to replace; `None` replaces the wall under `(x, y)`.
        replace_wall: Option<u8>,
    },
    /// Clear particles and walls in the given pixel rectangle.
    ClearRect {
        /// Top-left x coordinate in pixels.
        x: i32,
        /// Top-left y coordinate in pixels.
        y: i32,
        /// Width in pixels.
        w: i32,
        /// Height in pixels.
        h: i32,
    },
    /// Stamp one captured canvas region at `(x, y)`.
    ///
    /// Ports `Simulation::Load` (`Simulation.cpp:49-340`) for the clipboard and
    /// stamp placement paths: the destination rectangle is cleared first
    /// (`Simulation.cpp:63-103`), every saved particle is recreated and patched
    /// (`Simulation.cpp:110-260`), the region's wall cells and air grids are
    /// copied (`Simulation.cpp:292-321`) and the signs anchored inside the
    /// rectangle are appended (`Simulation.cpp:277-291`).
    PlaceRegion {
        /// Top-left x coordinate of the region in pixels.
        x: i32,
        /// Top-left y coordinate of the region in pixels.
        y: i32,
        /// Region width in pixels.
        w: u32,
        /// Region height in pixels.
        h: u32,
        /// Particles with region-relative pixel coordinates.
        particles: Vec<GpuParticle>,
        /// Wall ids, row-major over `w * h` pixels (`0` = no wall).
        walls: Vec<u32>,
        /// Cropped air grids; an empty payload leaves the destination air
        /// untouched (`Simulation.cpp:301-321` only writes them when
        /// `includePressure` is on).
        air: RegionAir,
        /// Signs anchored inside the region, with region-relative anchors.
        signs: Vec<Sign>,
    },
    /// Flood-fill particle decoration colour.
    FloodDecorate {
        /// Seed pixel x coordinate.
        x: i32,
        /// Seed pixel y coordinate.
        y: i32,
        /// Packed `0xAABBGGRR` decoration colour.
        colour: u32,
    },
    /// Reset particle temperatures to element defaults.
    ResetTemp {
        /// Only reset conductive/non-insulating elements.
        only_conductors: bool,
    },
    /// Reset Eulerian air pressure in a cell rectangle to ambient/edge pressure.
    ResetPressure {
        /// Cell x coordinate.
        x: i32,
        /// Cell y coordinate.
        y: i32,
        /// Width in cells.
        w: i32,
        /// Height in cells.
        h: i32,
    },
    /// Zero Eulerian air velocity in a cell rectangle.
    ResetVelocity {
        /// Cell x coordinate.
        x: i32,
        /// Cell y coordinate.
        y: i32,
        /// Width in cells.
        w: i32,
        /// Height in cells.
        h: i32,
    },
    /// Clear electrical sparks across the simulation.
    ResetSpark,
    /// Negate the pressure and both velocity components of every air cell.
    ///
    /// `GameController::InvertAirSim` -> `Air::Invert` (`Air.cpp:499-513`),
    /// bound to `I` (`GameView.cpp:1617-1618`).
    InvertAir,
    /// Reset every ambient heat cell to the configured ambient temperature.
    ///
    /// `GameController::ResetAHeat` -> `GameModel::ResetAHeat` ->
    /// `Air::ClearAirH` (`Air.cpp:71-74`), bound to Ctrl+U
    /// (`GameView.cpp:1539-1542`).
    ResetAmbientHeat,
    /// Spark the conductor occupying a pixel.
    ///
    /// Mirrors `Simulation::create_part(-1, x, y, PT_SPRK)` from a `{b|..}`
    /// button sign (`GameController.cpp:583-585`): an empty, walled or
    /// already-charged pixel is left untouched.
    SparkAt {
        /// Pixel x coordinate.
        x: i32,
        /// Pixel y coordinate.
        y: i32,
    },
    /// Set the live command word of a player stickman.
    ///
    /// Ports `GameController::KeyPress` / `KeyRelease` writing
    /// `sim->player.comm` / `sim->player2.comm`
    /// (`GameController.cpp:611-730`): bits `0x01` left, `0x02` right,
    /// `0x04` jump, `0x08` use element. `pcomm` is the direction word
    /// saved when a left/right key is released.
    SetStickmanControl {
        /// `0` = `STKM` (arrow keys), `1` = `STKM2` (WASD).
        player: u8,
        /// Current command bits.
        comm: u32,
        /// Previous command word (element-spawn direction memory).
        pcomm: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::DecoMode;

    #[test]
    fn deco_mode_index_round_trips() {
        for (index, mode) in DecoMode::ALL.into_iter().enumerate() {
            assert_eq!(mode.index(), index as u32);
            assert_eq!(DecoMode::from_index(index as u32), mode);
        }
        // Out-of-range `Decoration.Mode` values fall back to `Draw`.
        assert_eq!(DecoMode::from_index(7), DecoMode::Draw);
        assert_eq!(DecoMode::from_index(u32::MAX), DecoMode::Draw);
    }

    #[test]
    fn deco_mode_indices_match_the_reference_constants() {
        // `DECO_DRAW` .. `DECO_SMUDGE` in `SimulationData.h`.
        assert_eq!(DecoMode::Draw.index(), 0);
        assert_eq!(DecoMode::Clear.index(), 1);
        assert_eq!(DecoMode::Add.index(), 2);
        assert_eq!(DecoMode::Subtract.index(), 3);
        assert_eq!(DecoMode::Multiply.index(), 4);
        assert_eq!(DecoMode::Divide.index(), 5);
        assert_eq!(DecoMode::Smudge.index(), 6);
    }
}
