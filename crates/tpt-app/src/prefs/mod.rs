//! Persistent user preferences for the Rust front-end.
//!
//! The C++ game keeps its settings in `prefs.json` inside its data directory.
//! This module mirrors that format for the settings exposed by the Options and
//! Render windows, using the same dotted keys (`Simulation.*`, `Renderer.*`,
//! `Decoration.*`, ...). The file is stored in the process working directory,
//! next to the `saves` folder.
//!
//! The preference file uses a small JSON reader/writer instead of derive
//! macros. Only a flat object of scalar values is needed, and unknown keys (for
//! example prefs written by the C++ game or by other tools) are preserved
//! verbatim across save/load.
//!
//! Every operation is best effort: a missing, unreadable or corrupt file falls
//! back to the defaults and never panics.
//!
//! The key/default/coercion schema lives in `schema`; the file reader/writer
//! and the per-section key mapping live in `io`.

mod io;
mod schema;

use tpt_core::commands::DecoMode;
use tpt_core::particle::{AirMode, EdgeMode, GravityMode, SimulationConfig};

use self::io::JsonValue;
use self::schema::clean_f32;
use crate::app_state::GameMode;
use crate::i18n::Language;

/// Name of the preference file, relative to the process working directory.
pub const PREFS_FILE: &str = "prefs.json";

/// Lowest sensible simulation framerate cap (TPT `FpsLimitExplicit::minSane`).
pub const MIN_SANE_FPS: f32 = 3.0;
/// Highest sensible simulation framerate cap (TPT `FpsLimitExplicit::maxSane`).
pub const MAX_SANE_FPS: f32 = 1000.0;
/// Lowest sensible rendering framerate cap (TPT `DrawLimitExplicit::minSane`).
pub const MIN_SANE_DRAW_FPS: u32 = 1;
/// Highest rendering framerate cap, in frames per second.
///
/// TPT lets the cap run to 1000; the port hard-locks it at 240 so the renderer
/// never outruns a 240 Hz display (the extra presents would be pure GPU work).
pub const MAX_SANE_DRAW_FPS: u32 = 240;
/// Default rendering framerate cap: the 240 fps ceiling, independent of the
/// display refresh rate.
pub const DEFAULT_DRAW_FPS: u32 = MAX_SANE_DRAW_FPS;
/// TPT caps the undo history at 200 snapshots to bound memory use.
pub const MAX_UNDO_HISTORY_LIMIT: u32 = 200;
/// TPT's default undo history limit.
pub const DEFAULT_UNDO_HISTORY_LIMIT: u32 = 5;
/// Default 3D render scale (full viewport resolution).
pub const DEFAULT_VOXEL_RENDER_SCALE: f32 = 1.0;
/// Default number of ambient-occlusion rays in the 3D renderer.
pub const DEFAULT_VOXEL_AO_RAYS: u32 = 2;
/// Default Jacobi iteration count of the 3D air solve.
pub const DEFAULT_VOXEL_AIR_ITERATIONS: u32 = 12;
/// Lowest settable 3D air iteration count.
pub const MIN_VOXEL_AIR_ITERATIONS: u32 = 1;
/// Highest settable 3D air iteration count.
pub const MAX_VOXEL_AIR_ITERATIONS: u32 = 64;
/// Render scales offered by the 3D options page.
pub const VOXEL_RENDER_SCALES: [f32; 3] = [0.5, 0.75, 1.0];
/// Ambient-occlusion ray counts offered by the 3D options page.
pub const VOXEL_AO_CHOICES: [u32; 3] = [0, 2, 4];
/// Lowest window scale factor offered by the Options window.
pub const MIN_WINDOW_SCALE: u32 = 1;
/// Highest window scale factor offered by the Options window.
pub const MAX_WINDOW_SCALE: u32 = 8;

/// Snaps an arbitrary render scale to the closest offered value.
pub fn snap_voxel_render_scale(scale: f32) -> f32 {
    if !scale.is_finite() {
        return DEFAULT_VOXEL_RENDER_SCALE;
    }
    VOXEL_RENDER_SCALES
        .into_iter()
        .min_by(|a, b| {
            (a - scale)
                .abs()
                .partial_cmp(&(b - scale).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(DEFAULT_VOXEL_RENDER_SCALE)
}

/// Snaps an arbitrary ray count to the closest offered value.
pub fn snap_voxel_ao_rays(rays: u32) -> u32 {
    VOXEL_AO_CHOICES
        .into_iter()
        .min_by_key(|value| value.abs_diff(rays))
        .unwrap_or(DEFAULT_VOXEL_AO_RAYS)
}

/// Simulation framerate cap, mirroring TPT's `SimFpsLimit`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FpsLimit {
    /// Run the simulation at most `fps` times per second.
    Exact(f32),
    /// Advance the simulation once per rendered frame.
    Uncapped,
}

impl Default for FpsLimit {
    fn default() -> Self {
        Self::Exact(60.0)
    }
}

impl FpsLimit {
    /// Minimum interval between two simulation steps, or `None` when uncapped.
    pub fn interval_secs(self) -> Option<f32> {
        match self {
            Self::Exact(fps) => Some(1.0 / fps.clamp(MIN_SANE_FPS, MAX_SANE_FPS)),
            Self::Uncapped => None,
        }
    }

    /// i18n key of the combo-box label.
    pub fn label_key(self) -> &'static str {
        match self {
            Self::Exact(_) => "options.limit_exact",
            Self::Uncapped => "options.limit_uncapped",
        }
    }

    /// TPT stores uncapped as the sentinel value `2`; anything else is an
    /// explicit frames-per-second value.
    fn from_pref_value(value: f64) -> Self {
        if (value - 2.0).abs() < f64::EPSILON {
            Self::Uncapped
        } else if value >= MIN_SANE_FPS as f64 && value <= MAX_SANE_FPS as f64 {
            Self::Exact(value as f32)
        } else {
            Self::default()
        }
    }

    /// Cap as shared with the engine (`2.0` = uncapped).
    pub fn to_cap_value(self) -> f32 {
        match self {
            Self::Exact(fps) => fps,
            Self::Uncapped => 2.0,
        }
    }

    /// Reads a cap value set by a Lua script (`2.0` = uncapped).
    ///
    /// Values above/below the sane range are clamped (not defaulted) so a Lua
    /// `tpt.fpsCap(1000)` keeps its meaning while an absurd value cannot stall
    /// or spin the simulation.
    pub fn from_cap_value(value: f32) -> Self {
        if (value - 2.0).abs() < f32::EPSILON {
            Self::Uncapped
        } else {
            Self::Exact(value.clamp(MIN_SANE_FPS, MAX_SANE_FPS))
        }
    }

    fn to_pref_value(self) -> f64 {
        match self {
            Self::Exact(fps) => clean_f32(fps),
            Self::Uncapped => 2.0,
        }
    }
}

/// Interval between two rendered frames for a cap in frames per second.
///
/// The value is clamped into [`MIN_SANE_DRAW_FPS`]..=[`MAX_SANE_DRAW_FPS`], so
/// no preference can push the renderer past 240 fps.
pub fn draw_interval_secs(fps: u32) -> f32 {
    1.0 / fps.clamp(MIN_SANE_DRAW_FPS, MAX_SANE_DRAW_FPS) as f32
}

///
/// Field defaults match the C++ game's defaults. The config-backed subset is
/// synchronised with [`SimulationConfig`] through [`Self::apply_to_config`] and
/// [`Self::capture_from_config`]; window-level fields (resizable, fullscreen,
/// ...) are persisted for the platform layer.
#[derive(Clone, Debug)]
pub struct Prefs {
    /// `Simulation.NewtonianGravity`: FFT Poisson gravity solve.
    pub newtonian_gravity: bool,
    /// `Simulation.AmbientHeat`: ambient heat simulation.
    pub ambient_heat: bool,
    /// `Simulation.AmbientAirTemp`: ambient air temperature in Kelvin.
    pub ambient_air_temp: f32,
    /// `Simulation.EdgePressure`: pressure applied at the canvas border.
    pub edge_pressure: f32,
    /// `Simulation.EdgeVelocityX`: ambient air velocity at the border.
    pub edge_velocity_x: f32,
    /// `Simulation.EdgeVelocityY`: ambient air velocity at the border.
    pub edge_velocity_y: f32,
    /// `Simulation.VorticityCoeff`: vorticity confinement coefficient.
    pub vorticity_coeff: f32,
    /// `Simulation.ConvectionMode`: `AIRC_*` air heat convection mode.
    pub convection_mode: u32,
    /// `Simulation.EdgeMode`: `EDGE_*` boundary behaviour.
    pub edge_mode: u32,
    /// `Simulation.IncludePressure`: include pressure when copying/stamping.
    pub include_pressure: bool,
    /// `Simulation.UndoHistoryLimit`: number of undo snapshots kept.
    pub undo_history_limit: u32,
    /// `Simulation.PrettyPowder`: prettier (slower) powder rendering.
    pub pretty_powder: bool,
    /// `Simulation.EnableModElements` (port-only): allow creating the Cracker
    /// TPT mod elements. Saved particles keep simulating either way.
    pub mod_elements_enabled: bool,
    /// `Simulation.HeatSimulation`: heat conduction and phase changes.
    ///
    /// The C++ game does not persist this one, but keeping it makes the
    /// Options window behave consistently across restarts.
    pub heat_enable: bool,
    /// `Simulation.WaterEqualisation`: TPT water equalisation.
    pub water_equal_test: bool,
    /// `Simulation.AirMode`: `AIR_*` Eulerian air simulation mode.
    pub air_mode: u32,
    /// `Simulation.GravityMode`: `GRAV_*` gravity model.
    pub gravity_mode: u32,
    /// `Simulation.CustomGravityX`: custom gravity direction (x).
    pub custom_gravity_x: f32,
    /// `Simulation.CustomGravityY`: custom gravity direction (y).
    pub custom_gravity_y: f32,
    /// `Renderer.DisplayMode`: `DISPLAY_*` bitmask.
    pub display_mode: u32,
    /// `Renderer.RenderMode`: `RENDER_*` bitmask.
    pub render_mode: u32,
    /// `Renderer.ColourMode`: `COLOUR_*` bitmask.
    pub colour_mode: u32,
    /// `Renderer.Decorations`: draw particle decorations.
    pub decorations: bool,
    /// `Renderer.GravityField`: draw the Newtonian gravity force vectors.
    pub gravity_field: bool,
    /// `Renderer.GridCheckerboard`: checkerboard background on the grid.
    pub grid_checkerboard: bool,
    /// `Renderer.TemperatureScale`: `TEMPSCALE_*`. The port defaults to
    /// Kelvin.
    pub temperature_scale: u32,
    /// `Renderer.FireIntensity`: additive fire glow strength (`0.0..=4.0`).
    pub fire_intensity: f32,
    /// `MouseClickRequired`: "sticky categories" (click instead of hover).
    pub mouse_click_required: bool,
    /// `Decoration.Red`: default decoration colour, red channel.
    pub deco_red: u8,
    /// `Decoration.Green`: default decoration colour, green channel.
    pub deco_green: u8,
    /// `Decoration.Blue`: default decoration colour, blue channel.
    pub deco_blue: u8,
    /// `Decoration.Alpha`: default decoration colour, alpha channel.
    pub deco_alpha: u8,
    /// `Decoration.Mode`: decoration blend mode the palette binds by default
    /// (`DECO_*` index, `0` = draw). The bound tool carries its own copy, so a
    /// palette click also refreshes this value.
    pub deco_mode: u32,
    /// `Prop.Type`: field index edited by the property tool
    /// ([`crate::app_state::PropertyKind::index`]; `0` = `tmp`, `1` = `life`,
    /// `2` = `ctype`, `3` = `tmp2`, ...).
    pub prop_field: u32,
    /// `Prop.Value`: value the property tool writes to the selected field.
    pub prop_value: f32,
    /// `Prop.Fill`: whether the property tool floods connected matching particles.
    pub prop_fill: bool,
    /// `Tool.SpawnElement`: element id created by the single-particle tool.
    pub spawn_element: u32,
    /// `Tool.Primary`: preference key of the tool bound to the left button.
    pub primary_tool: String,
    /// `Tool.Secondary`: preference key of the tool bound to the right button.
    pub secondary_tool: String,
    /// `FpsLimit`: simulation steps per second (logical TPS) cap.
    ///
    /// Canonical and persisted; Lua's `tpt.fpsCap` reaches it through the live
    /// `SimulationEngine::fps_cap` mirror, which
    /// [`crate::app_state::AppState::persist_script_caps`] folds back here.
    pub fps_limit: FpsLimit,
    /// `DrawLimit`: rendering framerate cap, clamped to 1..=`MAX_SANE_DRAW_FPS`.
    ///
    /// Persisted, like [`Self::fps_limit`]; Lua writes the
    /// `SimulationEngine::draw_cap` mirror (`0` = display).
    pub draw_fps: u32,
    /// `BlurryScaling`: use blurry instead of nearest-neighbour scaling.
    pub blurry_scaling: bool,
    /// `Resizable`: allow resizing and maximising the window.
    pub resizable: bool,
    /// `Fullscreen`: start in fullscreen mode.
    pub fullscreen: bool,
    /// `NativeClipboard.Enabled`: region copy/paste goes through the platform
    /// clipboard (`OptionsView.cpp:413-415`). When off the clipboard stays
    /// in-process, like a build without `PLATFORM_CLIPBOARD`.
    pub native_clipboard: bool,
    /// `GraveExitsConsole`: the key under Esc closes the console.
    pub grave_exits_console: bool,
    /// `Language`: interface language, stored as its [`Language::code`].
    pub language: Language,
    /// `Game.Mode`: simulation flavour restored on the next launch.
    pub game_mode: GameMode,
    /// `WindowX`: last window outer x coordinate, in physical pixels
    /// (`PowderToy.cpp:44`). `None` until the first clean exit.
    pub window_x: Option<i32>,
    /// `WindowY`: last window outer y coordinate, in physical pixels
    /// (`PowderToy.cpp:45`).
    pub window_y: Option<i32>,
    /// `WindowWidth`: last window inner width, in physical pixels. The C++
    /// game recreates resizable windows at their previous size
    /// (`PowderToySDL.cpp:212-216`); the port keeps it across launches.
    pub window_width: Option<u32>,
    /// `WindowHeight`: last window inner height, in physical pixels.
    pub window_height: Option<u32>,
    /// `Scale`: window scale factor for larger screens
    /// (`OptionsView.cpp:340-362`). Drives the simulation viewport blit.
    pub scale: u32,
    /// `Renderer3D.RenderScale`: 3D render resolution scale (`0.5`, `0.75`,
    /// `1.0`).
    pub voxel_render_scale: f32,
    /// `Renderer3D.AORays`: sparse ambient-occlusion rays (`0`, `2` or `4`).
    pub voxel_ao_rays: u32,
    /// `Renderer3D.Shadows`: sun shadow rays in the 3D renderer.
    pub voxel_shadows: bool,
    /// `Renderer3D.AirIterations`: Jacobi iterations of the 3D air solve.
    pub voxel_air_iterations: u32,

    /// Raw entries read from disk, so unknown keys survive a rewrite.
    entries: Vec<(String, JsonValue)>,
}

impl Prefs {
    /// Applies every config-backed preference to `config`.
    ///
    /// Canvas size, pause state and any other save-specific field are left
    /// untouched.
    pub fn apply_to_config(&self, config: &mut SimulationConfig) {
        config.heat_enable = self.heat_enable;
        config.newton_grav = self.newtonian_gravity;
        config.water_equal_test = self.water_equal_test;
        config.aheat_enable = self.ambient_heat;
        config.air_mode = self.air_mode();
        config.ambient_air_temp = self.ambient_air_temp;
        config.edge_pressure = self.edge_pressure;
        config.edge_velocity_x = self.edge_velocity_x;
        config.edge_velocity_y = self.edge_velocity_y;
        config.vorticity_coeff = self.vorticity_coeff;
        config.convection_mode = self.convection_mode;
        config.gravity_mode = self.gravity_mode();
        config.custom_gravity_x = self.custom_gravity_x;
        config.custom_gravity_y = self.custom_gravity_y;
        config.edge_mode = self.edge_mode();
        config.gravity_field_enabled = self.gravity_field;
        config.display_mode = self.display_mode;
        config.colour_mode = self.colour_mode;
        config.render_mode = self.render_mode;
        config.fire_intensity = self.fire_intensity;
        config.pretty_powder = self.pretty_powder;
        config.mod_elements_enabled = self.mod_elements_enabled;
        config.temperature_scale = self.temperature_scale;
    }

    /// Refreshes the config-backed fields from a live simulation config.
    pub fn capture_from_config(&mut self, config: &SimulationConfig) {
        self.heat_enable = config.heat_enable;
        self.newtonian_gravity = config.newton_grav;
        self.water_equal_test = config.water_equal_test;
        self.ambient_heat = config.aheat_enable;
        self.air_mode = config.air_mode as u32;
        self.ambient_air_temp = config.ambient_air_temp;
        self.edge_pressure = config.edge_pressure;
        self.edge_velocity_x = config.edge_velocity_x;
        self.edge_velocity_y = config.edge_velocity_y;
        self.vorticity_coeff = config.vorticity_coeff;
        self.convection_mode = config.convection_mode;
        self.gravity_mode = config.gravity_mode as u32;
        self.custom_gravity_x = config.custom_gravity_x;
        self.custom_gravity_y = config.custom_gravity_y;
        self.edge_mode = config.edge_mode as u32;
        self.gravity_field = config.gravity_field_enabled;
        self.display_mode = config.display_mode;
        self.colour_mode = config.colour_mode;
        self.render_mode = config.render_mode;
        self.fire_intensity = config.fire_intensity;
        self.pretty_powder = config.pretty_powder;
        self.mod_elements_enabled = config.mod_elements_enabled;
        self.temperature_scale = config.temperature_scale;
    }

    /// Boundary behaviour selected by [`Self::edge_mode`].
    pub fn edge_mode(&self) -> EdgeMode {
        match self.edge_mode {
            1 => EdgeMode::Solid,
            2 => EdgeMode::Loop,
            _ => EdgeMode::Void,
        }
    }

    /// Gravity model selected by [`Self::gravity_mode`].
    pub fn gravity_mode(&self) -> GravityMode {
        match self.gravity_mode {
            1 => GravityMode::Off,
            2 => GravityMode::Radial,
            3 => GravityMode::Custom,
            _ => GravityMode::Vertical,
        }
    }

    /// Air simulation mode selected by [`Self::air_mode`].
    pub fn air_mode(&self) -> AirMode {
        match self.air_mode {
            1 => AirMode::PressureOff,
            2 => AirMode::VelocityOff,
            3 => AirMode::Off,
            4 => AirMode::NoUpdate,
            _ => AirMode::On,
        }
    }

    /// Decoration colour as `[r, g, b, a]`.
    pub fn deco_colour(&self) -> [u8; 4] {
        [
            self.deco_red,
            self.deco_green,
            self.deco_blue,
            self.deco_alpha,
        ]
    }

    /// Decoration blend mode selected by [`Self::deco_mode`].
    pub fn deco_mode(&self) -> DecoMode {
        DecoMode::from_index(self.deco_mode)
    }

    /// Screen pixels per simulation pixel for the viewport blit.
    ///
    /// `fit` is the viewport's aspect-preserving fit scale. `Scale`
    /// (`OptionsView.cpp:340-362`) multiplies it, mirroring the C++ window
    /// scale factor. egui rasterises the texture through its own linear
    /// sampler, so the port has no integer-scaling mode: the image always
    /// fills the available area at the requested fraction. [`Self::scale`] is
    /// clamped by the reader, but a hand-built `Prefs` still cannot produce a
    /// zero or negative factor.
    pub fn viewport_blit_scale(&self, fit: f32) -> f32 {
        let fit = if fit.is_finite() && fit > 0.0 {
            fit
        } else {
            1.0
        };
        fit * self.scale.max(MIN_WINDOW_SCALE) as f32
    }
}
