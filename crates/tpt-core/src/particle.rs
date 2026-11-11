//! Particle records, simulation configuration and rigid-body data structures.
//!
//! [`GpuParticle`] is the AoS mirror of the SoA GPU particle pool used for
//! save/load, Lua access and rigid-body coupling. It is exactly 64 bytes so a
//! `[GpuParticle]` slice can be uploaded verbatim to a storage buffer.

use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::constants::*;

/// Smallest accepted canvas edge length, in pixels.
pub const MIN_CANVAS_DIM: u32 = 64;

/// Mass below which a body is treated as massless.
const MASS_EPSILON: f32 = 1e-6;

/// Moment of inertia of a single unit-square pixel about its own centre,
/// i.e. `(1/12) * (1^2 + 1^2) = 1/6` for unit mass.
const PIXEL_INERTIA: f32 = 1.0 / 6.0;

/// A single particle as stored on the host and mirrored to the GPU.
///
/// Layout: 16 x 32-bit words = 64 bytes, `#[repr(C)]`, no padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct GpuParticle {
    /// Element type (`PT_*`).
    pub ty: u32,
    /// Remaining life; meaning depends on the element.
    pub life: i32,
    /// Element spawned on transition / clone target.
    pub ctype: u32,
    /// Pixel x coordinate.
    pub x: f32,
    /// Pixel y coordinate.
    pub y: f32,
    /// Horizontal velocity.
    pub vx: f32,
    /// Vertical velocity.
    pub vy: f32,
    /// Temperature in Kelvin.
    pub temp: f32,
    /// `FLAG_*` bits.
    pub flags: u32,
    /// Element-specific scratch value.
    pub tmp: i32,
    /// Element-specific scratch value.
    pub tmp2: i32,
    /// Element-specific scratch value.
    pub tmp3: i32,
    /// Element-specific scratch value.
    pub tmp4: i32,
    /// Decoration colour (`0x00RRGGBB`, 0 = none; a non-zero high byte acts as
    /// an alpha/blend factor).
    pub dcolour: u32,
    /// Padding, keeps the record at 64 bytes.
    pub _pad0: u32,
    /// Padding, keeps the record at 64 bytes.
    pub _pad1: u32,
}

// The GPU storage-buffer layout requires exactly 64 bytes.
const _: () = assert!(std::mem::size_of::<GpuParticle>() == 64);
const _: () = assert!(std::mem::align_of::<GpuParticle>() == 4);

impl GpuParticle {
    /// Number of 32-bit words in one particle record.
    pub const WORDS: usize = 16;

    /// Packs the record into 16 words for GPU upload, in field order.
    ///
    /// Floating-point fields are stored as their IEEE-754 bit patterns, so the
    /// result is identical to the in-memory `#[repr(C)]` layout.
    #[inline]
    pub fn to_words(&self) -> [u32; Self::WORDS] {
        [
            self.ty,
            self.life as u32,
            self.ctype,
            self.x.to_bits(),
            self.y.to_bits(),
            self.vx.to_bits(),
            self.vy.to_bits(),
            self.temp.to_bits(),
            self.flags,
            self.tmp as u32,
            self.tmp2 as u32,
            self.tmp3 as u32,
            self.tmp4 as u32,
            self.dcolour,
            self._pad0,
            self._pad1,
        ]
    }

    /// Rebuilds a particle from 16 words produced by [`GpuParticle::to_words`].
    #[inline]
    pub fn from_words(words: [u32; Self::WORDS]) -> Self {
        Self {
            ty: words[0],
            life: words[1] as i32,
            ctype: words[2],
            x: f32::from_bits(words[3]),
            y: f32::from_bits(words[4]),
            vx: f32::from_bits(words[5]),
            vy: f32::from_bits(words[6]),
            temp: f32::from_bits(words[7]),
            flags: words[8],
            tmp: words[9] as i32,
            tmp2: words[10] as i32,
            tmp3: words[11] as i32,
            tmp4: words[12] as i32,
            dcolour: words[13],
            _pad0: words[14],
            _pad1: words[15],
        }
    }
}

/// Behaviour of particles that leave the canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeMode {
    /// Particles are destroyed at the border.
    #[default]
    Void = 0,
    /// Particles bounce off the border.
    Solid = 1,
    /// Particles wrap around to the opposite side.
    Loop = 2,
}

/// Direction of gravity applied to particles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GravityMode {
    /// Gravity points towards the bottom of the canvas.
    #[default]
    Vertical = 0,
    /// No gravity.
    Off = 1,
    /// Gravity points towards the centre of the canvas.
    Radial = 2,
    /// Gravity follows the custom direction in the simulation config.
    Custom = 3,
}

/// Which parts of the Eulerian air simulation are active.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AirMode {
    /// Pressure and velocity are both simulated.
    #[default]
    On = 0,
    /// Pressure is held constant, velocity is simulated.
    PressureOff = 1,
    /// Velocity is held constant, pressure is simulated.
    VelocityOff = 2,
    /// Air simulation is disabled entirely.
    Off = 3,
    /// Air fields are kept but no longer advanced.
    NoUpdate = 4,
}

/// Canvas dimensions and global simulation toggles.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Canvas width in pixels; always a multiple of [`CELL`].
    pub xres: u32,
    /// Canvas height in pixels; always a multiple of [`CELL`].
    pub yres: u32,
    /// Whether the simulation is paused.
    pub paused: bool,
    /// Whether ambient heat exchange is enabled.
    pub aheat_enable: bool,
    /// Whether the prettier (slower) powder rendering is enabled.
    pub pretty_powder: bool,
    /// Whether the player may create the Cracker1000 mod elements
    /// ([`crate::constants::is_mod_element`]). Existing particles always keep
    /// simulating; this only hides them from the menus and refuses new ones.
    ///
    /// A global preference, so it stays out of the save layout: bincode is not
    /// self-describing and adding a serialized field would make every existing
    /// `.tptrs` file undecodable.
    #[serde(skip, default = "mod_elements_default")]
    pub mod_elements_enabled: bool,
    /// Border behaviour for particles.
    pub edge_mode: EdgeMode,
    /// Gravity direction.
    pub gravity_mode: GravityMode,
    /// Air simulation mode.
    pub air_mode: AirMode,
    /// Ambient air temperature in Kelvin.
    pub ambient_air_temp: f32,
    /// Pressure applied at the canvas border.
    pub edge_pressure: f32,
    /// `DISPLAY_*` bitmask.
    pub display_mode: u32,
    /// `COLOUR_*` bitmask.
    pub colour_mode: u32,
    /// `RENDER_*` bitmask controlling the fancy rendering passes.
    pub render_mode: u32,
    /// Strength of the additive fire glow (`0.0..=4.0`, TPT `fireIntensity`).
    pub fire_intensity: f32,
    /// Heat conduction and phase-change simulation.
    pub heat_enable: bool,
    /// Newtonian gravity (FFT Poisson solve over the coarse grid).
    pub newton_grav: bool,
    /// TPT water equalisation (`flood_water` connected-vessel levelling).
    pub water_equal_test: bool,
    /// Vorticity confinement coefficient for the air solver.
    pub vorticity_coeff: f32,
    /// `AIRC_*` air heat convection mode.
    pub convection_mode: u32,
    /// Ambient air velocity applied at the canvas border.
    pub edge_velocity_x: f32,
    /// Ambient air velocity applied at the canvas border (y axis).
    pub edge_velocity_y: f32,
    /// Custom gravity direction (used by [`GravityMode::Custom`]).
    pub custom_gravity_x: f32,
    /// Custom gravity direction (used by [`GravityMode::Custom`]).
    pub custom_gravity_y: f32,
    /// `TEMPSCALE_*` used by the HUD and options dialog.
    pub temperature_scale: u32,
    /// Whether particle decorations (`dcolour`) are rendered.
    pub decorations: bool,
    /// Simulation pixel under the pointer, `(-1, -1)` when it is off-canvas
    /// (TPT `Simulation::mousePos`, used by the portal debug lines).
    pub mouse_x: i32,
    /// Simulation pixel under the pointer, `(-1, -1)` when it is off-canvas.
    pub mouse_y: i32,
    /// Whether the Newtonian gravity force vectors are drawn
    /// (`RendererSettings::gravityFieldEnabled`).
    pub gravity_field_enabled: bool,
    /// Whether gravity-tool cells are marked with the 4x4 checker
    /// (`RendererSettings::gravityZonesEnabled`).
    pub gravity_zones_enabled: bool,
    /// Whether the element grid overlay is rendered.
    pub grid: bool,
    /// Grid cell size factor (TPT `gridSize`).
    pub grid_size: i32,
    /// Whether the checkerboard grid variant is rendered.
    pub grid_checkerboard: bool,
    /// Whether the in-game HUD (toolbars, telemetry) is visible.
    pub hud: bool,
    /// Whether the debug HUD telemetry overlay is visible.
    pub debug_hud: bool,
    /// Whether the brush cursor outline is rendered.
    pub show_brush: bool,
    /// Whether the zoom scope and magnification window are enabled.
    pub zoom_enabled: bool,
    /// Canvas X coordinate of the zoom magnification window.
    pub zoom_window_x: i32,
    /// Canvas Y coordinate of the zoom magnification window.
    pub zoom_window_y: i32,
    /// Magnification factor for the zoom window (TPT `ZFACTOR`).
    pub zoom_factor: i32,
    /// Canvas X coordinate of the zoom inspection scope.
    pub zoom_scope_x: i32,
    /// Canvas Y coordinate of the zoom inspection scope.
    pub zoom_scope_y: i32,
    /// Edge size in pixels of the square zoom inspection scope (TPT `zoomScopeSize`).
    pub zoom_scope_size: i32,
    /// Minimum temperature limit for heat display mode (`None` = auto).
    pub heat_display_limit_min: Option<f32>,
    /// Maximum temperature limit for heat display mode (`None` = auto).
    pub heat_display_limit_max: Option<f32>,
    /// Canvas X origin of the auto heat display limit sampling area.
    pub heat_display_auto_x: i32,
    /// Canvas Y origin of the auto heat display limit sampling area.
    pub heat_display_auto_y: i32,
    /// Width of the auto heat display limit sampling area.
    pub heat_display_auto_w: i32,
    /// Height of the auto heat display limit sampling area.
    pub heat_display_auto_h: i32,
}

/// Serde default for [`SimulationConfig::mod_elements_enabled`] when a payload
/// predates the field: the port ships the mod elements enabled.
#[inline]
fn mod_elements_default() -> bool {
    true
}

impl SimulationConfig {
    /// Creates a configuration for the given canvas size.
    ///
    /// `xres` and `yres` are clamped to at least [`MIN_CANVAS_DIM`] and rounded
    /// up to the next multiple of [`CELL`] so the air/wall grid tiles the canvas
    /// exactly.
    pub fn new(xres: u32, yres: u32) -> Self {
        Self {
            xres: align_dim(xres),
            yres: align_dim(yres),
            paused: false,
            aheat_enable: false,
            pretty_powder: false,
            // The port ships the mod elements, so they are available until the
            // user turns them off.
            mod_elements_enabled: true,
            edge_mode: EdgeMode::Void,
            gravity_mode: GravityMode::Vertical,
            air_mode: AirMode::On,
            ambient_air_temp: DEFAULT_TEMP_K,
            edge_pressure: 0.0,
            display_mode: 0,
            colour_mode: COLOUR_DEFAULT,
            render_mode: RENDER_DEFAULT,
            fire_intensity: 1.0,
            heat_enable: true,
            newton_grav: false,
            water_equal_test: false,
            vorticity_coeff: 0.1,
            convection_mode: AIRC_BOUSSINESQ,
            edge_velocity_x: 0.0,
            edge_velocity_y: 0.0,
            custom_gravity_x: 0.0,
            custom_gravity_y: 1.0,
            // The port displays Kelvin by default; `GameModel.cpp` defaults to
            // Celsius, but the scale is a preference, not a mechanic.
            temperature_scale: TEMPSCALE_KELVIN,
            decorations: true,
            mouse_x: -1,
            mouse_y: -1,
            gravity_field_enabled: false,
            gravity_zones_enabled: false,
            grid: false,
            grid_size: 1,
            grid_checkerboard: false,
            hud: true,
            debug_hud: false,
            show_brush: true,
            zoom_enabled: false,
            zoom_window_x: 0,
            zoom_window_y: 0,
            zoom_factor: 8,
            zoom_scope_x: 0,
            zoom_scope_y: 0,
            zoom_scope_size: 32,
            heat_display_limit_min: Some(MIN_TEMP),
            heat_display_limit_max: Some(MAX_TEMP),
            heat_display_auto_x: 0,
            heat_display_auto_y: 0,
            heat_display_auto_w: align_dim(xres) as i32,
            heat_display_auto_h: align_dim(yres) as i32,
        }
    }

    /// Number of air/wall cells along the x axis.
    #[inline]
    pub fn xcells(&self) -> u32 {
        self.xres / CELL
    }

    /// Number of air/wall cells along the y axis.
    #[inline]
    pub fn ycells(&self) -> u32 {
        self.yres / CELL
    }

    /// Total number of pixels on the canvas.
    #[inline]
    pub fn num_pixels(&self) -> usize {
        (self.xres * self.yres) as usize
    }

    /// Total number of air/wall cells on the canvas.
    #[inline]
    pub fn num_cells(&self) -> usize {
        (self.xcells() * self.ycells()) as usize
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self::new(DEFAULT_XRES, DEFAULT_YRES)
    }
}

/// Rounds a requested canvas dimension up to a valid, cell-aligned size.
#[inline]
fn align_dim(value: u32) -> u32 {
    value.max(MIN_CANVAS_DIM).next_multiple_of(CELL)
}

/// One pixel of a rigid-body cluster.
///
/// `local_x` / `local_y` are offsets from the body's centre of mass in body
/// space; the world position is
/// `(cm_x, cm_y) + rotate(angle) * (local_x, local_y)`.
/// `grid_x` / `grid_y` cache the last rasterised pixel coordinates so the
/// previous frame's spatial-map entries can be cleared.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidPixel {
    /// Local x offset from the centre of mass.
    pub local_x: f32,
    /// Local y offset from the centre of mass.
    pub local_y: f32,
    /// Element type of this pixel.
    pub elem_ty: u32,
    /// Temperature in Kelvin.
    pub temp: f32,
    /// Remaining life.
    pub life: i32,
    /// Element spawned on transition.
    pub ctype: u32,
    /// Decoration colour.
    pub dcolour: u32,
    /// Last rasterised world x coordinate.
    pub grid_x: i32,
    /// Last rasterised world y coordinate.
    pub grid_y: i32,
}

/// A Noita-style pixel-cluster rigid body.
///
/// The body pose is `(cm_x, cm_y, angle)`; every pixel is transformed from body
/// space into world space around the centre of mass, so the centre of mass is
/// always the origin of the local frame after
/// [`RigidBody2D::recompute_mass_properties`] has run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RigidBody2D {
    /// Unique body identifier.
    pub id: u32,
    /// Whether the body still participates in the simulation.
    pub active: bool,
    /// World-space centre-of-mass x coordinate.
    pub cm_x: f32,
    /// World-space centre-of-mass y coordinate.
    pub cm_y: f32,
    /// Linear velocity along x.
    pub vx: f32,
    /// Linear velocity along y.
    pub vy: f32,
    /// Rotation angle in radians.
    pub angle: f32,
    /// Angular velocity in radians per second.
    pub omega: f32,
    /// Total mass.
    pub mass: f32,
    /// Inverse total mass (`0` for massless bodies).
    pub inv_mass: f32,
    /// Rotational inertia about the centre of mass.
    pub inertia: f32,
    /// Inverse rotational inertia (`0` for degenerate bodies).
    pub inv_inertia: f32,
    /// Constituent pixels.
    pub pixels: Vec<RigidPixel>,
    /// Whether the pixels above have ever been backed by real particles.
    ///
    /// A body is appended to the simulation by
    /// `SimulationEngine::spawn_rigid_body`, which only records the pixel
    /// layout; the first rigid-body sync pass allocates the particles and sets
    /// this flag. It is what tells a *new* body (materialise it) apart from a
    /// body whose pixels have all been destroyed (deactivate it), which would
    /// otherwise look identical: no live rigid particle anywhere on the body.
    ///
    /// Derived state, not saved data: a restored body always comes back with
    /// its particles, so the first sync pass after loading re-establishes it.
    #[serde(skip)]
    pub materialised: bool,
}

impl RigidBody2D {
    /// Builds a solid rectangular body of `(2 * half_w + 1) x (2 * half_h + 1)`
    /// pixels centred on `(center_x, center_y)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_box(
        id: u32,
        center_x: f32,
        center_y: f32,
        half_w: i32,
        half_h: i32,
        elem_ty: u32,
        temp: f32,
        unit_weight: f32,
    ) -> Self {
        let half_w = half_w.max(0);
        let half_h = half_h.max(0);
        let width = (2 * half_w + 1) as usize;
        let height = (2 * half_h + 1) as usize;

        let mut pixels = Vec::with_capacity(width * height);
        for gy in -half_h..=half_h {
            for gx in -half_w..=half_w {
                pixels.push(new_pixel(center_x, center_y, gx, gy, elem_ty, temp));
            }
        }

        let mut body = Self::empty(id, center_x, center_y, pixels);
        body.recompute_mass_properties(|_| unit_weight);
        body
    }

    /// Builds a filled circular body of the given radius, centred on
    /// `(center_x, center_y)`.
    pub fn new_circle(
        id: u32,
        center_x: f32,
        center_y: f32,
        radius: i32,
        elem_ty: u32,
        temp: f32,
        unit_weight: f32,
    ) -> Self {
        let radius = radius.max(0);
        let r = radius as i64;
        let r_sq = r * r;

        let mut pixels = Vec::new();
        for gy in -radius..=radius {
            for gx in -radius..=radius {
                let gx64 = gx as i64;
                let gy64 = gy as i64;
                if gx64 * gx64 + gy64 * gy64 <= r_sq {
                    pixels.push(new_pixel(center_x, center_y, gx, gy, elem_ty, temp));
                }
            }
        }

        let mut body = Self::empty(id, center_x, center_y, pixels);
        body.recompute_mass_properties(|_| unit_weight);
        body
    }

    /// Creates a body with the given pixels and no motion, without computing
    /// mass properties.
    fn empty(id: u32, center_x: f32, center_y: f32, pixels: Vec<RigidPixel>) -> Self {
        Self {
            id,
            active: true,
            cm_x: center_x,
            cm_y: center_y,
            vx: 0.0,
            vy: 0.0,
            angle: 0.0,
            omega: 0.0,
            mass: 0.0,
            inv_mass: 0.0,
            inertia: 0.0,
            inv_inertia: 0.0,
            pixels,
            materialised: false,
        }
    }

    /// Recomputes total mass, centre of mass and rotational inertia.
    ///
    /// `unit_weight_fn` maps an element type to the mass of one pixel. Pixels
    /// are re-centred around the newly computed centre of mass, and the
    /// world-space `(cm_x, cm_y)` is moved by the rotated local offset so the
    /// body does not visually jump. The rotational inertia is the exact 2D
    /// parallel-axis sum
    /// `I = sum(m_i * (r_ix^2 + r_iy^2 + 1/6))`,
    /// where `1/6` is the moment of a unit square pixel about its own centre.
    pub fn recompute_mass_properties(&mut self, unit_weight_fn: impl Fn(u32) -> f32) {
        let mut total_mass = 0.0f32;
        let mut moment_x = 0.0f32;
        let mut moment_y = 0.0f32;

        for pixel in &self.pixels {
            let mass = unit_weight_fn(pixel.elem_ty).max(0.0);
            total_mass += mass;
            moment_x += mass * pixel.local_x;
            moment_y += mass * pixel.local_y;
        }

        if total_mass <= MASS_EPSILON {
            self.mass = 0.0;
            self.inv_mass = 0.0;
            self.inertia = 0.0;
            self.inv_inertia = 0.0;
            return;
        }

        // Local offset of the centre of mass relative to the current origin.
        let offset_x = moment_x / total_mass;
        let offset_y = moment_y / total_mass;

        if offset_x != 0.0 || offset_y != 0.0 {
            for pixel in &mut self.pixels {
                pixel.local_x -= offset_x;
                pixel.local_y -= offset_y;
            }
            // Keep the world pose unchanged: move the centre of mass by the
            // rotated local offset.
            let (sin_a, cos_a) = self.angle.sin_cos();
            self.cm_x += cos_a * offset_x - sin_a * offset_y;
            self.cm_y += sin_a * offset_x + cos_a * offset_y;
        }

        let mut inertia = 0.0f32;
        for pixel in &self.pixels {
            let mass = unit_weight_fn(pixel.elem_ty).max(0.0);
            let radius_sq = pixel.local_x * pixel.local_x + pixel.local_y * pixel.local_y;
            inertia += mass * (radius_sq + PIXEL_INERTIA);
        }

        self.mass = total_mass;
        self.inv_mass = 1.0 / total_mass;
        self.inertia = inertia;
        self.inv_inertia = if inertia > MASS_EPSILON {
            1.0 / inertia
        } else {
            0.0
        };
    }

    /// Transforms a point from body space into world space.
    #[inline]
    pub fn local_to_world(&self, local_x: f32, local_y: f32) -> Vec2 {
        let (sin_a, cos_a) = self.angle.sin_cos();
        Vec2::new(
            self.cm_x + cos_a * local_x - sin_a * local_y,
            self.cm_y + sin_a * local_x + cos_a * local_y,
        )
    }

    /// Velocity of a body-space point from the current linear and angular
    /// velocity: `v = v_cm + omega x r`.
    #[inline]
    pub fn point_velocity(&self, local_x: f32, local_y: f32) -> Vec2 {
        Vec2::new(
            self.vx - self.omega * local_y,
            self.vy + self.omega * local_x,
        )
    }
}

/// Builds one rigid-body pixel at the given body-space grid offset.
#[inline]
fn new_pixel(
    center_x: f32,
    center_y: f32,
    gx: i32,
    gy: i32,
    elem_ty: u32,
    temp: f32,
) -> RigidPixel {
    RigidPixel {
        local_x: gx as f32,
        local_y: gy as f32,
        elem_ty,
        temp,
        life: 0,
        ctype: PT_NONE,
        dcolour: 0,
        grid_x: (center_x + gx as f32).round() as i32,
        grid_y: (center_y + gy as f32).round() as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_particle_word_roundtrip() {
        let particle = GpuParticle {
            ty: PT_LAVA,
            life: -17,
            ctype: PT_STNE,
            x: 12.5,
            y: 300.25,
            vx: -0.75,
            vy: 2.0,
            temp: 1500.0,
            flags: FLAG_MOVABLE | FLAG_STAGNANT,
            tmp: 1,
            tmp2: -2,
            tmp3: 3,
            tmp4: -4,
            dcolour: 0x0012_34AB,
            _pad0: 0,
            _pad1: 0,
        };

        let words = particle.to_words();
        assert_eq!(words.len(), GpuParticle::WORDS);
        assert_eq!(words[3], 12.5f32.to_bits());
        assert_eq!(GpuParticle::from_words(words), particle);
    }

    #[test]
    fn simulation_config_aligns_and_clamps() {
        let config = SimulationConfig::new(100, 101);
        assert_eq!(config.xres, 100);
        assert_eq!(config.yres, 104);
        assert_eq!(config.xcells(), 25);
        assert_eq!(config.ycells(), 26);

        let tiny = SimulationConfig::new(1, 7);
        assert_eq!((tiny.xres, tiny.yres), (MIN_CANVAS_DIM, MIN_CANVAS_DIM));

        let default = SimulationConfig::default();
        assert_eq!((default.xres, default.yres), (DEFAULT_XRES, DEFAULT_YRES));
        assert_eq!(default.num_pixels(), 612 * 384);
        assert_eq!(default.num_cells(), 153 * 96);
    }

    #[test]
    fn box_mass_properties_are_exact() {
        let body = RigidBody2D::new_box(1, 100.0, 200.0, 1, 1, PT_STNE, 295.15, 1.0);

        assert_eq!(body.pixels.len(), 9);
        assert!(body.active);
        assert_eq!((body.cm_x, body.cm_y), (100.0, 200.0));
        assert!((body.mass - 9.0).abs() < 1e-6);
        assert!((body.inv_mass - 1.0 / 9.0).abs() < 1e-6);
        // sum(x^2 + y^2) over the 3x3 grid is 12, plus 9 pixels * 1/6.
        assert!((body.inertia - 13.5).abs() < 1e-4);
        assert!((body.inv_inertia - 1.0 / 13.5).abs() < 1e-4);
    }

    #[test]
    fn recompute_recentres_pixels_on_center_of_mass() {
        let mut body = RigidBody2D::new_box(2, 50.0, 50.0, 2, 2, PT_WOOD, 300.0, 2.0);
        // Remove a corner so the centre of mass is no longer at the origin.
        assert!(body.pixels.pop().is_some());
        body.recompute_mass_properties(|_| 2.0);

        let (mut sx, mut sy) = (0.0f32, 0.0f32);
        for pixel in &body.pixels {
            sx += pixel.local_x;
            sy += pixel.local_y;
        }
        assert!(sx.abs() < 1e-4, "local frame not centred: {sx}");
        assert!(sy.abs() < 1e-4, "local frame not centred: {sy}");
        assert!((body.mass - 48.0).abs() < 1e-6);
    }

    #[test]
    fn circle_has_symmetric_mass_and_velocity_helpers() {
        let mut body = RigidBody2D::new_circle(3, 64.0, 64.0, 4, PT_DMND, 295.15, 1.0);
        assert!(!body.pixels.is_empty());
        assert!((body.cm_x - 64.0).abs() < 1e-6);
        assert!((body.cm_y - 64.0).abs() < 1e-6);
        assert!((body.mass - body.pixels.len() as f32).abs() < 1e-6);

        body.vx = 2.0;
        body.vy = 1.0;
        body.omega = 0.5;
        let v = body.point_velocity(1.0, 0.0);
        assert!((v.x - 2.0).abs() < 1e-6);
        assert!((v.y - 1.5).abs() < 1e-6);

        body.angle = std::f32::consts::FRAC_PI_2;
        let world = body.local_to_world(1.0, 0.0);
        assert!((world.x - 64.0).abs() < 1e-4);
        assert!((world.y - 65.0).abs() < 1e-4);
    }
}
