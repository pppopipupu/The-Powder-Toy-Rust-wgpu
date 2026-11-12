//! The Powder Toy — GPU simulation core.
//!
//! `tpt-sim` owns the whole simulation: the GPU-resident buffers
//! ([`SimBuffers`]), the CubeCL kernel passes (air, thermal, reactions,
//! movement, rigid bodies, rendering) and the [`SimulationEngine`] that drives
//! them in the right order.
//!
//! ```no_run
//! use cubecl::client::Client;
//! use tpt_core::commands::SimCommand;
//! use tpt_core::constants::PT_SAND;
//! use tpt_core::particle::SimulationConfig;
//! use tpt_sim::SimulationEngine;
//!
//! # fn demo(client: Client) {
//! let mut engine = SimulationEngine::new(client, SimulationConfig::new(612, 384));
//! engine.apply_command(SimCommand::PlaceParticles {
//!     points: vec![(300, 300)],
//!     elem_ty: PT_SAND,
//!     replace_mode: false,
//!     dcolour: 0,
//! });
//! engine.step();
//! let rgba: &[u8] = engine.render_rgba();
//! assert_eq!(rgba.len(), 612 * 384 * 4);
//! # }
//! ```

// AGENTS.md's kernel-constant mirror convention requires compile-time guards
// (`const _: () = assert!(C_PT_X == PT_X)` and table tests such as
// `assert!(SPAWN_SNOW_K + 2.0 < 273.15)`). Those assertions are constant by
// design, so clippy's `assertions_on_constants` warning does not apply.
#![allow(clippy::assertions_on_constants)]

pub mod air_kernel;
pub mod buffers;
pub mod edit_kernel;
pub mod flood_kernel;
pub mod gol_kernel;
pub mod gravity_gpu;
pub mod gravity_kernel;
pub mod live_count;
pub mod move_kernel;
pub mod reaction_kernel;
pub mod render_kernel;
pub mod rigid_body;
pub mod thermal_kernel;
pub mod tool_kernel;
pub mod voxel3d;
pub mod water_equal_gpu;

mod engine_commands;
mod engine_render;

pub use buffers::{MIN_PARTICLE_CAPACITY, SimBuffers};

use cubecl::client::Client;

use tpt_core::constants::TYPE_ENERGY;
use tpt_core::elements::{ElementRegistry, GpuElementProps};
use tpt_core::particle::SimulationConfig;
use tpt_core::sign::{MAX_SIGNS, Sign};
use tpt_core::walls::{WallDef, default_walls};

/// Custom Game of Life rule added through the Lua API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomGolRule {
    /// Rule name without `DEFAULT_PT_LIFECUST_` prefix.
    pub name: String,
    /// Packed ruleset bits.
    pub ruleset: u32,
    /// Primary colour (`dcolour`).
    pub colour: u32,
    /// Secondary colour (`tmp`).
    pub colour2: u32,
}

/// The complete simulation: buffers, element definitions, wall definitions and
/// the frame counter used to seed the stochastic kernel passes.
pub struct SimulationEngine {
    /// GPU + host simulation state.
    pub bufs: SimBuffers,
    /// Element definitions (physics properties, colours, transitions).
    pub registry: ElementRegistry,
    /// Wall definitions (`WL_*` metadata used by the renderer and the HUD).
    pub walls: Vec<WallDef>,
    /// Number of simulation steps executed so far.
    pub frame_count: u64,
    /// Tick of the next allowed lightning bolt (TPT `lightningRecreate`).
    pub lightning_recreate: u64,
    /// Device-resident Newtonian-gravity FFT solver (padded transform buffers
    /// and the cached point-mass spectra).
    pub gravity: gravity_gpu::GravityFft,
    /// Device-resident rigid-body state (body/pixel records, claim and owner
    /// maps). The host [`tpt_core::particle::RigidBody2D`] list is only the
    /// description that is uploaded when it changes.
    pub rigid_gpu: rigid_body::RigidGpu,
    /// Game of Life scratch space and generation bookkeeping.
    pub gol: gol_kernel::GolState,
    /// Game of Life speed ratio (GSPEED).
    pub gspeed: u32,
    /// Live simulation framerate cap (`2.0` = uncapped, TPT `SimFpsLimit`).
    ///
    /// The application owns the persisted preference and seeds this mirror at
    /// startup; Lua's `tpt.fpsCap` reads and writes the mirror because the
    /// scripting bridge can only reach the engine. [`Self::caps_dirty`] tells
    /// the host to write the new value back to its preferences.
    pub fps_cap: f32,
    /// Live rendering framerate cap (`0` = follow the display).
    pub draw_cap: u32,
    /// [`Self::fps_cap`] / [`Self::draw_cap`] were changed by a script.
    pub caps_dirty: bool,
    /// Determinism flag for RNG / simulation.
    pub ensure_determinism: bool,
    /// 128-bit RNG state (4 x 32-bit words).
    pub rng_state: [u32; 4],
    /// UI replace mode flags (REPLACE_MODE = 1, SPECIFIC_DELETE = 2).
    pub replace_mode_flags: u32,
    /// Decoration colour space (0=sRGB, 1=Linear, 2=Gamma2.2, 3=Gamma1.8).
    pub deco_space: u32,
    /// Gravity grid display mode (0=off, 1=on).
    pub gravity_grid: u32,
    /// Number of frames queued to simulate.
    pub queued_frames: i32,
    /// Current save identifier and version (if any).
    pub current_save_id: Option<(i32, i32)>,
    /// Last particle index updated during step.
    pub last_updated_id: Option<i32>,
    /// Custom GoL rules.
    pub custom_gol: Vec<CustomGolRule>,
    /// Canvas text signs (`Simulation::signs`).
    ///
    /// Annotations, never physics, but part of the simulation state: the save
    /// file, the undo history and Lua's `sim.signs` all read them here.
    pub signs: Vec<Sign>,
}

impl SimulationEngine {
    /// Build a fresh, empty simulation.
    pub fn new(client: Client, config: SimulationConfig) -> Self {
        let registry = ElementRegistry::new_default();
        let walls = default_walls();
        let bufs = SimBuffers::new(client, config, &registry);
        Self {
            bufs,
            registry,
            walls,
            frame_count: 0,
            lightning_recreate: 0,
            gravity: gravity_gpu::GravityFft::new(),
            rigid_gpu: rigid_body::RigidGpu::new(),
            gol: gol_kernel::GolState::new(),
            gspeed: 1,
            // TPT's defaults; the application overwrites these from its
            // persisted preferences when it builds the state.
            fps_cap: 60.0,
            draw_cap: 0,
            caps_dirty: false,
            ensure_determinism: false,
            rng_state: [0x1234_5678, 0x9abc_def0, 0x0fed_cba9, 0x8765_4321],
            replace_mode_flags: 0,
            deco_space: 0,
            gravity_grid: 0,
            queued_frames: 0,
            current_save_id: None,
            last_updated_id: None,
            custom_gol: Vec::new(),
            signs: Vec::new(),
        }
    }

    /// Read-only access to the live configuration.
    pub fn config(&self) -> &SimulationConfig {
        &self.bufs.config
    }

    /// Mutable access to the live configuration (display mode, pause, gravity, …).
    pub fn config_mut(&mut self) -> &mut SimulationConfig {
        &mut self.bufs.config
    }

    /// Number of live particles.
    pub fn particle_count(&self) -> usize {
        self.bufs.active_count
    }

    /// Number of simulated rigid bodies.
    pub fn rigid_body_count(&self) -> usize {
        self.bufs.rigid_bodies.len()
    }

    /// The canvas signs.
    pub fn signs(&self) -> &[Sign] {
        &self.signs
    }

    /// Moves the sign at `index` to `(x, y)` (`SignWindow`'s move handle).
    pub fn move_sign(&mut self, index: usize, x: i32, y: i32) {
        if let Some(sign) = self.signs.get_mut(index) {
            sign.x = x;
            sign.y = y;
        }
    }

    /// Replaces every sign (`sim.signs = {...}`, load, undo).
    pub fn set_signs(&mut self, signs: Vec<Sign>) {
        self.signs = signs.into_iter().take(MAX_SIGNS).collect();
    }

    /// Appends a sign and returns its 1-based index, or `None` when the canvas
    /// already holds [`MAX_SIGNS`] (`LuaSimulation.cpp:1922-1938`).
    pub fn add_sign(&mut self, x: i32, y: i32, text: &str, ju: u8) -> Option<usize> {
        if self.signs.len() >= MAX_SIGNS {
            return None;
        }
        self.signs
            .push(Sign::new(x, y, text).with_justification(ju));
        Some(self.signs.len())
    }

    /// Removes the sign at 0-based `index`; out-of-range indices are ignored.
    pub fn remove_sign(&mut self, index: usize) -> bool {
        if index < self.signs.len() {
            self.signs.remove(index);
            true
        } else {
            false
        }
    }

    /// Push element property edits (Lua `elem.property` / `elem.allocate`) to the
    /// GPU element tables, but only when the registry actually changed.
    fn sync_element_tables(&mut self) {
        // This runs every tick, so the unchanged case must not allocate: the
        // displacement matrix alone is PT_NUM^2 words (~150 KB) and used to be
        // rebuilt and compared on every frame. Compare in place and only build
        // the owned tables when something really differs.
        let host_props = &self.bufs.elem_props_host;
        let host_can_move = &self.bufs.can_move_host;
        let registry_can_move = &self.registry.can_move;
        let props_same = self.registry.elements.len() == host_props.len()
            && self
                .registry
                .elements
                .iter()
                .zip(host_props.iter())
                .all(|(def, host)| {
                    bytemuck::bytes_of(&GpuElementProps::from_def(def)) == bytemuck::bytes_of(host)
                });
        // `to_gpu_can_move_u32` zero-fills past the registry matrix, so a host
        // copy of a different length can never match.
        let can_move_same = host_can_move.len()
            == tpt_core::constants::PT_NUM * tpt_core::constants::PT_NUM
            && host_can_move
                .iter()
                .zip(registry_can_move.iter())
                .all(|(&host, &reg)| host == u32::from(reg))
            && host_can_move
                .iter()
                .skip(registry_can_move.len())
                .all(|&host| host == 0);
        if props_same && can_move_same {
            return;
        }
        let props = self.registry.to_gpu_props();
        let can_move = self.registry.to_gpu_can_move_u32();
        self.bufs.elem_props_host = props;
        self.bufs.can_move_host = can_move;
        self.bufs.elem_is_energy = self
            .registry
            .elements
            .iter()
            .map(|def| def.properties & TYPE_ENERGY != 0)
            .collect();
        self.bufs.upload_element_tables();
        self.bufs.move_tables_stale = true;
    }

    /// Advance the simulation by one frame.
    ///
    /// The pass order follows TPT's `GameModel::BeforeSim` +
    /// `Simulation::UpdateParticles` (`Simulation.cpp:2295-2430`,
    /// `:3727-3918`) as closely as a device-resident, lock-free pipeline can:
    ///
    /// 1. upload host edits (brushes, walls, rigid bodies, save restores),
    /// 2. air pressure/velocity/ambient-heat pass plus the particle `HotAir`
    ///    contribution; the same step decays the wall electrical state and
    ///    refreshes the air/heat wall masks (`Air::update_air` + `update_airh`,
    ///    `Simulation.cpp:2347-2358`; the `NoUpdate` / `Off` gating lives
    ///    inside the pass),
    /// 3. Newtonian gravity dispatch (the mass snapshot is one frame behind,
    ///    matching `BeforeSim:3737-3742`),
    /// 4. `RecalcFreeParticles` equivalent: life decay, `pmap` / `photons`
    ///    rebuild, `pmap_count`, and the 1-in-10 `CheckStacking` pass,
    /// 5. heat conduction + phase/pressure transitions (`TransitionPhase`),
    /// 6. element reactions (`elements[t].Update`),
    /// 7. `Element::legacyUpdate` when heat simulation is off,
    /// 8. Game of Life generation (`SimulateGoL`, gated by `GSPEED`/`CGOL`),
    /// 9. lock-free particle movement,
    /// 10. water equalisation (`flood_water`; TPT calls it inside
    ///     `MovementPhase`, the batched port reads the rebuilt post-movement
    ///     map),
    /// 11. rigid-body reconciliation / integration / contacts / rasterisation,
    /// 12. the `EMP` screen-flash decay.
    ///
    /// # Known deviation: reaction/movement interleaving
    ///
    /// TPT processes **one particle at a time**: air deposit, velocity,
    /// `TransitionPhase`, `Update` and `MovementPhase` all happen for particle
    /// `i` before particle `i+1` starts, and later particles see the earlier
    /// ones' new positions, types, heats and deaths through `pmap`
    /// (`Simulation.cpp:2295-2430`). This port keeps the simulation
    /// device-resident and lock-free, so the thermal, reaction and movement
    /// phases are **batched**: every particle's reaction reads the pre-frame
    /// snapshot, then one winner-takes-the-pixel commit pass arbitrates all
    /// moves at once. The per-particle rules are C++-faithful, but two
    /// consequences of the batching remain and cannot be removed without
    /// reintroducing a host round trip or a sequential device scan:
    ///
    /// * a reaction triggered at slot `i` is not visible to slot `i+1` until
    ///   the next frame (a fire igniting one pixel cannot ignite its neighbour
    ///   in the same tick),
    /// * a liquid that vacates a pixel this frame is not entered by a later
    ///   liquid until the next frame (cyclic swaps resolve with a one-frame
    ///   lag).
    ///
    /// Everything else in this list is ordered like the reference.
    /// Uploads host edits without advancing the simulation.
    ///
    /// A paused canvas still has to show brush strokes and simulation tools:
    /// the reference writes the live `Simulation` state, so its renderer picks
    /// the edit up immediately even while paused. [`Self::step`] returns before
    /// the upload when paused, so the frame loop flushes them itself.
    pub fn flush_host_edits(&mut self) {
        if self.bufs.host_dirty {
            self.bufs.sync_host_to_gpu();
        }
    }

    pub fn step(&mut self) {
        if self.bufs.config.paused {
            return;
        }

        // A clear `gpu_dirty` means the caller explicitly read the device back
        // after the previous frame. Record it before the kernel passes set the
        // flag again; the rigid pass refreshes its host description once.
        if !self.bufs.gpu_dirty {
            self.rigid_gpu.note_explicit_sync();
        }

        let seed = (self.frame_count as u32)
            .wrapping_mul(0x9E37_79B9)
            .rotate_left(13)
            ^ 0x85EB_CA6B;

        // Element table edits (Lua). The wall electrical decay runs on the
        // device inside the air pass's wall-mask kernel.
        self.sync_element_tables();

        // Host -> GPU only for the categories a brush, wall edit, rigid body
        // or save actually touched.
        if self.bufs.host_dirty {
            self.bufs.sync_host_to_gpu();
        }

        // Air (`Air::update_air` + `update_airh`). The pass itself skips the
        // `update_air` kernels for `AIR_NOUPDATE` and still runs the ambient
        // heat kernel whenever `aheat_enable`, matching
        // `Simulation.cpp:3734-3736`.
        air_kernel::step_air_gpu(&mut self.bufs);

        // Newtonian gravity forces. TPT dispatches the FFT at the start of
        // `BeforeSim` from the mass accumulated by the previous frame's element
        // updates, then clears `gravIn.mass` (`Simulation.cpp:3737-3742`); the
        // solver consumes the previous frame's snapshot.
        if self.bufs.config.newton_grav {
            gravity_kernel::step_gravity_gpu(
                &mut self.bufs,
                &self.registry,
                &mut self.gravity,
                seed,
            );
        } else {
            // `Simulation::EnableNewtonianGravity(false)` resets `gravOut`
            // (`Simulation.cpp:3975-3983`). The force field feeds element
            // reactions (`DEUT`) and the fancy lensing pass even while the
            // model is off, so the last solve must be erased here as well as
            // at render time (a paused frame never reaches this branch).
            self.bufs.zero_gravity_forces();
        }

        // `Simulation::RecalcFreeParticles` (`Simulation.cpp:3347-3415`):
        // life decay, `pmap` / `photons` rebuild, `pmap_count`, then the
        // 1-in-10 `CheckStacking` pass (`Simulation.cpp:3620-3681`).
        let stacking_check = self.frame_count.is_multiple_of(10);
        self.bufs.recalc_free_particles_device(seed, stacking_check);

        // Heat + phase transitions (`TransitionPhase`). The conduction and
        // heat-exchange half is skipped when `heat_enable` is off, but the
        // transition kernel still runs the `LIFE` drain, wall sparks,
        // detonation and pressure rules (`Simulation.cpp:2767-2866`), exactly
        // like TPT's `legacy_enable` path; `Element::legacyUpdate` then runs
        // after the reactions below.
        // `legacy_enable` is read from the config before the mutable buffer
        // borrow so the call cannot overlap with `&mut self.bufs`.
        let legacy_enable = !self.bufs.config.heat_enable;
        thermal_kernel::step_thermal_and_transitions_gpu(
            &mut self.bufs,
            &self.registry,
            seed,
            legacy_enable,
        );

        // Reactions (`elements[t].Update`).
        reaction_kernel::step_reactions_gpu(&mut self.bufs, &self.registry, seed);

        // Legacy heat physics (`Element::legacyUpdate`,
        // `Simulation.cpp:2398-2400`): TPT runs it after the element update and
        // before movement whenever heat simulation is off.
        if !self.bufs.config.heat_enable {
            thermal_kernel::step_legacy_update_gpu(&mut self.bufs, &self.registry, seed);
        }

        // Game of Life. TPT runs `SimulateGoL` at the end of `BeforeSim`,
        // i.e. after the reactions and before the particle update, so a
        // generation's births join this frame's movement/map rebuild. The
        // generation gate lives in the device `counters[0]` word, so the host
        // never reads it back; `GSPEED` / `CGOL` decide how many frames pass
        // between generations (`Simulation.cpp:3885-3888`).
        // GoL births go through the same device creation queue as the reaction
        // passes: arm a fresh queue and free-slot window, run the generation,
        // then materialise its births in this tick.
        reaction_kernel::begin_pending_creates(&self.bufs);
        gol_kernel::step_gol_gpu(&mut self.bufs, &mut self.gol, &self.registry, self.gspeed);
        reaction_kernel::drain_pending_creates(&self.bufs, seed);

        // Movement (lock-free intent/winner/commit). The `e == 2` pair
        // resolution (`Simulation::try_move`, `Simulation.cpp:1258-1381`) is
        // part of this pass: it launches between the intent and commit kernels,
        // before either slot's position is written.
        move_kernel::step_movement_gpu(&mut self.bufs, &self.registry, seed);

        // Water equalisation (`flood_water`). TPT calls it from inside
        // `MovementPhase`, against the live map the movement loop is mutating
        // (`Simulation.cpp:3133-3135`); a batched device pass cannot read a
        // live map, so it reads the complete map `step_movement_gpu` has just
        // rebuilt for the frame. Its intent kernel still filters the
        // destination through `eval_move`, which reads the movement LUT this
        // same pass refreshed.
        if self.bufs.config.water_equal_test {
            water_equal_gpu::step_water_equal_gpu(&mut self.bufs, seed);
        }

        // The kernels changed device state; the host shadows are stale until a
        // subsystem that actually needs them asks for a read-back. Rendering is
        // fully GPU-resident, so the common frame never copies anything back.
        self.bufs.gpu_dirty = true;
        self.bufs.gpu_particles_dirty = true;
        self.bufs.gpu_maps_dirty = true;
        self.bufs.gpu_air_dirty = true;

        // Keep the HUD particle counter honest without blocking the render
        // loop: poll any finished reduction every tick and launch the next
        // async readback every half second.
        live_count::poll_live_count(&mut self.bufs);
        if self.frame_count.is_multiple_of(30) {
            live_count::start_live_count_async(&mut self.bufs);
        }

        // Rigid bodies. The device owns the steady state: the host solver is
        // only invoked when the rigid body description changed (spawn /
        // edit / save restore / destroy), and no particle, map or air buffer
        // is copied back on the per-frame path.
        rigid_body::step_rigid_bodies(&mut self.bufs, &mut self.rigid_gpu, &self.registry, 1.0);

        // TPT `Simulation::Update`: the `EMP` screen flash decays every tick.
        if self.bufs.emp_decor > 0 {
            self.bufs.emp_decor -= self.bufs.emp_decor / 25 + 2;
        }
        if self.bufs.emp_decor < 0 {
            self.bufs.emp_decor = 0;
        }

        self.frame_count = self.frame_count.wrapping_add(1);
        self.bufs.sim_tick = self.bufs.sim_tick.wrapping_add(1);
    }

    /// Resize the simulation canvas (aligned to `CELL`) and clear all state.
    pub fn resize_canvas(&mut self, xres: u32, yres: u32) {
        self.bufs.resize(xres, yres);
        self.rigid_gpu.invalidate();
        self.gol.reset();
        self.bufs.sync_host_to_gpu();
        self.frame_count = 0;
    }

    /// Pulls the device-resident rigid-body transforms / live pixel lists into
    /// `bufs.rigid_bodies`.
    ///
    /// Save capture, undo snapshots and debug inspectors call this after
    /// [`Self::ensure_host_current`]; the per-frame pipeline never needs it
    /// because the device is authoritative for rigid bodies.
    pub fn sync_rigid_bodies_to_host(&mut self) {
        self.rigid_gpu.sync_host(&mut self.bufs);
    }

    /// Remove every particle, wall, air value, glow and rigid body.
    pub fn clear(&mut self) {
        self.bufs.reset_all();
        self.rigid_gpu.invalidate();

        self.gol.reset();
        self.bufs.sync_host_to_gpu();
    }
    /// Computes a deterministic 32-bit hash of the current simulation state.
    pub fn compute_hash(&self) -> u32 {
        let mut h = 0x811c_9dc5u32;
        let prime = 0x0100_0193u32;
        h = (h ^ (self.bufs.active_count as u32)).wrapping_mul(prime);
        for i in 0..self.bufs.capacity.min(1024) {
            if self.bufs.part_ty[i] != 0 {
                h = (h ^ self.bufs.part_ty[i]).wrapping_mul(prime);
                h = (h ^ (self.bufs.part_x[i] as u32)).wrapping_mul(prime);
                h = (h ^ (self.bufs.part_y[i] as u32)).wrapping_mul(prime);
                h = (h ^ (self.bufs.part_temp[i] as u32)).wrapping_mul(prime);
            }
        }
        h
    }
}
