//! GPU-resident simulation buffers for the CubeCL + wgpu Powder Toy core.
//!
//! [`SimBuffers`] owns two mirrored copies of every simulation array:
//!
//! * a **host shadow** (`Vec<T>`) used for brush edits, Lua queries, rigid-body
//!   coupling, save/load and the HUD, and
//! * a **device buffer** (`cubecl::server::Handle`) that the CubeCL kernels read
//!   and write.
//!
//! Dirty tracking decides which side wins:
//!
//! | flag         | meaning                                                        |
//! |--------------|----------------------------------------------------------------|
//! | `host_dirty` | the host shadow changed and must be uploaded before kernels run |
//! | `gpu_dirty`  | a kernel ran and the host shadow must be read back first        |
//!
//! Particle edits are tracked per slot in [`SimBuffers::host_dirty_slots`], so a
//! brush stroke uploads only the handful of slots it created or freed instead of
//! rewinding the device pool to the last host read-back. [`FULL_PARTICLE_UPLOAD`]
//! in that list forces a whole-array upload for host passes that rewrite fields
//! directly.
//!
//! The particle pool grows on demand: [`SimBuffers::ensure_particle_capacity`]
//! doubles the pool and reallocates every GPU handle whenever a spawn needs more
//! room, up to [`SimBuffers::MAX_PARTICLE_CAPACITY`] (`0x7F_FFFF` slots, the
//! limit of the 23-bit `pmap` id field).
//!
//! Layout conventions (shared by every kernel in this crate):
//!
//! * particle arrays are SoA, indexed by particle id in `0..capacity`; a slot is
//!   free when `part_ty[id] == PT_NONE (0)`;
//! * `pmap` / `photons` are indexed `y * xres + x` and store
//!   [`tpt_core::constants::pmap_pack`] values (matter in `pmap`, `TYPE_ENERGY`
//!   in `photons`);
//! * air, wall, fan and fire grids are indexed `cy * xcells + cx` (cell = 4 px);
//! * `wall_bmap` holds the wall id in its low byte, `wall_emap` the electrical
//!   state/life in its low byte;
//! * `framebuf_rgba` is packed `0xAABBGGRR` (little-endian `[R, G, B, A]`) for
//!   `wgpu::TextureFormat::Rgba8Unorm`.

pub(crate) mod gpu_sync;
mod layout;
mod particles;
pub use particles::CreationSource;
mod particle_patch;
mod pool_clear;
mod spatial;

use cubecl::client::Client;
use cubecl::server::Handle;

use tpt_core::constants::{CELL, PT_NONE};
use tpt_core::elements::GpuElementProps;
use tpt_core::particle::{GpuParticle, RigidBody2D, SimulationConfig};

/// Smallest particle pool the engine will ever allocate (65 536 slots).
pub const MIN_PARTICLE_CAPACITY: usize = 65_536;

/// Bytes per element of every `u32`/`f32`/`i32` simulation array.
const WORD: usize = core::mem::size_of::<u32>();

/// Sentinel pushed into [`SimBuffers::host_dirty_slots`] when the whole particle
/// SoA has to be uploaded (a host-side pass wrote arbitrary fields directly).
pub const FULL_PARTICLE_UPLOAD: u32 = u32::MAX;

/// Snapshot of one pixel returned by [`SimBuffers::probe_pixel`].
#[derive(Clone, Debug)]
pub struct PixelProbe {
    /// Particle occupying the pixel, if any.
    pub particle: Option<GpuParticle>,
    /// Wall id in the pixel's `CELL x CELL` block (`0` = none).
    pub wall: u8,
    /// Wall electrical state in the same block.
    pub emap: u8,
    /// Air pressure in the block.
    pub pressure: f32,
}

pub struct SimBuffers {
    /// CubeCL client used to allocate, upload and read back device memory.
    pub client: Client,
    /// Live simulation configuration (resolution, modes, ambient temperature).
    pub config: SimulationConfig,

    /// Number of allocated particle slots (grows dynamically, never shrinks).
    pub capacity: usize,
    /// Number of live particles (slots whose `part_ty != PT_NONE`).
    pub active_count: usize,
    /// Highest slot ever handed out; kernels should iterate `0..capacity` and
    /// skip `part_ty == 0`, because live slots are not a contiguous prefix.
    pub next_slot: usize,
    /// Slots freed by [`SimBuffers::kill_particle`], reused before `next_slot`.
    pub free_list: Vec<usize>,
    /// Staging buffer for scattered particle uploads (see `particle_patch`).
    patch_buf: Option<Handle>,
    /// Capacity of [`Self::patch_buf`] in words; the buffer only ever grows.
    patch_words: usize,
    /// Particle slots whose host fields changed since the last upload.
    ///
    /// Filled by [`SimBuffers::allocate_particle`], [`SimBuffers::kill_particle`]
    /// and the command helpers; drained by `upload_particles`, which uploads
    /// only these sub-ranges of each SoA array. [`FULL_PARTICLE_UPLOAD`] forces a
    /// whole-array upload instead.
    pub host_dirty_slots: Vec<u32>,
    /// The host shadow changed and must be uploaded before the next kernel pass.
    pub host_dirty: bool,
    /// Host particle SoA changed (positions, types, life, …).
    pub host_particles_dirty: bool,
    /// Host `pmap` / `photons` changed.
    pub host_maps_dirty: bool,
    /// Host air grids (`air_*`, `fan_*`) changed.
    pub host_air_dirty: bool,
    /// Host wall grids (`wall_*`, `block_air*`) changed.
    pub host_walls_dirty: bool,
    /// Only the electrical wall state + masks changed (per-frame emap decay).
    pub host_wall_state_dirty: bool,
    /// A kernel wrote device memory; the host shadow is stale until
    /// [`SimBuffers::sync_gpu_to_host`] runs.
    pub gpu_dirty: bool,
    /// Device particle SoA changed since the last particle read-back.
    pub gpu_particles_dirty: bool,
    /// Device `pmap` / `photons` changed since the last map read-back.
    pub gpu_maps_dirty: bool,
    /// Device air grids changed since the last air read-back.
    pub gpu_air_dirty: bool,
    /// `true` once a device reaction step has started since the last reset,
    /// meaning slots inside [`Self::GPU_SPAWN_START`]..[`Self::GPU_SPAWN_END`]
    /// may hold (or are about to hold) device-spawned particles.
    pub gpu_spawn_used: bool,

    pub part_ty: Vec<u32>,
    pub gpu_part_ty: Handle,
    pub part_life: Vec<i32>,
    pub gpu_part_life: Handle,
    pub part_ctype: Vec<u32>,
    pub gpu_part_ctype: Handle,
    pub part_x: Vec<f32>,
    pub gpu_part_x: Handle,
    pub part_y: Vec<f32>,
    pub gpu_part_y: Handle,
    pub part_vx: Vec<f32>,
    pub gpu_part_vx: Handle,
    pub part_vy: Vec<f32>,
    pub gpu_part_vy: Handle,
    pub part_temp: Vec<f32>,
    pub gpu_part_temp: Handle,
    /// Per-slot equilibrium temperature computed by the thermal conduction pass
    /// (`TransitionPhase`'s `pt`), `-1` when the conduction gate did not pass.
    /// Pure device scratch: the host shadow is never read back.
    pub part_equil: Vec<f32>,
    pub gpu_part_equil: Handle,
    pub part_flags: Vec<u32>,
    pub gpu_part_flags: Handle,
    pub part_tmp: Vec<i32>,
    pub gpu_part_tmp: Handle,
    pub part_tmp2: Vec<i32>,
    pub gpu_part_tmp2: Handle,
    /// Element-specific scratch value (TPT `tmp3`).
    pub part_tmp3: Vec<i32>,
    pub gpu_part_tmp3: Handle,
    /// Element-specific scratch value (TPT `tmp4`).
    pub part_tmp4: Vec<i32>,
    pub gpu_part_tmp4: Handle,
    pub part_dcolour: Vec<u32>,
    pub gpu_part_dcolour: Handle,

    /// Matter particle per pixel (`y * xres + x`), [`pmap_pack`] encoded.
    pub pmap: Vec<u32>,
    pub gpu_pmap: Handle,
    /// `TYPE_ENERGY` particle per pixel, [`pmap_pack`] encoded.
    pub photons: Vec<u32>,
    pub gpu_photons: Handle,
    /// Number of stacked matter particles per pixel (`Simulation::pmap_count`).
    ///
    /// Rebuilt every frame by
    /// [`SimBuffers::recalc_free_particles_device`]; `CheckStacking` uses it to
    /// collapse extreme stacks into `NBHL`. Excludes `THDR` / `EMBR` / `FIGH` /
    /// `PLSM`, exactly like `RecalcFreeParticles` (`Simulation.cpp:3381-3384`).
    pub pmap_count: Vec<u32>,
    pub gpu_pmap_count: Handle,
    /// Per particle desired target pixel (`ny * xres + nx`, `0xFFFF_FFFF` = stay).
    pub intent_target: Handle,
    /// Per pixel atomic move winner: `(weight_rank << 23) | (part_id + 1)`.
    pub cell_winner: Handle,

    pub air_vx: Vec<f32>,
    pub gpu_air_vx: Handle,
    pub air_vy: Vec<f32>,
    pub gpu_air_vy: Handle,
    pub air_pv: Vec<f32>,
    pub gpu_air_pv: Handle,
    pub air_hv: Vec<f32>,
    pub gpu_air_hv: Handle,
    pub air_ovx: Vec<f32>,
    pub gpu_air_ovx: Handle,
    pub air_ovy: Vec<f32>,
    pub gpu_air_ovy: Handle,
    pub air_opv: Vec<f32>,
    pub gpu_air_opv: Handle,
    pub air_ohv: Vec<f32>,
    pub gpu_air_ohv: Handle,
    pub fan_vx: Vec<f32>,
    pub gpu_fan_vx: Handle,
    pub fan_vy: Vec<f32>,
    pub gpu_fan_vy: Handle,
    /// Wall id in the low byte (`WL_*`, `0` = none).
    pub wall_bmap: Vec<u32>,
    pub gpu_wall_bmap: Handle,
    /// Wall electrical state / life in the low byte.
    pub wall_emap: Vec<u32>,
    pub gpu_wall_emap: Handle,
    /// `1` where the wall blocks air movement.
    pub block_air: Vec<u32>,
    pub gpu_block_air: Handle,
    /// `1` where the wall blocks heat conduction.
    pub block_airh: Vec<u32>,
    pub gpu_block_airh: Handle,
    /// Element-driven `bmap_blockair` writes (`WALL` / `RUBR` / `ALUM`), one
    /// `u32` per air cell.
    ///
    /// Device-only scratch: the element passes stamp it on the device, the next
    /// frame's air mask rebuild folds it into the wall mask, and a canvas
    /// resize recreates it zeroed. There is no host shadow because no host pass
    /// ever needs the value.
    pub gpu_elem_block_air: Handle,
    /// Element-driven `bmap_blockairh` writes, one `u32` per air cell.
    ///
    /// Device-only, exactly like [`Self::gpu_elem_block_air`]: the element
    /// passes write it and the heat-conduction mask rebuild consumes it.
    pub gpu_elem_block_airh: Handle,

    pub fire_r: Vec<u32>,
    pub gpu_fire_r: Handle,
    pub fire_g: Vec<u32>,
    pub gpu_fire_g: Handle,
    pub fire_b: Vec<u32>,
    pub gpu_fire_b: Handle,
    /// Per-cell effect flags, one `u32` per coarse cell.
    ///
    /// Bit 0 marks a pixel with a `PMODE_BLOB`/`PMODE_GLOW`/`PMODE_BLUR` draw
    /// mode, bit 1 one with `PMODE_SPARK`/`PMODE_FLARE`/`PMODE_LFLARE`. The
    /// particle passes write them while drawing, and the glow and spark gathers
    /// read the 3x3 neighbourhood to skip regions with no effect at all, which
    /// is what makes a canvas of sand or steam cheap to render.
    pub fx_cell: Vec<u32>,
    pub gpu_fx_cell: Handle,
    /// Rows (`0..yres`) followed by columns (`yres..yres + xres`) that hold at
    /// least one spark/flare pixel.
    ///
    /// The ray pass walks up to `RAY_STEPS` pixels along the four axis
    /// directions, so a pixel can only receive a ray from its own row or its
    /// own column. Both halves share one buffer to stay inside the 16-binding
    /// ceiling of the particle passes.
    pub fx_axis: Vec<u32>,
    pub gpu_fx_axis: Handle,
    /// Composite of the fire glow, one packed `0xAABBGGRR` word per pixel.
    ///
    /// The glow composite is a nine-tap gather over the cell grid; it only
    /// changes when the fire cells change, i.e. once per simulation tick, while
    /// the framebuffer is redrawn at the display rate. Caching it here keeps
    /// the per-frame cost at one add per pixel instead of nine taps.
    pub fire_layer: Vec<u32>,
    pub gpu_fire_layer: Handle,
    /// Packed `0xAABBGGRR` framebuffer, `xres * yres` entries.
    pub framebuf_rgba: Vec<u32>,
    pub gpu_framebuf: Handle,
    /// TPT `Simulation::emp_decor`: screen-wide `EMP` flash strength.
    ///
    /// Grows by `3` per triggered `EMP` (capped at `40`) and decays by
    /// `emp_decor / 25 + 2` per tick; the render pass blends a blue tint over
    /// the whole canvas while it is positive.
    pub emp_decor: i32,

    /// Simulation steps taken so far.
    ///
    /// TPT renders and simulates in lockstep, so its fire buffer (accumulate,
    /// bloom, decay) advances exactly once per tick. The port presents frames
    /// faster than it steps, so the fire passes are gated on this counter
    /// instead of on the render call: see [`Self::fire_tick`].
    pub sim_tick: u64,
    /// [`Self::sim_tick`] value the fire buffer was last advanced at.
    pub fire_tick: u64,

    /// Mass density per air cell (`gravIn.mass`), accumulated by element updates.
    pub grav_mass: Vec<f32>,
    pub gpu_grav_mass: Handle,
    /// Snapshot of the previous frame's mass grid.
    ///
    /// TPT dispatches the FFT at the start of `BeforeSim` and clears
    /// `gravIn.mass` immediately afterwards (`Simulation.cpp:3737-3742`), so
    /// frame N's forces come from the mass written during frame N-1. The port
    /// keeps the current build in [`Self::gpu_grav_mass`] and the snapshot the
    /// solver consumes here.
    pub grav_mass_prev: Vec<f32>,
    pub gpu_grav_mass_prev: Handle,
    /// Mass the `PGRV`/`NGRV` tools wrote into `gravIn.mass` for the next solve.
    ///
    /// TPT clears `gravIn.mass` right after each gravity dispatch, so a painted
    /// well lasts exactly one solve. The port keeps the tool contributions in
    /// this separate grid: [`crate::gravity_kernel::step_gravity_gpu`] uploads
    /// it once, seeds the device accumulator with it, and the seeding kernel
    /// zeroes it again on the device.
    pub grav_tool_mass: Vec<f32>,
    pub gpu_grav_tool_mass: Handle,
    /// `true` while [`Self::grav_tool_mass`] holds writes not yet uploaded.
    pub grav_tool_mass_dirty: bool,
    /// One-shot mass the element passes inject into the next gravity solve,
    /// one `i32` per air cell in units of `1/65536` mass.
    ///
    /// This is the shared channel for every element whose `Update` assigns
    /// `gravIn.mass[cell]`: `CEXP` (`CEXP.cpp:80`) and `GSNS`
    /// (`GSNS.cpp:104`). Writers go through
    /// [`crate::gravity_kernel::set_grav_mass_override`] during the element
    /// pass; [`crate::gravity_kernel::step_gravity_gpu`] folds exactly one
    /// tick of writes into the accumulator it snapshots (after the element
    /// build, before the copy the next solve reads), and the modext clear
    /// kernel zeroes the grid on the device at the start of the next element
    /// pass. The injection therefore lasts one tick and there is no host
    /// shadow; every canvas resize recreates it zeroed.
    pub gpu_grav_mass_override: Handle,
    /// `1` in each cell [`Self::gpu_grav_mass_override`] holds a write for.
    ///
    /// TPT's elements *assign* `gravIn.mass[cell]`, and a forced `0` is a
    /// meaningful value (`CEXP` with `tmp == 0`, a `GSNS` reading zero), so the
    /// value grid needs a companion flag to tell "written zero" from "not
    /// written this tick". [`Self::gpu_grav_mass_override`] and this grid are
    /// written together by the element passes and zeroed together by the
    /// modext clear kernel; one `u32` per air cell, device-only.
    pub gpu_grav_mass_override_set: Handle,
    /// Gravity propagation mask per air cell (`gravIn.mask`, `0` = blocked).
    pub grav_mask: Vec<u32>,
    pub gpu_grav_mask: Handle,
    /// Force per air cell in pixels/step² (`gravOut.forceX`).
    pub grav_fx: Vec<f32>,
    pub gpu_grav_fx: Handle,
    /// Force per air cell in pixels/step² (`gravOut.forceY`).
    pub grav_fy: Vec<f32>,
    pub gpu_grav_fy: Handle,
    /// `true` when the wall layout changed and [`Self::grav_mask`] must be rebuilt.
    pub grav_mask_dirty: bool,
    /// `true` when the device mass grid changed since the last solver run.
    pub grav_mass_dirty: bool,
    /// `true` while [`Self::grav_mass`], [`Self::grav_fx`] and [`Self::grav_fy`]
    /// are stale.
    ///
    /// The gravity solve runs entirely on the device, so the host shadows are
    /// only refreshed by [`Self::sync_gravity_gpu_to_host`] when a Lua grid
    /// getter or a save capture asks for them.
    pub grav_shadow_dirty: bool,
    /// `true` while the device force grids may hold a force from an earlier
    /// solve.
    ///
    /// TPT resets `gravOut` to zero when Newtonian gravity is switched off
    /// (`Simulation::EnableNewtonianGravity`, `Simulation.cpp:3975-3983`); the
    /// renderer's lensing pass reads `gravOut` regardless of the model
    /// (`Renderer.cpp:42-47`), so [`Self::zero_gravity_forces`] uploads a zero
    /// field once the model is off. `false` means both sides are zero.
    pub grav_force_stale: bool,

    /// Transient per-cell `DMG` blast origin, one word per air cell.
    ///
    /// The word is `(strength << 24) | (x << 12) | y` — the blast strength in
    /// the high byte and the pixel the charge detonated on in the low `24`
    /// bits — and `0` means "this cell is not inside a blast". The reaction
    /// pass's force stage zeroes the grid on the device at
    /// the start of every frame, stamps the strongest blast into the cells it
    /// covers and reads it back in the same frame, so the grid is never saved
    /// and the host shadow below is never uploaded or read back: it exists only
    /// so this grid has a host mirror like every other array here. A scene reset
    /// therefore does not have to clear it — the device copy is rebuilt before
    /// anything reads it, so a stale shadow can never reach the device.
    pub blast_cell: Vec<u32>,
    pub gpu_blast_cell: Handle,

    /// One word: the longest `FRAY` beam any emitter can fire this frame,
    /// recomputed on the device by `fray_range_kernel` and read by the `FRAY`
    /// gather to bound its ray walk. Device-only scratch; never uploaded.
    pub gpu_fray_range: Handle,

    /// Scratch buffer for water equalisation: per-column surface info.
    pub gpu_water_col_surf: Handle,
    /// Scratch buffer for water equalisation: per-column destination target.
    pub gpu_water_col_target: Handle,
    /// Scratch buffer for water equalisation: per-pixel arbitration winner.
    pub gpu_water_winner: Handle,

    /// Device free-slot table for zero-copy particle creation (`[avail, cursor, slots..]`).
    pub gpu_free_slots: Handle,
    /// One-word live-particle counter, reduced on the device by
    /// [`crate::live_count`] so the HUD never reads `part_ty` back.
    pub gpu_live_count: Handle,
    /// One-word gate for the wire `set_emap` closure: set by the parallel
    /// [`crate::move_kernel::emap_gate_kernel`] when an uncharged wire span sits
    /// next to charged wire, and consumed (and cleared) by the single-threaded
    /// closure, which otherwise has nothing to propagate. Device-only.
    pub gpu_emap_gate: Handle,
    /// Scratch list of packed `(x, y)` brush pixels for the device-side edit
    /// kernels (`[x | y << 16]`), [`Self::EDIT_POINT_CAPACITY`] entries.
    pub gpu_edit_points: Handle,
    /// RNG rolls for the `MIX` tool kernel, three words per edited pixel
    /// ([`Self::tool_rolls_capacity`] entries).
    pub gpu_edit_rolls: Handle,
    /// Movement LUT (element meta + `PT_NUM * PT_NUM` can-move matrix), rebuilt
    /// only when the element registry changes.
    pub gpu_move_lut: Handle,
    /// Legacy per-element physics allocation. The physics words live in the
    /// tail of `gpu_move_lut`, so only word 0 is bound: the `LOVE` / `LOLZ`
    /// `elementCount` gate (`reaction_kernel::love`).
    pub gpu_elem_phys: Handle,
    /// Per-element "lives in the photons layer" mask.
    pub gpu_photon_mask: Handle,
    /// `true` when the three movement tables above are out of date.
    pub move_tables_stale: bool,
    /// Deferred device-side particle creations: `[count, record..]`,
    /// `GPU_CREATE_COUNT * CREATE_WORDS + 1` `u32` words (see
    /// `reaction_kernel::shared::request_create`).
    pub gpu_create_queue: Handle,
    /// Work list of the per-tick connected-component floods
    /// (`reaction_kernel::flood_component`), one pixel index per word. Each
    /// pixel is pushed at most once per flood, so `pixels` words are enough.
    pub gpu_flood_queue: Handle,
    /// `[head, tail]` atomic counters of [`Self::gpu_flood_queue`]; the drain
    /// resets both when it finishes, so the next seed scan starts from zero.
    pub gpu_flood_counters: Handle,
    /// Per-pixel visited marks of the flood drain, `pixels` words. The drain
    /// clears a pixel's mark when it pops it, which leaves the buffer zero for
    /// the next flood.
    pub gpu_flood_marks: Handle,
    /// Portal slot header word (`((donated_slot + 1) << 9) | ty`), `PORTAL_CAPACITY` entries.
    pub gpu_portal_ty: Handle,
    /// Portal payload words (`PORTAL_CAPACITY * PORTAL_WORDS` entries:
    /// `life, ctype, temp_bits, tmp, tmp2, vx_bits, vy_bits, dcolour`).
    pub gpu_portal_data: Handle,
    /// Wireless channel state (`PORTAL_CHANNELS * 2` entries: `[active_now, active_next]`).
    pub gpu_wireless: Handle,
    /// Stickman (`FIGH` / `STKM` / `STKM2`) leg positions (`STKM_SLOTS * STKM_LEGS_STRIDE` `f32`s).
    pub gpu_stkm_legs: Handle,
    /// Stickman leg accelerations (`STKM_SLOTS * STKM_ACCS_STRIDE` `f32`s).
    pub gpu_stkm_accs: Handle,
    /// Stickman metadata (`STKM_SLOTS * STKM_META_STRIDE` `u32`s:
    /// `spwn, comm, pcomm, elem, frames, rocket_boots, fan, spawn_id`).
    pub gpu_stkm_meta: Handle,
    /// Host-owned live control words for the two player stickmen,
    /// `[comm1, pcomm1, comm2, pcomm2]`. Ports
    /// `Simulation::player.comm` / `player2.comm` (`Stickman.h:4-16`,
    /// `GameController.cpp:611-730`). Passed to the stickman kernel as
    /// scalars every tick so no full-state readback is needed.
    pub stickman_control: [u32; 4],

    /// `GpuElementProps[PT_NUM]` uploaded once per registry change.
    pub gpu_elem_props: Handle,
    /// Flat `PT_NUM * PT_NUM` can-move matrix (`0` block, `1` swap, `2` pass, `3` special).
    pub gpu_can_move: Handle,
    /// Host copy of the element property table (used to refresh the GPU buffer).
    pub elem_props_host: Vec<GpuElementProps>,
    /// Host copy of the can-move matrix.
    pub can_move_host: Vec<u32>,
    /// `true` for element ids whose state flag contains [`TYPE_ENERGY`].
    pub elem_is_energy: Vec<bool>,
    /// `SPAWN_DEFAULT_WORDS` `u32`s per element: TPT's `DefaultProperties`, read
    /// by the device-side creation drain (`ElementRegistry::to_gpu_spawn_defaults_u32`).
    pub gpu_spawn_defaults: Handle,
    /// Host copy of [`Self::gpu_spawn_defaults`].
    pub spawn_defaults_host: Vec<u32>,

    pub rigid_bodies: Vec<RigidBody2D>,
    pub next_rigid_id: u32,
}

impl SimBuffers {
    /// First particle slot reserved for zero-copy device-side allocation
    /// (`ARAY`, `DRAY`, `PSTN`, `PRTO`, `FIGH` and the creation-queue drain).
    pub const GPU_SPAWN_START: usize = 1024;
    /// Number of particle slots in the device-side allocation window. It matches
    /// [`Self::GPU_CREATE_COUNT`] so one frame's deferred creations cannot starve
    /// the passes that allocate directly (`portals`, `rays`, `stickman`).
    pub const GPU_SPAWN_COUNT: usize = 16384;
    /// One-past-the-end slot of the device-side allocation window (`17 408`).
    pub const GPU_SPAWN_END: usize = Self::GPU_SPAWN_START + Self::GPU_SPAWN_COUNT;
    /// Records the deferred creation queue can hold in one frame. Reactions push
    /// a record per particle they create; the drain at the end of the reaction
    /// step allocates them from [`Self::GPU_SPAWN_START`]..[`Self::GPU_SPAWN_END`].
    /// Excess pushes are dropped, exactly as TPT drops creations once the pool
    /// is full.
    pub const GPU_CREATE_COUNT: usize = 16384;

    /// Pixels one brush stroke can queue for a device-side edit kernel.
    ///
    /// A stroke covers at most one frame's brush footprint; the largest brush
    /// the UI offers is far below this, and a longer list is truncated rather
    /// than growing the buffer mid-stroke.
    pub const EDIT_POINT_CAPACITY: usize = 1 << 16;

    /// Words reserved for the `MIX` tool's per-pixel RNG rolls.
    pub const fn tool_rolls_capacity() -> usize {
        3 * Self::EDIT_POINT_CAPACITY
    }
    /// `u32` words per creation record: `slot, ty, x, y, mask, life, ctype,
    /// temp, tmp, tmp2, tmp3, tmp4, dcolour, flags, vx, vy` (field indices and
    /// the mask layout live in `reaction_kernel::shared`).
    pub const CREATE_WORDS: usize = 16;

    /// Number of live particles (scans the host shadow).
    pub fn count_active_particles(&self) -> usize {
        self.part_ty.iter().filter(|ty| **ty != PT_NONE).count()
    }

    /// Number of live particles as maintained incrementally.
    pub fn particle_count(&self) -> usize {
        self.active_count
    }

    /// Highest slot ever used; kernels should still iterate `0..capacity`.
    pub fn slot_bound(&self) -> usize {
        self.next_slot
    }

    /// Prefix of the particle pool that may contain live slots on the device.
    ///
    /// Includes the device spawn window (`GPU_SPAWN_START..GPU_SPAWN_END`) once
    /// any reaction step has run.
    #[inline]
    pub fn used_slot_bound(&self) -> usize {
        let bound = if self.gpu_spawn_used {
            self.next_slot.max(Self::GPU_SPAWN_END)
        } else {
            self.next_slot
        };
        bound.clamp(1, self.capacity)
    }

    /// Pixel index for `(x, y)`, `None` when outside the canvas.
    #[inline]
    pub fn pixel_index(&self, x: i32, y: i32) -> Option<usize> {
        if x < 0 || y < 0 || x >= self.config.xres as i32 || y >= self.config.yres as i32 {
            return None;
        }
        Some(y as usize * self.config.xres as usize + x as usize)
    }

    /// Cell index (`xcells * ycells` grid) for pixel `(x, y)`.
    #[inline]
    pub fn cell_index(&self, x: i32, y: i32) -> Option<usize> {
        if x < 0 || y < 0 || x >= self.config.xres as i32 || y >= self.config.yres as i32 {
            return None;
        }
        let cx = x as usize / CELL as usize;
        let cy = y as usize / CELL as usize;
        Some(cy * self.config.xcells() as usize + cx)
    }
}
