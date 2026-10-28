//! Element reactions — the chemical layer of the simulation.
//!
//! The passes run after the air and thermal kernels and before the movement
//! kernel. [`energy_entry`] is the exception: it is the movement step's
//! `try_move` `e == 2` pair resolution, launched between the intent and commit
//! launches from `move_kernel::step_movement_gpu`, not from this driver. It is split into several CubeCL kernels because each one only reads
//! the SoA arrays it actually needs, which keeps every shader well inside the
//! guaranteed `max_storage_buffers_per_shader_stage` budget:
//!
//! | kernel | work |
//!--------|------|
//! | [`life::life_and_local_reaction_kernel`] | `PROP_LIFE_DEC` / `PROP_LIFE_KILL*`, fire dying into smoke/steam, spark discharging back into its conductor |
//! | [`combustion`] | `FIRE` / `PLSM` / `LAVA` heating + 5x5 ignition, `H2` / `O2` chemistry, explosives and cords, `SOAP` / `GEL` / `TRON` / `LIGH` / fireworks |
//! | [`fluid::fluid_reaction_kernel`] | water extinguishing fire, salt dissolving, ice/snow melting, `CLNE` capture/spawn, `VOID` eating |
//! | [`corrosion`] | `ACID` / `CAUS` corroding neighbours and igniting explosives, `ACID` life timers, `BMTL` rusting |
//! | [`energy::energy_reaction_kernel`] | `ELEC` sparking/electrolysis, `NEUT` transmutations, `BIZR`/`BIZRG`/`BIZRS` decoration painting |
//! | [`electronics`] | the electrical layer: staged spark conduction, switches, powered devices, storage, sensors (`FILT` serialisation), arcs, `ETRD` plasma bolts |
//! | [`nuclear`] | decay chains, shields, isotopes, `PTNM`, `PROT`, viruses, `WARP`, converters |
//! | [`special`] | force/pressure materials, plants, residue powders, `DMG`, `SEED` / `VINE`, refraction, `QRTZ`, photon sources |
//! | [`love`] | the `LOVE` / `LOLZ` 9x9 rule-table blocks |
//! | [`force`] | gather-model `ACEL` / `DCEL` / `HEAC` / `RPEL` / `FRAY`, magnets, projectiles |
//! | [`pipe`] | staged `PIPE` / `PPIP` handshake transport and pumps |
//! | [`shared::drain_gpu_creates`] | allocates and initialises every creation the passes above queued |
//! | [`spawn`] | expands the `part_flags` spawn-request bits of the passes without a creation-queue binding into queue records |
//! | [`portals`] | channel-indexed `PRTI` / `PRTO` queues plus the `WIFI` bus |
//! | [`rays`] | `ARAY` / `DRAY` wavelength rays and the `PSTN` / `FRME` piston rigs |
//! | [`stickman`] | `FIGH` / `STKM` / `STKM2` Verlet rigs and fighter AI |
//! | [`wall::wall_conduction_kernel`] | conductive walls charging from a spark, electrified walls sparking conductors |
//! | [`energy_entry::pair_resolve_kernel`] | `try_move`'s shared-pixel (`e == 2`) pair resolution; runs between the movement intent and commit launches, not with the passes above |
//!
//! # Module map
//!
//! [`shared`] holds the constants and the `#[cube]` helpers used by several
//! passes (element-word offsets, wall mirrors, hashes, spawn bits, the creation
//! queue). Each other module is one pass; this file is only the launch driver.
//!
//! # Design
//!
//! * The kernels only *read* the spatial maps. Every particle type change or
//!   death is expressed by writing `part_ty`, so the GPU pass stays free of
//!   cross-thread map races, and the host never pulls the pass back to service
//!   a creation: both creation routes resolve on the device.
//! * Particle *creation* has two routes. The device-side creation queue
//!   ([`shared::request_create`], drained by [`shared::drain_gpu_creates`] right
//!   after [`shared::refresh_gpu_free_slots`]) allocates from
//!   `SimBuffers::GPU_SPAWN_START..` and writes a full field set: the element's
//!   `DefaultProperties` plus whatever the record overrides (`life`, `ctype`,
//!   `temp`, `tmp`..`tmp4`, `dcolour`, `vx`/`vy`), so a queued creation is what
//!   TPT's `create_part` would have produced, live for the movement and render
//!   passes of the same frame. The second route is a request bit in `part_flags`,
//!   used where a pass has no room for the queue binding; it can only express a
//!   handful of shapes (a clone of `ctype`, a copy of `tmp`, a `type|life` word),
//!   and [`spawn`] expands it into queue records later in the same tick. A spawn
//!   frame refreshes the particle and map shadows with the narrow read-backs
//!   first, because allocating on the host mutates both and a stale
//!   upload would clobber the device state.
//! * Particle *death* is simply `part_ty = PT_NONE`; the next particle
//!   read-back reconciles the pool, so no explicit free-list bookkeeping is
//!   required on the device.

mod combustion;
mod corrosion;
mod electronics;
mod energy;
pub(crate) mod energy_entry;
mod fluid;
mod force;
mod glow;
mod life;
mod love;
mod neut;
mod nuclear;
mod pipe;
mod portals;
mod rays;
pub(crate) mod shared;
mod spawn;
mod special;
mod stickman;
mod wall;

use cubecl::prelude::*;
use tpt_core::constants::*;
use tpt_core::elements::ElementRegistry;

use crate::buffers::SimBuffers;

use shared::{
    CREATE_QUEUE_WORDS, ELEM_PROP_WORDS, drain_gpu_creates, launch_grid, refresh_gpu_free_slots,
    reset_gpu_create_queue,
};

/// Sizes shared by every launch of one reaction step.
struct PassDims {
    capacity: usize,
    /// Prefix of the pool that can hold live slots
    /// ([`SimBuffers::used_slot_bound`]). Slot-parallel passes launch over this
    /// instead of `capacity`: a 1024x768 canvas reserves 1.57 M slots but
    /// typically holds tens of thousands, and the kernels only ever visit a
    /// slot below the bound anyway.
    used_slots: usize,
    pixels: usize,
    cells: usize,
    xres: u32,
    yres: u32,
    xcells: u32,
    elem_len: usize,
    frame_seed: u32,
}

/// Arms the creation queue for a pass that runs after the reaction step.
///
/// The reaction step drains its own records, so the queue is empty again; the
/// Game of Life births are queued afterwards and materialised by
/// [`drain_pending_creates`] in the same tick. The free-slot window is rebuilt
/// because the reaction drain consumed part of it.
pub(crate) fn begin_pending_creates(bufs: &SimBuffers) {
    reset_gpu_create_queue(bufs);
    refresh_gpu_free_slots(bufs);
}

/// Materialises every creation queued by [`begin_pending_creates`]'s users.
pub(crate) fn drain_pending_creates(bufs: &SimBuffers, frame_seed: u32) {
    let dims = PassDims {
        capacity: bufs.capacity,
        used_slots: bufs.used_slot_bound(),
        pixels: bufs.config.num_pixels(),
        cells: bufs.config.num_cells(),
        xres: bufs.config.xres,
        yres: bufs.config.yres,
        xcells: bufs.config.xcells(),
        elem_len: PT_NUM * ELEM_PROP_WORDS,
        frame_seed,
    };
    drain_gpu_creates(bufs, &dims);
}

/// 1. Life timers, fire burnout, spark discharge.
fn launch_life_pass(bufs: &SimBuffers, dims: &PassDims) {
    let (cubes, dim) = launch_grid(dims.used_slots);
    unsafe {
        life::life_and_local_reaction_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.frame_seed,
            dims.used_slots as u32,
            dims.xcells,
        );
    }
}

/// Gravity direction index for the launch kick of `FIRW` / `FWRK`.
///
/// Mirrors `special::gravity_dir_index`: 0..7 are the eight compass directions
/// and 8 means "undecided" (`GravityMode::Off` / `Radial`), which the kernel
/// replaces with a random kick. The firework kernel now evaluates the full
/// `GetGravityField` vector, so this helper only backs the table test.
#[cfg(test)]
fn launch_gravity_dir(config: &tpt_core::particle::SimulationConfig) -> u32 {
    use tpt_core::particle::GravityMode;

    let (gx, gy) = match config.gravity_mode {
        GravityMode::Vertical => (0.0f32, 1.0f32),
        GravityMode::Custom => (config.custom_gravity_x, config.custom_gravity_y),
        GravityMode::Off | GravityMode::Radial => (0.0f32, 0.0f32),
    };
    let len = (gx * gx + gy * gy).sqrt();
    let (mut nx, mut ny) = (0.0f32, 0.0f32);
    if len > 0.0f32 {
        nx = gx / len;
        ny = gy / len;
    }
    let mut gxi = 1u32;
    let mut gyi = 1u32;
    if nx > 0.5f32 {
        gxi = 2u32;
    }
    if nx < -0.5f32 {
        gxi = 0u32;
    }
    if ny > 0.5f32 {
        gyi = 2u32;
    }
    if ny < -0.5f32 {
        gyi = 0u32;
    }
    match (gxi, gyi) {
        (0u32, 0u32) => 1u32,
        (0u32, 1u32) => 2u32,
        (0u32, 2u32) => 3u32,
        (1u32, 0u32) => 0u32,
        (1u32, 1u32) => 8u32,
        (1u32, 2u32) => 4u32,
        (2u32, 0u32) => 7u32,
        (2u32, 1u32) => 6u32,
        _ => 5u32,
    }
}

/// 2. Fire / plasma / lava ignition, hydrogen and oxygen chemistry.
///
/// The explosive and special-element `Update` functions follow the base pass.
/// Each of them stays inside the sixteen-buffer ceiling, so the pass is split
/// by binding set rather than by element:
///
/// | kernel | bindings | elements |
/// |--------|----------|----------|
/// | [`combustion::combustion_reaction_kernel`] | 16 | `FIRE` / `PLSM` / `LAVA` / `H2` / `O2` / `GOO` / `SPNG` |
/// | [`combustion::o2_burst_kernel`] | 15 | white-hot `O2` graviton burst (`gravOut` force field) |
/// | [`combustion::firework_kernel`] | 8 | `FIRW` / `FWRK` |
/// | [`combustion::reactive_liquid_kernel`] | 9 | `LITH` / `EXOT` / `SING` |
/// | [`combustion::exot_freeze_kernel`] | 5 | cold `EXOT` stops moving |
/// | [`combustion::soap_kernel`] | 12 | `SOAP` bubble state, attach and the `OIL` foam coupling |
/// | [`combustion::soap_detach_kernel`] | 9 | `SOAP` pop on contact |
/// | [`combustion::soap_spring_kernel`] | 7 | `SOAP` bubble spring |
/// | [`combustion::gel_kernel`] | 7 | `GEL` |
/// | [`combustion::ligh_heat_kernel`] | 11 | `LIGH` heating |
/// | [`combustion::ligh_prop_kernel`] | 7 | `LIGH` segments, `C5` photon |
/// | [`combustion::explosive_reaction_kernel`] | 11 | `BOMB` / `DEST` / `BANG` / `EMBR` |
/// | [`combustion::fuse_kernel`] | 7 | `FUSE` / `FSEP` / `IGNT` (creates go through the queue) |
/// | [`combustion::pressure_gas_kernel`] | 6 | `ANAR` / `BOYL` / `C5` cold trigger |
/// | [`combustion::thermite_kernel`] | 7 | `THRM` burning, `LAVA(THRM/PLUT)` |
/// | [`combustion::tron_kernel`] | 8 | `TRON` |
/// | [`combustion::emp_kernel`] | 8 | `EMP` pulse |
/// | [`combustion::carbonated_kernel`] | 10 | `CBNW` degassing, blast timer/charge, rubber ignition |
/// | [`combustion::glass_kernel`] | 9 | `GLAS` pressure shatter, `LNTG` condensation |
fn launch_combustion_pass(bufs: &SimBuffers, dims: &PassDims) {
    let ycells = (dims.cells / dims.xcells as usize) as u32;
    let aheat = u32::from(bufs.config.aheat_enable);
    // `FIRW`/`FWRK` use the full `GetGravityField` vector, not just a compass
    // index (`FWRK.cpp:58`, `FIRW.cpp:69`).
    let grav_mode = bufs.config.gravity_mode as u32;
    let custom_gx = bufs.config.custom_gravity_x;
    let custom_gy = bufs.config.custom_gravity_y;

    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        combustion::combustion_reaction_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(
                bufs.gpu_spawn_defaults.clone(),
                PT_NUM * tpt_core::elements::SPAWN_DEFAULT_WORDS,
            ),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    // `CBNW.cpp:51-145` and `GLAS.cpp:50-90`: the full soda-water update and
    // the glass pressure/condensation update. They run after the base pass so
    // a type or `life` change it made (the sponge turning soda into `CO2`,
    // molten salt strengthening the pane) is what they read.
    let legacy = u32::from(!bufs.config.heat_enable);
    unsafe {
        combustion::carbonated_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
            legacy,
        );
    }
    unsafe {
        combustion::glass_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_hv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    // `O2` white-hot graviton burst (`O2.cpp:80-123`). It needs the solved
    // force field, which the sixteen-binding shared combustion pass has no
    // room for, so it runs as a small follow-up pass over the same pixels.
    unsafe {
        combustion::o2_burst_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_fx.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_fy.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            BufferArg::from_raw_parts(
                bufs.gpu_spawn_defaults.clone(),
                PT_NUM * tpt_core::elements::SPAWN_DEFAULT_WORDS,
            ),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    // `FIRW` / `FWRK` launch and burst.
    unsafe {
        combustion::firework_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), dims.cells),
            BufferArg::from_raw_parts(
                bufs.gpu_move_lut.clone(),
                crate::move_kernel::build_move_lut_len(),
            ),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
            grav_mode,
            custom_gx,
            custom_gy,
        );
    }

    // `LITH` / `EXOT` / `SING`.
    unsafe {
        combustion::reactive_liquid_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            // Signature order is `air_pv` then `create_queue`; swapping them
            // sends every `request_create` into the pressure buffer and every
            // pressure write into the queue.
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            BufferArg::from_raw_parts(
                bufs.gpu_spawn_defaults.clone(),
                PT_NUM * tpt_core::elements::SPAWN_DEFAULT_WORDS,
            ),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.xcells,
            ycells,
            dims.frame_seed,
        );
    }

    // `SOAP` bubble state, then the slot-parallel spring.
    unsafe {
        combustion::soap_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_fx.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_fy.clone(), dims.cells),
            bufs.config.gravity_mode as u32,
            bufs.config.custom_gravity_x,
            bufs.config.custom_gravity_y,
            u32::from(bufs.config.newton_grav),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.used_slots as u32,
        );
    }
    // Pop-on-contact needs the element table, so it is its own pass.
    unsafe {
        combustion::soap_detach_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
            ycells,
        );
    }
    let (slot_cubes, slot_dim) = launch_grid(dims.used_slots);
    // Cold `EXOT` freezes in place (`EXOT.cpp:200-210`).
    unsafe {
        combustion::exot_freeze_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            dims.used_slots as u32,
        );
    }
    unsafe {
        combustion::soap_spring_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            dims.used_slots as u32,
            dims.xres,
            dims.yres,
            bufs.config.edge_mode as u32,
        );
    }

    // `GEL`.
    unsafe {
        combustion::gel_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }

    // `LIGH`: heating and ignition first, then the segment state machine.
    unsafe {
        combustion::ligh_heat_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_hv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
            aheat,
        );
    }
    unsafe {
        combustion::ligh_prop_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }

    // `BANG.cpp:74-80`: `flood_prop(x, y, FIELD_TMP, 2)` arms the whole
    // connected run of `BANG` in one call. A dispatch carries the mark a single
    // 4-connected hop, so the flood runs `BANG_FLOOD_ROUNDS` times per frame —
    // the `pipe::pipe_flood_kernel` cascade pattern. The mark is monotone and
    // stays on the armed pixels, so a charge wider than the per-frame reach
    // still converges on the following frames. The flood writes the
    // `BANG_ARMED_NEXT_FRAME` / `BANG_REARMED` sentinels rather than the final
    // `2`, because it runs before the explosive pass: `explosive_reaction_kernel`
    // turns them into `2`, and only the pixels that were already at `2` advance
    // to `3` this frame, so a run armed by the flood detonates on the same tick
    // as the seed that started it.
    for _ in 0..combustion::BANG_FLOOD_ROUNDS {
        unsafe {
            combustion::bang_flood_kernel::launch_unchecked(
                &bufs.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(dim),
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
                BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
                dims.xres,
                dims.yres,
            );
        }
    }

    // Explosives and fuses.
    unsafe {
        combustion::explosive_reaction_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    // `FUSE` / `FSEP` / `IGNT`.
    unsafe {
        combustion::fuse_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    // `ANAR` / `BOYL` / the `C5` cold trigger.
    unsafe {
        combustion::pressure_gas_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.xcells,
            ycells,
            dims.frame_seed,
        );
    }

    // `THRM`.
    unsafe {
        combustion::thermite_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    // `TRON`.
    unsafe {
        combustion::tron_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.xcells,
            ycells,
            dims.frame_seed,
        );
    }

    // `EMP` pulse (slot parallel, inert unless a fresh EMP latched a pulse).
    unsafe {
        combustion::emp_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.used_slots as u32,
            dims.frame_seed,
        );
    }
}

/// 3. Water, salt, ice, snow, clones and voids.
fn launch_fluid_pass(bufs: &SimBuffers, dims: &PassDims) {
    let legacy = u32::from(!bufs.config.heat_enable);
    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        fluid::fluid_reaction_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(
                bufs.gpu_spawn_defaults.clone(),
                PT_NUM * tpt_core::elements::SPAWN_DEFAULT_WORDS,
            ),
            dims.xres,
            dims.yres,
            dims.frame_seed,
            legacy,
        );
    }
}

/// 5. Electronics.
fn launch_energy_pass(bufs: &SimBuffers, dims: &PassDims) {
    let legacy = u32::from(!bufs.config.heat_enable);
    let (cubes, dim) = launch_grid(dims.used_slots);
    unsafe {
        energy::energy_reaction_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            BufferArg::from_raw_parts(
                bufs.gpu_spawn_defaults.clone(),
                PT_NUM * tpt_core::elements::SPAWN_DEFAULT_WORDS,
            ),
            dims.used_slots as u32,
            dims.xres,
            dims.yres,
            dims.frame_seed,
            legacy,
        );
    }
}

/// 6. Electrical layer: staged spark conduction, switches, powered devices,
///    storage, sensors, arcs and detectors.
///
/// The sub-passes run in TPT's dependency order: the snapshot freezes the frame,
/// spark conduction consumes it, the devices react to those sparks, the sensors
/// compute their trigger state, source ignition turns triggers and batteries
/// into sparks, and the two newest passes (arcs/delays and detectors) run last
/// because they can create sparks that nothing else in this frame depends on.
fn launch_electronics_passes(bufs: &SimBuffers, dims: &PassDims) {
    // 6.1 Snapshot every slot into the `intent_target` scratch. The movement
    // pass fully rewrites `intent_target` before it reads it, and the
    // reaction pass runs first in the frame, so borrowing it here is safe.
    let (cubes, dim) = launch_grid(dims.used_slots);
    unsafe {
        electronics::spark_stage_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.intent_target.clone(), dims.capacity),
            dims.used_slots as u32,
        );
    }

    // 6.2 `SPRK.cpp`: conduction, receiver side.
    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        electronics::spark_conduction_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.intent_target.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    // 6.2b `SPRK.cpp:377-381` / `Simulation.cpp:586-700`: the conduction pass
    // has just charged the `INST` pixel it reached; carry that spark along the
    // rest of the connected `INST` run.
    electronics::launch_flood_inst_pass(bufs, dims);

    unsafe {
        electronics::switch_device_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    unsafe {
        electronics::storage_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.capacity as u32,
        );
    }

    // 6.5 `PSNS` (`FILT` broadcast) and `LSNS` (life scan). Two launches, five
    // and seven storage buffers, so both stay inside the device limit.
    unsafe {
        electronics::sensor_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.xcells,
        );
    }
    unsafe {
        electronics::light_sensor_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            dims.xres,
            dims.yres,
        );
    }

    // 6.5b `TSNS` and `VSNS`: the spark latch, the temperature / velocity scan
    // and the `FILT` chain serialisation (`TSNS.cpp`, `VSNS.cpp`).
    unsafe {
        electronics::temp_sensor_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
        );
    }
    unsafe {
        electronics::velocity_sensor_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
        );
    }

    unsafe {
        electronics::source_ignition_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    // 6.6 `WIRE.cpp` / `PUMP.cpp` / `THDR.cpp`: the element updates this layer
    // has no other pass for (`BTRY.cpp` is the receiver branch of the source
    // pass just above). They run after the sources they can react to.
    electronics::launch_element_update_pass(bufs, dims);

    unsafe {
        electronics::arc_and_delay_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    // 6.8 `LSNS` modes 1 and 3 (`FILT` life serialisation). Separate launch:
    // `sensor_kernel` above binds six buffers and has no room for the
    // cross-pixel `ctype` writes this mode performs.
    unsafe {
        electronics::lsns_filter_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            dims.xres,
            dims.yres,
        );
    }

    // 6.9 `ETRD` plasma bolts. Pushes `PLSM` creations into the queue; the drain
    // in `step_reactions_gpu` allocates them after the reaction passes.
    electronics::bolt::launch_etrd_bolt_pass(bufs, dims);

    unsafe {
        electronics::detector_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
        );
    }
}

/// 7. Wall conduction.
fn launch_wall_pass(bufs: &SimBuffers, dims: &PassDims) {
    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        wall::wall_conduction_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.xcells,
        );
    }
}

/// Run the whole reaction pass for one frame.
///
/// Order matters: life timers run first so a particle that dies this frame
/// cannot ignite anything, the electrical pass runs after the energy pass
/// because sparks are created there, and the wall pass runs last because it
/// reacts to the sparks the energy and electronics passes just created.
pub fn step_reactions_gpu(bufs: &mut SimBuffers, registry: &ElementRegistry, frame_seed: u32) {
    let capacity = bufs.capacity;
    let xres = bufs.config.xres;
    let yres = bufs.config.yres;
    if capacity == 0 || xres == 0 || yres == 0 {
        return;
    }
    let xcells = bufs.config.xcells();
    let ycells = bufs.config.ycells();
    if xcells == 0 || ycells == 0 {
        return;
    }

    // Keep the device copy of the element table in step with the registry. The
    // engine normally uploads it, but a freshly allocated pool or a custom
    // element would otherwise react with stale properties.
    if bufs.elem_props_host.len() != registry.elements.len() {
        bufs.refresh_element_tables(registry);
    }

    // The drain below creates into the reserved window this very tick, so the
    // bound has to cover it before the passes are sized. Right after a read-back
    // the window is clean and `next_slot` alone would leave the slots a creation
    // lands in unvisited for one frame.
    bufs.gpu_spawn_used = true;
    let dims = PassDims {
        capacity,
        used_slots: bufs.used_slot_bound(),
        pixels: (xres * yres) as usize,
        cells: (xcells * ycells) as usize,
        xres,
        yres,
        xcells,
        elem_len: PT_NUM * ELEM_PROP_WORDS,
        frame_seed,
    };

    // The creation queue is per-frame: empty it before any pass can push.
    reset_gpu_create_queue(bufs);

    // `Simulation::BeforeSim` runs the LOVE/LOLZ block scan before the `WIRE`
    // tally (`Simulation.cpp:3862`) and before every element update, so it sees
    // the map `RecalcFreeParticles` just rebuilt. The port keeps it first in the
    // reaction step for the same reason: a pass that has already killed a
    // particle leaves its `pmap` word behind, and that cell must still look
    // occupied to the rule table here.
    love::launch(bufs, &dims);
    launch_life_pass(bufs, &dims);
    launch_combustion_pass(bufs, &dims);
    launch_fluid_pass(bufs, &dims);
    glow::launch(bufs, &dims);
    corrosion::launch_corrosion_pass(bufs, &dims);
    launch_energy_pass(bufs, &dims);
    neut::launch(bufs, &dims);
    launch_electronics_passes(bufs, &dims);
    nuclear::launch(bufs, &dims);
    special::launch(bufs, &dims);
    force::launch(bufs, &dims);
    pipe::launch(bufs, &dims);
    spawn::expand_spawn_requests(bufs, &dims);
    refresh_gpu_free_slots(bufs);
    // Every creation the passes above asked for is allocated here, before the
    // passes that allocate from the device window themselves.
    drain_gpu_creates(bufs, &dims);
    portals::launch(bufs, &dims);
    rays::launch(bufs, &dims);
    stickman::launch(bufs, &dims);
    launch_wall_pass(bufs, &dims);

    // They run after the life timers, which may have just killed particles,
    // and before movement — the order of TPT's `UpdateParticles`.
    crate::gravity_kernel::step_gravity_element_updates(bufs, frame_seed);

    // Every kernel above only touched device memory; the host shadows stay
    // stale until a subsystem asks for a narrow read-back.
    bufs.gpu_particles_dirty = true;
    bufs.gpu_maps_dirty = true;
    bufs.gpu_dirty = true;
}

#[cfg(test)]
mod tests {
    use super::shared::{CUBE_DIM, ELEM_PROP_WORDS, WORD_FLAMMABLE, WORD_HARDNESS, launch_grid};
    use super::shared::{
        W_ALLOWALLELEC, W_ALLOWLIQUID, W_DETECT, W_EHOLE, W_EWALL, W_WALLELEC, WORD_EXPLOSIVE,
        WORD_PROPERTIES,
    };
    use tpt_core::constants::*;

    #[test]
    fn wall_mirrors_match_core_constants() {
        assert_eq!(W_WALLELEC, WL_WALLELEC as u32);
        assert_eq!(W_EWALL, WL_EWALL as u32);
        assert_eq!(W_DETECT, WL_DETECT as u32);
        assert_eq!(W_ALLOWLIQUID, WL_ALLOWLIQUID as u32);
        assert_eq!(W_ALLOWALLELEC, WL_ALLOWALLELEC as u32);
        assert_eq!(W_EHOLE, WL_EHOLE as u32);
    }

    #[test]
    fn element_word_offsets_match_gpu_layout() {
        use core::mem::size_of;
        use tpt_core::elements::GpuElementProps;

        assert_eq!(size_of::<GpuElementProps>(), ELEM_PROP_WORDS * 4);
        let props = GpuElementProps::default();
        let base = &props as *const GpuElementProps as usize;
        assert_eq!(
            (&props.properties as *const u32 as usize) - base,
            WORD_PROPERTIES as usize * 4
        );
        assert_eq!(
            (&props.flammable as *const i32 as usize) - base,
            WORD_FLAMMABLE as usize * 4
        );
        assert_eq!(
            (&props.explosive as *const i32 as usize) - base,
            WORD_EXPLOSIVE as usize * 4
        );
        assert_eq!(
            (&props.hardness as *const i32 as usize) - base,
            WORD_HARDNESS as usize * 4
        );
    }

    #[test]
    fn launch_grid_covers_units() {
        assert_eq!(launch_grid(0), (1, CUBE_DIM));
        assert_eq!(launch_grid(1), (1, CUBE_DIM));
        assert_eq!(launch_grid(CUBE_DIM as usize), (1, CUBE_DIM));
        assert_eq!(launch_grid(CUBE_DIM as usize + 1), (2, CUBE_DIM));
    }

    #[test]
    fn launch_gravity_dir_matches_the_special_table() {
        use tpt_core::particle::{GravityMode, SimulationConfig};

        let vertical = SimulationConfig {
            gravity_mode: GravityMode::Vertical,
            ..Default::default()
        };
        assert_eq!(super::launch_gravity_dir(&vertical), 4);
        for mode in [GravityMode::Off, GravityMode::Radial] {
            let config = SimulationConfig {
                gravity_mode: mode,
                ..Default::default()
            };
            assert_eq!(super::launch_gravity_dir(&config), 8);
        }

        // The eight compass directions, mirroring `special::gravity_dir_index`.
        let cases = [
            (0.0f32, -1.0f32, 0u32),
            (-1.0, -1.0, 1),
            (-1.0, 0.0, 2),
            (-1.0, 1.0, 3),
            (0.0, 1.0, 4),
            (1.0, 1.0, 5),
            (1.0, 0.0, 6),
            (1.0, -1.0, 7),
            (0.0, 0.0, 8),
        ];
        for (gx, gy, expected) in cases {
            let config = SimulationConfig {
                gravity_mode: GravityMode::Custom,
                custom_gravity_x: gx,
                custom_gravity_y: gy,
                ..Default::default()
            };
            assert_eq!(
                super::launch_gravity_dir(&config),
                expected,
                "gravity ({gx}, {gy})"
            );
        }
    }
}
