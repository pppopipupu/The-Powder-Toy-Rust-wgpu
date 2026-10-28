//! Special / force element updates (`GRAV`, `VINE`, `SEED`, `BASE`, ...).
//!
//! This pass ports the TPT `Update` functions that the other reaction passes do
//! not bind: the plants, the pressure materials, the light emitters and the
//! residue powders. It is split into seven kernel modules so every one stays inside
//! ten storage buffers:
//!
//! | module / kernel | work | storage buffers |
//! |-----------------|------|-----------------|
//! | [`force::special_force_kernel`] | `GRAV`, `VIBR`/`BVBR`, `MERC` state | 9 |
//! | [`force::merc_trade_snapshot_kernel`] | `MERC` volume snapshot for the staged exchange | 3 |
//! | [`force::merc_trade_kernel`] | `MERC` four-trade volume equalisation | 4 |
//! | [`air::special_air_kernel`] | `RFRG`/`RFGL`, `BASE`, `BREC`, `SEED` (pressure), `VIBR` detonation | 12 |
//! | [`pressure::special_pressure_kernel`] | `TUNG` (melt, `O2` detonation, `tmp3` brittleness, velocity kick), `QRTZ` pressure rupture | 10 |
//! | [`qrtz::qrtz_kernel`] | `QRTZ`/`PQRT` salt absorption, growth and the diffusion *send* | 10 |
//! | [`qrtz::qrtz_receive_kernel`] | `QRTZ` diffusion *receive* | 4 |
//! | [`transform::special_transform_kernel`] | `PLNT` (tree growth, seeds, leaves), `GOLD`, `RSST`/`RSSS`, `VINE`, `SEED`, `SLCN`, `CRAY`, `VIBR`/`BVBR`, `MERC` (spill), `BREC`/`BRMT` | 13 |
//! | [`transform::seed_breed_kernel`] | `SEED` breeding genome swap | 8 |
//! | [`photon::special_photon_kernel`] | `GOLD` neutron shielding, the `POLO` decay chain and its neutron releases | 10 |
//!
//! The `QRTZ` crystal spreads its `tmp` (absorbed salt) through the crystal by
//! *message passing*: [`qrtz::qrtz_kernel`] lets every crystal pick one partner and
//! write the transfer into its own `tmp4`, then [`qrtz::qrtz_receive_kernel`] credits
//! the receiver. Both sides write only their own slot and read the other's
//! pre-dispatch value, so the transfer is exact and needs no cross-slot
//! read-modify-write; the two dispatches are strictly ordered inside
//! [`launch`].
//!
//! `MERC`'s four-trade volume equalisation (`MERC.cpp:120-145`) needs no
//! messages: [`force::merc_trade_snapshot_kernel`] freezes every mercury's
//! `tmp` into the `intent_target` scratch and [`force::merc_trade_kernel`]
//! replays, for each sender in a 5x5 window, the deterministic `between(-2, 2)`
//! draws that sender would make. Both sides therefore subtract and credit the
//! same amount, and a mercury writes only its own `tmp`.
//!
//! `PLNT`'s branching (`PLNT.cpp:120-165`) pushes one creation record per set
//! genome bit into the device queue. TPT's `create_part` calls see each other;
//! the queue is drained after the reaction step, so all branches of one plant
//! are emitted from the pre-pass pixel map. Two bits of one plant always name
//! different directions, so the multi-branch case is exact; when two different
//! plants (or a plant and another pass) target the same pixel, the drain keeps
//! the first queued record instead of TPT's slot order. The child inherits the
//! programme word of its phase (`tmp`..`tmp4`) and the parent's temperature.
//!
//! `SEED`'s breeding swap (`SEED.cpp:145-244`) runs in
//! [`transform::seed_breed_kernel`] after the transform pass. Each seed writes
//! only its own genome: the lowest-slot unclaimed seed whose chosen partner is
//! a pixel writes the child row 0 there, and the partner writes row 1 of its
//! own slot. Both sides derive the same meiosis draws from the pair seed, so
//! the swap needs no cross-slot read-modify-write and no `Atomic`.
//!
//! `BASE`'s concentration diffusion (`BASE.cpp:219-241`) uses
//! `Atomic<i32>::fetch_add`/`fetch_sub` on both halves, so concurrent trades
//! conserve the total concentration. `GLOW`'s `WATR -> DEUT` / `GEL -> RSST`
//! claim uses `Atomic<u32>::compare_exchange` on the neighbour's type, so a
//! water pixel touched by two glows is converted once and only the winning glow
//! dies. `MERC` spills one pixel per free neighbour (`MERC.cpp:100-118`), and
//! `VINE` creates its child at the sampled `(x+rx, y+ry)` with the parent's
//! temperature in the same pass (`VINE.cpp:66-72`).
//!
//! # TPT references
//!
//! * `GRAV.cpp:44-52` — velocity gauge sets a 48 frame `life`.
//! * `VIBR.cpp:43-84` — heat absorption, the pressure absorption (`pv > 2.5`
//!   charges 7 and spends one unit, the vacuum mirror drains 2 and returns
//!   one), the 1000 unit explosion arming and the `tmp` clamp;
//!   `VIBR.cpp:170-235` graphics.
//! * `VIBR.cpp:88-95`, `177-182` — `VIBR`/`BVBR` next to `ANAR` becomes `BVBR`;
//!   next to `EXOT` (1-in-25) becomes `EXOT`.
//! * `MERC.cpp:55-146` — temperature clamp, volume target, the condense /
//!   spill branches and the four-trade `tmp` equalisation.
//! * `TUNG.cpp:57-104` — the `O2` detonation, the melting threshold, the
//!   random `FIRE`/`LAVA` split and the `tmp3` pressure-brittle path.
//! * `TUNG.cpp:55` — the melting point is read from the element table, not
//!   hard-coded.
//! * `RFRG.cpp:22-36` — temperature scaling by the pressure change stored in
//!   the low `tmp` bits (`RFGL` shares the update).
//! * `BASE.cpp:56-215` — evaporation, freezing and the neutralisation table.
//! * `BREC.cpp:62-74` / `BRMT.cpp:51-78` — pressure heating and the `EXOT`
//!   conversion; hot `BREC` or hot `BRMT` (> 250 °C) react into `THRM`.
//! * `PLNT.cpp:61-262` / `SEED.cpp:106-281` / `VINE.cpp:55-77` / `SLTW.cpp:64-67` — tree growth,
//!   branching, stem conversion to `WOOD`/`GOO`, seed release, leaf coloring, spontaneous vine
//!   growth, and salt water plant death.
//! * `SEED.cpp:64-104` — high temperature/pressure wax conversion and growth.
//! * `SEED.cpp:106-281` — water intake and the `PLNT` hand-over.
//! * `GOLD.cpp:64-113` — rusted metal reversal and spark re-emission.
//! * `RSST.cpp:57-99` — gunpowder/carbon reactions and clone programming.
//! * `RSSS.cpp:27-60` — clone programming.
//! * `VINE.cpp:55-77` — spontaneous growth into `PLNT` and the `tmp2` heat
//!   memory the graphics pass reads.
//! * `SLCN.cpp:58-99` — the sparkle state machine and spark conduction.
//! * `QRTZ.cpp:57-155` — salt absorption, the `life` clamp, the `tmp3`
//!   pressure rupture, growth into a free pixel and the `tmp` diffusion
//!   (`PQRT` shares the update; `QRTZ` additionally does the rupture).
//! * `CRAY.cpp:62-79` — latching the element to emit from a neighbour.
//! * `POLO.cpp:55-108` — decay into `PLUT`, the slow self-heating, `PROT`
//!   absorption and the two neutron releases (see [`photon::special_photon_kernel`]).
//!
//! # What a pixel-parallel / per-slot order cannot express
//!
//! * **Force materials** (`FRAY.cpp:49-88`, `RPEL.cpp:41-73`,
//!   `ACEL.cpp:55-102`, `DCEL.cpp:55-99`) and **`DMG.cpp:44-116`** /
//!   **`HEAC.cpp:144-179`**: every one of them *adds* to the velocity (or the
//!   temperature) of an arbitrary neighbour, which this per-slot pass cannot
//!   do. They live in [`super::force`], which inverts the relation and has the
//!   target resolve the force elements around itself; nothing about them is
//!   implemented here.
//! * **`VIBR.cpp:96-126`** sparking and neighbour energy sharing: the sparks
//!   and the `tmp * 3` heat hand-off rewrite a *neighbour* slot, and the
//!   `tmp`/`tmp2` propagation between adjacent `VIBR` slots is a cross-slot
//!   read-modify-write. The detonation itself (`VIBR.cpp:118-142`, `EXOT` plus
//!   `ELEC`/`PHOT`/`BREC`) is ported through the creation queue in
//!   [`air::special_air_kernel`].
//! * **`BASE.cpp:230-260`**: the concentration diffusion writes a neighbour's
//!   `life`, which the pass performs with atomic `fetch_add`/`fetch_sub` on
//!   both halves of a trade, so two bases trading into the same pixel conserve
//!   the total concentration. The reaction *type* changes and deaths are ported
//!   the same way as the fluid pass (neighbouring `part_ty` writes). The
//!   `BMTL` rust seed in `tmp` and `GOLD`'s reset of it remain plain
//!   read-modify-writes, resolved in dispatch order rather than TPT's slot
//!   order.
//! * **`SAWD.cpp`**: has no `Update` (the combustion pass already treats a
//!   `SAWD` as flammable sawdust), so there is nothing to add here.
//! * `EQVE.cpp` (`E116`), `SPAWN.cpp`, `SPAWN2.cpp`, `LOLZ.cpp`, `PSTS.cpp`
//!   and `FILT.cpp` have no `Update`; their `Create`/`ChangeType` hooks are
//!   host-side placement rules, and the `FILT`/`PSTS`/`EQVE` visuals are
//!   handled by the phase-transition and render passes.
//! * `CRAY.cpp:59-79` (the `ctype` latch) samples `photons` *before* `pmap`
//!   and rejects an invalid id; the beam itself (`CRAY.cpp:84-140`, including
//!   the `FILT` colouring and the `PSCN`/`INST`/`INWR` spark modes) lives in
//!   [`cray`]. `ElementDef::enabled` is a host-only flag, so the device test is
//!   the id range, exactly as in `pipe::pump`.
//! * Element `Create` hooks (`QRTZ::create`, `SEED::create`, `TUNG::create`,
//!   `FILT::create`, ...) are host allocation defaults; this GPU pass cannot
//!   seed `tmp2`/`tmp3`/the seed genome at allocation time. `allocate_particle`
//!   zeroes `tmp3`/`tmp4`, so `TUNG`'s and `QRTZ`'s `tmp3 = int(pv*64)` seed
//!   simply happens on the first update instead (both kernels write it
//!   unconditionally), while `QRTZ`'s `tmp2 = between(0,10)` speckle allele
//!   stays 0 and `SEED`'s genome starts empty.

use core::mem::offset_of;

use cubecl::prelude::*;
use tpt_core::elements::GpuElementProps;
use tpt_core::particle::{GravityMode, SimulationConfig};

use super::PassDims;
use super::shared::{launch_grid, stasis_frozen};
use crate::buffers::SimBuffers;
use crate::reaction_kernel::shared::CREATE_QUEUE_WORDS;

pub(super) mod air;
pub(super) mod cray;
pub(super) mod force;
pub(super) mod photon;
pub(super) mod pressure;
pub(super) mod qrtz;
pub(super) mod transform;

use air::special_air_kernel;
use cray::cray_beam_kernel;
use force::{merc_trade_kernel, merc_trade_snapshot_kernel, special_force_kernel};
use photon::special_photon_kernel;
use pressure::special_pressure_kernel;
use qrtz::{qrtz_kernel, qrtz_receive_kernel};
use transform::{seed_breed_kernel, special_transform_kernel};

/// Word offset of `GpuElementProps::high_temp` (used by `TUNG`).
pub(super) const WORD_HIGH_TEMP: u32 = 24;

const _: () = assert!(offset_of!(GpuElementProps, high_temp) == WORD_HIGH_TEMP as usize * 4);

/// `PLNT.h`: bit positions inside a plant's packed `ctype`.
pub(super) const PLNT_PHASE: u32 = 1;
pub(super) const PLNT_DIR: u32 = 3;
pub(super) const PLNT_COLOUR: u32 = 6;
pub(super) const PLNT_LIFE: u32 = 12;
/// `PLNT.h:PLNT_TOTAL_TMP`: width of the `tmp`..`tmp4` genome words.
pub(super) const PLNT_TOTAL_TMP: u32 = 15;

/// `MERC.cpp`: maximum number of particles condensed into one mercury pixel.
pub(super) const MERC_ABSORB_SCALE: i32 = 10_000;

/// Type field of a packed `pmap` word (`PMAPMASK`).
pub(super) const PMAP_TYPE: u32 = 0x1FF;

/// `clamp(v, lo, hi)` for floats (CubeCL has no `clamp`).
#[cube]
pub(super) fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    let mut r = v;
    if r < lo {
        r = lo;
    }
    if r > hi {
        r = hi;
    }
    r
}

/// `clamp(v, lo, hi)` for integers.
#[cube]
pub(super) fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    let mut r = v;
    if r < lo {
        r = lo;
    }
    if r > hi {
        r = hi;
    }
    r
}

/// TPT `Simulation::rng.chance(1, denom)` from a deterministic hash.
#[cube]
pub(super) fn chance1(seed: u32, denom: u32) -> bool {
    seed.is_multiple_of(denom)
}

/// TPT `rng.chance(num, denom)` from a deterministic hash.
#[cube]
pub(super) fn chance(seed: u32, num: u32, denom: u32) -> bool {
    (seed % denom) < num
}

/// `dir3x3[d].X` (`PLNT.h`): counter-clockwise from the top.
#[cube]
pub(super) fn dir_x(d: u32) -> i32 {
    let mut v = 0i32;
    if d == 1u32 || d == 2u32 || d == 3u32 {
        v = -1i32;
    } else if d == 5u32 || d == 6u32 || d == 7u32 {
        v = 1i32;
    }
    v
}

/// `dir3x3[d].Y` (`PLNT.h`).
#[cube]
pub(super) fn dir_y(d: u32) -> i32 {
    let mut v = 0i32;
    if d == 0u32 || d == 1u32 || d == 7u32 {
        v = -1i32;
    } else if d == 3u32 || d == 4u32 || d == 5u32 {
        v = 1i32;
    }
    v
}

/// `Element_PLNT_detectDown` direction index for a normalised gravity vector.
///
/// Mirrors `PLNT.cpp:406-445`: the sign of each component picks a row/column of
/// `invDir`, and a vector that is not clearly axis-aligned yields 8 ("undecided").
fn gravity_dir_index(gx: f32, gy: f32) -> u32 {
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

/// Map the active gravity mode to a direction index for `SEED`'s down axis.
///
/// Radial gravity has no single direction, and custom gravity is a vector `SimBuffers`
/// cannot carry, so it reports "undecided" (8) and the seed falls back to down.
fn config_down_dir(config: &SimulationConfig) -> u32 {
    match config.gravity_mode {
        GravityMode::Vertical => gravity_dir_index(0.0f32, 1.0f32),
        GravityMode::Off => 8u32,
        GravityMode::Radial => 8u32,
        GravityMode::Custom => gravity_dir_index(config.custom_gravity_x, config.custom_gravity_y),
    }
}

/// Entry point called once per reaction step.
pub(super) fn launch(bufs: &SimBuffers, dims: &PassDims) {
    let (cubes, dim) = launch_grid(dims.used_slots);
    unsafe {
        special_force_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            dims.used_slots as u32,
            dims.xcells,
            dims.frame_seed,
        );
    }

    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        special_air_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    unsafe {
        special_pressure_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    // Quartz: salt absorption, growth and the diffusion send. The receive
    // pass reads the messages the send pass just wrote, so it must be a
    // separate dispatch.
    unsafe {
        qrtz_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }
    unsafe {
        qrtz_receive_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            dims.xres,
            dims.yres,
        );
    }

    let down_dir = config_down_dir(&bufs.config);
    unsafe {
        special_transform_kernel::launch_unchecked(
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
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.frame_seed,
            down_dir,
        );
    }

    // `SEED.cpp:145-244`: the breeding swap. Runs after the transform pass so
    // the water count it reads is the one the seed drank this frame. The
    // lowest-slot unclaimed initiator writes both child rows; its partner
    // writes nothing, so each slot has exactly one writer.
    unsafe {
        seed_breed_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }

    // `MERC.cpp:120-145`: the staged volume exchange. The snapshot pass freezes
    // every mercury's post-condense volume into the `intent_target` scratch and
    // the trade pass resolves all four trades per pixel from it, so neither
    // side of a trade ever writes the other's slot.
    let (slot_cubes, slot_dim) = launch_grid(dims.used_slots);
    unsafe {
        merc_trade_snapshot_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.intent_target.clone(), dims.capacity),
            dims.used_slots as u32,
        );
    }
    unsafe {
        merc_trade_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.intent_target.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }

    // `CRAY.cpp:84-140`: the beams. Their creations go through the device
    // queue, which the driver drains after this pass.
    unsafe {
        cray_beam_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
        );
    }

    // Energy-layer interactions: `GOLD` shielding and the `POLO` decay
    // chain. Slot parallel, but it needs the `photons` layer, so it cannot
    // live in the pixel-parallel transform pass above.
    let (cubes, dim) = launch_grid(dims.used_slots);
    unsafe {
        special_photon_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            dims.used_slots as u32,
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plant_down_direction_follows_gravity() {
        // `PLNT.cpp:406-445` + `invDir`: +y is dir 4, +x is dir 6.
        assert_eq!(gravity_dir_index(0.0f32, 1.0f32), 4);
        assert_eq!(gravity_dir_index(0.0f32, -1.0f32), 0);
        assert_eq!(gravity_dir_index(1.0f32, 0.0f32), 6);
        assert_eq!(gravity_dir_index(-1.0f32, 0.0f32), 2);
        assert_eq!(gravity_dir_index(1.0f32, 1.0f32), 5);
        assert_eq!(gravity_dir_index(-1.0f32, -1.0f32), 1);
        assert_eq!(gravity_dir_index(1.0f32, -1.0f32), 7);
        assert_eq!(gravity_dir_index(-1.0f32, 1.0f32), 3);
        // Only a zero gravity vector is undecided: a non-zero vector always
        // has a component above 0.5 after normalisation.
        assert_eq!(gravity_dir_index(0.0f32, 0.0f32), 8);
        assert_eq!(gravity_dir_index(1.0f32, 0.4f32), 6);
    }

    #[test]
    fn config_down_direction_maps_modes() {
        let mut config = SimulationConfig::new(64, 64);
        assert_eq!(config_down_dir(&config), 4);
        config.gravity_mode = GravityMode::Off;
        assert_eq!(config_down_dir(&config), 8);
        config.gravity_mode = GravityMode::Radial;
        assert_eq!(config_down_dir(&config), 8);
        config.gravity_mode = GravityMode::Custom;
        config.custom_gravity_x = 1.0f32;
        config.custom_gravity_y = 0.0f32;
        assert_eq!(config_down_dir(&config), 6);
    }

    #[test]
    fn plant_bit_layout_matches_plnt_header() {
        // `PLNT.h`: PLNT_PHASE=1, PLNT_DIR=3, PLNT_COLOUR=6, PLNT_LIFE=12.
        assert_eq!(PLNT_PHASE, 1);
        assert_eq!(PLNT_DIR, 3);
        assert_eq!(PLNT_LIFE, 12);
        // `SEED.cpp:create`: ctype = `0b111011'000'00'0`, i.e. a colour
        // allele mask with zero direction/phase and no stored water yet.
        let ctype: u32 = 0b1110_1100_0000;
        assert_eq!((ctype >> PLNT_LIFE) & 0xFF, 0);
        assert_eq!((ctype >> PLNT_DIR) & 7, 0);
        assert_eq!((ctype >> 6) & 0x3F, 0b111011);
        assert_eq!(ctype & 1, 0);
        assert_eq!((ctype >> PLNT_DIR) & 7, 0);
    }
}
