//! Device-side indirect dispatch table for conditional reaction and GoL passes.
//!
//! `thermal_kernel::refresh_pmap_types_kernel` reduces the active element types
//! across `0..used_slots` into the 256-bit `SimBuffers::gpu_elem_mask` (`8 x u32`).
//! [`prepare_sim_dispatch`] runs a single 64-thread workgroup that intersects
//! that bitmask with each subsystem's element-mask row in `gpu_sim_dispatch_lut`
//! and writes `[cubes, 1, 1, 0]` (or `[0, 1, 1, 0]` when absent) into
//! `SimBuffers::gpu_sim_dispatch`.
//!
//! Every gated pass then dispatches with [`sim_grid`], which binds its 12-byte
//! indirect-dispatch slice via `CubeCount::Dynamic`. Absent subsystems launch
//! zero workgroups on the device with zero host read-back.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::PassDims;
use super::shared::launch_grid;
use crate::buffers::SimBuffers;

/// Number of 16-byte (`4 x u32`) indirect-dispatch slots in `gpu_sim_dispatch`.
pub const SIM_DISPATCH_SLOTS: usize = 56;
/// Total `u32` words in `SimBuffers::gpu_sim_dispatch`.
pub const SIM_DISPATCH_WORDS: usize = SIM_DISPATCH_SLOTS * 4;
/// Words per slot in the constant LUT: `8` mask words + `1` grid-kind word.
pub const SIM_LUT_STRIDE: usize = 9;
/// Total `u32` words in `SimBuffers::gpu_sim_dispatch_lut`.
pub const SIM_LUT_WORDS: usize = SIM_DISPATCH_SLOTS * SIM_LUT_STRIDE;

const SIM_SLOTS_U32: u32 = 56;
const SIM_LUT_STRIDE_U32: u32 = 9;
const _: () = assert!(SIM_SLOTS_U32 as usize == SIM_DISPATCH_SLOTS);
const _: () = assert!(SIM_LUT_STRIDE_U32 as usize == SIM_LUT_STRIDE);

const GRID_PIXELS: u32 = 0;
const GRID_SLOTS: u32 = 1;
const GRID_LOVE: u32 = 2;
const GRID_SPAWN_WIN: u32 = 3;

pub(crate) const SIM_SLOT_LOVE: usize = 0;
pub(crate) const SIM_SLOT_CBNW: usize = 1;
pub(crate) const SIM_SLOT_GLAS: usize = 2;
pub(crate) const SIM_SLOT_O2: usize = 3;
pub(crate) const SIM_SLOT_FIRW: usize = 4;
pub(crate) const SIM_SLOT_REACTIVE: usize = 5;
pub(crate) const SIM_SLOT_SOAP_PIX: usize = 6;
pub(crate) const SIM_SLOT_EXOT_SOAP_SLOT: usize = 7;
pub(crate) const SIM_SLOT_GEL: usize = 8;
pub(crate) const SIM_SLOT_LIGH: usize = 9;
pub(crate) const SIM_SLOT_BANG: usize = 10;
pub(crate) const SIM_SLOT_FUSE_GROUP: usize = 11;
pub(crate) const SIM_SLOT_EMP: usize = 12;
pub(crate) const SIM_SLOT_GLOW: usize = 13;
pub(crate) const SIM_SLOT_CORROSION: usize = 14;
pub(crate) const SIM_SLOT_NEUT: usize = 15;
pub(crate) const SIM_SLOT_ELEC_SENSORS: usize = 16;
pub(crate) const SIM_SLOT_ELEC_LSNS: usize = 17;
pub(crate) const SIM_SLOT_ELEC_INST: usize = 18;
pub(crate) const SIM_SLOT_ELEC_ETRD: usize = 19;
pub(crate) const SIM_SLOT_NUC_PRESSURE: usize = 20;
pub(crate) const SIM_SLOT_NUC_NBLE_PIX: usize = 21;
pub(crate) const SIM_SLOT_NUC_NBLE_SLOT: usize = 22;
pub(crate) const SIM_SLOT_NUC_ISOTOPE: usize = 23;
pub(crate) const SIM_SLOT_NUC_SHIELD: usize = 24;
pub(crate) const SIM_SLOT_NUC_ANNIHILATE: usize = 25;
pub(crate) const SIM_SLOT_NUC_VIRUS: usize = 26;
pub(crate) const SIM_SLOT_NUC_CONV: usize = 27;
pub(crate) const SIM_SLOT_NUC_WARP: usize = 28;
pub(crate) const SIM_SLOT_NUC_DEUT: usize = 29;
pub(crate) const SIM_SLOT_NUC_PTNM: usize = 30;
pub(crate) const SIM_SLOT_NUC_PROT_PIX: usize = 31;
pub(crate) const SIM_SLOT_NUC_PROT_SLOT: usize = 32;
pub(crate) const SIM_SLOT_BISM: usize = 33;
pub(crate) const SIM_SLOT_MODEXT_PIX: usize = 34;
pub(crate) const SIM_SLOT_MODEXT_SLOT: usize = 35;
pub(crate) const SIM_SLOT_FORCE_PIX: usize = 36;
pub(crate) const SIM_SLOT_FORCE_SLOT: usize = 37;
pub(crate) const SIM_SLOT_PIPE_PIX: usize = 38;
pub(crate) const SIM_SLOT_PIPE_SLOT: usize = 39;
pub(crate) const SIM_SLOT_PORTALS_PIX: usize = 40;
pub(crate) const SIM_SLOT_RAYS_PIX: usize = 41;
pub(crate) const SIM_SLOT_STICKMAN_SLOT: usize = 42;
pub(crate) const SIM_SLOT_GOL_PIX: usize = 43;
pub(crate) const SIM_SLOT_SPEC_MERC_SLOT: usize = 44;
pub(crate) const SIM_SLOT_SPEC_QRTZ_PIX: usize = 45;
pub(crate) const SIM_SLOT_SPEC_SEED_PIX: usize = 46;
pub(crate) const SIM_SLOT_NUC_FREE_WIN: usize = 47;
pub(crate) const SIM_SLOT_SPEC_FORCE_SLOT: usize = 48;
pub(crate) const SIM_SLOT_SPEC_AIR_PIX: usize = 49;
pub(crate) const SIM_SLOT_SPEC_PRESSURE_PIX: usize = 50;
pub(crate) const SIM_SLOT_SPEC_TRANSFORM_PIX: usize = 51;
pub(crate) const SIM_SLOT_SPEC_MERC_PIX: usize = 52;
pub(crate) const SIM_SLOT_SPEC_PHOTON_SLOT: usize = 53;
pub(crate) const SIM_SLOT_ELEC_THDR_PIX: usize = 54;
pub(crate) const SIM_SLOT_COMB_EXPLOSIVE_PIX: usize = 55;

/// Builds a 256-bit element mask (`8 x u32`) from a slice of element IDs.
const fn mask_of(ids: &[u32]) -> [u32; 8] {
    let mut m = [0u32; 8];
    let mut i = 0;
    while i < ids.len() {
        let id = ids[i] as usize;
        if id < 256 {
            m[id >> 5] |= 1u32 << (id & 31);
        }
        i += 1;
    }
    m
}

/// Mask covering the 44 Cracker1000 modext elements (`195..=238`).
const fn modext_mask() -> [u32; 8] {
    let mut m = [0u32; 8];
    let mut id = 195usize;
    while id <= 238 {
        m[id >> 5] |= 1u32 << (id & 31);
        id += 1;
    }
    m
}

const fn write_row(lut: &mut [u32; SIM_LUT_WORDS], slot: usize, mask: [u32; 8], kind: u32) {
    let base = slot * SIM_LUT_STRIDE;
    let mut w = 0;
    while w < 8 {
        lut[base + w] = mask[w];
        w += 1;
    }
    lut[base + 8] = kind;
}

/// Builds the static host copy of `SimBuffers::gpu_sim_dispatch_lut`.
pub fn build_sim_dispatch_lut() -> [u32; SIM_LUT_WORDS] {
    let mut lut = [0u32; SIM_LUT_WORDS];
    write_row(&mut lut, SIM_SLOT_LOVE, mask_of(&[PT_LOVE, PT_LOLZ]), GRID_LOVE);
    write_row(&mut lut, SIM_SLOT_CBNW, mask_of(&[PT_CBNW]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_GLAS, mask_of(&[PT_GLAS]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_O2, mask_of(&[PT_O2]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_FIRW, mask_of(&[PT_FIRW, PT_FWRK]), GRID_PIXELS);
    write_row(
        &mut lut,
        SIM_SLOT_REACTIVE,
        mask_of(&[PT_LITH, PT_EXOT, PT_SING]),
        GRID_PIXELS,
    );
    write_row(&mut lut, SIM_SLOT_SOAP_PIX, mask_of(&[PT_SOAP]), GRID_PIXELS);
    write_row(
        &mut lut,
        SIM_SLOT_EXOT_SOAP_SLOT,
        mask_of(&[PT_EXOT, PT_SOAP]),
        GRID_SLOTS,
    );
    write_row(&mut lut, SIM_SLOT_GEL, mask_of(&[PT_GEL]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_LIGH, mask_of(&[PT_LIGH, PT_C5]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_BANG, mask_of(&[PT_BANG]), GRID_PIXELS);
    write_row(
        &mut lut,
        SIM_SLOT_FUSE_GROUP,
        mask_of(&[
            PT_FUSE, PT_FSEP, PT_IGNT, PT_ANAR, PT_BOYL, PT_C5, PT_THRM, PT_LAVA, PT_TRON,
        ]),
        GRID_PIXELS,
    );
    write_row(&mut lut, SIM_SLOT_EMP, mask_of(&[PT_EMP]), GRID_SLOTS);
    write_row(&mut lut, SIM_SLOT_GLOW, mask_of(&[PT_GLOW]), GRID_PIXELS);
    write_row(
        &mut lut,
        SIM_SLOT_CORROSION,
        mask_of(&[PT_ACID, PT_CAUS, PT_BMTL, PT_IRON, PT_METL]),
        GRID_PIXELS,
    );
    write_row(&mut lut, SIM_SLOT_NEUT, mask_of(&[PT_NEUT]), GRID_PIXELS);
    write_row(
        &mut lut,
        SIM_SLOT_ELEC_SENSORS,
        mask_of(&[PT_PSNS, PT_LSNS, PT_TSNS, PT_VSNS]),
        GRID_PIXELS,
    );
    write_row(&mut lut, SIM_SLOT_ELEC_LSNS, mask_of(&[PT_LSNS]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_ELEC_INST, mask_of(&[PT_INST]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_ELEC_ETRD, mask_of(&[PT_ETRD]), GRID_SLOTS);
    write_row(
        &mut lut,
        SIM_SLOT_NUC_PRESSURE,
        mask_of(&[PT_URAN, PT_PLUT, PT_WARP, PT_INVIS]),
        GRID_PIXELS,
    );
    write_row(&mut lut, SIM_SLOT_NUC_NBLE_PIX, mask_of(&[PT_NBLE]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_NUC_NBLE_SLOT, mask_of(&[PT_NBLE]), GRID_SLOTS);
    write_row(
        &mut lut,
        SIM_SLOT_NUC_ISOTOPE,
        mask_of(&[PT_ISOZ, PT_ISZS, PT_PRTI, PT_PRTO]),
        GRID_PIXELS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_NUC_SHIELD,
        mask_of(&[PT_SHLD1, PT_SHLD2, PT_SHLD3, PT_SHLD4]),
        GRID_PIXELS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_NUC_ANNIHILATE,
        mask_of(&[PT_AMTR, PT_EMBR, PT_BANG, PT_BOMB, PT_DEST]),
        GRID_PIXELS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_NUC_VIRUS,
        mask_of(&[PT_VIRS, PT_VRSS, PT_VRSG]),
        GRID_PIXELS,
    );
    write_row(&mut lut, SIM_SLOT_NUC_CONV, mask_of(&[PT_CONV]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_NUC_WARP, mask_of(&[PT_WARP]), GRID_PIXELS);
    write_row(
        &mut lut,
        SIM_SLOT_NUC_DEUT,
        mask_of(&[PT_DEUT, PT_GLOW]),
        GRID_PIXELS,
    );
    write_row(&mut lut, SIM_SLOT_NUC_PTNM, mask_of(&[PT_PTNM]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_NUC_PROT_PIX, mask_of(&[PT_PROT]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_NUC_PROT_SLOT, mask_of(&[PT_PROT]), GRID_SLOTS);
    write_row(&mut lut, SIM_SLOT_BISM, mask_of(&[PT_BISM, PT_LAVA]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_MODEXT_PIX, modext_mask(), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_MODEXT_SLOT, modext_mask(), GRID_SLOTS);
    write_row(
        &mut lut,
        SIM_SLOT_FORCE_PIX,
        mask_of(&[PT_ACEL, PT_DCEL, PT_RPEL, PT_FRAY, PT_HEAC, PT_DMG]),
        GRID_PIXELS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_FORCE_SLOT,
        mask_of(&[PT_ACEL, PT_DCEL, PT_RPEL, PT_FRAY, PT_HEAC, PT_DMG]),
        GRID_SLOTS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_PIPE_PIX,
        mask_of(&[PT_PIPE, PT_PPIP]),
        GRID_PIXELS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_PIPE_SLOT,
        mask_of(&[PT_PIPE, PT_PPIP]),
        GRID_SLOTS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_PORTALS_PIX,
        mask_of(&[PT_PRTI, PT_PRTO, PT_WIFI, PT_PPTI, PT_PPTO]),
        GRID_PIXELS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_RAYS_PIX,
        mask_of(&[PT_ARAY, PT_DRAY, PT_PSTN, PT_FRME]),
        GRID_PIXELS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_STICKMAN_SLOT,
        mask_of(&[PT_STKM, PT_STKM2, PT_FIGH, PT_SPAWN, PT_SPAWN2]),
        GRID_SLOTS,
    );
    write_row(&mut lut, SIM_SLOT_GOL_PIX, mask_of(&[PT_LIFE]), GRID_PIXELS);
    write_row(&mut lut, SIM_SLOT_SPEC_MERC_SLOT, mask_of(&[PT_MERC]), GRID_SLOTS);
    write_row(
        &mut lut,
        SIM_SLOT_SPEC_QRTZ_PIX,
        mask_of(&[PT_QRTZ, PT_PQRT]),
        GRID_PIXELS,
    );
    write_row(&mut lut, SIM_SLOT_SPEC_SEED_PIX, mask_of(&[PT_SEED]), GRID_PIXELS);
    write_row(
        &mut lut,
        SIM_SLOT_NUC_FREE_WIN,
        mask_of(&[PT_NBLE, PT_PROT]),
        GRID_SPAWN_WIN,
    );
    write_row(
        &mut lut,
        SIM_SLOT_SPEC_FORCE_SLOT,
        mask_of(&[PT_GRAV, PT_VIBR, PT_BVBR, PT_MERC]),
        GRID_SLOTS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_SPEC_AIR_PIX,
        mask_of(&[
            PT_RFRG, PT_RFGL, PT_BASE, PT_BREC, PT_SEED, PT_VIBR, PT_BVBR, PT_MORT,
        ]),
        GRID_PIXELS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_SPEC_PRESSURE_PIX,
        mask_of(&[PT_TUNG, PT_QRTZ]),
        GRID_PIXELS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_SPEC_TRANSFORM_PIX,
        mask_of(&[
            PT_GOLD, PT_RSST, PT_RSSS, PT_VINE, PT_SEED, PT_PLNT, PT_SLCN, PT_CRAY, PT_VIBR,
            PT_BVBR, PT_MERC, PT_BREC, PT_BRMT, PT_BASE, PT_GLOW,
        ]),
        GRID_PIXELS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_SPEC_MERC_PIX,
        mask_of(&[PT_MERC, PT_BASE]),
        GRID_PIXELS,
    );
    write_row(
        &mut lut,
        SIM_SLOT_SPEC_PHOTON_SLOT,
        mask_of(&[PT_GOLD, PT_POLO]),
        GRID_SLOTS,
    );
    write_row(&mut lut, SIM_SLOT_ELEC_THDR_PIX, mask_of(&[PT_THDR]), GRID_PIXELS);
    write_row(
        &mut lut,
        SIM_SLOT_COMB_EXPLOSIVE_PIX,
        mask_of(&[PT_BOMB, PT_DEST, PT_BANG, PT_EMBR]),
        GRID_PIXELS,
    );
    lut
}

#[cube]
#[allow(clippy::erasing_op)]
fn seed_u32(anchor: u32, value: u32) -> u32 {
    anchor * 0u32 + value
}

/// Populates all 56 indirect-dispatch slots in `sim_dispatch` from `elem_mask`
/// and the static `lut`. One 64-thread workgroup covers every slot.
#[cube(launch_unchecked)]
fn prepare_sim_dispatch_kernel(
    elem_mask: &[Atomic<u32>],
    lut: &[u32],
    sim_dispatch: &mut [u32],
    pixel_cubes: u32,
    slot_cubes: u32,
    love_cubes: u32,
    spawn_win_cubes: u32,
) {
    let tid = UNIT_POS_X;
    if tid >= SIM_SLOTS_U32 {
        terminate!();
    }
    let row = (tid * SIM_LUT_STRIDE_U32) as usize;
    let mut hit = seed_u32(tid, 0u32);
    for w in 0u32..8u32 {
        let mask_word = lut[row + w as usize];
        if mask_word != 0u32 {
            hit |= elem_mask[w as usize].fetch_max(0u32) & mask_word;
        }
    }
    let kind = lut[row + 8usize];
    let mut cubes = seed_u32(tid, 0u32);
    if hit != 0u32 {
        if kind == 0u32 {
            cubes = pixel_cubes;
        } else if kind == 1u32 {
            cubes = slot_cubes;
        } else if kind == 2u32 {
            cubes = love_cubes;
        } else {
            cubes = spawn_win_cubes;
        }
    }
    let out = (tid * 4u32) as usize;
    sim_dispatch[out] = cubes;
    sim_dispatch[out + 1usize] = 1u32;
    sim_dispatch[out + 2usize] = 1u32;
    sim_dispatch[out + 3usize] = 0u32;
}

/// Computes the number of 9x9 `LOVE`/`LOLZ` blocks for the given canvas.
/// Mirrors `love::launch`'s `(xres - 4).div_ceil(9) * (yres - 4).div_ceil(9)`.
fn love_block_count(xres: u32, yres: u32) -> usize {
    if xres < 4 || yres < 4 {
        return 0;
    }
    let bx_count = (xres - 4).div_ceil(9) as usize;
    let by_count = (yres - 4).div_ceil(9) as usize;
    bx_count.saturating_mul(by_count)
}

/// Refreshes `SimBuffers::gpu_sim_dispatch` from `SimBuffers::gpu_elem_mask`.
///
/// Called once at the start of `step_reactions_gpu`, and once again after
/// `drain_gpu_creates` so passes that run after the drain (`portals`, `rays`,
/// `stickman`, and `gol_kernel`) see newly created element types in the same
/// tick.
pub(super) fn prepare_sim_dispatch(bufs: &SimBuffers, dims: &PassDims) {
    let (pixel_cubes, _) = launch_grid(dims.pixels);
    let (slot_cubes, _) = launch_grid(dims.used_slots);
    let love_blocks = love_block_count(dims.xres, dims.yres);
    let love_cubes = if love_blocks == 0 {
        0u32
    } else {
        launch_grid(love_blocks).0
    };
    let (spawn_win_cubes, _) = launch_grid(SimBuffers::GPU_SPAWN_COUNT);
    unsafe {
        prepare_sim_dispatch_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            BufferArg::from_raw_parts(bufs.gpu_elem_mask.clone(), 8),
            BufferArg::from_raw_parts(bufs.gpu_sim_dispatch_lut.clone(), SIM_LUT_WORDS),
            BufferArg::from_raw_parts(bufs.gpu_sim_dispatch.clone(), SIM_DISPATCH_WORDS),
            pixel_cubes,
            slot_cubes,
            love_cubes,
            spawn_win_cubes,
        );
    }
}

/// Builds a `CubeCount::Dynamic` pointing at one 12-byte indirect dispatch slot
/// inside `SimBuffers::gpu_sim_dispatch`.
pub(crate) fn sim_grid(bufs: &SimBuffers, slot: usize) -> CubeCount {
    debug_assert!(slot < SIM_DISPATCH_SLOTS);
    let start = (slot * 16) as u64;
    let total = (SIM_DISPATCH_WORDS * 4) as u64;
    let end = total - (start + 12);
    CubeCount::Dynamic(
        bufs.gpu_sim_dispatch
            .clone()
            .offset_start(start)
            .offset_end(end)
            .binding(),
    )
}
