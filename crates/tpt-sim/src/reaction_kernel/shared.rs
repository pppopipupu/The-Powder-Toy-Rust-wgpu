//! Shared constants and helpers of the reaction pass.
//!
//! Everything here is used by at least two of the per-pass modules
//! ([`crate::reaction_kernel::life`], `combustion`, `fluid`, `corrosion`,
//! `energy`, `electronics`, `wall`): the packed `GpuElementProps` word offsets,
//! the wall-id mirrors with their compile-time assertions, the device-side
//! hash/PRNG helpers, the insulation test, the spawn-request bits, the
//! creation-queue records ([`request_create`] for a fresh slot,
//! [`request_replace`] for re-creating an existing one) and the host-side spawn
//! materialisation.
//!
//! The `#[cube]` helpers are re-exported from the parent module, so
//! `reaction_kernel::elem_word` & friends keep their original paths.

use cubecl::prelude::*;
use tpt_core::constants::*;

use crate::buffers::SimBuffers;

use super::PassDims;

/// `u32` words per [`tpt_core::elements::GpuElementProps`] entry (128 bytes).
pub(super) const ELEM_PROP_WORDS: usize = 32;
/// Word offset of `GpuElementProps::properties`.
pub(super) const WORD_PROPERTIES: u32 = 1;
/// Word offset of `GpuElementProps::flammable`.
pub(super) const WORD_FLAMMABLE: u32 = 12;
/// Word offset of `GpuElementProps::explosive`.
pub(super) const WORD_EXPLOSIVE: u32 = 13;
/// Word offset of `GpuElementProps::hardness`.
pub(super) const WORD_HARDNESS: u32 = 14;
/// Word offset of `GpuElementProps::default_temp`.
pub(super) const WORD_DEFAULT_TEMP: u32 = 16;
/// Word offset of `GpuElementProps::high_temp_transition`
/// (`CLNE.cpp:86-87` tests it before cloning `LAVA`).
pub(crate) const WORD_HIGH_TEMP_TRANSITION: u32 = 25;
/// Word offset of `GpuElementProps::default_life`.
pub(super) const WORD_DEFAULT_LIFE: u32 = 17;
/// Word offset of `GpuElementProps::carries_ctype` (`1` when a created element
/// restores a stored `ctype`, TPT's `CarriesTypeIn & (1 << FIELD_CTYPE)`).
pub(super) const WORD_CTYPE_IN: u32 = 31;

/// Number of element slots addressable through `gpu_elem_props`.
pub(super) const ELEM_SLOTS: u32 = PT_NUM as u32;

// The kernel indexes the property table by element id; keep the mirror honest.
const _: () = assert!(ELEM_SLOTS as usize == PT_NUM);
// `elem_word` strides by the literal `32` (words per `GpuElementProps` row).
const _: () = assert!(ELEM_PROP_WORDS == 32);

/// Work units per cube. 256 is the portable WebGPU minimum for a 1-D cube.
pub(super) const CUBE_DIM: u32 = 256;

/// A clone asked for one copy of `part_ctype` in an adjacent free pixel.
pub(super) const FLAG_SPAWN_CLONE: u32 = 0x0000_0100;
/// A particle asked for one copy of its **own** element in an adjacent free
/// pixel (`YEST` growth). Unlike [`FLAG_SPAWN_CLONE`] this needs no `ctype`,
/// which the growth pass does not have bound.
pub(super) const FLAG_SPAWN_SELF: u32 = 0x0000_0400;
/// A particle asked for one pixel of the element stored in its `tmp`
/// (`SPNG` releasing water/steam, `PLNT` puffing oxygen). Used where `ctype`
/// is either not bound or already carries the element's own state.
pub(super) const FLAG_SPAWN_TMP: u32 = 0x0000_0800;
/// A sparked `TESC` asked for one `LIGH` bolt; the bolt's power comes from the
/// coil's `tmp` and the requested direction from its `tmp2`.
pub(super) const FLAG_SPAWN_LIGHT: u32 = 0x0000_2000;
/// A `POLO` asked for one `NEUT` on its **own** pixel (`POLO.cpp:62-92`,
/// `create_part(-3, x, y, PT_NEUT)`). Energy particles live in the `photons`
/// layer, so the requester's pixel stays occupied in `pmap`; the spawned
/// neutron inherits the requester's temperature and, when the release consumed
/// an existing photon, that photon's velocity (its slot + 1 is parked in the
/// requester's `tmp3`).
pub(super) const FLAG_SPAWN_NEUT: u32 = 0x0000_4000;
/// A `STOR` released its contents. `storage_kernel` stores the captured
/// particle exactly as `STOR.cpp:77-81` does — `tmp` = type, `temp` = temperature,
/// `tmp2` = life, `tmp3` = `tmp`, `tmp4` = `ctype` — and the host copies those
/// five words into the released pixel.
pub(super) const FLAG_SPAWN_STOR: u32 = 0x0000_1000;
//
// TPT creates particles in place: `create_part(-1, x, y, t)` writes a full
// `Particle` and every field the caller then overrides (`life`, `temp`, `ctype`,
// `tmp`, `dcolour`) lands before the next particle is updated. A GPU pass cannot
// do that — a thread cannot reserve a pool slot without an allocator, and the
// host `FLAG_SPAWN_*` bits can only express a handful of shapes (one clone,
// one `ctype`, one packed `type|life`), always at the requester's own or a
// neighbouring pixel, with the registry defaults for everything else.
//
// The queue closes that gap. A kernel pushes a record per creation through
// [`request_create`]; `step_reactions_gpu` drains it right after
// `refresh_gpu_free_slots`, so the slots come from the device allocation window
// and the particles are live for the movement and render passes of the same
// frame, exactly as TPT's would be.
//
// The queue carries two creation shapes. A [`request_create`] record asks the
// drain for a *fresh* slot out of that window. A [`request_replace`] record
// re-creates the particle that already owns a slot and mirrors
// `Simulation::create_part(p, x, y, t)` with `p >= 0`
// (`Simulation.cpp:1918-1945`): the record arrives with [`CR_DST`] already set
// to `slot + 1`, the drain re-initialises that slot from the target element's
// `DefaultProperties`, runs the target's `Create` hook, and rewrites the map
// word at `(x, y)` in the layer of the new element. The old element's
// `ChangeType` hook is NOT run: the mod elements that replace themselves this
// way have no `ChangeType` on their old types, so TPT's notification would be a
// no-op. The drain applies no occupancy test to the requested pixel: the caller
// owns the slot and names the pixel it is taking over.
//
/// Word index of a record's slot: `0` = still pending, `0xFFFF_FFFF` = dropped
/// (no slot, pixel taken or out of bounds), otherwise `slot + 1`. A
/// [`request_replace`] record arrives with this word already filled in.
pub(crate) const CR_DST: u32 = 0;
/// Word index of the element id.
pub(crate) const CR_TY: u32 = 1;
/// Word indices of the requested pixel.
pub(crate) const CR_X: u32 = 2;
/// See [`CR_X`].
pub(crate) const CR_Y: u32 = 3;
/// Word index of the record's mask: the `CR_SET_*` field overrides plus
/// [`CR_NO_OCCUPANCY`] and [`CR_REPLACE`].
pub(crate) const CR_MASK: u32 = 4;
/// Word indices of the optional field overrides.
pub(crate) const CR_LIFE: u32 = 5;
/// See [`CR_LIFE`].
pub(crate) const CR_CTYPE: u32 = 6;
/// See [`CR_LIFE`] (`f32` bits).
pub(crate) const CR_TEMP: u32 = 7;
/// See [`CR_LIFE`].
pub(crate) const CR_TMP: u32 = 8;
/// See [`CR_LIFE`].
pub(crate) const CR_TMP2: u32 = 9;
/// See [`CR_LIFE`].
pub(crate) const CR_TMP3: u32 = 10;
/// See [`CR_LIFE`].
pub(crate) const CR_TMP4: u32 = 11;
/// See [`CR_LIFE`].
pub(crate) const CR_DCOLOUR: u32 = 12;
/// Word index of the `part_flags` word the new particle gets. The drain stores
/// it, so a replace record drops the old slot's flags instead of OR-ing them.
pub(crate) const CR_FLAGS: u32 = 13;
/// Word indices of the optional velocity (`f32` bits).
pub(crate) const CR_VX: u32 = 14;
/// See [`CR_VX`].
pub(crate) const CR_VY: u32 = 15;

/// [`CR_MASK`] bits: the record overrides the element's `DefaultProperties`.
pub(crate) const CR_SET_LIFE: u32 = 0x01;
/// See [`CR_SET_LIFE`].
pub(crate) const CR_SET_CTYPE: u32 = 0x02;
/// See [`CR_SET_LIFE`].
pub(crate) const CR_SET_TEMP: u32 = 0x04;
/// See [`CR_SET_LIFE`].
pub(crate) const CR_SET_TMP: u32 = 0x08;
/// See [`CR_SET_LIFE`].
pub(crate) const CR_SET_TMP2: u32 = 0x10;
/// See [`CR_SET_LIFE`].
pub(crate) const CR_SET_TMP3: u32 = 0x20;
/// See [`CR_SET_LIFE`].
pub(crate) const CR_SET_TMP4: u32 = 0x40;
/// See [`CR_SET_LIFE`].
pub(crate) const CR_SET_DCOLOUR: u32 = 0x80;
/// See [`CR_SET_LIFE`].
pub(crate) const CR_SET_VX: u32 = 0x100;
/// See [`CR_SET_LIFE`].
pub(crate) const CR_SET_VY: u32 = 0x200;
/// Skip the "pixel already occupied" test.
///
/// TPT's `create_part(-3, ...)` skips the map checks outright and overwrites
/// the pixel's map word, which is how `DeutImplosion` (`PROT.cpp:211-227`)
/// creates a whole burst of protons in one pixel. The default drain refuses a
/// taken pixel, so a burst has to opt out explicitly.
pub(super) const CR_NO_OCCUPANCY: u32 = 0x400;
/// Reuse the slot named by [`CR_DST`] instead of allocating one.
///
/// Mirrors `Simulation::create_part(p, x, y, t)` with `p >= 0`
/// (`Simulation.cpp:1918-1945`): the record arrives with `CR_DST = slot + 1`
/// and the drain skips both the free-slot allocation and the occupancy test,
/// handing the slot to the target element's `DefaultProperties` row and
/// `Create` hook. The old element's `ChangeType` hook is not run (see the
/// module comment).
pub(crate) const CR_REPLACE: u32 = 0x800;
// The replace bit shares the mask word with the field overrides; it must not
// alias one, or the drain would take an override for a slot reuse.
const _: () = assert!(
    (CR_REPLACE
        & (CR_SET_LIFE
            | CR_SET_CTYPE
            | CR_SET_TEMP
            | CR_SET_TMP
            | CR_SET_TMP2
            | CR_SET_TMP3
            | CR_SET_TMP4
            | CR_SET_DCOLOUR
            | CR_SET_VX
            | CR_SET_VY
            | CR_NO_OCCUPANCY))
        == 0
);

/// `SimBuffers::GPU_CREATE_COUNT` mirrored for kernel bodies (rule 1).
const _: () = assert!(SimBuffers::GPU_CREATE_COUNT == 16384);
const GPU_CREATE_COUNT_U32: u32 = 16384;
/// `SimBuffers::CREATE_WORDS` mirrored for kernel bodies.
const _: () = assert!(SimBuffers::CREATE_WORDS == 16);
const CREATE_WORDS_U32: u32 = 16;
/// Word count of the whole queue: `[count, record * GPU_CREATE_COUNT]`.
const _: () = assert!(SimBuffers::GPU_CREATE_COUNT * SimBuffers::CREATE_WORDS + 1 == 262_145);
pub(super) const CREATE_QUEUE_WORDS: usize =
    SimBuffers::GPU_CREATE_COUNT * SimBuffers::CREATE_WORDS + 1;
/// `tpt_core::elements::SPAWN_DEFAULT_WORDS` mirrored for kernel bodies.
const _: () = assert!(tpt_core::elements::SPAWN_DEFAULT_WORDS == 8);
const SPAWN_DEFAULT_WORDS_U32: u32 = 8;
/// Sentinels stored in [`CR_DST`].
const CREATE_PENDING_U32: u32 = 0;
const CREATE_FAILED_U32: u32 = 0xFFFF_FFFF;

/// Write one record into queue slot `rec`, when the queue has room for it.
///
/// `dst` is [`CREATE_PENDING_U32`] for a fresh-slot [`request_create`] and
/// `slot + 1` for a [`request_replace`], whose destination is allocated before
/// the drain runs. Both shapes share the [`CR_DST`]..[`CR_VY`] layout, so the
/// writer is shared too and the two cannot drift apart.
#[cube]
#[allow(clippy::too_many_arguments)]
fn write_create_record(
    create_queue: &mut [Atomic<u32>],
    rec: u32,
    dst: u32,
    ty: u32,
    x: i32,
    y: i32,
    mask: u32,
    life: i32,
    ctype: u32,
    temp: f32,
    tmp: i32,
    tmp2: i32,
    tmp3: i32,
    tmp4: i32,
    dcolour: u32,
    flags: u32,
    vx: f32,
    vy: f32,
) {
    if rec < GPU_CREATE_COUNT_U32 {
        let base = (1u32 + rec * CREATE_WORDS_U32) as usize;
        create_queue[base + CR_DST as usize].store(dst);
        create_queue[base + CR_TY as usize].store(ty);
        create_queue[base + CR_X as usize].store(x as u32);
        create_queue[base + CR_Y as usize].store(y as u32);
        create_queue[base + CR_MASK as usize].store(mask);
        create_queue[base + CR_LIFE as usize].store(life as u32);
        create_queue[base + CR_CTYPE as usize].store(ctype);
        create_queue[base + CR_TEMP as usize].store(u32::reinterpret(temp));
        create_queue[base + CR_TMP as usize].store(tmp as u32);
        create_queue[base + CR_TMP2 as usize].store(tmp2 as u32);
        create_queue[base + CR_TMP3 as usize].store(tmp3 as u32);
        create_queue[base + CR_TMP4 as usize].store(tmp4 as u32);
        create_queue[base + CR_DCOLOUR as usize].store(dcolour);
        create_queue[base + CR_FLAGS as usize].store(flags);
        create_queue[base + CR_VX as usize].store(u32::reinterpret(vx));
        create_queue[base + CR_VY as usize].store(u32::reinterpret(vy));
    }
}

/// Ask the drain for one particle with TPT's `create_part` semantics.
///
/// Every field whose [`CR_MASK`] bit is set is taken from the record; the rest
/// come from the element's `DefaultProperties`, so a caller that only needs an
/// element id and a pixel passes zeros for the rest. The call is fire-and-forget:
/// when the queue is full, the pixel is occupied, or the pool has no free slot
/// left in the device window, the creation is dropped — the same outcome TPT
/// reports as `create_part` returning `-1`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn request_create(
    create_queue: &mut [Atomic<u32>],
    ty: u32,
    x: i32,
    y: i32,
    mask: u32,
    life: i32,
    ctype: u32,
    temp: f32,
    tmp: i32,
    tmp2: i32,
    tmp3: i32,
    tmp4: i32,
    dcolour: u32,
    flags: u32,
    vx: f32,
    vy: f32,
) {
    let rec = create_queue[0].fetch_add(1u32);
    write_create_record(
        create_queue,
        rec,
        CREATE_PENDING_U32,
        ty,
        x,
        y,
        mask,
        life,
        ctype,
        temp,
        tmp,
        tmp2,
        tmp3,
        tmp4,
        dcolour,
        flags,
        vx,
        vy,
    );
}

/// Ask the drain to re-create the particle already living in `slot`.
///
/// This is TPT's `Simulation::create_part(p, x, y, t)` with `p >= 0`
/// (`Simulation.cpp:1918-1945`): the slot is not reallocated, it starts over
/// from the target element's `DefaultProperties` plus the [`CR_MASK`]
/// overrides, and the target's `Create` hook runs — exactly like
/// [`request_create`]. The drain rewrites the map word at `(x, y)` in the
/// layer of the new element without testing occupancy, so the caller is
/// responsible for naming a pixel the old particle frees or takes over.
///
/// The old element's `ChangeType` hook is not run (see the module comment).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn request_replace(
    create_queue: &mut [Atomic<u32>],
    slot: u32,
    ty: u32,
    x: i32,
    y: i32,
    mask: u32,
    life: i32,
    ctype: u32,
    temp: f32,
    tmp: i32,
    tmp2: i32,
    tmp3: i32,
    tmp4: i32,
    dcolour: u32,
    flags: u32,
    vx: f32,
    vy: f32,
) {
    let rec = create_queue[0].fetch_add(1u32);
    write_create_record(
        create_queue,
        rec,
        slot + 1u32,
        ty,
        x,
        y,
        mask | CR_REPLACE,
        life,
        ctype,
        temp,
        tmp,
        tmp2,
        tmp3,
        tmp4,
        dcolour,
        flags,
        vx,
        vy,
    );
}

/// [`request_create`] for the common case: an element at a pixel, no overrides.
#[cube]
pub(crate) fn request_create_plain(create_queue: &mut [Atomic<u32>], ty: u32, x: i32, y: i32) {
    request_create(
        create_queue,
        ty,
        x,
        y,
        0u32,
        0i32,
        0u32,
        0.0f32,
        0i32,
        0i32,
        0i32,
        0i32,
        0u32,
        0u32,
        0.0f32,
        0.0f32,
    );
}

/// `Simulation::create_part` bounds/type guards mirrored for kernel bodies.
const PT_NONE_U32: u32 = 0;
const PT_NUM_U32: u32 = PT_NUM as u32;
const PT_PLSM_U32: u32 = 28;
const PT_TRON_U32: u32 = 151;
const PT_HFLM_U32: u32 = 100;
const PT_PHOT_U32: u32 = 29;
const PT_NEUT_U32: u32 = 30;
const PT_ELEC_U32: u32 = 31;
const PT_LAVA_U32: u32 = 6;
const PT_FIRE_U32: u32 = 4;
const PT_ACID_U32: u32 = 20;
const PT_PROT_U32: u32 = 173;
const PT_GRVT_U32: u32 = 39;
const PT_WARP_U32: u32 = 112;
const PT_SING_U32: u32 = 140;
const PT_CLST_U32: u32 = 157;
const PT_PTNM_U32: u32 = 187;
/// Second-set element ids the `Create` hooks below test (`PT_FNTC`..`PT_MGNT`).
const PT_CLNT_U32: u32 = 196;
const PT_LED_U32: u32 = 197;
const PT_PRMT_U32: u32 = 212;
const PT_CWIR_U32: u32 = 214;
const PT_GRPH_U32: u32 = 218;
const PT_DFOM_U32: u32 = 220;
const PT_CEXP_U32: u32 = 222;
const PT_UVRD_U32: u32 = 225;
const PT_BFLM_U32: u32 = 229;
const PT_ALMP_U32: u32 = 233;
const PT_MISL_U32: u32 = 237;
const _: () = assert!(
    PT_CLNT == 196
        && PT_LED == 197
        && PT_PRMT == 212
        && PT_CWIR == 214
        && PT_GRPH == 218
        && PT_DFOM == 220
        && PT_CEXP == 222
        && PT_UVRD == 225
        && PT_BFLM == 229
        && PT_ALMP == 233
        && PT_MISL == 237
);
/// `PT_SEED`; the create hook seeds its genome (`SEED.cpp:254-275`).
const PT_SEED_U32: u32 = 194;
/// `PLNT_DIR` (`PLNT.h`): bit offset of the growth direction inside `ctype`.
const PLNT_DIR_U32: u32 = 3;
const _: () = assert!(PT_SEED == PT_SEED_U32 && super::special::PLNT_DIR == PLNT_DIR_U32);
const PT_SLCN_U32: u32 = 186;
const PT_CRMC_U32: u32 = 178;
const PT_PQRT_U32: u32 = 142;
const PT_ROCK_U32: u32 = 189;
const PT_QRTZ_U32: u32 = 141;
const PT_GLAS_U32: u32 = 24;
const PT_TUNG_U32: u32 = 171;
/// `CELL`: the air grid is four pixels per cell.
const CELL_U32: u32 = 4;
const TYPE_ENERGY_U32: u32 = 0x0000_0010;
const _: () = assert!(PT_NONE == 0);
const _: () = assert!(PT_NUM_U32 as usize == PT_NUM);
const _: () = assert!(PT_PLSM == 28);
const _: () = assert!(PT_TRON == 151);
const _: () = assert!(PT_HFLM == 100);
const _: () = assert!(PT_PHOT == 29);
const _: () = assert!(PT_NEUT == 30);
const _: () = assert!(PT_ELEC == 31);
const _: () = assert!(PT_LAVA == 6);
const _: () = assert!(PT_FIRE == 4);
const _: () = assert!(PT_ACID == 20);
const _: () = assert!(PT_PROT == 173);
const _: () = assert!(PT_GRVT == 39);
const _: () = assert!(PT_WARP == 112);
const _: () = assert!(PT_SING == 140);
const _: () = assert!(PT_CLST == 157);
const _: () = assert!(PT_PTNM == 187);
const _: () = assert!(PT_SLCN == 186);
const _: () = assert!(PT_CRMC == 178);
const _: () = assert!(PT_PQRT == 142);
const _: () = assert!(PT_ROCK == 189);
const _: () = assert!(PT_QRTZ == 141);
const _: () = assert!(PT_GLAS == 24);
const _: () = assert!(PT_TUNG == 171);
const _: () = assert!(CELL == 4);
const _: () = assert!(TYPE_ENERGY == 0x0000_0010);
/// `pmap_pack` stride: the id field is 23 bits, the type field 9.
const _: () = assert!(pmap_pack(0, 0) == 1u32 << 9);
/// The alloc kernel rebuilds a replace record's map word as
/// `(slot + 1) << 9 | ty`; keep that shift in step with `pmap_pack`.
const _: () = assert!(pmap_pack(41, 0) == 42u32 << 9);

/// Seed a `u32` device local from a runtime anchor.
///
/// CubeCL expands `let mut v = <host literal>;` to a plain integer, so the first
/// runtime assignment to `v` fails to compile with `expected u32, found
/// NativeExpand<u32>` (AGENTS rule 4). `anchor * 0 + value` is the value itself
/// on the device and keeps the local a device variable.
#[cube]
#[allow(clippy::erasing_op)]
pub(crate) fn seed_u32(anchor: u32, value: u32) -> u32 {
    anchor * 0u32 + value
}

/// `f32` variant of [`seed_u32`].
#[cube]
#[allow(clippy::erasing_op)]
pub(super) fn seed_f32(anchor: f32, value: f32) -> f32 {
    anchor * 0.0f32 + value
}

/// Drain step 1: give every new record of this frame its slot and its
/// spatial-map entry.
///
/// A [`request_create`] record takes a slot from the device window and is
/// scattered into the layer of its element. A [`request_replace`] record
/// arrives with its slot already set and re-scatters that same slot, which is
/// `create_part(p >= 0, ...)` (`Simulation.cpp:1918-1945`).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn create_alloc_kernel(
    create_queue: &mut [u32],
    free_slots: &mut [Atomic<u32>],
    elem_props: &[u32],
    pmap: &mut [u32],
    photons: &mut [u32],
    part_ty: &mut [u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_flags: &mut [u32],
    xres: u32,
    yres: u32,
    records: u32,
) {
    let rec = ABSOLUTE_POS as u32;
    // `reset_gpu_create_queue` clears the count word but not the records, so a
    // record belongs to this frame iff its index is below the count. Testing
    // "the destination is still pending" would also skip every replace record,
    // which arrives already allocated.
    if rec < records && rec < create_queue[0] {
        let base = (1u32 + rec * CREATE_WORDS_U32) as usize;
        let ty = create_queue[base + CR_TY as usize];
        let x = create_queue[base + CR_X as usize] as i32;
        let y = create_queue[base + CR_Y as usize] as i32;
        let mask = create_queue[base + CR_MASK as usize];
        let mut dst = seed_u32(ty, CREATE_FAILED_U32);
        let in_bounds = x >= 0i32 && y >= 0i32 && (x as u32) < xres && (y as u32) < yres;
        if in_bounds && ty != PT_NONE_U32 && ty < PT_NUM_U32 {
            let px = (y as u32 * xres + x as u32) as usize;
            let energy = (elem_word(elem_props, ty, WORD_PROPERTIES) & TYPE_ENERGY_U32) != 0u32;
            if (mask & CR_REPLACE) != 0u32 {
                // Slot reuse. `request_replace` already stored `slot + 1`, so
                // no allocation and no occupancy test: the caller owns the
                // slot. Drop the pixel's entry in either layer while it still
                // names this slot, then write the new element's word into the
                // layer its `TYPE_ENERGY` bit selects.
                let replace_dst = create_queue[base + CR_DST as usize];
                if replace_dst != CREATE_PENDING_U32 && replace_dst != CREATE_FAILED_U32 {
                    dst = replace_dst;
                    let s = (replace_dst - 1u32) as usize;
                    part_ty[s] = ty;
                    part_x[s] = x as f32;
                    part_y[s] = y as f32;
                    part_flags[s] = create_queue[base + CR_FLAGS as usize];
                    // The id field of a packed word is `slot + 1`.
                    if (pmap[px] >> 9u32) == replace_dst {
                        pmap[px] = 0u32;
                    }
                    if (photons[px] >> 9u32) == replace_dst {
                        photons[px] = 0u32;
                    }
                    let packed = (replace_dst << 9u32) | ty;
                    if energy {
                        photons[px] = packed;
                    } else {
                        pmap[px] = packed;
                    }
                }
            } else {
                let mut taken = if energy {
                    photons[px] != 0u32
                } else {
                    pmap[px] != 0u32
                };
                if (mask & CR_NO_OCCUPANCY) != 0u32 {
                    taken = false;
                }
                if !taken {
                    let slot = gpu_alloc_slot(free_slots);
                    if slot != CREATE_FAILED_U32 {
                        dst = slot + 1u32;
                        let s = slot as usize;
                        part_ty[s] = ty;
                        part_x[s] = x as f32;
                        part_y[s] = y as f32;
                        part_flags[s] = create_queue[base + CR_FLAGS as usize];
                        let packed = ((slot + 1u32) << 9u32) | ty;
                        if energy {
                            photons[px] = packed;
                        } else {
                            pmap[px] = packed;
                        }
                    }
                }
            }
        }
        create_queue[base] = dst;
    }
}

/// Drain step 2: fill the fields of the records this frame created or
/// re-created, taking the ones the record overrides and the element's
/// `DefaultProperties` for the rest. Stale records left in the queue by earlier
/// frames are skipped: only the first `create_queue[0]` records belong to this
/// frame.
///
/// Both creation shapes land here — a fresh allocation and a [`request_replace`]
/// record, whose destination word already names its slot — and the reset is the
/// same for both. This is what clears `tmp3`, `tmp4` and `dcolour` on a *reused*
/// slot: it is recycled from whatever the old element left there, and TPT's
/// `create_part` starts from a clean default row.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn create_fields_kernel(
    create_queue: &mut [u32],
    spawn_defaults: &[u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_dcolour: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    air_pv: &[f32],
    frame_seed: u32,
    xcells: u32,
) {
    let rec = ABSOLUTE_POS as u32;
    // `reset_gpu_create_queue` clears the counter but not the per-record
    // destination words. A frame with fewer requests than the last would
    // otherwise replay stale records and overwrite fields a later pass
    // changed -- the LIGH segment countdown in `part_tmp2`, for one -- so
    // only this frame's requests may be filled.
    let records = create_queue[0];
    if rec < records && rec < GPU_CREATE_COUNT_U32 {
        let base = (1u32 + rec * CREATE_WORDS_U32) as usize;
        let dst = create_queue[base];
        // A replace record arrives with its destination already set, so it
        // passes this test and re-initialises the slot it names.
        if dst != CREATE_PENDING_U32 && dst != CREATE_FAILED_U32 {
            let s = (dst - 1u32) as usize;
            let ty = create_queue[base + CR_TY as usize];
            let mask = create_queue[base + CR_MASK as usize];
            let db = (ty * SPAWN_DEFAULT_WORDS_U32) as usize;
            part_life[s] = if (mask & CR_SET_LIFE) != 0u32 {
                create_queue[base + CR_LIFE as usize] as i32
            } else {
                spawn_defaults[db + 1usize] as i32
            };
            part_ctype[s] = if (mask & CR_SET_CTYPE) != 0u32 {
                create_queue[base + CR_CTYPE as usize]
            } else {
                spawn_defaults[db + 2usize]
            };
            if (mask & CR_SET_CTYPE) == 0u32 && ty == PT_SEED_U32 {
                // `SEED.cpp:254-266`: the genome header plus the growth
                // direction, which points against gravity. The device has no
                // gravity field at creation time, so the direction is the random
                // fallback TPT itself uses when `detectDown` cannot decide.
                let h = hash_u32(frame_seed ^ (dst << 17u32) ^ 0x5EED_0001u32);
                let down = h % 8u32;
                let up = (down + 4u32) % 8u32;
                part_ctype[s] = 0b1110_1100_0000u32 | ((up & 7u32) << PLNT_DIR_U32);
            }
            if (mask & CR_SET_CTYPE) == 0u32 && ty == PT_CEXP_U32 {
                // `CEXP.cpp:91-97`: the charge starts as plasma instead of a
                // solid, whatever the data row says.
                part_ctype[s] = PT_PLSM_U32;
            }
            part_temp[s] = if (mask & CR_SET_TEMP) != 0u32 {
                f32::reinterpret(create_queue[base + CR_TEMP as usize])
            } else {
                f32::reinterpret(spawn_defaults[db])
            };
            if (mask & CR_SET_TEMP) == 0u32 && ty == PT_CEXP_U32 {
                // `CEXP.cpp:93`: the blast temperature the fuse waits for.
                part_temp[s] = 9720.0f32;
            }
            part_tmp[s] = if (mask & CR_SET_TMP) != 0u32 {
                create_queue[base + CR_TMP as usize] as i32
            } else {
                spawn_defaults[db + 3usize] as i32
            };
            part_tmp2[s] = if (mask & CR_SET_TMP2) != 0u32 {
                create_queue[base + CR_TMP2 as usize] as i32
            } else {
                spawn_defaults[db + 4usize] as i32
            };
            part_tmp3[s] = if (mask & CR_SET_TMP3) != 0u32 {
                create_queue[base + CR_TMP3 as usize] as i32
            } else {
                spawn_defaults[db + 5usize] as i32
            };
            part_tmp4[s] = if (mask & CR_SET_TMP4) != 0u32 {
                create_queue[base + CR_TMP4 as usize] as i32
            } else {
                spawn_defaults[db + 6usize] as i32
            };

            part_dcolour[s] = if (mask & CR_SET_DCOLOUR) != 0u32 {
                create_queue[base + CR_DCOLOUR as usize]
            } else if ty == PT_LED_U32 {
                // `LED.cpp:95-98`: a fresh LED is white.
                0xFFFF_FFFFu32
            } else {
                0u32
            };
            // `create_part` copies `DefaultProperties`, whose velocity is always
            // zero, and then runs the element's `Create` hook. A queued creation
            // has to reproduce both or a spawned photon would sit still.
            let roll = hash_u32(frame_seed ^ (dst << 7u32) ^ (ty << 3u32));
            let mut rvx = seed_f32(ty as f32, 0.0f32);
            let mut rvy = seed_f32(ty as f32, 0.0f32);
            if (mask & (CR_SET_VX | CR_SET_VY)) != 0u32 {
                rvx = f32::reinterpret(create_queue[base + CR_VX as usize]);
                rvy = f32::reinterpret(create_queue[base + CR_VY as usize]);
            } else if ty == PT_PHOT_U32 {
                // `PHOT.cpp:176-184`: 3 px/frame along one of eight directions.
                let a = (roll % 8u32) as f32 * core::f32::consts::FRAC_PI_4;
                rvx = 3.0f32 * a.cos();
                rvy = 3.0f32 * a.sin();
            } else if ty == PT_ELEC_U32 || ty == PT_PROT_U32 || ty == PT_GRVT_U32 {
                // `ELEC.cpp:44-50` / `PROT.cpp:236-242`: 2 px/frame at any angle,
                // quantised to 10 degree steps for a proton.
                let mut div = 360u32;
                if ty == PT_PROT_U32 {
                    div = 36u32;
                }
                let a = (roll % div) as f32 / (div as f32) * core::f32::consts::TAU;
                rvx = 2.0f32 * a.cos();
                rvy = 2.0f32 * a.sin();
            } else if ty == PT_NEUT_U32 {
                let h = hash_u32(frame_seed ^ (dst << 11u32) ^ ty);
                let angle = (h & 0xFFFFu32) as f32 * (core::f32::consts::TAU / 65_535.0f32);
                // `NEUT.cpp:255-257`: `r = rng.between(128, 255) / 127.0f`, so a
                // fresh neutron leaves at 1.008..2.008 px/frame.
                let radius = (128.0f32 + ((h >> 16u32) & 0x7Fu32) as f32) / 127.0f32;
                rvx = angle.cos() * radius;
                rvy = angle.sin() * radius;
            } else if ty == PT_LAVA_U32 || ty == PT_FIRE_U32 || ty == PT_ACID_U32 {
                let h = hash_u32(frame_seed ^ (dst << 13u32) ^ (ty << 1u32));
                rvx = ((h & 0xFFu32) as f32 * (1.0f32 / 255.0f32) - 0.5f32) * 0.2f32;
                rvy = (((h >> 8u32) & 0xFFu32) as f32 * (1.0f32 / 255.0f32) - 0.5f32) * 0.2f32;
            } else if ty == PT_UVRD_U32 {
                // `UVRD.cpp:129-135`: 2 px/frame along
                // `rng.between(0, 359)` degrees.
                let a = (roll % 360u32) as f32 * (core::f32::consts::PI / 180.0f32);
                rvx = 2.0f32 * a.cos();
                rvy = 2.0f32 * a.sin();
            }
            part_vx[s] = rvx;
            part_vy[s] = rvy;
            // The element `Create` hooks the host allocator runs
            // (`SimBuffers::run_create_hook`) are random or multi-field, so a
            // queued creation has to reproduce them here. `VIRS` / `VRSS` /
            // `VRSG` need no hook any more: their `tmp4` timer is a plain
            // `DefaultProperties` value in the table above.
            if (mask & CR_SET_LIFE) == 0u32 {
                if ty == PT_PLSM_U32 || ty == PT_HFLM_U32 {
                    // `CFLM.cpp:69-72` / `PLSM.cpp:47-50`.
                    part_life[s] = 50i32 + (roll % 150u32) as i32;
                } else if ty == PT_FIRE_U32 {
                    // `FIRE.cpp:64-67`.
                    part_life[s] = 120i32 + (roll % 50u32) as i32;
                } else if ty == PT_TRON_U32 {
                    part_life[s] = 5i32;
                } else if ty == PT_NEUT_U32 {
                    // `NEUT.cpp:252-258`: `life = rng.between(480, 959)`.
                    part_life[s] = 480i32 + (roll % 480u32) as i32;
                } else if ty == PT_LAVA_U32 {
                    // `LAVA.cpp:73-76`: `life = rng.between(240, 359)`.
                    part_life[s] = 240i32 + (roll % 120u32) as i32;
                } else if ty == PT_WARP_U32 {
                    // `WARP.cpp:47-50`.
                    part_life[s] = 70i32 + (roll % 95u32) as i32;
                } else if ty == PT_SING_U32 {
                    // `SING.cpp:44-47`.
                    part_life[s] = 60i32 + (roll % 50u32) as i32;
                } else if ty == PT_ELEC_U32 || ty == PT_PROT_U32 {
                    // `ELEC.cpp:44-50` / `PROT.cpp:236-242`.
                    part_life[s] = 680i32;
                } else if ty == PT_GRVT_U32 {
                    // `GRVT.cpp:44-50`.
                    part_life[s] = 250i32 + (roll % 200u32) as i32;
                } else if ty == PT_CEXP_U32 {
                    // `CEXP.cpp:96`: life is the blast pressure.
                    part_life[s] = 256i32;
                } else if ty == PT_BFLM_U32 {
                    // `BFLM.cpp:103-106`: `life = rng.between(80, 120)`.
                    part_life[s] = 80i32 + (roll % 41u32) as i32;
                } else if ty == PT_UVRD_U32 {
                    // `UVRD.cpp:132`: five frames of travel.
                    part_life[s] = 5i32;
                }
            }
            // The `tmp`-seeded hooks. A caller that overrides the field keeps its
            // value, exactly as a `CR_SET_*` override does for `create_part`.
            if (mask & CR_SET_TMP) == 0u32 {
                if ty == PT_TRON_U32 {
                    // `TRON.cpp:194-205`: head marker, direction and hue.
                    let hue = (roll % 360u32) as i32;
                    let dir = ((roll / 360u32) % 4u32) as i32;
                    part_tmp[s] = 1i32 | (dir << 5i32) | (hue << 7i32);
                } else if ty == PT_CLST_U32 {
                    // `CLST.cpp:47-50`.
                    part_tmp[s] = (roll % 7u32) as i32;
                } else if ty == PT_PTNM_U32 {
                    // `PTNM.cpp:44-50`: one in fifteen starts as plutonium.
                    let hit = roll % 15u32;
                    if hit == 0u32 {
                        part_tmp[s] = 1i32;
                    }
                } else if ty == PT_SLCN_U32 {
                    // `SLCN.cpp:59-66`: the sparkle phase word.
                    part_tmp[s] = 0x10_0000i32 + (roll % 0x90_0000u32) as i32;
                } else if ty == PT_SEED_U32 {
                    // `SEED.cpp:267-275`: every genome bit is a coin flip, so a
                    // uniform 15-bit word has the same distribution.
                    part_tmp[s] = (roll & 0x7FFF_u32) as i32;
                } else if ty == PT_CLNT_U32 {
                    // `CLNT.cpp:86-89`: default cooling target, in Celsius.
                    part_tmp[s] = 22i32;
                } else if ty == PT_CEXP_U32 {
                    // `CEXP.cpp:95`: fuse length.
                    part_tmp[s] = 100i32;
                } else if ty == PT_DFOM_U32 {
                    // `DFOM.cpp:160-165`: the spawn pixel, `tmp = x`.
                    part_tmp[s] = create_queue[base + CR_X as usize] as i32;
                } else if ty == PT_MISL_U32 {
                    // `MISL.cpp:148-152`: target x in 1..=610.
                    part_tmp[s] = 1i32 + (roll % 610u32) as i32;
                }
            }
            if (mask & CR_SET_TMP2) == 0u32 {
                if ty == PT_SEED_U32 {
                    // `SEED.cpp:268-275`: the other three genome words.
                    part_tmp2[s] = ((roll >> 7u32) & 0x7FFF_u32) as i32;
                } else if ty == PT_TRON_U32 {
                    part_tmp2[s] = 4i32;
                } else if ty == PT_CRMC_U32 {
                    // `CRMC.cpp:47-50`.
                    part_tmp2[s] = (roll % 5u32) as i32;
                } else if ty == PT_PQRT_U32 || ty == PT_ROCK_U32 || ty == PT_QRTZ_U32 {
                    // `PQRT.cpp` / `ROCK.cpp` / `QRTZ.cpp:44-50`.
                    part_tmp2[s] = (roll % 11u32) as i32;
                } else if ty == PT_CWIR_U32 {
                    // `CWIR.cpp:93-96`: melting point, in Kelvin.
                    part_tmp2[s] = 2000i32;
                } else if ty == PT_GRPH_U32 {
                    // `GRPH.cpp:169-173`: `tmp2 = rng.between(0, 5)`.
                    part_tmp2[s] = (roll % 6u32) as i32;
                } else if ty == PT_DFOM_U32 {
                    // `DFOM.cpp:160-165`: the spawn pixel, `tmp2 = y`.
                    part_tmp2[s] = create_queue[base + CR_Y as usize] as i32;
                } else if ty == PT_MISL_U32 {
                    // `MISL.cpp:148-152`: target y in 1..=380.
                    part_tmp2[s] = 1i32 + (roll % 380u32) as i32;
                } else if ty == PT_PRMT_U32 {
                    // `PRMT.cpp:107-110`: `tmp2 = rng.between(0, 10)`.
                    part_tmp2[s] = (roll % 11u32) as i32;
                } else if ty == PT_ALMP_U32 {
                    // `ALMP.cpp:158-161`: `tmp2 = rng.between(0, 6)`.
                    part_tmp2[s] = (roll % 7u32) as i32;
                }
            }
            if (mask & CR_SET_TMP3) == 0u32 {
                // `SEED.cpp:270-275`: the third genome word.
                if ty == PT_SEED_U32 {
                    part_tmp3[s] = ((roll >> 14u32) & 0x7FFF_u32) as i32;
                }
                // `SEED.cpp:272-275`: the fourth genome word.
                if ty == PT_SEED_U32 && (mask & CR_SET_TMP4) == 0u32 {
                    part_tmp4[s] = ((roll >> 21u32) & 0x7FFF_u32) as i32;
                }
                // `GLAS.cpp` / `TUNG.cpp` / `QRTZ.cpp`: the cell pressure the
                // pixel was created under, as `int(pv * 64)`.
                if ty == PT_GLAS_U32 || ty == PT_TUNG_U32 || ty == PT_QRTZ_U32 {
                    let px = create_queue[base + CR_X as usize];
                    let py = create_queue[base + CR_Y as usize];
                    let cell = ((py / CELL_U32) * xcells + (px / CELL_U32)) as usize;
                    part_tmp3[s] = (air_pv[cell] * 64.0f32) as i32;
                }
                // `GRPH.cpp:172`: the ignition countdown starts full.
                if ty == PT_GRPH_U32 {
                    part_tmp3[s] = 100i32;
                }
                // `DFOM.cpp:164`: the coalesce timer.
                if ty == PT_DFOM_U32 {
                    part_tmp3[s] = 40i32;
                }
            }
        }
    }
}

#[cube(launch_unchecked)]
fn reset_create_queue_kernel(create_queue: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0usize {
        create_queue[0].store(0u32);
    }
}

/// Empty the creation queue. Called at the top of the reaction step.
pub(crate) fn reset_gpu_create_queue(bufs: &SimBuffers) {
    unsafe {
        reset_create_queue_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
        );
    }
}

/// Allocate and initialise every particle the reaction passes asked for: a
/// fresh slot for each [`request_create`] record, an in-place reset of the named
/// slot for each [`request_replace`].
///
/// Must run after [`refresh_gpu_free_slots`] (so the device window holds this
/// frame's free slots) and before the passes that allocate from the window
/// themselves, so a burst of creations cannot starve `portals` / `rays` /
/// `stickman`.
pub(super) fn drain_gpu_creates(bufs: &SimBuffers, dims: &PassDims) {
    let defaults_len = PT_NUM * tpt_core::elements::SPAWN_DEFAULT_WORDS;
    let (cubes, dim) = launch_grid(SimBuffers::GPU_CREATE_COUNT);
    unsafe {
        create_alloc_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            BufferArg::from_raw_parts(bufs.gpu_free_slots.clone(), SimBuffers::GPU_SPAWN_COUNT + 2),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            GPU_CREATE_COUNT_U32,
        );
        create_fields_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            BufferArg::from_raw_parts(bufs.gpu_spawn_defaults.clone(), defaults_len),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            dims.frame_seed,
            dims.xcells,
        );
    }
}

// `wall_bmap` stores the wall id in its low byte; the kernel compares against
// `u32` words, so the ids are mirrored here. The `const` assertions below keep
// the mirrors honest.
pub(super) const W_WALLELEC: u32 = 1;
pub(super) const W_EWALL: u32 = 2;
pub(super) const W_DETECT: u32 = 3;
pub(super) const W_ALLOWLIQUID: u32 = 6;
pub(super) const W_ALLOWALLELEC: u32 = 11;
pub(super) const W_EHOLE: u32 = 12;

const _: () = assert!(W_WALLELEC == WL_WALLELEC as u32);
const _: () = assert!(W_EWALL == WL_EWALL as u32);
const _: () = assert!(W_DETECT == WL_DETECT as u32);
const _: () = assert!(W_ALLOWLIQUID == WL_ALLOWLIQUID as u32);
const _: () = assert!(W_ALLOWALLELEC == WL_ALLOWALLELEC as u32);
const _: () = assert!(W_EHOLE == WL_EHOLE as u32);

/// `wall_emap` value meaning "this wall is carrying a spark".
pub(super) const EMAP_SPARKING: u32 = 12;
/// `wall_emap` value written when a wall is charged by an adjacent spark.
pub(super) const EMAP_CHARGED: u32 = 16;

/// Deterministic integer avalanche (the finaliser of `xxh3`).
///
/// CubeCL has no `wrapping_mul`; integer multiplication in the IR is already
/// modular, so a plain `*` is the wrapping variant on every backend.
#[cube]
pub(crate) fn hash_u32(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 16u32;
    h *= 2_246_822_507u32;
    h ^= h >> 13u32;
    h *= 3_266_489_909u32;
    h ^= h >> 16u32;
    h
}

/// `index * salt`, kept behind a call so the product is never constant-folded.
///
/// In the unexpanded (host) build of a `#[cube]` function `ABSOLUTE_POS` is the
/// literal `2`, so writing `(idx as u32) * 2_246_822_519u32` inline makes the
/// compiler evaluate `2 * 2_246_822_519` and reject the build with
/// `arithmetic_overflow`. Integer multiplication in the IR is modular, so this
/// is the wrapping product on every backend.
#[cube]
pub(super) fn mix_index(index: u32, salt: u32) -> u32 {
    index * salt
}

/// Uniform pseudo random number in `[0, 1)` derived from `seed`.
#[cube]
pub(crate) fn rand_unit(seed: u32) -> f32 {
    (hash_u32(seed) & 0x00FF_FFFFu32) as f32 * (1.0f32 / 16_777_216.0f32)
}

/// `|v|` for a signed integer (CubeCL has no `i32::abs`).
#[cube]
pub(crate) fn iabs(v: i32) -> i32 {
    let mut r = v;
    if v < 0i32 {
        r = -v;
    }
    r
}

/// Read one `u32` word of the element property table.
///
/// Out-of-range element ids read as zero, i.e. "no properties", which makes
/// every property test fail instead of reading past the buffer.
#[cube]
pub(crate) fn elem_word(words: &[u32], ty: u32, word: u32) -> u32 {
    let mut v = 0u32;
    if ty < ELEM_SLOTS {
        v = words[((ty * 32u32) + word) as usize];
    }
    v
}

/// Packed spatial-map entry at `(x, y)`, or `0` when the pixel is outside the
/// canvas or empty.
#[cube]
pub(crate) fn pmap_at(pmap: &[u32], xres: u32, yres: u32, x: i32, y: i32) -> u32 {
    let mut v = 0u32;
    if x >= 0i32 && y >= 0i32 && (x as u32) < xres && (y as u32) < yres {
        v = pmap[((y as u32) * xres + (x as u32)) as usize];
    }
    v
}

/// Particle slot stored in a packed spatial-map word.
///
/// A word with no id bits is malformed (the packing always stores `id + 1`), so
/// it degrades to slot 0 instead of wrapping to a huge index.
#[cube]
pub(crate) fn pmap_slot(packed: u32) -> usize {
    let raw = packed >> 9u32;
    let mut slot = 0usize;
    if raw > 0u32 {
        slot = (raw - 1u32) as usize;
    }
    slot
}

/// `true` when `INSL` (or unmeltable `RSSS`) sits between two adjacent pixels.
///
/// This is TPT's `parts_avg(i, j, PT_INSL)` test: it samples the pixel at the
/// midpoint of the two particle positions, so a one-pixel insulator wall blocks
/// conduction across it while a conductor that merely *touches* insulation is
/// unaffected.
#[cube]
pub(crate) fn insulated(
    pmap: &[u32],
    xres: u32,
    yres: u32,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
) -> bool {
    let mx = (x1 + x2) / 2i32;
    let my = (y1 + y2) / 2i32;
    let v = pmap_at(pmap, xres, yres, mx, my);
    let mut blocked = false;
    if v != 0u32 {
        let t = v & 0x1FFu32;
        if t == PT_INSL || t == PT_RSSS {
            blocked = true;
        }
    }
    blocked
}

/// Sentinel returned by [`gpu_alloc_slot`] when the device spawn window is full.
pub(super) const GPU_ALLOC_NONE: u32 = 0xFFFF_FFFF;

/// `true` when the slot sits in an unpowered `WL_STASIS` cell.
///
/// The flag is refreshed by the life pass (`Simulation.cpp:2335-2338`), which
/// is the first pass to visit every slot, so the later reaction passes test one
/// bit instead of binding the wall maps.
#[cube]
pub(crate) fn stasis_frozen(part_flags: &mut [u32], id: usize) -> bool {
    (part_flags[id] & 0x20u32) != 0u32
}

/// `SOAP.cpp:73-88` `Element_SOAP_detach`: unlink a bubble from its mates.
///
/// `PIPE.cpp:255`, `STOR.cpp:75` and `PRTI.cpp:93` call this immediately
/// before they consume a `SOAP` particle. `ctype` bit 2 marks a live `tmp`
/// link and bit 4 a live `tmp2` link; the mate's reciprocal bit is cleared,
/// then the consumed bubble's `ctype` is zeroed. `n_slots` is TPT's
/// `validIndex` bound, i.e. the particle pool capacity.
///
/// Every slot has at most one mate per direction, so the two cross-slot
/// clears touch slots no other thread of the consumer kernels owns; the
/// consumed slot itself is killed or stored by the caller afterwards.
#[cube]
pub(crate) fn soap_detach(
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    slot: usize,
    n_slots: u32,
) {
    let ctype = part_ctype[slot];
    if (ctype & 2u32) != 0u32 {
        let mate = part_tmp[slot];
        if mate >= 0i32
            && (mate as u32) < n_slots
            && part_ty[mate as usize] == PT_SOAP
            && (part_ctype[mate as usize] & 4u32) != 0u32
        {
            part_ctype[mate as usize] ^= 4u32;
        }
    }
    if (ctype & 4u32) != 0u32 {
        let mate = part_tmp2[slot];
        if mate >= 0i32
            && (mate as u32) < n_slots
            && part_ty[mate as usize] == PT_SOAP
            && (part_ctype[mate as usize] & 2u32) != 0u32
        {
            part_ctype[mate as usize] ^= 2u32;
        }
    }
    part_ctype[slot] = 0u32;
}

/// Pop one free particle slot on the device from `free_slots`.
///
/// `free_slots[0]` is the number of available slots collected by
/// [`collect_gpu_free_slots_kernel`], `free_slots[1]` is the atomic allocation
/// cursor, and `free_slots[2..]` holds the slot ids.
#[cube]
pub(super) fn gpu_alloc_slot(free_slots: &mut [Atomic<u32>]) -> u32 {
    let avail = free_slots[0].load();
    let idx = free_slots[1].fetch_add(1u32);
    // Both halves accumulate with `|=` instead of overwriting a seeded local:
    // a plain assignment that lands at this depth makes CubeCL emit a store
    // through the initialiser constant, which fails IR verification
    // (`Expected operand type cube.ptr, but found builtin.integer`) and poisons
    // every buffer the kernel was bound to.
    let mut slot = 0u32;
    if idx < avail {
        slot |= free_slots[(idx + 2u32) as usize].load();
    }
    let mut missing = 0u32;
    if idx >= avail {
        missing |= GPU_ALLOC_NONE;
    }
    slot | missing
}

#[cube(launch_unchecked)]
fn reset_gpu_free_slots_kernel(free_slots: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0usize {
        free_slots[0].store(0u32);
        free_slots[1].store(0u32);
    }
}

#[cube(launch_unchecked)]
fn collect_gpu_free_slots_kernel(
    part_ty: &[u32],
    part_flags: &[u32],
    free_slots: &mut [Atomic<u32>],
    start_slot: u32,
    count: u32,
    capacity: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < count {
        let slot = start_slot + idx;
        if slot < capacity {
            let s = slot as usize;
            // `0` = `PT_NONE`, `0x04` = `FLAG_PORTAL_HELD`.
            if part_ty[s] == 0u32 && (part_flags[s] & 0x04u32) == 0u32 {
                let pos = free_slots[0].fetch_add(1u32);
                if pos < count {
                    free_slots[(pos + 2u32) as usize].store(slot);
                }
            }
        }
    }
}

/// Rebuild the device free-slot table (`SimBuffers::gpu_free_slots`) over the
/// reserved window [`SimBuffers::GPU_SPAWN_START`]..[`SimBuffers::GPU_SPAWN_END`].
pub(crate) fn refresh_gpu_free_slots(bufs: &SimBuffers) {
    let count = SimBuffers::GPU_SPAWN_COUNT;
    let free_len = count + 2;
    let (cubes, dim) = launch_grid(count);
    unsafe {
        reset_gpu_free_slots_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.gpu_free_slots.clone(), free_len),
        );
    }
    unsafe {
        collect_gpu_free_slots_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_free_slots.clone(), free_len),
            SimBuffers::GPU_SPAWN_START as u32,
            count as u32,
            bufs.capacity as u32,
        );
    }
}

/// Number of cubes and units per cube needed to cover `units` work items.
pub(crate) fn launch_grid(units: usize) -> (u32, u32) {
    let cubes = units.div_ceil(CUBE_DIM as usize).max(1) as u32;
    (cubes, CUBE_DIM)
}
