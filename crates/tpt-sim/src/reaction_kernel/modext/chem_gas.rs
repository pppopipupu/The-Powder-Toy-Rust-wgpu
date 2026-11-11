//! Reactive gases of the second set: `CHLR`, `ACTY`, `NTRG`, `RADN`, `DFOM`,
//! `CLUD`.
//!
//! Every `Update` is the reference's, in its own switch order:
//!
//! | kernel | reference | behaviour |
//! |--------|-----------|-----------|
//! | [`chlr_kernel`] | `CHLR.cpp:51-161` | photochemical `H2` reaction, water family below/above 50 C, `IRON`/`BMTL` rust, organic decay, the `ACTY` heat trickle |
//! | [`acty_kernel`] | `ACTY.cpp:52-110` | acetylene's 3500 C burn with the `tmp2` fuse, `O2` ignition, `SPRK`/`SMKE`/`FIRE`/`PLSM` ignition at 1800 C |
//! | [`ntrg_kernel`] | `NTRG.cpp:48-77` | `H2 -> NITR` (the hydrogen dies) and fire/plasma extinguishing |
//! | [`radn_kernel`] | `RADN.cpp:51-127` | radon's `tmp`/`tmp2` decay chain and UV absorption |
//! | [`dfom_kernel`] | `DFOM.cpp:51-129` | defence foam's advection, temperature relaxation, timers, jitter, recovery and cold snap |
//! | [`clud_kernel`] | `CLUD.cpp:50-92` | the cloud's charge / lightning / rain state machine |
//!
//! `NTRG`'s `HighPressure`/`LowTemperature` conversions into `LNTG`
//! (`NTRG.cpp:36-41`) are data-table transitions run by the phase-transition
//! pass, not by the `Update`.
//!
//! # Shape
//!
//! All six `Update`s are slot parallel over the live pool, like the rest of
//! this group: the reference walks the particle array once per frame, and a
//! pixel-parallel scan could visit one slot twice while a stale `pmap` word
//! still names it (a replacing `create_part(i, x, y, t)` writes the new pixel
//! and leaves the old map word behind until the movement pass rebuilds the
//! map). Positions come from `part_x`/`part_y` with `UPDATE_FUNC_ARGS`'s
//! `(int)(x + 0.5f)` rounding.
//!
//! Cross-slot writes are the reference's: type changes, deaths and the
//! `CHLR`/`RADN` `tmp` reads. `create_part(i, ...)` (slot reuse) goes through
//! [`request_replace`] and `create_part(-1, ...)` (fresh particle) through
//! [`request_create`], both drained at the end of the reaction step.
//!
//! # Deviations
//!
//! * `kill_part` frees the slot but the reference keeps executing the rest of
//!   that particle's `Update` on it (`parts[i]` is reused by the next
//!   allocation). `CHLR` and `ACTY` can kill themselves mid-scan; the port
//!   stops the scan there. The reference's own `part_change_type` on the freed
//!   slot fails outright, so nothing it does after the death is visible on the
//!   dead slot — only a later neighbour conversion or kill that the reference
//!   would still have performed is dropped here (the per-contact odds are
//!   unchanged).
//! * `create_part(i, ...)` mid-scan resets the slot before the following
//!   neighbours are read; a queued replacement only lands at the drain. The
//!   port stops the scan after queueing one, so no neighbour roll is decided
//!   from a half-reset slot.
//! * The reference's `create_part(-1, x, y, ...)` occupancy test comes from
//!   `eval_move`, which lets a neutron share the pixel of a
//!   `PROP_NEUTPASS` element (`RADN.cpp:55-58`); the drain's own test only
//!   sees the map word, so that one creation opts out with
//!   `CR_NO_OCCUPANCY`.
//! * The `Created` hook of `LIGH` (`LIGH.cpp:312-336`) has no drain branch; a
//!   cloud queues the bolt's three hook fields itself (see
//!   [`queue_ligh`]), but the `gravOut` term of `GetGravityField` is not
//!   included, the same omission as `combustion::fireworks`'s
//!   `config_gravity`.

// Collapsing a nested `if` changes what the CubeCL macro expands, and the
// device IR has no `RangeInclusive::contains`, so the literal branches stay.
#![allow(clippy::collapsible_if)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::PassDims;
use super::super::combustion::{chance, rand_range};
use super::super::shared::{
    CR_NO_OCCUPANCY, CR_SET_LIFE, CR_SET_TEMP, CR_SET_TMP, CR_SET_TMP2, CREATE_QUEUE_WORDS,
    hash_u32, launch_grid, mix_index, pmap_at, pmap_slot, request_create, request_replace,
    seed_u32, stasis_frozen,
};
use crate::buffers::SimBuffers;

/// `GravityMode` discriminants (`tpt_core::particle`), mirrored for the
/// kernel body.
const GRAV_VERTICAL_U32: u32 = 0;
const GRAV_RADIAL_U32: u32 = 2;
const GRAV_CUSTOM_U32: u32 = 3;
const _: () = assert!(
    GRAV_VERTICAL_U32 == tpt_core::particle::GravityMode::Vertical as u32
        && GRAV_RADIAL_U32 == tpt_core::particle::GravityMode::Radial as u32
        && GRAV_CUSTOM_U32 == tpt_core::particle::GravityMode::Custom as u32
);

/// `LIGH.cpp:315-322`: `v = -1` (`create_part`'s default) gives every bolt a
/// 30-frame life.
const LIGH_LIFE_I32: i32 = 30;
/// `LIGH.cpp:323`: `temp = life * 150`, the bolt's power as a temperature.
const LIGH_TEMP_F32: f32 = 4_500.0f32;
const _: () = assert!(LIGH_TEMP_F32 == LIGH_LIFE_I32 as f32 * 150.0f32);
/// `LIGH.cpp:335`: the deferred-bending state a fresh head carries.
const LIGH_TMP2_I32: i32 = 4;
/// `LIGH.cpp:326`: the field strength below which a bolt picks a random aim.
const LIGH_WEAK_FIELD_F32: f32 = 0.0016f32;
/// `LIGH.cpp:331`: the random kick's radius on a weak field.
const LIGH_WEAK_KICK_F32: f32 = 0.04f32;
// `Simulation.cpp:1918-1945` is the reference for [`request_replace`]: the
// slot keeps its id and restarts from the target's defaults, so a queued
// replacement runs the target's `Create` hook in the drain. Every call below
// therefore passes a bare mask or the hook fields the target needs.

/// `CHLR.cpp:51-161`: chlorine.
///
/// The `tmp` countdown and the below-freezing `vy` pinch run first, then the
/// radius-2 ring in the reference's `rx`-outer/`ry`-inner order. The water
/// family's branch is decided by the chlorine's own temperature, so a hot
/// chlorine flashes one neighbour per successful roll into `ACID` (1 in 400,
/// with the cell pressure spiked to 6.0) while a cold one freezes it to
/// `DSTW` and dies 1 in 300. `H2` is the photochemical partner: `tmp` resets
/// to 50 and 1 in 400 swaps the chlorine for `ACID` at the hydrogen's pixel.
/// `ACTY` heats the chlorine by 1 K per frame, and 1 in 700 swaps the chlorine
/// for `LRBD` at the acetylene's pixel.
/// Stickmen die 1 in 70, `IRON`/`BMTL` rust to `BRMT` 1 in 1000 (consuming
/// the chlorine) and the organics go 1 in 200.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn chlr_kernel(
    pmap: &[u32],
    part_ty: &mut [Atomic<u32>],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_vy: &mut [f32],
    part_flags: &mut [u32],
    part_x: &[f32],
    part_y: &[f32],
    air_pv: &mut [f32],
    create_queue: &mut [Atomic<u32>],
    used_slots: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i < used_slots as usize {
        if part_ty[i].load() == PT_CHLR {
            if stasis_frozen(part_flags, i) {
                terminate!();
            }
            let px = i32::cast_from(part_x[i] + 0.5f32);
            let py = i32::cast_from(part_y[i] + 0.5f32);
            let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
            let seed = hash_u32(frame_seed ^ mix_index(i as u32, 0x4348_4C52u32));
            let mut temp = part_temp[i];
            // `CHLR.cpp:53-54`: the chlorine's own `tmp` countdown.
            if part_tmp[i] > 0i32 {
                part_tmp[i] -= 1i32;
            }
            // `CHLR.cpp:56-59`: below freezing the gas sinks.
            if temp < 273.15f32 {
                part_vy[i] = 0.1f32;
            }
            let mut dead = false;
            for n in 0u32..25u32 {
                // `CHLR.cpp:61-63`: `rx` outer, `ry` inner.
                let dx = (n / 5u32) as i32 - 2i32;
                let dy = (n % 5u32) as i32 - 2i32;
                if (dx != 0i32 || dy != 0i32) && !dead {
                    let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                    if nv != 0u32 {
                        let nid = pmap_slot(nv);
                        let nty = part_ty[nid].load();
                        let roll = seed ^ (n << 8u32);
                        if nty == PT_WATR
                            || nty == PT_SLTW
                            || nty == PT_CBNW
                            || nty == PT_DSTW
                            || nty == PT_WTRV
                        {
                            // `CHLR.cpp:71-96`: above 50 C the water flashes
                            // into acid, below it the chlorine freezes it.
                            if temp > 323.15f32 {
                                if chance(roll ^ 0x11u32, 1u32, 400u32) {
                                    air_pv[cidx] = 6.0f32;
                                    part_ty[nid].store(PT_NONE);
                                    part_life[i] = 200i32;
                                    part_ty[i].store(PT_ACID);
                                    dead = true;
                                }
                            } else if temp < 323.15f32 {
                                part_ty[nid].store(PT_DSTW);
                                if chance(roll ^ 0x12u32, 1u32, 300u32) {
                                    part_ty[i].store(PT_NONE);
                                    dead = true;
                                }
                            }
                        } else if nty == PT_H2 {
                            // `CHLR.cpp:99-107`: the photochemical reaction
                            // arms the chlorine's `tmp` and swaps it for acid
                            // at the hydrogen's pixel.
                            part_tmp[i] = 50i32;
                            if chance(roll ^ 0x13u32, 1u32, 400u32) {
                                part_ty[nid].store(PT_NONE);
                                request_replace(
                                    create_queue,
                                    i as u32,
                                    PT_ACID,
                                    px + dx,
                                    py + dy,
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
                                dead = true;
                            }
                        } else if nty == PT_ACTY {
                            // `CHLR.cpp:110-119`: acetylene heats the
                            // chlorine; one in seven hundred makes `LRBD`.
                            temp += 1.0f32;
                            if chance(roll ^ 0x14u32, 1u32, 700u32) {
                                part_ty[nid].store(PT_NONE);
                                request_replace(
                                    create_queue,
                                    i as u32,
                                    PT_LRBD,
                                    px + dx,
                                    py + dy,
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
                                dead = true;
                            }
                        } else if nty == PT_STKM || nty == PT_STKM2 || nty == PT_FIGH {
                            // `CHLR.cpp:121-129`.
                            if chance(roll ^ 0x15u32, 1u32, 70u32) {
                                part_ty[nid].store(PT_NONE);
                            }
                        } else if nty == PT_IRON || nty == PT_BMTL {
                            // `CHLR.cpp:132-140`: rust the metal and die.
                            if chance(roll ^ 0x16u32, 1u32, 1000u32) {
                                part_ty[nid].store(PT_BRMT);
                                part_ty[i].store(PT_NONE);
                                dead = true;
                            }
                        } else if nty == PT_PLNT
                            || nty == PT_VINE
                            || nty == PT_WOOD
                            || nty == PT_YEST
                            || nty == PT_SEED
                        {
                            // `CHLR.cpp:143-154`: both sides go.
                            if chance(roll ^ 0x17u32, 1u32, 200u32) {
                                part_ty[nid].store(PT_NONE);
                                part_ty[i].store(PT_NONE);
                                dead = true;
                            }
                        }
                    }
                }
            }
            if !dead {
                part_temp[i] = temp;
            }
        }
    }
}

/// `ACTY.cpp:52-110`: acetylene.
///
/// The `tmp` flag and the `tmp2` fuse are a burn state: while `tmp` is set the
/// gas holds 3500 C and the fuse counts down, and an empty fuse kills the
/// slot. `O2` contact sets `tmp` and consumes the oxygen, `ACTY` contact
/// spreads the flag, and a spark, smoke, fire or plasma neighbour not already
/// burning at 3500 C relights the gas at 1800 C. An ignited (`tmp == 1`) gas
/// next to anything but `ACTY`/`FIRE`/`PLSM`/`O2` turns into `FIRE`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn acty_kernel(
    pmap: &[u32],
    part_ty: &mut [Atomic<u32>],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_flags: &mut [u32],
    part_x: &[f32],
    part_y: &[f32],
    used_slots: u32,
    xres: u32,
    yres: u32,
) {
    let i = ABSOLUTE_POS;
    if i < used_slots as usize {
        if part_ty[i].load() == PT_ACTY {
            if stasis_frozen(part_flags, i) {
                terminate!();
            }
            let px = i32::cast_from(part_x[i] + 0.5f32);
            let py = i32::cast_from(part_y[i] + 0.5f32);
            let mut temp = part_temp[i];
            let mut tmp = part_tmp[i];
            let mut tmp2 = part_tmp2[i];
            // `ACTY.cpp:54-58`: the burning gas holds 3500 C and burns its
            // fuse. `tmp2` starts at 40 (`ACTY.cpp:33`).
            if tmp > 0i32 {
                temp = 3500.0f32 + 273.15f32;
                tmp2 -= 1i32;
            }
            // `ACTY.cpp:60-63`: an empty fuse kills the gas. The reference
            // runs the rest of the update on the freed slot, where
            // `part_change_type` fails; the port stops here.
            if tmp2 <= 0i32 {
                part_ty[i].store(PT_NONE);
                terminate!();
            }
            for n in 0u32..25u32 {
                // `ACTY.cpp:65-67`: `rx` outer, `ry` inner, radius 2.
                let dx = (n / 5u32) as i32 - 2i32;
                let dy = (n % 5u32) as i32 - 2i32;
                if dx != 0i32 || dy != 0i32 {
                    let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                    if nv != 0u32 {
                        let nid = pmap_slot(nv);
                        let nty = part_ty[nid].load();
                        // `ACTY.cpp:73-76`: an ignited gas beside anything
                        // that is not fuel, fire, plasma or oxygen turns into
                        // fire.
                        if nty != PT_NONE
                            && nty != PT_ACTY
                            && nty != PT_FIRE
                            && nty != PT_PLSM
                            && nty != PT_O2
                            && tmp == 1i32
                        {
                            part_ty[i].store(PT_FIRE);
                        }
                        if nty == PT_O2 {
                            // `ACTY.cpp:79-84`: oxygen ignites and is
                            // consumed.
                            tmp = 1i32;
                            part_ty[nid].store(PT_NONE);
                        } else if nty == PT_ACTY {
                            // `ACTY.cpp:85-90`: the flame spreads through a
                            // burning neighbour.
                            if part_tmp[nid] > 0i32 {
                                tmp = 1i32;
                            }
                        } else if nty == PT_SPRK
                            || nty == PT_SMKE
                            || nty == PT_FIRE
                            || nty == PT_PLSM
                        {
                            // `ACTY.cpp:92-103`: relight at 1800 C unless the
                            // gas is already at its 3500 C burn.
                            if tmp != 1i32 {
                                part_life[i] = 35i32;
                                temp = 1800.0f32 + 273.15f32;
                                part_ty[i].store(PT_FIRE);
                            }
                        }
                    }
                }
            }
            part_temp[i] = temp;
            part_tmp[i] = tmp;
            part_tmp2[i] = tmp2;
        }
    }
}

/// `NTRG.cpp:48-77`: nitrogen.
///
/// `H2` becomes `NITR` — the reference's `part_change_type(i, ...)` rewrites
/// the nitrogen's own slot and then kills the hydrogen — and a `FIRE`/`PLSM`
/// neighbour is extinguished. The scan keeps running after the nitrogen has
/// become nitre, exactly like the reference (it never returns early).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn ntrg_kernel(
    pmap: &[u32],
    part_ty: &mut [Atomic<u32>],
    part_flags: &mut [u32],
    part_x: &[f32],
    part_y: &[f32],
    used_slots: u32,
    xres: u32,
    yres: u32,
) {
    let i = ABSOLUTE_POS;
    if i < used_slots as usize {
        if part_ty[i].load() == PT_NTRG {
            if stasis_frozen(part_flags, i) {
                terminate!();
            }
            let px = i32::cast_from(part_x[i] + 0.5f32);
            let py = i32::cast_from(part_y[i] + 0.5f32);
            for n in 0u32..9u32 {
                // `NTRG.cpp:51-53`: `rx` outer, `ry` inner.
                let dx = (n / 3u32) as i32 - 1i32;
                let dy = (n % 3u32) as i32 - 1i32;
                if dx != 0i32 || dy != 0i32 {
                    let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                    if nv != 0u32 {
                        let nid = pmap_slot(nv);
                        let nty = part_ty[nid].load();
                        if nty == PT_H2 {
                            // `NTRG.cpp:58-63`.
                            part_ty[i].store(PT_NITR);
                            part_ty[nid].store(PT_NONE);
                        } else if nty == PT_FIRE || nty == PT_PLSM {
                            // `NTRG.cpp:65-69`: fire and plasma go out.
                            part_ty[nid].store(PT_NONE);
                        }
                    }
                }
            }
        }
    }
}

/// `RADN.cpp:51-127`: radon.
///
/// A fresh atom (`tmp == 0`) leaks one neutron in three hundred and latches
/// `tmp = 1` at the same odds. Absorption of a `UVRD` photon sets `tmp2 = 700`
/// (`RADN.cpp:119-124`); the reduced timer then fires six independent
/// creations per frame while it is below 500 — `PHOT` 1 in 100, `EMBR` 1 in
/// 190, `GRVT` 1 in 120, `PROT` 1 in 90, `BVBR` 1 in 290 and `NEUT` 1 in 140.
/// Every one of them is `create_part(i, ...)`, so each restarts the same slot
/// and the last success wins; the port queues exactly that one record.
/// Stickmen and fighters die 1 in 70.
///
/// The reference skips the photon probe whenever the scanned pixel has no
/// particle (`RADN.cpp:104-105` `continue`), so a photon over an empty pixel is
/// never absorbed; the port keeps the probe inside the same `nv != 0` guard.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn radn_kernel(
    pmap: &[u32],
    photons: &mut [Atomic<u32>],
    part_ty: &mut [Atomic<u32>],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_flags: &mut [u32],
    part_x: &[f32],
    part_y: &[f32],
    create_queue: &mut [Atomic<u32>],
    used_slots: u32,
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i < used_slots as usize {
        if part_ty[i].load() == PT_RADN {
            if stasis_frozen(part_flags, i) {
                terminate!();
            }
            let px = i32::cast_from(part_x[i] + 0.5f32);
            let py = i32::cast_from(part_y[i] + 0.5f32);
            let seed = hash_u32(frame_seed ^ mix_index(i as u32, 0x5241_444Eu32));
            let mut tmp = part_tmp[i];
            let mut tmp2 = part_tmp2[i];
            // `RADN.cpp:53-62`: the one-way decay latch.
            if tmp == 0i32 {
                if chance(seed ^ 0x01u32, 1u32, 300u32) {
                    // `RADN.cpp:55-58`: `create_part(-1, x, y, PT_NEUT)`.
                    // Radon's `PROP_NEUTPASS` lets the neutron share its
                    // pixel, which `eval_move` allows and the drain's map
                    // test does not, hence the occupancy opt-out.
                    request_create(
                        create_queue,
                        PT_NEUT,
                        px,
                        py,
                        CR_NO_OCCUPANCY,
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
                if chance(seed ^ 0x02u32, 1u32, 300u32) {
                    tmp = 1i32;
                }
            }
            let mut replaced = false;
            // `RADN.cpp:65-95`: the ionised countdown.
            if tmp2 > 0i32 {
                tmp2 -= 1i32;
                if tmp2 < 500i32 {
                    let mut spawn_ty = seed_u32(i as u32, PT_NONE);
                    if chance(seed ^ 0x03u32, 1u32, 100u32) {
                        spawn_ty = PT_PHOT;
                    }
                    if chance(seed ^ 0x04u32, 1u32, 190u32) {
                        spawn_ty = PT_EMBR;
                    }
                    if chance(seed ^ 0x05u32, 1u32, 120u32) {
                        spawn_ty = PT_GRVT;
                    }
                    if chance(seed ^ 0x06u32, 1u32, 90u32) {
                        spawn_ty = PT_PROT;
                    }
                    if chance(seed ^ 0x07u32, 1u32, 290u32) {
                        spawn_ty = PT_BVBR;
                    }
                    if chance(seed ^ 0x08u32, 1u32, 140u32) {
                        spawn_ty = PT_NEUT;
                    }
                    if spawn_ty != PT_NONE {
                        // The last successful `create_part(i, ...)` in the
                        // reference's order is the one left in the slot.
                        request_replace(
                            create_queue,
                            i as u32,
                            spawn_ty,
                            px,
                            py,
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
                        replaced = true;
                    }
                }
            }
            // `RADN.cpp:91-126`: stickmen die, UV is absorbed.
            for n in 0u32..25u32 {
                // `RADN.cpp:96-98`: `rx` outer, `ry` inner.
                let dx = (n / 5u32) as i32 - 2i32;
                let dy = (n % 5u32) as i32 - 2i32;
                if dx != 0i32 || dy != 0i32 {
                    let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                    if nv != 0u32 {
                        let nid = pmap_slot(nv);
                        let nty = part_ty[nid].load();
                        if nty == PT_STKM || nty == PT_STKM2 || nty == PT_FIGH {
                            // `RADN.cpp:106-115`.
                            if chance(seed ^ (n << 8u32) ^ 0x11u32, 1u32, 70u32) {
                                part_ty[nid].store(PT_NONE);
                            }
                        }
                        // `RADN.cpp:119-124`: the matter layer's
                        // `continue` above also skips the photon probe.
                        let pidx = (((py + dy) as u32) * xres + (px + dx) as u32) as usize;
                        let rp = photons[pidx].load();
                        if (rp & 0x1FFu32) == PT_UVRD {
                            // Only one radon may claim the photon.
                            if photons[pidx].compare_exchange_weak(rp, 0u32) == rp {
                                tmp2 = 700i32;
                                part_ty[pmap_slot(rp)].store(PT_NONE);
                            }
                        }
                    }
                }
            }
            // A queued replacement resets `tmp`/`tmp2` at the drain, exactly
            // where the reference's `create_part` does.
            if !replaced {
                part_tmp[i] = tmp;
                part_tmp2[i] = tmp2;
            }
        }
    }
}

/// `DFOM.cpp:51-129`: defence foam.
///
/// Pressure above one unit or 60 C refreshes the 250-frame absorption timer and
/// drags the foam along the air velocity (`ADVECTION = 0.4`). The foam then
/// relaxes towards 22 C (`-4` K above it, `+1` K below), ticks `tmp3`/`tmp4`
/// down, shoves itself away from an adjacent foam pixel while `tmp3` is in
/// (1, 15), pulls itself back towards the spawn pixel (`tmp`/`tmp2`) while
/// `tmp4` is in (0, 200) on one roll in three, and re-latches the spawn pixel
/// whenever it is colder than 0 C.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn dfom_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_flags: &mut [u32],
    air_pv: &[f32],
    air_vx: &[f32],
    air_vy: &[f32],
    used_slots: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i < used_slots as usize {
        if part_ty[i] == PT_DFOM {
            if stasis_frozen(part_flags, i) {
                terminate!();
            }
            let px = i32::cast_from(part_x[i] + 0.5f32);
            let py = i32::cast_from(part_y[i] + 0.5f32);
            let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
            let seed = hash_u32(frame_seed ^ mix_index(i as u32, 0x4446_4F4Du32));
            let mut temp = part_temp[i];
            let mut tmp = part_tmp[i];
            let mut tmp2 = part_tmp2[i];
            let mut tmp3 = part_tmp3[i];
            let mut tmp4 = part_tmp4[i];
            let mut vx = part_vx[i];
            let mut vy = part_vy[i];
            // `DFOM.cpp:53-58`: a shock refreshes the timer and the air drags
            // the foam along.
            if air_pv[cidx] > 1.0f32 || temp > 60.0f32 + 273.15f32 {
                tmp4 = 250i32;
                vx += 0.4f32 * air_vx[cidx];
                vy += 0.4f32 * air_vy[cidx];
            }
            // `DFOM.cpp:60-67`: relax towards 22 C.
            if temp > 22.0f32 + 273.15f32 {
                temp -= 4.0f32;
            }
            if temp < 22.0f32 + 273.15f32 {
                temp += 1.0f32;
            }
            // `DFOM.cpp:68-75`.
            if tmp3 > 0i32 {
                tmp3 -= 1i32;
            }
            if tmp4 > 0i32 {
                tmp4 -= 1i32;
            }
            // `DFOM.cpp:77-96`: the uneven texture. Neighbours are sampled in
            // the reference's `rx`-outer/`ry`-inner order over `rx, ry in
            // {-1, 0}`, and the last foam neighbour wins both axes.
            if tmp3 < 15i32 && tmp3 > 1i32 {
                // `DFOM.cpp:79-83` is unreachable behind `tmp3 > 1`; the
                // reference still carries it, so the port keeps the shape.
                if tmp3 == 1i32 {
                    tmp = part_x[i] as i32;
                    tmp2 = part_y[i] as i32;
                }
                for n in 0u32..4u32 {
                    let dx = (n / 2u32) as i32 - 1i32;
                    let dy = (n % 2u32) as i32 - 1i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            if part_ty[nid] == PT_DFOM {
                                vx = (-dx * 2i32) as f32;
                                vy = (-dy * 2i32) as f32;
                            }
                        }
                    }
                }
            }
            // `DFOM.cpp:99-116`: structure recovery, one roll in three.
            if tmp4 < 200i32 && tmp4 > 0i32 {
                if chance(seed, 1u32, 3u32) {
                    // C++ compares the float position against the int latch.
                    if part_x[i] > tmp as f32 {
                        vx -= 2.0f32;
                    } else {
                        vx += 2.0f32;
                    }
                    if part_y[i] > tmp2 as f32 {
                        vy -= 2.0f32;
                    } else {
                        vy += 2.0f32;
                    }
                }
            }
            // `DFOM.cpp:123-127`: below 0 C the foam sets its shape.
            if temp < 273.15f32 {
                tmp = part_x[i] as i32;
                tmp2 = part_y[i] as i32;
            }
            part_temp[i] = temp;
            part_tmp[i] = tmp;
            part_tmp2[i] = tmp2;
            part_tmp3[i] = tmp3;
            part_tmp4[i] = tmp4;
            part_vx[i] = vx;
            part_vy[i] = vy;
        }
    }
}

/// `Simulation::GetGravityField` (`Simulation.cpp:2092-2127`) for a particle of
/// gravity `1.0`, the vector `LIGH.cpp:324` aims a fresh bolt along.
///
/// Vertical uses `(0, 1)`, radial points at the canvas centre with the
/// reference's signed `0.01 - hypot` denominator, custom is the configured
/// vector and `Off` stays zero (which falls through to the weak-field kick in
/// [`ligh_aim`]). Like `combustion::fireworks`'s `config_gravity` the
/// `newtonGrav` term is not included.
#[cube]
#[allow(clippy::too_many_arguments)]
fn gravity_at(
    mode: u32,
    custom_gx: f32,
    custom_gy: f32,
    px: i32,
    py: i32,
    xres: u32,
    yres: u32,
) -> (f32, f32) {
    let mut gx = 0.0f32;
    let mut gy = 0.0f32;
    if mode == GRAV_VERTICAL_U32 {
        gy = 1.0f32;
    } else if mode == GRAV_RADIAL_U32 {
        let dx = (px - (xres / 2u32) as i32) as f32;
        let dy = (py - (yres / 2u32) as i32) as f32;
        if dx != 0.0f32 || dy != 0.0f32 {
            let d = 0.01f32 - (dx * dx + dy * dy).sqrt();
            gx = dx / d;
            gy = dy / d;
        }
    } else if mode == GRAV_CUSTOM_U32 {
        gx = custom_gx;
        gy = custom_gy;
    }
    (gx, gy)
}

/// `LIGH.cpp:324-334` plus the `tmp` line: the bolt's aim.
///
/// A field weaker than `0.0016` gets the reference's random kick of up to
/// `0.04` before `atan2(-gy, gx)` decides the direction; the result is the
/// angle in degrees plus `rng.between(-20, 20)`.
#[cube]
fn ligh_aim(seed: u32, gx0: f32, gy0: f32) -> i32 {
    let mut gx = gx0;
    let mut gy = gy0;
    let gsize = gx * gx + gy * gy;
    if gsize < LIGH_WEAK_FIELD_F32 {
        let angle = rand_range(seed, 0i32, 6283i32) as f32 * 0.001f32;
        let weak = gsize.sqrt();
        gx += angle.cos() * (LIGH_WEAK_KICK_F32 - weak);
        gy += angle.sin() * (LIGH_WEAK_KICK_F32 - weak);
    }
    let aim = ((-gy).atan2(gx) * 180.0f32 / core::f32::consts::PI) as i32;
    let jitter = rand_range(seed ^ 0x9E37_79B9u32, -20i32, 20i32);
    (aim + jitter + 360i32) % 360i32
}

/// `create_part(-1, x, y, PT_LIGH)` for [`clud_kernel`].
///
/// The drain has no `LIGH` create hook, so the record carries the three fields
/// the hook would set: `life = 30` (`v = -1`), `temp = life * 150` and
/// `tmp2 = 4`. The bolt's aim comes from the gravity field at the creation
/// pixel, exactly like the host allocator's hook
/// (`buffers::particles::run_create_hook`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn queue_ligh(
    create_queue: &mut [Atomic<u32>],
    grav_mode: u32,
    custom_gx: f32,
    custom_gy: f32,
    px: i32,
    py: i32,
    xres: u32,
    yres: u32,
    seed: u32,
) {
    // `create_part` rejects an out-of-bounds pixel before the hook runs, so
    // the field is only sampled in bounds.
    if px >= 0i32 && py >= 0i32 && (px as u32) < xres && (py as u32) < yres {
        let (gx, gy) = gravity_at(grav_mode, custom_gx, custom_gy, px, py, xres, yres);
        let aim = ligh_aim(seed, gx, gy);
        request_create(
            create_queue,
            PT_LIGH,
            px,
            py,
            CR_SET_LIFE | CR_SET_TEMP | CR_SET_TMP | CR_SET_TMP2,
            LIGH_LIFE_I32,
            0u32,
            LIGH_TEMP_F32,
            aim,
            LIGH_TMP2_I32,
            0i32,
            0i32,
            0u32,
            0u32,
            0.0f32,
            0.0f32,
        );
    }
}

/// `CLUD.cpp:50-92`: cloud.
///
/// `tmp` is the charge: it climbs one per frame to 1000 (`tmp2 != 1`), fires a
/// lightning bolt 25 pixels below at exactly 1000 (1 in 1500) and a second
/// chance 30 pixels below on the frame it passes 995 (1 in 1000). `tmp2 = 1`
/// starts the rain phase: `tmp` counts back down while it is above -5, the
/// cell pressure follows `tmp / 150`, and one roll in fifty drops water below
/// the cloud, or snow while it is under 0 C. A negative `tmp` resets `tmp2`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn clud_kernel(
    part_ty: &[u32],
    part_temp: &[f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_flags: &mut [u32],
    part_x: &[f32],
    part_y: &[f32],
    air_pv: &mut [f32],
    create_queue: &mut [Atomic<u32>],
    grav_mode: u32,
    custom_gx: f32,
    custom_gy: f32,
    used_slots: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i < used_slots as usize {
        if part_ty[i] == PT_CLUD {
            if stasis_frozen(part_flags, i) {
                terminate!();
            }
            let px = i32::cast_from(part_x[i] + 0.5f32);
            let py = i32::cast_from(part_y[i] + 0.5f32);
            let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
            let seed = hash_u32(frame_seed ^ mix_index(i as u32, 0x434C_5544u32));
            let mut tmp = part_tmp[i];
            let mut tmp2 = part_tmp2[i];
            // `CLUD.cpp:52-55`: charge while the rain phase has not started.
            if tmp < 1000i32 && tmp2 != 1i32 {
                tmp += 1i32;
            }
            if tmp == 1000i32 {
                // `CLUD.cpp:56-63`: the fully charged cloud starts raining and
                // may strike 25 pixels below.
                tmp2 = 1i32;
                if chance(seed ^ 0x01u32, 1u32, 1500u32) {
                    queue_ligh(
                        create_queue,
                        grav_mode,
                        custom_gx,
                        custom_gy,
                        px,
                        py + 25i32,
                        xres,
                        yres,
                        seed ^ 0x11u32,
                    );
                }
            } else if tmp == 995i32 {
                // `CLUD.cpp:64-69`: one last bolt on the way up.
                if chance(seed ^ 0x02u32, 1u32, 1000u32) {
                    queue_ligh(
                        create_queue,
                        grav_mode,
                        custom_gx,
                        custom_gy,
                        px,
                        py + 30i32,
                        xres,
                        yres,
                        seed ^ 0x12u32,
                    );
                }
            }
            // `CLUD.cpp:70-84`: the rain phase.
            if tmp2 == 1i32 && tmp > -5i32 {
                tmp -= 1i32;
                air_pv[cidx] = tmp as f32 / 150.0f32;
                if chance(seed ^ 0x03u32, 1u32, 50u32) {
                    if part_temp[i] >= 273.15f32 {
                        request_create(
                            create_queue,
                            PT_WATR,
                            px,
                            py + 1i32,
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
                    } else {
                        request_create(
                            create_queue,
                            PT_SNOW,
                            px,
                            py + 1i32,
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
                }
            }
            // `CLUD.cpp:85-88`: the spent charge resets the phase.
            if tmp < 0i32 {
                tmp2 = 0i32;
            }
            part_tmp[i] = tmp;
            part_tmp2[i] = tmp2;
        }
    }
}

/// Launch the second set's reactive gases for one reaction step.
///
/// The six launches run in reference slot order — `CHLR` 204, `ACTY` 205,
/// `NTRG` 211, `RADN` 217, `DFOM` 220, `CLUD` 223 — so a pass that reads a
/// neighbour's `temp`/`tmp` sees this frame's value, exactly like the
/// reference's sequential particle loop (e.g. `CHLR`'s `ACTY` heat trickle
/// reads an `ACTY` that has already burnt this frame).
pub(crate) fn launch(bufs: &SimBuffers, dims: &PassDims, seed: u32) {
    if dims.used_slots == 0 || dims.capacity == 0 {
        return;
    }
    let (cubes, dim) = launch_grid(dims.used_slots);
    let used = dims.used_slots as u32;
    // `LIGH`'s create hook samples `GetGravityField`; the kernel reproduces
    // the configured part of it (`gravity_at`).
    let grav_mode = bufs.config.gravity_mode as u32;
    let custom_gx = bufs.config.custom_gravity_x;
    let custom_gy = bufs.config.custom_gravity_y;

    unsafe {
        chlr_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            used,
            dims.xres,
            dims.yres,
            dims.xcells,
            seed,
        );
    }
    unsafe {
        acty_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            used,
            dims.xres,
            dims.yres,
        );
    }
    unsafe {
        ntrg_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            used,
            dims.xres,
            dims.yres,
        );
    }
    unsafe {
        radn_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            used,
            dims.xres,
            dims.yres,
            seed,
        );
    }
    unsafe {
        dfom_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), dims.cells),
            used,
            dims.xres,
            dims.yres,
            dims.xcells,
            seed,
        );
    }
    unsafe {
        clud_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            grav_mode,
            custom_gx,
            custom_gy,
            used,
            dims.xres,
            dims.yres,
            dims.xcells,
            seed,
        );
    }
}
