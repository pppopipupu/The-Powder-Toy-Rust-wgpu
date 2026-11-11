//! Powders, solids and sticky liquids of the second set: `PHOS`, `CMNT`,
//! `RUBR`, `GRPH`, `ALUM`, `ALMP`, `CLRC`, `NAPM`.
//!
//! Ports phosphorus's slow reddening and melt (`PHOS.cpp:49-120`), cement
//! setting in water (`CMNT.cpp:51-82`), rubber's bounce state
//! (`RUBR.cpp:51-115`), graphite's conduction and ignition (`GRPH.cpp:52-148`),
//! aluminium's pressure-burst shell (`ALUM.cpp:71-269`), aluminium powder's
//! sparking burn (`ALMP.cpp:66-125`), the clear protective coat
//! (`CLRC.cpp:51-80`) and napalm's stick-and-burn (`NAPM.cpp:47-116`).
//!
//! # Three passes
//!
//! The eight updates are split by the buffer set each one reads, because a
//! single pass would need nineteen storage bindings and the particle passes
//! live under a sixteen-binding ceiling:
//!
//! * [`chem_solid_update_kernel`] — `PHOS`, `CMNT`, `GRPH`, `CLRC`, `NAPM`
//!   (fifteen bindings, including the photon layer and the element table).
//! * [`chem_solid_burn_kernel`] — `ALMP`, `RUBR` (sixteen bindings: the burn
//!   roll needs the element table for the `TYPE_SOLID` / `TYPE_ENERGY` bounce
//!   test and the air block masks).
//! * [`alum_kernel`] — `ALUM` alone (sixteen bindings: the nine-cell pressure
//!   sample needs `pv` *and* both air velocity components, and the powder
//!   break still needs the block masks).
//!
//! The temperature melts of `PHOS` (317 K), `CMNT` (2887.15 K) and `GRPH`
//! (3950.15 K) are the element table's `high_temp` / `high_temp_transition`
//! row, applied by the thermal phase-transition pass; `ALUM` and `ALMP` carry
//! `ITH` there and melt from their own updates below.
//!
//! The burn pass runs **before** the aluminium pass: `ALUM.cpp:153` changes a
//! broken shell pixel into `ALMP` in place, and TPT never re-visits a slot it
//! has already updated this frame, so the fresh powder must not be picked up by
//! `ALMP.cpp`'s ignition scan in the same tick.
//!
//! # Conversions
//!
//! TPT's `create_part(i, x, y, t)` rebuilds the slot from the target's
//! `DefaultProperties` row and runs its `Create` hook (`Simulation.cpp:1941`),
//! so the `PHOS`/`ALMP` self-ignitions and the `PHOS` clear-flame swap go
//! through [`request_replace`]. `create_part(-1, ...)` is a brand-new particle
//! and becomes a [`request_create`] record. `part_change_type` only rewrites
//! the type field and keeps every other field, so those conversions (`PHOS` to
//! `VINE`, `GRPH` to `FIRE`/`SPRK`/`CO2`/`UVRD`, `ALUM`/`ALMP` to `LAVA`,
//! `NAPM` to `FIRE`, `CMNT`'s water kill, `GRPH`'s oxygen/neutron kill) are
//! direct field writes, the convention [`super::thermo`] and `combustion`
//! already use.
//!
//! Two reference quirks are kept because they change the outcome:
//!
//! * `PHOS.cpp:78` adds the *pixel* offsets `rx`/`ry` to the phosphor's own
//!   *cell* index (`pv[(y / CELL) + ry][(x / CELL) + rx]`) instead of sampling
//!   the oxygen's cell. The stamp is reproduced literally and clipped to the
//!   air grid, like the `AMBE`/`FUEL` stamps of [`super::thermo`].
//! * `CLRC.cpp:73` only reads the photon layer after the pixel passed the
//!   "there is a matter particle here" test, so a `UVRD` over an empty pixel
//!   does not coat the liquid. `PHOS.cpp:115` has the same shape: its `!r`
//!   guard `continue`s past the photon read, so only a `UVRD` that shares a
//!   pixel with matter charges the phosphor. Both are reproduced with the
//!   photon read inside the matter guard.

// Same rationale as `pixel_mode.rs`: collapsing a nested `if` changes the
// CubeCL expansion, and `x = k * x` / the redundant `as u32` are the forms the
// device IR accepts.
#![allow(
    clippy::collapsible_if,
    clippy::assign_op_pattern,
    clippy::unnecessary_cast
)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::PassDims;
use super::super::combustion::{chance, clampi, rand_range};
use super::super::shared::{
    CR_SET_CTYPE, CR_SET_LIFE, CR_SET_TMP, CR_SET_TMP2, CR_SET_VX, CR_SET_VY, CREATE_QUEUE_WORDS,
    WORD_PROPERTIES, elem_word, hash_u32, launch_grid, mix_index, pmap_at, pmap_slot,
    request_create, request_create_plain, request_replace, seed_u32, stasis_frozen,
};
use crate::buffers::SimBuffers;

/// `ALMP.cpp:8`: burn health a fresh powder particle starts with (also its
/// `DefaultProperties.tmp`).
const ALMP_BURN_HEALTH: i32 = 40;
/// `ALUM.cpp:26`: `BaseStrength`.
const ALUM_BASE_STRENGTH: f32 = 10.0f32;

/// `(x, y)` word of the photon layer, or `0` outside the canvas.
///
/// Mirrors [`pmap_at`] for the `TYPE_ENERGY` map: `photons` is one packed word
/// per pixel and a zero word means the pixel holds no energy particle.
#[cube]
fn photon_at(photons: &mut [u32], xres: u32, yres: u32, x: i32, y: i32) -> u32 {
    let mut v = 0u32;
    if x >= 0i32 && y >= 0i32 && (x as u32) < xres && (y as u32) < yres {
        v = photons[((y as u32) * xres + (x as u32)) as usize];
    }
    v
}

/// `create_part(slot, x, y, ty)` for the slot's own pixel: no field overrides.
#[cube]
fn replace_plain(create_queue: &mut [Atomic<u32>], slot: u32, ty: u32, x: i32, y: i32) {
    request_replace(
        create_queue,
        slot,
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

/// `PHOS`, `CMNT`, `GRPH`, `CLRC` and `NAPM` (`PHOS.cpp:49-120`,
/// `CMNT.cpp:51-82`, `GRPH.cpp:52-148`, `CLRC.cpp:51-80`, `NAPM.cpp:47-116`).
///
/// Pixel parallel over the spatial map. Every thread owns exactly one slot, so
/// the cross-pixel writes (`VINE` conversion, water/oxygen/neutron kills, the
/// rubber/phosphorus neighbour scans) are the same writes the reference makes
/// by slot id; two writers of one slot resolve by dispatch order, exactly like
/// TPT's slot order.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn chem_solid_update_kernel(
    pmap: &[u32],
    photons: &mut [u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_temp: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_flags: &mut [u32],
    air_pv: &mut [f32],
    elem_words: &[u32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty != PT_NONE {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                // Per-particle, per-frame roll; every reference `rng` call
                // below derives its own seed from it.
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 0x9E37_79B9u32));

                if ty == PT_PHOS {
                    // `PHOS.cpp:51-52`: the reddening/UV timer.
                    if part_tmp3[id] > 0i32 {
                        part_tmp3[id] -= 1i32;
                    }
                    // `PHOS.cpp:54-61`: an ignited phosphor turns to fire.
                    if part_tmp2[id] == 1i32 && chance(roll ^ 0x01u32, 1u32, 500u32) {
                        part_life[id] = 150i32;
                        part_ty[id] = PT_FIRE;
                    }
                    // `PHOS.cpp:63-66`: slow reddening.
                    if part_tmp[id] < 255i32 && chance(roll ^ 0x02u32, 1u32, 8u32) {
                        part_tmp[id] += 1i32;
                    }
                    for n in 0u32..25u32 {
                        let rx = (n % 5u32) as i32 - 2i32;
                        let ry = (n / 5u32) as i32 - 2i32;
                        if rx != 0i32 || ry != 0i32 {
                            let nx = px + rx;
                            let ny = py + ry;
                            let nv = pmap_at(pmap, xres, yres, nx, ny);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let nroll = roll ^ mix_index(n, 0x85EB_CA6Bu32);
                                if nty == PT_O2 {
                                    // `PHOS.cpp:76-81`: the reference adds the
                                    // pixel offsets to the phosphor's *cell*
                                    // index; clipped to the grid here.
                                    let cx = (px as u32 / CELL) as i32 + rx;
                                    let cy = (py as u32 / CELL) as i32 + ry;
                                    if cx >= 0i32
                                        && cy >= 0i32
                                        && (cx as u32) < xcells
                                        && (cy as u32) < ycells
                                    {
                                        air_pv[((cy as u32) * xcells + (cx as u32)) as usize] +=
                                            10.0f32;
                                    }
                                    // `create_part(i, x, y, PT_FIRE)`: the row
                                    // is rebuilt with `FIRE`'s create hook.
                                    replace_plain(create_queue, id as u32, PT_FIRE, px, py);
                                } else if nty == PT_OIL {
                                    // `PHOS.cpp:82-87`.
                                    if part_tmp[id] > 0i32 {
                                        part_tmp[id] -= 1i32;
                                    }
                                } else if nty == PT_CFLM {
                                    // `PHOS.cpp:88-92`.
                                    replace_plain(create_queue, id as u32, PT_CFLM, px, py);
                                } else if nty == PT_FIRE || nty == PT_PLSM {
                                    // `PHOS.cpp:93-98`.
                                    part_tmp2[id] = 1i32;
                                } else if nty == PT_PHOS {
                                    // `PHOS.cpp:99-104`: the neighbour's word is
                                    // read live, a same-frame ignite spreads.
                                    if part_tmp2[nid] > 0i32 {
                                        part_tmp2[id] = 1i32;
                                    }
                                } else if nty == PT_PLNT {
                                    // `PHOS.cpp:105-113`: one in a hundred
                                    // feedings fertilises the plant and the
                                    // phosphor is consumed.
                                    if chance(nroll, 1u32, 100u32) {
                                        part_ty[nid] = PT_VINE;
                                        part_ty[id] = PT_NONE;
                                    }
                                }
                                // `PHOS.cpp:115-119`: the photon test runs
                                // only for the offsets that hold a matter
                                // particle, because the `!r` guard above
                                // `continue`s past it.
                                let phv = photon_at(photons, xres, yres, nx, ny);
                                if phv != 0u32 {
                                    let phid = pmap_slot(phv);
                                    if part_ty[phid] == PT_UVRD {
                                        part_tmp3[id] += 2i32;
                                    }
                                }
                            }
                        }
                    }
                } else if ty == PT_CMNT {
                    // `CMNT.cpp:53-60`: once wetted, the cement counter runs
                    // up to 300 and slowly warms below 90C.
                    if part_tmp2[id] == 1i32 && part_tmp[id] < 300i32 {
                        if part_temp[id] < 363.15f32 {
                            part_temp[id] += 1.0f32;
                        }
                        part_tmp[id] += 1i32;
                    }
                    // `CMNT.cpp:61-65`: fully set cement stops moving.
                    if part_tmp[id] == 300i32 {
                        part_vx[id] = 0.0f32;
                        part_vy[id] = 0.0f32;
                    }
                    // `CMNT.cpp:66-81`: the first water pixel in the 5x5 is
                    // drunk and sets the cement; `tmp2` is re-read each step so
                    // only one pixel is consumed per frame.
                    for n in 0u32..25u32 {
                        let rx = (n % 5u32) as i32 - 2i32;
                        let ry = (n / 5u32) as i32 - 2i32;
                        if rx != 0i32 || ry != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if (nty == PT_WATR
                                    || nty == PT_DSTW
                                    || nty == PT_SLTW
                                    || nty == PT_CBNW)
                                    && part_tmp2[id] != 1i32
                                {
                                    part_tmp2[id] = 1i32;
                                    part_ty[nid] = PT_NONE;
                                }
                            }
                        }
                    }
                } else if ty == PT_GRPH {
                    // `GRPH.cpp:54-55`: the glow timer decays above 100.
                    if part_tmp3[id] > 100i32 {
                        part_tmp3[id] -= 1i32;
                    }
                    // `GRPH.cpp:57-72`: a lit graphite pixel spits fire/carbon
                    // dioxide and eventually burns itself out.
                    if part_tmp[id] == 1i32 {
                        if chance(roll ^ 0x11u32, 1u32, 80u32) {
                            request_create_plain(create_queue, PT_FIRE, px, py - 1i32);
                        }
                        if chance(roll ^ 0x12u32, 1u32, 500u32) {
                            request_create_plain(create_queue, PT_CO2, px, py - 1i32);
                        }
                        if chance(roll ^ 0x13u32, 1u32, 700u32) {
                            part_life[id] = 70i32;
                            part_ty[id] = PT_FIRE;
                        }
                    }
                    // `GRPH.cpp:74-90`: a spark four pixels away turns the
                    // graphite into the conductor.
                    if part_life[id] == 0i32 {
                        for j in 0u32..4u32 {
                            let mut ox = 0i32;
                            let mut oy = 0i32;
                            if j == 0u32 {
                                ox = -4i32;
                            } else if j == 1u32 {
                                ox = 4i32;
                            } else if j == 2u32 {
                                oy = -4i32;
                            } else {
                                oy = 4i32;
                            }
                            let nv = pmap_at(pmap, xres, yres, px + ox, py + oy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nlife = part_life[nid];
                                if part_ty[nid] == PT_SPRK && nlife != 0i32 && nlife < 4i32 {
                                    part_ty[id] = PT_SPRK;
                                    part_life[id] = 4i32;
                                    part_ctype[id] = PT_GRPH;
                                }
                            }
                        }
                    }
                    // `GRPH.cpp:92-146`: the 5x5 reaction table. A pixel only
                    // falls back to the photon layer when it has no matter.
                    for n in 0u32..25u32 {
                        let rx = (n % 5u32) as i32 - 2i32;
                        let ry = (n / 5u32) as i32 - 2i32;
                        if rx != 0i32 || ry != 0i32 {
                            let nx = px + rx;
                            let ny = py + ry;
                            let mut nv = pmap_at(pmap, xres, yres, nx, ny);
                            let mut from_photon = false;
                            if nv == 0u32 {
                                nv = photon_at(photons, xres, yres, nx, ny);
                                from_photon = nv != 0u32;
                            }
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let nroll = roll ^ mix_index(n, 0x2545_F491u32);
                                if nty == PT_FIRE || nty == PT_SMKE || nty == PT_PLSM {
                                    // `GRPH.cpp:104-111`.
                                    if part_temp[id] > 693.15f32 {
                                        part_tmp[id] = 1i32;
                                    }
                                } else if nty == PT_GRPH {
                                    // `GRPH.cpp:112-117`.
                                    if part_tmp[nid] > 0i32 {
                                        part_tmp[id] = 1i32;
                                    }
                                } else if nty == PT_SPRK {
                                    // `GRPH.cpp:118-122`.
                                    part_tmp3[id] = 255i32;
                                } else if nty == PT_PHOT {
                                    // `GRPH.cpp:123-127`: the photon becomes a
                                    // UV ray. TPT passes the graphite's pixel
                                    // to `part_change_type`; the port changes
                                    // the photon where it was found.
                                    part_ty[nid] = PT_UVRD;
                                } else if nty == PT_O2 {
                                    // `GRPH.cpp:128-136`: graphite burns the
                                    // oxygen and becomes carbon dioxide.
                                    if chance(nroll, 1u32, 40u32) {
                                        part_ty[id] = PT_CO2;
                                        part_ty[nid] = PT_NONE;
                                    }
                                } else if nty == PT_NEUT {
                                    // `GRPH.cpp:137-144`; a neutron lives in
                                    // the photon layer, so its map word is
                                    // cleared with it.
                                    if chance(nroll, 1u32, 4u32) {
                                        part_ty[nid] = PT_NONE;
                                        if from_photon {
                                            let pidx = ((ny as u32) * xres + (nx as u32)) as usize;
                                            photons[pidx] = 0u32;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if ty == PT_CLRC {
                    // `CLRC.cpp:55-59`: a fresh coat is still sticky.
                    if part_life[id] > 1i32 {
                        part_vx[id] = 0.0f32;
                        part_vy[id] = 0.0f32;
                    }
                    // `CLRC.cpp:61-78`: solids (and cold copper) refresh the
                    // lifetime, and a UV ray under a matter particle makes the
                    // coat invisible.
                    for n in 0u32..25u32 {
                        let rx = (n % 5u32) as i32 - 2i32;
                        let ry = (n / 5u32) as i32 - 2i32;
                        if rx != 0i32 || ry != 0i32 {
                            let nx = px + rx;
                            let ny = py + ry;
                            let nv = pmap_at(pmap, xres, yres, nx, ny);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let solid = (elem_word(elem_words, nty, WORD_PROPERTIES)
                                    & TYPE_SOLID)
                                    != 0u32;
                                if solid || (nty == PT_COPR && part_tmp2[nid] == 0i32) {
                                    part_life[id] = 5i32;
                                }
                                let phv = photon_at(photons, xres, yres, nx, ny);
                                if phv != 0u32 && part_ty[pmap_slot(phv)] == PT_UVRD {
                                    part_tmp[id] = 1i32;
                                }
                            }
                        }
                    }
                } else if ty == PT_NAPM {
                    // `NAPM.cpp:51-59`.
                    if part_life[id] > 1i32 {
                        part_vx[id] = 0.0f32;
                        part_vy[id] = 0.0f32;
                    }
                    if part_temp[id] > 374.15f32 {
                        part_tmp[id] = 1i32;
                    }
                    // `NAPM.cpp:61-80`: burning napalm pressurises its own air
                    // cell, heats up to 1473.15 K and sheds fire and embers.
                    if part_tmp[id] == 1i32 {
                        let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                        air_pv[cidx] = 3.0f32;
                        if part_temp[id] < 1200.0f32 + 273.15f32 {
                            part_temp[id] += 30.15f32;
                        }
                        if chance(roll ^ 0x31u32, 1u32, 35u32) {
                            request_create_plain(create_queue, PT_FIRE, px, py - 1i32);
                        }
                        if chance(roll ^ 0x32u32, 1u32, 10u32) {
                            request_create_plain(create_queue, PT_EMBR, px, py - 1i32);
                        }
                        if chance(roll ^ 0x33u32, 1u32, 900u32) {
                            part_ty[id] = PT_FIRE;
                        }
                    }
                    // `NAPM.cpp:82-113`: solids make it stick, fire ignites it
                    // and a burning neighbour spreads the flame.
                    for n in 0u32..25u32 {
                        let rx = (n % 5u32) as i32 - 2i32;
                        let ry = (n / 5u32) as i32 - 2i32;
                        if rx != 0i32 || ry != 0i32 {
                            let nx = px + rx;
                            let ny = py + ry;
                            let mut nv = pmap_at(pmap, xres, yres, nx, ny);
                            if nv == 0u32 {
                                nv = photon_at(photons, xres, yres, nx, ny);
                            }
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if (elem_word(elem_words, nty, WORD_PROPERTIES) & TYPE_SOLID)
                                    != 0u32
                                {
                                    part_life[id] = 5i32;
                                }
                                if nty == PT_FIRE
                                    || nty == PT_SMKE
                                    || nty == PT_PLSM
                                    || nty == PT_SPRK
                                {
                                    part_tmp[id] = 1i32;
                                } else if nty == PT_NAPM {
                                    if part_tmp[nid] > 0i32 {
                                        part_tmp[id] = 1i32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `ALMP` and `RUBR` (`ALMP.cpp:66-125`, `RUBR.cpp:51-115`).
///
/// Runs before [`alum_kernel`] so the powder a broken aluminium shell leaves
/// behind (`ALUM.cpp:153`) is not scanned for ignition in the same frame.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn chem_solid_burn_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_temp: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_flags: &mut [u32],
    air_pv: &mut [f32],
    elem_words: &[u32],
    elem_block_air: &mut [u32],
    elem_block_airh: &mut [u32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty != PT_NONE {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 0x85EB_CA6Bu32));

                if ty == PT_ALMP {
                    // `ALMP.cpp:70`: oxidised powder (the oxide tower's cells)
                    // is inert and only melts back.
                    if part_ctype[id] != PT_O2 {
                        if part_tmp[id] >= ALMP_BURN_HEALTH {
                            // `ALMP.cpp:72-89`: any flame, plasma, spark or
                            // lightning in the 3x3 lights the powder. The roll
                            // is retaken for every hit, as in the reference.
                            for n in 0u32..9u32 {
                                let rx = (n % 3u32) as i32 - 1i32;
                                let ry = (n / 3u32) as i32 - 1i32;
                                if rx != 0i32 || ry != 0i32 {
                                    let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                                    if nv != 0u32 {
                                        let nid = pmap_slot(nv);
                                        let nty = part_ty[nid];
                                        if nty == PT_FIRE
                                            || nty == PT_PLSM
                                            || nty == PT_SPRK
                                            || nty == PT_LIGH
                                        {
                                            part_tmp[id] = ALMP_BURN_HEALTH
                                                - rand_range(
                                                    roll ^ mix_index(n, 0x2545_F491u32),
                                                    1i32,
                                                    10i32,
                                                );
                                        }
                                    }
                                }
                            }
                        } else if part_tmp[id] <= 0i32 {
                            // `ALMP.cpp:90-93`: burnt out, the pixel flashes
                            // into fire and the pressure spikes.
                            let cidx =
                                (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                            replace_plain(create_queue, id as u32, PT_FIRE, px, py);
                            air_pv[cidx] += 2.0f32;
                        } else if part_tmp[id] < ALMP_BURN_HEALTH
                            && chance(roll ^ 0x41u32, 1u32, 2u32)
                        {
                            // `ALMP.cpp:94-108`: erratic burn; two times in
                            // three the pixel throws a short-lived ember.
                            part_tmp[id] -= 1i32;
                            if chance(roll ^ 0x42u32, 2u32, 3u32) {
                                let cidx =
                                    (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                                air_pv[cidx] += 0.2f32;
                                let ex = px + rand_range(roll ^ 0x43u32, -1i32, 1i32);
                                let ey = py + rand_range(roll ^ 0x44u32, -1i32, 1i32);
                                let evx = rand_range(roll ^ 0x45u32, -4i32, 4i32) as f32;
                                let evy = rand_range(roll ^ 0x46u32, -4i32, 4i32) as f32;
                                request_create(
                                    create_queue,
                                    PT_EMBR,
                                    ex,
                                    ey,
                                    CR_SET_TMP | CR_SET_LIFE | CR_SET_VX | CR_SET_VY,
                                    50i32,
                                    0u32,
                                    0.0f32,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0u32,
                                    0u32,
                                    evx,
                                    evy,
                                );
                            } else {
                                let fx = px + rand_range(roll ^ 0x47u32, -1i32, 1i32);
                                let fy = py + rand_range(roll ^ 0x48u32, -1i32, 1i32);
                                request_create_plain(create_queue, PT_FIRE, fx, fy);
                            }
                        }
                    } else if part_temp[id] > 2072.0f32 + 273.15f32 {
                        // `ALMP.cpp:113-122`: the oxide melts back into
                        // aluminium; `part_change_type` keeps the fields the
                        // reference leaves alone.
                        part_ty[id] = PT_LAVA;
                        part_ctype[id] = PT_ALUM;
                        part_tmp[id] = 0i32;
                        part_tmp2[id] = 0i32;
                        part_tmp3[id] = 0i32;
                    }
                } else if ty == PT_RUBR {
                    // `RUBR.cpp:55-70`: burning rubber spits fire and carbon
                    // dioxide above itself and finally ignites.
                    if part_tmp2[id] == 1i32 {
                        if chance(roll ^ 0x51u32, 1u32, 50u32) {
                            request_create_plain(create_queue, PT_FIRE, px, py - 1i32);
                        }
                        if chance(roll ^ 0x52u32, 1u32, 300u32) {
                            request_create_plain(create_queue, PT_CO2, px, py - 1i32);
                        }
                        if chance(roll ^ 0x53u32, 1u32, 500u32) {
                            part_life[id] = 70i32;
                            part_ty[id] = PT_FIRE;
                        }
                    }
                    // `RUBR.cpp:72-78`: set rubber seals its cell against air
                    // movement and heat.
                    if part_tmp[id] != 0i32 {
                        part_vx[id] = 0.0f32;
                        part_vy[id] = 0.0f32;
                        let cx = (px as u32 / CELL) as i32;
                        let cy = (py as u32 / CELL) as i32;
                        if cx >= 0i32 && cy >= 0i32 && (cx as u32) < xcells && (cy as u32) < ycells
                        {
                            let cidx = ((cy as u32) * xcells + (cx as u32)) as usize;
                            elem_block_air[cidx] = 1u32;
                            elem_block_airh[cidx] = 0x8u32;
                        }
                    }
                    // `RUBR.cpp:79-82`: 230C cures the latex.
                    if part_temp[id] > 230.0f32 + 273.15f32 && chance(roll ^ 0x54u32, 1u32, 100u32)
                    {
                        part_tmp[id] = 1i32;
                    }
                    // `RUBR.cpp:84-113`.
                    for n in 0u32..9u32 {
                        let rx = (n % 3u32) as i32 - 1i32;
                        let ry = (n / 3u32) as i32 - 1i32;
                        if rx != 0i32 || ry != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let nprops = elem_word(elem_words, nty, WORD_PROPERTIES);
                                let nroll = roll ^ mix_index(n, 0x9E37_79B9u32);
                                // `RUBR.cpp:92-96`: cured rubber bounces every
                                // movable neighbour away.
                                if nty != PT_RUBR
                                    && part_tmp[id] == 1i32
                                    && (nprops & TYPE_SOLID) == 0u32
                                    && (nprops & TYPE_ENERGY) == 0u32
                                {
                                    part_vx[nid] = -1.3f32 * part_vx[nid];
                                    part_vy[nid] = -1.3f32 * part_vy[nid];
                                }
                                // `RUBR.cpp:97-104`: heat or fire sets it
                                // burning, and a burning neighbour spreads it.
                                if part_temp[id] >= 800.15f32
                                    || nty == PT_FIRE
                                    || nty == PT_PLSM
                                    || nty == PT_SMKE
                                {
                                    part_tmp2[id] = 1i32;
                                }
                                if nty == PT_RUBR && part_tmp2[nid] == 1i32 {
                                    part_tmp2[id] = 1i32;
                                }
                                // `RUBR.cpp:105-111`: gas and oil dissolve it.
                                if (nty == PT_GAS || nty == PT_OIL) && chance(nroll, 1u32, 150u32) {
                                    part_ty[id] = PT_NONE;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `ALUM.cpp:71-269`: the pressure-burst shell.
///
/// The particle samples its nine surrounding air cells for pressure and
/// velocity, folds them into a strain value and compares it against the
/// element's nominal strength (`BaseStrength + tmp3 * 10`, or `0.2` for the
/// oxide). Above it the shell deforms with the air, below it the cell is
/// sealed; a pixel whose whole 3x3 neighbourhood is free aluminium breaks into
/// powder (`ALMP`) or, when it is oxide, simply disappears.
///
/// The reference reads `pv`/`vx`/`vy` one cell outside the grid at the canvas
/// border (the arrays have no border). A device buffer cannot, so an
/// out-of-grid sample reads as a still, pressureless cell, and an out-of-grid
/// block stamp is dropped — the same clipping [`super::thermo`] applies.
///
/// The sixteen-binding budget is fully spent on the fields the update reads
/// (`part_life` included: the oxide walker pokes a blocked aluminium pixel's
/// conduction timer), so this pass has no `part_flags` binding and runs its
/// shells even inside a stasis cell.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn alum_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_temp: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    air_pv: &mut [f32],
    air_vx: &[f32],
    air_vy: &[f32],
    elem_block_air: &mut [u32],
    elem_block_airh: &mut [u32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] != PT_ALUM {
                terminate!();
            }
            let px = (idx as u32 % xres) as i32;
            let py = (idx as u32 / xres) as i32;
            let cx = (px as u32 / CELL) as i32;
            let cy = (py as u32 / CELL) as i32;
            let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 0xC2B2_AE35u32));

            // `ALUM.cpp:73-87`: orthogonally adjacent aluminium. The reference
            // comment says "including self", but `!rx != !ry` is false at
            // `(0, 0)`, so only the four edge neighbours count; reaching four
            // clamps to three (`alum = 3; break;`).
            let mut alum = 0i32;
            for n in 0u32..9u32 {
                let rx = (n % 3u32) as i32 - 1i32;
                let ry = (n / 3u32) as i32 - 1i32;
                let orthogonal = (rx == 0i32 && ry != 0i32) || (rx != 0i32 && ry == 0i32);
                if orthogonal {
                    let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                    if nv != 0u32 && part_ty[pmap_slot(nv)] == PT_ALUM {
                        alum += 1i32;
                    }
                    if alum == 4i32 {
                        alum = 3i32;
                    }
                }
            }

            // `ALUM.cpp:89-113`: the nine-cell pressure / velocity sample.
            let mut pressure_avg = 0.0f32;
            let mut velocity_avg = 0.0f32;
            let mut velocity_sum = 0.0f32;
            for n in 0u32..9u32 {
                let rx = (n % 3u32) as i32 - 1i32;
                let ry = (n / 3u32) as i32 - 1i32;
                let mut pressure = 0.0f32;
                let mut avx = 0.0f32;
                let mut avy = 0.0f32;
                let gx = cx + rx;
                let gy = cy + ry;
                if gx >= 0i32 && gy >= 0i32 && (gx as u32) < xcells && (gy as u32) < ycells {
                    let gi = ((gy as u32) * xcells + (gx as u32)) as usize;
                    pressure = air_pv[gi];
                    avx = air_vx[gi];
                    avy = air_vy[gi];
                }
                pressure_avg += pressure;
                let strength = (avx * avx + avy * avy).sqrt();
                velocity_avg += strength;
                if strength > 0.1f32 {
                    velocity_sum += (rx as f32 * avx + ry as f32 * avy) / strength;
                }
            }
            pressure_avg /= 9.0f32;
            // `ALUM.cpp:116-122`: the reference stores the nine pressures and
            // squares their deviations; nothing writes `pv` in between, so the
            // second read sees the same words.
            let mut pressure_stdev = 0.0f32;
            for n in 0u32..9u32 {
                let rx = (n % 3u32) as i32 - 1i32;
                let ry = (n / 3u32) as i32 - 1i32;
                let mut pressure = 0.0f32;
                let gx = cx + rx;
                let gy = cy + ry;
                if gx >= 0i32 && gy >= 0i32 && (gx as u32) < xcells && (gy as u32) < ycells {
                    pressure = air_pv[((gy as u32) * xcells + (gx as u32)) as usize];
                }
                let dev = pressure - pressure_avg;
                pressure_stdev += dev * dev;
            }
            pressure_stdev = pressure_stdev.sqrt();

            // `ALUM.cpp:126-132`.
            let strain = (pressure_stdev + velocity_sum + pressure_stdev * velocity_avg * 0.05f32)
                / (alum as f32 + 1.0f32);
            let mut strength = ALUM_BASE_STRENGTH + (part_tmp3[id] * 10i32) as f32;
            if part_ctype[id] == PT_O2 {
                strength = 0.2f32;
            }

            // `ALUM.cpp:134-143`: deform with the air, or seal the cell.
            let own = ((cy as u32) * xcells + (cx as u32)) as usize;
            if strain > strength {
                part_vx[id] += 0.4f32 * air_vx[own];
                part_vy[id] += 0.4f32 * air_vy[own];
            } else if alum >= 2i32 {
                elem_block_air[own] = 1u32;
                elem_block_airh[own] = 0x8u32;
            }

            // `ALUM.cpp:145-157`: a lone pixel under double the strain bursts.
            // `nt` counts the eight neighbours that are not aluminium, so
            // `nt == 8` means no orthogonal or diagonal aluminium contact.
            let mut nt = 0i32;
            for n in 0u32..9u32 {
                let rx = (n % 3u32) as i32 - 1i32;
                let ry = (n / 3u32) as i32 - 1i32;
                if rx != 0i32 || ry != 0i32 {
                    let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                    let mut nty = seed_u32(px as u32, PT_NONE);
                    if nv != 0u32 {
                        nty = part_ty[pmap_slot(nv)];
                    }
                    if nty != PT_ALUM {
                        nt += 1i32;
                    }
                }
            }
            let mut burst = false;
            if nt == 8i32
                && (pressure_stdev + velocity_avg) / (alum as f32 + 1.0f32) > strength * 2.0f32
            {
                burst = true;
                if part_ctype[id] == PT_O2 {
                    // `ALUM.cpp:148-150`: the oxide tower must not grow.
                    part_ty[id] = PT_NONE;
                } else {
                    // `ALUM.cpp:152-156`: `part_change_type` keeps the fields;
                    // the reference sets the powder's health and glow word.
                    part_ty[id] = PT_ALMP;
                    part_tmp[id] = ALMP_BURN_HEALTH;
                    part_tmp2[id] = rand_range(roll ^ 0x61u32, 0i32, 6i32);
                }
            }

            // `ALUM.cpp:160-257`: one random pixel of the 5x5 neighbourhood
            // drives the mercury absorption, the oxidation and the oxide
            // growth.
            if !burst {
                let rrx = px + rand_range(roll ^ 0x62u32, -2i32, 2i32);
                let rry = py + rand_range(roll ^ 0x63u32, -2i32, 2i32);
                let rv = pmap_at(pmap, xres, yres, rrx, rry);
                if rv != 0u32 {
                    let nid = pmap_slot(rv);
                    let nty = part_ty[nid];
                    // `ALUM.cpp:166-175`: absorb mercury until it is used up.
                    if part_ctype[id] != PT_O2 && nty == PT_MERC && part_tmp[id] < 1i32 {
                        part_ctype[id] = PT_MERC;
                        part_tmp2[id] += 1i32;
                        part_tmp[nid] -= 1i32;
                        if part_tmp[nid] < 1i32 {
                            part_ty[nid] = PT_NONE;
                        }
                    }
                    // `ALUM.cpp:177-180`.
                    if nty == PT_O2 && part_tmp[id] < 1i32 {
                        part_tmp[id] = 1i32;
                    }
                    // `ALUM.cpp:182-243`: the oxide "traveller". It walks up
                    // to thirty steps of one pixel each through the mercury
                    // soaked tower and seeds one oxide pixel. `create_part`
                    // reports no slot on the device, so the drain's own
                    // acceptance test — an empty, in-bounds pixel — decides
                    // between the success and the retry arm.
                    if part_ctype[id] == PT_O2
                        && nty == PT_ALUM
                        && part_ctype[nid] == PT_MERC
                        && chance(
                            roll ^ 0x64u32,
                            clampi(part_tmp2[nid], 0i32, 3000i32) as u32,
                            3000u32,
                        )
                    {
                        let mut tx = px as f32 + 0.5f32;
                        let mut tyf = py as f32 + 0.5f32;
                        let mut tdx = (px - rrx) as f32;
                        let mut tdy = (py - rry) as f32;
                        let mut done = false;
                        for j in 0u32..30u32 {
                            if !done {
                                // `ALUM.cpp:193-207`: five random probes steer
                                // the walker towards aluminium that is not yet
                                // oxide.
                                for k in 0u32..5u32 {
                                    let s = roll ^ mix_index(j * 5u32 + k, 0x2545_F491u32);
                                    let gx = (tx + rand_range(s, -5i32, 5i32) as f32) as i32;
                                    let gy = (tyf + rand_range(s ^ 0x9E37u32, -5i32, 5i32) as f32)
                                        as i32;
                                    let gv = pmap_at(pmap, xres, yres, gx, gy);
                                    let mut gty = seed_u32(gx as u32, PT_NONE);
                                    let mut gslot = 0usize;
                                    if gv != 0u32 {
                                        gslot = pmap_slot(gv);
                                        gty = part_ty[gslot];
                                    }
                                    if gty != PT_ALUM {
                                        tdx += (tx - gx as f32 - 0.5f32) * 0.02f32;
                                        tdy += (tyf - gy as f32 - 0.5f32) * 0.02f32;
                                    }
                                    if gty == PT_ALUM && part_ctype[gslot] != PT_O2 {
                                        tdx += (tx - gx as f32 - 0.5f32) * 0.1f32;
                                        tdy += (tyf - gy as f32 - 0.5f32) * 0.1f32;
                                    }
                                }
                                // `ALUM.cpp:209-217`.
                                let mut mag = (tdx * tdx + tdy * tdy).sqrt();
                                if mag == 0.0f32 {
                                    mag = 1.0f32;
                                }
                                tdx = tdx / mag;
                                tdy = tdy / mag;
                                tx += tdx;
                                tyf += tdy;
                                let npx = tx as i32;
                                let npy = tyf as i32;
                                let target = pmap_at(pmap, xres, yres, npx, npy);
                                let fits = npx >= 0i32
                                    && npy >= 0i32
                                    && (npx as u32) < xres
                                    && (npy as u32) < yres;
                                if fits && target == 0u32 {
                                    // `ALUM.cpp:218-225`: the new oxide pixel
                                    // takes one unit of the tower's mercury.
                                    request_create(
                                        create_queue,
                                        PT_ALUM,
                                        npx,
                                        npy,
                                        CR_SET_CTYPE | CR_SET_TMP2,
                                        0i32,
                                        PT_O2,
                                        0.0f32,
                                        0i32,
                                        1i32,
                                        0i32,
                                        0i32,
                                        0u32,
                                        0u32,
                                        0.0f32,
                                        0.0f32,
                                    );
                                    part_tmp2[nid] -= 1i32;
                                    done = true;
                                } else {
                                    // `ALUM.cpp:226-241`: a blocked step pokes
                                    // the aluminium it ran into; oxide lets the
                                    // walk continue.
                                    let mut stop = true;
                                    if target != 0u32 {
                                        let ppslot = pmap_slot(target);
                                        if part_ty[ppslot] == PT_ALUM {
                                            part_life[ppslot] = 10i32;
                                            if part_ctype[ppslot] == PT_O2 {
                                                stop = false;
                                            }
                                        }
                                    }
                                    if stop {
                                        done = true;
                                    }
                                }
                            }
                        }
                    }
                }
                // `ALUM.cpp:246-257`: a fully soaked pixel grows one more
                // oxide pixel in its own 3x3.
                if part_ctype[id] == PT_MERC
                    && chance(
                        roll ^ 0x65u32,
                        clampi(part_tmp2[id] - 10i32, 0i32, 3000i32) as u32,
                        3000u32,
                    )
                {
                    let gx = px + rand_range(roll ^ 0x66u32, -1i32, 1i32);
                    let gy = py + rand_range(roll ^ 0x67u32, -1i32, 1i32);
                    let fits = gx >= 0i32 && gy >= 0i32 && (gx as u32) < xres && (gy as u32) < yres;
                    if fits && pmap_at(pmap, xres, yres, gx, gy) == 0u32 {
                        request_create(
                            create_queue,
                            PT_ALUM,
                            gx,
                            gy,
                            CR_SET_CTYPE | CR_SET_TMP2,
                            0i32,
                            PT_O2,
                            0.0f32,
                            0i32,
                            10i32,
                            0i32,
                            0i32,
                            0u32,
                            0u32,
                            0.0f32,
                            0.0f32,
                        );
                        part_tmp2[id] -= 10i32;
                    }
                }
                // `ALUM.cpp:260-266`: the alloy's melting point rises by
                // 200 K per alloy level; the oxide never melts this way.
                if part_temp[id] > 660.32f32 + 273.15f32 + (part_tmp3[id] as f32) * 200.0f32
                    && part_ctype[id] != PT_O2
                {
                    part_ty[id] = PT_LAVA;
                    part_ctype[id] = PT_ALUM;
                    part_tmp[id] = 0i32;
                }
            }
        }
    }
}

/// Launch the three second-set solid passes.
///
/// The order matters: `ALMP` (in the burn pass) runs before `ALUM`, so the
/// powder `ALUM.cpp:153` leaves in its slot is not updated again this frame.
pub(crate) fn launch(bufs: &SimBuffers, dims: &PassDims, _seed: u32) {
    if dims.pixels == 0 || dims.xcells == 0 {
        return;
    }
    let xcells = dims.xcells as u32;
    let ycells = (dims.cells / dims.xcells as usize) as u32;
    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        chem_solid_update_kernel::launch_unchecked(
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
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            xcells,
            ycells,
            dims.frame_seed,
        );
        chem_solid_burn_kernel::launch_unchecked(
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
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_elem_block_air.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_block_airh.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            xcells,
            ycells,
            dims.frame_seed,
        );
        alum_kernel::launch_unchecked(
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
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_block_air.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_block_airh.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            xcells,
            ycells,
            dims.frame_seed,
        );
    }
}
