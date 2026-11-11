//! Force and explosive second-set elements: `MGNT`, `MISL`, `PROJ`, `PRMT`,
//! `CEXP`, `BFLM`.
//!
//! Ports the electromagnet's field (`MGNT.cpp`), missile guidance (`MISL.cpp`,
//! target in `tmp`/`tmp2`), the projectile's `ctype` conversion (`PROJ.cpp`),
//! promethium's velocity-triggered fire and neutron emission (`PRMT.cpp`), the
//! custom explosive's `temp`/`life` blast (`CEXP.cpp`) and black flames'
//! propagation (`BFLM.cpp`).
//!
//! The launches run before [`super::super::force::launch`], so a force change
//! made here is seen by the gather pass in the same tick, exactly like the
//! `special` force updates that precede it.
//!
//! # The two connected floods
//!
//! `MGNT.cpp:96`/`106` and `CEXP.cpp:73` call `Simulation::flood_prop`
//! (`Simulation.cpp:520-570`), which rewrites one field of **every** particle
//! of the connected run in a single call. Both are reproduced by
//! `reaction_kernel::flood_component`: a slot-parallel seed scan queues the
//! pixels that carry the trigger value and one device thread drains the queue,
//! expanding the 4-connected run without a per-frame round budget.
//!
//! `MGNT` writes a sentinel instead of `3` because its `tmp`/`tmp2` are also
//! the attract/repel switch (`> 0`): the consuming pass
//! ([`mgnt_attract_kernel`]) turns the sentinel into the reference's `3` before
//! any force is applied. `CEXP` needs no separate seed: `CEXP.cpp:70-74` arms a
//! charge that touches an armed one, so every armed pixel (`10`, or the
//! in-flight sentinel) seeds the scan and the drain arms the whole run.
//!
//! # Field preservation on conversion
//!
//! Every conversion in this group except `PRMT`'s is
//! `Simulation::part_change_type` (`Simulation.cpp:1801-1845`): it rewrites
//! only the element id — and the particle's map layer, which the per-frame
//! `pmap`/`photons` rebuild reproduces — and keeps `temp`, `vx`, `vy`, `life`,
//! `tmp`, `tmp2`, `flags` and `dcolour`. Those kernels therefore assign
//! `part_ty` directly and write only the fields the reference writes on top.
//! That is what carries `PROJ`'s momentum and `CEXP`'s 9720 K blast
//! temperature into the new element. The `ChangeType` hooks are not run: only
//! `ALUM` has one and only for `from == LAVA`.
//!
//! `PRMT.cpp:62-89` instead uses `create_part(i, ...)`, whose whole-row reset
//! — defaults plus the target's `Create` hook — is exactly what a
//! [`request_replace`] record reproduces, so those three conversions are
//! queued.

// Same rationale as `pixel_mode.rs`: collapsing a nested `if` changes the
// CubeCL expansion, and the device IR has no `RangeInclusive::contains` or
// `f32::clamp`.
#![allow(
    clippy::collapsible_if,
    clippy::manual_range_contains,
    clippy::manual_clamp
)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::PassDims;
use super::super::combustion::chance;
use super::super::shared::{
    CREATE_QUEUE_WORDS, hash_u32, insulated, launch_grid, mix_index, pmap_at, pmap_slot,
    request_create, request_replace, seed_f32, seed_u32, stasis_frozen,
};
use crate::buffers::SimBuffers;
use crate::gravity_kernel::set_grav_mass_override;
use crate::reaction_kernel::flood_component::{
    FLOOD_FIELD_TMP, FLOOD_FIELD_TMP2, FLOOD_MODE_TMP, FLOOD_MODE_TMP2, launch_drain,
    launch_seed_scan,
};

// The element ids this module's comments cite from the C++ sources.
const _: () = assert!(
    PT_MGNT == 238
        && PT_MISL == 237
        && PT_PROJ == 227
        && PT_PRMT == 212
        && PT_CEXP == 222
        && PT_BFLM == 229
);

/// `MGNT.cpp:50`: one frame in five decays the magnet's fields.
const MGNT_DECAY_DEN: u32 = 5;
/// `MGNT.cpp:67`: the radius of the attract/repel field is capped at 40.
const MGNT_LIFE_MAX: i32 = 40;
/// `MGNT.cpp:73`: both fields charged heats the magnet by 15.15 K per frame.
const MGNT_FIELD_HEAT: f32 = 15.15f32;
/// `MGNT.cpp:91`/`101`: a spark heats the magnet by 3 K.
const MGNT_SPARK_HEAT: f32 = 3.0f32;
/// `MGNT.cpp:91`/`101`: ... until it reaches 673.15 K.
const MGNT_SPARK_HEAT_CAP: f32 = 673.15f32;
/// `MGNT.cpp:96`/`106`: `flood_prop` writes this into the whole connected run.
const MGNT_FIELD_ON: i32 = 3;
/// Sentinel that stands in for [`MGNT_FIELD_ON`] until the flood is over.
const MGNT_FLOOD: i32 = -2;
/// `MGNT.cpp:171`/`:172`: a sparked metal heats by 35.15 K and sparks for 4.
const MGNT_METAL_HEAT: f32 = 35.15f32;
/// See [`MGNT_METAL_HEAT`].
const MGNT_SPARK_LIFE: i32 = 4;

/// `MISL.cpp:52-57`: the default/clamp values of the target coordinates.
const MISL_MAX_TMP: i32 = 610;
/// See [`MISL_MAX_TMP`].
const MISL_MAX_TMP2: i32 = 380;
/// `MISL.cpp:52-57`: the clamped target defaults.
const MISL_DEFAULT_TMP: i32 = 100;
/// `MISL.cpp:66`: arrival window around the target.
const MISL_ARRIVE_X: f32 = 3.0f32;
/// See [`MISL_ARRIVE_X`].
const MISL_ARRIVE_Y: f32 = 2.0f32;
/// `MISL.cpp:66`: the flight timer that forces the detonation.
const MISL_MAX_TMP4: i32 = 300;
/// `MISL.cpp:66`: a hot missile detonates as well.
const MISL_DETONATE_TEMP: f32 = 873.15f32;
/// `MISL.cpp:68`: the pressure pulse of the `SING` conversion.
const MISL_BLAST_PRESSURE: f32 = 270.0f32;
/// `MISL.cpp:70`: the `tmp` the `SING` starts with (its explosion radius).
const MISL_SING_TMP: i32 = 400;
/// `MISL.cpp:77-83`/`103-109`: the seek window and the two speeds.
const MISL_SEEK_WINDOW: f32 = 6.0f32;
/// See [`MISL_SEEK_WINDOW`].
const MISL_VEL_FAR: f32 = 3.0f32;
/// See [`MISL_SEEK_WINDOW`].
const MISL_VEL_NEAR: f32 = 1.0f32;

/// `PROJ.cpp:57-60`: the clamped `tmp` / temperature window.
const PROJ_MAX_TMP: i32 = 100;
/// See [`PROJ_MAX_TMP`].
const PROJ_DEFAULT_TMP: i32 = 10;
/// `PROJ.cpp:59`: outside this window the temperature resets.
const PROJ_MIN_TEMP: f32 = 273.15f32;
/// See [`PROJ_MIN_TEMP`].
const PROJ_MAX_TEMP: f32 = 373.15f32;
/// See [`PROJ_MIN_TEMP`].
const PROJ_DEFAULT_TEMP: f32 = 293.15f32;
/// `PROJ.cpp:93`: `vx = tmp4 * ((temp - 273.15) / 10)`.
const PROJ_SPEED_DIV: f32 = 10.0f32;

/// `PRMT.cpp:60`: a promethium this fast catches fire.
const PRMT_IGNITE_VEL: f32 = 12.0f32;
/// `PRMT.cpp:64`: below this it collapses into a `SING`.
const PRMT_COLD_TEMP: f32 = 274.15f32;
/// `PRMT.cpp:70`: above this it emits neutrons one frame in two hundred.
const PRMT_HOT_TEMP: f32 = 874.15f32;
/// `PRMT.cpp:72`: the neutron emission chance.
const PRMT_NEUT_DEN: u32 = 200;
/// `PRMT.cpp:90`: a `PLUT` neighbour is sucked into a neutron.
const PRMT_PLUT_PRESSURE: f32 = -2.0f32;

/// `CEXP.cpp:70`: an armed charge arms its connected neighbours.
const CEXP_ARMED: i32 = 10;
/// Sentinel that stands in for [`CEXP_ARMED`] while the flood is running.
const CEXP_FLOOD: i32 = -2;

/// `BFLM.cpp:70`: an adjacent particle refreshes the flame's fuel.
const BFLM_FUEL: i32 = 5;
/// `BFLM.cpp:71`: ... and catches fire one frame in thirteen.
const BFLM_SPREAD_DEN: u32 = 13;
/// `BFLM.cpp:73`: the life a converted neighbour starts with.
const BFLM_CHILD_LIFE: i32 = 55;

/// `PT_NUM` as a `u32` for the kernel-side element-id guard.
///
/// The host constant is a `usize`; comparing it with the `u32` element id
/// inside a `#[cube]` body would not type-check.
const PT_NUM_U32: u32 = PT_NUM as u32;
const _: () = assert!(PT_NUM_U32 as usize == PT_NUM);

/// `abs(v)` for floats (CubeCL has no `f32::abs` in the portable subset).
#[cube]
fn fabsf(v: f32) -> f32 {
    let mut r = v;
    if v < 0.0f32 {
        r = -v;
    }
    r
}

/// `MGNT.cpp:50-110`: the electromagnet's own state and spark pickup.
///
/// Slot-owned, pixel parallel. The 1/5 decay, the `life` clamp and the
/// "both fields charged" heating run first, then the 4x4 spark scan
/// (`MGNT.cpp:76-110`). Two things from that scan cannot finish here:
///
/// * a `PSCN`/`NSCN` spark asks for `flood_prop` over the whole connected
///   magnet; this kernel stores [`MGNT_FLOOD`] on the sparked pixel and the
///   `flood_component` seed scan + drain spreads it through the run, with
///   [`mgnt_attract_kernel`] turning the sentinel into `3`;
/// * `parts[i].tmp3 = 1` is a sticky flag in the reference (nothing clears it)
///   and gates the attract loop, so it is written here and never cleared.
///
/// `nt` (`ElementDefs.h:47`, `Simulation.cpp:2284-2293`) is the count of the
/// eight ring neighbours whose type differs from `MGNT`; empty pixels count,
/// so `nt > 0` fails only inside a solid magnet block.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn mgnt_update_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_ctype: &[u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_flags: &mut [u32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_MGNT {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 0x9E37_79B9u32));
                let mut life = part_life[id];
                let mut temp = part_temp[id];
                let mut tmp = part_tmp[id];
                let mut tmp2 = part_tmp2[id];

                // `MGNT.cpp:50-66`: one frame in five, both fields and the
                // field radius leak one unit each (the two `life += 2` refunds
                // are what keeps a charged magnet's radius stable).
                if chance(roll, 1u32, MGNT_DECAY_DEN) {
                    if tmp > 0i32 {
                        tmp -= 1i32;
                        life += 2i32;
                    }
                    if tmp2 > 0i32 {
                        tmp2 -= 1i32;
                        life += 2i32;
                    }
                    if life > 0i32 {
                        life -= 1i32;
                    }
                }
                // `MGNT.cpp:67-70`: the radius is clamped, not wrapped.
                if life > MGNT_LIFE_MAX || life < 0i32 {
                    life = MGNT_LIFE_MAX;
                }
                // `MGNT.cpp:71-74`: a fully charged magnet runs hot.
                if tmp > 0i32 && tmp2 > 0i32 {
                    temp += MGNT_FIELD_HEAT;
                }

                // `Simulation.cpp:2284-2293`: `nt` counts the eight ring
                // neighbours of a different type (`TYP == 0` included).
                let mut nt = 0u32;
                for oy in 0u32..3u32 {
                    for ox in 0u32..3u32 {
                        if ox != 1u32 || oy != 1u32 {
                            let nv = pmap_at(
                                pmap,
                                xres,
                                yres,
                                px + ox as i32 - 1i32,
                                py + oy as i32 - 1i32,
                            );
                            let mut nty = seed_u32(px as u32, PT_NONE);
                            if nv != 0u32 {
                                nty = part_ty[pmap_slot(nv)];
                            }
                            if nty != PT_MGNT {
                                nt += 1u32;
                            }
                        }
                    }
                }

                // `MGNT.cpp:76-110`: the 4x4 spark scan. `tmp3` is set by any
                // non-empty sample, the two counters drive the heating and the
                // flood seeds.
                let mut any_neighbour = false;
                let mut pscn = 0u32;
                let mut nscn = 0u32;
                for oy in 0u32..4u32 {
                    for ox in 0u32..4u32 {
                        let dx = ox as i32 - 1i32;
                        let dy = oy as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                any_neighbour = true;
                                let nid = pmap_slot(nv);
                                if part_ty[nid] == PT_SPRK {
                                    let ct = part_ctype[nid];
                                    if ct == PT_PSCN {
                                        pscn += 1u32;
                                    } else if ct == PT_NSCN {
                                        nscn += 1u32;
                                    }
                                }
                            }
                        }
                    }
                }
                // `MGNT.cpp:91-94` / `101-104`: +3 K per sparked sensor, up to
                // the 673.15 K cap, then the flood request.
                for _k in 0u32..pscn {
                    if temp < MGNT_SPARK_HEAT_CAP {
                        temp += MGNT_SPARK_HEAT;
                    }
                }
                for _k in 0u32..nscn {
                    if temp < MGNT_SPARK_HEAT_CAP {
                        temp += MGNT_SPARK_HEAT;
                    }
                }
                // `MGNT.cpp:96` `FIELD_TMP` / `:106` `FIELD_TMP2`.
                if pscn > 0u32 {
                    tmp = MGNT_FLOOD;
                }
                if nscn > 0u32 {
                    tmp2 = MGNT_FLOOD;
                }
                // `MGNT.cpp:83-86`: sticky, never cleared by the reference.
                if nt > 0u32 && any_neighbour {
                    part_tmp3[id] = 1i32;
                }

                part_life[id] = life;
                part_temp[id] = temp;
                part_tmp[id] = tmp;
                part_tmp2[id] = tmp2;
            }
        }
    }
}

/// `MGNT.cpp:111-181`: the attract/repel field and the metal sparking.
///
/// Runs after the flood rounds. The sentinel becomes `3` first, so every
/// charged pixel of the run is in the reference's final state before any
/// force is applied. The scan radius is `life` (clamped to `[0, 40]`), the
/// targets are read from `pmap` and then `photons` (`MGNT.cpp:117-121`), and
/// the velocity writes are the reference's absolute `+-0.5` (attract) and
/// `+-1` (repel) assignments. All ten target types of `MGNT.cpp:124-133` are
/// covered, including the energy particles that only live in the photon layer.
///
/// The last case (`MGNT.cpp:161-178`) sparks a metal: the reference sets
/// `temp += 35.15`, `life = 4`, `ctype = <metal>` and converts the neighbour
/// with `part_change_type`, which keeps the rest of the row, so this is a
/// direct field write rather than a queued replacement. The electronics pass
/// has already run this tick, so the new `SPRK` starts conducting on the next
/// one — the same one-slot-order ambiguity the reference has.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn mgnt_attract_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_ctype: &mut [u32],
    part_flags: &mut [u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_MGNT {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                // The flood is over; every marked pixel finishes at `3`.
                let mut tmp = part_tmp[id];
                if tmp == MGNT_FLOOD {
                    tmp = MGNT_FIELD_ON;
                    part_tmp[id] = MGNT_FIELD_ON;
                }
                let mut tmp2 = part_tmp2[id];
                if tmp2 == MGNT_FLOOD {
                    tmp2 = MGNT_FIELD_ON;
                    part_tmp2[id] = MGNT_FIELD_ON;
                }
                if part_tmp3[id] == 1i32 {
                    let px = (idx as u32 % xres) as i32;
                    let py = (idx as u32 / xres) as i32;
                    let self_x = part_x[id];
                    let self_y = part_y[id];
                    // `MGNT.cpp:75`: `checkrad = life`, already clamped by
                    // the state pass above; the local clamp keeps the span
                    // valid even if this kernel runs on its own.
                    let mut checkrad = part_life[id];
                    if checkrad < 0i32 {
                        checkrad = 0i32;
                    }
                    if checkrad > MGNT_LIFE_MAX {
                        checkrad = MGNT_LIFE_MAX;
                    }
                    let span = (2i32 * checkrad + 1i32) as u32;
                    for oy in 0u32..span {
                        let dy = oy as i32 - checkrad;
                        for ox in 0u32..span {
                            let dx = ox as i32 - checkrad;
                            if dx != 0i32 || dy != 0i32 {
                                // `MGNT.cpp:115`: bounded explicitly here.
                                let mut nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv == 0u32 {
                                    nv = pmap_at(photons, xres, yres, px + dx, py + dy);
                                }
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    let nt = part_ty[nid];
                                    let attracted = nt == PT_BRMT
                                        || nt == PT_BREC
                                        || nt == PT_SLCN
                                        || nt == PT_PQRT
                                        || nt == PT_COPR
                                        || nt == PT_ELEC
                                        || nt == PT_NEUT
                                        || nt == PT_GRVT
                                        || nt == PT_PHOT
                                        || nt == PT_PLSM;
                                    if attracted {
                                        // `MGNT.cpp:135-146`: attract at 0.5.
                                        if tmp > 0i32 {
                                            if part_y[nid] > self_y {
                                                part_vy[nid] = -0.5f32;
                                            } else if part_y[nid] < self_y {
                                                part_vy[nid] = 0.5f32;
                                            }
                                            if part_x[nid] > self_x {
                                                part_vx[nid] = -0.5f32;
                                            } else if part_x[nid] < self_x {
                                                part_vx[nid] = 0.5f32;
                                            }
                                        }
                                        // `MGNT.cpp:147-158`: repel at 1.
                                        if tmp2 > 0i32 {
                                            if part_y[nid] > self_y {
                                                part_vy[nid] = 1.0f32;
                                            } else if part_y[nid] < self_y {
                                                part_vy[nid] = -1.0f32;
                                            }
                                            if part_x[nid] > self_x {
                                                part_vx[nid] = 0.5f32;
                                            } else if part_x[nid] < self_x {
                                                part_vx[nid] = -0.5f32;
                                            }
                                        }
                                    } else if nt == PT_METL
                                        || nt == PT_BMTL
                                        || nt == PT_IRON
                                        || nt == PT_TUNG
                                        || nt == PT_GOLD
                                    {
                                        // `MGNT.cpp:167-176`.
                                        if tmp > 0i32 && tmp2 > 0i32 && part_life[nid] == 0i32 {
                                            part_temp[nid] += MGNT_METAL_HEAT;
                                            part_life[nid] = MGNT_SPARK_LIFE;
                                            part_ctype[nid] = nt;
                                            part_ty[nid] = PT_SPRK;
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
}

/// `MISL.cpp:49-146`: missile guidance, the `SING` conversion and the trail.
///
/// The target (`tmp`, `tmp2`) is clamped to the canvas, `tmp4` is the flight
/// timer, and `life` selects the phase: `20` seeks vertically and drops `BRAY`
/// above/below itself, `10` seeks horizontally and drops it left/right, and
/// any other non-zero life only waits. The detonation replaces the missile
/// with a `SING` (`part_change_type`, so the row is kept) and stamps 270
/// pressure units into the missile's own air cell.
///
/// The 5x5 scan picks up a `PSCN` spark at `life == 3` (restarting the seek)
/// and counts `tmp4` up once per adjacent particle while the missile is live;
/// the insulator test is `parts_avg(ID(r), i, PT_INSL)` (`MISL.cpp:128`).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn misl_update_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &[f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp4: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_x: &[f32],
    part_y: &[f32],
    part_ctype: &[u32],
    part_flags: &mut [u32],
    air_pv: &mut [f32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_MISL {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let x = part_x[id];
                let y = part_y[id];
                let mut tmp = part_tmp[id];
                let mut tmp2 = part_tmp2[id];
                let mut tmp4 = part_tmp4[id];
                let mut life = part_life[id];

                // `MISL.cpp:52-57`.
                if tmp <= 0i32 || tmp > MISL_MAX_TMP {
                    tmp = MISL_DEFAULT_TMP;
                }
                if tmp2 <= 0i32 || tmp2 > MISL_MAX_TMP2 {
                    tmp2 = MISL_DEFAULT_TMP;
                }
                if tmp4 < 1i32 {
                    tmp4 = 1i32;
                }
                // `MISL.cpp:58-62`: a missile that has not been launched
                // cannot be displaced by the air.
                if life == 0i32 {
                    part_vx[id] = 0.0f32;
                    part_vy[id] = 0.0f32;
                }
                // `MISL.cpp:64-73`.
                if life > 0i32 {
                    let arrived = fabsf(x - tmp as f32) <= MISL_ARRIVE_X
                        && fabsf(y - tmp2 as f32) <= MISL_ARRIVE_Y;
                    if arrived || tmp4 > MISL_MAX_TMP4 || part_temp[id] >= MISL_DETONATE_TEMP {
                        air_pv[cidx] = MISL_BLAST_PRESSURE;
                        life = 1i32;
                        tmp = MISL_SING_TMP;
                        // `part_change_type(i, x, y, PT_SING)` keeps the row.
                        part_ty[id] = PT_SING;
                    }
                }

                // `MISL.cpp:74-121`: the vertical (life 20) and horizontal
                // (life 10) seek phases with their trail particles.
                if life == 20i32 {
                    let mut vel = seed_f32(y, MISL_VEL_NEAR);
                    if fabsf(y - tmp2 as f32) > MISL_SEEK_WINDOW {
                        vel = MISL_VEL_FAR;
                    }
                    if y > tmp2 as f32 {
                        part_vy[id] = -vel;
                        // `MISL.cpp:88`: `create_part(-1, x, y + 1, PT_BRAY)`.
                        request_create(
                            create_queue,
                            PT_BRAY,
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
                    } else if y < tmp2 as f32 {
                        part_vy[id] = vel;
                        // `MISL.cpp:93`: `create_part(-1, x, y - 1, PT_BRAY)`.
                        request_create(
                            create_queue,
                            PT_BRAY,
                            px,
                            py - 1i32,
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
                    if y == tmp2 as f32 {
                        life = 10i32;
                    }
                } else if life == 10i32 {
                    let mut vel = seed_f32(x, MISL_VEL_NEAR);
                    if fabsf(x - tmp as f32) > MISL_SEEK_WINDOW {
                        vel = MISL_VEL_FAR;
                    }
                    if x < tmp as f32 {
                        part_vx[id] = vel;
                        // `MISL.cpp:114`: `create_part(-1, x - 1, y, PT_BRAY)`.
                        request_create(
                            create_queue,
                            PT_BRAY,
                            px - 1i32,
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
                    } else if x > tmp as f32 {
                        part_vx[id] = -vel;
                        // `MISL.cpp:119`: `create_part(-1, x + 1, y, PT_BRAY)`.
                        request_create(
                            create_queue,
                            PT_BRAY,
                            px + 1i32,
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
                    }
                }

                // `MISL.cpp:123-144`: the 5x5 trigger and flight counter.
                for oy in 0u32..5u32 {
                    let dy = oy as i32 - 2i32;
                    for ox in 0u32..5u32 {
                        let dx = ox as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                if !insulated(pmap, xres, yres, px, py, px + dx, py + dy) {
                                    let nid = pmap_slot(nv);
                                    let nt = part_ty[nid];
                                    // `MISL.cpp:130-133`.
                                    if nt == PT_SPRK
                                        && part_ctype[nid] == PT_PSCN
                                        && part_life[nid] == 3i32
                                    {
                                        life = 20i32;
                                    }
                                    // `MISL.cpp:135-138`.
                                    if nt == PT_PSCN && life == 0i32 {
                                        tmp4 = 0i32;
                                    }
                                    // `MISL.cpp:139-143`: `TYP(r)` is non-zero
                                    // here, so every neighbour counts.
                                    if life > 0i32 && nt != PT_NONE {
                                        tmp4 += 1i32;
                                    }
                                }
                            }
                        }
                    }
                }

                part_life[id] = life;
                part_tmp[id] = tmp;
                part_tmp2[id] = tmp2;
                part_tmp4[id] = tmp4;
            }
        }
    }
}

/// `PROJ.cpp:54-97`: the projectile's arming, collision and flight.
///
/// A `PSCN` spark at `life == 3` in the 3x3 neighbourhood arms the projectile
/// and stores the launch direction in `tmp4`/`tmp3` (`PROJ.cpp:69-74`); the 8x8
/// scan then converts the projectile into its `ctype` on any non-`EMBR`,
/// non-`SPRK`, non-`PSCN` particle (`PROJ.cpp:83-88`).
///
/// The conversion is `Simulation::part_change_type`
/// (`Simulation.cpp:1801-1845`): only `type` (and the particle's map layer,
/// which the per-frame `pmap`/`photons` rebuild reproduces) changes, so `temp`,
/// `tmp`, `tmp2`, `tmp3`, `tmp4` and the velocity survive into the new element
/// and the flight writes below still land on the converted row. The reference
/// calls it once per colliding neighbour with the same `ctype` each time, so
/// the scan degenerates to one assignment; the `ChangeType` hooks are not run
/// (only `ALUM` has one and only for `from == LAVA`).
///
/// An armed projectile flies with `vx = tmp4 * (temp - 273.15) / 10` and
/// `vy = tmp3 + 0.2 * (tmp2 / tmp)` (`PROJ.cpp:92-95`).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn proj_update_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_ctype: &[u32],
    part_flags: &mut [u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_PROJ {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let mut tmp = part_tmp[id];
                let mut temp = part_temp[id];
                let mut life = part_life[id];
                let mut tmp3 = part_tmp3[id];
                let mut tmp4 = part_tmp4[id];

                // `PROJ.cpp:57-60`.
                if tmp <= 0i32 || tmp > PROJ_MAX_TMP {
                    tmp = PROJ_DEFAULT_TMP;
                }
                if temp <= PROJ_MIN_TEMP || temp > PROJ_MAX_TEMP {
                    temp = PROJ_DEFAULT_TEMP;
                }

                // `PROJ.cpp:62-75`: `for rx` is the outer loop, so the last
                // matching spark wins the direction.
                for ox in 0u32..3u32 {
                    let dx = ox as i32 - 1i32;
                    for oy in 0u32..3u32 {
                        let dy = oy as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                if part_ty[nid] == PT_SPRK
                                    && part_ctype[nid] == PT_PSCN
                                    && part_life[nid] == 3i32
                                {
                                    tmp4 = -dx;
                                    tmp3 = -dy;
                                    life = 10i32;
                                }
                            }
                        }
                    }
                }

                // `PROJ.cpp:76-88`: the 8x8 collision scan. The reference
                // calls `part_change_type` for every matching neighbour, but
                // they all name the same `ctype`, so one assignment is enough.
                let ctype = part_ctype[id];
                let mut hit = false;
                if life == 10i32 && ctype != PT_NONE && ctype < PT_NUM_U32 {
                    for ox in 0u32..8u32 {
                        let dx = ox as i32 - 3i32;
                        for oy in 0u32..8u32 {
                            let dy = oy as i32 - 3i32;
                            if (dx != 0i32 || dy != 0i32) && !hit {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nt = part_ty[pmap_slot(nv)];
                                    if nt != PT_NONE
                                        && nt != PT_EMBR
                                        && nt != PT_SPRK
                                        && nt != PT_PSCN
                                    {
                                        hit = true;
                                    }
                                }
                            }
                        }
                    }
                }
                // `PROJ.cpp:86`: `part_change_type(i, x, y, ctype)` inside the
                // scan, before the flight block below. The last colliding
                // neighbour wins, exactly like the sequential reference calling
                // it once per match; `part_change_type` rejects an id outside
                // `1..PT_NUM`, which the guard reproduces. The energy-layer
                // move (`pmap` -> `photons`) is reproduced by the per-frame map
                // rebuild, so no layer write is needed here.
                if hit && ctype != PT_NONE && ctype < PT_NUM_U32 {
                    part_ty[id] = ctype;
                }

                // `PROJ.cpp:90-96`.
                if life == 10i32 {
                    let tmp2 = part_tmp2[id] + 1i32;
                    part_tmp2[id] = tmp2;
                    part_vx[id] = tmp4 as f32 * ((temp - PROJ_MIN_TEMP) / PROJ_SPEED_DIV);
                    part_vy[id] = tmp3 as f32 + 0.2f32 * ((tmp2 / tmp) as f32);
                }

                part_life[id] = life;
                part_temp[id] = temp;
                part_tmp[id] = tmp;
                part_tmp3[id] = tmp3;
                part_tmp4[id] = tmp4;
            }
        }
    }
}

/// `PRMT.cpp:58-96`: promethium's three fates plus the `PLUT` reaction.
///
/// The order is the reference's: a fast promethium becomes `FIRE` (a whole-row
/// `create_part(i, ...)`, so the record carries no overrides), a cold one
/// collapses into `SING` with `tmp = 20` (a `part_change_type`, so the row is
/// written directly), a hot one emits a `NEUT` one frame in
/// [`PRMT_NEUT_DEN`], and every `PLUT` neighbour both emits a neutron and
/// pulls two pressure units out of its own air cell.
///
/// The reference keeps running the rest of the update after a conversion, on
/// the row the new element now owns: after `create_part(i, PT_FIRE)` the two
/// temperature branches read `FIRE`'s 695.15 K default and skip, which is what
/// the `fired` guard reproduces. The `PLUT` scan still runs, and its `NEUT`
/// record is queued after the `FIRE` one — the drain replays the records in
/// order, so the neutron wins exactly like two `create_part` calls on one
/// slot.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn prmt_update_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_vx: &[f32],
    part_vy: &[f32],
    part_flags: &mut [u32],
    air_pv: &mut [f32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    xcells: u32,
    cells: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_PRMT {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 0x51ED_2701u32));
                let temp = part_temp[id];

                // `PRMT.cpp:60-63`: `create_part(i, x, y, PT_FIRE)`. The
                // conversion resets the row, so the two temperature branches
                // below read `FIRE`'s default 695.15 K and skip; `fired`
                // reproduces that without binding the element table.
                let mut fired = false;
                if part_vx[id] >= PRMT_IGNITE_VEL || part_vy[id] >= PRMT_IGNITE_VEL {
                    fired = true;
                    request_replace(
                        create_queue,
                        id as u32,
                        PT_FIRE,
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
                }
                // `PRMT.cpp:64-68`: a cold promethium collapses. The reference
                // keeps the row (`part_change_type`) and sets `tmp` first, so
                // both writes are direct.
                if !fired && temp <= PRMT_COLD_TEMP {
                    part_tmp[id] = 20i32;
                    part_ty[id] = PT_SING;
                }
                // `PRMT.cpp:70-76`: one neutron in two hundred.
                if !fired && temp >= PRMT_HOT_TEMP && chance(roll, 1u32, PRMT_NEUT_DEN) {
                    request_replace(
                        create_queue,
                        id as u32,
                        PT_NEUT,
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
                }

                // `PRMT.cpp:78-94`: every `PLUT` in the 5x5.
                for oy in 0u32..5u32 {
                    let dy = oy as i32 - 2i32;
                    for ox in 0u32..5u32 {
                        let dx = ox as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                if part_ty[pmap_slot(nv)] == PT_PLUT {
                                    // `PRMT.cpp:89`: `create_part(i, x, y,
                                    // PT_NEUT)`.
                                    request_replace(
                                        create_queue,
                                        id as u32,
                                        PT_NEUT,
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
                                    // `PRMT.cpp:90`: the neighbour's cell loses
                                    // two pressure units.
                                    let ncx = (px as u32 / CELL) as i32 + dx;
                                    let ncy = (py as u32 / CELL) as i32 + dy;
                                    if ncx >= 0i32 && ncy >= 0i32 {
                                        let ncell = (ncy as u32 * xcells + ncx as u32) as usize;
                                        if ncell < cells as usize {
                                            air_pv[ncell] = PRMT_PLUT_PRESSURE;
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
}

/// `CEXP.cpp:49-88`: the custom explosive's arming.
///
/// Phase 1: fix a self-referential `ctype`, then the 5x5 scan. A spark, fire,
/// plasma, thunder or lightning neighbour arms the charge (`tmp2 = 10`), and so
/// does a connected *armed* charge; the seed scan + drain in `launch` then arm
/// the rest of that 4-connected run.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn cexp_trigger_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_ctype: &mut [u32],
    part_tmp2: &mut [i32],
    part_flags: &mut [u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_CEXP {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                // `CEXP.cpp:49-52`: a charge cannot explode into itself.
                if part_ctype[id] == PT_CEXP {
                    part_ctype[id] = PT_PLSM;
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let mut tmp2 = part_tmp2[id];
                for oy in 0u32..5u32 {
                    let dy = oy as i32 - 2i32;
                    for ox in 0u32..5u32 {
                        let dx = ox as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let mut nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv == 0u32 {
                                nv = pmap_at(photons, xres, yres, px + dx, py + dy);
                            }
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nt = part_ty[nid];
                                // `CEXP.cpp:65-68`.
                                if nt == PT_SPRK
                                    || nt == PT_FIRE
                                    || nt == PT_PLSM
                                    || nt == PT_THDR
                                    || nt == PT_LIGH
                                {
                                    tmp2 = CEXP_ARMED;
                                }
                                // `CEXP.cpp:70-74`: an armed neighbour arms
                                // this charge and floods the connected run.
                                if nt == PT_CEXP {
                                    let n2 = part_tmp2[nid];
                                    if n2 > 0i32 || n2 == CEXP_FLOOD {
                                        tmp2 = CEXP_ARMED;
                                    }
                                }
                            }
                        }
                    }
                }
                part_tmp2[id] = tmp2;
            }
        }
    }
}

/// `CEXP.cpp:76-83`: the blast, once a charge is armed.
///
/// Every non-empty 5x5 neighbour gets the charge's `life` as air pressure and
/// its `temp` as ambient heat in *that neighbour's* air cell, and takes the
/// charge's temperature; the charge's own cell gets the mass override
/// (`gravIn.mass[cell] = tmp`, resolved by [`set_grav_mass_override`]); finally
/// the charge becomes its `ctype`.
///
/// The conversion is `Simulation::part_change_type` (`Simulation.cpp:1801-1845`),
/// not `create_part`: it rewrites only `type` (and the particle's map layer,
/// which the per-frame `pmap`/`photons` rebuild reproduces) and keeps `temp`,
/// `vx`, `vy`, `life`, `tmp`, `tmp2`, `flags` and `dcolour`. That is what
/// carries the 9720 K blast temperature into the created element and keeps
/// `CEXP`'s chain reaction intact, so the port assigns the type directly
/// instead of queueing a whole-row replacement. The `ChangeType` hooks are not
/// run: only `ALUM` has one and only for `from == LAVA`.
///
/// The reference does all of this once per non-empty neighbour, inside the
/// scan; the writes are identical each time, so the port does them once per
/// armed charge (the mass override in particular is a store, not an add). The
/// type assignment comes last, after the pressure/heat/gravity writes, and the
/// reference keeps looping afterwards (`return 0` follows the scan), so this
/// kernel never exits early.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn cexp_blast_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &[i32],
    part_tmp2: &mut [i32],
    part_ctype: &[u32],
    part_flags: &mut [u32],
    air_pv: &mut [f32],
    air_hv: &mut [f32],
    override_mass: &mut [i32],
    override_set: &mut [u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_CEXP {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let mut tmp2 = part_tmp2[id];
                if tmp2 == CEXP_FLOOD {
                    tmp2 = CEXP_ARMED;
                    part_tmp2[id] = CEXP_ARMED;
                }
                if tmp2 > 0i32 {
                    let life = part_life[id];
                    let temp = part_temp[id];
                    let mut any_neighbour = false;
                    for oy in 0u32..5u32 {
                        let dy = oy as i32 - 2i32;
                        for ox in 0u32..5u32 {
                            let dx = ox as i32 - 2i32;
                            if dx != 0i32 || dy != 0i32 {
                                let mut nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv == 0u32 {
                                    nv = pmap_at(photons, xres, yres, px + dx, py + dy);
                                }
                                if nv != 0u32 {
                                    any_neighbour = true;
                                    let nid = pmap_slot(nv);
                                    // `CEXP.cpp:81`: the neighbour keeps the
                                    // charge's temperature.
                                    part_temp[nid] = temp;
                                    // `CEXP.cpp:78-79`: pressure and ambient
                                    // heat land in the neighbour's air cell.
                                    let ncx = (px as u32 / CELL) as i32 + dx;
                                    let ncy = (py as u32 / CELL) as i32 + dy;
                                    if ncx >= 0i32
                                        && ncy >= 0i32
                                        && (ncx as u32) < xcells
                                        && (ncy as u32) < ycells
                                    {
                                        let ncell = ((ncy as u32) * xcells + (ncx as u32)) as usize;
                                        air_pv[ncell] = life as f32;
                                        air_hv[ncell] = temp;
                                    }
                                }
                            }
                        }
                    }
                    if any_neighbour {
                        // `CEXP.cpp:80`: `gravIn.mass[x/CELL][y/CELL] = tmp`.
                        let cell = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                        set_grav_mass_override(
                            override_mass,
                            override_set,
                            cell,
                            part_tmp[id] as f32,
                        );
                        // `CEXP.cpp:82`: `part_change_type(i, x, y, ctype)`.
                        // `Simulation.cpp:1801-1845` keeps the whole row and
                        // rewrites only the type; `part_change_type` rejects an
                        // id outside `1..PT_NUM`, which the guard reproduces.
                        // The energy-layer move (`pmap` -> `photons`) is
                        // reproduced by the per-frame map rebuild, so no layer
                        // write is needed here.
                        let ctype = part_ctype[id];
                        if ctype != PT_NONE && ctype < PT_NUM_U32 {
                            part_ty[id] = ctype;
                        }
                    }
                }
            }
        }
    }
}

/// `BFLM.cpp:52-80`: black flames.
///
/// `tmp` is the fuel gauge: it burns down by one per frame and is refilled to
/// [`BFLM_FUEL`] by any 4x4 neighbour that is not another black flame, `DMRN`
/// or `WALL`. One such neighbour in [`BFLM_SPREAD_DEN`] is converted in place —
/// `parts[ID(r)].life = 55; part_change_type(ID(r), ..., PT_BFLM)`, which keeps
/// the neighbour's other fields, so this is a direct two-field write rather
/// than a queued replacement.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn bflm_update_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_tmp: &mut [i32],
    part_flags: &mut [u32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_BFLM {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 0x1B87_3593u32));
                let mut tmp = part_tmp[id];
                // `BFLM.cpp:54-57`.
                if tmp > 0i32 {
                    tmp -= 1i32;
                }
                // `BFLM.cpp:58-78`: the 4x4 scan (`rx`/`ry` run `-2 .. 1`).
                for oy in 0u32..4u32 {
                    let dy = oy as i32 - 2i32;
                    for ox in 0u32..4u32 {
                        let dx = ox as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let mut nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv == 0u32 {
                                nv = pmap_at(photons, xres, yres, px + dx, py + dy);
                            }
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nt = part_ty[nid];
                                if nt != PT_NONE && nt != PT_BFLM && nt != PT_DMRN && nt != PT_WALL
                                {
                                    tmp = BFLM_FUEL;
                                    if chance(roll ^ (nid as u32), 1u32, BFLM_SPREAD_DEN) {
                                        part_life[nid] = BFLM_CHILD_LIFE;
                                        part_ty[nid] = PT_BFLM;
                                    }
                                }
                            }
                        }
                    }
                }
                part_tmp[id] = tmp;
            }
        }
    }
}

/// Launch every force/explosive second-set pass for one frame.
///
/// Order: `MGNT` (state, flood, field), `MISL`, `PROJ`, `PRMT`, `CEXP`
/// (trigger, flood, blast), `BFLM`. The reference runs the element `Update`s in
/// slot order, so the cross-element ordering is only exact within one element;
/// the group order above keeps each pass after the state it consumes.
///
/// The one-shot gravity override grid this pass writes is emptied by
/// [`super::launch`]'s `clear_elem_block_kernel` at the very top of the modext
/// pass, before any group writes to it, so a `CEXP` injection survives exactly
/// the one gravity solve that consumes it.
pub(crate) fn launch(bufs: &SimBuffers, dims: &PassDims, seed: u32) {
    let (cubes, dim) = launch_grid(dims.pixels);
    let cells = dims.cells as u32;
    let xres = dims.xres;
    let yres = dims.yres;
    let xcells = dims.xcells;
    let ycells = cells / xcells.max(1u32);

    // `MGNT.cpp:48-183`.
    unsafe {
        mgnt_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            xres,
            yres,
            seed,
        );
    }
    // `MGNT.cpp:96` / `:106`: the two `flood_prop` calls rewrite their whole
    // 4-connected run in the frame the spark fires; each field gets its own
    // seed scan and drain because the two seeds can sit in different runs.
    launch_seed_scan(bufs, dims, FLOOD_FIELD_TMP, MGNT_FLOOD, MGNT_FLOOD, PT_MGNT);
    launch_drain(bufs, dims, FLOOD_MODE_TMP, PT_MGNT, MGNT_FLOOD, 0i32);
    launch_seed_scan(
        bufs,
        dims,
        FLOOD_FIELD_TMP2,
        MGNT_FLOOD,
        MGNT_FLOOD,
        PT_MGNT,
    );
    launch_drain(bufs, dims, FLOOD_MODE_TMP2, PT_MGNT, MGNT_FLOOD, 0i32);
    unsafe {
        mgnt_attract_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            xres,
            yres,
        );
    }

    // `MISL.cpp:49-146`.
    unsafe {
        misl_update_kernel::launch_unchecked(
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
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            xres,
            yres,
            xcells,
        );
    }

    // `PROJ.cpp:54-97`.
    unsafe {
        proj_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            xres,
            yres,
        );
    }

    // `PRMT.cpp:58-96`.
    unsafe {
        prmt_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            xres,
            yres,
            xcells,
            cells,
            seed,
        );
    }

    // `CEXP.cpp:47-89`.
    unsafe {
        cexp_trigger_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            xres,
            yres,
        );
    }
    // `CEXP.cpp:70-74`: the armed run arms the rest of its 4-connected
    // component in the same frame.
    launch_seed_scan(bufs, dims, FLOOD_FIELD_TMP2, CEXP_ARMED, CEXP_FLOOD, PT_CEXP);
    launch_drain(bufs, dims, FLOOD_MODE_TMP2, PT_CEXP, CEXP_FLOOD, 0i32);
    unsafe {
        cexp_blast_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_hv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_mass_override.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_mass_override_set.clone(), dims.cells),
            xres,
            yres,
            xcells,
            ycells,
        );
    }

    // `BFLM.cpp:52-80`.
    unsafe {
        bflm_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            xres,
            yres,
            seed,
        );
    }
}
