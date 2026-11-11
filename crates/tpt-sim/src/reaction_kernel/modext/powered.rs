//! Powered second-set devices: `LED`, `TIMC`, `PINV`, `PPTI`, `PPTO`, `PCON`.
//!
//! Ports the `PSCN`-driven half of `LED.cpp`, `TIMC.cpp` (sparked conversion
//! into `ctype`), `PINV.cpp` (pixel transparency while powered), `PPTI.cpp` /
//! `PPTO.cpp` (the powered portal pair) and `PCON.cpp` (the powered converter).
//!
//! # TPT references
//!
//! * `LED.cpp:56-83`: clamp the case temperature to `274.15..374.15 K`, run
//!   `life` down, and latch `life = 8` from a `PSCN`-carrying spark in the
//!   `rx, ry in -2..2` window (TPT's half-open loops make that a 4x4 block).
//!   The latch runs `flood_prop(x, y, FIELD_LIFE ,8)`, a 4-connected flood over
//!   the whole `LED` component, reproduced by `flood_component`'s seed scan and
//!   device drain.
//! * `TIMC.cpp:52-77`: `tmp2` is a countdown driven by sparks (`life 1..3`,
//!   `ctype == PSCN`) and by the 5x5 `TIMC` handshake, and the crystal turns
//!   into its `ctype` when the countdown reaches `1`.
//! * `PINV.cpp:49-75`: `life == 10` is the open state; the 5x5 ring passes the
//!   open state on (an idle neighbour is opened, an open one sinks to `9`).
//! * `PCON.cpp:66-125`: same 5x5 handshake on `tmp2`, then either latch the
//!   first valid neighbour type into `ctype` or convert the 3x3 ring into
//!   `ctype` (respecting the `tmp` restriction), skipping `PCON`, `DMND` and
//!   the target type itself. The conversion is `create_part(ID(r), ..., v)`:
//!   [`request_replace`] has no `v` argument, so the payload is decoded into
//!   the record's field overrides ([`life_create_payload`] for a `LIFE`
//!   target, `tmp` for `TESC`/`FILT`).
//! * `PPTI.cpp:74-176` / `PPTO.cpp:81-205`: the powered portal pair. Both run
//!   the identical capture/release rings as `PRTI`/`PRTO` against the same
//!   `portalp` FIFO (`portals::portal_capture_ring` / `portals::spark_ring`),
//!   gated behind `tmp2 == 0` and animated with the same orbital steps.
//!
//! # Why the portal pair launches here
//!
//! `PPTO`'s release is `create_part(-1, ...)`, which the device expresses
//! through [`request_create`]; the record is filled by
//! `shared::drain_gpu_creates`, which `step_reactions_gpu` runs *before*
//! `portals::launch`. Launching `PPTI`/`PPTO` from this group (which runs
//! before the drain) is what makes a released particle live in the same tick;
//! the pixel-parallel `PRTI`/`PRTO` kernels stay in the portal pass.
//! A capture marks its source slot `FLAG_PORTAL_HELD` and clears `part_ty`, so
//! a release clears that bit on the donated slot and lets the drain allocate
//! the new particle — the port cannot name a slot through the creation queue.

// Same rationale as `pixel_mode.rs`: collapsing a nested `if` changes the
// CubeCL expansion, and the device IR has neither `f32::clamp` nor compound
// assignment through an array element.
#![allow(
    clippy::collapsible_if,
    clippy::manual_clamp,
    clippy::assign_op_pattern
)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use crate::buffers::SimBuffers;
use crate::gol_kernel::NGOL;
use crate::reaction_kernel::flood_component::{
    FLOOD_FIELD_LIFE, FLOOD_MODE_LIFE, launch_drain, launch_seed_scan,
};
use crate::reaction_kernel::portals::{
    portal_animate_in, portal_animate_out, portal_capture_ring, portal_offset, spark_ring,
    temp_channel,
};
use crate::reaction_kernel::shared::{
    CR_SET_CTYPE, CR_SET_DCOLOUR, CR_SET_LIFE, CR_SET_TEMP, CR_SET_TMP, CR_SET_TMP2, CR_SET_VX,
    CR_SET_VY, CREATE_QUEUE_WORDS, hash_u32, insulated, launch_grid, mix_index, pmap_at, pmap_slot,
    request_create, request_replace,
};

use super::super::PassDims;

/// `LED`, the only element in this group with a `flood_prop` latch.
const LED_U32: u32 = PT_LED;
/// `TIMC`, the time crystal.
const TIMC_U32: u32 = PT_TIMC;
/// `PINV`, the powered invisible pixel.
const PINV_U32: u32 = PT_PINV;
/// `PCON`, the powered converter.
const PCON_U32: u32 = PT_PCON;
/// `PPTI`, powered `PRTI`.
const PPTI_U32: u32 = PT_PPTI;
/// `PPTO`, powered `PRTO`.
const PPTO_U32: u32 = PT_PPTO;
/// `SPRK`, the spark all three trigger scans look for.
const SPRK_U32: u32 = PT_SPRK;
/// `PSCN`, the only conductor a device spark is allowed to come from.
const PSCN_U32: u32 = PT_PSCN;
/// `NSCN`, excluded from the `PCON` ctype latch.
const NSCN_U32: u32 = PT_NSCN;
/// `CLNE`, excluded from the `PCON` ctype latch (clones carry no type).
const CLNE_U32: u32 = PT_CLNE;
/// `PCLN`, excluded from the `PCON` ctype latch.
const PCLN_U32: u32 = PT_PCLN;
/// `BCLN`, excluded from the `PCON` ctype latch.
const BCLN_U32: u32 = PT_BCLN;
/// `PBCN`, excluded from the `PCON` ctype latch.
const PBCN_U32: u32 = PT_PBCN;
/// `STKM`, excluded from the `PCON` ctype latch (stickman rigs).
const STKM_U32: u32 = PT_STKM;
/// `STKM2`, excluded from the `PCON` ctype latch.
const STKM2_U32: u32 = PT_STKM2;
/// `LIFE`, whose `Create` hook packs its ruleset into the high `ctype` bits.
const LIFE_U32: u32 = PT_LIFE;
/// `TESC`, whose `Create` hook takes `tmp` from the payload.
const TESC_U32: u32 = PT_TESC;
/// `FILT`, whose `Create` hook takes the filter mode from the payload.
const FILT_U32: u32 = PT_FILT;
/// `DMND`, never converted by a `PCON`.
const DMND_U32: u32 = PT_DMND;
/// `PT_NONE`.
const NONE_U32: u32 = PT_NONE;
/// `PT_NUM`.
const NUM_U32: u32 = PT_NUM as u32;
/// `PORTAL_CHANNELS`, the clamp [`temp_channel`] already applies.
const PORTAL_CHANNELS_U32: u32 = PORTAL_CHANNELS as u32;
/// `FLAG_PORTAL_HELD`, set by a capture and cleared by a release.
const FLAG_PORTAL_HELD_U32: u32 = FLAG_PORTAL_HELD;

const _: () = assert!(LED_U32 == PT_LED);
const _: () = assert!(TIMC_U32 == PT_TIMC);
const _: () = assert!(PINV_U32 == PT_PINV);
const _: () = assert!(PCON_U32 == PT_PCON);
const _: () = assert!(PPTI_U32 == PT_PPTI);
const _: () = assert!(PPTO_U32 == PT_PPTO);
const _: () = assert!(SPRK_U32 == PT_SPRK);
const _: () = assert!(PSCN_U32 == PT_PSCN);
const _: () = assert!(NSCN_U32 == PT_NSCN);
const _: () = assert!(CLNE_U32 == PT_CLNE);
const _: () = assert!(PCLN_U32 == PT_PCLN);
const _: () = assert!(BCLN_U32 == PT_BCLN);
const _: () = assert!(PBCN_U32 == PT_PBCN);
const _: () = assert!(STKM_U32 == PT_STKM);
const _: () = assert!(STKM2_U32 == PT_STKM2);
const _: () = assert!(LIFE_U32 == PT_LIFE);
const _: () = assert!(TESC_U32 == PT_TESC);
const _: () = assert!(FILT_U32 == PT_FILT);
const _: () = assert!(DMND_U32 == PT_DMND);
const _: () = assert!(NONE_U32 == PT_NONE);
const _: () = assert!(NUM_U32 as usize == PT_NUM);
const _: () = assert!(FLAG_PORTAL_HELD_U32 == FLAG_PORTAL_HELD);
// `portals::temp_channel` clamps its hardcoded `102` to `PORTAL_CHANNELS - 1`;
// keep the mirror honest (`PRTI.cpp:68-72`).
const _: () = assert!(PORTAL_CHANNELS_U32 == 103);
/// `LED` case temperature floor (`LED.cpp:60-61`).
const LED_TEMP_MIN: f32 = 274.15f32;
/// `LED` case temperature ceiling (`LED.cpp:58-59`).
const LED_TEMP_MAX: f32 = 374.15f32;
/// `LED` life a lit run is refreshed to (`LED.cpp:79`).
const LED_LIT_LIFE: i32 = 8;
/// `TIMC` countdown a `PSCN` spark sets (`TIMC.cpp:64`).
const TIMC_SPARK_TMP2: i32 = 100;
/// `TIMC` countdown that converts the crystal (`TIMC.cpp:74-76`).
const TIMC_CONVERT_TMP2: i32 = 1;
/// `PINV` `life` that means "open" (`PINV.cpp:51`).
const PINV_OPEN_LIFE: i32 = 10;
/// `PINV` `life` an opened pixel sinks to when a neighbour is opening
/// (`PINV.cpp:67`).
const PINV_SINK_LIFE: i32 = 9;
/// `PCON` `tmp2` that means "powered" (`PCON.cpp:70`).
const POWERED_TMP2: i32 = 10;
/// `PCON`/`PPTI`/`PPTO` `tmp2` a powered device sinks to while a neighbour is
/// counting (`PCON.cpp:79`, `PPTI.cpp:90`, `PPTO.cpp:97`).
const POWERED_SINK_TMP2: i32 = 9;
/// `LIFE`'s ruleset lives in the high `ctype` bits (`Simulation.cpp:1990-1995`,
/// `PMAPID(x) == x << 9`); `Element_LIFE()`'s `Create` hook masks it to 21 bits.
const LIFE_RULESET_BITS: u32 = 0x1F_FFFFu32;

/// `LED.cpp:56-66`: clamp the case temperature and run the glow timer down.
///
/// The clamp is what keeps a decor-coloured LED's `firea` (the `temp - 273.15`
/// glow, `LED.cpp:88`) inside the palette range.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn led_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_life: &mut [i32],
    part_ctype: &[u32],
    part_temp: &mut [f32],
    part_x: &[f32],
    part_y: &[f32],
    xres: u32,
    yres: u32,
    n_slots: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize && part_ty[i] == LED_U32 {
        let mut temp = part_temp[i];
        if temp > LED_TEMP_MAX {
            temp = LED_TEMP_MAX;
        }
        if temp < LED_TEMP_MIN {
            temp = LED_TEMP_MIN;
        }
        part_temp[i] = temp;

        let mut life = part_life[i];
        if life > 0i32 {
            life -= 1i32;
        }

        let px = i32::cast_from(part_x[i] + 0.5f32);
        let py = i32::cast_from(part_y[i] + 0.5f32);
        let mut lit = false;
        // `LED.cpp:68-70`: `rx, ry in -2..2`, i.e. the 4x4 block up-left of the
        // pixel; the `rx || ry` guard only skips its own cell.
        for ox in 0u32..4u32 {
            for oy in 0u32..4u32 {
                let rx = ox as i32 - 2i32;
                let ry = oy as i32 - 2i32;
                if (rx != 0i32 || ry != 0i32) && !lit {
                    let nx = px + rx;
                    let ny = py + ry;
                    let nv = pmap_at(pmap, xres, yres, nx, ny);
                    // `parts_avg(ID(r), i, PT_INSL) == PT_INSL` skips a spark
                    // behind a one-pixel insulator wall.
                    if nv != 0u32 && !insulated(pmap, xres, yres, nx, ny, px, py) {
                        let nid = pmap_slot(nv);
                        if part_ty[nid] == SPRK_U32
                            && part_life[nid] > 0i32
                            && part_ctype[nid] == PSCN_U32
                        {
                            lit = true;
                        }
                    }
                }
            }
        }
        if lit {
            life = LED_LIT_LIFE;
        }
        part_life[i] = life;
    }
}

/// The `tmp2 != 10 / == 10` handshake of `PCON.cpp:70-88`, `PPTI.cpp:80-98`
/// and `PPTO.cpp:87-105`.
///
/// A powered device (`tmp2 == 10`) kicks its 5x5 ring: a counting neighbour
/// (`0 < tmp2 < 10`) makes this one sink to `9`, an idle one (`tmp2 == 0`) is
/// kicked to `10`. That is the two-phase ripple a line of devices is built
/// from, and it is why a device only counts down on the frames it is not
/// powered.
#[cube]
#[allow(clippy::too_many_arguments)]
fn spread_tmp2_5x5(
    pmap: &[u32],
    part_tmp2: &mut [i32],
    device_ty: u32,
    px: i32,
    py: i32,
    xres: u32,
    yres: u32,
    s: usize,
) {
    for ox in 0u32..5u32 {
        for oy in 0u32..5u32 {
            let rx = ox as i32 - 2i32;
            let ry = oy as i32 - 2i32;
            if rx != 0i32 || ry != 0i32 {
                let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                if nv != 0u32 && (nv & 0x1FFu32) == device_ty {
                    let nid = pmap_slot(nv);
                    let other = part_tmp2[nid];
                    if other < POWERED_TMP2 && other > 0i32 {
                        part_tmp2[s] = POWERED_SINK_TMP2;
                    } else if other == 0i32 {
                        part_tmp2[nid] = POWERED_TMP2;
                    }
                }
            }
        }
    }
}

/// `PINV.cpp:49-75`'s `life` ring: an open pixel (`life == 10`) opens an idle
/// `PINV` neighbour and sinks to `9` next to one that is already opening.
#[cube]
#[allow(clippy::too_many_arguments)]
fn spread_life_5x5(
    pmap: &[u32],
    part_life: &mut [i32],
    device_ty: u32,
    px: i32,
    py: i32,
    xres: u32,
    yres: u32,
    s: usize,
) {
    for ox in 0u32..5u32 {
        for oy in 0u32..5u32 {
            let rx = ox as i32 - 2i32;
            let ry = oy as i32 - 2i32;
            if rx != 0i32 || ry != 0i32 {
                let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                if nv != 0u32 && (nv & 0x1FFu32) == device_ty {
                    let nid = pmap_slot(nv);
                    let other = part_life[nid];
                    if other < PINV_OPEN_LIFE && other > 0i32 {
                        part_life[s] = PINV_SINK_LIFE;
                    } else if other == 0i32 {
                        part_life[nid] = PINV_OPEN_LIFE;
                    }
                }
            }
        }
    }
}

/// `TIMC.cpp:52-77`: the time crystal.
///
/// The countdown runs down every frame; a `PSCN` spark with `life 1..3` behind
/// no insulator refills it to `100`, and the 5x5 `TIMC` ring copies a running
/// countdown between neighbours with TPT's slot-order nudge (a higher slot
/// hands over one less, and receives one more, so a handshake can only move
/// the count in one direction). At `1` the crystal is re-created as its
/// `ctype`.
///
/// TPT runs the conversion inside the neighbour loop, so a crystal whose
/// neighbours are all gone never converts; the `saw_neighbour` flag keeps that
/// edge. The conversion itself runs once, after the ring, instead of once per
/// neighbour — the later iterations of `create_part(i, ...)` only ever see the
/// new element's own `tmp2`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn timc_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_life: &[i32],
    part_ctype: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    part_tmp2: &mut [i32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    n_slots: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize && part_ty[i] == TIMC_U32 {
        let px = i32::cast_from(part_x[i] + 0.5f32);
        let py = i32::cast_from(part_y[i] + 0.5f32);

        if part_tmp2[i] > 0i32 {
            part_tmp2[i] = part_tmp2[i] - 1i32;
        }

        let mut saw_neighbour = false;
        for ox in 0u32..5u32 {
            for oy in 0u32..5u32 {
                let rx = ox as i32 - 2i32;
                let ry = oy as i32 - 2i32;
                if rx != 0i32 || ry != 0i32 {
                    let nx = px + rx;
                    let ny = py + ry;
                    let nv = pmap_at(pmap, xres, yres, nx, ny);
                    // `TIMC.cpp:57-58`: an empty or insulated neighbour is not
                    // visited at all, which is also what gates the conversion.
                    if nv != 0u32 && !insulated(pmap, xres, yres, nx, ny, px, py) {
                        saw_neighbour = true;
                        let nid = pmap_slot(nv);
                        let nt = nv & 0x1FFu32;
                        if nt == SPRK_U32 {
                            if part_life[i] == 0i32
                                && part_life[nid] > 0i32
                                && part_life[nid] < 4i32
                                && part_ctype[nid] == PSCN_U32
                            {
                                part_tmp2[i] = TIMC_SPARK_TMP2;
                            }
                        } else if nt == TIMC_U32 && part_ty[nid] == TIMC_U32 {
                            let other = part_tmp2[nid];
                            if part_tmp2[i] == 0i32 {
                                if other != 0i32 {
                                    let mut next = other;
                                    if nid > i {
                                        next -= 1i32;
                                    }
                                    part_tmp2[i] = next;
                                }
                            } else if other == 0i32 {
                                let mut next = part_tmp2[i];
                                if nid > i {
                                    next += 1i32;
                                }
                                part_tmp2[nid] = next;
                            }
                        }
                    }
                }
            }
        }

        let cty = part_ctype[i];
        if saw_neighbour && part_tmp2[i] == TIMC_CONVERT_TMP2 && cty > NONE_U32 && cty < NUM_U32 {
            // `TIMC.cpp:74-76`: `create_part(i, x, y, parts[i].ctype)` — the
            // crystal's own slot is re-created as the element its `ctype`
            // names (default `CHLR`). The four-argument overload leaves
            // `create_part`'s `v` at `-1`, which `LIFE`'s `Create` treats as
            // ruleset `0` (`LIFE.cpp:97-98`); every other target ignores it.
            let mut mask = 0u32;
            let mut ov_ctype = 0u32;
            let mut ov_tmp = 0i32;
            let mut ov_tmp2 = 0i32;
            let mut ov_dcolour = 0u32;
            if cty == LIFE_U32 {
                let (rule, tmp2, dcolour, tmp) = life_create_payload(0u32);
                mask |= CR_SET_CTYPE;
                mask |= CR_SET_TMP2;
                mask |= CR_SET_DCOLOUR;
                mask |= CR_SET_TMP;
                ov_ctype |= rule;
                ov_tmp2 |= tmp2;
                ov_tmp |= tmp;
                ov_dcolour |= dcolour;
            }
            request_replace(
                create_queue,
                i as u32,
                cty,
                px,
                py,
                mask,
                0i32,
                ov_ctype,
                0.0f32,
                ov_tmp,
                ov_tmp2,
                0i32,
                0i32,
                ov_dcolour,
                0u32,
                0.0f32,
                0.0f32,
            );
        }
    }
}

/// `PINV.cpp:49-75`: the powered invisible pixel.
///
/// `life == 10` is transparent and passes the open state around; any other
/// value runs down, so a pixel closes again five frames after it was last
/// refreshed by the ring.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pinv_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_life: &mut [i32],
    part_x: &[f32],
    part_y: &[f32],
    xres: u32,
    yres: u32,
    n_slots: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize && part_ty[i] == PINV_U32 {
        if part_life[i] != PINV_OPEN_LIFE {
            if part_life[i] > 0i32 {
                part_life[i] = part_life[i] - 1i32;
            }
        } else {
            let px = i32::cast_from(part_x[i] + 0.5f32);
            let py = i32::cast_from(part_y[i] + 0.5f32);
            spread_life_5x5(pmap, part_life, PINV_U32, px, py, xres, yres, i);
        }
    }
}

/// `NGOL` mirrored for kernel bodies (rule 1).
const GOL_NGOL_U32: u32 = NGOL as u32;
const _: () = assert!(GOL_NGOL_U32 as usize == NGOL);

/// `BUILTIN_GOL[rule].(colour, colour2)`, mirrored for the device `LIFE`
/// `Create` payload decode.
///
/// The reaction pass has no Game of Life colour buffer bound — the table lives
/// in `gol_kernel`'s private scratch — so the 24 pairs are branches here, and
/// the `gol_colour_mirror_matches_the_builtin_table` test pins them to
/// `gol_kernel::BUILTIN_GOL`. An out-of-range rule yields `(0, 0)`, which the
/// caller turns into "leave the element defaults".
#[cube]
fn gol_builtin_colours(rule: u32) -> (u32, u32) {
    let mut colour = 0u32;
    let mut colour2 = 0u32;
    if rule == 0u32 {
        colour |= 0x0CAC00u32;
        colour2 |= 0x0CAC00u32;
    } else if rule == 1u32 {
        colour |= 0xFF0000u32;
        colour2 |= 0xFF0000u32;
    } else if rule == 2u32 {
        colour |= 0x0000FFu32;
        colour2 |= 0x0000FFu32;
    } else if rule == 3u32 {
        colour |= 0xFFFF00u32;
        colour2 |= 0xFFFF00u32;
    } else if rule == 4u32 {
        colour |= 0x00FFFFu32;
        colour2 |= 0x00FFFFu32;
    } else if rule == 5u32 {
        colour |= 0xFF00FFu32;
        colour2 |= 0xFF00FFu32;
    } else if rule == 6u32 {
        colour |= 0xFFFFFFu32;
        colour2 |= 0xFFFFFFu32;
    } else if rule == 7u32 {
        colour |= 0xE05010u32;
        colour2 |= 0xE05010u32;
    } else if rule == 8u32 {
        colour |= 0x500000u32;
        colour2 |= 0x500000u32;
    } else if rule == 9u32 {
        colour |= 0x500050u32;
        colour2 |= 0x500050u32;
    } else if rule == 10u32 {
        colour |= 0x505050u32;
        colour2 |= 0x505050u32;
    } else if rule == 11u32 {
        colour |= 0x5000FFu32;
        colour2 |= 0x5000FFu32;
    } else if rule == 12u32 {
        colour |= 0xFBEC7Du32;
        colour2 |= 0xFBEC7Du32;
    } else if rule == 13u32 {
        colour |= 0xA8E4A0u32;
        colour2 |= 0xA8E4A0u32;
    } else if rule == 14u32 {
        colour |= 0x9ACD32u32;
        colour2 |= 0x9ACD32u32;
    } else if rule == 15u32 {
        colour |= 0x0047ABu32;
        colour2 |= 0x0047ABu32;
    } else if rule == 16u32 {
        colour |= 0xE5B73Bu32;
        colour2 |= 0xE5B73Bu32;
    } else if rule == 17u32 {
        colour |= 0x259588u32;
        colour2 |= 0x259588u32;
    } else if rule == 18u32 {
        colour |= 0x0C3C00u32;
        colour2 |= 0x0C3C00u32;
    } else if rule == 19u32 {
        colour |= 0xFF0000u32;
        colour2 |= 0xFFFF00u32;
    } else if rule == 20u32 {
        colour |= 0x006432u32;
        colour2 |= 0x00FF5Au32;
    } else if rule == 21u32 {
        colour |= 0x000040u32;
        colour2 |= 0x0000E6u32;
    } else if rule == 22u32 {
        colour |= 0x006400u32;
        colour2 |= 0x00FF00u32;
    } else if rule == 23u32 {
        colour |= 0xFFFF00u32;
        colour2 |= 0x969600u32;
    }
    (colour, colour2)
}

/// `LIFE`'s `Create` payload (`LIFE.cpp:93-119`), exactly as the host
/// `create_part` arm decodes it (`buffers/particles.rs:810-824`).
///
/// `v` is the packed-`ctype` var payload `ID(ctype) = ctype >> 9` that
/// `PCON.cpp:123` hands to `create_part`. Returns
/// `(ctype, tmp2, dcolour, tmp)`:
///
/// * `ctype = v & 0x1F_FFFF` — the ruleset id (builtin index or packed rule);
/// * `tmp2 = ((ctype >> 17) & 0xF) + 1` — the number of live states;
/// * `dcolour` / `tmp` are the builtin rule's live/dying colours and are only
///   meaningful when `ctype < NGOL`; the port has no custom-GOL colour lookup,
///   matching the host arm, which leaves the defaults in that case.
///
/// The device creation drain has no `PT_LIFE` arm, so the only caller is
/// `pcon_kernel`. `shared::create_fields_kernel` can reuse it for a `PT_LIFE`
/// record once this module path is exposed (`mod powered` is private to
/// `modext`; a `pub(crate) use powered::life_create_payload;` in
/// `modext/mod.rs` is all that is needed).
#[cube]
pub(crate) fn life_create_payload(v: u32) -> (u32, i32, u32, i32) {
    let rule = v & LIFE_RULESET_BITS;
    let (colour, colour2) = gol_builtin_colours(rule);
    let mut dcolour = 0u32;
    dcolour |= colour;
    let mut tmp = 0i32;
    tmp |= colour2 as i32;
    let tmp2 = (((rule >> 17u32) & 0xFu32) + 1u32) as i32;
    (rule, tmp2, dcolour, tmp)
}

/// `PCON.cpp:66-125`: the powered converter.
///
/// `tmp2` is the power latch (the same 5x5 handshake as the portals). While it
/// is `0` the converter either latches the first neighbour type it accepts into
/// `ctype` (when the current `ctype` is empty, out of range or a `PCON`), or it
/// converts the 3x3 ring into `ctype`. The `tmp` field, when it names a live
/// element, restricts which neighbours are converted.
///
/// `ElementDef::enabled` is a host-only flag in this port, so the device test
/// for `elements[ctype].Enabled` is the id range, exactly as in
/// `pipe::pump` and `special::cray`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pcon_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_ctype: &mut [u32],
    part_x: &[f32],
    part_y: &[f32],
    part_tmp: &[i32],
    part_tmp2: &mut [i32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    n_slots: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize && part_ty[i] == PCON_U32 {
        let px = i32::cast_from(part_x[i] + 0.5f32);
        let py = i32::cast_from(part_y[i] + 0.5f32);

        if part_tmp2[i] != POWERED_TMP2 {
            if part_tmp2[i] > 0i32 {
                part_tmp2[i] = part_tmp2[i] - 1i32;
            }
        } else {
            spread_tmp2_5x5(pmap, part_tmp2, PCON_U32, px, py, xres, yres, i);
        }

        if part_tmp2[i] == 0i32 {
            // `PCON.cpp:92`: `TYP(parts[i].ctype)` — the target element.
            let target = part_ctype[i] & 0x1FFu32;
            if target == NONE_U32 || target >= NUM_U32 || target == PCON_U32 {
                // `PCON.cpp:94-112`: latch the first neighbour that is not a
                // clone, a stickman, a conductor, a spark or another converter.
                // TPT assigns `parts[i].ctype` on every match, so the last one
                // in the 3x3 sweep wins; writing it here keeps that order and
                // needs no scratch value.
                for ox in 0u32..3u32 {
                    for oy in 0u32..3u32 {
                        let rx = ox as i32 - 1i32;
                        let ry = oy as i32 - 1i32;
                        if rx != 0i32 || ry != 0i32 {
                            let mut r = pmap_at(photons, xres, yres, px + rx, py + ry);
                            if r == 0u32 {
                                r = pmap_at(pmap, xres, yres, px + rx, py + ry);
                            }
                            if r != 0u32 {
                                let rt = r & 0x1FFu32;
                                if rt > NONE_U32
                                    && rt < NUM_U32
                                    && rt != CLNE_U32
                                    && rt != PCLN_U32
                                    && rt != BCLN_U32
                                    && rt != STKM_U32
                                    && rt != PSCN_U32
                                    && rt != NSCN_U32
                                    && rt != SPRK_U32
                                    && rt != PBCN_U32
                                    && rt != STKM2_U32
                                    && rt != PCON_U32
                                {
                                    part_ctype[i] = rt;
                                    if rt == LIFE_U32 {
                                        // `PCON.cpp:108-109`:
                                        // `ctype |= PMAPID(parts[ID(r)].ctype)`
                                        // keeps the ruleset the neighbour runs.
                                        part_ctype[i] |=
                                            (part_ctype[pmap_slot(r)] & LIFE_RULESET_BITS) << 9u32;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // `PCON.cpp:116-124`: `int restrictElement = IsElement(tmp) ? tmp : 0`.
                let restrict = if part_tmp[i] > 0i32 && (part_tmp[i] as u32) < NUM_U32 {
                    part_tmp[i] as u32
                } else {
                    NONE_U32
                };
                // `ID(parts[i].ctype)`: `create_part`'s `v` payload, decoded
                // once because the target `ctype` is loop-invariant. Only
                // `LIFE`, `TESC` and `FILT` read `v` (see the module note on
                // `LIGH`); every other target ignores it, so its override mask
                // stays empty and the drain uses the element defaults.
                let v = (part_ctype[i] >> 9u32) & LIFE_RULESET_BITS;
                let mut mask = 0u32;
                let mut ov_ctype = 0u32;
                let mut ov_tmp = 0i32;
                let mut ov_tmp2 = 0i32;
                let mut ov_dcolour = 0u32;
                if target == LIFE_U32 {
                    let (rule, tmp2, dcolour, tmp) = life_create_payload(v);
                    mask |= CR_SET_CTYPE;
                    mask |= CR_SET_TMP2;
                    ov_ctype |= rule;
                    ov_tmp2 |= tmp2;
                    ov_tmp |= tmp;
                    if rule < GOL_NGOL_U32 {
                        mask |= CR_SET_DCOLOUR;
                        mask |= CR_SET_TMP;
                        ov_dcolour |= dcolour;
                    }
                } else if target == TESC_U32 {
                    // `TESC.cpp:44-52`: `tmp = v.min(300)`.
                    mask |= CR_SET_TMP;
                    if v < 300u32 {
                        ov_tmp |= v as i32;
                    } else {
                        ov_tmp |= 300i32;
                    }
                } else if target == FILT_U32 {
                    // `FILT.cpp:44-47`: `tmp = v` (the filter mode).
                    mask |= CR_SET_TMP;
                    ov_tmp |= v as i32;
                }
                // `LIGH.cpp:312-336` also reads `v` (the bolt power), but its
                // `Create` aims the bolt along the gravity field, which this
                // pass does not bind; a hand-set `PCON` ctype with a `LIGH`
                // payload therefore gets the `DefaultProperties` bolt, exactly
                // like `create_part` with `v == 0`.
                for ox in 0u32..3u32 {
                    for oy in 0u32..3u32 {
                        let rx = ox as i32 - 1i32;
                        let ry = oy as i32 - 1i32;
                        if rx != 0i32 || ry != 0i32 {
                            let mut r = pmap_at(photons, xres, yres, px + rx, py + ry);
                            if r == 0u32 || (restrict != NONE_U32 && (r & 0x1FFu32) != restrict) {
                                r = pmap_at(pmap, xres, yres, px + rx, py + ry);
                            }
                            if r != 0u32 && (restrict == NONE_U32 || (r & 0x1FFu32) == restrict) {
                                let rt = r & 0x1FFu32;
                                let nslot = pmap_slot(r);
                                if rt != PCON_U32
                                    && rt != DMND_U32
                                    && rt != target
                                    && nslot < part_ty.len()
                                {
                                    // `PCON.cpp:123`:
                                    // `create_part(ID(r), x + rx, y + ry,
                                    // TYP(ctype), ID(ctype))` — the neighbour
                                    // slot is re-created as the target, with
                                    // the `v` payload decoded into the fields
                                    // the target's `Create` hook would set.
                                    request_replace(
                                        create_queue,
                                        nslot as u32,
                                        target,
                                        px + rx,
                                        py + ry,
                                        mask,
                                        0i32,
                                        ov_ctype,
                                        0.0f32,
                                        ov_tmp,
                                        ov_tmp2,
                                        0i32,
                                        0i32,
                                        ov_dcolour,
                                        0u32,
                                        0.0f32,
                                        0.0f32,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `PPTI.cpp:74-176`: the powered `PRTI`.
///
/// The power gate is `tmp2 == 0`; while it is closed the portal only runs its
/// countdown/handshake and its orbital dots are cleared. The capture ring is
/// the `PRTI` one, against the same `portalp` FIFO.
#[cube(launch_unchecked)]
#[allow(clippy::manual_clamp, clippy::too_many_arguments)]
fn ppti_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &[f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_vx: &[f32],
    part_vy: &[f32],
    part_flags: &mut [u32],
    part_dcolour: &[u32],
    portal_ty: &mut [Atomic<u32>],
    portal_data: &mut [u32],
    elem_props: &[u32],
    xres: u32,
    yres: u32,
    n_slots: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (xres * yres) as usize {
        terminate!();
    }
    let v = pmap[idx];
    if v == 0u32 {
        terminate!();
    }
    if (v & 0x1FFu32) != PPTI_U32 {
        terminate!();
    }
    let s = pmap_slot(v);
    if part_ty[s] != PPTI_U32 {
        terminate!();
    }
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;

    if part_tmp2[s] != POWERED_TMP2 {
        if part_tmp2[s] > 0i32 {
            part_tmp2[s] = part_tmp2[s] - 1i32;
        }
    } else {
        spread_tmp2_5x5(pmap, part_tmp2, PPTI_U32, px, py, xres, yres, s);
    }

    let mut fe = 0u32;
    if part_tmp2[s] == 0i32 {
        let ch = temp_channel(part_temp[s]);
        fe |= portal_capture_ring(
            pmap,
            photons,
            part_ty,
            part_life,
            part_ctype,
            part_temp,
            part_tmp,
            part_tmp2,
            part_tmp3,
            part_tmp4,
            part_vx,
            part_vy,
            part_flags,
            part_dcolour,
            portal_ty,
            portal_data,
            elem_props,
            PPTI_U32,
            xres,
            yres,
            n_slots,
            s,
            px,
            py,
            ch,
        );
    }
    portal_animate_in(part_life, part_ctype, s, frame_seed, fe);
}

/// `PPTO.cpp:81-205`: the powered `PRTO`.
///
/// The release ring matches `PRTO`'s, with two deliberate differences from the
/// plain portal:
///
/// * it fires only into a pixel whose matter map is empty (`if (!r)`, without
///   `PRTO.cpp`'s photon test);
/// * the emitted particle is re-created through [`request_create`] with the
///   queue entry's fields as overrides. TPT copies the whole stored `Particle`
///   (`PPTO.cpp:176-186`) and only keeps the fresh particle's velocity when the
///   stored one is zero, which the mask reproduces by leaving `VX`/`VY` out.
///
/// TPT's `parts[np].x/y = x + rx, y + ry` is the drain's own pixel write, so no
/// override is needed; the donor slot a capture marked `FLAG_PORTAL_HELD` is
/// released back to the pool once the entry is consumed.
#[cube(launch_unchecked)]
#[allow(
    clippy::manual_clamp,
    clippy::too_many_arguments,
    clippy::approx_constant
)]
fn ppto_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &[f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_flags: &mut [u32],
    portal_ty: &mut [Atomic<u32>],
    portal_data: &[u32],
    create_queue: &mut [Atomic<u32>],
    elem_props: &[u32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (xres * yres) as usize {
        terminate!();
    }
    let v = pmap[idx];
    if v == 0u32 {
        terminate!();
    }
    if (v & 0x1FFu32) != PPTO_U32 {
        terminate!();
    }
    let s = pmap_slot(v);
    if part_ty[s] != PPTO_U32 {
        terminate!();
    }
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;

    if part_tmp2[s] != POWERED_TMP2 {
        if part_tmp2[s] > 0i32 {
            part_tmp2[s] = part_tmp2[s] - 1i32;
        }
    } else {
        spread_tmp2_5x5(pmap, part_tmp2, PPTO_U32, px, py, xres, yres, s);
    }

    let mut fe = 0u32;
    if part_tmp2[s] == 0i32 {
        let ch = temp_channel(part_temp[s]);
        part_tmp[s] = ch as i32;
        for count in 0u32..8u32 {
            let (rx, ry) = portal_offset(count);
            let nx = px + rx;
            let ny = py + ry;
            if nx >= 0i32 && nx < xres as i32 && ny >= 0i32 && ny < yres as i32 {
                if pmap_at(pmap, xres, yres, nx, ny) == 0u32 {
                    fe = 1u32;
                    let mut emitted = false;
                    for nnx in 0u32..80u32 {
                        if !emitted {
                            let rnd = hash_u32(
                                frame_seed
                                    ^ mix_index(s as u32, 0x2545_F491u32)
                                    ^ (count << 8u32)
                                    ^ nnx,
                            );
                            let jitter = ((rnd % 3u32) as i32) - 1i32;
                            let randomness = ((count as i32 + jitter + 4i32) & 7i32) as u32;
                            let base_lane = (ch * 8u32 + randomness) * 80u32;
                            let pidx = (base_lane + nnx) as usize;

                            let old = portal_ty[pidx].load();
                            if old != 0u32
                                && portal_ty[pidx].compare_exchange_weak(old, 0u32) == old
                            {
                                let emit_ty = old & 0x1FFu32;
                                if emit_ty == SPRK_U32 {
                                    // `PPTO.cpp:132-149`: a stored spark comes
                                    // back as a ring of sparks around the
                                    // portal, never as a particle.
                                    spark_ring(
                                        pmap, part_ty, part_life, part_ctype, elem_props, px, py,
                                        xres, yres,
                                    );
                                } else {
                                    let db = pidx * 8usize;
                                    let svx = f32::reinterpret(portal_data[db + 5usize]);
                                    let svy = f32::reinterpret(portal_data[db + 6usize]);
                                    let mut mask = 0u32;
                                    mask |= CR_SET_LIFE;
                                    mask |= CR_SET_CTYPE;
                                    mask |= CR_SET_TEMP;
                                    mask |= CR_SET_TMP;
                                    mask |= CR_SET_TMP2;
                                    mask |= CR_SET_DCOLOUR;
                                    if svx != 0.0f32 || svy != 0.0f32 {
                                        mask |= CR_SET_VX;
                                        mask |= CR_SET_VY;
                                    }
                                    request_create(
                                        create_queue,
                                        emit_ty,
                                        nx,
                                        ny,
                                        mask,
                                        portal_data[db] as i32,
                                        portal_data[db + 1usize],
                                        f32::reinterpret(portal_data[db + 2usize]),
                                        portal_data[db + 3usize] as i32,
                                        portal_data[db + 4usize] as i32,
                                        0i32,
                                        0i32,
                                        portal_data[db + 7usize],
                                        0u32,
                                        svx,
                                        svy,
                                    );
                                    // The capture held the donor slot; the
                                    // drain allocates a fresh one, so hand the
                                    // old slot back to the pool.
                                    let donated = old >> 9u32;
                                    if donated > 0u32 {
                                        let cand = donated - 1u32;
                                        if (cand as usize) < part_flags.len() {
                                            part_flags[cand as usize] &= !FLAG_PORTAL_HELD_U32;
                                        }
                                    }
                                }
                                emitted = true;
                            }
                        }
                    }
                }
            }
        }
    }
    portal_animate_out(part_life, part_ctype, s, frame_seed, fe);
}

/// Launch the powered group's passes for one frame.
///
/// Order: the LED latch (and its wavefront replay), the time crystal, the
/// invisible pixel, the converter, then the powered portal pair. The portal
/// pair is last because its release records have to reach the same tick's
/// creation drain, and nothing below writes the queue.
pub(crate) fn launch(bufs: &SimBuffers, dims: &PassDims, seed: u32) {
    let (slot_cubes, slot_dim) = launch_grid(dims.used_slots);
    let (pixel_cubes, pixel_dim) = launch_grid(dims.pixels);

    unsafe {
        led_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.used_slots as u32,
        );
    }

    // `LED.cpp:78-81`: `flood_prop(x, y, FIELD_LIFE, 8)` refreshes the whole
    // 4-connected run in the trigger frame. The seed scan picks up every LED
    // whose `life` the latch just wrote to 8 and the drain walks the component
    // without a per-frame round budget.
    launch_seed_scan(
        bufs,
        dims,
        FLOOD_FIELD_LIFE,
        LED_LIT_LIFE,
        LED_LIT_LIFE,
        LED_U32,
    );
    launch_drain(bufs, dims, FLOOD_MODE_LIFE, LED_U32, LED_LIT_LIFE, 0i32);

    unsafe {
        timc_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.used_slots as u32,
        );
    }

    unsafe {
        pinv_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.used_slots as u32,
        );
    }

    unsafe {
        pcon_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.used_slots as u32,
        );
    }

    // The powered portal pair: same `portalp` ring as `PRTI`/`PRTO`
    // (`portals::launch` runs after the drain, so the release has to be
    // launched here to materialise this tick).
    unsafe {
        ppti_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
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
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_portal_ty.clone(), PORTAL_CAPACITY),
            BufferArg::from_raw_parts(bufs.gpu_portal_data.clone(), PORTAL_CAPACITY * PORTAL_WORDS),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.capacity as u32,
            seed,
        );
    }

    unsafe {
        ppto_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_portal_ty.clone(), PORTAL_CAPACITY),
            BufferArg::from_raw_parts(bufs.gpu_portal_data.clone(), PORTAL_CAPACITY * PORTAL_WORDS),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            seed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gol_kernel::BUILTIN_GOL;

    /// The device mirror of `BUILTIN_GOL` must stay in step with the table
    /// `gol_kernel` owns; a hand-edited colour would otherwise desync the
    /// `PCON` -> `LIFE` conversion from the `LIFE` brush.
    #[test]
    fn gol_colour_mirror_matches_the_builtin_table() {
        assert_eq!(GOL_NGOL_U32 as usize, BUILTIN_GOL.len());
        for (index, builtin) in BUILTIN_GOL.iter().enumerate() {
            let (colour, colour2) = gol_builtin_colours(index as u32);
            assert_eq!(colour, builtin.colour, "rule {}", index);
            assert_eq!(colour2, builtin.colour2, "rule {}", index);
        }
        // Out of range: no colour, which the caller reads as "keep defaults".
        let (colour, colour2) = gol_builtin_colours(GOL_NGOL_U32);
        assert_eq!(colour, 0);
        assert_eq!(colour2, 0);
    }

    /// `LIFE`'s payload decode must match the host `create_part` arm
    /// (`buffers/particles.rs:810-824`).
    #[test]
    fn life_create_payload_matches_the_host_arm() {
        // A builtin index: the ruleset id stays the index, the colours come
        // from the table row and `tmp2` from the low state-count bits.
        let (ctype, tmp2, dcolour, tmp) = life_create_payload(19);
        assert_eq!(ctype, 19);
        assert_eq!(tmp2, 1);
        assert_eq!(dcolour, 0xFF_0000);
        assert_eq!(tmp, 0xFF_FF00);

        // A packed custom rule: 4 live states (`(rule >> 17) & 0xF == 3`),
        // no table colour.
        let rule = 0x6_1234u32;
        let (ctype, tmp2, dcolour, tmp) = life_create_payload(rule);
        assert_eq!(ctype, 0x6_1234);
        assert_eq!(tmp2, 4);
        assert_eq!(dcolour, 0);
        assert_eq!(tmp, 0);

        // The payload mask drops the "skip colour lookup" bit TPT keeps at
        // `0x200000`, exactly like the host arm.
        let (ctype, _, _, _) = life_create_payload(0x20_0000);
        assert_eq!(ctype, 0);
    }
}
