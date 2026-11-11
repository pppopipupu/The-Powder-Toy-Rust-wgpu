//! Second-set sensors: `CSNS`, `TMPS`, `GSNS`, `ECLR`.
//!
//! Ports the four element `Update` functions that read a nearby particle's
//! stored value and spark a conductor (or, for `ECLR`, erase an area):
//!
//! * `CSNS.cpp:53-161` — ctype sensor. The `tmp2` radius scan compares
//!   neighbours' `ctype` against the sensor's own (`tmp == 0`), inverted with
//!   `tmp == 2`, or serialises the last captured `ctype` into an adjacent
//!   `FILT` chain (`tmp == 1`) and deserialises a chain back onto one
//!   neighbour (`tmp == 3`). A latched sensor (`life != 0`) consumes the latch
//!   and sparks every idle conductor in its 5x5 window.
//! * `TMPS.cpp:52-228` — the same machine, but `tmp3` selects which of
//!   `tmp`/`tmp2`/`tmp3`/`tmp4` is compared (values `0`/`1` both mean `tmp`)
//!   and written back on deserialisation, and the comparison is against
//!   `temp - 273.15`.
//! * `GSNS.cpp:50-109` — gravity sensor. The trigger is the Newtonian mass of
//!   the sensor's own coarse cell against `temp - 273.15` (`tmp == 0` at or
//!   above, `tmp == 2` below); `tmp == 1` serialises `round(mass)` into a
//!   `FILT` chain and `tmp == 3` writes a chain value back into the mass grid.
//! * `ECLR.cpp:50-73` — electronic eraser. A sparked eraser (`life == 10`)
//!   kills every particle in the `tmp`-radius square except sparks and the two
//!   conductor-source elements.
//!
//! The 5x5 spark injection is the same block in `CSNS.cpp:58-80`,
//! `TMPS.cpp:57-79` and `GSNS.cpp:55-74`: every neighbour with
//! `PROP_CONDUCTS`, excluding the water family, the thermistors and `INWR`,
//! that is idle and not insulated from the sensor becomes a `SPRK` with
//! `life = 4` and `ctype = <its old type>`. It is expressed receiver-side
//! exactly like [`super::super::electronics::sensors::source_ignition_kernel`]:
//! one branch per conductor pixel, no `pmap` rewrite.
//!
//! `CSNS`'s `CtypeDraw` (`CSNS.cpp:163-174`, which ORs `PMAPID(30)` into a
//! `LIGH` power word) is a draw-time concern and stays with the render pass.
//!
//! `FILT` chain writes use the shared `0x1000_0000` tag of the sensor family
//! (`CSNS.cpp:141`, `TMPS.cpp:191`, `GSNS.cpp:92`); the deserialisers subtract
//! it back in `i32` arithmetic, matching TPT's signed `ctype`.

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
use super::super::electronics::{SENSOR_MAX_RANGE, SENSOR_SCAN};
use super::super::shared::{
    WORD_PROPERTIES, elem_word, insulated, launch_grid, pmap_at, pmap_slot,
};
use crate::buffers::SimBuffers;
use crate::gravity_kernel::{grav_mass_at, set_grav_mass_override};

/// Serialisation tag of the `FILT` chain protocol (`CSNS.cpp:141`).
const FILT_SERIAL_TAG: u32 = 0x1000_0000;
/// [`FILT_SERIAL_TAG`] in the signed arithmetic TPT's `ctype` field uses.
const FILT_SERIAL_TAG_I32: i32 = 0x1000_0000;
const _: () = assert!(FILT_SERIAL_TAG as i32 == FILT_SERIAL_TAG_I32);

// A chain write advances one pixel per step in a fixed direction, so
// `xres + yres` steps always reach the canvas edge. TPT's
// `while (TYP(r) == PT_FILT)` is bounded only by the canvas; the device loops
// use this bound, so no chain is truncated.

/// Depth of `ECLR`'s repaired `tmp` window (`ECLR.cpp:52-53` resets anything
/// outside `0..=200` to `10`).
const ECLR_MAX_RANGE: i32 = 200;

const PT_WATR_U32: u32 = 2;
const PT_METL_U32: u32 = 11;
const PT_SPRK_U32: u32 = 12;
const PT_SLTW_U32: u32 = 19;
const PT_PSCN_U32: u32 = 50;
const PT_NSCN_U32: u32 = 51;
const PT_NTCT_U32: u32 = 53;
const PT_PTCT_U32: u32 = 54;
const PT_INWR_U32: u32 = 63;
const PT_FILT_U32: u32 = 134;
const PT_NONE_U32: u32 = 0;
const PT_TMPS_U32: u32 = 208;
const PT_CSNS_U32: u32 = 213;
const PT_ECLR_U32: u32 = 221;
const PT_GSNS_U32: u32 = 228;

const _: () = assert!(PT_WATR == PT_WATR_U32);
const _: () = assert!(PT_METL == PT_METL_U32);
const _: () = assert!(PT_SPRK == PT_SPRK_U32);
const _: () = assert!(PT_SLTW == PT_SLTW_U32);
const _: () = assert!(PT_PSCN == PT_PSCN_U32);
const _: () = assert!(PT_NSCN == PT_NSCN_U32);
const _: () = assert!(PT_NTCT == PT_NTCT_U32);
const _: () = assert!(PT_PTCT == PT_PTCT_U32);
const _: () = assert!(PT_INWR == PT_INWR_U32);
const _: () = assert!(PT_FILT == PT_FILT_U32);
const _: () = assert!(PT_NONE == PT_NONE_U32);
const _: () = assert!(PT_TMPS == PT_TMPS_U32);
const _: () = assert!(PT_CSNS == PT_CSNS_U32);
const _: () = assert!(PT_ECLR == PT_ECLR_U32);
const _: () = assert!(PT_GSNS == PT_GSNS_U32);

/// Runtime-seeded `0` for a device `i32` local.
///
/// CubeCL expands `let mut x = <host literal>` to a plain integer, so the first
/// runtime assignment to it fails to compile; `anchor * 0` stays a device value
/// (same trick as `shared::seed_u32`).
#[cube]
#[allow(clippy::erasing_op)]
fn zero_i32(anchor: i32) -> i32 {
    anchor * 0i32
}

/// `(signed int)(round(v))` — `GSNS.cpp:92` uses C `round`, i.e. halves go away
/// from zero while `as i32` truncates towards zero, so bias before the cast.
#[cube]
fn round_half_away(v: f32) -> i32 {
    let mut r = v;
    if r >= 0.0f32 {
        r += 0.5f32;
    } else {
        r -= 0.5f32;
    }
    r as i32
}

/// `CSNS.cpp:58-80` / `TMPS.cpp:57-79` / `GSNS.cpp:55-74`: spark the idle
/// conductors of the 5x5 window around `(px, py)`.
///
/// The neighbour's type is TPT's `TYP(r)` read from the live slot (the map
/// word's copy goes stale when this pass kills or converts a neighbour), the
/// insulation test is `parts_avg(i, ID(r), PT_INSL) != PT_INSL`, and the
/// in-place change is `part_change_type(ID(r), ..., PT_SPRK)` — the same
/// direct field write [`super::super::electronics::sensors`] uses for
/// `TSNS` / `VSNS`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn spark_conductors_5x5(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    elem_words: &[u32],
    px: i32,
    py: i32,
    xres: u32,
    yres: u32,
) {
    for n in 0u32..25u32 {
        let rx = (n % 5u32) as i32 - 2i32;
        let ry = (n / 5u32) as i32 - 2i32;
        if rx != 0i32 || ry != 0i32 {
            let nx = px + rx;
            let ny = py + ry;
            let rv = pmap_at(pmap, xres, yres, nx, ny);
            if rv != 0u32 {
                let slot = pmap_slot(rv);
                // Reads the live slot type rather than the map word's copy:
                // TPT's map is cleared by `kill_part` and rewritten by
                // `part_change_type`, which this pass models by writing
                // `part_ty`, so the SoA type is the one the C++ reads.
                let rt = part_ty[slot];
                if !insulated(pmap, xres, yres, px, py, nx, ny) {
                    if rt != PT_NONE_U32
                        && (elem_word(elem_words, rt, WORD_PROPERTIES) & PROP_CONDUCTS) != 0u32
                        && rt != PT_WATR_U32
                        && rt != PT_SLTW_U32
                        && rt != PT_NTCT_U32
                        && rt != PT_PTCT_U32
                        && rt != PT_INWR_U32
                        && part_life[slot] == 0i32
                    {
                        part_life[slot] = 4i32;
                        part_ctype[slot] = rt;
                        part_ty[slot] = PT_SPRK_U32;
                    }
                }
            }
        }
    }
}

/// `CSNS.cpp:53-161`.
///
/// Pixel parallel: each thread owns one `pmap` pixel, so the scan reads the
/// other pixels' pre-dispatch state exactly like TPT's sequential loop does for
/// a sensor whose neighbours it has not reached yet. The latch is consumed
/// before the scan, so a condition found this frame sparks on the next one,
/// matching TPT's per-particle order.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn csns_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &[i32],
    part_tmp2: &mut [i32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_CSNS_U32 {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;

                // `CSNS.cpp:56-57`: only the upper bound is clamped; a negative
                // `tmp2` makes the scan window empty, as in TPT.
                let mut rd = part_tmp2[id];
                if rd > SENSOR_MAX_RANGE as i32 {
                    rd = SENSOR_MAX_RANGE as i32;
                    part_tmp2[id] = rd;
                }

                // `CSNS.cpp:58-80`: a latched sensor sparks, then clears the
                // latch. The scan below can arm it again for the next frame.
                if part_life[id] != 0i32 {
                    part_life[id] = 0i32;
                    spark_conductors_5x5(
                        pmap, part_ty, part_life, part_ctype, elem_words, px, py, xres, yres,
                    );
                }

                let mode = part_tmp[id];
                let self_ctype = part_ctype[id];
                let mut do_ser = false;
                let mut do_deser = false;
                let mut tmppr = zero_i32(rd);

                // `CSNS.cpp:85-86`: `rx` is the outer loop, `ry` the inner one.
                for n in 0u32..(SENSOR_SCAN * SENSOR_SCAN) {
                    let rx = (n / SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                    let ry = (n % SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                    if rx >= -rd && rx <= rd && ry >= -rd && ry <= rd && (rx != 0i32 || ry != 0i32)
                    {
                        let nx = px + rx;
                        let ny = py + ry;
                        let mut rv = pmap_at(pmap, xres, yres, nx, ny);
                        if rv == 0u32 {
                            rv = pmap_at(photons, xres, yres, nx, ny);
                        }
                        if rv != 0u32 {
                            let slot = pmap_slot(rv);
                            let rt = part_ty[slot];
                            // A slot killed earlier in the frame is an empty
                            // pixel in TPT; skip it exactly like `if (!r)`.
                            if rt != PT_NONE_U32 {
                                if mode == 1i32 {
                                    // `CSNS.cpp:97-104`: `ctype` serialisation.
                                    if rt != PT_CSNS_U32 && rt != PT_FILT_U32 {
                                        do_ser = true;
                                        tmppr = part_ctype[slot] as i32;
                                    }
                                } else if mode == 3i32 {
                                    // `CSNS.cpp:105-112`: `ctype` deserialisation.
                                    if rt == PT_FILT_U32 {
                                        do_deser = true;
                                        tmppr = part_ctype[slot] as i32;
                                    }
                                } else if mode == 2i32 {
                                    // `CSNS.cpp:113-119`: inverted comparison.
                                    if rt != PT_METL_U32 && part_ctype[slot] != self_ctype {
                                        part_life[id] = 1i32;
                                    } else {
                                        part_life[id] = 0i32;
                                    }
                                } else {
                                    // `CSNS.cpp:120-124`: normal comparison. The
                                    // latch is only ever set here (never cleared).
                                    if rt != PT_METL_U32 && part_ctype[slot] == self_ctype {
                                        part_life[id] = 1i32;
                                    }
                                }
                            }
                        }
                    }
                }

                // `CSNS.cpp:127-159`: write into the 8 adjacent `FILT` chains.
                if do_ser || do_deser {
                    for cx_i in 0u32..3u32 {
                        let rx = cx_i as i32 - 1i32;
                        // TPT's deserialisation `break` leaves only the inner
                        // `ry` row, so one neighbour per column can be written.
                        let mut wrote = false;
                        for cy_i in 0u32..3u32 {
                            let ry = cy_i as i32 - 1i32;
                            if (rx != 0i32 || ry != 0i32) && !wrote {
                                let nx = px + rx;
                                let ny = py + ry;
                                let rv = pmap_at(pmap, xres, yres, nx, ny);
                                if rv != 0u32 {
                                    let slot = pmap_slot(rv);
                                    let rty = part_ty[slot];
                                    if rty != PT_NONE_U32 {
                                        if do_ser {
                                            // `CSNS.cpp:137-148`: walk the chain the
                                            // same direction, stopping at a
                                            // non-`FILT` pixel or the canvas edge.
                                            let mut wx = nx;
                                            let mut wy = ny;
                                            let mut walking = true;
                                            for _ in 0u32..(xres + yres) {
                                                if walking {
                                                    let cv = pmap_at(pmap, xres, yres, wx, wy);
                                                    if cv != 0u32
                                                        && part_ty[pmap_slot(cv)] == PT_FILT_U32
                                                    {
                                                        part_ctype[pmap_slot(cv)] =
                                                            FILT_SERIAL_TAG + (tmppr as u32);
                                                        wx += rx;
                                                        wy += ry;
                                                    } else {
                                                        walking = false;
                                                    }
                                                }
                                            }
                                        }
                                        if do_deser {
                                            // `CSNS.cpp:150-158`: the first non-`FILT`
                                            // neighbour of the column takes the value.
                                            if rty != PT_FILT_U32 {
                                                part_ctype[slot] =
                                                    (tmppr - FILT_SERIAL_TAG_I32) as u32;
                                                wrote = true;
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
}

/// `TMPS.cpp:52-228`.
///
/// Same machine as [`csns_kernel`], with `tmp3` selecting the compared and
/// (de)serialised field. `tmp3 == 0` and `tmp3 == 1` both select `tmp`; any
/// other value outside `2..=4` makes the comparison modes do nothing and
/// serialises the initial `0`, exactly like the `tmpp` TPT leaves untouched.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tmps_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_temp: &[f32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_TMPS_U32 {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;

                // `TMPS.cpp:55-56`.
                let mut rd = part_tmp2[id];
                if rd > SENSOR_MAX_RANGE as i32 {
                    rd = SENSOR_MAX_RANGE as i32;
                    part_tmp2[id] = rd;
                }

                // `TMPS.cpp:57-79`.
                if part_life[id] != 0i32 {
                    part_life[id] = 0i32;
                    spark_conductors_5x5(
                        pmap, part_ty, part_life, part_ctype, elem_words, px, py, xres, yres,
                    );
                }

                let mode = part_tmp[id];
                let sel = part_tmp3[id];
                let threshold = part_temp[id] - 273.15f32;
                let mut do_ser = false;
                let mut do_deser = false;
                let mut tmpp = zero_i32(rd);

                // `TMPS.cpp:84-86`: `rx` outer, `ry` inner.
                for n in 0u32..(SENSOR_SCAN * SENSOR_SCAN) {
                    let rx = (n / SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                    let ry = (n % SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                    if rx >= -rd && rx <= rd && ry >= -rd && ry <= rd && (rx != 0i32 || ry != 0i32)
                    {
                        let nx = px + rx;
                        let ny = py + ry;
                        let mut rv = pmap_at(pmap, xres, yres, nx, ny);
                        if rv == 0u32 {
                            rv = pmap_at(photons, xres, yres, nx, ny);
                        }
                        if rv != 0u32 {
                            let slot = pmap_slot(rv);
                            let rt = part_ty[slot];
                            // A killed slot behaves like an empty pixel.
                            if rt != PT_NONE_U32 {
                                if mode == 1i32 {
                                    // `TMPS.cpp:96-118`.
                                    if rt != PT_TMPS_U32 && rt != PT_FILT_U32 {
                                        do_ser = true;
                                        if sel == 0i32 || sel == 1i32 {
                                            tmpp = part_tmp[slot];
                                        } else if sel == 2i32 {
                                            tmpp = part_tmp2[slot];
                                        } else if sel == 3i32 {
                                            tmpp = part_tmp3[slot];
                                        } else if sel == 4i32 {
                                            tmpp = part_tmp4[slot];
                                        }
                                    }
                                } else if mode == 3i32 {
                                    // `TMPS.cpp:119-126`.
                                    if rt == PT_FILT_U32 {
                                        do_deser = true;
                                        tmpp = part_ctype[slot] as i32;
                                    }
                                } else if mode == 2i32 {
                                    // `TMPS.cpp:127-149`: inverted comparison.
                                    if sel == 0i32 || sel == 1i32 {
                                        if rt != PT_METL_U32 && (part_tmp[slot] as f32) <= threshold
                                        {
                                            part_life[id] = 1i32;
                                        }
                                    } else if sel == 2i32 {
                                        if rt != PT_METL_U32
                                            && (part_tmp2[slot] as f32) <= threshold
                                        {
                                            part_life[id] = 1i32;
                                        }
                                    } else if sel == 3i32 {
                                        if rt != PT_METL_U32
                                            && (part_tmp3[slot] as f32) <= threshold
                                        {
                                            part_life[id] = 1i32;
                                        }
                                    } else if sel == 4i32 {
                                        if rt != PT_METL_U32
                                            && (part_tmp4[slot] as f32) <= threshold
                                        {
                                            part_life[id] = 1i32;
                                        }
                                    }
                                } else {
                                    // `TMPS.cpp:150-172`: normal comparison.
                                    if sel == 0i32 || sel == 1i32 {
                                        if rt != PT_METL_U32 && (part_tmp[slot] as f32) > threshold
                                        {
                                            part_life[id] = 1i32;
                                        }
                                    } else if sel == 2i32 {
                                        if rt != PT_METL_U32 && (part_tmp2[slot] as f32) > threshold
                                        {
                                            part_life[id] = 1i32;
                                        }
                                    } else if sel == 3i32 {
                                        if rt != PT_METL_U32 && (part_tmp3[slot] as f32) > threshold
                                        {
                                            part_life[id] = 1i32;
                                        }
                                    } else if sel == 4i32 {
                                        if rt != PT_METL_U32 && (part_tmp4[slot] as f32) > threshold
                                        {
                                            part_life[id] = 1i32;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // `TMPS.cpp:176-225`.
                if do_ser || do_deser {
                    for cx_i in 0u32..3u32 {
                        let rx = cx_i as i32 - 1i32;
                        let mut wrote = false;
                        for cy_i in 0u32..3u32 {
                            let ry = cy_i as i32 - 1i32;
                            if (rx != 0i32 || ry != 0i32) && !wrote {
                                let nx = px + rx;
                                let ny = py + ry;
                                let rv = pmap_at(pmap, xres, yres, nx, ny);
                                if rv != 0u32 {
                                    let slot = pmap_slot(rv);
                                    let rty = part_ty[slot];
                                    if rty != PT_NONE_U32 {
                                        if do_ser {
                                            let mut wx = nx;
                                            let mut wy = ny;
                                            let mut walking = true;
                                            for _ in 0u32..(xres + yres) {
                                                if walking {
                                                    let cv = pmap_at(pmap, xres, yres, wx, wy);
                                                    if cv != 0u32
                                                        && part_ty[pmap_slot(cv)] == PT_FILT_U32
                                                    {
                                                        part_ctype[pmap_slot(cv)] =
                                                            FILT_SERIAL_TAG + (tmpp as u32);
                                                        wx += rx;
                                                        wy += ry;
                                                    } else {
                                                        walking = false;
                                                    }
                                                }
                                            }
                                        }
                                        if do_deser {
                                            if rty != PT_FILT_U32 {
                                                let value = tmpp - FILT_SERIAL_TAG_I32;
                                                // `TMPS.cpp:205-220`: write the
                                                // selected field. The `break`
                                                // at :221 happens even when the
                                                // selector names no field.
                                                if sel == 0i32 || sel == 1i32 {
                                                    part_tmp[slot] = value;
                                                } else if sel == 2i32 {
                                                    part_tmp2[slot] = value;
                                                } else if sel == 3i32 {
                                                    part_tmp3[slot] = value;
                                                } else if sel == 4i32 {
                                                    part_tmp4[slot] = value;
                                                }
                                                wrote = true;
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
}

/// `GSNS.cpp:50-109`.
///
/// The trigger compares the Newtonian mass of the sensor's own coarse cell
/// against `temp - 273.15`: `tmp == 0` fires at or above it, `tmp == 2` below
/// it. `tmp == 1` writes `round(mass)` into the adjacent `FILT` chains and
/// `tmp == 3` writes a chain value back into the mass grid.
///
/// The mass grid is read from `gpu_grav_mass_prev`, the snapshot the last solve
/// consumed: TPT accumulates `gravIn.mass` from the element updates, dispatches
/// the FFT at the start of the next frame, then clears it
/// (`Simulation.cpp:3739-3743`), while this pass runs before the port's gravity
/// step that rebuilds `gpu_grav_mass`. Deserialisation publishes a one-shot
/// override for the next solve, which is TPT's direct assignment (the `set`
/// flag lets a forced `0` survive the "unset means no override" test).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gsns_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &[i32],
    part_temp: &[f32],
    elem_words: &[u32],
    grav_mass_prev: &[Atomic<i32>],
    grav_override: &mut [i32],
    grav_override_set: &mut [u32],
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_GSNS_U32 {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                // `GSNS.cpp:53`: `gravIn.mass[Vec2{x, y} / CELL]`.
                let cell = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let mass = grav_mass_at(grav_mass_prev, cell);
                let mode = part_tmp[id];
                let threshold = part_temp[id] - 273.15f32;

                // `GSNS.cpp:53`: `(tmp == 0 && mass >= temp - 273.15)` or
                // `(tmp == 2 && mass < temp - 273.15)`.
                let mut trigger = false;
                if mode == 0i32 && mass >= threshold {
                    trigger = true;
                }
                if mode == 2i32 && mass < threshold {
                    trigger = true;
                }

                // `GSNS.cpp:55-74`.
                if trigger {
                    part_life[id] = 0i32;
                    spark_conductors_5x5(
                        pmap, part_ty, part_life, part_ctype, elem_words, px, py, xres, yres,
                    );
                }

                // `GSNS.cpp:75-107`: the 8 neighbours.
                let serial = round_half_away(mass);
                for n in 0u32..9u32 {
                    let rx = (n % 3u32) as i32 - 1i32;
                    let ry = (n / 3u32) as i32 - 1i32;
                    if rx != 0i32 || ry != 0i32 {
                        let nx = px + rx;
                        let ny = py + ry;
                        let mut rv = pmap_at(pmap, xres, yres, nx, ny);
                        if rv == 0u32 {
                            rv = pmap_at(photons, xres, yres, nx, ny);
                        }
                        if rv != 0u32 {
                            if mode == 1i32 {
                                // `GSNS.cpp:84-99`: chain write of `round(mass)`
                                // from the sensor's own cell.
                                let mut wx = nx;
                                let mut wy = ny;
                                let mut walking = true;
                                for _ in 0u32..(xres + yres) {
                                    if walking {
                                        let cv = pmap_at(pmap, xres, yres, wx, wy);
                                        if cv != 0u32 && part_ty[pmap_slot(cv)] == PT_FILT_U32 {
                                            part_ctype[pmap_slot(cv)] =
                                                FILT_SERIAL_TAG + (serial as u32);
                                            wx += rx;
                                            wy += ry;
                                        } else {
                                            walking = false;
                                        }
                                    }
                                }
                            } else if mode == 3i32 {
                                // `GSNS.cpp:100-106`.
                                if part_ty[pmap_slot(rv)] == PT_FILT_U32 {
                                    let value =
                                        part_ctype[pmap_slot(rv)] as i32 - FILT_SERIAL_TAG_I32;
                                    set_grav_mass_override(
                                        grav_override,
                                        grav_override_set,
                                        cell,
                                        value as f32,
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

/// `ECLR.cpp:50-73`: the sparked electronic eraser.
///
/// `tmp` is the eraser radius and is repaired to `10` outside `0..=200`; TPT
/// snapshots `range` before that repair, so the repairing frame still scans the
/// stale radius. The device loop clamps the bound to the repair window — every
/// in-range radius is exact, while a hostile `tmp = i32::MAX` cannot stall the
/// dispatch.
///
/// Every particle in the square except `SPRK`, `PSCN` and `NSCN` is killed
/// (matter first, `photons` when the pixel map is empty), i.e. its slot is
/// marked `PT_NONE` exactly like the other reaction kills; the map word itself
/// is left for the pool rebuild, as everywhere else in this pass.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn eclr_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &[i32],
    part_tmp: &mut [i32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_ECLR_U32 {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;

                // `ECLR.cpp:51-53`: `range` is read before the repair.
                let raw = part_tmp[id];
                if raw > ECLR_MAX_RANGE || raw < 0i32 {
                    part_tmp[id] = 10i32;
                }
                let mut range = raw;
                if range > ECLR_MAX_RANGE {
                    range = ECLR_MAX_RANGE;
                }
                if range < 0i32 {
                    range = 0i32;
                }

                if part_life[id] == 10i32 {
                    let span = 2u32 * (range as u32) + 1u32;
                    for oy in 0u32..span {
                        let ry = oy as i32 - range;
                        for ox in 0u32..span {
                            let rx = ox as i32 - range;
                            if rx != 0i32 || ry != 0i32 {
                                let nx = px + rx;
                                let ny = py + ry;
                                let mut rv = pmap_at(pmap, xres, yres, nx, ny);
                                if rv == 0u32 {
                                    rv = pmap_at(photons, xres, yres, nx, ny);
                                }
                                if rv != 0u32 {
                                    let slot = pmap_slot(rv);
                                    let nty = part_ty[slot];
                                    // `ECLR.cpp:65-69`: `kill_part(ID(r))`.
                                    if nty != PT_SPRK_U32
                                        && nty != PT_PSCN_U32
                                        && nty != PT_NSCN_U32
                                    {
                                        part_ty[slot] = PT_NONE_U32;
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

/// Launch the four sensor kernels for one frame.
///
/// They are pixel parallel over `pmap` and independent of each other, so one
/// launch group covers them all. `GSNS` additionally binds the gravity mass
/// snapshot and the one-shot override pair.
pub(crate) fn launch(bufs: &SimBuffers, dims: &PassDims, _seed: u32) {
    if dims.pixels == 0 || dims.capacity == 0 || dims.elem_len == 0 {
        return;
    }
    if bufs.pmap.len() < dims.pixels
        || bufs.photons.len() < dims.pixels
        || bufs.part_ty.len() < dims.capacity
        || bufs.part_life.len() < dims.capacity
        || bufs.part_ctype.len() < dims.capacity
        || bufs.part_tmp.len() < dims.capacity
        || bufs.part_tmp2.len() < dims.capacity
        || bufs.part_tmp3.len() < dims.capacity
        || bufs.part_tmp4.len() < dims.capacity
        || bufs.part_temp.len() < dims.capacity
    {
        return;
    }

    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        csns_kernel::launch_unchecked(
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

    unsafe {
        tmps_kernel::launch_unchecked(
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
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
        );
    }

    unsafe {
        eclr_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            dims.xres,
            dims.yres,
        );
    }

    // The override pair is device-only (no host shadow); its extent is
    // `dims.cells`, the same grid the mass snapshot uses.
    if dims.cells > 0 && dims.xcells > 0 && bufs.grav_mass_prev.len() >= dims.cells {
        unsafe {
            gsns_kernel::launch_unchecked(
                &bufs.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(dim),
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
                BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
                BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
                BufferArg::from_raw_parts(bufs.gpu_grav_mass_prev.clone(), dims.cells),
                BufferArg::from_raw_parts(bufs.gpu_grav_mass_override.clone(), dims.cells),
                BufferArg::from_raw_parts(bufs.gpu_grav_mass_override_set.clone(), dims.cells),
                dims.xres,
                dims.yres,
                dims.xcells,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filt_tag_round_trips_signed_values() {
        for value in [0i32, 1, 4, 10, 255, 1000, -5, -0x1000_0000] {
            let packed = FILT_SERIAL_TAG.wrapping_add(value as u32);
            let recovered = packed as i32 - FILT_SERIAL_TAG_I32;
            assert_eq!(recovered, value);
        }
    }

    /// `GSNS.cpp:92`'s `round` on the values the mass grid can hold.
    #[test]
    fn gsns_rounding_is_half_away_from_zero() {
        fn host_round(v: f32) -> i32 {
            let mut r = v;
            if r >= 0.0 {
                r += 0.5;
            } else {
                r -= 0.5;
            }
            r as i32
        }
        assert_eq!(host_round(0.4), 0);
        assert_eq!(host_round(0.5), 1);
        assert_eq!(host_round(1.5), 2);
        assert_eq!(host_round(-0.5), -1);
        assert_eq!(host_round(-1.5), -2);
        assert_eq!(host_round(51.2), 51);
    }

    /// `ECLR.cpp:51-53` repairs `tmp` outside `0..=200` to `10`.
    #[test]
    fn eclr_repairs_out_of_range_tmp() {
        fn repaired(tmp: i32) -> i32 {
            if tmp > ECLR_MAX_RANGE || tmp < 0 {
                10
            } else {
                tmp
            }
        }
        assert_eq!(repaired(-1), 10);
        assert_eq!(repaired(0), 0);
        assert_eq!(repaired(10), 10);
        assert_eq!(repaired(200), 200);
        assert_eq!(repaired(201), 10);
        assert_eq!(repaired(i32::MAX), 10);
    }

    /// `CSNS.cpp:56-57` / `TMPS.cpp:55-56` cap the scan radius at 25.
    #[test]
    fn sensor_scan_radius_caps_at_25() {
        fn cap(rd: i32) -> i32 {
            if rd > SENSOR_MAX_RANGE as i32 {
                SENSOR_MAX_RANGE as i32
            } else {
                rd
            }
        }
        assert_eq!(cap(2), 2);
        assert_eq!(cap(25), 25);
        assert_eq!(cap(26), 25);
        assert_eq!(cap(1000), 25);
        assert_eq!(SENSOR_SCAN, 51);
    }
}
