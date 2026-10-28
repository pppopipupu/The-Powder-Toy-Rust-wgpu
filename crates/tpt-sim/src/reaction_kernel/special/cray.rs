//! `CRAY` laser beam emission (`CRAY.cpp:84-140`).
//!
//! The `ctype` latch (which element the laser fires) lives in
//! [`super::transform`]; this pass is the beam itself: **every** neighbour
//! spark with `life == 3` (`CRAY.cpp:88-100`) casts its own straight line of
//! the stored element away from itself, one particle per step, up to `tmp`
//! creations (255 when `tmp` is 0) after skipping `tmp2` cells. The 3x3 scan
//! and the march are one loop: a spark writes its direction code into `tmp4`
//! and the march runs immediately, so the beams are cast in the same
//! row-major order and with the same shared map as TPT.
//!
//! TPT keeps the beam's running state (the remaining shots and the colour
//! picked up from a `FILT`) in local variables of its per-particle loop. A
//! CubeCL kernel cannot mutate a local across a nested loop (see the kernel
//! rules in `AGENTS.md`), so this pass keeps that state in the laser's own
//! scratch fields instead: `tmp4` holds the shot count under the direction
//! code (and is rewritten for every beam), and `tmp3` holds the `dcolour` the
//! beam applies after passing a filter (reset at the start of each beam).
//! Both are rewritten on every beam, so nothing leaks between beams or frames.
//!
//! Creations made by two beams aimed at the same pixel are resolved by the
//! drain in queue order: the first record occupies the pixel and the second
//! fails like a `create_part` that returned `-1`. TPT, by contrast, sees a
//! beam's creations in `pmap` immediately, so a later beam of the same frame
//! can stop on them. The queue is drained once per frame, so that
//! intra-frame feedback does not exist here. The queue call is also
//! fire-and-forget: a beam spends its shot budget even when the shared queue
//! (16384 records for the whole step) is full, while TPT only decrements
//! `partsRemaining` after a successful `create_part`.
//!
//! Beam special cases, straight from TPT:
//!
//! * a `FILT` in the path colours every particle created after it (a filter
//!   with `dcolour == 0xFF000000` makes the beam black, one with `tmp != 0`
//!   leaves the colour alone) and flashes `life = 4`;
//! * `PSCN` sparks destroy what they hit instead of creating (`DMND` survives);
//! * `INST` sparks keep going through occupied cells;
//! * `INWR` sparks create on top of what is in the way.
//!
//! Creations go through the device queue, which the reaction driver drains
//! after this pass, so the beam is live for the movement and render passes of
//! the same frame.

use cubecl::prelude::*;
use tpt_core::constants::*;

use crate::move_kernel::wall_blocks_particle;
use crate::reaction_kernel::shared::{
    CR_SET_DCOLOUR, CR_SET_LIFE, CR_SET_TEMP, WORD_PROPERTIES, elem_word, pmap_at, pmap_slot,
    request_create,
};
use crate::render_kernel::particles::elements_basic::{filt_wavelengths, photon_graphics};

/// Bits of `tmp4` that hold the remaining shot count; the direction code sits
/// above them.
const CRAY_COUNT_MASK: i32 = 0xFFFF;

/// Upper bound of the beam march. The march is a straight line, so it always
/// leaves the canvas within `xres + yres` cells; a constant bound keeps the
/// device loop counter in range (a runtime bound also costs the loop's phi
/// nodes a pointer argument, which the CubeCL verifier rejects once a creation
/// sits inside the loop's branch).
const CRAY_MAX_STEPS: u32 = 4096;

/// `wavelengthToDecoColour` (`CRAY.cpp:143-170`): the packed `0xAARRGGBB`
/// decoration colour for a 12+12+12-bit wavelength mask.
///
/// The colour maths is `PHOT::graphics`, which TPT duplicates verbatim in
/// `CRAY.cpp`; the port reuses the single implementation.
#[cube]
fn wavelength_deco_colour(wavelength: u32) -> u32 {
    let (r, g, b) = photon_graphics(wavelength);
    (255u32 << 24u32)
        | (u32::cast_from(r) << 16u32)
        | (u32::cast_from(g) << 8u32)
        | u32::cast_from(b)
}

/// `CRAY.cpp:84-140`: cast the beam from a sparking laser.
#[cube(launch_unchecked)]
pub(super) fn cray_beam_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &[i32],
    part_tmp2: &[i32],
    part_tmp3: &mut [u32],
    part_tmp4: &mut [i32],
    part_temp: &[f32],
    part_dcolour: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    elem_words: &[u32],
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
            let emit = part_ctype[id];
            if part_ty[id] == PT_CRAY && emit != 0u32 && emit < PT_NUM as u32 {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;

                // `CRAY.cpp:86-98`: **every** neighbouring spark with
                // `life == 3` fires its own beam away from itself, in the
                // row-major order of the 3x3 scan. The scan stores the
                // direction code (`n + 1`) and the shot budget in `tmp4`; a
                // laser with no spark resets it to 0. The march below runs
                // once per qualifying neighbour and checks the code it just
                // wrote, so beams are cast in TPT's order.
                part_tmp4[id] = 0i32;
                for n in 0u32..9u32 {
                    let rx = (n % 3u32) as i32 - 1i32;
                    let ry = (n / 3u32) as i32 - 1i32;
                    if rx != 0i32 || ry != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            if part_ty[nid] == PT_SPRK && part_life[nid] == 3i32 {
                                let tmp = part_tmp[id];
                                let limit = if tmp > 0i32 { tmp } else { 255i32 };
                                part_tmp4[id] = (((n as i32) + 1i32) << 16i32) | limit;
                                part_tmp3[id] = 0u32;
                            }
                        }
                    }

                    if (part_tmp4[id] >> 16i32) == (n as i32) + 1i32 {
                        let code = part_tmp4[id] >> 16i32;
                        let rx = ((code - 1i32) % 3i32) - 1i32;
                        let ry = ((code - 1i32) / 3i32) - 1i32;
                        let nxi = -rx;
                        let nyi = -ry;
                        let spaces = part_tmp2[id];
                        let spark_ctype =
                            part_ctype[pmap_slot(pmap_at(pmap, xres, yres, px + rx, py + ry))];
                        let destroy = spark_ctype == PT_PSCN;
                        let nostop = spark_ctype == PT_INST;
                        let create_spark = spark_ctype == PT_INWR;
                        let emit_props = elem_word(elem_words, emit, WORD_PROPERTIES);
                        let life = part_life[id];

                        // The march is bounded by the canvas diagonal; TPT relies on
                        // the bounds check alone, but a fixed upper bound keeps the
                        // device loop counter in range.
                        for step in 0u32..CRAY_MAX_STEPS {
                            let offset = 1i32 + spaces + step as i32;
                            let bx = px + nxi * offset;
                            let by = py + nyi * offset;
                            if bx < 0i32 || by < 0i32 || bx >= xres as i32 || by >= yres as i32 {
                                break;
                            }
                            let ux = bx as u32;
                            let uy = by as u32;
                            let bi = (uy * xres + ux) as usize;
                            let cell = ((uy / 4u32) * xcells + (ux / 4u32)) as usize;
                            let wall = wall_bmap[cell] & 0xFFu32;
                            let emap = wall_emap[cell] & 0xFFu32;
                            let blocked =
                                wall != 0u32 && wall_blocks_particle(wall, emap, emit_props);
                            let r = pmap[bi];
                            let rt = r & 0x1FFu32;
                            // TPT's four-way branch, resolved before any side
                            // effect so the loop body stays a list of plain `if`
                            // blocks: 0 stop, 1 create, 2 `FILT`, 3 pass through,
                            // 4 `PSCN` destroy.
                            let create_ok = !blocked && (r == 0u32 || create_spark);
                            let action = if create_ok {
                                1i32
                            } else if rt == PT_FILT {
                                2i32
                            } else if rt == PT_CRAY || nostop {
                                3i32
                            } else if destroy && r != 0u32 && rt != PT_DMND {
                                4i32
                            } else {
                                0i32
                            };
                            if action == 0i32 {
                                break;
                            }
                            if action == 2i32 {
                                // `CRAY.cpp:126-136`: a filter recolours the rest of
                                // the beam. A black-decorated filter blacks it out, a
                                // filter with `tmp != 0` (its "filter out" mode)
                                // leaves the colour alone and clears a previous
                                // black; the filter itself flashes `life = 4`.
                                let fid = pmap_slot(r);
                                if part_dcolour[fid] == 0xFF00_0000u32 {
                                    part_tmp3[id] = 0xFF00_0000u32;
                                } else if part_tmp[fid] == 0i32 {
                                    part_tmp3[id] = wavelength_deco_colour(filt_wavelengths(
                                        part_ctype[fid],
                                        part_temp[fid],
                                    ));
                                } else if part_tmp3[id] == 0xFF00_0000u32 {
                                    part_tmp3[id] = 0u32;
                                }
                                part_life[fid] = 4i32;
                            }
                            if action == 4i32 {
                                part_ty[pmap_slot(r)] = PT_NONE;
                            }
                            let colored = part_tmp3[id];
                            let mask = CR_SET_TEMP
                                + (if colored != 0u32 {
                                    CR_SET_DCOLOUR
                                } else {
                                    0u32
                                })
                                + (if life > 0i32 { CR_SET_LIFE } else { 0u32 });
                            // `request_create` keeps a device-side atomic, so it
                            // must sit at the loop's top level: wrapped in a branch
                            // CubeCL gives the atomic's pointer phi node an integer
                            // argument and the verifier rejects the kernel. A type
                            // of 0 marks a record the drain discards, which only
                            // happens on the rare pass-through cells (`FILT`,
                            // `CRAY`, `INST`) — a stop cell breaks above.
                            let ty = if action == 1i32 { emit } else { 0u32 };
                            request_create(
                                create_queue,
                                ty,
                                bx,
                                by,
                                mask,
                                life,
                                0u32,
                                part_temp[id],
                                0i32,
                                0i32,
                                0i32,
                                0i32,
                                colored,
                                0u32,
                                0.0f32,
                                0.0f32,
                            );
                            if action == 1i32 || action == 4i32 {
                                part_tmp4[id] -= 1i32;
                                if (part_tmp4[id] & CRAY_COUNT_MASK) <= 0i32 {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
