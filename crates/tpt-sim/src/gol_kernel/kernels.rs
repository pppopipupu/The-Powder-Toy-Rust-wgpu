//! The device stages of a generation: [`gol_analyze_kernel`] stages the map,
//! [`gol_update_kernel`] applies it, [`gol_birth_kernel`] turns the surviving
//! birth votes into creation requests and [`gol_settle_kernel`] folds the
//! survivor count into the device arm word.
//!
//! All three per-pixel stages read `counters[0]` first and terminate when it is
//! clear, so a canvas without `LIFE` pays one atomic load per thread instead of
//! a host round trip.

use cubecl::prelude::*;

use crate::reaction_kernel::shared::{
    CR_SET_CTYPE, CR_SET_DCOLOUR, CR_SET_TMP, CR_SET_TMP2, request_create,
};

use super::{
    CELL_U32, META_BORDER, META_COUNT_MASK, META_HAS_BIRTH, META_OWN_LIFE, META_PAYLOAD_MASK,
    META_PAYLOAD_SHIFT, PMAP_TYPE_MASK, PT_LIFE_U32, PT_NONE_U32, WL_STASIS_U32,
};

/// `NGOL` mirrored for kernel bodies (rule 1).
const NGOL_U32: u32 = 24;
const _: () = assert!(NGOL_U32 as usize == super::NGOL);

/// `GOL_RULE_MASK` mirrored for kernel bodies.
const GOL_RULE_MASK_U32: u32 = 0x001F_FFFF;
const _: () = assert!(GOL_RULE_MASK_U32 == super::GOL_RULE_MASK);

/// Wraps a coordinate onto the Life torus (`SimulateGoL`'s modulo arithmetic).
///
/// Callers only pass coordinates `>= 0`, so the unsigned remainder is the
/// mathematical modulo and the result is always inside
/// `CELL .. extent - CELL`.
#[cube]
fn gol_wrap(coord: i32, extent: u32) -> u32 {
    let span = extent - 2u32 * CELL_U32;
    (((coord as u32) + extent - 3u32 * CELL_U32) % span) + CELL_U32
}

/// Pixel index of direction `d` (`0..9`; `4` is the centre, `d % 3` is dx and
/// `d / 3` is dy, both biased by one).
#[cube]
fn gol_neighbour_index(x: u32, y: u32, d: u32, xres: u32, yres: u32) -> u32 {
    let dx = (d % 3u32) as i32 - 1i32;
    let dy = (d / 3u32) as i32 - 1i32;
    gol_wrap(x as i32 + dx, xres) + gol_wrap(y as i32 + dy, yres) * xres
}

/// Resolves a `ctype` (builtin index or packed ruleset) to a ruleset.
///
/// CubeCL only accepts `|=`/`+=` style assignments on host-seeded locals, so
/// the table read is accumulated into a zero seed instead of assigned.
#[cube]
fn gol_resolve(ctype: u32, rules: &[u32]) -> u32 {
    let mut rule = 0u32;
    if ctype < NGOL_U32 {
        rule |= rules[ctype as usize];
    } else {
        rule |= ctype;
    }
    rule & GOL_RULE_MASK_U32
}

/// `ctype + 1` of a *live* `LIFE` neighbour, or `0` when there is none.
///
/// The `+ 1` bias keeps a valid `ctype` of `0` (the builtin GOL rule)
/// distinguishable from "no neighbour" without a mutable sentinel.
///
/// `SimulateGoL` only lets a cell with `tmp2 == states` contribute to its
/// neighbours' population lists, which is what this guard reproduces. A cell
/// whose `tmp2` is still zero (created without the create hook) is *dying* in
/// the reference, not live: the update stage advances it to `-1` and kills it.
#[cube]
fn gol_live_neighbour(
    pmap: &[u32],
    part_ctype: &[u32],
    part_tmp2: &[i32],
    rules: &[u32],
    nidx: u32,
    capacity: u32,
) -> u32 {
    let mut found = 0u32;
    let entry = pmap[nidx as usize];
    if entry != 0u32 && (entry & PMAP_TYPE_MASK) == PT_LIFE_U32 {
        let slot = (entry >> 9u32) - 1u32;
        if slot < capacity {
            let ctype = part_ctype[slot as usize];
            let rule = gol_resolve(ctype, rules);
            let max_state = (((rule >> 17u32) & 0xFu32) + 1u32) as i32;
            let t2 = part_tmp2[slot as usize];
            if t2 == max_state {
                found += ctype + 1u32;
            }
        }
    }
    found
}

/// Stage 1: the staged read of the current map.
///
/// For every pixel this writes one `gol_meta` word:
///
/// * a pixel of the play area, empty or holding `LIFE`, carries its number of
///   live `LIFE` neighbours (`0..=8`) in the low nibble,
/// * `META_OWN_LIFE` + the particle slot for a pixel holding `LIFE`
///   (`META_BORDER`, and no count, when it sits in the inert border ring),
/// * `META_HAS_BIRTH` + the winning `ctype + 1` for an empty pixel that should
///   be born into (`SimulateGoL`'s majority vote, preferring the lowest
///   `ctype`); the count is consumed by the vote and not kept,
/// * `0` for an empty or non-`LIFE` pixel that cannot change.
///
/// Work item `0` clears the survivor counter the update stage fills in. The two
/// dispatches are ordered, so the clear happens before any `fetch_add`; the
/// arm word is *not* cleared here because every thread reads it to decide
/// whether to run at all.
///
/// The vote accumulates its winner in the pixel's own word instead of a local:
/// a local declared outside these loops loses its address when it is assigned
/// from this depth, and CubeCL emits a store through the initialiser constant
/// (`memory.store (0xFFFFFFFF, ...)`), which fails IR verification and poisons
/// every later buffer read.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn gol_analyze_kernel(
    pmap: &[u32],
    part_ctype: &[u32],
    part_tmp2: &[i32],
    gol_rules: &[u32],
    gol_meta: &mut [u32],
    counters: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    capacity: u32,
    num_pixels: u32,
) {
    let i = ABSOLUTE_POS;
    if i == 0usize {
        counters[1].store(0u32);
    }
    // `counters[0]` is the device arm word: the host sets it on a paint or an
    // edit, and `gol_settle_kernel` keeps it set while the last generation left
    // a survivor or a birth. When it is clear the whole pass is a no-op.
    if counters[0].load() == 0u32 {
        terminate!();
    }
    if i >= num_pixels as usize {
        terminate!();
    }

    let idx = i as u32;
    let x = idx % xres;
    let y = idx / xres;
    let entry = pmap[i];
    let inside = x >= CELL_U32 && y >= CELL_U32 && x < xres - CELL_U32 && y < yres - CELL_U32;

    let is_life = entry != 0u32 && (entry & PMAP_TYPE_MASK) == PT_LIFE_U32;
    if inside && (is_life || entry == 0u32) {
        // Live `LIFE` neighbours, for a cell of the play area. `SimulateGoL`
        // counts them for occupied and empty cells alike, because a cell that
        // has neighbours also has to know whether it survives.
        let mut neighbours = 0u32;
        for d in 0u32..9u32 {
            if d != 4u32 {
                let nidx = gol_neighbour_index(x, y, d, xres, yres);
                if gol_live_neighbour(pmap, part_ctype, part_tmp2, gol_rules, nidx, capacity)
                    != 0u32
                {
                    neighbours += 1u32;
                }
            }
        }
        if is_life {
            let slot = (entry >> 9u32) - 1u32;
            gol_meta[i] =
                META_OWN_LIFE | ((slot & META_PAYLOAD_MASK) << META_PAYLOAD_SHIFT) | neighbours;
        } else {
            gol_meta[i] = neighbours;
            if neighbours > 0u32 {
                // `SimulateGoL`'s majority vote: the smallest ruleset among those
                // whose `B` bit for this population passes and that hold at least
                // `ceil(neighbours / 2)` cells.
                //
                // The winner is accumulated in this pixel's own word rather than a
                // local: CubeCL resolves a local declared outside these loops to
                // its initialiser when it is assigned from this depth, which emits
                // a store through a constant. Each invocation only ever touches its
                // own word, so the read-modify-write is race free.
                let majority = neighbours / 2u32 + neighbours % 2u32;
                for d in 0u32..9u32 {
                    if d != 4u32 {
                        let nidx = gol_neighbour_index(x, y, d, xres, yres);
                        let nctype = gol_live_neighbour(
                            pmap, part_ctype, part_tmp2, gol_rules, nidx, capacity,
                        );
                        if nctype != 0u32 {
                            // Only the first cell of each ruleset in the
                            // neighbourhood proposes that ruleset, like the
                            // population list `SimulateGoL` builds.
                            let mut seen = false;
                            for e in 0u32..d {
                                if e != 4u32 {
                                    let eidx = gol_neighbour_index(x, y, e, xres, yres);
                                    let ectype = gol_live_neighbour(
                                        pmap, part_ctype, part_tmp2, gol_rules, eidx, capacity,
                                    );
                                    if ectype == nctype {
                                        seen = true;
                                    }
                                }
                            }
                            if !seen {
                                let rule = gol_resolve(nctype - 1u32, gol_rules);
                                if ((rule >> (neighbours + 8u32)) & 1u32) != 0u32 {
                                    let mut population = 0u32;
                                    for f in 0u32..9u32 {
                                        if f != 4u32 {
                                            let fidx = gol_neighbour_index(x, y, f, xres, yres);
                                            let fctype = gol_live_neighbour(
                                                pmap, part_ctype, part_tmp2, gol_rules, fidx,
                                                capacity,
                                            );
                                            if fctype == nctype {
                                                population += 1u32;
                                            }
                                        }
                                    }
                                    // The payload holds the winner's `ctype + 1`
                                    // so that zero still means "none".
                                    let best =
                                        (gol_meta[i] >> META_PAYLOAD_SHIFT) & META_PAYLOAD_MASK;
                                    if population >= majority && (best == 0u32 || nctype < best) {
                                        gol_meta[i] = META_HAS_BIRTH
                                            | ((nctype & META_PAYLOAD_MASK) << META_PAYLOAD_SHIFT);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if is_life {
        // A `LIFE` pixel in the inert border ring: it never evolves but still
        // keeps the pass armed.
        let slot = (entry >> 9u32) - 1u32;
        gol_meta[i] =
            META_OWN_LIFE | META_BORDER | ((slot & META_PAYLOAD_MASK) << META_PAYLOAD_SHIFT);
    } else {
        // Occupied by something other than `LIFE`, or outside the play area.
        gol_meta[i] = 0u32;
    }
}

/// Stage 2: apply one generation from the staged analysis.
///
/// No thread reads another thread's output: every decision comes from
/// `gol_meta`, so the pass is race free even though it kills particles. A
/// birth vote on a stasis-frozen cell is cleared from `gol_meta` so the birth
/// stage does not need the wall maps.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn gol_update_kernel(
    part_ty: &mut [u32],
    part_tmp2: &mut [i32],
    part_ctype: &[u32],
    gol_rules: &[u32],
    gol_meta: &mut [u32],
    counters: &mut [Atomic<u32>],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    xres: u32,
    xcells: u32,
    capacity: u32,
    num_pixels: u32,
) {
    if counters[0].load() == 0u32 {
        terminate!();
    }
    let i = ABSOLUTE_POS;
    if i >= num_pixels as usize {
        terminate!();
    }
    let meta = gol_meta[i];
    if meta == 0u32 {
        terminate!();
    }

    let idx = i as u32;
    let x = idx % xres;
    let y = idx / xres;
    let cell = ((y / CELL_U32) * xcells + (x / CELL_U32)) as usize;
    // `WL_STASIS` freezes the cell while its electrical state is below 8.
    let frozen = wall_bmap[cell] == WL_STASIS_U32 && wall_emap[cell] < 8u32;

    let neighbours = meta & META_COUNT_MASK;
    let payload = (meta >> META_PAYLOAD_SHIFT) & META_PAYLOAD_MASK;

    if (meta & META_OWN_LIFE) != 0u32 {
        let slot = payload as usize;
        if payload < capacity && part_ty[slot] == PT_LIFE_U32 {
            if (meta & META_BORDER) != 0u32 {
                // Border cells never evolve but still keep the pass alive.
                counters[1].fetch_add(1u32);
            } else {
                let rule = gol_resolve(part_ctype[slot], gol_rules);
                let max_state = (((rule >> 17u32) & 0xFu32) + 1u32) as i32;
                // `SimulateGoL:3462` only treats a cell with
                // `tmp2 == states` as live. A cell created without the create
                // hook has `tmp2 == 0` and is already dying, so the reference
                // advances it to `-1` and kills it (`Simulation.cpp:3505-3511`,
                // `:3595-3603`); no revive branch here.
                let mut t2 = 0i32;
                t2 += part_tmp2[slot];
                if t2 < max_state {
                    // A dying cell advances one state per generation (unless a
                    // stasis wall freezes it).
                    let mut next = 0i32;
                    next += t2;
                    if !frozen {
                        next -= 1i32;
                    }
                    part_tmp2[slot] = next;
                    if next <= 0i32 {
                        part_ty[slot] = PT_NONE_U32;
                    } else {
                        counters[1].fetch_add(1u32);
                    }
                } else if t2 == max_state && !frozen && ((rule >> neighbours) & 1u32) == 0u32 {
                    // Start the death sequence.
                    let next = max_state - 1i32;
                    part_tmp2[slot] = next;
                    if next <= 0i32 {
                        part_ty[slot] = PT_NONE_U32;
                    } else {
                        counters[1].fetch_add(1u32);
                    }
                } else {
                    counters[1].fetch_add(1u32);
                }
            }
        }
    } else if (meta & META_HAS_BIRTH) != 0u32 && frozen {
        // A stasis wall suspends the whole cell, births included.
        gol_meta[i] = meta - META_HAS_BIRTH;
    }
}

/// Stage 3: turn every surviving birth vote into a creation request.
///
/// The record carries TPT's `create_part` overrides for a `LIFE` cell
/// (`ctype`, `tmp2`, `dcolour`, `tmp`); the drain fills the rest from the
/// element's `DefaultProperties` and the `LIFE::create` hook values, exactly as
/// a host `create_part` would. The first live neighbour of the same ruleset
/// provides the colours, with the builtin menu colours as the fallback.
///
/// The sample slot is accumulated in the pixel's own `gol_meta` word for the
/// same reason the vote is (see [`gol_analyze_kernel`]); the birth flag has
/// already been consumed at that point.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn gol_birth_kernel(
    pmap: &[u32],
    part_ctype: &[u32],
    part_tmp: &[i32],
    part_dcolour: &[u32],
    gol_rules: &[u32],
    gol_meta: &mut [u32],
    create_queue: &mut [Atomic<u32>],
    counters: &mut [Atomic<u32>],
    capacity: u32,
    xres: u32,
    yres: u32,
    num_pixels: u32,
) {
    if counters[0].load() == 0u32 {
        terminate!();
    }
    let i = ABSOLUTE_POS;
    if i >= num_pixels as usize {
        terminate!();
    }
    let meta = gol_meta[i];
    if (meta & META_HAS_BIRTH) == 0u32 {
        terminate!();
    }
    // The analysis stores the winner's `ctype + 1`; the host path masked the
    // decoded value with `GOL_RULE_MASK` before stamping it on the particle.
    let payload = (meta >> META_PAYLOAD_SHIFT) & META_PAYLOAD_MASK;
    let ctype = (payload - 1u32) & GOL_RULE_MASK_U32;
    let idx = i as u32;
    let x = idx % xres;
    let y = idx / xres;

    gol_meta[i] = 0u32;
    for d in 0u32..9u32 {
        if d != 4u32 {
            let nidx = gol_neighbour_index(x, y, d, xres, yres);
            let entry = pmap[nidx as usize];
            if gol_meta[i] == 0u32 && entry != 0u32 && (entry & PMAP_TYPE_MASK) == PT_LIFE_U32 {
                let slot = (entry >> 9u32) - 1u32;
                if slot < capacity && part_ctype[slot as usize] == ctype {
                    gol_meta[i] = slot + 1u32;
                }
            }
        }
    }
    let sample = gol_meta[i];
    let rule = gol_resolve(ctype, gol_rules);
    let max_state = (((rule >> 17u32) & 0xFu32) + 1u32) as i32;
    // `LIFE.cpp:93-111`: a builtin rule carries the menu colours; a custom
    // ruleset keeps the colours of the sampled neighbour or zero.
    let mut colour1 = 0u32;
    let mut colour2 = 0i32;
    if sample != 0u32 {
        let slot = (sample - 1u32) as usize;
        colour1 = part_dcolour[slot];
        colour2 = part_tmp[slot];
    } else if ctype < NGOL_U32 {
        let base = (NGOL_U32 + ctype * 2u32) as usize;
        colour1 = gol_rules[base];
        colour2 = gol_rules[base + 1usize] as i32;
    }

    request_create(
        create_queue,
        PT_LIFE_U32,
        x as i32,
        y as i32,
        CR_SET_CTYPE | CR_SET_TMP | CR_SET_TMP2 | CR_SET_DCOLOUR,
        0i32,
        ctype,
        0.0f32,
        colour2,
        max_state,
        0i32,
        0i32,
        colour1,
        0u32,
        0.0f32,
        0.0f32,
    );
    // A queued birth is a reason to run the next generation even when every
    // current cell died producing it.
    counters[1].fetch_add(1u32);
}

/// Stage 4: copy the generation's survivor count into the arm word.
///
/// Runs as a single thread after the update and birth stages, so the next
/// frame's per-pixel stages can decide on the device whether the pass has
/// anything to do. The host never reads this back.
#[cube(launch_unchecked)]
pub fn gol_settle_kernel(counters: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0usize {
        let alive = counters[1].load();
        let armed = counters[0].load();
        let mut next = 0u32;
        if armed != 0u32 && alive != 0u32 {
            next = 1u32;
        }
        counters[0].store(next);
        if next == 0u32 {
            counters[1].store(0u32);
        }
    }
}
