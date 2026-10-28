//! Damage charges (`DMG`) and blast-cell stamping.

use core::mem::offset_of;

use cubecl::prelude::*;
use tpt_core::constants::*;
use tpt_core::elements::GpuElementProps;

use super::ring_dir;
use crate::buffers::SimBuffers;
use crate::reaction_kernel::shared::{elem_word, pmap_at, pmap_slot};

/// `u32` word offset of [`GpuElementProps::high_pressure_transition`].
pub(super) const WORD_HIGH_PRESSURE_TRANSITION: u32 = 21;

const _: () = assert!(
    offset_of!(GpuElementProps, high_pressure_transition)
        == WORD_HIGH_PRESSURE_TRANSITION as usize * 4
);

/// `PT_NUM` as a `u32`, so kernel-side range checks stay in one integer type.
const PT_NUM_U32: u32 = PT_NUM as u32;

/// `DMG.cpp:55`: blast radius in pixels.
const DMG_RAD: i32 = 25;

/// `DMG.cpp:80-82`: blast impulse magnitude (`cos(angle) * 7.0f`), in pixels
/// per step.
const DMG_STRENGTH: u32 = 7;

/// Squared pixel radius of the blast.
///
/// TPT tests `int(sqrt(dx² + dy²)) <= rad` (`DMG.cpp:74-75`), which for the
/// integer `d2 = dx² + dy²` is exactly `d2 < (rad + 1)²`: `676` is the first
/// square that truncates to `26`. The truncated test reaches one pixel further
/// than `d2 <= rad²` — `(25, 5)` (`d2 = 650`) is inside the blast.
const DMG_DISC: i32 = (DMG_RAD + 1) * (DMG_RAD + 1);

/// Bits per axis of the packed blast origin.
const BLAST_AXIS_BITS: u32 = 12;
/// Mask of one packed blast origin axis.
const BLAST_AXIS_MASK: u32 = (1u32 << BLAST_AXIS_BITS) - 1u32;

// The blast grid stores the origin in 12 bits per axis, so the canvas must
// stay inside that field for the packing to be lossless.
const _: () = assert!(SimBuffers::MAX_SUPPORTED_CANVAS_SIDE <= BLAST_AXIS_MASK);

/// Pack one `blast_cell` word: `strength` in the high byte, the detonation
/// pixel in the low `24` bits (`x` in bits 12..24, `y` in bits 0..12).
///
/// `0` is the "no blast" sentinel. [`DMG_STRENGTH`] is non-zero, so a real
/// blast never packs to zero, not even at the top-left corner.
///
/// The strength byte is the `fetch_max` rank: a stronger blast overwrites a
/// weaker one in the same cell. TPT's `DMG` impulse is the constant `7.0f` of
/// `DMG.cpp:80-82`, so every blast this port emits has the same strength and
/// ties fall through to the packed origin bits (see [`dmg_blast_kernel`]).
#[cube]
pub(super) fn blast_pack(strength: u32, x: i32, y: i32) -> u32 {
    ((strength & 0xFFu32) << 24u32)
        | (((x as u32) & BLAST_AXIS_MASK) << BLAST_AXIS_BITS)
        | ((y as u32) & BLAST_AXIS_MASK)
}

/// Strength byte of a [`blast_pack`] word.
#[cube]
pub(super) fn blast_strength(word: u32) -> u32 {
    (word >> 24u32) & 0xFFu32
}

/// Detonation `x` of a [`blast_pack`] word.
#[cube]
pub(super) fn blast_origin_x(word: u32) -> i32 {
    ((word >> BLAST_AXIS_BITS) & BLAST_AXIS_MASK) as i32
}

/// Detonation `y` of a [`blast_pack`] word.
#[cube]
pub(super) fn blast_origin_y(word: u32) -> i32 {
    (word & BLAST_AXIS_MASK) as i32
}

/// `DMG.cpp:65`: the neighbours that do **not** set a charge off.
///
/// `EMBR` is the interesting one: a `BOMB` blast fills its radius with embers
/// (`BOMB.cpp:92-121`), so a charge that survives a nearby explosion is
/// surrounded by `EMBR` and stays inert until something else touches it. A
/// `BOMB` itself is not on the list, so a `BOMB` next to a `DMG` detonates it.
#[cube]
pub(super) fn dmg_immune(ty: u32) -> bool {
    ty == PT_DMG
        || ty == PT_EMBR
        || ty == PT_DMND
        || ty == PT_CLNE
        || ty == PT_PCLN
        || ty == PT_BCLN
}

/// 1. Zero the transient per-cell blast grid.
///
/// The grid is never saved and only the `DMG` pass fills it, so it starts empty
/// every frame. `Atomic<u32>` because the stamping pass runs one thread per
/// detonating charge and their discs overlap.
#[cube(launch_unchecked)]
pub fn blast_clear_kernel(blast_cell: &mut [Atomic<u32>], num_cells: u32) {
    let idx = ABSOLUTE_POS;
    if idx < num_cells as usize {
        blast_cell[idx].store(0u32);
    }
}

/// 2. `DMG` detonation: publish the blast and kill the charge.
///
/// `DMG.cpp:56-66` scans the eight neighbours of the charge and sets it off
/// when any of them holds a matter particle that is not one of the
/// [`dmg_immune`] types; a pixel that is empty — or whose slot is already dead
/// this frame, which TPT sees as an empty `pmap` word after `kill_part` — is
/// skipped. The charge then dies exactly like `kill_part` (`DMG.cpp:67`):
/// `part_ty = PT_NONE`.
///
/// The disc of `DMG.cpp:68-110` is walked once per detonation. Every covered
/// pixel stamps its origin into the cell it belongs to, and every covered pixel
/// that holds a live particle adds the `DMG.cpp:87` pressure pulse to that cell
/// (TPT adds `pv` only where `pmap` is set, so an empty pixel contributes no
/// pressure). `Atomic::fetch_max` keeps the strongest blast per cell.
///
/// **Collapse:** a cell holds one origin, so when two blasts cover the same
/// cell only the stronger one is applied there — and because every `DMG` blast
/// has the same strength, the tie falls through to the packed origin, i.e. the
/// larger `(x, y)` wins. TPT applies every blast's push to every particle it
/// covers, so a particle inside two overlapping blasts receives one push here.
#[cube(launch_unchecked)]
pub fn dmg_blast_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    blast_cell: &mut [Atomic<u32>],
    air_pv: &mut [f32],
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_DMG {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;

                // The 3x3 trigger (`DMG.cpp:56-66`).
                let mut trigger = false;
                for n in 0u32..9u32 {
                    let (dx, dy) = ring_dir(n);
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 {
                            let nt = part_ty[pmap_slot(nv)];
                            if nt != PT_NONE && !dmg_immune(nt) {
                                trigger = true;
                            }
                        }
                    }
                }

                if trigger {
                    // `kill_part(i)` before the blast (`DMG.cpp:67`).
                    part_ty[id] = PT_NONE;

                    // Stamp the blast over its disc (`DMG.cpp:68-110`).
                    let word = blast_pack(DMG_STRENGTH, px, py);
                    for oy in 0u32..(2u32 * DMG_RAD as u32 + 1u32) {
                        let dy = oy as i32 - DMG_RAD;
                        for ox in 0u32..(2u32 * DMG_RAD as u32 + 1u32) {
                            let dx = ox as i32 - DMG_RAD;
                            let nx = px + dx;
                            let ny = py + dy;
                            // `(nxi || nxj)`: the origin pixel is skipped, and
                            // `int(sqrt(d2)) <= rad` is `d2 < DMG_DISC`.
                            if (dx != 0i32 || dy != 0i32)
                                && dx * dx + dy * dy < DMG_DISC
                                && nx >= 0i32
                                && ny >= 0i32
                                && nx < xres as i32
                                && ny < yres as i32
                            {
                                let cidx =
                                    (((ny as u32) / CELL) * xcells + ((nx as u32) / CELL)) as usize;
                                blast_cell[cidx].fetch_max(word);
                                let nv = pmap[(ny as u32 * xres + nx as u32) as usize];
                                if nv != 0u32 && part_ty[pmap_slot(nv)] != PT_NONE {
                                    // `DMG.cpp:87`: `pv[cell] += 1.0f` per
                                    // covered particle pixel. Overlapping
                                    // detonations race on this read-modify-write
                                    // exactly like the combustion kernels' `pv`
                                    // writes; the lost updates are bounded by
                                    // the number of charges in one cell.
                                    air_pv[cidx] += 1.0f32;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// 3. Apply the blast to every particle inside it.
///
/// Each matter particle reads the blast of its own cell and recomputes its
/// exact pixel offset from the stored origin, so the cell granularity of the
/// grid never widens the blast: a particle in a cell that the disc only
/// partially covers is skipped unless its own pixel is inside the disc.
///
/// The impulse is TPT's `DMG.cpp:80-82`:
///
/// ```text
/// angle = atan2(dy, dx);  vx += cos(angle) * 7;  vy += sin(angle) * 7
/// ```
///
/// i.e. a constant magnitude of [`DMG_STRENGTH`] along the unit vector from the
/// origin, with no distance falloff. It is evaluated as `dx * 7 / sqrt(d2)`
/// (`sin`/`cos` of `atan2` is exactly the normalised offset), which avoids the
/// `atan2` round trip. Unlike `ACEL`/`RPEL`/`FRAY` the result is **not**
/// clamped: `DMG.cpp` adds to the velocity unclamped, exactly like TPT.
///
/// The type transitions of `DMG.cpp:88-105` follow, in TPT's order: the
/// element's own `HighPressureTransition` wins, then the hard-coded
/// `BMTL`/`GLAS`/`COAL`/`QRTZ`/`TUNG`/`WOOD` pairings. A transition to
/// `PT_NONE` (the `SHLD` shields) is `part_change_type`'s kill path
/// (`Simulation.cpp:1804-1807`). `DMG.cpp` changes no temperature, so there is
/// no heat term here.
#[cube(launch_unchecked)]
pub fn dmg_apply_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_ctype: &mut [u32],
    // `&mut` even though this kernel only loads: WGSL allows `atomic<u32>` only
    // in read-write storage, and a `&[Atomic<u32>]` binding is emitted as a
    // read-only storage buffer that wgpu rejects at pipeline creation.
    blast_cell: &mut [Atomic<u32>],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let slot = pmap_slot(v);
            let ty = part_ty[slot];
            // A slot killed earlier in this dispatch (the charge itself, or a
            // blast victim) keeps its `pmap` word until the movement pass
            // rebuilds the map, so the liveness test is on `part_ty`.
            if ty != PT_NONE {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let word = blast_cell[cidx].load();
                if word != 0u32 {
                    let dx = px - blast_origin_x(word);
                    let dy = py - blast_origin_y(word);
                    let d2 = dx * dx + dy * dy;
                    if (dx != 0i32 || dy != 0i32) && d2 < DMG_DISC {
                        let inv = (blast_strength(word) as f32) / (d2 as f32).sqrt();
                        part_vx[slot] += (dx as f32) * inv;
                        part_vy[slot] += (dy as f32) * inv;

                        // `DMG.cpp:89-90`: the element table's own
                        // high-pressure transition, checked before the
                        // hard-coded pairings. `NT` (`-1`) means "none"; a
                        // target of `PT_NONE` kills.
                        let hp = if ty < PT_NUM_U32 {
                            elem_word(elem_words, ty, WORD_HIGH_PRESSURE_TRANSITION) as i32
                        } else {
                            NT
                        };
                        if hp > -1i32 && (hp as u32) < PT_NUM_U32 {
                            part_ty[slot] = hp as u32;
                        } else if ty == PT_BMTL {
                            // `DMG.cpp:91-92`: `BMTL` carries
                            // `HighPressureTransition = ST`, and `ST` is
                            // `PT_NUM` (`TransitionConstants.h:13`), so it fails
                            // the `< PT_NUM` test above and this pairing runs.
                            part_ty[slot] = PT_BRMT;
                        } else if ty == PT_GLAS {
                            part_ty[slot] = PT_BGLA;
                        } else if ty == PT_COAL {
                            part_ty[slot] = PT_BCOL;
                        } else if ty == PT_QRTZ {
                            part_ty[slot] = PT_PQRT;
                        } else if ty == PT_TUNG {
                            part_ty[slot] = PT_BRMT;
                            part_ctype[slot] = PT_TUNG;
                        } else if ty == PT_WOOD {
                            part_ty[slot] = PT_SAWD;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reaction_kernel::force::accumulators::WORD_HEAT_CONDUCT;
    use crate::reaction_kernel::shared::{ELEM_PROP_WORDS, WORD_PROPERTIES};
    use tpt_core::elements::ElementRegistry;

    /// `gpu_elem_props` mirror with the words this pass reads.
    fn words_table() -> Vec<u32> {
        let props = ElementRegistry::new_default().to_gpu_props();
        let mut words = vec![0u32; PT_NUM * ELEM_PROP_WORDS];
        for (i, p) in props.iter().enumerate() {
            let base = i * ELEM_PROP_WORDS;
            words[base + WORD_PROPERTIES as usize] = p.properties;
            words[base + WORD_HEAT_CONDUCT as usize] = p.heat_conduct;
            words[base + WORD_HIGH_PRESSURE_TRANSITION as usize] =
                p.high_pressure_transition as u32;
        }
        words
    }

    #[test]
    fn blast_pack_round_trips_every_field() {
        // `(strength << 24) | (x << 12) | y`: strength, origin x, origin y.
        let word = blast_pack(DMG_STRENGTH, 1234, 567);
        assert_eq!(blast_strength(word), DMG_STRENGTH);
        assert_eq!(blast_origin_x(word), 1234);
        assert_eq!(blast_origin_y(word), 567);

        // Both axis fields are 12 bits wide, so the largest canvas pixel fits.
        let word = blast_pack(DMG_STRENGTH, 4095, 4095);
        assert_eq!(blast_origin_x(word), 4095);
        assert_eq!(blast_origin_y(word), 4095);
        assert_eq!(blast_strength(word), DMG_STRENGTH);

        // Out-of-range origins are masked instead of spilling into the next
        // field (the canvas cap keeps this unreachable, but the packing must
        // not alias two different origins).
        let word = blast_pack(DMG_STRENGTH, 4096, 8192);
        assert_eq!(blast_origin_x(word), 0);
        assert_eq!(blast_origin_y(word), 0);
        assert_eq!(blast_strength(word), DMG_STRENGTH);
    }

    #[test]
    fn blast_zero_is_the_empty_cell_sentinel() {
        // A real blast never packs to `0`, not even at the top-left corner:
        // the strength byte is non-zero, so `0` can mean "no blast".
        assert_eq!(blast_pack(0, 0, 0), 0);
        assert_ne!(blast_pack(DMG_STRENGTH, 0, 0), 0);
        // Every origin and strength a `DMG` can emit stays off the sentinel.
        for x in [0i32, 1, 2048, 4095] {
            for y in [0i32, 1, 2048, 4095] {
                assert_ne!(blast_pack(DMG_STRENGTH, x, y), 0, "({x}, {y})");
            }
        }
    }

    #[test]
    fn fetch_max_keeps_the_strongest_blast() {
        // The strength byte is the high byte, so `max` ranks by strength first.
        let weak = blast_pack(1, 4095, 4095);
        let strong = blast_pack(DMG_STRENGTH, 0, 0);
        assert!(strong > weak);
        assert_eq!(weak.max(strong), strong);
        assert_eq!(strong.max(weak), strong);

        // TPT's impulse is a constant, so two blasts of one cell tie on
        // strength and the packed origin breaks the tie deterministically
        // (larger `x` first, then larger `y`).
        let left = blast_pack(DMG_STRENGTH, 10, 20);
        let right = blast_pack(DMG_STRENGTH, 11, 1);
        assert_eq!(left.max(right), right);
        let low = blast_pack(DMG_STRENGTH, 11, 1);
        let high = blast_pack(DMG_STRENGTH, 11, 2);
        assert_eq!(low.max(high), high);
    }

    #[test]
    fn blast_disc_matches_tpt_truncated_sqrt() {
        // `DMG.cpp:74-75`: `int(sqrt(dx² + dy²)) <= rad`, evaluated over TPT's
        // `[-rad, rad]²` loop bounds. `d2 < (rad + 1)²` is the exact integer
        // form; `d2 <= rad²` would clip the rim (`(25, 5)` is inside).
        assert_eq!(DMG_RAD, 25);
        assert_eq!(DMG_DISC, 676);
        let mut inside = 0u32;
        for dy in -DMG_RAD..=DMG_RAD {
            for dx in -DMG_RAD..=DMG_RAD {
                let d2 = dx * dx + dy * dy;
                let tpt = ((d2 as f32).sqrt() as i32) <= DMG_RAD;
                assert_eq!(d2 < DMG_DISC, tpt, "({dx}, {dy})");
                if tpt {
                    inside += 1u32;
                }
            }
        }
        // The truncated disc holds 2109 pixels, i.e. the origin — which
        // `DMG.cpp:72` skips with `(nxi || nxj)` — plus 2108 pushed ones. It is
        // wider than the 1961 pixels of the strict `d2 <= 625` circle.
        assert_eq!(inside, 2109);
    }

    #[test]
    fn dmg_trigger_ignores_the_immune_elements() {
        // `DMG.cpp:65`: these neighbours do not set the charge off. `EMBR` is
        // in the list because a `BOMB` blast carpets its radius with embers.
        for ty in [PT_DMG, PT_EMBR, PT_DMND, PT_CLNE, PT_PCLN, PT_BCLN] {
            assert!(dmg_immune(ty), "{ty}");
        }
        // Everything else does — including `BOMB`, which is what makes a bomb
        // next to a charge detonate it.
        for ty in [PT_NONE, PT_BOMB, PT_WATR, PT_DUST, PT_GLAS, PT_STNE] {
            assert!(!dmg_immune(ty), "{ty}");
        }
    }

    #[test]
    fn dmg_transitions_come_from_the_element_table() {
        let words = words_table();
        // `DMG.cpp:89-90`: a `HighPressureTransition` below `PT_NUM` wins over
        // the hard-coded pairings. `BRCK` carries `STNE` (`BRCK.cpp:39`).
        assert_eq!(
            elem_word(&words, PT_BRCK, WORD_HIGH_PRESSURE_TRANSITION) as i32,
            PT_STNE as i32
        );
        // `PT_NONE` (0) is a valid target: `part_change_type` kills the
        // particle (`Simulation.cpp:1804-1807`). The `SHLD` shields use it.
        assert_eq!(
            elem_word(&words, PT_SHLD1, WORD_HIGH_PRESSURE_TRANSITION) as i32,
            PT_NONE as i32
        );
        // `ST` is `PT_NUM` in TPT (`TransitionConstants.h:13`), so `BMTL` fails
        // the `< PT_NUM` test and its `BMTL -> BRMT` pairing
        // (`DMG.cpp:91-92`) runs — in TPT and here alike.
        assert_eq!(
            elem_word(&words, PT_BMTL, WORD_HIGH_PRESSURE_TRANSITION) as i32,
            ST
        );
        assert!((ST as u32) >= PT_NUM_U32);
        // The remaining hard-coded pairings need `NT` here, or they would never
        // run.
        for ty in [PT_GLAS, PT_COAL, PT_QRTZ, PT_TUNG, PT_WOOD] {
            assert_eq!(
                elem_word(&words, ty, WORD_HIGH_PRESSURE_TRANSITION) as i32,
                NT,
                "{ty}"
            );
        }
        // `TUNG` is the one pairing that also writes `ctype` (`DMG.cpp:99-103`).
        assert_eq!(PT_TUNG, 171);
    }
}
