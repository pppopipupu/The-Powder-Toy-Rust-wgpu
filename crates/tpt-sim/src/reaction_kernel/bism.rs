//! `BISM` (bismuth) — a port addition above the reference / mod element tables.
//!
//! Bismuth is a solid whose `ctype` is the thickness of its surface oxide film,
//! `0..=100` (`0` = bare silvery metal, `>= 80` = passivated `Bi2O3`). The
//! state-machine shape follows `LITH.cpp:44-260` (one field, three timers) and
//! the neighbour catalysis follows `PTNM.cpp:52-265`; the chemistry itself is
//! the real element:
//!
//! | reaction | source behaviour |
//! |----------|------------------|
//! | `O2` tarnishes: consumed, `ctype += 1`, stops at 80 | `4Bi + 3O2 -> 2Bi2O3`, the film passivates the metal |
//! | hot `WTRV`: `ctype += 2`, steam -> `H2` | `2Bi + 3H2O -> Bi2O3 + 3H2` (red heat) |
//! | `H2` reduces: `ctype -= 8`, `H2` -> `WTRV` | `Bi2O3 + 3H2 -> 2Bi + 3H2O` |
//! | `ACID` only dissolves the film, bare metal is inert | bismuth resists dilute acids |
//! | burning metal spits blue `FIRE` and keeps oxidising | bismuth burns in air |
//! | a spark passing through heats the metal | high resistivity |
//! | a temperature gradient charges a thermocouple | Seebeck / Peltier |
//! | a cooling melt grows crystals | bismuth's stepped hopper crystals |
//! | molten bismuth absorbs a neutron and becomes `POLO` | `Bi-209 -> Bi-210 -> Po-210` |
//!
//! The melt solidifies through its own front rather than through the generic
//! `LAVA` freeze (the thermal pass skips `LAVA(BISM)` there): a cell below
//! 544.7 K freezes onto a bismuth grain or nucleates at the melt's surface, so
//! a puddle solidifies from its boundary inward as blocks of differently
//! oriented grains, and the fresh metal starts bare so the `Bi2O3` film then
//! grows again wherever `O2` reaches it.
//!
//! A crystalline region of a slow-cooling melt may instead request a
//! pixel-cluster rigid body ([`crate::crystal_spawn`]): the device writes one
//! request record per frame at most, and the host materialises the body.
//!
//! Pixel parallel: every lane writes only its own fields and neighbour `part_ty`
//! changes, exactly like the reference passes.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::PassDims;
use super::shared::{
    CREATE_QUEUE_WORDS, hash_u32, launch_grid, mix_index, pmap_at, pmap_slot, request_create_plain,
};
use crate::buffers::SimBuffers;
use crate::crystal_spawn::{CRYSTAL_REQUEST_CAP, CRYSTAL_REQUEST_RECORD};

// Element ids are mirrored as literals because a `#[cube]` body cannot read a
// host constant; the assertions below keep the mirrors honest.
const PT_NONE_C: u32 = 0;
const PT_WATR_C: u32 = 2;
const PT_FIRE_C: u32 = 4;
const PT_LAVA_C: u32 = 6;
const PT_SPRK_C: u32 = 12;
const PT_SLTW_C: u32 = 19;
const PT_ACID_C: u32 = 20;
const PT_WTRV_C: u32 = 21;
const PT_PLSM_C: u32 = 28;
const PT_NEUT_C: u32 = 30;
const PT_ELEC_C: u32 = 31;
const PT_H2_C: u32 = 32;
const PT_O2_C: u32 = 33;
const PT_DSTW_C: u32 = 35;
const PT_CBNW_C: u32 = 73;
const PT_POLO_C: u32 = 181;
const PT_BISM_C: u32 = 239;

const _: () = assert!(PT_NONE_C == PT_NONE);
const _: () = assert!(PT_WATR_C == PT_WATR);
const _: () = assert!(PT_FIRE_C == PT_FIRE);
const _: () = assert!(PT_LAVA_C == PT_LAVA);
const _: () = assert!(PT_SPRK_C == PT_SPRK);
const _: () = assert!(PT_SLTW_C == PT_SLTW);
const _: () = assert!(PT_ACID_C == PT_ACID);
const _: () = assert!(PT_WTRV_C == PT_WTRV);
const _: () = assert!(PT_PLSM_C == PT_PLSM);
const _: () = assert!(PT_NEUT_C == PT_NEUT);
const _: () = assert!(PT_ELEC_C == PT_ELEC);
const _: () = assert!(PT_H2_C == PT_H2);
const _: () = assert!(PT_O2_C == PT_O2);
const _: () = assert!(PT_DSTW_C == PT_DSTW);
const _: () = assert!(PT_CBNW_C == PT_CBNW);
const _: () = assert!(PT_POLO_C == PT_POLO);
const _: () = assert!(PT_BISM_C == PT_BISM);

/// `ctype` at which the `Bi2O3` film passivates the surface.
const PASSIVE_OXIDE: i32 = 80;
/// `ctype` ceiling (the film stops growing here).
const MAX_OXIDE: i32 = 100;
/// `BISM`'s melting point in Kelvin, mirrored from `ElementDef.high_temp` in
/// `elements/data/bism.rs`. The melt solidifies just below it; the generic
/// `LAVA_FREEZE_K` (973 K) is skipped for `LAVA(BISM)` so the crystal front is
/// what freezes the pool.
const BISM_MELT_K: f32 = 544.7;
/// Freeze threshold: a hair below the melting point, so a cell that just
/// crystallised cannot melt again on the same tick.
const BISM_FREEZE_K: f32 = BISM_MELT_K - 0.1f32;
/// `BISM_GRAIN_SHIFT` mirrored: the nucleation hash and the `Create` hook use
/// the same `1 << 4` crystal cell.
const _: () = assert!(BISM_GRAIN_SHIFT == 4);
/// `MAX_TEMP` (`ElementDefs.h:6`), mirrored for the clamp.
const _: () = assert!(MAX_TEMP.to_bits() == 9999.0f32.to_bits());
/// `0` = no record; records start after the `[sequence, gate]` header.
const CRYSTAL_HEADER: usize = 2;
const _: () = assert!(CRYSTAL_REQUEST_CAP == 3);
const _: () = assert!(CRYSTAL_REQUEST_RECORD == 4);

/// `hash_u32(seed) % den < num`.
#[cube]
fn chance(seed: u32, num: u32, den: u32) -> bool {
    (hash_u32(seed) % den) < num
}

/// Clamp a float without `f32::clamp` (not available in the device IR).
#[cube]
fn fclamp(v: f32, lo: f32, hi: f32) -> f32 {
    let mut r = v;
    if r < lo {
        r = lo;
    }
    if r > hi {
        r = hi;
    }
    r
}

/// One `BISM` particle's update: oxidation, reduction, burning, spark heating
/// and the thermocouple.
#[cube]
// `clamp` and `is_multiple_of` are host-only methods; a kernel body must keep
// the comparisons and the `%` it can lower to the device IR.
#[allow(
    clippy::too_many_arguments,
    clippy::manual_clamp,
    clippy::manual_is_multiple_of
)]
fn bism_update(
    id: usize,
    px: i32,
    py: i32,
    seed: u32,
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp2: &mut [i32],
    part_temp: &mut [f32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
) {
    // `ctype` is the oxide film. Clamp values the property tool or Lua may set.
    let mut oxide = part_ctype[id] as i32;
    if oxide < 0i32 {
        oxide = 0i32;
    }
    if oxide > MAX_OXIDE {
        oxide = MAX_OXIDE;
    }
    let mut temp = part_temp[id];
    if temp < 0.0f32 {
        temp = 0.0f32;
    }

    // `life > 1000` is the burn timer (`LITH.cpp`'s dual-purpose `life`).
    let burning = part_life[id] > 1000i32;

    // --- 3x3 surface chemistry -------------------------------------------------
    for n in 0u32..9u32 {
        let dx = (n % 3u32) as i32 - 1i32;
        let dy = (n / 3u32) as i32 - 1i32;
        if dx != 0i32 || dy != 0i32 {
            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
            if nv != 0u32 {
                let nid = pmap_slot(nv);
                let nty = part_ty[nid];
                let nseed = seed ^ mix_index(n, 0x9E37_79B9u32);

                if nty == PT_O2_C {
                    // Tarnish: one oxygen neighbour per roll, until the film
                    // passivates.
                    if oxide < PASSIVE_OXIDE && chance(nseed, 1u32, 6u32) {
                        part_ty[nid] = PT_NONE_C;
                        oxide += 1i32;
                        temp += 1.5f32;
                    }
                } else if nty == PT_WTRV_C {
                    // `2Bi + 3H2O -> Bi2O3 + 3H2` at red heat.
                    if oxide < PASSIVE_OXIDE && temp >= 373.15f32 && chance(nseed, 1u32, 4u32) {
                        part_ty[nid] = PT_H2_C;
                        part_temp[nid] = 440.0f32;
                        oxide += 2i32;
                        temp += 2.0f32;
                    }
                } else if nty == PT_H2_C {
                    // `Bi2O3 + 3H2 -> 2Bi + 3H2O`, strongly exothermic.
                    if oxide > 0i32 && chance(nseed, 1u32, 4u32) {
                        oxide -= 8i32;
                        if oxide < 0i32 {
                            oxide = 0i32;
                        }
                        part_ty[nid] = PT_WTRV_C;
                        part_temp[nid] = temp + 25.0f32;
                        temp += 25.0f32;
                    }
                } else if nty == PT_ACID_C {
                    // Acid dissolves the oxide film and nothing else: bare
                    // bismuth resists dilute acids.
                    if oxide > 0i32 && chance(nseed, 1u32, 8u32) {
                        oxide -= 4i32;
                        if oxide < 0i32 {
                            oxide = 0i32;
                        }
                        part_life[nid] -= 1i32;
                        if part_life[nid] <= 0i32 {
                            part_ty[nid] = PT_NONE_C;
                        }
                    }
                } else if nty == PT_FIRE_C || nty == PT_PLSM_C {
                    // Ignition of a thick film above 600 C.
                    if oxide >= 20i32 && temp >= 873.15f32 && !burning && chance(nseed, 1u32, 8u32)
                    {
                        part_life[id] = 1024i32 + (nseed % 32u32) as i32;
                    }
                } else if nty == PT_SPRK_C {
                    // Resistive heating: a spark leaving a bismuth conductor
                    // warms it (`heat_conduct` is the worst of the metals).
                    if chance(nseed, 1u32, 2u32) {
                        temp += 2.0f32;
                    }
                } else if nty == PT_WATR_C
                    || nty == PT_SLTW_C
                    || nty == PT_DSTW_C
                    || nty == PT_CBNW_C
                {
                    // Quench: burning bismuth and water cannot coexist.
                    if burning {
                        part_ty[nid] = PT_WTRV_C;
                        part_temp[nid] = 440.0f32;
                        part_life[id] = 0i32;
                        temp -= 20.0f32;
                    }
                }
            }
        }
    }

    // --- Seebeck / Peltier thermocouple ---------------------------------------
    //
    // A hot particle on one side and a cold one directly opposite mean a
    // temperature gradient across the bismuth. The pair charges `tmp2`; at 100
    // it fires an electron into an empty cell past the cold side and moves heat
    // from the hot side to the cold one.
    if oxide > 0i32 && part_tmp2[id] < 100i32 {
        for n in 0u32..9u32 {
            let dx = (n % 3u32) as i32 - 1i32;
            let dy = (n / 3u32) as i32 - 1i32;
            if dx != 0i32 || dy != 0i32 {
                let hot = pmap_at(pmap, xres, yres, px + dx, py + dy);
                if hot != 0u32 {
                    let hid = pmap_slot(hot);
                    if part_temp[hid] >= 450.0f32 {
                        let cold = pmap_at(pmap, xres, yres, px - dx, py - dy);
                        if cold != 0u32 {
                            let cid = pmap_slot(cold);
                            if part_temp[cid] <= 320.0f32
                                && part_temp[hid] - part_temp[cid] >= 80.0f32
                                && chance(seed ^ mix_index(n, 0x85EB_CA6Bu32), 1u32, 2u32)
                            {
                                part_tmp2[id] += 1i32;
                                if part_tmp2[id] >= 100i32 {
                                    part_tmp2[id] = 0i32;
                                    // Peltier: heat is carried from the hot
                                    // junction to the cold one.
                                    part_temp[hid] -= 15.0f32;
                                    part_temp[cid] += 15.0f32;
                                    let ex = px - dx * 2i32;
                                    let ey = py - dy * 2i32;
                                    if pmap_at(pmap, xres, yres, ex, ey) == 0u32 {
                                        request_create_plain(create_queue, PT_ELEC_C, ex, ey);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Burning ---------------------------------------------------------------
    if part_life[id] > 1000i32 {
        for n in 0u32..25u32 {
            let dx = (n % 5u32) as i32 - 2i32;
            let dy = (n / 5u32) as i32 - 2i32;
            if (dx != 0i32 || dy != 0i32)
                && pmap_at(pmap, xres, yres, px + dx, py + dy) == 0u32
                && chance(seed ^ mix_index(n, 0x27D4_EB2Fu32), 1u32, 10u32)
            {
                request_create_plain(create_queue, PT_FIRE_C, px + dx, py + dy);
            }
        }
        // The burning metal keeps oxidising; the film is what burns.
        if oxide < MAX_OXIDE && chance(seed ^ 0x51ED_2701u32, 1u32, 4u32) {
            oxide += 2i32;
            if oxide > MAX_OXIDE {
                oxide = MAX_OXIDE;
            }
        }
    }

    // --- Commit ----------------------------------------------------------------
    if oxide > MAX_OXIDE {
        oxide = MAX_OXIDE;
    }
    part_ctype[id] = oxide as u32;
    part_temp[id] = fclamp(temp, 0.0f32, 9999.0f32);
}

/// One molten-bismuth (`LAVA` with `ctype == BISM`) cell: crystal growth, the
/// rigid-crystal request and neutron capture.
#[cube]
// `is_multiple_of` is host-only; the kernel keeps the `%` the IR can lower.
#[allow(clippy::too_many_arguments, clippy::manual_is_multiple_of)]
fn bism_molten_update(
    id: usize,
    px: i32,
    py: i32,
    seed: u32,
    frame_seed: u32,
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_temp: &mut [f32],
    crystal_requests: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
) {
    // `Bi-209 + n -> Bi-210 -> Po-210`: a neutron absorbed by the melt turns it
    // into polonium and heats the cell.
    for n in 0u32..9u32 {
        let dx = (n % 3u32) as i32 - 1i32;
        let dy = (n / 3u32) as i32 - 1i32;
        if (dx != 0i32 || dy != 0i32) && chance(seed ^ mix_index(n, 0x1656_67B1u32), 1u32, 10u32) {
            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
            if nv != 0u32 {
                let nid = pmap_slot(nv);
                if part_ty[nid] == PT_NEUT_C && part_ty[id] == PT_LAVA_C {
                    part_ty[nid] = PT_NONE_C;
                    part_ty[id] = PT_POLO_C;
                    part_ctype[id] = 0u32;
                    part_life[id] = 0i32;
                    part_tmp[id] = 0i32;
                    part_temp[id] = fclamp(part_temp[id] + 250.0f32, 0.0f32, 9999.0f32);
                }
            }
        }
    }

    // Crystal growth: below the melting point a melt cell freezes onto a
    // bismuth grain and inherits its orientation (`tmp`), and a cell on the
    // melt's boundary nucleates a grain of its own. `thermal_kernel` skips the
    // generic `LAVA` freeze for `LAVA(BISM)` so this front is what solidifies
    // the pool: a cooling puddle grows a mass of crystals from its surface
    // inward instead of snapping into one plain block in a single tick.
    if part_ty[id] == PT_LAVA_C && part_temp[id] < BISM_FREEZE_K {
        // One nucleation roll per cell and frame; the neighbour loop only
        // decides on which side the crystal starts. The rate rises with the
        // supercooling: a melt far below its melting point solidifies within a
        // few frames, a puddle hovering at the melting point lingers and grows
        // larger crystals.
        let mut den = 32u32;
        if BISM_MELT_K - part_temp[id] >= 100.0f32 {
            den = 4u32;
        }
        let nucleate = chance(seed ^ 0x8B15_2711u32, 1u32, den);
        for n in 0u32..9u32 {
            let dx = (n % 3u32) as i32 - 1i32;
            let dy = (n / 3u32) as i32 - 1i32;
            if (dx != 0i32 || dy != 0i32) && part_ty[id] == PT_LAVA_C {
                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                // A seedless cell starts its own grain; contact growth
                // overwrites it with the grain it freezes onto.
                let cx = (px as u32) >> 4u32;
                let cy = (py as u32) >> 4u32;
                let mut grain =
                    (hash_u32(mix_index(cx, mix_index(cy, 0x8B15_0001u32))) % 8u32) as i32;
                let mut freezes = false;
                if nv == 0u32 {
                    freezes = nucleate;
                } else {
                    let nid = pmap_slot(nv);
                    if part_ty[nid] == PT_BISM_C {
                        if chance(seed ^ mix_index(n, 0x2545_F491u32), 1u32, 8u32) {
                            freezes = true;
                            grain = part_tmp[nid];
                        }
                    } else if part_ty[nid] != PT_LAVA_C || part_ctype[nid] != PT_BISM_C {
                        // Anything that is not this melt is a cold boundary.
                        freezes = nucleate;
                    }
                }
                if freezes {
                    part_ty[id] = PT_BISM_C;
                    part_ctype[id] = 0u32;
                    part_tmp[id] = grain;
                    // `life` is LAVA's shimmer/temperature feedback; a frozen
                    // cell must not inherit a burn timer.
                    part_life[id] = 0i32;
                    // Latent heat of fusion, capped below the melting point so
                    // the fresh crystal cannot melt again on the next tick.
                    let mut t2 = part_temp[id] + 4.0f32;
                    if t2 > BISM_FREEZE_K {
                        t2 = BISM_FREEZE_K;
                    }
                    part_temp[id] = t2;
                }
            }
        }
    }

    // Rare rigid crystal: the atomic gate lets exactly one lane try per tick,
    // and a modest chance turns that into a body while the melt stays
    // supercooled and big enough. The host materialises the body from this
    // record; the body's own pixels replace the melt cells, so a cooling
    // puddle ends up as a cluster of stepped crystals rather than a single
    // slab.
    if part_ty[id] == PT_LAVA_C && part_temp[id] < BISM_FREEZE_K {
        let prev = crystal_requests[1].fetch_max(frame_seed);
        if prev != frame_seed && chance(seed ^ 0x1D2E_3F40u32, 1u32, 16u32) {
            let mut melt = 0i32;
            for n in 0u32..25u32 {
                let dx = (n % 5u32) as i32 - 2i32;
                let dy = (n / 5u32) as i32 - 2i32;
                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                if nv != 0u32 {
                    let nid = pmap_slot(nv);
                    if part_ty[nid] == PT_LAVA_C && part_ctype[nid] == PT_BISM_C {
                        melt += 1i32;
                    }
                }
            }
            if melt >= 8i32 {
                let seq = crystal_requests[0].fetch_add(1u32);
                let slot = (seq as usize) % CRYSTAL_REQUEST_CAP;
                let base = CRYSTAL_HEADER + slot * CRYSTAL_REQUEST_RECORD;
                crystal_requests[base].store(px as u32);
                crystal_requests[base + 1].store(py as u32);
                // Radius 3..=5 from the frame roll.
                crystal_requests[base + 2].store(3u32 + (hash_u32(seed) % 3u32));
                crystal_requests[base + 3].store(part_temp[id] as u32);
            }
        }
    }
}

/// Launch the `BISM` pass for one frame.
pub(super) fn launch(bufs: &SimBuffers, dims: &PassDims) {
    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        bism_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            BufferArg::from_raw_parts(
                bufs.gpu_crystal_requests.clone(),
                crate::crystal_spawn::CRYSTAL_REQUEST_WORDS,
            ),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn bism_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_temp: &mut [f32],
    create_queue: &mut [Atomic<u32>],
    crystal_requests: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty == PT_BISM_C || (ty == PT_LAVA_C && part_ctype[id] == PT_BISM_C) {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 0xC2B2_AE35u32));
                if ty == PT_BISM_C {
                    bism_update(
                        id,
                        px,
                        py,
                        seed,
                        pmap,
                        part_ty,
                        part_life,
                        part_ctype,
                        part_tmp2,
                        part_temp,
                        create_queue,
                        xres,
                        yres,
                    );
                } else {
                    bism_molten_update(
                        id,
                        px,
                        py,
                        seed,
                        frame_seed,
                        pmap,
                        part_ty,
                        part_life,
                        part_ctype,
                        part_tmp,
                        part_temp,
                        crystal_requests,
                        xres,
                        yres,
                    );
                }
            }
        }
    }
}
