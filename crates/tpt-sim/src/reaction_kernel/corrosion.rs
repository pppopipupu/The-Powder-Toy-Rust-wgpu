//! `ACID` / `CAUS` corrosion and `BMTL` rusting.
//!
//! Ports `ACID.cpp:50-146`, `CAUS.cpp:50-105` and `BMTL.cpp:48-77`. The two
//! corrosives walk the radius-2 ring: a neighbour with a hardness is dissolved
//! (one roll in `1000/hardness`), the exothermic heat `(60 - hardness) * 7` is
//! added to the acid, and a `GLAS` pixel on the midpoint shields the target.
//! Once the acid's `life` budget is spent it dies on its next contact. `ACID`
//! additionally flashes secondary explosives, is neutralised into `CAUS` by
//! `WTRV`, and `CAUS` turns `GAS` under pressure into `RFRG`.
//!
//! `BMTL` runs its `tmp` rust timer and converts adjacent `METL`/`IRON` into
//! `BMTL` on a 1-in-100 roll, or itself into `BRMT` on a 1-in-1000 roll.
//!
//! Cross-slot writes are limited to type changes, deaths and the `life` the
//! `FIRE` conversion of a neighbour carries — exactly the convention of the
//! other reaction passes. `BMTL`'s rusting is the one relation that is
//! inverted instead: the `METL`/`IRON` target resolves the `BMTL` source and
//! rewrites its own slot (`BMTL.cpp:62-66`), so no thread mutates a neighbour
//! there. The source and the target derive the roll from both slot ids
//! ([`bmtl_roll`]), which lets the source still detect the `tmp = 1` snap the
//! reference performs on itself.
//!
//! # Design notes
//!
//! * `CAUS.cpp:105` returns `converted` to skip the remainder of `UpdateParticles`
//!   (movement) in TPT's host loop; in this port reaction and movement are distinct
//!   GPU passes so no return channel is required.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::PassDims;
use super::shared::{
    WORD_HARDNESS, elem_word, hash_u32, launch_grid, mix_index, pmap_at, pmap_slot, stasis_frozen,
};
use crate::buffers::SimBuffers;

/// Type field of a packed `pmap` word (`PMAPMASK`).
const PMAP_TYPE: u32 = 0x1FF;

/// Radius of the corrosion ring (`ACID.cpp:59`, `CAUS.cpp:57`).
const RING: u32 = 2;

/// TPT `rng.chance(n, d)`: `roll % d < n`.
#[cube]
fn chance(roll: u32, n: u32, d: u32) -> bool {
    let mut hit = false;
    if d > 0u32 {
        hit = (roll % d) < n;
    }
    hit
}

/// `parts_avg(i, j, PT_GLAS)`: `true` when the midpoint of the two pixels holds
/// glass, which shields the target from acid.
///
/// TPT samples `(int)((x1 + x2)/2 + 0.5)` (the non-`INSL` branch of
/// `Simulation.cpp:997`), so a distance-2 neighbour is shielded by the pixel
/// between the two and an adjacent one by the neighbour itself.
#[cube]
fn glass_between(pmap: &[u32], xres: u32, yres: u32, x1: i32, y1: i32, x2: i32, y2: i32) -> bool {
    let mx = (((x1 + x2) as f32) / 2.0f32 + 0.5f32) as i32;
    let my = (((y1 + y2) as f32) / 2.0f32 + 0.5f32) as i32;
    let v = pmap_at(pmap, xres, yres, mx, my);
    let mut blocked = false;
    if v != 0u32 && (v & PMAP_TYPE) == PT_GLAS {
        blocked = true;
    }
    blocked
}

/// `ACID.cpp:87` / `CAUS.cpp:76`: the neighbour classes that are never
/// dissolved whatever their hardness says.
///
/// `DMND`, `GLAS` and the other hardness-0 materials are already immune through
/// the `chance(0, 1000)` roll; this covers explicit exclusions (`CLNE`, `PCLN`)
/// and aged `FOG`/`RIME` (`tmp > 5`).
#[cube]
fn corrosion_immune(ty: u32, tmp: i32) -> bool {
    let mut immune = false;
    if ty == PT_CLNE || ty == PT_PCLN {
        immune = true;
    }
    if (ty == PT_FOG || ty == PT_RIME) && tmp > 5i32 {
        immune = true;
    }
    immune
}

/// `BMTL.cpp:62` / `BMTL.cpp:72`: the rusting roll of one `BMTL`/metal pair.
///
/// The two sides of the inverted relation must agree on the same roll: the
/// source (`BMTL`) needs it to know whether it may drop its `tmp` to 1, the
/// target (`METL`/`IRON`) needs it to decide whether it turns into `BMTL`
/// (`BMTL.cpp:48-77`). Deriving it from both slot ids gives them the same value
/// without a staged buffer.
#[cube]
fn bmtl_roll(frame_seed: u32, src: u32, dst: u32) -> u32 {
    hash_u32(frame_seed ^ mix_index(src, 1_103_515_245u32) ^ mix_index(dst, 2_246_822_519u32))
}

/// `ACID` / `CAUS`: dissolve neighbours, release the dissolution heat, flash
/// secondary explosives (`ACID.cpp:50-146`, `CAUS.cpp:50-105`); `BMTL`: rust
/// adjacent `METL`/`IRON` and break into `BRMT` (`BMTL.cpp:48-77`).
#[cube(launch_unchecked)]
pub(super) fn corrosion_reaction_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_flags: &mut [u32],
    air_pv: &[f32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if stasis_frozen(part_flags, id) {
                terminate!();
            }
            let px = (idx as u32 % xres) as i32;
            let py = (idx as u32 / xres) as i32;
            if ty == PT_ACID || ty == PT_CAUS {
                let mut life = part_life[id];
                let mut dead = false;
                let span = RING * 2u32 + 1u32;

                for n in 0u32..(span * span) {
                    let dx = (n % span) as i32 - RING as i32;
                    let dy = (n / span) as i32 - RING as i32;
                    if (dx != 0i32 || dy != 0i32) && !dead {
                        let nx = px + dx;
                        let ny = py + dy;
                        let nv = pmap_at(pmap, xres, yres, nx, ny);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            let nty = part_ty[nid];
                            // `rng.gen()` per contact, derived from the slot
                            // and the ring position.
                            let roll = hash_u32(frame_seed ^ ((nid as u32) * 1_103_515_245u32) ^ n);

                            if ty == PT_ACID {
                                if nty != PT_ACID && nty != PT_CAUS {
                                    if nty == PT_C4
                                        || nty == PT_NITR
                                        || nty == PT_GUNP
                                        || nty == PT_RBDM
                                        || nty == PT_LRBD
                                    {
                                        // `ACID.cpp:68-76`: both sides flash.
                                        // TPT does not return here, so the next
                                        // contact can still hit the `life`
                                        // kill below.
                                        part_ty[id] = PT_FIRE;
                                        life = 4i32;
                                        part_ty[nid] = PT_FIRE;
                                        part_life[nid] = 4i32;
                                    } else if nty == PT_WTRV {
                                        // `ACID.cpp:77-85`: steam neutralises
                                        // the acid into caustic.
                                        if chance(roll, 1u32, 250u32) {
                                            part_ty[id] = PT_CAUS;
                                            life = 25i32 + (hash_u32(roll) % 50u32) as i32;
                                            part_ty[nid] = PT_NONE;
                                        }
                                    } else if nty == PT_BASE {
                                        // `BASE.cpp:134-149`: base neutralises acid into salt water.
                                        if life > 50i32 && part_life[nid] > 0i32 {
                                            life -= 1i32;
                                            part_life[nid] -= 1i32;
                                        }
                                        if life <= 50i32 {
                                            part_ty[id] = PT_SLTW;
                                            dead = true;
                                        }
                                        if part_life[nid] <= 0i32 {
                                            part_ty[nid] = PT_SLTW;
                                        }
                                    } else if !corrosion_immune(nty, part_tmp[nid])
                                        && life > 50i32
                                        && chance(
                                            roll,
                                            elem_word(elem_words, nty, WORD_HARDNESS),
                                            1000u32,
                                        )
                                    {
                                        // `ACID.cpp:87-114`: glass on the
                                        // midpoint shields the target.
                                        if !glass_between(pmap, xres, yres, px, py, nx, ny) {
                                            let hardness =
                                                elem_word(elem_words, nty, WORD_HARDNESS) as f32;
                                            let mut heat = (60.0f32 - hardness) * 7.0f32;
                                            if heat < 0.0f32 {
                                                heat = 0.0f32;
                                            }
                                            part_temp[id] += heat;
                                            life -= 1i32;
                                            if nty == PT_LITH {
                                                // `ACID.cpp:99-102`: lithium
                                                // releases hydrogen instead.
                                                part_ty[nid] = PT_H2;
                                            } else {
                                                part_ty[nid] = PT_NONE;
                                            }
                                        }
                                    } else if life <= 50i32 {
                                        // `ACID.cpp:116-120`: a spent acid
                                        // burns itself out on contact.
                                        part_ty[id] = PT_NONE;
                                        dead = true;
                                    }
                                }
                            } else if nty == PT_GAS {
                                // `CAUS.cpp:63-72`: compressed gas is absorbed
                                // and both pixels become refrigerant.
                                let ncidx =
                                    (((ny as u32) / CELL) * xcells + ((nx as u32) / CELL)) as usize;
                                if air_pv[ncidx] > 3.0f32 {
                                    part_ty[nid] = PT_RFRG;
                                    part_ty[id] = PT_RFRG;
                                }
                            } else if nty == PT_BASE {
                                // `BASE.cpp:151-166`: base neutralises caustic into salt water.
                                if life > 50i32 && part_life[nid] > 0i32 {
                                    life -= 1i32;
                                    part_life[nid] -= 1i32;
                                }
                                if life <= 50i32 {
                                    part_ty[id] = PT_NONE;
                                    dead = true;
                                }
                                if part_life[nid] <= 0i32 {
                                    part_ty[nid] = PT_SLTW;
                                }
                            } else if nty != PT_ACID
                                && nty != PT_CAUS
                                && nty != PT_RFRG
                                && nty != PT_RFGL
                            {
                                if !corrosion_immune(nty, part_tmp[nid])
                                    && life > 50i32
                                    && chance(
                                        roll,
                                        elem_word(elem_words, nty, WORD_HARDNESS),
                                        1000u32,
                                    )
                                {
                                    if !glass_between(pmap, xres, yres, px, py, nx, ny) {
                                        let hardness =
                                            elem_word(elem_words, nty, WORD_HARDNESS) as f32;
                                        let mut heat = (60.0f32 - hardness) * 7.0f32;
                                        if heat < 0.0f32 {
                                            heat = 0.0f32;
                                        }
                                        part_temp[id] += heat;
                                        life -= 1i32;
                                        part_ty[nid] = PT_NONE;
                                    }
                                } else if life <= 50i32 {
                                    part_ty[id] = PT_NONE;
                                    dead = true;
                                }
                            }
                        }
                    }
                }

                if !dead {
                    part_life[id] = life;
                }
            } else if ty == PT_BMTL {
                // `BMTL.cpp:50-76`: the rust timer. `tmp > 1` burns one unit
                // per frame while the eight neighbours are offered the 1-in-100
                // rust roll; `tmp == 1` is the 1-in-1000 break into `BRMT`.
                let mut tmp = part_tmp[id];
                if tmp > 1i32 {
                    tmp -= 1i32;
                    if tmp <= 7i32 {
                        // `BMTL.cpp:66`: a successful rust while the timer is
                        // already low snaps it to 1. The target cannot write
                        // the source, so the source re-runs the same rolls for
                        // its metal neighbours.
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if dx != 0i32 || dy != 0i32 {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    let nty = part_ty[nid];
                                    if (nty == PT_METL || nty == PT_IRON)
                                        && chance(
                                            bmtl_roll(frame_seed, id as u32, nid as u32),
                                            1u32,
                                            100u32,
                                        )
                                    {
                                        tmp = 1i32;
                                    }
                                }
                            }
                        }
                    }
                    part_tmp[id] = tmp;
                } else if tmp == 1i32
                    && chance(
                        hash_u32(frame_seed ^ mix_index(idx as u32, 0x5BD1_E995u32)),
                        1u32,
                        1000u32,
                    )
                {
                    // `BMTL.cpp:72-76`: the weakened metal breaks.
                    part_tmp[id] = 0i32;
                    part_ty[id] = PT_BRMT;
                }
            } else if ty == PT_METL || ty == PT_IRON {
                // `BMTL.cpp:62-66`, target side: an adjacent `BMTL` with a
                // running timer rusts this metal with one chance in a hundred.
                // Only this thread writes the slot, which removes the neighbour
                // write the reference performs from the `BMTL`.
                let mut rusted = false;
                let mut next_tmp = 0i32;
                for n in 0u32..9u32 {
                    let dx = (n % 3u32) as i32 - 1i32;
                    let dy = (n / 3u32) as i32 - 1i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            if part_ty[nid] == PT_BMTL && part_tmp[nid] > 1i32 {
                                // The source has already decremented, so its
                                // live `tmp` is the value TPT uses here; a
                                // read that loses that race is off by one,
                                // which the timer tolerates.
                                let src = part_tmp[nid];
                                if chance(
                                    bmtl_roll(frame_seed, nid as u32, id as u32),
                                    1u32,
                                    100u32,
                                ) {
                                    rusted = true;
                                    if src <= 7i32 {
                                        next_tmp = 1i32;
                                    } else {
                                        next_tmp = src
                                            - (hash_u32(bmtl_roll(
                                                frame_seed, id as u32, nid as u32,
                                            )) % 5u32)
                                                as i32;
                                    }
                                }
                            }
                        }
                    }
                }
                if rusted {
                    part_ty[id] = PT_BMTL;
                    part_tmp[id] = next_tmp;
                }
            }
        }
    }
}

/// `ACID`: life diffusion between neighbouring acid particles (`ACID.cpp:119-143`).
///
/// Two trades per frame with random neighbours in the radius-2 square.
/// Only the particle with higher life initiates the transfer, ensuring each
/// pair has a single initiator; `Atomic<i32>` updates preserve total life
/// across concurrent threads.
#[cube(launch_unchecked)]
pub(super) fn acid_diffusion_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_life: &mut [Atomic<i32>],
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
            if ty == PT_ACID {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 1_103_515_245u32));

                for trade in 0u32..2u32 {
                    let rx =
                        (hash_u32(roll ^ mix_index(trade, 2_246_822_519u32)) % 5u32) as i32 - 2i32;
                    let ry =
                        (hash_u32(roll ^ mix_index(trade, 3_266_489_909u32)) % 5u32) as i32 - 2i32;
                    if rx != 0i32 || ry != 0i32 {
                        let nx = px + rx;
                        let ny = py + ry;
                        let nv = pmap_at(pmap, xres, yres, nx, ny);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            if nid != id && part_ty[nid] == PT_ACID {
                                let my_life = part_life[id].load();
                                let neighbour_life = part_life[nid].load();
                                if my_life > neighbour_life && my_life > 0i32 {
                                    let temp = my_life - neighbour_life;
                                    let delta = if temp == 1i32 { 1i32 } else { temp / 2i32 };
                                    part_life[id].fetch_add(0i32 - delta);
                                    part_life[nid].fetch_add(delta);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Launch corrosion reaction and acid life diffusion passes.
pub(super) fn launch_corrosion_pass(bufs: &SimBuffers, dims: &PassDims) {
    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        corrosion_reaction_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
        acid_diffusion_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host mirror of [`glass_between`]'s midpoint rounding.
    fn glass_midpoint(x1: i32, x2: i32) -> i32 {
        (((x1 + x2) as f32) / 2.0f32 + 0.5f32) as i32
    }

    #[test]
    fn glass_shield_samples_the_pixel_between() {
        // Adjacent pixels: TPT's `+0.5` rounds onto the neighbour itself (for a
        // positive offset) or onto the acid (for a negative one).
        assert_eq!(glass_midpoint(10, 11), 11);
        assert_eq!(glass_midpoint(10, 9), 10);
        // Distance 2: the sampled pixel is the one between the two.
        assert_eq!(glass_midpoint(10, 12), 11);
        assert_eq!(glass_midpoint(10, 8), 9);
        // Diagonal neighbours use the same rounding on both axes.
        assert_eq!(glass_midpoint(10, 12), 11);
    }

    #[test]
    fn ring_covers_the_radius_two_square() {
        // `ACID.cpp:59` walks `rx/ry` over `-2..=2`, i.e. 24 neighbours.
        let span = RING * 2 + 1;
        assert_eq!(span, 5);
        let mut count = 0;
        for n in 0u32..(span * span) {
            let dx = (n % span) as i32 - RING as i32;
            let dy = (n / span) as i32 - RING as i32;
            if dx != 0 || dy != 0 {
                count += 1;
            }
        }
        assert_eq!(count, 24);
        // The corners are part of the ring.
        assert_eq!((0u32 % span) as i32 - RING as i32, -2);
        assert_eq!((24u32 / span) as i32 - RING as i32, 2);
    }

    #[test]
    fn corrosion_immunity_matches_the_reference() {
        // Clones are unconditionally immune.
        assert!(corrosion_immune_host(PT_CLNE, 0));
        assert!(corrosion_immune_host(PT_PCLN, 0));
        // Hardness-0 materials (DMND, GLAS, ...) are immune through the
        // chance(0, 1000) roll, not through this predicate.
        assert!(!corrosion_immune_host(PT_DMND, 0));
        assert!(!corrosion_immune_host(PT_GLAS, 0));
        assert!(!corrosion_immune_host(PT_LITH, 0));
        assert!(!corrosion_immune_host(PT_ACID, 0));

        // FOG and RIME resist acid when aged (tmp > 5).
        assert!(!corrosion_immune_host(PT_FOG, 0), "fresh fog dissolves");
        assert!(!corrosion_immune_host(PT_FOG, 5), "fog at tmp 5 dissolves");
        assert!(corrosion_immune_host(PT_FOG, 6), "aged fog resists");
        assert!(corrosion_immune_host(PT_FOG, 10), "aged fog resists");

        assert!(!corrosion_immune_host(PT_RIME, 0), "fresh rime dissolves");
        assert!(
            !corrosion_immune_host(PT_RIME, 5),
            "rime at tmp 5 dissolves"
        );
        assert!(corrosion_immune_host(PT_RIME, 6), "aged rime resists");
        assert!(corrosion_immune_host(PT_RIME, 10), "aged rime resists");
    }

    fn corrosion_immune_host(ty: u32, tmp: i32) -> bool {
        ty == PT_CLNE || ty == PT_PCLN || ((ty == PT_FOG || ty == PT_RIME) && tmp > 5)
    }

    /// Host mirror of the acid life diffusion transfer step (`ACID.cpp:128-141`).
    fn acid_diffuse_step(my_life: i32, neighbour_life: i32) -> (i32, i32) {
        if my_life > neighbour_life && my_life > 0 {
            let temp = my_life - neighbour_life;
            let delta = if temp == 1 {
                1
            } else if temp > 1 {
                temp / 2
            } else {
                0
            };
            (my_life - delta, neighbour_life + delta)
        } else {
            (my_life, neighbour_life)
        }
    }

    #[test]
    fn acid_diffusion_conserves_total_life() {
        let cases = [
            (75, 74), // temp == 1
            (75, 73), // temp == 2 (temp > 1)
            (75, 70), // temp == 5 (temp > 1)
            (75, 50), // temp == 25
            (60, 60), // equal life, no transfer
            (50, 75), // lower life, no transfer
            (0, -5),  // non-positive life, no transfer
        ];

        for (a, b) in cases {
            let (new_a, new_b) = acid_diffuse_step(a, b);
            assert_eq!(
                a + b,
                new_a + new_b,
                "total life must be strictly conserved for ({a}, {b}) -> ({new_a}, {new_b})"
            );
        }
    }

    #[test]
    fn acid_diffusion_branches() {
        // temp == 1: exactly 1 unit transferred.
        let (a1, b1) = acid_diffuse_step(75, 74);
        assert_eq!((a1, b1), (74, 75));

        // temp == 2: temp / 2 = 1 unit transferred.
        let (a2, b2) = acid_diffuse_step(75, 73);
        assert_eq!((a2, b2), (74, 74));

        // temp == 5: temp / 2 = 2 units transferred.
        let (a3, b3) = acid_diffuse_step(75, 70);
        assert_eq!((a3, b3), (73, 72));

        // temp == 10: temp / 2 = 5 units transferred.
        let (a4, b4) = acid_diffuse_step(75, 65);
        assert_eq!((a4, b4), (70, 70));

        // Equal or lower life initiates no transfer.
        assert_eq!(acid_diffuse_step(70, 70), (70, 70));
        assert_eq!(acid_diffuse_step(60, 75), (60, 75));

        // Initiator with life <= 0 initiates no transfer.
        assert_eq!(acid_diffuse_step(0, -2), (0, -2));
        assert_eq!(acid_diffuse_step(-1, -5), (-1, -5));
    }

    #[test]
    fn dissolution_heat_matches_the_reference() {
        // `ACID.cpp:104-112`: `(60 - hardness) * 7`, clamped at 0.
        let heat = |hardness: i32| ((60.0f32 - hardness as f32) * 7.0f32).max(0.0f32);
        assert_eq!(heat(14), 322.0f32); // LITH
        assert_eq!(heat(4), 392.0f32); // WTRV
        assert_eq!(heat(60), 0.0f32); // exactly neutral
        assert_eq!(heat(80), 0.0f32, "a hard target cannot cool the acid");
    }

    #[test]
    fn hardness_roll_scales_with_the_target() {
        // `rng.chance(hardness, 1000)`.
        let chance_host = |roll: u32, n: u32, d: u32| d > 0 && roll % d < n;
        assert!(!chance_host(999, 0, 1000), "hardness 0 never dissolves");
        assert!(chance_host(13, 14, 1000));
        assert!(!chance_host(14, 14, 1000));
        assert!(chance_host(0, 14, 1000));
    }

    #[test]
    fn base_neutralisation_produces_sltw() {
        // When acid reaches life <= 50, it turns into SLTW.
        let mut acid_life = 51;
        let mut base_life = 5;
        if acid_life > 50 && base_life > 0 {
            acid_life -= 1;
            base_life -= 1;
        }
        assert_eq!(acid_life, 50);
        assert_eq!(base_life, 4);
        // Spent acid turns to SLTW.
        let acid_ty = if acid_life <= 50 { PT_SLTW } else { PT_ACID };
        assert_eq!(acid_ty, PT_SLTW);
    }
}
