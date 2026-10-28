//! 3D `SC_SPECIAL` elements: VOID, CLNE, DEST, AMTR, EXOT, SING, WARP and the
//! gravity-element bookkeeping (`GRVT`, `GBMB`, `GPMP`).
//!
//! Every branch is a literal port of the C++ update named in its comment. The
//! pass runs after the movement commit, so self writes go straight into
//! `voxel_a`; neighbour writes append to the mutation queue and are applied by
//! [`super::super::mutation`] before the mask rebuild.
//!
//! # Grid adaptations (documented, no invented mechanics)
//!
//! * C++ scans `rx`/`ry` in `-1..=1` (or `-2..=2`). The 3D port scans the full
//!   `3³`/`5³` box, so a diagonal neighbour is a neighbour too.
//! * `SING`'s pop spawns up to 3019 energy particles in one cell because C++
//!   keeps energy particles in a separate layer. A voxel grid has one particle
//!   per cell, so the spawn count is capped at the 27 cells around the
//!   singularity and each created particle occupies its own cell.
//! * `DEST`'s `DEUT`, `EXOT`'s `PROT`/`VIBR`/`TTAN`/`GOLD` and `WARP`'s `ELEC`
//!   branch are omitted because those elements are not in the 59-row table.
//! * `EXOT`'s life is a full-range `int` in C++ (`EXOT.cpp:48,78,104,127,134`:
//!   1000 on `Create`, 1500 after the neighbour reaction, and the `> 1337` /
//!   `< 1001` tests), which the 8-bit word field cannot hold. Every other
//!   element keeps the packed 8-bit counter; `EXOT` stores its life in the
//!   `tmp3` slot (`aux_tmp3`), a field `EXOT.cpp` never reads.

use cubecl::prelude::*;

use super::super::Voxel3dBuffers;
use super::super::engine::Voxel3dEngine;
use super::super::mutation::{MODE_CREATE, MODE_KILL, MODE_REPLACE, push as push_mutation};
use super::{
    C_BRICK_COUNT, C_ELEM_FLAGS, C_ELEM_HEAT_CONDUCT, C_ELEM_MELT_PRODUCT, C_ELEM_WORDS,
    C_FLAG_STATIC, C_LIFE_MASK, C_LIFE_SHIFT, C_NONE, C_PT_AMTR, C_PT_BOMB, C_PT_BREC, C_PT_CLNE,
    C_PT_DEST, C_PT_EXOT, C_PT_GBMB, C_PT_GPMP, C_PT_GRVT, C_PT_LAVA, C_PT_LIGH, C_PT_NBHL,
    C_PT_NEUT, C_PT_PHOT, C_PT_PLUT, C_PT_SING, C_PT_VOID, C_PT_WARP, C_TEMP_MASK, C_TEMP_SHIFT,
    C_TY_MASK, clampf, hash_u32, mix_index, seed_f32, seed_i32, seed_u32,
};

/// Voxels along one brick edge.
const BRICK: u32 = 8;
/// Threads per brick: one per voxel.
const CUBE_DIM: u32 = 8;
/// World edge length.
const WORLD: i32 = 256;
/// TPT `MAX_TEMP`.
const MAX_TEMP_K: f32 = 9999.0;
/// `MAX_TEMP / 2`, used by the SING pop (`SING.cpp:121`).
const HALF_MAX_TEMP_K: f32 = 4999.5;
/// Fixed-point units per TPT pressure unit.
const PV_SCALE_F: f32 = 1024.0;
/// TPT pressure clamp (`Air.cpp:315-327`).
const MAX_PRESSURE: f32 = 256.0;
/// `CFDS = 4 / CELL` with `CELL = 4`.
const CFDS: f32 = 1.0;
/// Temperature of a freshly created particle (`Element.cpp:54`).
const DEFAULT_TEMP_K: f32 = 295.15;
/// Photon default life (`PHOT.cpp:50`).
const PHOT_LIFE: u32 = 680;
/// Photon wavelength mask (`PHOT.cpp:51`).
const PHOT_WAVELENGTH: u32 = 0x3FFF_FFFF;
/// Neutron default life roll (`NEUT.cpp:265-267`).
const RNG_SALT: u32 = 0x9E37_79B1u32;
/// `std::numbers::pi` of the SING pop (`SING.cpp:89`).
const PI_F32: f32 = core::f32::consts::PI;

/// Fixed-point temperature to Kelvin.
#[cube]
fn temp_of(word: u32) -> f32 {
    ((word >> C_TEMP_SHIFT) & C_TEMP_MASK) as f32 * 0.125f32
}

/// Packed word of `ty` at `temp_k` with `life`.
#[cube]
fn pack_word(ty: u32, temp_k: f32, life: u32) -> u32 {
    let mut q = seed_f32(temp_k, 0.0f32);
    if temp_k > 0.0f32 {
        q = (temp_k * 8.0f32).round();
    }
    let mut temp_q = q as u32;
    if temp_q > 65_535u32 {
        temp_q = 65_535u32;
    }
    let mut life_out = seed_u32(life, life);
    if life_out > C_LIFE_MASK {
        life_out = C_LIFE_MASK;
    }
    ty | (temp_q << C_TEMP_SHIFT) | (life_out << C_LIFE_SHIFT)
}

/// TPT `rng.chance(a, b)`.
#[cube]
fn chance(seed: u32, a: u32, b: u32) -> bool {
    (hash_u32(seed) % b) < a
}

/// TPT `rng.between(lo, hi)`.
#[cube]
fn between(seed: u32, lo: u32, hi: u32) -> u32 {
    lo + hash_u32(seed) % (hi - lo + 1u32)
}

/// Coarse cell index (`CELL = 4`) of a world coordinate.
#[cube]
fn air_cell(x: u32, y: u32, z: u32) -> u32 {
    ((y >> 2u32) << 12u32) | ((z >> 2u32) << 6u32) | (x >> 2u32)
}

/// TPT `pv` pressure, in pressure units.
#[cube]
fn pv_at(pv: &mut [Atomic<i32>], x: u32, y: u32, z: u32) -> f32 {
    pv[air_cell(x, y, z) as usize].load() as f32 * (1.0f32 / PV_SCALE_F)
}

/// Adds a pressure impulse to the cell of `(x, y, z)`.
#[cube]
fn pv_add(pv: &mut [Atomic<i32>], x: u32, y: u32, z: u32, impulse: f32) {
    pv[air_cell(x, y, z) as usize].fetch_add((impulse * PV_SCALE_F) as i32);
}

/// Clamps a signed integer without `i32::clamp`.
#[cube]
fn clamp_i(value: i32, lo: i32, hi: i32) -> i32 {
    let mut out = seed_i32(value, value);
    if out < lo {
        out = lo;
    }
    if out > hi {
        out = hi;
    }
    out
}

/// World index, or [`C_NONE`].
#[cube]
fn world_index(x: i32, y: i32, z: i32) -> u32 {
    let mut out = seed_i32(x, C_NONE as i32);
    // `x as u32` maps negatives to huge values, so one unsigned compare
    // replaces each `coordinate >= 0 && coordinate < WORLD` pair.
    if (x as u32) < WORLD as u32 && (y as u32) < WORLD as u32 && (z as u32) < WORLD as u32 {
        out = (((y as u32) << 16u32) | ((z as u32) << 8u32) | x as u32) as i32;
    }
    out as u32
}

/// Whether `id` has a row in the 256-row table.
#[cube]
fn is_known(table: &[u32], id: u32) -> bool {
    id > 0u32 && id < 256u32 && table[(id as usize) * C_ELEM_WORDS] == id
}

/// Whether `id` is a solid, used by DEST's life cost (`DEST.cpp:86`).
#[cube]
fn is_solid(table: &[u32], id: u32) -> bool {
    (table[(id as usize) * C_ELEM_WORDS + C_ELEM_FLAGS] & C_FLAG_STATIC) != 0u32
}

/// One thread per voxel of every simulated brick.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn special_kernel(
    voxel_a: &mut [u32],
    table: &[u32],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    pv: &mut [Atomic<i32>],
    sim_list: &[u32],
    mutation_queue: &mut [u32],
    mutation_count: &mut [Atomic<u32>],
    tick: u32,
) {
    let brick = sim_list[CUBE_POS_X as usize];
    let x = (brick & 31u32) * BRICK + UNIT_POS_X;
    let y = ((brick >> 10u32) & 31u32) * BRICK + UNIT_POS_Y;
    let z = ((brick >> 5u32) & 31u32) * BRICK + UNIT_POS_Z;
    let index = (y << 16u32) | (z << 8u32) | x;
    let word = voxel_a[index as usize];
    let ty = word & C_TY_MASK;
    let xi = x as i32;
    let yi = y as i32;
    let zi = z as i32;
    let life = (word >> C_LIFE_SHIFT) & C_LIFE_MASK;
    let self_temp = temp_of(word);
    let mut rng = seed_u32(
        index,
        hash_u32(mix_index(index, RNG_SALT) ^ tick ^ (ty << 8u32)),
    );

    // --- GRVT (`GRVT.cpp:53-56`): tmp is clamped to +-100. ----------------
    if ty == C_PT_GRVT {
        let tmp = clamp_i(
            seed_i32(index as i32, aux_tmp[index as usize]),
            -100i32,
            100i32,
        );
        aux_tmp[index as usize] = tmp;
    }

    // --- GBMB (`GBMB.cpp:53-68`). ----------------------------------------
    if ty == C_PT_GBMB && life == 0u32 {
        let mut latch = seed_u32(index, 0u32);
        for dx in -1i32..=1i32 {
            for dy in -1i32..=1i32 {
                for dz in -1i32..=1i32 {
                    let ni = world_index(xi + dx, yi + dy, zi + dz);
                    if ni != C_NONE && latch == 0u32 {
                        let nty = voxel_a[ni as usize] & C_TY_MASK;
                        if nty != 0u32 && nty != C_PT_GBMB && nty != C_PT_CLNE && nty != C_PT_BOMB {
                            latch = 1u32;
                        }
                    }
                }
            }
        }
        if latch != 0u32 {
            voxel_a[index as usize] = pack_word(C_PT_GBMB, self_temp, 60u32);
        }
    }

    // --- GPMP (`GPMP.cpp:53-84`). ---------------------------------------
    if ty == C_PT_GPMP {
        if life != 10u32 {
            if life > 0u32 {
                voxel_a[index as usize] = pack_word(C_PT_GPMP, self_temp, life - 1u32);
            }
        } else {
            let temp = clampf(
                seed_f32(self_temp, self_temp),
                -256.0f32 + 273.15f32,
                256.0f32 + 273.15f32,
            );
            let mut next_life = seed_u32(index, 10u32);
            for dx in -2i32..=2i32 {
                for dy in -2i32..=2i32 {
                    for dz in -2i32..=2i32 {
                        let ni = world_index(xi + dx, yi + dy, zi + dz);
                        if ni != C_NONE {
                            let nword = voxel_a[ni as usize];
                            if (nword & C_TY_MASK) == C_PT_GPMP {
                                let nlife = (nword >> C_LIFE_SHIFT) & C_LIFE_MASK;
                                if nlife < 10u32 && nlife > 0u32 {
                                    next_life = 9u32;
                                } else if nlife == 0u32 {
                                    push_mutation(
                                        mutation_queue,
                                        mutation_count,
                                        ni,
                                        MODE_REPLACE,
                                        pack_word(C_PT_GPMP, temp_of(nword), 10u32),
                                        aux_ctype[ni as usize],
                                        aux_tmp[ni as usize],
                                        aux_tmp2[ni as usize],
                                        aux_tmp3[ni as usize],
                                    );
                                }
                            }
                        }
                    }
                }
            }
            voxel_a[index as usize] = pack_word(C_PT_GPMP, temp, next_life);
        }
    }

    // --- VOID (`VOID.cpp`; the kill is `Simulation.cpp:1387-1394`). ------
    if ty == C_PT_VOID {
        // C++ kills a particle when its move enters the void cell. The 3D
        // movement pass steps by one voxel, so a face neighbour is exactly the
        // cell that would have been entered.
        for ox in -1i32..=1i32 {
            for oy in -1i32..=1i32 {
                for oz in -1i32..=1i32 {
                    let faces = (ox == 0i32) as u32 + (oy == 0i32) as u32 + (oz == 0i32) as u32;
                    if faces == 2u32 {
                        let ni = world_index(xi + ox, yi + oy, zi + oz);
                        if ni != C_NONE {
                            let nty = voxel_a[ni as usize] & C_TY_MASK;
                            if nty != 0u32 && nty != C_PT_VOID {
                                push_mutation(
                                    mutation_queue,
                                    mutation_count,
                                    ni,
                                    MODE_KILL,
                                    0u32,
                                    0u32,
                                    0i32,
                                    0i32,
                                    0i32,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // --- CLNE (`CLNE.cpp:52-90`). --------------------------------------
    if ty == C_PT_CLNE {
        let ctype = aux_ctype[index as usize];
        if !is_known(table, ctype) {
            // Learn a type from a neighbour.
            for dx in -1i32..=1i32 {
                for dy in -1i32..=1i32 {
                    for dz in -1i32..=1i32 {
                        let ni = world_index(xi + dx, yi + dy, zi + dz);
                        if ni != C_NONE {
                            let nword = voxel_a[ni as usize];
                            let nty = nword & C_TY_MASK;
                            if is_known(table, nty) && nty != C_PT_CLNE {
                                aux_ctype[index as usize] = nty;
                                if nty == C_PT_LAVA {
                                    aux_tmp[index as usize] = aux_ctype[ni as usize] as i32;
                                }
                            }
                        }
                    }
                }
            }
        } else if ctype != C_PT_LIGH || chance(rng, 1u32, 30u32) {
            // `CLNE.cpp:82-90`: clone, except LIGH only once in thirty ticks.
            let ox = between(rng, 0u32, 2u32) as i32 - 1i32;
            rng = hash_u32(rng);
            let oy = between(rng, 0u32, 2u32) as i32 - 1i32;
            rng = hash_u32(rng);
            let oz = between(rng, 0u32, 2u32) as i32 - 1i32;
            rng = hash_u32(rng);
            let ni = world_index(xi + ox, yi + oy, zi + oz);
            if ni != C_NONE {
                let mut spawn_ctype = seed_u32(ctype, 0u32);
                if ctype == C_PT_LAVA {
                    let tmp = aux_tmp[index as usize];
                    let tmp_id = tmp as u32;
                    if tmp_id > 0u32 && tmp_id < 256u32 {
                        let tmp_base = (tmp_id as usize) * C_ELEM_WORDS;
                        let tmp_melt = table[tmp_base + C_ELEM_MELT_PRODUCT];
                        if tmp_melt == C_PT_LAVA {
                            spawn_ctype = tmp_id;
                        }
                    }
                }
                push_mutation(
                    mutation_queue,
                    mutation_count,
                    ni,
                    MODE_CREATE,
                    pack_word(ctype, DEFAULT_TEMP_K, 0u32),
                    spawn_ctype,
                    0i32,
                    0i32,
                    0i32,
                );
            }
        }
    }

    // --- DEST (`DEST.cpp:51-98`). --------------------------------------
    if ty == C_PT_DEST {
        let ox = between(rng, 0u32, 4u32) as i32 - 2i32;
        rng = hash_u32(rng);
        let oy = between(rng, 0u32, 4u32) as i32 - 2i32;
        rng = hash_u32(rng);
        let oz = between(rng, 0u32, 4u32) as i32 - 2i32;
        rng = hash_u32(rng);
        let ni = world_index(xi + ox, yi + oy, zi + oz);
        if ni != C_NONE {
            let nword = voxel_a[ni as usize];
            let nty = nword & C_TY_MASK;
            if nty != 0u32 && nty != C_PT_DEST && nty != C_PT_CLNE {
                let mut next_life = seed_u32(life, life);
                if life == 0u32 || life > 37u32 {
                    next_life = between(rng, 30u32, 49u32);
                    rng = hash_u32(rng);
                    pv_add(pv, x, y, z, 60.0f32);
                }
                if nty == C_PT_PLUT {
                    pv_add(pv, x, y, z, 20.0f32);
                    if chance(rng, 1u32, 2u32) {
                        rng = hash_u32(rng);
                        // `create_part(ID(r), ..., PT_NEUT)` + MAX_TEMP.
                        push_mutation(
                            mutation_queue,
                            mutation_count,
                            ni,
                            MODE_REPLACE,
                            pack_word(C_PT_NEUT, MAX_TEMP_K, between(rng, 480u32, 959u32)),
                            0u32,
                            0i32,
                            (hash_u32(rng) % 6u32) as i32,
                            1i32,
                        );
                        pv_add(pv, x, y, z, 10.0f32);
                        if next_life >= 4u32 {
                            next_life -= 4u32;
                        } else {
                            next_life = 0u32;
                        }
                    }
                } else if chance(rng, 1u32, 3u32) {
                    rng = hash_u32(rng);
                    push_mutation(
                        mutation_queue,
                        mutation_count,
                        ni,
                        MODE_KILL,
                        0u32,
                        0u32,
                        0i32,
                        0i32,
                        0i32,
                    );
                    let cost = if is_solid(table, nty) { 12u32 } else { 4u32 };
                    if next_life > cost {
                        next_life -= cost;
                    } else {
                        next_life = 1u32;
                    }
                } else if table[(nty as usize) * C_ELEM_WORDS + C_ELEM_HEAT_CONDUCT] != 0u32 {
                    rng = hash_u32(rng);
                    // `parts[ID(r)].temp = MAX_TEMP` keeps the neighbour.
                    push_mutation(
                        mutation_queue,
                        mutation_count,
                        ni,
                        MODE_REPLACE,
                        pack_word(nty, MAX_TEMP_K, (nword >> C_LIFE_SHIFT) & C_LIFE_MASK),
                        aux_ctype[ni as usize],
                        aux_tmp[ni as usize],
                        aux_tmp2[ni as usize],
                        aux_tmp3[ni as usize],
                    );
                }
                voxel_a[index as usize] = pack_word(C_PT_DEST, MAX_TEMP_K, next_life);
                let pressure = pv_at(pv, x, y, z);
                let target = clampf(pressure + 80.0f32, -MAX_PRESSURE, MAX_PRESSURE);
                let delta = target - pressure;
                pv_add(pv, x, y, z, delta);
            }
        }
    }

    // --- AMTR (`AMTR.cpp:53-84`). ---------------------------------------
    if ty == C_PT_AMTR {
        let mut amtr_life = seed_u32(index, life);
        let mut dead = seed_u32(index, 0u32);
        for dx in -1i32..=1i32 {
            for dy in -1i32..=1i32 {
                for dz in -1i32..=1i32 {
                    if (dx != 0i32 || dy != 0i32 || dz != 0i32) && dead == 0u32 {
                        let ni = world_index(xi + dx, yi + dy, zi + dz);
                        if ni != C_NONE {
                            let nword = voxel_a[ni as usize];
                            let nty = nword & C_TY_MASK;
                            if nty != 0u32
                                && nty != C_PT_AMTR
                                && nty != C_PT_CLNE
                                && nty != C_PT_VOID
                                && nty != C_PT_NBHL
                            {
                                amtr_life += 1u32;
                                if amtr_life == 4u32 {
                                    dead = 1u32;
                                } else {
                                    let nseed = hash_u32(mix_index(ni, RNG_SALT) ^ tick ^ index);
                                    if chance(nseed, 1u32, 10u32) {
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_REPLACE,
                                            pack_word(C_PT_PHOT, temp_of(nword), PHOT_LIFE),
                                            PHOT_WAVELENGTH,
                                            0i32,
                                            (hash_u32(nseed) % 6u32) as i32,
                                            1i32,
                                        );
                                    } else {
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_KILL,
                                            0u32,
                                            0u32,
                                            0i32,
                                            0i32,
                                            0i32,
                                        );
                                    }
                                    pv_add(pv, x, y, z, -2.0f32);
                                }
                            }
                        }
                    }
                }
            }
        }
        if dead != 0u32 {
            voxel_a[index as usize] = 0u32;
            aux_ctype[index as usize] = 0u32;
            aux_tmp[index as usize] = 0i32;
            aux_tmp2[index as usize] = 0i32;
            aux_tmp3[index as usize] = 0i32;
        } else if amtr_life != life {
            voxel_a[index as usize] = pack_word(C_PT_AMTR, self_temp, amtr_life);
        }
    }

    // --- EXOT (`EXOT.cpp:56-195`). --------------------------------------
    //
    // C++ stores EXOT's life in a full-range `int` (`EXOT.cpp:48,78,104,127,134`):
    // 1000 from `Create`, 1500 from the neighbour rule, and both the `> 1337`
    // and `< 1001` branches need values the 8-bit word field cannot hold. The
    // counter therefore lives in the `tmp3` slot, which `EXOT.cpp` never
    // reads. A `tmp3` of 0 marks a freshly painted/spawned voxel and becomes
    // the `Create` default 1000; no EXOT path ever writes a smaller life.
    if ty == C_PT_EXOT {
        let mut tmp = seed_i32(index as i32, aux_tmp[index as usize]);
        let mut tmp2 = seed_i32(index as i32, aux_tmp2[index as usize]);
        let mut exot_life = seed_i32(index as i32, aux_tmp3[index as usize]);
        if exot_life == 0i32 {
            exot_life = 1000i32;
        }
        let mut become_ty = seed_u32(index, 0u32);
        for dx in -2i32..=2i32 {
            for dy in -2i32..=2i32 {
                for dz in -2i32..=2i32 {
                    if dx != 0i32 || dy != 0i32 || dz != 0i32 {
                        let ni = world_index(xi + dx, yi + dy, zi + dz);
                        if ni != C_NONE {
                            let nword = voxel_a[ni as usize];
                            let nty = nword & C_TY_MASK;
                            if nty == C_PT_WARP {
                                if aux_tmp2[ni as usize] > 2000i32 && chance(rng, 1u32, 100u32) {
                                    rng = hash_u32(rng);
                                    tmp2 += 100i32;
                                }
                            } else if nty == C_PT_EXOT
                                && aux_tmp3[ni as usize] == 1500i32
                                && chance(rng, 1u32, 1000u32)
                            {
                                rng = hash_u32(rng);
                                exot_life = 1500i32;
                            }
                            // `EXOT + LAVA(TTAN/GOLD/VIBR)` needs elements that
                            // are not part of this table.
                            if tmp > 245i32
                                && exot_life > 1337i32
                                && become_ty == 0u32
                                && nty != 0u32
                                && nty != C_PT_EXOT
                                && nty != C_PT_BREC
                                && nty != C_PT_CLNE
                                && nty != C_PT_VOID
                                && nty != C_PT_NBHL
                                && nty != C_PT_WARP
                                && is_known(table, nty)
                            {
                                // `create_part(i, x, y, rt)` replaces self.
                                become_ty = nty;
                            }
                        }
                    }
                }
            }
        }
        if become_ty != 0u32 {
            voxel_a[index as usize] = pack_word(become_ty, DEFAULT_TEMP_K, 0u32);
            aux_ctype[index as usize] = 0u32;
            aux_tmp[index as usize] = 0i32;
            aux_tmp2[index as usize] = 0i32;
            aux_tmp3[index as usize] = 0i32;
        } else {
            let mut converted = seed_u32(index, 0u32);
            tmp -= 1i32;
            tmp2 -= 1i32;
            // `if (tmp < 1 || tmp > 250) tmp = 250` (`EXOT.cpp:119-120`),
            // kept as two plain comparisons so the kernel body does not rely
            // on `RangeInclusive::contains`.
            if tmp < 1i32 {
                tmp = 250i32;
            }
            if tmp > 250i32 {
                tmp = 250i32;
            }
            if tmp2 < 1i32 {
                tmp2 = 1i32;
            } else if tmp2 > 6000i32 {
                tmp2 = 10000i32;
                if exot_life < 1001i32 {
                    // `EXOT -> WARP` (`part_change_type`, so life and ctype
                    // survive the change).
                    voxel_a[index as usize] = pack_word(C_PT_WARP, self_temp, exot_life as u32);
                    converted = 1u32;
                }
            } else if exot_life < 1001i32 {
                pv_add(pv, x, y, z, (tmp2 as f32 * CFDS) / 160000.0f32);
            }
            let pressure = pv_at(pv, x, y, z);
            if converted == 0u32 && pressure > 200.0f32 && self_temp > 9000.0f32 && tmp2 > 200i32 {
                tmp2 = 6000i32;
                voxel_a[index as usize] = pack_word(C_PT_WARP, self_temp, exot_life as u32);
                converted = 1u32;
            }
            if converted == 0u32 && tmp2 > 100i32 {
                let mut trades = seed_u32(index, 0u32);
                while trades < 9u32 {
                    let ox = between(rng, 0u32, 4u32) as i32 - 2i32;
                    rng = hash_u32(rng);
                    let oy = between(rng, 0u32, 4u32) as i32 - 2i32;
                    rng = hash_u32(rng);
                    let oz = between(rng, 0u32, 4u32) as i32 - 2i32;
                    rng = hash_u32(rng);
                    let ni = world_index(xi + ox, yi + oy, zi + oz);
                    if ni != C_NONE {
                        let nty = voxel_a[ni as usize] & C_TY_MASK;
                        if nty == C_PT_EXOT {
                            let other = aux_tmp2[ni as usize];
                            if tmp2 > other && other >= 0i32 {
                                let diff = tmp2 - other;
                                let half = diff / 2i32;
                                aux_tmp2[ni as usize] = other + half;
                                tmp2 -= half;
                                trades = 9u32;
                            }
                        }
                    }
                    trades += 1u32;
                }
            }
            if self_temp < 273.15f32 {
                pv_add(pv, x, y, z, -0.01f32);
                tmp -= 1i32;
            }
            if converted == 0u32 {
                voxel_a[index as usize] = pack_word(C_PT_EXOT, self_temp, exot_life as u32);
                aux_tmp[index as usize] = tmp;
                aux_tmp2[index as usize] = tmp2;
                // The word keeps the saturated 8-bit copy for generic readers;
                // the authoritative full-range counter is `tmp3`.
                aux_tmp3[index as usize] = exot_life;
            }
        }
    }

    // --- SING (`SING.cpp:51-152`). --------------------------------------
    if ty == C_PT_SING {
        let singularity = -(life as f32);
        let ccx = (x >> 2u32) as i32;
        let ccy = (y >> 2u32) as i32;
        let ccz = (z >> 2u32) as i32;
        for ox in -1i32..=1i32 {
            for oy in -1i32..=1i32 {
                for oz in -1i32..=1i32 {
                    let cx = clamp_i(ccx + ox, 0i32, 63i32) as u32;
                    let cy = clamp_i(ccy + oy, 0i32, 63i32) as u32;
                    let cz = clamp_i(ccz + oz, 0i32, 63i32) as u32;
                    let c = (cy << 12u32) | (cz << 6u32) | cx;
                    let current = pv[c as usize].load() as f32 * (1.0f32 / PV_SCALE_F);
                    if current < singularity {
                        pv[c as usize]
                            .fetch_add((0.1f32 * (singularity - current) * PV_SCALE_F) as i32);
                    }
                }
            }
        }
        if life < 1u32 {
            // `SING.cpp:85-129`: the pop. Grid adaptation documented above.
            for ox in -1i32..=1i32 {
                for oy in -1i32..=1i32 {
                    for oz in -1i32..=1i32 {
                        let ni = world_index(xi + ox, yi + oy, zi + oz);
                        if ni != C_NONE {
                            let cx = ni & 255u32;
                            let cy = (ni >> 16u32) & 255u32;
                            let cz = (ni >> 8u32) & 255u32;
                            pv_add(pv, cx, cy, cz, aux_tmp[index as usize] as f32);
                        }
                    }
                }
            }
            let mut pop = seed_i32(index as i32, aux_tmp[index as usize]);
            if pop < 0i32 {
                pop = -pop;
            }
            let mut count = seed_u32(index, 3019u32);
            if pop <= 255i32 {
                // `int(pow(pop / 8, 2) * pi)` (`SING.cpp:88-89`).
                let n = (pop / 8) as f32;
                count = (n * n * PI_F32) as u32;
            }
            // The 3D mutation queue is finite; a single voxel can only place
            // one particle per cell, so the pop is capped at one per cell of
            // the surrounding 4³ region.
            if count > 64u32 {
                count = 64u32;
            }
            let mut spawned = seed_u32(index, 0u32);
            let mut tries = seed_u32(index, 0u32);
            while spawned < count && tries < 512u32 {
                let ox = between(rng, 0u32, 6u32) as i32 - 3i32;
                rng = hash_u32(rng);
                let oy = between(rng, 0u32, 6u32) as i32 - 3i32;
                rng = hash_u32(rng);
                let oz = between(rng, 0u32, 6u32) as i32 - 3i32;
                rng = hash_u32(rng);
                let ni = world_index(xi + ox, yi + oy, zi + oz);
                if ni != C_NONE {
                    let kind_seed = hash_u32(rng ^ (tries + 29u32));
                    let kind = if (kind_seed & 1u32) == 0u32 {
                        C_PT_PHOT
                    } else {
                        C_PT_NEUT
                    };
                    let spawn_life = between(kind_seed, 0u32, 299u32);
                    let spawn_ctype = if kind == C_PT_PHOT {
                        PHOT_WAVELENGTH
                    } else {
                        0u32
                    };
                    push_mutation(
                        mutation_queue,
                        mutation_count,
                        ni,
                        MODE_CREATE,
                        pack_word(kind, HALF_MAX_TEMP_K, spawn_life),
                        spawn_ctype,
                        0i32,
                        (hash_u32(kind_seed) % 6u32) as i32,
                        1i32,
                    );
                    spawned += 1u32;
                }
                tries += 1u32;
            }
            voxel_a[index as usize] = 0u32;
            aux_ctype[index as usize] = 0u32;
            aux_tmp[index as usize] = 0i32;
            aux_tmp2[index as usize] = 0i32;
            aux_tmp3[index as usize] = 0i32;
        } else {
            let mut sing_life = seed_u32(index, life);
            let mut sing_tmp = seed_i32(index as i32, aux_tmp[index as usize]);
            let mut sing_temp = seed_f32(self_temp, self_temp);
            for dx in -1i32..=1i32 {
                for dy in -1i32..=1i32 {
                    for dz in -1i32..=1i32 {
                        if (dx != 0i32 || dy != 0i32 || dz != 0i32) && chance(rng, 1u32, 3u32) {
                            let ni = world_index(xi + dx, yi + dy, zi + dz);
                            if ni != C_NONE {
                                let nword = voxel_a[ni as usize];
                                let nty = nword & C_TY_MASK;
                                if nty != 0u32 {
                                    let nlife = (nword >> C_LIFE_SHIFT) & C_LIFE_MASK;
                                    if nty == C_PT_SING && nlife > 10u32 {
                                        if sing_life + nlife <= 255u32 {
                                            sing_life += nlife;
                                        }
                                    } else if sing_life + 3u32 > 255u32 {
                                        if nty != C_PT_SING && chance(rng, 1u32, 1000u32) {
                                            push_mutation(
                                                mutation_queue,
                                                mutation_count,
                                                ni,
                                                MODE_REPLACE,
                                                pack_word(
                                                    C_PT_SING,
                                                    temp_of(nword),
                                                    between(rng, 60u32, 109u32),
                                                ),
                                                aux_ctype[ni as usize],
                                                aux_tmp[ni as usize],
                                                aux_tmp2[ni as usize],
                                                aux_tmp3[ni as usize],
                                            );
                                        }
                                    } else {
                                        sing_life += 3u32;
                                        sing_tmp += 1i32;
                                        sing_temp += temp_of(nword);
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_KILL,
                                            0u32,
                                            0u32,
                                            0i32,
                                            0i32,
                                            0i32,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            voxel_a[index as usize] = pack_word(C_PT_SING, sing_temp, sing_life);
            aux_tmp[index as usize] = sing_tmp;
        }
    }

    // --- WARP (`WARP.cpp:53-87`). ---------------------------------------
    if ty == C_PT_WARP {
        let mut warp_life = seed_u32(index, life);
        let mut warp_temp = seed_f32(self_temp, self_temp);
        let mut wrote = seed_u32(index, 0u32);
        let tmp2 = aux_tmp2[index as usize];
        if tmp2 > 2000i32 {
            let pressure = pv_at(pv, x, y, z);
            let target = clampf(
                pressure + (tmp2 as f32 / 5000.0f32) * CFDS,
                -MAX_PRESSURE,
                MAX_PRESSURE,
            );
            pv_add(pv, x, y, z, target - pressure);
            warp_temp = 10000.0f32;
            wrote = 1u32;
            // `WARP.cpp:63-64` creates ELEC; that element is not part of the
            // 59-row table, so the spark branch is omitted.
        }
        let mut swapped = seed_u32(index, 0u32);
        let mut trade = seed_u32(index, 0u32);
        while trade < 5u32 && swapped == 0u32 {
            let ox = between(rng, 0u32, 2u32) as i32 - 1i32;
            rng = hash_u32(rng);
            let oy = between(rng, 0u32, 2u32) as i32 - 1i32;
            rng = hash_u32(rng);
            let oz = between(rng, 0u32, 2u32) as i32 - 1i32;
            rng = hash_u32(rng);
            let ni = world_index(xi + ox, yi + oy, zi + oz);
            if ni != C_NONE {
                let nword = voxel_a[ni as usize];
                let nty = nword & C_TY_MASK;
                if nty != 0u32 && nty != C_PT_WARP && nty != C_PT_CLNE {
                    // Exchange the two voxel words and their auxiliary fields.
                    push_mutation(
                        mutation_queue,
                        mutation_count,
                        ni,
                        MODE_REPLACE,
                        pack_word(C_PT_WARP, warp_temp, warp_life),
                        aux_ctype[index as usize],
                        aux_tmp[index as usize],
                        aux_tmp2[index as usize],
                        aux_tmp3[index as usize],
                    );
                    push_mutation(
                        mutation_queue,
                        mutation_count,
                        index,
                        MODE_REPLACE,
                        nword,
                        aux_ctype[ni as usize],
                        aux_tmp[ni as usize],
                        aux_tmp2[ni as usize],
                        aux_tmp3[ni as usize],
                    );
                    warp_life += 4u32;
                    swapped = 1u32;
                }
            }
            trade += 1u32;
        }
        if swapped != 0u32 || wrote != 0u32 {
            voxel_a[index as usize] = pack_word(C_PT_WARP, warp_temp, warp_life);
        }
    }
}

/// Runs the special-element pass over the tick's bricks.
pub(crate) fn apply(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    unsafe {
        special_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Dynamic(bufs.sim_count.clone().binding()),
            CubeDim::new_3d(CUBE_DIM, CUBE_DIM, CUBE_DIM),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.elements.table.clone(), C_ELEM_WORDS * 256),
            BufferArg::from_raw_parts(bufs.aux_ctype.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp2.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp3.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.air.pv.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(bufs.sim_list.clone(), C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(
                bufs.mutation_queue.clone(),
                (super::super::mutation::MUTATION_CAPACITY
                    * super::super::mutation::MUTATION_STRIDE) as usize,
            ),
            BufferArg::from_raw_parts(bufs.mutation_count.clone(), 3),
            engine.tick as u32,
        );
    }
}
