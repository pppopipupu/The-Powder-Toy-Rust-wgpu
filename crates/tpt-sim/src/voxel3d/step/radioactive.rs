//! 3D radioactive elements (`SC_NUCLEAR`): URAN, ISOZ/ISZS, NEUT and PHOT.
//!
//! This is the user-priority category, so every branch is a literal port of the
//! C++ update and is named with its source file. The pass runs after thermal
//! conduction and before the movement passes (C++ runs every element update
//! before particle movement):
//!
//! * self changes (temperature, decay, life) are written straight into
//!   `voxel_a`, because this pass owns its cell after the commit;
//! * neighbour effects append to the mutation queue, exactly like TPT's
//!   `create_part`/`part_change_type`/`kill_part` (`Simulation.cpp:1797-1960`).
//!
//! Energy movement: TPT stores photons and neutrons in a separate `photons`
//! layer and integrates their `vx/vy` (`Simulation.cpp:2951-3160`). The 3D
//! model has one voxel per cell, so each energy voxel keeps a direction in
//! `aux_tmp2` (`0..=5`, `+x,-x,+y,-y,+z,-z`) and an initialised flag in
//! `aux_tmp3`, and advances up to three cells per tick along that direction;
//! photons pass through `PROP_PHOTPASS` matter and are absorbed by the first
//! opaque cell. Neutrons pass through `PROP_NEUTPENETRATE` matter by swapping
//! with it (`Simulation.cpp:1475-1492`).

use cubecl::prelude::*;

use super::super::Voxel3dBuffers;
use super::super::engine::Voxel3dEngine;
use super::super::mutation::{MODE_CREATE, MODE_KILL, MODE_REPLACE, push as push_mutation};
use super::{
    C_BRICK_COUNT, C_ELEM_FLAMMABLE, C_ELEM_TPT_PROPS, C_ELEM_WORDS, C_LIFE_MASK, C_LIFE_SHIFT,
    C_NONE, C_PT_ACID, C_PT_COAL, C_PT_DSTW, C_PT_DUST, C_PT_EXOT, C_PT_FIRE, C_PT_GAS, C_PT_GLAS,
    C_PT_GPMP, C_PT_GRVT, C_PT_GUNP, C_PT_ISOZ, C_PT_ISZS, C_PT_LAVA, C_PT_NEUT, C_PT_NSCN,
    C_PT_OIL, C_PT_PHOT, C_PT_PLUT, C_PT_PSCN, C_PT_SPRK, C_PT_URAN, C_PT_WATR, C_PT_WOOD,
    C_TEMP_MASK, C_TEMP_SHIFT, C_TPT_NEUTPENETRATE, C_TPT_PHOTPASS, C_TY_MASK, clampf, hash_u32,
    mix_index, seed_f32, seed_i32, seed_u32,
};

/// Voxels along one brick edge.
const BRICK: u32 = 8;
/// Threads per brick: one per voxel.
const CUBE_DIM: u32 = 8;
/// World edge length.
const WORLD: i32 = 256;
/// TPT `MAX_TEMP` (`ElementDefs.h:5`); the packed field saturates earlier.
const MAX_TEMP_K: f32 = 9999.0;
/// TPT `MIN_TEMP`.
const MIN_TEMP_K: f32 = 0.0;
/// Fixed-point units per TPT pressure unit.
const PV_SCALE_F: f32 = 1024.0;
/// `CFDS = 4 / CELL` (`SimulationConfig.h:33`) with `CELL = 4`.
const CFDS: f32 = 1.0;
/// Temperature of a freshly created photon (`PHOT.cpp:35`).
const PHOT_DEFAULT_TEMP_K: f32 = 1195.15;
/// Life of a freshly created photon (`PHOT.cpp:50`).
const PHOT_DEFAULT_LIFE: u32 = 680;
/// Photon wavelength mask (`PHOT.cpp:51`).
const PHOT_WAVELENGTH: u32 = 0x3FFF_FFFF;
/// Temperature of a freshly created neutron (`NEUT.cpp:36`).
const NEUT_DEFAULT_TEMP_K: f32 = 299.15;
/// Fire temperature used by the ignition branch (`FIRE.cpp:312`).
const FIRE_DEFAULT_TEMP_K: f32 = 695.15;
/// RNG salt of the per-neighbour chain.
const NEIGHBOUR_SALT: u32 = 0xC2B2_AE35u32;

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

/// TPT `pv` at a world coordinate, in pressure units.
#[cube]
fn pv_at(pv: &mut [Atomic<i32>], x: u32, y: u32, z: u32) -> f32 {
    pv[air_cell(x, y, z) as usize].load() as f32 * (1.0f32 / PV_SCALE_F)
}

/// Adds a TPT pressure impulse to the coarse cell of `(x, y, z)`.
#[cube]
fn pv_add(pv: &mut [Atomic<i32>], x: u32, y: u32, z: u32, impulse: f32) {
    pv[air_cell(x, y, z) as usize].fetch_add((impulse * PV_SCALE_F) as i32);
}

/// Direction stored in `aux_tmp2`: `0..=5` as `+x,-x,+y,-y,+z,-z`.
#[cube]
fn dir_delta(dir: u32) -> (i32, i32, i32) {
    let mut dx = seed_i32(dir as i32, 0i32);
    let mut dy = seed_i32(dir as i32, 0i32);
    let mut dz = seed_i32(dir as i32, 0i32);
    if dir == 0u32 {
        dx = 1i32;
    } else if dir == 1u32 {
        dx = -1i32;
    } else if dir == 2u32 {
        dy = 1i32;
    } else if dir == 3u32 {
        dy = -1i32;
    } else if dir == 4u32 {
        dz = 1i32;
    } else {
        dz = -1i32;
    }
    (dx, dy, dz)
}

/// World index of a coordinate, or [`C_NONE`] outside the world.
#[cube]
fn world_index(x: i32, y: i32, z: i32) -> u32 {
    let mut out = seed_i32(x, C_NONE as i32);
    // A negative coordinate wraps to a huge u32 and fails the bound.
    if (x as u32) < WORLD as u32 && (y as u32) < WORLD as u32 && (z as u32) < WORLD as u32 {
        out = (((y as u32) << 16u32) | ((z as u32) << 8u32) | x as u32) as i32;
    }
    out as u32
}

/// `PROP_*` bits of element `ty` from the GPU table.
#[cube]
fn props_of(table: &[u32], ty: u32) -> u32 {
    table[(ty as usize) * C_ELEM_WORDS + C_ELEM_TPT_PROPS]
}

/// One thread per voxel of every simulated brick.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn radioactive_kernel(
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

    // --- URAN (`URAN.cpp:40-51`). ----------------------------------------
    if ty == C_PT_URAN {
        let pressure = pv_at(pv, x, y, z);
        if pressure > 0.0f32 {
            let old_temp = temp_of(word);
            // The seed must be a value the branch actually keeps: `MIN_TEMP`
            // resets to 0.01 K, everything else is heated and clamped.
            let mut next_temp = seed_f32(old_temp, 0.01f32);
            if old_temp != MIN_TEMP_K {
                next_temp = clampf(
                    old_temp * (1.0f32 + pressure / 2000.0f32) + MIN_TEMP_K,
                    MIN_TEMP_K,
                    MAX_TEMP_K,
                );
            }
            voxel_a[index as usize] =
                pack_word(C_PT_URAN, next_temp, (word >> C_LIFE_SHIFT) & C_LIFE_MASK);
        }
    }

    // --- ISOZ / ISZS (`ISOZ.cpp:39-52`, `ISZS.cpp:39-52`). ---------------
    if ty == C_PT_ISOZ || ty == C_PT_ISZS {
        let seed_a = hash_u32(mix_index(index, 0x9E37_79B1u32) ^ tick ^ (ty << 8u32));
        let seed_b = hash_u32(seed_a);
        let pressure = pv_at(pv, x, y, z);
        let numerator = (-4.0f32 * pressure) as i32;
        if chance(seed_a, 1u32, 200u32)
            && numerator > 0i32
            && chance(seed_b, numerator as u32, 1000u32)
        {
            // `create_part(i, x, y, PT_PHOT)` resets the cell to PHOT's
            // DefaultProperties; the velocity roll becomes a random direction.
            voxel_a[index as usize] = pack_word(C_PT_PHOT, PHOT_DEFAULT_TEMP_K, PHOT_DEFAULT_LIFE);
            aux_ctype[index as usize] = PHOT_WAVELENGTH;
            aux_tmp[index as usize] = 0i32;
            aux_tmp2[index as usize] = (hash_u32(seed_b ^ index) % 6u32) as i32;
            aux_tmp3[index as usize] = 1i32;
        }
    }

    // --- NEUT (`NEUT.cpp:57-249`). ---------------------------------------
    if ty == C_PT_NEUT {
        let mut dir = aux_tmp2[index as usize] as u32;
        if aux_tmp3[index as usize] != 1i32 {
            dir = hash_u32(mix_index(index, 0x27D4_EB2Fu32) ^ tick) % 6u32;
        }
        aux_tmp2[index as usize] = dir as i32;
        aux_tmp3[index as usize] = 1i32;

        let xi = x as i32;
        let yi = y as i32;
        let zi = z as i32;
        let pressure_factor = (3.0f32 + pv_at(pv, x, y, z)) as i32;

        for dx in -1i32..=1i32 {
            for dy in -1i32..=1i32 {
                for dz in -1i32..=1i32 {
                    if dx != 0i32 || dy != 0i32 || dz != 0i32 {
                        let ni = world_index(xi + dx, yi + dy, zi + dz);
                        if ni != C_NONE {
                            let nword = voxel_a[ni as usize];
                            let nty = nword & C_TY_MASK;
                            if nty != 0u32 {
                                let nseed = hash_u32(mix_index(ni, NEIGHBOUR_SALT) ^ tick ^ index);
                                if nty == C_PT_WATR {
                                    // `NEUT.cpp:70-72`: 3/20 chance to distil.
                                    if chance(nseed, 3u32, 20u32) {
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_REPLACE,
                                            pack_word(
                                                C_PT_DSTW,
                                                temp_of(nword),
                                                (nword >> C_LIFE_SHIFT) & C_LIFE_MASK,
                                            ),
                                            aux_ctype[ni as usize],
                                            aux_tmp[ni as usize],
                                            aux_tmp2[ni as usize],
                                            aux_tmp3[ni as usize],
                                        );
                                    }
                                } else if nty == C_PT_GUNP {
                                    // `NEUT.cpp:115-118`: 3/200 chance to DUST.
                                    if chance(nseed, 3u32, 200u32) {
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_REPLACE,
                                            pack_word(
                                                C_PT_DUST,
                                                temp_of(nword),
                                                (nword >> C_LIFE_SHIFT) & C_LIFE_MASK,
                                            ),
                                            aux_ctype[ni as usize],
                                            aux_tmp[ni as usize],
                                            aux_tmp2[ni as usize],
                                            aux_tmp3[ni as usize],
                                        );
                                    }
                                } else if nty == C_PT_COAL {
                                    // `NEUT.cpp:151-154`: 1/20 chance to WOOD.
                                    if chance(nseed, 1u32, 20u32) {
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_REPLACE,
                                            pack_word(
                                                C_PT_WOOD,
                                                temp_of(nword),
                                                (nword >> C_LIFE_SHIFT) & C_LIFE_MASK,
                                            ),
                                            aux_ctype[ni as usize],
                                            aux_tmp[ni as usize],
                                            aux_tmp2[ni as usize],
                                            aux_tmp3[ni as usize],
                                        );
                                    }
                                } else if nty == C_PT_OIL {
                                    // `NEUT.cpp:144-148`: 3/200 chance to GAS.
                                    if chance(nseed, 3u32, 200u32) {
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_REPLACE,
                                            pack_word(
                                                C_PT_GAS,
                                                temp_of(nword),
                                                (nword >> C_LIFE_SHIFT) & C_LIFE_MASK,
                                            ),
                                            aux_ctype[ni as usize],
                                            aux_tmp[ni as usize],
                                            aux_tmp2[ni as usize],
                                            aux_tmp3[ni as usize],
                                        );
                                    }
                                } else if nty == C_PT_ACID {
                                    // `NEUT.cpp:170-174`: 1/20 chance to ISOZ.
                                    if chance(nseed, 1u32, 20u32) {
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_REPLACE,
                                            pack_word(
                                                C_PT_ISOZ,
                                                temp_of(nword),
                                                (nword >> C_LIFE_SHIFT) & C_LIFE_MASK,
                                            ),
                                            aux_ctype[ni as usize],
                                            aux_tmp[ni as usize],
                                            aux_tmp2[ni as usize],
                                            aux_tmp3[ni as usize],
                                        );
                                    }
                                } else if nty == C_PT_EXOT {
                                    // `NEUT.cpp:181-185`: 1/20 chance life = 1500.
                                    // EXOT keeps its full-range life in `tmp3`
                                    // (`EXOT.cpp:48,78`), so the rewrite must
                                    // land there, not in the 8-bit word field.
                                    if chance(nseed, 1u32, 20u32) {
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_REPLACE,
                                            pack_word(C_PT_EXOT, temp_of(nword), 1500u32),
                                            aux_ctype[ni as usize],
                                            aux_tmp[ni as usize],
                                            aux_tmp2[ni as usize],
                                            1500i32,
                                        );
                                    }
                                } else if nty == C_PT_PLUT {
                                    // `NEUT.cpp:75-107`: fission under pressure.
                                    if pressure_factor > 0i32
                                        && chance(nseed, pressure_factor as u32, 1000u32)
                                    {
                                        let second = hash_u32(nseed ^ 0x5BD1_E995u32);
                                        if chance(second, 1u32, 3u32) {
                                            let third = hash_u32(second);
                                            if chance(third, 2u32, 3u32) {
                                                // LAVA(PLUT): tmp = 100, ctype = PLUT.
                                                push_mutation(
                                                    mutation_queue,
                                                    mutation_count,
                                                    ni,
                                                    MODE_REPLACE,
                                                    pack_word(
                                                        C_PT_LAVA,
                                                        MAX_TEMP_K,
                                                        between(hash_u32(nseed), 240u32, 359u32),
                                                    ),
                                                    C_PT_PLUT,
                                                    100i32,
                                                    aux_tmp2[ni as usize],
                                                    aux_tmp3[ni as usize],
                                                );
                                            } else {
                                                push_mutation(
                                                    mutation_queue,
                                                    mutation_count,
                                                    ni,
                                                    MODE_REPLACE,
                                                    pack_word(
                                                        C_PT_URAN,
                                                        MAX_TEMP_K,
                                                        (nword >> C_LIFE_SHIFT) & C_LIFE_MASK,
                                                    ),
                                                    aux_ctype[ni as usize],
                                                    aux_tmp[ni as usize],
                                                    aux_tmp2[ni as usize],
                                                    aux_tmp3[ni as usize],
                                                );
                                            }
                                        } else {
                                            // 2xNEUT: the target becomes a neutron.
                                            push_mutation(
                                                mutation_queue,
                                                mutation_count,
                                                ni,
                                                MODE_REPLACE,
                                                pack_word(
                                                    C_PT_NEUT,
                                                    NEUT_DEFAULT_TEMP_K,
                                                    between(nseed, 480u32, 959u32),
                                                ),
                                                aux_ctype[ni as usize],
                                                0i32,
                                                (hash_u32(nseed) % 6u32) as i32,
                                                1i32,
                                            );
                                        }
                                        pv_add(pv, x, y, z, 10.0f32 * CFDS);
                                    }
                                } else if nty == C_PT_GLAS {
                                    // `Simulation.cpp:1370-1374`: Cherenkov light.
                                    if chance(nseed, 1u32, 10u32) {
                                        let crev = hash_u32(nseed ^ 0x85EB_CA6Bu32);
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_CREATE,
                                            pack_word(C_PT_PHOT, temp_of(nword), PHOT_DEFAULT_LIFE),
                                            PHOT_WAVELENGTH,
                                            0i32,
                                            (crev % 6u32) as i32,
                                            1i32,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Movement (`Simulation.cpp:1466-1492`). The neutron swaps with
        // NEUTPENETRATE matter and is stopped by everything else.
        let (ddx, ddy, ddz) = dir_delta(dir);
        let dest = world_index(xi + ddx, yi + ddy, zi + ddz);
        if dest == C_NONE {
            // Off-screen energy particles are killed (`Simulation.cpp:1550-1556`).
            push_mutation(
                mutation_queue,
                mutation_count,
                index,
                MODE_KILL,
                0u32,
                0u32,
                0i32,
                0i32,
                0i32,
            );
        } else {
            let dword = voxel_a[dest as usize];
            let dty = dword & C_TY_MASK;
            let self_word = pack_word(
                C_PT_NEUT,
                temp_of(word),
                (word >> C_LIFE_SHIFT) & C_LIFE_MASK,
            );
            if dty == 0u32 {
                push_mutation(
                    mutation_queue,
                    mutation_count,
                    dest,
                    MODE_CREATE,
                    self_word,
                    aux_ctype[index as usize],
                    aux_tmp[index as usize],
                    aux_tmp2[index as usize],
                    aux_tmp3[index as usize],
                );
                push_mutation(
                    mutation_queue,
                    mutation_count,
                    index,
                    MODE_KILL,
                    0u32,
                    0u32,
                    0i32,
                    0i32,
                    0i32,
                );
            } else if (props_of(table, dty) & C_TPT_NEUTPENETRATE) != 0u32 {
                push_mutation(
                    mutation_queue,
                    mutation_count,
                    dest,
                    MODE_REPLACE,
                    self_word,
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
                    dword,
                    aux_ctype[dest as usize],
                    aux_tmp[dest as usize],
                    aux_tmp2[dest as usize],
                    aux_tmp3[dest as usize],
                );
            }
        }
    }

    // --- PHOT (`PHOT.cpp:59-146`, wavelength masked). ---------------------
    if ty == C_PT_PHOT {
        let wavelength = aux_ctype[index as usize];
        if (wavelength & PHOT_WAVELENGTH) == 0u32 {
            // `PHOT.cpp:63-66`: a photon with no wavelength dies.
            voxel_a[index as usize] = 0u32;
            aux_ctype[index as usize] = 0u32;
            aux_tmp2[index as usize] = 0i32;
            aux_tmp3[index as usize] = 0i32;
        } else {
            let mut dir = aux_tmp2[index as usize] as u32;
            if aux_tmp3[index as usize] != 1i32 {
                dir = hash_u32(mix_index(index, 0x9E37_79B1u32) ^ tick) % 6u32;
            }
            aux_tmp2[index as usize] = dir as i32;
            aux_tmp3[index as usize] = 1i32;
            let (ddx, ddy, ddz) = dir_delta(dir);
            let self_temp = temp_of(word);
            let mut cx = x as i32;
            let mut cy = y as i32;
            let mut cz = z as i32;
            let mut dest = seed_u32(index, C_NONE);
            let mut blocked = seed_u32(index, 0u32);
            let mut steps = seed_u32(index, 0u32);
            while steps < 3u32 && blocked == 0u32 {
                cx += ddx;
                cy += ddy;
                cz += ddz;
                let ni = world_index(cx, cy, cz);
                if ni == C_NONE {
                    blocked = 2u32;
                } else {
                    let nword = voxel_a[ni as usize];
                    let nty = nword & C_TY_MASK;
                    if nty == 0u32 {
                        dest = ni;
                    } else if (props_of(table, nty) & C_TPT_PHOTPASS) != 0u32 {
                        // Glass/water and other PHOTPASS matter let the photon
                        // continue; it still ends the tick in the last empty
                        // cell it crossed.
                    } else {
                        blocked = 1u32;
                        let nseed = hash_u32(mix_index(ni, NEIGHBOUR_SALT) ^ tick ^ index);
                        if (nty == C_PT_ISOZ || nty == C_PT_ISZS) && chance(nseed, 1u32, 400u32) {
                            // `PHOT.cpp:78-95`: 2xPHOT, pressure drop.
                            let nnseed = hash_u32(nseed ^ 0x27D4_EB2Fu32);
                            push_mutation(
                                mutation_queue,
                                mutation_count,
                                ni,
                                MODE_REPLACE,
                                pack_word(C_PT_PHOT, temp_of(nword), PHOT_DEFAULT_LIFE),
                                PHOT_WAVELENGTH,
                                0i32,
                                (nnseed % 6u32) as i32,
                                1i32,
                            );
                            pv_add(pv, x, y, z, -15.0f32 * CFDS);
                        } else if nty == C_PT_PSCN {
                            // `Simulation.cpp:1568-1580` photoelectric effect:
                            // a PSCN with a face-adjacent NSCN sparks.
                            let nx = ni & 255u32;
                            let nz = (ni >> 8u32) & 255u32;
                            let ny = (ni >> 16u32) & 255u32;
                            let mut has_nscn = seed_u32(ni, 0u32);
                            for ox in -1i32..=1i32 {
                                for oz in -1i32..=1i32 {
                                    for oy in -1i32..=1i32 {
                                        let faces = (ox == 0i32) as u32
                                            + (oy == 0i32) as u32
                                            + (oz == 0i32) as u32;
                                        if faces == 2u32 {
                                            let ci = world_index(
                                                nx as i32 + ox,
                                                ny as i32 + oy,
                                                nz as i32 + oz,
                                            );
                                            if ci != C_NONE
                                                && (voxel_a[ci as usize] & C_TY_MASK) == C_PT_NSCN
                                            {
                                                has_nscn = 1u32;
                                            }
                                        }
                                    }
                                }
                            }
                            if has_nscn != 0u32 && ((nword >> C_LIFE_SHIFT) & C_LIFE_MASK) == 0u32 {
                                push_mutation(
                                    mutation_queue,
                                    mutation_count,
                                    ni,
                                    MODE_REPLACE,
                                    pack_word(C_PT_SPRK, temp_of(nword), 4u32),
                                    C_PT_PSCN,
                                    aux_tmp[ni as usize],
                                    aux_tmp2[ni as usize],
                                    aux_tmp3[ni as usize],
                                );
                            }
                        } else if nty == C_PT_GPMP
                            && ((nword >> C_LIFE_SHIFT) & C_LIFE_MASK) == 0u32
                        {
                            // `Simulation.cpp:1355-1362`: PHOT + GPMP -> GRVT.
                            // This replaces the photon in place, so the pass
                            // must not run its absorption kill afterwards.
                            voxel_a[index as usize] = pack_word(
                                C_PT_GRVT,
                                temp_of(word),
                                (word >> C_LIFE_SHIFT) & C_LIFE_MASK,
                            );
                            aux_ctype[index as usize] = 0u32;
                            aux_tmp[index as usize] = (temp_of(nword) - 273.15f32) as i32;
                            blocked = 3u32;
                        } else {
                            // `FIRE.cpp:304-321` via `PHOT.cpp:65-67`: a hot
                            // photon ignites flammable matter one time in ten.
                            let flammable = table[(nty as usize) * C_ELEM_WORDS + C_ELEM_FLAMMABLE];
                            if self_temp > 506.0f32
                                && chance(hash_u32(nseed ^ 0x85EB_CA6Bu32), 1u32, 10u32)
                                && flammable > 0u32
                                && chance(nseed, flammable, 1000u32)
                            {
                                push_mutation(
                                    mutation_queue,
                                    mutation_count,
                                    ni,
                                    MODE_REPLACE,
                                    pack_word(
                                        C_PT_FIRE,
                                        FIRE_DEFAULT_TEMP_K + flammable as f32 * 0.5f32,
                                        between(nseed, 180u32, 259u32),
                                    ),
                                    0u32,
                                    0i32,
                                    aux_tmp2[ni as usize],
                                    aux_tmp3[ni as usize],
                                );
                            }
                        }
                    }
                }
                steps += 1u32;
            }
            if blocked != 3u32 && dest != C_NONE && dest != index {
                // The photon ends the tick in the last empty cell it crossed;
                // a solid in front is absorbed next tick (`Simulation.cpp`
                // moves photons along `vx/vy` and only interacts on contact).
                push_mutation(
                    mutation_queue,
                    mutation_count,
                    dest,
                    MODE_CREATE,
                    pack_word(C_PT_PHOT, self_temp, (word >> C_LIFE_SHIFT) & C_LIFE_MASK),
                    aux_ctype[index as usize],
                    aux_tmp[index as usize],
                    aux_tmp2[index as usize],
                    aux_tmp3[index as usize],
                );
                push_mutation(
                    mutation_queue,
                    mutation_count,
                    index,
                    MODE_KILL,
                    0u32,
                    0u32,
                    0i32,
                    0i32,
                    0i32,
                );
            } else if blocked != 0u32 && blocked != 3u32 {
                push_mutation(
                    mutation_queue,
                    mutation_count,
                    index,
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

/// Runs the radioactive pass over the tick's bricks.
pub(crate) fn apply(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    unsafe {
        radioactive_kernel::launch_unchecked(
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
