//! Simple 3D electronics: `SPRK` conduction through the two silicons.
//!
//! The user priority is a deliberately small electronics subset: sparks that
//! travel along `METL`, `PSCN` and `NSCN`. There is no battery, switch,
//! wireless, array, delay or sensor here; the pass is a literal port of
//! `SPRK.cpp`'s update with the unimplemented receivers removed:
//!
//! * life exhaustion restores the conductor the spark was sitting on
//!   (`SPRK.cpp:63-86`),
//! * `SPRK` on `SPRK` kills the spark (`SPRK.cpp:90-92`),
//! * the 5x5x5 conduction scan with the `abs(rx)+abs(ry) >= 4` range limit
//!   (`SPRK.cpp:199-291`),
//! * `tryConduct`'s P/N gating, where `PSCN` refuses an `NSCN` sender and
//!   every other pair conducts (`SPRK.cpp:327-360`),
//! * the water/salt-water special cases (`SPRK.cpp:368-376`) and the +10 K
//!   conductor heating (`SPRK.cpp:391-397`).
//!
//! The `pavg` insulation checks need `INSL`/`RSSS`, which are not part of the
//! 59-element 3D table, so they are skipped; every other branch is unchanged.
//!
//! # Ordering
//!
//! The pass runs after the movement commit and before the mutation drain, so
//! self changes are visible immediately and a newly sparked neighbour is
//! written by [`super::super::mutation`] in the same tick. A sparked voxel
//! starts with life 4; the next tick's life pass decrements it so the spark
//! conducts for three updates before restoring its conductor, exactly like
//! `PROP_LIFE_DEC` on the C++ side (`SPRK.cpp:44`).

use cubecl::prelude::*;

use super::super::Voxel3dBuffers;
use super::super::engine::Voxel3dEngine;
use super::super::mutation::{MODE_REPLACE, push as push_mutation};
use super::{
    C_BRICK_COUNT, C_ELEM_TPT_PROPS, C_ELEM_WORDS, C_LIFE_MASK, C_LIFE_SHIFT, C_PT_BMTL, C_PT_BRMT,
    C_PT_METL, C_PT_NSCN, C_PT_PSCN, C_PT_SLTW, C_PT_SPRK, C_PT_WATR, C_TEMP_MASK, C_TEMP_SHIFT,
    C_TPT_CONDUCTS, C_TY_MASK, iabsf, seed_f32, seed_u32,
};

/// Voxels along one brick edge.
const BRICK: u32 = 8;
/// Threads per brick: one per voxel.
const CUBE_DIM: u32 = 8;
/// Temperature a spark restores its conductor to (`SPRK.cpp:65-66`).
const SPARK_RESET_TEMP_K: f32 = 295.15;
/// The coldest conductor a spark can heat (`SPRK.cpp:395`).
const SPARK_HEAT_LIMIT_K: f32 = 673.0f32;
/// Per-conduction temperature increase (`SPRK.cpp:396`).
const SPARK_HEAT_K: f32 = 10.0f32;

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

/// Whether `sender` may conduct into `receiver`.
///
/// `SPRK.cpp:327-360`: `PSCN` is the only receiver in this subset with an
/// input rule (`if (sender != PT_NSCN) return true; return false;`); every
/// other conductor accepts every sender.
#[cube]
fn conducts_to(sender: u32, receiver: u32) -> bool {
    let mut ok = seed_u32(receiver, 1u32);
    if sender == C_PT_NSCN && receiver == C_PT_PSCN {
        ok = 0u32;
    }
    ok != 0u32
}

/// Conductors that gain 10 K when a spark reaches them (`SPRK.cpp:395`).
#[cube]
fn heats_with_spark(ty: u32) -> bool {
    let mut hot = seed_u32(ty, 0u32);
    if ty == C_PT_METL || ty == C_PT_BMTL || ty == C_PT_BRMT || ty == C_PT_PSCN || ty == C_PT_NSCN {
        hot = 1u32;
    }
    hot != 0u32
}

/// One thread per voxel of every simulated brick.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn spark_kernel(
    voxel_a: &mut [u32],
    table: &[u32],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    sim_list: &[u32],
    mutation_queue: &mut [u32],
    mutation_count: &mut [Atomic<u32>],
) {
    let brick = sim_list[CUBE_POS_X as usize];
    let x = (brick & 31u32) * BRICK + UNIT_POS_X;
    let y = ((brick >> 10u32) & 31u32) * BRICK + UNIT_POS_Y;
    let z = ((brick >> 5u32) & 31u32) * BRICK + UNIT_POS_Z;
    let index = (y << 16u32) | (z << 8u32) | x;
    let word = voxel_a[index as usize];
    if (word & C_TY_MASK) == C_PT_SPRK {
        let life = (word >> C_LIFE_SHIFT) & C_LIFE_MASK;
        let self_temp = ((word >> C_TEMP_SHIFT) & C_TEMP_MASK) as f32 * 0.125f32;
        let ct = aux_ctype[index as usize];

        if life == 0u32 {
            // `SPRK.cpp:63-86`: the spark cycle is over. Restore the conductor
            // stored in ctype (METL when the type is invalid or missing) and
            // consume the ctype. Water and salt water keep their long life so
            // the next touch does not re-ignite them immediately.
            let mut target = seed_u32(ct, ct);
            if target == 0u32
                || target >= 256u32
                || table[(target as usize) * C_ELEM_WORDS] != target
            {
                target = C_PT_METL;
            }
            let mut next_temp = seed_f32(self_temp, self_temp);
            if ct == C_PT_WATR || ct == C_PT_SLTW || ct == C_PT_PSCN || ct == C_PT_NSCN {
                next_temp = SPARK_RESET_TEMP_K;
            }
            let mut next_life = seed_u32(life, 4u32);
            if target == C_PT_WATR {
                next_life = 64u32;
            } else if target == C_PT_SLTW {
                next_life = 54u32;
            }
            voxel_a[index as usize] = pack_word(target, next_temp, next_life);
            aux_ctype[index as usize] = 0u32;
        } else {
            if ct == C_PT_SPRK {
                // `SPRK.cpp:90-92`: a spark on a spark burns out.
                voxel_a[index as usize] = 0u32;
                aux_ctype[index as usize] = 0u32;
                aux_tmp[index as usize] = 0i32;
                aux_tmp2[index as usize] = 0i32;
                aux_tmp3[index as usize] = 0i32;
            } else if life < 4u32 {
                // `SPRK.cpp:199-291` only conducts while the life counter is
                // below the freshly created value of four.
                let xi = x as i32;
                let yi = y as i32;
                let zi = z as i32;
                for dx in -2i32..=2i32 {
                    for dy in -2i32..=2i32 {
                        for dz in -2i32..=2i32 {
                            let dist = iabsf(dx) + iabsf(dy) + iabsf(dz);
                            if (dx != 0i32 || dy != 0i32 || dz != 0i32) && dist < 4i32 {
                                let nx = xi + dx;
                                let ny = yi + dy;
                                let nz = zi + dz;
                                // A negative coordinate wraps to a huge u32
                                // and fails this bound, as does 256 or more.
                                if (nx as u32) < 256u32
                                    && (ny as u32) < 256u32
                                    && (nz as u32) < 256u32
                                {
                                    let ni =
                                        ((ny as u32) << 16u32) | ((nz as u32) << 8u32) | nx as u32;
                                    let nword = voxel_a[ni as usize];
                                    let nty = nword & C_TY_MASK;
                                    if nty != 0u32 && nty != C_PT_SPRK {
                                        let nbase = (nty as usize) * C_ELEM_WORDS;
                                        let nprops = table[nbase + C_ELEM_TPT_PROPS];
                                        let nlife = (nword >> C_LIFE_SHIFT) & C_LIFE_MASK;
                                        if (nprops & C_TPT_CONDUCTS) != 0u32
                                            && nlife == 0u32
                                            && conducts_to(ct, nty)
                                        {
                                            let ntemp_in = ((nword >> C_TEMP_SHIFT) & C_TEMP_MASK)
                                                as f32
                                                * 0.125f32;
                                            let mut ntemp = seed_f32(ntemp_in, ntemp_in);
                                            if ntemp + SPARK_HEAT_K < SPARK_HEAT_LIMIT_K
                                                && heats_with_spark(nty)
                                            {
                                                ntemp += SPARK_HEAT_K;
                                            }
                                            let mut spark_life = seed_u32(nty, 4u32);
                                            if nty == C_PT_WATR {
                                                spark_life = 6u32;
                                            } else if nty == C_PT_SLTW {
                                                spark_life = 5u32;
                                            }
                                            push_mutation(
                                                mutation_queue,
                                                mutation_count,
                                                ni,
                                                MODE_REPLACE,
                                                pack_word(C_PT_SPRK, ntemp, spark_life),
                                                nty,
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
                }
            }
        }
    }
}

/// Runs the spark pass over the tick's bricks.
pub(crate) fn apply(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    unsafe {
        spark_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Dynamic(bufs.sim_count.clone().binding()),
            CubeDim::new_3d(CUBE_DIM, CUBE_DIM, CUBE_DIM),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.elements.table.clone(), C_ELEM_WORDS * 256),
            BufferArg::from_raw_parts(bufs.aux_ctype.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp2.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp3.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.sim_list.clone(), C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(
                bufs.mutation_queue.clone(),
                (super::super::mutation::MUTATION_CAPACITY
                    * super::super::mutation::MUTATION_STRIDE) as usize,
            ),
            BufferArg::from_raw_parts(bufs.mutation_count.clone(), 3),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pn_gating_matches_sprk_cpp() {
        // `SPRK.cpp:353-356`: PSCN refuses an NSCN sender.
        assert!(!conducts_to(C_PT_NSCN, C_PT_PSCN));
        assert!(conducts_to(C_PT_METL, C_PT_PSCN));
        assert!(conducts_to(C_PT_PSCN, C_PT_NSCN));
        assert!(conducts_to(C_PT_NSCN, C_PT_NSCN));
        assert!(conducts_to(C_PT_METL, C_PT_METL));
    }

    #[test]
    fn only_the_cpp_conductor_list_heats() {
        assert!(heats_with_spark(C_PT_METL));
        assert!(heats_with_spark(C_PT_PSCN));
        assert!(!heats_with_spark(C_PT_WATR));
    }
}
