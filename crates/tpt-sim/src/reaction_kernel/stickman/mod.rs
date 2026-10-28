//! Stickman (`FIGH` / `STKM` / `STKM2` / `SPAWN` / `SPAWN2`) device-resident SoA pass.
//!
//! # Submodules
//!
//! * [`rig`]: Verlet leg integration, joint constraints, stickman lifecycle, and absorption.
//! * [`fighter`]: Fighter AI command selection and targeting.
//!
//! # TPT references
//!
//! * `STKM.cpp:120-633` — `Element_STKM_run_stickman`: gravity, Verlet leg integration,
//!   distance constraints, obstacle collisions, element absorption, rocket boots, attacks.
//! * `STKM.cpp:714-749` — `Element_STKM_init_legs`: initial joint positions for the 4-segment rig.
//! * `FIGH.cpp:60-154` — fighter AI: target selection seeking `STKM`/`STKM2`, left/right
//!   obstacle avoidance, jump triggers, attack radius (< 600 px²).
//! * `Stickman.h:3-16` — `MAX_FIGHTERS = 100`, player/fighter data structures.

#![allow(clippy::identity_op)]

mod fighter;
mod rig;

#[cfg(test)]
pub(crate) use fighter::figh_ai_select_command_host;
#[cfg(test)]
pub(crate) use rig::init_legs_host;
pub use rig::stickman_kernel;

use cubecl::prelude::*;
use tpt_core::constants::*;

use crate::buffers::SimBuffers;

use super::PassDims;
use super::shared::launch_grid;

const C_PT_NONE: u32 = 0;
const C_PT_DUST: u32 = 1;
const C_PT_GRVT: u32 = 39;
const C_PT_PLNT: u32 = 45;
const C_PT_SPAWN: u32 = 127;
const C_PT_STKM: u32 = 59;
const C_PT_STOR: u32 = 74;
const C_PT_PLUT: u32 = 89;
const C_PT_LIGH: u32 = 107;
const C_PT_STKM2: u32 = 137;
const C_PT_SPAWN2: u32 = 126;
const C_PT_FIGH: u32 = 159;
const C_PT_NUM: u32 = 200;

const C_STKM_MAX_FIGHTERS: usize = 100;
const C_STKM_PLAYER1: usize = 100;
const C_STKM_PLAYER2: usize = 101;
const C_STKM_SLOTS: usize = 102;
const C_STKM_LEGS_STRIDE: usize = 16;
const C_STKM_ACCS_STRIDE: usize = 8;
const C_STKM_META_STRIDE: usize = 8;

const C_PROP_DEADLY: u32 = 0x0000_0400;
const C_PROP_RADIOACTIVE: u32 = 0x0000_2000;
const C_FLAG_MOVABLE: u32 = 0x08;

const _: () = assert!(C_PT_NONE == PT_NONE);
const _: () = assert!(C_PT_DUST == PT_DUST);
const _: () = assert!(C_PT_GRVT == PT_GRVT);
const _: () = assert!(C_PT_PLNT == PT_PLNT);
const _: () = assert!(C_PT_SPAWN == PT_SPAWN);
const _: () = assert!(C_PT_STKM == PT_STKM);
const _: () = assert!(C_PT_STOR == PT_STOR);
const _: () = assert!(C_PT_PLUT == PT_PLUT);
const _: () = assert!(C_PT_LIGH == PT_LIGH);
const _: () = assert!(C_PT_STKM2 == PT_STKM2);
const _: () = assert!(C_PT_SPAWN2 == PT_SPAWN2);
const _: () = assert!(C_PT_FIGH == PT_FIGH);
const _: () = assert!(C_PT_NUM == PT_NUM as u32);

const _: () = assert!(C_STKM_MAX_FIGHTERS == STKM_MAX_FIGHTERS);
const _: () = assert!(C_STKM_PLAYER1 == STKM_PLAYER1);
const _: () = assert!(C_STKM_PLAYER2 == STKM_PLAYER2);
const _: () = assert!(C_STKM_SLOTS == STKM_SLOTS);
const _: () = assert!(C_STKM_LEGS_STRIDE == STKM_LEGS_STRIDE);
const _: () = assert!(C_STKM_ACCS_STRIDE == STKM_ACCS_STRIDE);
const _: () = assert!(C_STKM_META_STRIDE == STKM_META_STRIDE);

const _: () = assert!(C_PROP_DEADLY == PROP_DEADLY);
const _: () = assert!(C_PROP_RADIOACTIVE == PROP_RADIOACTIVE);
const _: () = assert!(C_FLAG_MOVABLE == FLAG_MOVABLE);

/// Check whether `(x, y)` contains a solid/powder/liquid obstacle.
#[cube]
pub fn is_obstacle(pmap: &[u32], xres: u32, yres: u32, x: i32, y: i32) -> bool {
    if x < 0i32 || x >= xres as i32 || y < 0i32 || y >= yres as i32 {
        true
    } else {
        let packed = pmap[(y as u32 * xres + x as u32) as usize];
        (packed & 0x1FFu32) != 0u32
    }
}

pub(super) fn launch(bufs: &SimBuffers, dims: &PassDims) {
    let (cubes, dim) = launch_grid(dims.used_slots);
    unsafe {
        stickman_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_stkm_legs.clone(), STKM_SLOTS * STKM_LEGS_STRIDE),
            BufferArg::from_raw_parts(bufs.gpu_stkm_accs.clone(), STKM_SLOTS * STKM_ACCS_STRIDE),
            BufferArg::from_raw_parts(bufs.gpu_stkm_meta.clone(), STKM_SLOTS * STKM_META_STRIDE),
            BufferArg::from_raw_parts(bufs.gpu_free_slots.clone(), SimBuffers::GPU_SPAWN_COUNT + 2),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.used_slots as u32,
            (dims.elem_len / 32) as u32,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leg_initialisation() {
        let mut legs = vec![0.0f32; STKM_SLOTS * STKM_LEGS_STRIDE];
        let mut accs = vec![0.0f32; STKM_SLOTS * STKM_ACCS_STRIDE];
        let mut meta = vec![0u32; STKM_SLOTS * STKM_META_STRIDE];

        let sid = 5usize;
        let hx = 100.0f32;
        let hy = 50.0f32;
        let elem = PT_FIRE;

        init_legs_host(&mut legs, &mut accs, &mut meta, sid, hx, hy, elem);

        let lb = sid * 16;
        assert_eq!(legs[lb + 0], 99.0);
        assert_eq!(legs[lb + 1], 56.0);
        assert_eq!(legs[lb + 2], 99.0);
        assert_eq!(legs[lb + 3], 56.0);

        assert_eq!(legs[lb + 4], 97.0);
        assert_eq!(legs[lb + 5], 62.0);
        assert_eq!(legs[lb + 6], 97.0);
        assert_eq!(legs[lb + 7], 62.0);

        assert_eq!(legs[lb + 8], 101.0);
        assert_eq!(legs[lb + 9], 56.0);
        assert_eq!(legs[lb + 10], 101.0);
        assert_eq!(legs[lb + 11], 56.0);

        assert_eq!(legs[lb + 12], 103.0);
        assert_eq!(legs[lb + 13], 62.0);
        assert_eq!(legs[lb + 14], 103.0);
        assert_eq!(legs[lb + 15], 62.0);

        let ab = sid * 8;
        for k in 0..8 {
            assert_eq!(accs[ab + k], 0.0);
        }

        let mb = sid * 8;
        assert_eq!(meta[mb + 0], 1); // spwn
        assert_eq!(meta[mb + 1], 0); // comm
        assert_eq!(meta[mb + 2], 0); // pcomm
        assert_eq!(meta[mb + 3], PT_FIRE); // elem
        assert_eq!(meta[mb + 4], 0); // frames
        assert_eq!(meta[mb + 5], 0); // rocket_boots
        assert_eq!(meta[mb + 6], 0); // fan
    }

    #[test]
    fn test_fighter_ai_command_selection() {
        let xres = 612u32;
        let yres = 384u32;
        let pmap = vec![0u32; (xres * yres) as usize];
        let mut legs = vec![0.0f32; STKM_SLOTS * STKM_LEGS_STRIDE];
        let mut accs = vec![0.0f32; STKM_SLOTS * STKM_ACCS_STRIDE];
        let mut meta = vec![0u32; STKM_SLOTS * STKM_META_STRIDE];

        let fid = 0usize;
        init_legs_host(&mut legs, &mut accs, &mut meta, fid, 100.0, 50.0, PT_DUST);

        // Case 1: Target to the left (STKM at x=50, y=50) -> seeks left (0x01)
        init_legs_host(
            &mut legs,
            &mut accs,
            &mut meta,
            STKM_PLAYER1,
            50.0,
            50.0,
            PT_DUST,
        );
        let comm_left =
            figh_ai_select_command_host(&pmap, 612, 384, 100.0, 50.0, &legs, &meta, fid);
        assert_eq!(comm_left & 0x01, 0x01);

        // Case 2: Target to the right (STKM at x=200, y=50) -> seeks right (0x02)
        init_legs_host(
            &mut legs,
            &mut accs,
            &mut meta,
            STKM_PLAYER1,
            200.0,
            50.0,
            PT_DUST,
        );
        let comm_right =
            figh_ai_select_command_host(&pmap, 612, 384, 100.0, 50.0, &legs, &meta, fid);
        assert_eq!(comm_right & 0x02, 0x02);

        // Case 3: Target close (dist^2 < 600, e.g. dx=10, dy=0) -> attack (0x08)
        init_legs_host(
            &mut legs,
            &mut accs,
            &mut meta,
            STKM_PLAYER1,
            110.0,
            50.0,
            PT_DUST,
        );
        let comm_atk = figh_ai_select_command_host(&pmap, 612, 384, 100.0, 50.0, &legs, &meta, fid);
        assert_eq!(comm_atk & 0x08, 0x08);
    }
}
