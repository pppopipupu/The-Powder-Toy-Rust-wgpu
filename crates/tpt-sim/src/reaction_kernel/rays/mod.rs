//! Ray (`ARAY` / `DRAY`) and piston (`PSTN` / `FRME`) device passes.
//!
//! Port of The Powder Toy's ray emitters and pistons:
//! * `ARAY.cpp`: Ray emitter firing `BRAY` beams, interacting with `FILT`
//!   wavelengths, releasing `STOR`, and sparking conductors.
//! * `DRAY.cpp`: Duplicator ray replicating lines of matter/energy particles.
//! * `PSTN.cpp` & `FRME.cpp`: Pistons extending and retracting arm particles,
//!   pushing and pulling stacks directly or across perpendicular `FRME` beams.
//!
//! # Submodules
//!
//! * [`emitter`]: Ray emitter firing `BRAY` beams, interacting with `FILT` wavelengths, releasing `STOR`, and sparking conductors.
//! * [`pistons`]: Pistons extending and retracting arm particles, pushing and pulling stacks directly or across perpendicular `FRME` beams.

mod emitter;
mod pistons;

pub use emitter::*;
pub use pistons::*;

use cubecl::prelude::*;
use tpt_core::constants::*;

use crate::buffers::SimBuffers;

use super::PassDims;
use super::shared::launch_grid;

const C_PT_NONE: u32 = 0;
const C_PT_SPRK: u32 = 12;
const C_PT_PSCN: u32 = 50;
const C_PT_STKM: u32 = 59;
const C_PT_SWCH: u32 = 60;
const C_PT_INWR: u32 = 63;
const C_PT_STOR: u32 = 74;
const C_PT_LIFE: u32 = 105;
const C_PT_INST: u32 = 115;
const C_PT_WIFI: u32 = 133;
const C_PT_FILT: u32 = 134;
const C_PT_ARAY: u32 = 135;
const C_PT_BRAY: u32 = 136;
const C_PT_STKM2: u32 = 137;
const C_PT_PSTN: u32 = 168;
const C_PT_FRME: u32 = 169;
const C_PT_DRAY: u32 = 177;

const _: () = assert!(C_PT_NONE == PT_NONE);
const _: () = assert!(C_PT_SPRK == PT_SPRK);
const _: () = assert!(C_PT_PSCN == PT_PSCN);
const _: () = assert!(C_PT_STKM == PT_STKM);
const _: () = assert!(C_PT_SWCH == PT_SWCH);
const _: () = assert!(C_PT_INWR == PT_INWR);
const _: () = assert!(C_PT_STOR == PT_STOR);
const _: () = assert!(C_PT_LIFE == PT_LIFE);
const _: () = assert!(C_PT_INST == PT_INST);
const _: () = assert!(C_PT_WIFI == PT_WIFI);
const _: () = assert!(C_PT_FILT == PT_FILT);
const _: () = assert!(C_PT_ARAY == PT_ARAY);
const _: () = assert!(C_PT_BRAY == PT_BRAY);
const _: () = assert!(C_PT_STKM2 == PT_STKM2);
const _: () = assert!(C_PT_PSTN == PT_PSTN);
const _: () = assert!(C_PT_FRME == PT_FRME);
const _: () = assert!(C_PT_DRAY == PT_DRAY);

const C_PROP_CONDUCTS: u32 = 0x0000_0020;
const C_TYPE_ENERGY: u32 = 0x0000_0010;
const _: () = assert!(C_PROP_CONDUCTS == PROP_CONDUCTS);
const _: () = assert!(C_TYPE_ENERGY == TYPE_ENERGY);

const C_ELEM_SLOTS: u32 = PT_NUM as u32;
const C_WORD_PROPERTIES: u32 = 1;
const C_GPU_ALLOC_NONE: u32 = 0xFFFF_FFFF;
const _: () = assert!(C_ELEM_SLOTS as usize == PT_NUM);
const _: () = assert!(C_WORD_PROPERTIES == super::shared::WORD_PROPERTIES);
const _: () = assert!(C_GPU_ALLOC_NONE == super::shared::GPU_ALLOC_NONE);

/// Seed for a mutable `i32` kernel local that is assigned a runtime value later.
///
/// CubeCL expands `let mut n = 0i32;` into a plain integer, so a later
/// `n = <runtime expression>` is rejected with
/// `expected i32, found NativeExpand<i32>`. Routing the initial value through a
/// call keeps the local a device variable.
#[cube]
#[allow(clippy::erasing_op)]
pub fn seed_i32(anchor: i32, value: i32) -> i32 {
    anchor * 0i32 + value
}

/// `u32` variant of [`seed_i32`].
#[cube]
#[allow(clippy::erasing_op)]
pub fn seed_u32(anchor: u32, value: u32) -> u32 {
    anchor * 0u32 + value
}

pub(super) fn launch(bufs: &SimBuffers, dims: &PassDims) {
    if dims.pixels == 0 || dims.capacity == 0 || dims.elem_len == 0 {
        return;
    }
    if bufs.pmap.len() < dims.pixels
        || bufs.photons.len() < dims.pixels
        || bufs.part_ty.len() < dims.capacity
        || bufs.part_life.len() < dims.capacity
        || bufs.part_ctype.len() < dims.capacity
        || bufs.part_temp.len() < dims.capacity
        || bufs.part_tmp.len() < dims.capacity
        || bufs.part_tmp2.len() < dims.capacity
        || bufs.part_tmp3.len() < dims.capacity
        || bufs.part_tmp4.len() < dims.capacity
        || bufs.part_x.len() < dims.capacity
        || bufs.part_y.len() < dims.capacity
        || bufs.part_vx.len() < dims.capacity
        || bufs.part_vy.len() < dims.capacity
        || bufs.part_flags.len() < dims.capacity
        || bufs.part_dcolour.len() < dims.capacity
    {
        return;
    }

    let (cubes, dim) = launch_grid(dims.pixels);
    let grid = CubeCount::Static(cubes, 1, 1);
    let cube_dim = CubeDim::new_1d(dim);
    let free_len = SimBuffers::GPU_SPAWN_COUNT + 2;

    unsafe {
        aray_kernel::launch_unchecked(
            &bufs.client,
            grid.clone(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_free_slots.clone(), free_len),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
        );

        dray_kernel::launch_unchecked(
            &bufs.client,
            grid.clone(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_free_slots.clone(), free_len),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
        );

        pstn_kernel::launch_unchecked(
            &bufs.client,
            grid,
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_free_slots.clone(), free_len),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pstn_temperature_force() {
        // At 0 C (273.15 K): (273.15 - 268.15) / 10 = 0.5 -> floor = 0
        assert_eq!(pstn_temp_force(273.15), 0);
        // At 10 C (283.15 K): (283.15 - 268.15) / 10 = 1.5 -> floor = 1
        assert_eq!(pstn_temp_force(283.15), 1);
        // At 20 C (293.15 K): (293.15 - 268.15) / 10 = 2.5 -> floor = 2
        assert_eq!(pstn_temp_force(293.15), 2);
        // At 100 C (373.15 K): (373.15 - 268.15) / 10 = 10.5 -> floor = 10
        assert_eq!(pstn_temp_force(373.15), 10);
        // At -5 C (268.15 K): (268.15 - 268.15) / 10 = 0.0 -> floor = 0
        assert_eq!(pstn_temp_force(268.15), 0);
        // Just below -5 C: (267.15 - 268.15) / 10 = -0.1 -> floor = -1
        assert_eq!(pstn_temp_force(267.15), -1);
        // At -15 C (258.15 K): (258.15 - 268.15) / 10 = -1.0 -> floor = -1
        assert_eq!(pstn_temp_force(258.15), -1);
        // At -16 C (257.15 K): (257.15 - 268.15) / 10 = -1.1 -> floor = -2
        assert_eq!(pstn_temp_force(257.15), -2);
    }

    #[test]
    fn test_filt_wavelength_interaction() {
        let filt_val = 0x0000_1234u32;
        let orig_val = 0x0000_5678u32;

        // Mode 0: Assign Colour
        assert_eq!(interact_wavelengths(0, filt_val, 300.0, orig_val), filt_val);

        // Mode 1: Filter Colour (AND)
        assert_eq!(
            interact_wavelengths(1, 0x0000_0FF0, 300.0, 0x0000_00FF),
            0x0000_00F0
        );

        // Mode 2: Add Colour (OR)
        assert_eq!(
            interact_wavelengths(2, 0x0000_0F00, 300.0, 0x0000_00F0),
            0x0000_0FF0
        );

        // Mode 3: Filter Colour other way (AND NOT)
        assert_eq!(
            interact_wavelengths(3, 0x0000_00F0, 300.0, 0x0000_0FF0),
            0x0000_0F00
        );

        // Mode 4: Red shift ((313 - 273) * 0.025 = 1 shift left)
        assert_eq!(interact_wavelengths(4, 0, 313.0, 0x0000_0001), 0x0000_0002);

        // Mode 5: Blue shift (1 shift right)
        assert_eq!(interact_wavelengths(5, 0, 313.0, 0x0000_0002), 0x0000_0001);

        // Mode 6: No change
        assert_eq!(interact_wavelengths(6, filt_val, 300.0, orig_val), orig_val);

        // Mode 7: XOR colours
        assert_eq!(
            interact_wavelengths(7, 0x0000_0FF0, 300.0, 0x0000_00FF),
            0x0000_0F0F
        );

        // Mode 8: Invert colours (masked to 30 bits)
        assert_eq!(interact_wavelengths(8, 0, 300.0, 0), 0x3FFF_FFFF);
    }

    #[test]
    fn test_pstn_darkened_dcolour() {
        // 0 colour stays 0
        assert_eq!(pstn_darken_dcolour(0), 0);

        // Max white colour: 0xFFFFFFFF
        // 0xFF - 0x3C = 0xC3 (255 - 60 = 195)
        assert_eq!(pstn_darken_dcolour(0xFFFF_FFFF), 0xFFC3_C3C3);

        // Values below 0x3C clamp to 0
        assert_eq!(pstn_darken_dcolour(0xFF20_2020), 0xFF00_0000);

        // Mixed channels
        // R: 0x40 - 0x3C = 0x04
        // G: 0x3C - 0x3C = 0x00
        // B: 0x3B - 0x3C (< 0) = 0x00
        assert_eq!(pstn_darken_dcolour(0x8040_3C3B), 0x8004_0000);
    }
}
