//! Portal (`PRTI` / `PRTO`), powered portal (`PPTI` / `PPTO`) and wireless
//! (`WIFI`) device-resident SoA passes.
//!
//! Portals and wireless transmitters transfer particles and electrical sparks
//! across the simulation grid via temperature-indexed channels.
//!
//! # Submodules
//!
//! * [`channels`]: Portal IN/OUT queues, ring offset calculations, and orbital animation.
//! * [`wifi`]: Wireless transmitter, receiver, and channel advance kernels.
//!
//! # TPT References
//!
//! * `WIFI.cpp:50-84`:
//!   - Channel calculation: `(temp - 73.15f) / 100 + 1`, clamped to `0..CHANNELS-1`.
//!   - Spark reception: `wireless[ch][0]` sparks adjacent `NSCN`, `PSCN`, and `INWR`
//!     conductors whose `life == 0`.
//!   - Spark transmission: adjacent sparks (`life >= 3`, `ctype != NSCN`) set `wireless[ch][1] = 1`.
//! * `PRTI.cpp:62-150` / `PRTO.cpp:60-179`:
//!   - Channel calculation: identical to `WIFI`.
//!   - `portal_rx` / `portal_ry`: 8 neighbor offsets in clockwise ring order.
//!   - Global FIFO `portalp[CHANNELS][8][80]`: mapped to `gpu_portal_ty` and `gpu_portal_data`.
//!   - Swirl animation: 4 orbital particles encoded in `life` (distances) and `ctype` (angles)
//!     via `orbitalparts.h`.
//! * `PPTI.cpp:74-176` / `PPTO.cpp:81-205`: the powered pair drives the same
//!   `portalp` FIFO — there is one queue, not two — behind a `tmp2` power gate
//!   (the 5x5 `tmp2 != 10` handshake, `== 10` to kick the neighbours). Their
//!   kernels live in `reaction_kernel::modext::powered`, which launches them
//!   before the creation drain so a release's `create_part(-1, ...)`
//!   materialises in the same tick; this module owns the shared ring bodies
//!   (`channels::portal_capture_ring` / `channels::spark_ring`) and the two
//!   animation steps (`channels::portal_animate_in` / `portal_animate_out`).

mod channels;
mod wifi;

pub use channels::*;
pub use wifi::*;

use cubecl::prelude::*;
use tpt_core::constants::*;

use crate::buffers::SimBuffers;

use super::PassDims;
use super::shared::launch_grid;

const PT_NONE_U32: u32 = PT_NONE;
const PT_PRTI_U32: u32 = PT_PRTI;
const PT_PRTO_U32: u32 = PT_PRTO;
const PT_WIFI_U32: u32 = PT_WIFI;
const PT_SPRK_U32: u32 = PT_SPRK;
const PT_STOR_U32: u32 = PT_STOR;
const PT_SOAP_U32: u32 = PT_SOAP;
const PT_NSCN_U32: u32 = PT_NSCN;
const PT_PSCN_U32: u32 = PT_PSCN;
const PT_INWR_U32: u32 = PT_INWR;
const PT_INST_U32: u32 = PT_INST;
const PT_STKM_U32: u32 = PT_STKM;
const PT_STKM2_U32: u32 = PT_STKM2;
const PT_FIGH_U32: u32 = PT_FIGH;
const PT_NUM_U32: u32 = PT_NUM as u32;

const FLAG_PORTAL_HELD_U32: u32 = FLAG_PORTAL_HELD;
const MOVABLE_MASK: u32 = TYPE_PART | TYPE_LIQUID | TYPE_GAS | TYPE_ENERGY;
const PROP_CONDUCTS_U32: u32 = PROP_CONDUCTS;

const PORTAL_CHANNELS_U32: u32 = PORTAL_CHANNELS as u32;
const PORTAL_DIRS_U32: u32 = PORTAL_DIRS as u32;
const PORTAL_SLOTS_U32: u32 = PORTAL_SLOTS as u32;
const PORTAL_CAPACITY_U32: u32 = PORTAL_CAPACITY as u32;
const PORTAL_WORDS_U32: u32 = PORTAL_WORDS as u32;

const _: () = assert!(PT_NONE_U32 == PT_NONE);
const _: () = assert!(PT_PRTI_U32 == PT_PRTI);
const _: () = assert!(PT_PRTO_U32 == PT_PRTO);
const _: () = assert!(PT_WIFI_U32 == PT_WIFI);
const _: () = assert!(PT_SPRK_U32 == PT_SPRK);
const _: () = assert!(PT_STOR_U32 == PT_STOR);
const _: () = assert!(PT_SOAP_U32 == PT_SOAP);
const _: () = assert!(PT_NSCN_U32 == PT_NSCN);
const _: () = assert!(PT_PSCN_U32 == PT_PSCN);
const _: () = assert!(PT_INWR_U32 == PT_INWR);
const _: () = assert!(PT_INST_U32 == PT_INST);
const _: () = assert!(PT_STKM_U32 == PT_STKM);
const _: () = assert!(PT_STKM2_U32 == PT_STKM2);
const _: () = assert!(PT_FIGH_U32 == PT_FIGH);
const _: () = assert!(PT_NUM_U32 == PT_NUM as u32);
const _: () = assert!(FLAG_PORTAL_HELD_U32 == FLAG_PORTAL_HELD);
const _: () = assert!(PROP_CONDUCTS_U32 == PROP_CONDUCTS);
const _: () = assert!(PORTAL_CHANNELS_U32 as usize == PORTAL_CHANNELS);
const _: () = assert!(PORTAL_DIRS_U32 as usize == PORTAL_DIRS);
const _: () = assert!(PORTAL_SLOTS_U32 as usize == PORTAL_SLOTS);
const _: () = assert!(PORTAL_CAPACITY_U32 as usize == PORTAL_CAPACITY);
const _: () = assert!(PORTAL_WORDS_U32 as usize == PORTAL_WORDS);

/// Temperature-dependent channel index (`WIFI.cpp:52-54`, `PRTI.cpp:68-72`).
///
/// Mirrors `int((temp - 73.15f) / 100.0f + 1)`, clamped to `0..102` (`PORTAL_CHANNELS - 1`).
#[cube]
pub fn temp_channel(temp: f32) -> u32 {
    let raw = ((temp - 73.15f32) / 100.0f32 + 1.0f32) as i32;
    let mut ch = 0u32;
    if raw > 102i32 {
        ch = 102u32;
    } else if raw > 0i32 {
        ch = raw as u32;
    }
    ch
}

/// Launch wireless and portal kernels for one reaction step.
pub(super) fn launch(bufs: &SimBuffers, dims: &PassDims) {
    let (pixel_cubes, pixel_dim) = launch_grid(dims.pixels);

    let (wifi_cubes, wifi_dim) = launch_grid(PORTAL_CHANNELS);
    unsafe {
        wifi_step_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(wifi_cubes, 1, 1),
            CubeDim::new_1d(wifi_dim),
            BufferArg::from_raw_parts(bufs.gpu_wireless.clone(), PORTAL_CHANNELS * 2),
        );
    }

    unsafe {
        wifi_receive_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_wireless.clone(), PORTAL_CHANNELS * 2),
            dims.xres,
            dims.yres,
        );
    }

    unsafe {
        wifi_transmit_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_wireless.clone(), PORTAL_CHANNELS * 2),
            dims.xres,
            dims.yres,
        );
    }

    unsafe {
        prti_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_portal_ty.clone(), PORTAL_CAPACITY),
            BufferArg::from_raw_parts(bufs.gpu_portal_data.clone(), PORTAL_CAPACITY * PORTAL_WORDS),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.capacity as u32,
            dims.frame_seed,
        );
    }

    unsafe {
        prto_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_portal_ty.clone(), PORTAL_CAPACITY),
            BufferArg::from_raw_parts(bufs.gpu_portal_data.clone(), PORTAL_CAPACITY * PORTAL_WORDS),
            BufferArg::from_raw_parts(bufs.gpu_free_slots.clone(), SimBuffers::GPU_SPAWN_COUNT + 2),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.frame_seed,
            dims.capacity as u32,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temp_channel_mapping() {
        assert_eq!(temp_channel(0.0f32), 0);
        assert_eq!(temp_channel(73.15f32), 1);
        assert_eq!(temp_channel(173.15f32), 2);
        assert_eq!(temp_channel(273.15f32), 3);
        assert_eq!(temp_channel(573.15f32), 6);
        assert_eq!(temp_channel(10173.15f32), 102);
        // Out of bounds / clamping checks
        assert_eq!(temp_channel(12000.0f32), 102);
        assert_eq!(temp_channel(-200.0f32), 0);
        assert_eq!(temp_channel(f32::NAN), 0);
    }

    #[test]
    fn test_portal_directions() {
        assert_eq!(portal_offset(0), (-1, -1));
        assert_eq!(portal_offset(1), (0, -1));
        assert_eq!(portal_offset(2), (1, -1));
        assert_eq!(portal_offset(3), (1, 0));
        assert_eq!(portal_offset(4), (1, 1));
        assert_eq!(portal_offset(5), (0, 1));
        assert_eq!(portal_offset(6), (-1, 1));
        assert_eq!(portal_offset(7), (-1, 0));
    }

    #[test]
    fn test_portal_header_packing() {
        // Slot 0, element PT_SPRK
        let packed_0 = portal_pack_ty(1, PT_SPRK);
        assert_eq!(portal_unpack_slot(packed_0), 0);
        assert_eq!(portal_unpack_ty(packed_0), PT_SPRK);

        // Slot 4095, element PT_PRTI
        let packed_max = portal_pack_ty(4096, PT_PRTI);
        assert_eq!(portal_unpack_slot(packed_max), 4095);
        assert_eq!(portal_unpack_ty(packed_max), PT_PRTI);

        // Donated slot 0 (none)
        let packed_none = portal_pack_ty(0, PT_STOR);
        assert_eq!(portal_unpack_slot(packed_none), 0xFFFF_FFFF);
        assert_eq!(portal_unpack_ty(packed_none), PT_STOR);
    }

    #[test]
    fn test_portal_payload_words() {
        let life = -42i32;
        let ctype = PT_PSCN;
        let temp = 1357.25f32;
        let tmp = 88i32;
        let tmp2 = -15i32;
        let vx = -3.5f32;
        let vy = 2.125f32;
        let dcolour = 0xAABBCCDDu32;

        let mut data = [0u32; PORTAL_WORDS];
        data[0] = life as u32;
        data[1] = ctype;
        data[2] = temp.to_bits();
        data[3] = tmp as u32;
        data[4] = tmp2 as u32;
        data[5] = vx.to_bits();
        data[6] = vy.to_bits();
        data[7] = dcolour;

        assert_eq!(data[0] as i32, life);
        assert_eq!(data[1], ctype);
        assert_eq!(f32::from_bits(data[2]), temp);
        assert_eq!(data[3] as i32, tmp);
        assert_eq!(data[4] as i32, tmp2);
        assert_eq!(f32::from_bits(data[5]), vx);
        assert_eq!(f32::from_bits(data[6]), vy);
        assert_eq!(data[7], dcolour);
    }

    #[test]
    fn test_orbital_get_set() {
        let (b1, b2) = orbital_set(128, 45, 0);
        assert_eq!(orbital_get(b1, b2, 0), (128, 45));

        let (b1_2, b2_2) = orbital_set(250, 200, 3);
        assert_eq!(orbital_get(b1_2, b2_2, 3), (250, 200));
    }

    #[test]
    fn test_orbital_stepping() {
        // Initial state: distance 200, angle 10 in slot 0
        let (life, ctype) = orbital_set(200, 10, 0);
        let (next_life, next_ctype) = orbital_step_prti(life as i32, ctype, 12345);
        let (d, l) = orbital_get(next_life as u32, next_ctype, 0);
        assert_eq!(d, 200 - 12);
        assert_eq!(l, 10 + 2);

        // PRTO stepping: distance 100, angle 20 in slot 0
        let (life_o, ctype_o) = orbital_set(100, 20, 0);
        let (next_life_o, next_ctype_o) = orbital_step_prto(life_o as i32, ctype_o, 54321);
        let (do_val, lo_val) = orbital_get(next_life_o as u32, next_ctype_o, 0);
        assert_eq!(do_val, 100 + 16);
        assert_eq!(lo_val, 20 + 1);
    }
}
