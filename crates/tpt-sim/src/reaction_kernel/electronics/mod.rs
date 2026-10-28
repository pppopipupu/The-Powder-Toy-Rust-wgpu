//! The electrical layer: sparks, conductors, powered devices and sensors.
//!
//! This module ports TPT's electricity model across submodules:
//!
//! * [`sparks`]: [`sparks::spark_stage_kernel`] snapshots the previous particle state into the
//!   `intent_target` scratch buffer, [`sparks::spark_conduction_kernel`] is the receiver-centric
//!   conduction loop, and [`sparks::flood_inst_kernel`] carries a spark along an `INST` wire
//!   (`FloodINST`).
//! * [`switches`]: [`switches::switch_device_kernel`] runs the `SWCH` / `HSWC` / `PVOD` / `PCLN` /
//!   `PBCN` / `LCRY` state machines.
//! * [`storage`]: [`storage::storage_kernel`] runs `STOR.cpp` (capture + release).
//! * [`sensors`]: [`sensors::sensor_kernel`] runs `PSNS.cpp` (`FILT` broadcast),
//!   [`sensors::light_sensor_kernel`] runs `LSNS.cpp` life scan, and
//!   [`sensors::source_ignition_kernel`] ignites conductors from sources.
//! * [`arcs`]: [`arcs::arc_and_delay_kernel`] runs the `TESC` / `ETRD` spark branches and `DLAY.cpp`.
//! * [`detectors`]: [`detectors::detector_kernel`] runs `DTEC.cpp` and `LDTC.cpp`.
//!
//! | kernel | module | work | storage buffers |
//! |--------|--------|------|-----------------|
//! | [`sparks::spark_stage_kernel`] | `sparks` | snapshot previous particle state into `intent_target` scratch | 5 |
//! | [`sparks::spark_conduction_kernel`] | `sparks` | receiver-centric spark conduction loop | 9 |
//! | [`sparks::flood_inst_kernel`] | `sparks` | `INST` wire cascade, one orthogonal hop per round | 4 |
//! | [`switches::switch_device_kernel`] | `switches` | state machines for `SWCH` / `HSWC` / `PVOD` / `PCLN` / `PBCN` / `LCRY` | 15 |
//! | [`storage::storage_kernel`] | `storage` | capture and release for `STOR` | 12 |
//! | [`sensors::sensor_kernel`] | `sensors` | pressure broadcast to `FILT` chains for `PSNS` | 5 |
//! | [`sensors::light_sensor_kernel`] | `sensors` | range scan for `LSNS` | 7 |
//! | [`sensors::lsns_filter_kernel`] | `sensors` | `FILT` serialisation and deserialisation for `LSNS` modes 1 and 3 | 7 |
//! | [`sensors::temp_sensor_kernel`] | `sensors` | `TSNS` spark latch, temperature scan and `FILT` serialisation | 9 |
//! | [`sensors::velocity_sensor_kernel`] | `sensors` | `VSNS` spark latch, velocity scan, `FILT` serialisation/deserialisation | 11 |
//! | [`sensors::source_ignition_kernel`] | `sensors` | ignite conductors from `BTRY` / `PSNS` / `LSNS`, thermistor cooling, `NBLE` | 8 |
//! | [`wire::wire_stage_kernel`] | `wire` | `BeforeSim`'s `tmp = ctype` for every WireWorld pixel | 3 |
//! | [`wire::wire_update_kernel`] | `wire` | one `WIRE.cpp` WireWorld generation | 6 |
//! | [`pump::pump_update_kernel`] | `pump` | `PUMP.cpp` pressure push, `FILT` set and chain | 7 |
//! | [`thunder::thunder_update_kernel`] | `thunder` | `THDR.cpp` conductor sparking, pressure burst and self-kill | 6 |
//! | [`arcs::arc_and_delay_kernel`] | `arcs` | `TESC` / `ETRD` arcing and `DLAY` delay wire | 10 |
//! | [`bolt::etrd_bolt_kernel`] | `bolt` | rasterise arcing `PLSM` bolts for sparked `ETRD` | 12 |
//! | [`detectors::detector_kernel`] | `detectors` | particle detection and conductor ignition for `DTEC` and `LDTC` | 8 |
//!
//! # Staged reads
//!
//! Every read of a *neighbour's* electrical state in the spark pass comes from
//! the snapshot, never from the live SoA, so the result does not depend on the
//! order in which the device schedules the threads. The cheaper device passes
//! ([`switches::switch_device_kernel`], [`arcs::arc_and_delay_kernel`]) read live values
//! instead: their cross-pixel writes are idempotent chain activations
//! (`life = 10`, `life = 9`, `tmp = 2`), exactly like TPT's sequential loop.
//! [`sparks::flood_inst_kernel`] also reads live types, but it only ever moves a
//! wire pixel from `INST` to `SPRK`, so a round that loses a race with a sibling
//! merely repeats the hop on the next round.
//!
//! # Temperature
//!
//! [`sparks::spark_conduction_kernel`] cannot afford a fourth temperature buffer, so
//! [`sparks::spark_stage_kernel`] folds the only three temperature thresholds the
//! conduction rules use (`NTCT` / `PTCT` at 373 K, `QRTZ` at 173.15 K) into
//! three flag bits of the snapshot word.
//!
//! # What this pass leaves to other passes
//!
//! `PSTN` (multi pixel stack pushes plus the `FRME` machinery and `tmp3`
//! flags) lives in `rays/pistons.rs`, `DMG` (a 25 pixel
//! radial impulse field that rewrites other particles' velocities) in
//! `force/projectiles.rs`, which inverts the relation and has each target
//! resolve the surrounding devices. `INST`'s `FloodINST` runs as a bounded
//! device cascade: [`sparks::flood_inst_kernel`] moves the spark one orthogonal
//! hop per round for [`sparks::INST_FLOOD_ROUNDS`] rounds, so a run longer than
//! the cascade finishes lighting from its frontier on the following frames
//! instead of stalling the pass on an unbounded span.
//!
//! Smaller omissions, all caused by the pass not spawning particles, not
//! writing velocities and not touching `pmap`:
//!
//! * Spark misc branches: `PPIP` flooding genuinely requires the pipe pass's
//!   multi-round cascade state, which is handled directly in
//!   `reaction_kernel/pipe/pump.rs`.
//! * Created particles (`PCLN` / `PBCN` clones, `TESC` lightning, `STOR`
//!   releases) are placed by the host materialiser in the first free
//!   neighbour, not at TPT's randomised or second-ring offsets.
//!
//! `SWCH` also relies on `life.rs`, which settles a spent spark back into the
//! conductor; this module only has to rescue the `SWCH` case early, because
//! `SWCH` is not `PROP_CONDUCTS` and would otherwise be deleted.

mod arcs;
pub(super) mod bolt;
mod detectors;
mod pump;
mod sensors;
mod sparks;
mod storage;
mod switches;
mod thunder;
mod wire;

pub(super) use arcs::arc_and_delay_kernel;
pub(super) use detectors::detector_kernel;
pub(super) use sensors::{
    light_sensor_kernel, lsns_filter_kernel, sensor_kernel, source_ignition_kernel,
    temp_sensor_kernel, velocity_sensor_kernel,
};
pub(super) use sparks::{spark_conduction_kernel, spark_stage_kernel};
pub(super) use storage::storage_kernel;
pub(super) use switches::switch_device_kernel;

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::PassDims;
use super::shared::launch_grid;
use crate::buffers::SimBuffers;

/// 6.2b `SPRK.cpp:377-381` → `Simulation.cpp:586-700`: after the conduction
/// pass has charged the `INST` pixel a `PSCN` spark conducts into, carry the
/// spark along the connected `INST` run.
///
/// One `flood_inst_kernel` dispatch moves the frontier a single orthogonal hop,
/// so [`sparks::INST_FLOOD_ROUNDS`] rounds carry it along the wire. The
/// cascade is monotone and slot-owned (a pixel that loses a round race simply
/// sparks on the next one), never reads back to the host and never allocates;
/// a run longer than the cascade keeps lighting from its frontier sparks on the
/// following frames.
pub(in crate::reaction_kernel) fn launch_flood_inst_pass(bufs: &SimBuffers, dims: &PassDims) {
    if dims.pixels == 0 {
        return;
    }
    let (cubes, dim) = launch_grid(dims.pixels);
    for _ in 0..sparks::INST_FLOOD_ROUNDS {
        unsafe {
            sparks::flood_inst_kernel::launch_unchecked(
                &bufs.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(dim),
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
                BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
                dims.xres,
                dims.yres,
            );
        }
    }
}

/// The four per-frame element updates this layer owns.
///
/// * `WIRE.cpp:50-94` (with `Simulation.cpp:3862-3876` for the `tmp = ctype`
///   staging that starts a generation) — [`wire::wire_stage_kernel`] then
///   [`wire::wire_update_kernel`].
/// * `PUMP.cpp:53-111` — [`pump::pump_update_kernel`].
/// * `THDR.cpp:51-93` — [`thunder::thunder_update_kernel`].
/// * `BTRY.cpp:48-76` is the receiver branch of
///   [`sensors::source_ignition_kernel`]: a conductor tests the radius-2
///   diamond around itself for a battery, which is the same rule without a
///   cross-pixel write.
///
/// The dispatch runs after the spark sources, so a conductor a battery or a
/// lightning sparks is not conducted again in the same frame; the batched
/// reaction pass is documented as one frame behind TPT's per-particle order in
/// `SimulationEngine::step`.
pub(in crate::reaction_kernel) fn launch_element_update_pass(bufs: &SimBuffers, dims: &PassDims) {
    if dims.pixels == 0 || dims.xcells == 0 {
        return;
    }
    // `BeforeSim`'s WireWorld staging is slot parallel; the generation below
    // is pixel parallel.
    let (slot_cubes, slot_dim) = launch_grid(dims.used_slots);
    unsafe {
        wire::wire_stage_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            dims.used_slots as u32,
        );
    }

    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        wire::wire_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            dims.xres,
            dims.yres,
        );
    }

    let ycells = (dims.cells / dims.xcells.max(1) as usize) as u32;
    unsafe {
        pump::pump_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
            ycells,
        );
    }

    unsafe {
        thunder::thunder_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
            if bufs.config.heat_enable { 0u32 } else { 1u32 },
        );
    }
}

/// `SPRK.cpp`: spark life, i.e. the spark conducts during the three frames
/// after it is created (`life < 4`).
pub(super) const SPARK_LIFE: i32 = 4;
/// `SPRK.cpp`: spark life of a spark sitting on `SWCH` once it turns back into
/// the switch, so the switch stays on for 14 frames after its pulse.
pub(super) const SWCH_RELEASE_LIFE: i32 = 14;
/// `SWCH.cpp`: life of a switch that has been told to turn off.
pub(super) const SWCH_OFF_LIFE: i32 = 9;
/// `SWCH.cpp` / `HSWC.cpp` / `PVOD.cpp` / `PCLN.cpp` / `PBCN.cpp`: a powered
/// device at `life >= 10` is on.
pub(super) const POWERED_ON_LIFE: i32 = 10;
/// `LSNS.cpp` / `DTEC.cpp`: the scan radius stored in `tmp2` is capped at 25.
pub(super) const SENSOR_MAX_RANGE: u32 = 25;
/// Width of the sensor scan window (`-25..=25`) for both sensor passes.
pub(super) const SENSOR_SCAN: u32 = 2 * SENSOR_MAX_RANGE + 1;

/// `SPRK.cpp`: the toggled elements (`PUMP`, `GPMP`, `HSWC`, `PBCN`) turn on
/// from a `PSCN` spark and off from an `NSCN` spark.
#[cube]
pub(super) fn powered_toggle_life(receiver_life: i32, sender: u32) -> i32 {
    let mut life = receiver_life;
    if sender == PT_PSCN {
        life = POWERED_ON_LIFE;
    } else if sender == PT_NSCN && receiver_life >= POWERED_ON_LIFE {
        life = SWCH_OFF_LIFE;
    }
    life
}

/// `SPRK.cpp` `case PT_TESC`: lightning life, `rng.between(0, 2 + power/15)`
/// plus `power/7`, capped at 60.
///
/// The bolt is spawned by `spawn::spawn_special_kernel`, whose `#[cube]`
/// `ligh_life` cannot be called from a host test; this mirror pins the formula.
#[cfg(test)]
pub(super) fn tesc_ligh_life(power: i32, roll: u32) -> i32 {
    let power = power.clamp(0, 300);
    let span = 2 + power / 15;
    let mut life = (roll % (span as u32 + 1)) as i32 + power / 7;
    if life > 60 {
        life = 60;
    }
    life
}

/// `SPRK.cpp` `case PT_TESC`: the arc roll denominator `power*power/20 + 6`.
///
/// The arc pass computes this inline (a kernel cannot call a host function), so
/// this mirror exists to pin the formula in tests.
#[cfg(test)]
pub(super) fn tesc_arc_denominator(power: i32) -> u32 {
    let power = power.clamp(0, 300);
    (power * power / 20 + 6) as u32
}

/// Host mirror of [`ihypot`] for offsets up to radius 16.
#[cfg(test)]
pub(super) fn ihypot_host(dx: i32, dy: i32) -> i32 {
    let d2 = dx * dx + dy * dy;
    let mut r = 0i32;
    for i in 1i32..18i32 {
        if i * i <= d2 {
            r = i;
        }
    }
    r
}

/// Pure helper for `SWCH.cpp:91-98` red BRAY toggle logic.
#[cfg(test)]
pub(super) fn swch_red_bray_toggle(
    top_left_empty: bool,
    top_right_empty: bool,
    vert_red_bray: bool,
    horiz_red_bray: bool,
    life: i32,
) -> Option<i32> {
    if top_left_empty && top_right_empty && vert_red_bray && horiz_red_bray {
        if life == POWERED_ON_LIFE {
            Some(SWCH_OFF_LIFE)
        } else if life <= 5 {
            Some(SWCH_RELEASE_LIFE)
        } else {
            None
        }
    } else {
        None
    }
}

/// Pure helper for `DTEC.cpp:98` match condition.
#[cfg(test)]
pub(super) fn dtec_detects(dtec_ctype: u32, dtec_tmp: i32, nty: u32, nctype: u32) -> bool {
    if dtec_ctype == PT_NONE || nty != dtec_ctype {
        return false;
    }
    if dtec_ctype == PT_LIFE && dtec_tmp != 0 && nctype != dtec_tmp as u32 {
        return false;
    }
    true
}

/// Pure helper for `DTEC.cpp:100` wavelength emission check.
#[cfg(test)]
pub(super) fn dtec_emits_wavelength(ty: u32, tmp: i32) -> bool {
    ty == PT_PHOT
        || (ty == PT_BRAY && tmp != 2)
        || ty == PT_BIZR
        || ty == PT_BIZRG
        || ty == PT_BIZRS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reaction_kernel::electronics::sparks::*;

    #[test]
    fn snapshot_word_layout_is_stable() {
        assert_eq!(STAGE_TY, 0x0000_00FF);
        assert_eq!(STAGE_LIFE, 0x0000_FF00);
        assert_eq!(STAGE_CTYPE, 0x00FF_0000);
        assert_eq!(STAGE_COLD_QUARTZ, 0x0100_0000);
        assert_eq!(STAGE_COLD, 0x0200_0000);
        assert_eq!(STAGE_HOT, 0x0400_0000);
        // Every element id fits in the low byte of the snapshot word.
        assert!(PT_NUM <= STAGE_TY as usize, "PT_NUM must fit in one byte");
    }

    /// Host mirror of the kernel's `heat_bits`, pinning the thresholds.
    fn heat_bits_host(t: f32) -> u32 {
        let mut bits = 0u32;
        if t < 173.15 {
            bits |= STAGE_COLD_QUARTZ;
        }
        if t < 373.0 {
            bits |= STAGE_COLD;
        }
        if t > 373.0 {
            bits |= STAGE_HOT;
        }
        bits
    }

    #[test]
    fn heat_classes_follow_the_cpp_thresholds() {
        // `QRTZ` conducts below 173.15 K, the thermistors switch at 373 K.
        assert_eq!(heat_bits_host(100.0), STAGE_COLD_QUARTZ | STAGE_COLD);
        assert_eq!(heat_bits_host(300.0), STAGE_COLD);
        assert_eq!(heat_bits_host(373.0), 0);
        assert_eq!(heat_bits_host(400.0), STAGE_HOT);
        assert_eq!(heat_bits_host(173.15), STAGE_COLD);
        assert_eq!(heat_bits_host(173.14), STAGE_COLD_QUARTZ | STAGE_COLD);
    }

    fn stage_encode(ty: u32, life: i32, ctype: u32, temp: f32) -> u32 {
        let l = life.clamp(0, 255) as u32;
        (ty & STAGE_TY) | (l << 8) | ((ctype & 0xFF) << 16) | heat_bits_host(temp)
    }

    fn stage_decode(word: u32) -> (u32, i32, u32) {
        (
            word & STAGE_TY,
            ((word & STAGE_LIFE) >> 8) as i32,
            (word & STAGE_CTYPE) >> 16,
        )
    }

    #[test]
    fn snapshot_round_trip() {
        let word = stage_encode(PT_SPRK, 3, PT_METL, 300.0);
        assert_eq!(stage_decode(word), (PT_SPRK, 3, PT_METL));
        // Out-of-range life saturates instead of bleeding into `ctype`.
        let hot = stage_encode(PT_DLAY, 4096, PT_PSCN, 400.0);
        assert_eq!(stage_decode(hot), (PT_DLAY, 255, PT_PSCN));
        let negative = stage_encode(PT_SWCH, -5, PT_NONE, 100.0);
        assert_eq!(stage_decode(negative), (PT_SWCH, 0, PT_NONE));
    }

    #[test]
    fn storer_field_layout_matches_tpt() {
        // TPT STOR.cpp:55-105:
        // tmp = type, temp = temp, tmp2 = life, tmp3 = tmp, tmp4 = ctype
        let src_ty = PT_WATR;
        let src_temp = 300.0f32;
        let src_life = 42i32;
        let src_tmp = 7i32;
        let src_ctype = PT_SALT;

        let stor_tmp = src_ty as i32;
        let stor_temp = src_temp;
        let stor_tmp2 = src_life;
        let stor_tmp3 = src_tmp;
        let stor_tmp4 = src_ctype as i32;

        assert_eq!(stor_tmp, PT_WATR as i32);
        assert_eq!(stor_temp, 300.0);
        assert_eq!(stor_tmp2, 42);
        assert_eq!(stor_tmp3, 7);
        assert_eq!(stor_tmp4, PT_SALT as i32);
    }

    #[test]
    fn tesla_arc_matches_the_cpp_ranges() {
        // `rng.between(0, 2 + tmp/15) + tmp/7`, capped at 60.
        assert_eq!(tesc_ligh_life(5, 0), 0);
        assert_eq!(tesc_ligh_life(300, 0), 42);
        assert_eq!(tesc_ligh_life(300, 22), 60);
        // The arc roll denominator grows with the coil power; the coil power is
        // already clamped to 300 where it is read, so the helper clamps too.
        assert_eq!(tesc_arc_denominator(0), 6);
        assert_eq!(tesc_arc_denominator(100), 506);
        assert_eq!(tesc_arc_denominator(1000), 4506);
    }
    #[test]
    fn integer_hypot_matches_exact_values() {
        assert_eq!(ihypot_host(0, 0), 0);
        assert_eq!(ihypot_host(3, 4), 5);
        assert_eq!(ihypot_host(4, 3), 5);
        assert_eq!(ihypot_host(1, 1), 1);
        assert_eq!(ihypot_host(12, 0), 12);
        assert_eq!(ihypot_host(0, 12), 12);
        assert_eq!(ihypot_host(12, 12), 16);
        assert_eq!(ihypot_host(5, 12), 13);
        assert_eq!(ihypot_host(6, 8), 10);
    }

    #[test]
    fn swch_red_bray_toggle_matches_cpp() {
        assert_eq!(
            swch_red_bray_toggle(true, true, true, true, 10),
            Some(SWCH_OFF_LIFE)
        );
        assert_eq!(
            swch_red_bray_toggle(true, true, true, true, 0),
            Some(SWCH_RELEASE_LIFE)
        );
        assert_eq!(
            swch_red_bray_toggle(true, true, true, true, 5),
            Some(SWCH_RELEASE_LIFE)
        );
        assert_eq!(swch_red_bray_toggle(true, true, true, true, 7), None);
        assert_eq!(swch_red_bray_toggle(false, true, true, true, 10), None);
        assert_eq!(swch_red_bray_toggle(true, false, true, true, 10), None);
        assert_eq!(swch_red_bray_toggle(true, true, false, true, 10), None);
        assert_eq!(swch_red_bray_toggle(true, true, true, false, 10), None);
    }

    #[test]
    fn dtec_detection_and_wavelength_filtering() {
        assert!(!dtec_detects(PT_NONE, 0, PT_WATR, 0));
        assert!(dtec_detects(PT_WATR, 0, PT_WATR, 0));
        assert!(!dtec_detects(PT_WATR, 0, PT_SLTW, 0));
        assert!(dtec_detects(PT_LIFE, 0, PT_LIFE, 5));
        assert!(dtec_detects(PT_LIFE, 3, PT_LIFE, 3));
        assert!(!dtec_detects(PT_LIFE, 3, PT_LIFE, 4));

        assert!(dtec_emits_wavelength(PT_PHOT, 0));
        assert!(dtec_emits_wavelength(PT_BRAY, 0));
        assert!(dtec_emits_wavelength(PT_BRAY, 1));
        assert!(!dtec_emits_wavelength(PT_BRAY, 2));
        assert!(dtec_emits_wavelength(PT_BIZR, 0));
        assert!(dtec_emits_wavelength(PT_BIZRG, 0));
        assert!(dtec_emits_wavelength(PT_BIZRS, 0));
        assert!(!dtec_emits_wavelength(PT_METL, 0));
    }
}
