//! GPU thermal conduction, ambient heat exchange and phase transitions.
//!
//! This module ports `Simulation::TransitionPhase` (`Simulation.cpp:2430-2866`)
//! and the `Element::legacyUpdate` fallback (`Element.cpp:111-220`).
//!
//! # Passes
//!
//! * [`thermal_conduction_kernel`] — the heat-transfer half of
//!   `TransitionPhase`: the `IsHeatInsulator` / `HeatConduct` gate, the ambient
//!   exchange with `air_hv`, the heat-capacity-weighted equilibrium `pt` over
//!   the eight `pmap` neighbours (with the FILT/BRAY/BIZR/ELEC/DEUT/HSWC pair
//!   exclusions), and the `blockairh` counter increment when the gate fails.
//!   `pt` is stored in `part_equil` (`-1` = gate failed).
//! * [`thermal_apply_kernel`] — writes `pt` back to every participant.
//!   `TransitionPhase` sets `parts[i].temp = pt` for the particle *and* all of
//!   its conducting neighbours; the parallel pass gives each pixel the `pt` of
//!   the highest slot that includes it, which is the same "last writer wins"
//!   rule the serial reference applies.
//! * [`phase_transition_kernel`] — the temperature/pressure transitions, the
//!   post-transition `ctype`/`life` bookkeeping, `LAVA`'s life feedback, the
//!   `LIFE` temperature drain, the wall-spark triggers and the pressure
//!   transitions (including the gravity-gated branches).
//! * [`refresh_pmap_types_kernel`] — republishes the element type half of the
//!   packed `pmap` word for particles whose `ty` changed in this pass.
//! * [`legacy_update_kernel`] — `Element::legacyUpdate`, launched instead of
//!   the heat passes when `heat_enable` is off.
//!
//! # Element data
//!
//! The kernels read the canonical element table that [`SimBuffers`] keeps on
//! the device (`gpu_elem_props`, one [`GpuElementProps`] row per element slot,
//! uploaded by `SimBuffers::upload_element_tables`). It is bound as a flat
//! `&[u32]` and addressed with the word offsets below; float columns are
//! recovered from their IEEE-754 bit patterns with `Reinterpret::reinterpret`.
//!
//! # Heat capacity
//!
//! `SimulationData::HeatCapacityOf` (`SimulationData.cpp:343-349`) returns the
//! element's `HeatCapacity`, plus the stored `ctype`'s capacity for
//! `PIPE`/`PPIP`. The only built-in override is `BASE = 1.5`
//! (`elements/BASE.cpp:33`); every other element keeps the `Element()` default
//! `1.0` (`Element.cpp:33`). The registry has no capacity column, so
//! [`heat_capacity`] encodes those two authoritative facts directly.
//!
//! # Units
//!
//! `part_temp`, `air_hv` and every `low_temp` / `high_temp` threshold are
//! **Kelvin**; `air_pv` uses the reference pressure scale. Live particle slots
//! are sparse, so every kernel scans `0..used_slot_bound()` and skips `PT_NONE`.

use cubecl::prelude::*;

use tpt_core::constants::*;
use tpt_core::elements::{ElementRegistry, GpuElementProps};

use crate::buffers::SimBuffers;

/// Number of `u32` words per element in the device-side element table.
pub const GPU_ELEM_STRIDE: usize = 32;

// Word offsets inside one `GpuElementProps` row (see `tpt_core::elements`).
// The const block after these definitions proves they match the struct layout.
/// Raw element `properties` bitflags (`PROP_NOAMBHEAT`, ...).
const COL_PROPERTIES: usize = 1;
/// `flammable` (`i32` bits) — the explosion ignition temperature bonus.
const COL_FLAMMABLE: usize = 12;
/// `explosive` (`i32` bits); bit 1 means "detonates under pressure".
const COL_EXPLOSIVE: usize = 13;
/// `heat_conduct` in `0..=255`; the conduction roll is compared against it.
const COL_HEAT_CONDUCT: usize = 15;
/// `default_temp` (`f32` bits) — the `FIRE` ignition temperature.
const COL_DEFAULT_TEMP: usize = 16;
/// `low_pressure` (`f32` bits).
const COL_LOW_PRESSURE: usize = 18;
/// `low_pressure_transition` (`i32` bits).
const COL_LOW_PRESSURE_TRANSITION: usize = 19;
/// `high_pressure` (`f32` bits).
const COL_HIGH_PRESSURE: usize = 20;
/// `high_pressure_transition` (`i32` bits).
const COL_HIGH_PRESSURE_TRANSITION: usize = 21;
/// `low_temp` (`f32` bits).
const COL_LOW_TEMP: usize = 22;
/// `low_temp_transition` (`i32` bits).
const COL_LOW_TEMP_TRANSITION: usize = 23;
/// `high_temp` (`f32` bits).
const COL_HIGH_TEMP: usize = 24;
/// `high_temp_transition` (`i32` bits).
const COL_HIGH_TEMP_TRANSITION: usize = 25;

// The kernels index `gpu_elem_props` as `&[u32]`, so the layout is load-bearing.
// Fail the build rather than silently reading the wrong column if the struct
// ever changes.
const _: () = {
    assert!(core::mem::size_of::<GpuElementProps>() == GPU_ELEM_STRIDE * 4);
    assert!(core::mem::offset_of!(GpuElementProps, properties) == COL_PROPERTIES * 4);
    assert!(core::mem::offset_of!(GpuElementProps, flammable) == COL_FLAMMABLE * 4);
    assert!(core::mem::offset_of!(GpuElementProps, explosive) == COL_EXPLOSIVE * 4);
    assert!(core::mem::offset_of!(GpuElementProps, heat_conduct) == COL_HEAT_CONDUCT * 4);
    assert!(core::mem::offset_of!(GpuElementProps, default_temp) == COL_DEFAULT_TEMP * 4);
    assert!(core::mem::offset_of!(GpuElementProps, low_pressure) == COL_LOW_PRESSURE * 4);
    assert!(
        core::mem::offset_of!(GpuElementProps, low_pressure_transition)
            == COL_LOW_PRESSURE_TRANSITION * 4
    );
    assert!(core::mem::offset_of!(GpuElementProps, high_pressure) == COL_HIGH_PRESSURE * 4);
    assert!(
        core::mem::offset_of!(GpuElementProps, high_pressure_transition)
            == COL_HIGH_PRESSURE_TRANSITION * 4
    );
    assert!(core::mem::offset_of!(GpuElementProps, low_temp) == COL_LOW_TEMP * 4);
    assert!(
        core::mem::offset_of!(GpuElementProps, low_temp_transition) == COL_LOW_TEMP_TRANSITION * 4
    );
    assert!(core::mem::offset_of!(GpuElementProps, high_temp) == COL_HIGH_TEMP * 4);
    assert!(
        core::mem::offset_of!(GpuElementProps, high_temp_transition)
            == COL_HIGH_TEMP_TRANSITION * 4
    );
};

/// `PT_NUM` as a `u32` so kernel-side range checks stay in one integer type.
const PT_NUM_U32: u32 = PT_NUM as u32;

/// `PT_TUNG` as a `u32`: the `BRMT` `ST` case reads tungsten's own melting
/// point so broken tungsten does not liquefy at the plain metal 1273 K.
const PT_TUNG_U32: u32 = PT_TUNG;
const _: () = assert!(PT_TUNG == 171);

/// `PT_BMTL` as a `u32`: `LAVA(THRM)` compares against broken metal's melting
/// point, not the thermite's own.
const PT_BMTL_U32: u32 = PT_BMTL;
const _: () = assert!(PT_BMTL == 46);
/// `PT_LN2` as a `u32`: liquid nitrogen boils under a pressure-shifted
/// threshold like a liquid/gas pair.
const PT_LN2_U32: u32 = PT_LN2;
const _: () = assert!(PT_LN2 == 38);
/// `PT_SLTW` as a `u32`: salt water uses the same pressure shift.
const PT_SLTW_U32: u32 = PT_SLTW;
const _: () = assert!(PT_SLTW == 19);
/// `PT_WTRV` as a `u32`: steam condenses under a pressure-shifted threshold.
const PT_WTRV_U32: u32 = PT_WTRV;
const _: () = assert!(PT_WTRV == 21);

/// Workgroup size used by every kernel in this module.
const WORKGROUP_SIZE: u32 = 256;

/// Fraction of the particle/air temperature gap exchanged per ambient tick:
/// `alpha = min(0.04, 0.4 * hc)` (`Simulation.cpp:2459-2463`).
const AMBIENT_ALPHA_MAX: f32 = 0.04;
/// The ambient exchange divides the particle side by `hc` but not the air side.
const AMBIENT_ALPHA_SCALE: f32 = 0.4;

/// Multiplier that decorrelates the per-particle hash from the frame seed.
const PARTICLE_HASH_SALT: u32 = 2_654_435_761;
/// Extra salt that keeps the transition roll independent of the conduction roll.
const TRANSITION_HASH_SALT: u32 = 0x51ED_2701;

/// Pressure (in the `air_pv` scale) above which `Explosive & 2` elements detonate.
const DETONATE_PRESSURE: f32 = 2.5;
/// Pressure added to the air cell by a detonation.
const DETONATE_PRESSURE_GAIN: f32 = 0.25;

/// Freezing point of `LAVA` (Kelvin) used by its `ST` low-temperature case.
const LAVA_FREEZE_K: f32 = 973.0;
/// Below this temperature condensing steam deposits as `RIME` instead of `DSTW`.
const STEAM_FREEZE_K: f32 = 273.0;
/// Melting point of plain water ice (Kelvin), used by the `ICEI`/`SNOW` `ST` case.
const WATER_MELT_K: f32 = 273.15;
/// `LAVA(VIBR)`/`LAVA(BVBR)` only resolidifies below water's freezing point
/// (`Simulation.cpp:2627`).
const VIBR_FREEZE_K: f32 = 273.15;
/// `LAVA(THRM)` holds its temperature while it burns down (`Simulation.cpp:2757`).
const THERMITE_TEMP: f32 = 3500.0;

/// Number of distinct random rolls taken from one hash word.
const CONDUCT_ROLL_MOD: u32 = 250;

/// `life` assigned to freshly molten `LAVA` (`240..=359`).
const LAVA_LIFE_BASE: i32 = 240;
const LAVA_LIFE_SPAN: u32 = 120;
/// `life` assigned to freshly ignited `FIRE` / `PLSM` / `CFLM` (`120..=169`).
const FIRE_LIFE_BASE: i32 = 120;
const FIRE_LIFE_SPAN: u32 = 50;
/// `life` assigned by the pressure detonation path (`180..=259`,
/// `Simulation.cpp:2778-2785`).
const BLAST_LIFE_BASE: i32 = 180;
const BLAST_LIFE_SPAN: u32 = 80;
/// `life` a sparking conductor keeps (`TransitionPhase` wall sparks).
const SPARK_LIFE: i32 = 4;
/// `emap` value `set_emap` writes (`Simulation.cpp:951`).
const EMAP_CHARGED: u32 = 16;
/// `emap` value that turns an adjacent conductor into `SPRK`
/// (`Simulation.cpp:2800`).
const EMAP_SPARKING: u32 = 12;

/// Mirrors of the element ids the transition rules name, with their ids
/// asserted so a renumbered table cannot silently change the physics.
macro_rules! pt_u32 {
    ($name:ident, $value:expr) => {
        const $name: u32 = $value;
        const _: () = assert!($name == $value);
    };
}
pt_u32!(PT_ICEI_U32, PT_ICEI);
pt_u32!(PT_SNOW_U32, PT_SNOW);
pt_u32!(PT_BRMT_U32, PT_BRMT);
pt_u32!(PT_RIME_U32, PT_RIME);
pt_u32!(PT_LAVA_U32, PT_LAVA);
pt_u32!(PT_CRMC_U32, PT_CRMC);
pt_u32!(PT_THRM_U32, PT_THRM);
pt_u32!(PT_PLUT_U32, PT_PLUT);
pt_u32!(PT_VIBR_U32, PT_VIBR);
pt_u32!(PT_BVBR_U32, PT_BVBR);
pt_u32!(PT_HEAC_U32, PT_HEAC);
pt_u32!(PT_STNE_U32, PT_STNE);
pt_u32!(PT_SAND_U32, PT_SAND);
pt_u32!(PT_BGLA_U32, PT_BGLA);
pt_u32!(PT_PQRT_U32, PT_PQRT);
pt_u32!(PT_GLAS_U32, PT_GLAS);
pt_u32!(PT_QRTZ_U32, PT_QRTZ);
pt_u32!(PT_LITH_U32, PT_LITH);
pt_u32!(PT_ACID_U32, PT_ACID);
pt_u32!(PT_FRZW_U32, PT_FRZW);
pt_u32!(PT_SEED_U32, PT_SEED);
pt_u32!(PT_FIRE_U32, PT_FIRE);
pt_u32!(PT_PLSM_U32, PT_PLSM);
pt_u32!(PT_CFLM_U32, PT_CFLM);
pt_u32!(PT_FOG_U32, PT_FOG);
pt_u32!(PT_LIFE_U32, PT_LIFE);
pt_u32!(PT_SPRK_U32, PT_SPRK);
pt_u32!(PT_GEL_U32, PT_GEL);
pt_u32!(PT_PIPE_U32, PT_PIPE);
pt_u32!(PT_PPIP_U32, PT_PPIP);
pt_u32!(PT_HSWC_U32, PT_HSWC);
pt_u32!(PT_FILT_U32, PT_FILT);
pt_u32!(PT_BRAY_U32, PT_BRAY);
pt_u32!(PT_BIZR_U32, PT_BIZR);
pt_u32!(PT_BIZRG_U32, PT_BIZRG);
pt_u32!(PT_PHOT_U32, PT_PHOT);
pt_u32!(PT_ELEC_U32, PT_ELEC);
pt_u32!(PT_DEUT_U32, PT_DEUT);
pt_u32!(PT_DSTW_U32, PT_DSTW);
pt_u32!(PT_WATR_U32, PT_WATR);
pt_u32!(PT_SALT_U32, PT_SALT);
pt_u32!(PT_GAS_U32, PT_GAS);
pt_u32!(PT_OIL_U32, PT_OIL);
pt_u32!(PT_DESL_U32, PT_DESL);
pt_u32!(PT_NONE_U32, PT_NONE);
pt_u32!(PT_BASE_U32, PT_BASE);
pt_u32!(PT_THDR_U32, PT_THDR);
pt_u32!(PT_EMBR_U32, PT_EMBR);
pt_u32!(PT_FIGH_U32, PT_FIGH);

/// `TYPE_ENERGY` / `TYPE_GAS` / `TYPE_LIQUID` / `TYPE_SOLID` mirrors.
const TYPE_ENERGY_U32: u32 = TYPE_ENERGY;
const TYPE_GAS_U32: u32 = TYPE_GAS;
const TYPE_LIQUID_U32: u32 = TYPE_LIQUID;
const _: () = assert!(TYPE_ENERGY_U32 == 0x10);
const _: () = assert!(TYPE_GAS_U32 == 0x08);
const _: () = assert!(TYPE_LIQUID_U32 == 0x02);
/// `PROP_NOAMBHEAT` / `PROP_CONDUCTS` mirrors.
const PROP_NOAMBHEAT_U32: u32 = PROP_NOAMBHEAT;
const PROP_CONDUCTS_U32: u32 = PROP_CONDUCTS;
const _: () = assert!(PROP_NOAMBHEAT_U32 == 0x0020_0000);
const _: () = assert!(PROP_CONDUCTS_U32 == 0x0000_0020);
/// `PFLAG_CAN_CONDUCT` (`elements/PIPE.h:18`).
const PFLAG_CAN_CONDUCT: i32 = 1;

/// `NT` / `ST` transition sentinels mirrored for kernel bodies.
const NT_I32: i32 = NT;
const ST_I32: i32 = ST;
const _: () = assert!(NT_I32 == -1);
const _: () = assert!(ST_I32 == 512);

/// `WL_*` ids the wall-spark trigger names.
const WL_STASIS_U32: u32 = WL_STASIS as u32;
const WL_DETECT_U32: u32 = WL_DETECT as u32;
const WL_EWALL_U32: u32 = WL_EWALL as u32;
const WL_ALLOWLIQUID_U32: u32 = WL_ALLOWLIQUID as u32;
const WL_WALLELEC_U32: u32 = WL_WALLELEC as u32;
const WL_ALLOWALLELEC_U32: u32 = WL_ALLOWALLELEC as u32;
const WL_EHOLE_U32: u32 = WL_EHOLE as u32;

/// Integer hash with good avalanche behaviour in `u32` (lowbias32-style).
#[cube]
fn gpu_hash(seed: u32) -> u32 {
    let mut h = seed ^ 0x9E37_79B9u32;
    h ^= h >> 16u32;
    h *= 0x7FEB_352Du32;
    h ^= h >> 15u32;
    h *= 0x846C_A68Bu32;
    h ^ (h >> 16u32)
}

/// `index * salt`, kept behind a call so the product is never constant-folded.
#[cube]
fn mix_index(index: u32, salt: u32) -> u32 {
    index * salt
}

/// Seed a runtime-assigned `i32` device local from a runtime anchor.
///
/// A `let mut` initialised from a host literal stays a host `i32` in the
/// expanded body, so the later runtime assignment mis-expands (AGENTS rule).
/// Routing the seed through this call keeps the local a `NativeExpand<i32>`.
#[cube]
#[allow(clippy::erasing_op)]
fn seed_i32(anchor: i32, value: i32) -> i32 {
    anchor * 0i32 + value
}

/// Clamp `v` into `[lo, hi]` (`f32::clamp` is unavailable inside a cube body).
#[cube]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    let mut out = v;
    if out < lo {
        out = lo;
    }
    if out > hi {
        out = hi;
    }
    out
}

/// `i32` clamp.
#[cube]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    let mut out = v;
    if out < lo {
        out = lo;
    }
    if out > hi {
        out = hi;
    }
    out
}

/// `f32` absolute value without `f32::abs` (unavailable in a cube body).
#[cube]
fn absf(v: f32) -> f32 {
    let mut out = v;
    if out < 0.0f32 {
        out = 0.0f32 - out;
    }
    out
}

/// `SimulationData::IsHeatInsulator` (`SimulationData.cpp:338-341`).
///
/// A particle is an insulator when its `HeatConduct` is zero, when it is a
/// `HSWC` whose `life` is not 10, or when it is a `PIPE`/`PPIP` without the
/// `PFLAG_CAN_CONDUCT` bit set.
#[cube]
fn is_heat_insulator(ty: u32, life: i32, tmp: i32, heat_conduct: u32) -> u32 {
    let mut out = 0u32;
    if heat_conduct == 0u32
        || (ty == PT_HSWC_U32 && life != 10i32)
        || ((ty == PT_PIPE_U32 || ty == PT_PPIP_U32) && (tmp & PFLAG_CAN_CONDUCT) == 0i32)
    {
        out = 1u32;
    }
    out
}

/// `SimulationData::HeatCapacityOf` (`SimulationData.cpp:343-349`).
///
/// `BASE` is the only built-in override (`elements/BASE.cpp:33`); `PIPE` and
/// `PPIP` add the capacity of the element stored in `ctype`.
#[cube]
fn heat_capacity(ty: u32, ctype: u32, elem_rows: u32) -> f32 {
    let mut hc = 1.0f32;
    if ty == PT_BASE_U32 {
        hc = 1.5f32;
    }
    if (ty == PT_PIPE_U32 || ty == PT_PPIP_U32) && ctype > 0u32 && ctype < elem_rows {
        let mut extra = 1.0f32;
        if ctype == PT_BASE_U32 {
            extra = 1.5f32;
        }
        hc += extra;
    }
    hc
}

/// `TransitionPhase`'s pairwise conduction exclusions
/// (`Simulation.cpp:2482-2490`).
#[cube]
fn conduct_pair_excluded(t: u32, rt: u32, self_tmp: i32, rt_tmp: i32) -> u32 {
    let mut out = 0u32;
    if (t == PT_FILT_U32 && (rt == PT_BRAY_U32 || rt == PT_BIZR_U32 || rt == PT_BIZRG_U32))
        || (rt == PT_FILT_U32
            && (t == PT_BRAY_U32 || t == PT_PHOT_U32 || t == PT_BIZR_U32 || t == PT_BIZRG_U32))
        || (t == PT_ELEC_U32 && rt == PT_DEUT_U32)
        || (t == PT_DEUT_U32 && rt == PT_ELEC_U32)
        || (t == PT_HSWC_U32 && rt == PT_FILT_U32 && self_tmp == 1i32)
        || (t == PT_FILT_U32 && rt == PT_HSWC_U32 && rt_tmp == 1i32)
    {
        out = 1u32;
    }
    out
}

/// `Simulation::GetGravityField(x, y, -2.0f, -2.0f, ...)` for the liquid
/// convection (`Simulation.cpp:2088-2125`, called at `:2438-2441`).
///
/// Returns `(gx, gy)` in pixel/step units: the particle-gravity term plus
/// `-2 * gravOut.force[cell]` (the reference passes `-2` as `newtonGrav` too).
#[cube]
#[allow(clippy::too_many_arguments)]
fn convection_gravity(
    x: i32,
    y: i32,
    xres: u32,
    yres: u32,
    ci: usize,
    grav_fx: &[f32],
    grav_fy: &[f32],
    gravity_mode: u32,
    custom_gravity_x: f32,
    custom_gravity_y: f32,
) -> (f32, f32) {
    let particle_grav = -2.0f32;
    let mut gx = 0.0f32;
    let mut gy = 0.0f32;
    if gravity_mode == 0u32 {
        // `GRAV_VERTICAL`.
        gy = particle_grav;
    } else if gravity_mode == 2u32 {
        // `GRAV_RADIAL`.
        let dx = x as f32 - xres as f32 * 0.5f32;
        let dy = y as f32 - yres as f32 * 0.5f32;
        if dx != 0.0f32 || dy != 0.0f32 {
            let d = 0.01f32 - (dx * dx + dy * dy).sqrt();
            gx = particle_grav * (dx / d);
            gy = particle_grav * (dy / d);
        }
    } else if gravity_mode == 3u32 {
        // `GRAV_CUSTOM`.
        gx = particle_grav * custom_gravity_x;
        gy = particle_grav * custom_gravity_y;
    }
    // `newtonGrav = -2` in the reference call.
    gx += -2.0f32 * grav_fx[ci];
    gy += -2.0f32 * grav_fy[ci];
    (gx, gy)
}

/// Heat conduction and ambient heat exchange for every live particle.
///
/// Writes the equilibrium temperature `pt` into `part_equil` (`-1` when the
/// conduction gate did not pass). The ambient exchange mutates `air_hv` exactly
/// like `Simulation.cpp:2455-2463`; the `blockairh` counter is incremented for
/// every particle that fails the gate (`Simulation.cpp:2770-2776`).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn thermal_conduction_kernel(
    part_ty: &[u32],
    part_life: &[i32],
    part_ctype: &[u32],
    part_tmp: &[i32],
    part_x: &[f32],
    part_y: &[f32],
    part_temp: &[f32],
    part_equil: &mut [f32],
    pmap: &[u32],
    air_hv: &mut [f32],
    block_airh: &mut [u32],
    elem_props: &[u32],
    slot_count: u32,
    elem_rows: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    aheat_enable: u32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i < slot_count as usize {
        let ty = part_ty[i];
        if ty != PT_NONE_U32 && ty < elem_rows {
            let base = ty as usize * GPU_ELEM_STRIDE;
            let heat_conduct = elem_props[base + COL_HEAT_CONDUCT];
            let x = part_x[i] as i32;
            let y = part_y[i] as i32;
            if x >= 0i32 && x < xres as i32 && y >= 0i32 && y < yres as i32 {
                let px = x as u32;
                let py = y as u32;
                let ci = ((py / CELL) * xcells + (px / CELL)) as usize;
                let life = part_life[i];
                let tmp = part_tmp[i];
                let ctype = part_ctype[i];

                // `gel_scale` (`Simulation.cpp:2433-2436`).
                let mut gel_scale = 1.0f32;
                if ty == PT_GEL_U32 {
                    gel_scale = tmp as f32 * 2.55f32;
                }
                let mut scaled = heat_conduct as f32 * gel_scale;
                if scaled < 0.0f32 {
                    scaled = 0.0f32;
                }
                let threshold = scaled as u32;

                let roll = gpu_hash(frame_seed ^ mix_index(i as u32, PARTICLE_HASH_SALT) ^ ty)
                    % CONDUCT_ROLL_MOD;
                let insulator = is_heat_insulator(ty, life, tmp, heat_conduct);
                let mut gate = 0u32;
                if insulator == 0u32 && roll < threshold {
                    gate = 1u32;
                }

                if gate != 0u32 {
                    // Ambient exchange (`Simulation.cpp:2455-2463`).
                    let mut self_temp = part_temp[i];
                    if aheat_enable != 0u32 {
                        let props = elem_props[base + COL_PROPERTIES];
                        if (props & PROP_NOAMBHEAT_U32) == 0u32 {
                            let dtemp = air_hv[ci] - part_temp[i];
                            let hc = heat_capacity(ty, ctype, elem_rows);
                            let mut alpha = AMBIENT_ALPHA_SCALE * hc;
                            if alpha > AMBIENT_ALPHA_MAX {
                                alpha = AMBIENT_ALPHA_MAX;
                            }
                            // The particle side is divided by `hc`, the air
                            // side is not (`Simulation.cpp:2462-2463`).
                            self_temp =
                                clampf(part_temp[i] + alpha * dtemp / hc, MIN_TEMP, MAX_TEMP);
                            air_hv[ci] = clampf(air_hv[ci] - alpha * dtemp, MIN_TEMP, MAX_TEMP);
                        }
                    }

                    // Weighted equilibrium over the particle and its eight
                    // conducting `pmap` neighbours (`Simulation.cpp:2466-2503`).
                    let hc_self = heat_capacity(ty, ctype, elem_rows);
                    let mut c_heat = self_temp * hc_self;
                    let mut hc_total = hc_self;
                    let mut ny = 0usize;
                    while ny < 3usize {
                        let mut nx = 0usize;
                        while nx < 3usize {
                            if nx != 1usize || ny != 1usize {
                                let nxi = x + nx as i32 - 1i32;
                                let nyi = y + ny as i32 - 1i32;
                                if nxi >= 0i32
                                    && nxi < xres as i32
                                    && nyi >= 0i32
                                    && nyi < yres as i32
                                {
                                    let pidx = (nyi as u32 * xres + nxi as u32) as usize;
                                    let packed = pmap[pidx];
                                    if (packed >> 9u32) != 0u32 {
                                        let nid = ((packed >> 9u32) - 1u32) as usize;
                                        if nid < slot_count as usize {
                                            let nty = part_ty[nid];
                                            if nty != PT_NONE_U32 && nty < elem_rows {
                                                let nbase = nty as usize * GPU_ELEM_STRIDE;
                                                let n_conduct =
                                                    elem_props[nbase + COL_HEAT_CONDUCT];
                                                let n_insulator = is_heat_insulator(
                                                    nty,
                                                    part_life[nid],
                                                    part_tmp[nid],
                                                    n_conduct,
                                                );
                                                let excluded = conduct_pair_excluded(
                                                    ty,
                                                    nty,
                                                    tmp,
                                                    part_tmp[nid],
                                                );
                                                if n_insulator == 0u32 && excluded == 0u32 {
                                                    let n_hc = heat_capacity(
                                                        nty,
                                                        part_ctype[nid],
                                                        elem_rows,
                                                    );
                                                    c_heat += part_temp[nid] * n_hc;
                                                    hc_total += n_hc;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            nx += 1usize;
                        }
                        ny += 1usize;
                    }
                    part_equil[i] = clampf(c_heat / hc_total, MIN_TEMP, MAX_TEMP);
                } else {
                    // `Simulation.cpp:2770-2776`: a particle that fails the
                    // gate (or is an insulator) saturates its cell's
                    // `blockairh` counter at 0x8.
                    if (block_airh[ci] & 0x8u32) == 0u32 {
                        block_airh[ci] += 1u32;
                    }
                    part_equil[i] = -1.0f32;
                }
            }
        }
    }
}

/// Applies the conduction equilibrium to every participant
/// (`Simulation.cpp:2505-2510`).
///
/// The reference sets `parts[i].temp = pt` for the particle and all of its
/// conducting neighbours, in slot order. The parallel pass gives each particle
/// the `pt` of the **highest slot** whose neighbourhood includes it, which is
/// exactly the value the last serial writer would leave behind.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn thermal_apply_kernel(
    part_ty: &[u32],
    part_life: &[i32],
    part_tmp: &[i32],
    part_x: &[f32],
    part_y: &[f32],
    part_temp: &mut [f32],
    part_equil: &[f32],
    pmap: &[u32],
    elem_props: &[u32],
    slot_count: u32,
    elem_rows: u32,
    xres: u32,
    yres: u32,
) {
    let j = ABSOLUTE_POS;
    if j < slot_count as usize {
        let jty = part_ty[j];
        if jty != PT_NONE_U32 && jty < elem_rows {
            let x = part_x[j] as i32;
            let y = part_y[j] as i32;
            if x >= 0i32 && x < xres as i32 && y >= 0i32 && y < yres as i32 {
                // `best_id` in the reference starts at -1 (no participant) and
                // records the highest slot whose neighbourhood covers `j`.
                // `has_best` carries the "found" bit so the scan never has to
                // hold a negative slot id in a runtime-typed local.
                let mut best_slot = j;
                let mut best_temp = part_temp[j];
                let mut has_best = 0u32;
                if part_equil[j] >= 0.0f32 {
                    best_temp = part_equil[j];
                    has_best = 1u32;
                }
                let j_conduct = elem_props[jty as usize * GPU_ELEM_STRIDE + COL_HEAT_CONDUCT];
                let j_insulator = is_heat_insulator(jty, part_life[j], part_tmp[j], j_conduct);
                let mut ny = 0usize;
                while ny < 3usize {
                    let mut nx = 0usize;
                    while nx < 3usize {
                        if nx != 1usize || ny != 1usize {
                            let nxi = x + nx as i32 - 1i32;
                            let nyi = y + ny as i32 - 1i32;
                            if nxi >= 0i32 && nxi < xres as i32 && nyi >= 0i32 && nyi < yres as i32
                            {
                                let pidx = (nyi as u32 * xres + nxi as u32) as usize;
                                let packed = pmap[pidx];
                                if (packed >> 9u32) != 0u32 {
                                    let nid = ((packed >> 9u32) - 1u32) as usize;
                                    if nid < slot_count as usize && nid != j {
                                        let nty = part_ty[nid];
                                        if nty != PT_NONE_U32
                                            && nty < elem_rows
                                            && (has_best == 0u32 || nid > best_slot)
                                            && part_equil[nid] >= 0.0f32
                                            && j_insulator == 0u32
                                        {
                                            let n_conduct = elem_props
                                                [nty as usize * GPU_ELEM_STRIDE + COL_HEAT_CONDUCT];
                                            let n_insulator = is_heat_insulator(
                                                nty,
                                                part_life[nid],
                                                part_tmp[nid],
                                                n_conduct,
                                            );
                                            let excluded = conduct_pair_excluded(
                                                nty,
                                                jty,
                                                part_tmp[nid],
                                                part_tmp[j],
                                            );
                                            if n_insulator == 0u32 && excluded == 0u32 {
                                                best_slot = nid;
                                                best_temp = part_equil[nid];
                                                has_best = 1u32;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        nx += 1usize;
                    }
                    ny += 1usize;
                }
                if has_best != 0u32 {
                    part_temp[j] = clampf(best_temp, MIN_TEMP, MAX_TEMP);
                } else {
                    part_temp[j] = clampf(part_temp[j], MIN_TEMP, MAX_TEMP);
                }
            }
        }
    }
}

/// Temperature, phase and pressure transitions for every live particle.
///
/// Ports the `!legacy_enable` half of `TransitionPhase` (`Simulation.cpp:2430-2866`)
/// when `legacy_enable == 0`, and always runs the `LIFE` drain, the wall-spark
/// triggers, the pressure detonation and the pressure/gravity transitions
/// (those blocks sit outside the legacy gate in the reference).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn phase_transition_kernel(
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_x: &[f32],
    part_y: &[f32],
    part_equil: &[f32],
    pmap: &[u32],
    air_pv: &mut [f32],
    grav_fx: &[f32],
    grav_fy: &[f32],
    wall_bmap: &[u32],
    wall_emap: &mut [u32],
    elem_props: &[u32],
    slot_count: u32,
    elem_rows: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    legacy_enable: u32,
    gravity_mode: u32,
    custom_gravity_x: f32,
    custom_gravity_y: f32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i < slot_count as usize {
        let mut t = part_ty[i];
        if t != PT_NONE_U32 && t < elem_rows {
            let x = part_x[i] as i32;
            let y = part_y[i] as i32;
            if x >= 0i32 && x < xres as i32 && y >= 0i32 && y < yres as i32 {
                let ci = (((y as u32) / CELL) * xcells + ((x as u32) / CELL)) as usize;
                let mut pv = air_pv[ci];
                let roll = gpu_hash(
                    frame_seed ^ mix_index(i as u32, PARTICLE_HASH_SALT) ^ t ^ TRANSITION_HASH_SALT,
                );

                if legacy_enable == 0u32 {
                    let props = elem_props[t as usize * GPU_ELEM_STRIDE + COL_PROPERTIES];
                    let mut ctype = part_ctype[i];
                    let tmp = part_tmp[i];
                    let tmp2 = part_tmp2[i];
                    let mut gel_scale = 1.0f32;
                    if t == PT_GEL_U32 {
                        gel_scale = tmp as f32 * 2.55f32;
                    }

                    // Liquid heat convection (`Simulation.cpp:2438-2449`).
                    if (props & TYPE_LIQUID_U32) != 0u32 {
                        let mut convect = 1u32;
                        if t == PT_GEL_U32 {
                            let gate_roll = gpu_hash(roll ^ 0x1B87_3593u32) % 255u32;
                            if gel_scale <= (1.0f32 + gate_roll as f32) {
                                convect = 0u32;
                            }
                        }
                        if convect != 0u32 {
                            let (gx, gy) = convection_gravity(
                                x,
                                y,
                                xres,
                                yres,
                                ci,
                                grav_fx,
                                grav_fy,
                                gravity_mode,
                                custom_gravity_x,
                                custom_gravity_y,
                            );
                            let rxf = gx + x as f32;
                            let ryf = gy + y as f32;
                            // `(int)(r + 0.5)` rounds toward the nearest cell.
                            let rx = if rxf >= 0.0f32 {
                                (rxf + 0.5f32) as i32
                            } else {
                                (rxf - 0.5f32) as i32
                            };
                            let ry = if ryf >= 0.0f32 {
                                (ryf + 0.5f32) as i32
                            } else {
                                (ryf - 0.5f32) as i32
                            };
                            let ox = clampi(rx, x - 1i32, x + 1i32);
                            let oy = clampi(ry, y - 1i32, y + 1i32);
                            if ox != x || oy != y {
                                let pidx = ((oy as u32) * xres + (ox as u32)) as usize;
                                let packed = pmap[pidx];
                                if packed != 0u32 {
                                    let nid = ((packed >> 9u32) - 1u32) as usize;
                                    if nid < slot_count as usize
                                        && part_ty[nid] == t
                                        && part_temp[i] > part_temp[nid]
                                    {
                                        let hot = part_temp[i];
                                        let cold = part_temp[nid];
                                        part_temp[i] = cold;
                                        part_temp[nid] = hot;
                                    }
                                }
                            }
                        }
                    }

                    // The conduction gate decides whether the transition half
                    // runs at all (`Simulation.cpp:2452`).
                    if part_equil[i] >= 0.0f32 {
                        let pt = part_equil[i];
                        // A fix for ice with `ctype = 0`
                        // (`Simulation.cpp:2528-2530`).
                        if (t == PT_ICEI_U32 || t == PT_SNOW_U32)
                            && (ctype == 0u32
                                || ctype >= elem_rows
                                || ctype == PT_ICEI_U32
                                || ctype == PT_SNOW_U32)
                        {
                            ctype = PT_WATR_U32;
                            part_ctype[i] = ctype;
                        }

                        let high_temp = f32::reinterpret(
                            elem_props[t as usize * GPU_ELEM_STRIDE + COL_HIGH_TEMP],
                        );
                        let high_transition = elem_props
                            [t as usize * GPU_ELEM_STRIDE + COL_HIGH_TEMP_TRANSITION]
                            as i32;
                        let low_temp = f32::reinterpret(
                            elem_props[t as usize * GPU_ELEM_STRIDE + COL_LOW_TEMP],
                        );
                        let low_transition = elem_props
                            [t as usize * GPU_ELEM_STRIDE + COL_LOW_TEMP_TRANSITION]
                            as i32;

                        // `Simulation.cpp:2511-2519`: the pressure shift only
                        // applies to a liquid boiling into a gas (and
                        // `LNTG`/`SLTW`) or a gas condensing into a liquid (and
                        // `WTRV`).
                        let pv_shift = 2.0f32 * pv;
                        let mut boil_shifted = 0u32;
                        if (props & TYPE_LIQUID_U32) != 0u32
                            && high_transition >= 0
                            && (high_transition as u32) < PT_NUM_U32
                            && (elem_props
                                [high_transition as usize * GPU_ELEM_STRIDE + COL_PROPERTIES]
                                & TYPE_GAS_U32)
                                != 0u32
                        {
                            boil_shifted = 1u32;
                        }
                        if t == PT_LN2_U32 || t == PT_SLTW_U32 {
                            boil_shifted = 1u32;
                        }
                        let mut condense_shifted = 0u32;
                        if (props & TYPE_GAS_U32) != 0u32
                            && low_transition >= 0
                            && (low_transition as u32) < PT_NUM_U32
                            && (elem_props
                                [low_transition as usize * GPU_ELEM_STRIDE + COL_PROPERTIES]
                                & TYPE_LIQUID_U32)
                                != 0u32
                        {
                            condense_shifted = 1u32;
                        }
                        if t == PT_WTRV_U32 {
                            condense_shifted = 1u32;
                        }
                        let ctemph = if boil_shifted != 0u32 {
                            pt - pv_shift
                        } else {
                            pt
                        };
                        let ctempl = if condense_shifted != 0u32 {
                            pt - pv_shift
                        } else {
                            pt
                        };

                        let mut old_ty = t;
                        let mut nt = t;
                        let mut new_life = part_life[i];
                        let mut new_ctype = ctype;
                        let mut new_tmp = tmp;
                        let mut new_tmp2 = tmp2;
                        let mut s = 0u32;

                        if high_transition != NT_I32 && ctemph >= high_temp {
                            // High-temperature transition
                            // (`Simulation.cpp:2532-2604`).
                            if high_transition != ST_I32 {
                                if t == PT_FOG_U32 {
                                    new_ctype = 0u32;
                                }
                                nt = high_transition as u32;
                                s = 1u32;
                            } else if t == PT_ICEI_U32 || t == PT_SNOW_U32 {
                                let mut melts = 0u32;
                                if new_ctype > 0u32 && new_ctype < PT_NUM_U32 && new_ctype != t {
                                    let cbase = new_ctype as usize * GPU_ELEM_STRIDE;
                                    let ct_trans =
                                        elem_props[cbase + COL_LOW_TEMP_TRANSITION] as i32;
                                    let ct_low = f32::reinterpret(elem_props[cbase + COL_LOW_TEMP]);
                                    if ct_trans == PT_ICEI as i32 || ct_trans == PT_SNOW as i32 {
                                        if pt >= ct_low {
                                            melts = 1u32;
                                        }
                                    } else if pt >= WATER_MELT_K {
                                        melts = 1u32;
                                    }
                                }
                                if melts != 0u32 {
                                    nt = new_ctype;
                                    new_ctype = PT_NONE_U32;
                                    new_life = 0i32;
                                    s = 1u32;
                                }
                            } else if t == PT_SLTW_U32 {
                                // `@ SLTW -> SALT/WTRV` (`Simulation.cpp:2560-2564`).
                                if roll.is_multiple_of(4u32) {
                                    nt = PT_SALT_U32;
                                } else {
                                    nt = PT_WTRV_U32;
                                }
                                s = 1u32;
                            } else if t == PT_BRMT_U32 {
                                // `@ BRMT(TUNG) -> LAVA(TUNG)`
                                // (`Simulation.cpp:2566-2586`).
                                if new_ctype == PT_TUNG_U32 {
                                    let tung_hot = f32::reinterpret(
                                        elem_props[PT_TUNG_U32 as usize * GPU_ELEM_STRIDE
                                            + COL_HIGH_TEMP],
                                    );
                                    if ctemph >= tung_hot {
                                        nt = PT_LAVA_U32;
                                        // `parts[i].type = PT_TUNG`: the s block
                                        // records TUNG as the lava's ctype.
                                        old_ty = PT_TUNG_U32;
                                        s = 1u32;
                                    }
                                } else if ctemph >= high_temp {
                                    nt = PT_LAVA_U32;
                                    s = 1u32;
                                }
                            } else if t == PT_CRMC_U32 {
                                // `@ CRMC -> LAVA` (`Simulation.cpp:2588-2595`).
                                let cy = (y as u32) / CELL;
                                let cx = (x as u32) / CELL;
                                let ym = max(y - 2i32, 0i32) as u32 / CELL;
                                let xm = max(x - 2i32, 0i32) as u32 / CELL;
                                let yp = min(y + 2i32, yres as i32 - 1i32) as u32 / CELL;
                                let xp = min(x + 2i32, xres as i32 - 1i32) as u32 / CELL;
                                let pres = max(
                                    (air_pv[(cy * xcells + cx) as usize]
                                        + air_pv[(ym * xcells + cx) as usize]
                                        + air_pv[(yp * xcells + cx) as usize]
                                        + air_pv[(cy * xcells + xm) as usize]
                                        + air_pv[(cy * xcells + xp) as usize])
                                        * 2.0f32,
                                    0.0f32,
                                );
                                if ctemph >= pres + high_temp {
                                    nt = PT_LAVA_U32;
                                    s = 1u32;
                                }
                            } else if t == PT_RIME_U32 {
                                // `@ RIME -> ACID` / `@ RIME -> WATR`
                                // (`Simulation.cpp:2597-2611`).
                                if new_tmp > 5i32 {
                                    nt = PT_ACID_U32;
                                    new_life = 25i32 + 5i32 * new_tmp;
                                    new_tmp = 0i32;
                                } else {
                                    if new_ctype == PT_DSTW_U32 {
                                        nt = PT_DSTW_U32;
                                    } else {
                                        nt = PT_WATR_U32;
                                    }
                                    new_ctype = 0u32;
                                }
                                s = 1u32;
                            }
                        } else if low_transition != NT_I32 && ctempl < low_temp {
                            // Low-temperature transition
                            // (`Simulation.cpp:2613-2698`).
                            if low_transition != ST_I32 {
                                nt = low_transition as u32;
                                s = 1u32;
                            } else if t == PT_WTRV_U32 {
                                // `@ WTRV -> RIME/DSTW`.
                                if pt < STEAM_FREEZE_K {
                                    nt = PT_RIME_U32;
                                } else {
                                    nt = PT_DSTW_U32;
                                }
                                s = 1u32;
                            } else if t == PT_LAVA_U32 {
                                if new_ctype > 0u32
                                    && new_ctype < PT_NUM_U32
                                    && new_ctype != PT_LAVA_U32
                                {
                                    let cbase = new_ctype as usize * GPU_ELEM_STRIDE;
                                    let c_high =
                                        f32::reinterpret(elem_props[cbase + COL_HIGH_TEMP]);
                                    let c_high_trans =
                                        elem_props[cbase + COL_HIGH_TEMP_TRANSITION] as i32;
                                    let bmtl_high = f32::reinterpret(
                                        elem_props[PT_BMTL_U32 as usize * GPU_ELEM_STRIDE
                                            + COL_HIGH_TEMP],
                                    );
                                    let mut freeze = 0u32;
                                    if new_ctype == PT_THRM_U32 {
                                        if pt < bmtl_high {
                                            freeze = 1u32;
                                        }
                                    } else if new_ctype == PT_VIBR_U32 || new_ctype == PT_BVBR_U32 {
                                        if pt < VIBR_FREEZE_K {
                                            freeze = 1u32;
                                        }
                                    } else if new_ctype == PT_TUNG_U32 {
                                        if pt < c_high {
                                            freeze = 1u32;
                                        }
                                    } else if new_ctype == PT_CRMC_U32 {
                                        let cy = (y as u32) / CELL;
                                        let cx = (x as u32) / CELL;
                                        let ym = max(y - 2i32, 0i32) as u32 / CELL;
                                        let xm = max(x - 2i32, 0i32) as u32 / CELL;
                                        let yp = min(y + 2i32, yres as i32 - 1i32) as u32 / CELL;
                                        let xp = min(x + 2i32, xres as i32 - 1i32) as u32 / CELL;
                                        let pres = max(
                                            (air_pv[(cy * xcells + cx) as usize]
                                                + air_pv[(ym * xcells + cx) as usize]
                                                + air_pv[(yp * xcells + cx) as usize]
                                                + air_pv[(cy * xcells + xm) as usize]
                                                + air_pv[(cy * xcells + xp) as usize])
                                                * 2.0f32,
                                            0.0f32,
                                        );
                                        if ctemph >= pres + c_high {
                                            freeze = 0u32;
                                        } else {
                                            freeze = 1u32;
                                        }
                                    } else if c_high_trans == PT_LAVA as i32
                                        || new_ctype == PT_HEAC_U32
                                    {
                                        if pt < c_high {
                                            freeze = 1u32;
                                        }
                                    } else if pt < LAVA_FREEZE_K {
                                        freeze = 1u32;
                                    }
                                    if freeze != 0u32 {
                                        if new_ctype == PT_THRM_U32 {
                                            nt = PT_BMTL_U32;
                                            new_tmp = 0i32;
                                        } else if new_ctype == PT_PLUT_U32 {
                                            nt = PT_LAVA_U32;
                                            new_tmp = 0i32;
                                        } else {
                                            nt = new_ctype;
                                        }
                                        new_ctype = PT_NONE_U32;
                                        s = 1u32;
                                    }
                                } else if pt < LAVA_FREEZE_K {
                                    nt = PT_STNE_U32;
                                    s = 1u32;
                                }
                            }
                        }

                        if s != 0u32 {
                            // Post-transition bookkeeping
                            // (`Simulation.cpp:2710-2745`).
                            if nt == PT_ICEI_U32 || nt == PT_LAVA_U32 || nt == PT_SNOW_U32 {
                                new_ctype = old_ty;
                            }
                            if nt == PT_RIME_U32 {
                                new_ctype = PT_DSTW_U32;
                            }
                            if !(nt == PT_ICEI_U32 && new_ctype == PT_FRZW_U32) && nt != PT_ACID_U32
                            {
                                new_life = 0i32;
                            }
                            if nt == PT_FIRE_U32 {
                                new_tmp = 0i32;
                                if old_ty == PT_SEED_U32 {
                                    new_ctype = 0u32;
                                    new_tmp2 = 0i32;
                                }
                            }
                            if (elem_props[nt as usize * GPU_ELEM_STRIDE + COL_PROPERTIES]
                                & TYPE_GAS_U32)
                                != 0u32
                                && (elem_props[old_ty as usize * GPU_ELEM_STRIDE + COL_PROPERTIES]
                                    & TYPE_GAS_U32)
                                    == 0u32
                            {
                                pv += 0.50f32;
                            }

                            if nt == PT_NONE_U32 {
                                part_ty[i] = PT_NONE_U32;
                                t = PT_NONE_U32;
                            } else {
                                part_ty[i] = nt;
                                t = nt;
                                if nt == PT_FIRE_U32 || nt == PT_PLSM_U32 || nt == PT_CFLM_U32 {
                                    new_life = FIRE_LIFE_BASE + (roll % FIRE_LIFE_SPAN) as i32;
                                }
                                if nt == PT_LAVA_U32 {
                                    if new_ctype == PT_BRMT_U32 {
                                        new_ctype = PT_BMTL_U32;
                                    } else if new_ctype == PT_SAND_U32
                                        || new_ctype == PT_BGLA_U32
                                        || (new_ctype == PT_LITH_U32 && new_tmp2 > 3i32)
                                    {
                                        new_ctype = PT_GLAS_U32;
                                    } else if new_ctype == PT_PQRT_U32 {
                                        new_ctype = PT_QRTZ_U32;
                                    }
                                    new_life = LAVA_LIFE_BASE + (roll % LAVA_LIFE_SPAN) as i32;
                                }
                            }
                            part_ctype[i] = new_ctype;
                            part_life[i] = new_life;
                            part_tmp[i] = new_tmp;
                            part_tmp2[i] = new_tmp2;
                        }

                        // `pt = parts[i].temp = restrict_flt(...)` plus the
                        // LAVA life feedback (`Simulation.cpp:2747-2765`).
                        // The conduction apply pass already wrote the
                        // highest-slot equilibrium into `part_temp`, which is
                        // the value the serial reference leaves behind; only
                        // the LAVA temperature overrides are written here.
                        let mut pt2 = clampf(pt, MIN_TEMP, MAX_TEMP);
                        if t == PT_LAVA_U32 {
                            let mut lava_life = ((pt2 - 700.0f32) / 7.0f32) as i32;
                            lava_life = clampi(lava_life, 0i32, 400i32);
                            part_life[i] = lava_life;
                            if part_ctype[i] == PT_THRM_U32 && part_tmp[i] > 0i32 {
                                part_tmp[i] -= 1i32;
                                pt2 = THERMITE_TEMP;
                            }
                            if part_ctype[i] == PT_PLUT_U32 && part_tmp[i] > 0i32 {
                                part_tmp[i] -= 1i32;
                                pt2 = MAX_TEMP;
                            }
                            part_temp[i] = pt2;
                        }
                    } else {
                        part_temp[i] = clampf(part_temp[i], MIN_TEMP, MAX_TEMP);
                    }
                }

                if t != PT_NONE_U32 && t < elem_rows {
                    // `if (t==PT_LIFE) temp -= 50` (`Simulation.cpp:2767-2770`).
                    if t == PT_LIFE_U32 {
                        part_temp[i] = clampf(part_temp[i] - 50.0f32, MIN_TEMP, MAX_TEMP);
                    }

                    // Spark updates from walls (`Simulation.cpp:2778-2811`).
                    let tprops = elem_props[t as usize * GPU_ELEM_STRIDE + COL_PROPERTIES];
                    if (tprops & PROP_CONDUCTS_U32) != 0u32 || t == PT_SPRK_U32 {
                        let mut nx = x % CELL as i32;
                        if nx == 0i32 {
                            nx = x / CELL as i32 - 1i32;
                        } else if nx == CELL as i32 - 1i32 {
                            nx = x / CELL as i32 + 1i32;
                        } else {
                            nx = x / CELL as i32;
                        }
                        let mut ny = y % CELL as i32;
                        if ny == 0i32 {
                            ny = y / CELL as i32 - 1i32;
                        } else if ny == CELL as i32 - 1i32 {
                            ny = y / CELL as i32 + 1i32;
                        } else {
                            ny = y / CELL as i32;
                        }
                        if nx >= 0i32 && ny >= 0i32 && nx < xcells as i32 && ny < ycells as i32 {
                            let wi = ((ny as u32) * xcells + (nx as u32)) as usize;
                            if t != PT_SPRK_U32 {
                                if (wall_emap[wi] & 0xFFu32) == EMAP_SPARKING
                                    && part_life[i] == 0i32
                                    && (wall_bmap[wi] & 0xFFu32) != WL_STASIS_U32
                                {
                                    part_ty[i] = PT_SPRK_U32;
                                    part_life[i] = SPARK_LIFE;
                                    part_ctype[i] = t;
                                    t = PT_SPRK_U32;
                                }
                            } else {
                                let wall = wall_bmap[wi] & 0xFFu32;
                                if wall == WL_DETECT_U32
                                    || wall == WL_EWALL_U32
                                    || wall == WL_ALLOWLIQUID_U32
                                    || wall == WL_WALLELEC_U32
                                    || wall == WL_ALLOWALLELEC_U32
                                    || wall == WL_EHOLE_U32
                                {
                                    wall_emap[wi] = EMAP_CHARGED;
                                }
                            }
                        }
                    }

                    // The basic explosion, from the `.explosive` variable
                    // (`Simulation.cpp:2813-2820`).
                    let explosive = elem_props[t as usize * GPU_ELEM_STRIDE + COL_EXPLOSIVE] as i32;
                    if (explosive & 2) != 0 && pv > DETONATE_PRESSURE {
                        let fire_temp = f32::reinterpret(
                            elem_props[PT_FIRE_U32 as usize * GPU_ELEM_STRIDE + COL_DEFAULT_TEMP],
                        );
                        let flammable =
                            elem_props[t as usize * GPU_ELEM_STRIDE + COL_FLAMMABLE] as i32;
                        part_temp[i] =
                            clampf(fire_temp + (flammable / 2i32) as f32, MIN_TEMP, MAX_TEMP);
                        part_life[i] = BLAST_LIFE_BASE + (roll % BLAST_LIFE_SPAN) as i32;
                        part_ty[i] = PT_FIRE_U32;
                        t = PT_FIRE_U32;
                        pv += DETONATE_PRESSURE_GAIN;
                    }

                    // Pressure / gravity transitions (`Simulation.cpp:2822-2866`).
                    let mut s = 0u32;
                    let mut nt = t;
                    let gravtot = absf(grav_fx[ci]) + absf(grav_fy[ci]);
                    let hpt = elem_props
                        [t as usize * GPU_ELEM_STRIDE + COL_HIGH_PRESSURE_TRANSITION]
                        as i32;
                    let hpt_val = f32::reinterpret(
                        elem_props[t as usize * GPU_ELEM_STRIDE + COL_HIGH_PRESSURE],
                    );
                    let lpt = elem_props[t as usize * GPU_ELEM_STRIDE + COL_LOW_PRESSURE_TRANSITION]
                        as i32;
                    let lpt_val = f32::reinterpret(
                        elem_props[t as usize * GPU_ELEM_STRIDE + COL_LOW_PRESSURE],
                    );
                    let tmp = part_tmp[i];
                    if hpt != NT_I32 && pv > hpt_val {
                        if hpt != ST_I32 {
                            nt = hpt as u32;
                            s = 1u32;
                        } else if t == PT_BMTL_U32 && (pv > 2.5f32 || (pv > 1.0f32 && tmp == 1i32))
                        {
                            nt = PT_BRMT_U32;
                            s = 1u32;
                        }
                    } else if lpt != NT_I32 && pv < lpt_val && gravtot <= lpt_val / 4.0f32 {
                        if lpt != ST_I32 {
                            nt = lpt as u32;
                            s = 1u32;
                        }
                    } else if hpt != NT_I32 && gravtot > hpt_val / 4.0f32 {
                        if hpt != ST_I32 {
                            nt = hpt as u32;
                            s = 1u32;
                        } else if t == PT_BMTL_U32
                            && (gravtot > 0.625f32 || (gravtot > 0.25f32 && tmp == 1i32))
                        {
                            nt = PT_BRMT_U32;
                            s = 1u32;
                        }
                    }
                    if s != 0u32 {
                        if nt == PT_NONE_U32 {
                            part_ty[i] = PT_NONE_U32;
                        } else {
                            part_life[i] = 0i32;
                            if nt == PT_BRMT_U32 {
                                part_ctype[i] = 0u32;
                                part_tmp[i] = 0i32;
                                part_tmp2[i] = 0i32;
                            }
                            part_ty[i] = nt;
                            if nt == PT_FIRE_U32 {
                                part_life[i] = FIRE_LIFE_BASE + (roll % FIRE_LIFE_SPAN) as i32;
                            }
                        }
                    }
                }

                if pv != air_pv[ci] {
                    air_pv[ci] = pv;
                }
            }
        }
    }
}

/// Republish the element-type half of every `pmap` word whose particle changed
/// type during this pass.
///
/// The `pmap` word is `(part_id + 1) << 9 | type`, so a transition invalidates
/// it until the next full rebuild. This kernel is cheap and idempotent: it only
/// touches words that are still owned by the particle and whose type differs.
#[cube(launch_unchecked)]
pub fn refresh_pmap_types_kernel(
    pmap: &mut [u32],
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    slot_count: u32,
    xres: u32,
    yres: u32,
) {
    let i = ABSOLUTE_POS;
    if i < slot_count as usize {
        let ty = part_ty[i];
        if ty != PT_NONE_U32 {
            let x = part_x[i] as i32;
            let y = part_y[i] as i32;
            if x >= 0i32 && x < xres as i32 && y >= 0i32 && y < yres as i32 {
                let pidx = ((y as u32) * xres + (x as u32)) as usize;
                let packed = pmap[pidx];
                if (packed >> 9u32) == (i as u32 + 1u32) && (packed & 0x1FFu32) != ty {
                    pmap[pidx] = ((i as u32 + 1u32) << 9u32) | (ty & 0x1FFu32);
                }
            }
        }
    }
}

/// `Element::legacyUpdate` (`Element.cpp:111-220`), the fallback that runs when
/// heat simulation is disabled (`Simulation.cpp:2398-2400`).
///
/// The five-element water/steam family scans its 5x5 neighbourhood (skipping
/// the centre) for the legacy phase changes, then the pressure rules turn
/// `WTRV`/`OIL`/`GAS`/`DESL` over.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn legacy_update_kernel(
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_x: &[f32],
    part_y: &[f32],
    pmap: &mut [u32],
    air_pv: &[f32],
    slot_count: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i < slot_count as usize {
        let t = part_ty[i];
        if t != PT_NONE_U32 {
            let x = part_x[i] as i32;
            let y = part_y[i] as i32;
            if x >= 0i32 && x < xres as i32 && y >= 0i32 && y < yres as i32 {
                let self_pidx = (y as u32 * xres + x as u32) as usize;
                let mut ry = seed_i32(x, -2i32);
                while ry <= 2i32 {
                    let mut rx = seed_i32(y, -2i32);
                    while rx <= 2i32 {
                        if rx != 0i32 || ry != 0i32 {
                            let nxi = x + rx;
                            let nyi = y + ry;
                            if nxi >= 0i32 && nyi > 0i32 && nxi < xres as i32 && nyi < yres as i32 {
                                let pidx = (nyi as u32 * xres + nxi as u32) as usize;
                                let packed = pmap[pidx];
                                if packed != 0u32 {
                                    let nid = ((packed >> 9u32) - 1u32) as usize;
                                    if nid < slot_count as usize {
                                        let rty = part_ty[nid];
                                        // `part_change_type` refreshes the
                                        // particle's own `pmap` word too
                                        // (`Simulation.cpp:1832-1838`), so both
                                        // words are written where the type
                                        // changes. No local is carried across
                                        // the nested loops.
                                        let roll = gpu_hash(
                                            frame_seed
                                                ^ mix_index(i as u32, PARTICLE_HASH_SALT)
                                                ^ mix_index(
                                                    ((rx + 2i32) * 5i32 + (ry + 2i32)) as u32,
                                                    0x9E37_79B9u32,
                                                ),
                                        );
                                        if t == PT_WTRV_U32
                                            && (rty == PT_WATR_U32
                                                || rty == PT_DSTW_U32
                                                || rty == PT_SLTW_U32)
                                            && roll.is_multiple_of(1000u32)
                                        {
                                            part_ty[i] = PT_WATR_U32;
                                            pmap[self_pidx] =
                                                ((i as u32 + 1u32) << 9u32) | PT_WATR_U32;
                                            part_ty[nid] = PT_WATR_U32;
                                            pmap[pidx] =
                                                ((nid as u32 + 1u32) << 9u32) | PT_WATR_U32;
                                        }
                                        if t == PT_WTRV_U32
                                            && (rty == PT_ICEI_U32 || rty == PT_SNOW_U32)
                                            && roll.is_multiple_of(1000u32)
                                        {
                                            part_ty[i] = PT_WATR_U32;
                                            pmap[self_pidx] =
                                                ((i as u32 + 1u32) << 9u32) | PT_WATR_U32;
                                            if gpu_hash(roll ^ 0x51ED_2701u32)
                                                .is_multiple_of(1000u32)
                                            {
                                                part_ty[nid] = PT_WATR_U32;
                                                pmap[pidx] =
                                                    ((nid as u32 + 1u32) << 9u32) | PT_WATR_U32;
                                            }
                                        }
                                        if (t == PT_WATR_U32 || t == PT_DSTW_U32)
                                            && (rty == PT_FIRE_U32 || rty == PT_LAVA_U32)
                                            && roll.is_multiple_of(10u32)
                                        {
                                            part_ty[i] = PT_WTRV_U32;
                                            pmap[self_pidx] =
                                                ((i as u32 + 1u32) << 9u32) | PT_WTRV_U32;
                                        }
                                        if t == PT_SLTW_U32
                                            && (rty == PT_FIRE_U32 || rty == PT_LAVA_U32)
                                            && roll.is_multiple_of(10u32)
                                        {
                                            if gpu_hash(roll ^ 0xA5A5_5A5Au32).is_multiple_of(4u32)
                                            {
                                                part_ty[i] = PT_SALT_U32;
                                                pmap[self_pidx] =
                                                    ((i as u32 + 1u32) << 9u32) | PT_SALT_U32;
                                            } else {
                                                part_ty[i] = PT_WTRV_U32;
                                                pmap[self_pidx] =
                                                    ((i as u32 + 1u32) << 9u32) | PT_WTRV_U32;
                                            }
                                        }
                                        if t == PT_ICEI_U32
                                            && (rty == PT_WATR_U32 || rty == PT_DSTW_U32)
                                            && roll.is_multiple_of(1000u32)
                                        {
                                            part_ty[i] = PT_ICEI_U32;
                                            pmap[self_pidx] =
                                                ((i as u32 + 1u32) << 9u32) | PT_ICEI_U32;
                                            part_ty[nid] = PT_ICEI_U32;
                                            pmap[pidx] =
                                                ((nid as u32 + 1u32) << 9u32) | PT_ICEI_U32;
                                        }
                                        if t == PT_SNOW_U32
                                            && (rty == PT_WATR_U32 || rty == PT_DSTW_U32)
                                        {
                                            if roll.is_multiple_of(1000u32) {
                                                part_ty[i] = PT_ICEI_U32;
                                                pmap[self_pidx] =
                                                    ((i as u32 + 1u32) << 9u32) | PT_ICEI_U32;
                                                part_ty[nid] = PT_ICEI_U32;
                                                pmap[pidx] =
                                                    ((nid as u32 + 1u32) << 9u32) | PT_ICEI_U32;
                                            }
                                            if roll % 200u32 < 3u32 {
                                                part_ty[i] = PT_WATR_U32;
                                                pmap[self_pidx] =
                                                    ((i as u32 + 1u32) << 9u32) | PT_WATR_U32;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        rx += 1i32;
                    }
                    ry += 1i32;
                }

                // Pressure rules (`Element.cpp:210-224`), keyed on the type the
                // particle had when the pass started, like the reference's `t`.
                let ci = (((y as u32) / CELL) * xcells + ((x as u32) / CELL)) as usize;
                let pv = air_pv[ci];
                if t == PT_WTRV_U32 && pv > 4.0f32 {
                    part_ty[i] = PT_DSTW_U32;
                    pmap[self_pidx] = ((i as u32 + 1u32) << 9u32) | PT_DSTW_U32;
                }
                if t == PT_OIL_U32 && pv < -6.0f32 {
                    part_ty[i] = PT_GAS_U32;
                    pmap[self_pidx] = ((i as u32 + 1u32) << 9u32) | PT_GAS_U32;
                }
                if t == PT_GAS_U32 && pv > 6.0f32 {
                    part_ty[i] = PT_OIL_U32;
                    pmap[self_pidx] = ((i as u32 + 1u32) << 9u32) | PT_OIL_U32;
                }
                if t == PT_DESL_U32 && pv > 12.0f32 {
                    part_ty[i] = PT_FIRE_U32;
                    pmap[self_pidx] = ((i as u32 + 1u32) << 9u32) | PT_FIRE_U32;
                    part_life[i] =
                        FIRE_LIFE_BASE + (roll_life(frame_seed, i as u32) % FIRE_LIFE_SPAN) as i32;
                }
            }
        }
    }
}

/// Life roll for the legacy diesel ignition.
#[cube]
fn roll_life(frame_seed: u32, index: u32) -> u32 {
    gpu_hash(frame_seed ^ mix_index(index, PARTICLE_HASH_SALT) ^ 0x1B87_3593u32)
}

/// Runs the thermal conduction, phase-transition and `pmap`-refresh kernels.
///
/// `frame_seed` decorrelates the per-particle stochastic rolls between frames.
/// When `legacy_enable` is set (TPT's `legacy_enable`, i.e. `heat_enable ==
/// false`) the conduction/apply passes are skipped and the transition kernel
/// only runs the `LIFE` drain, wall sparks, detonation and pressure rules; the
/// engine calls [`step_legacy_update_gpu`] after the reaction pass for the rest.
pub fn step_thermal_and_transitions_gpu(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    frame_seed: u32,
    legacy_enable: bool,
) {
    let capacity = bufs.capacity;
    if capacity == 0 {
        return;
    }

    let xres = bufs.config.xres;
    let yres = bufs.config.yres;
    let xcells = bufs.config.xcells();
    let ycells = bufs.config.ycells();
    if xres == 0 || yres == 0 || xcells == 0 || ycells == 0 {
        return;
    }

    let elem_rows = registry
        .elements
        .len()
        .min(bufs.elem_props_host.len())
        .min(PT_NUM);
    if elem_rows == 0 {
        return;
    }
    let elem_rows_u32 = elem_rows as u32;
    let elem_len = elem_rows * GPU_ELEM_STRIDE;

    let slot_count = bufs.used_slot_bound() as u32;
    let pmap_len = (xres * yres) as usize;
    let cell_len = (xcells * ycells) as usize;
    let aheat_enable = u32::from(bufs.config.aheat_enable);
    let legacy = u32::from(legacy_enable);
    let blocks = slot_count.div_ceil(WORKGROUP_SIZE);

    let client = bufs.client.clone();

    // SAFETY: every `BufferArg` length matches the allocation it was created
    // with, and every index in the kernels is bounded by `slot_count`,
    // `elem_rows` and the explicit pixel-range guards.
    unsafe {
        if legacy == 0u32 {
            thermal_conduction_kernel::launch_unchecked(
                &client,
                CubeCount::Static(blocks, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_equil.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pmap_len),
                BufferArg::from_raw_parts(bufs.gpu_air_hv.clone(), cell_len),
                BufferArg::from_raw_parts(bufs.gpu_block_airh.clone(), cell_len),
                BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), elem_len),
                slot_count,
                elem_rows_u32,
                xres,
                yres,
                xcells,
                aheat_enable,
                frame_seed,
            );

            thermal_apply_kernel::launch_unchecked(
                &client,
                CubeCount::Static(blocks, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_equil.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pmap_len),
                BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), elem_len),
                slot_count,
                elem_rows_u32,
                xres,
                yres,
            );
        }

        phase_transition_kernel::launch_unchecked(
            &client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_equil.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pmap_len),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), cell_len),
            BufferArg::from_raw_parts(bufs.gpu_grav_fx.clone(), cell_len),
            BufferArg::from_raw_parts(bufs.gpu_grav_fy.clone(), cell_len),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), cell_len),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), cell_len),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), elem_len),
            slot_count,
            elem_rows_u32,
            xres,
            yres,
            xcells,
            ycells,
            legacy,
            bufs.config.gravity_mode as u32,
            bufs.config.custom_gravity_x,
            bufs.config.custom_gravity_y,
            frame_seed,
        );

        refresh_pmap_types_kernel::launch_unchecked(
            &client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pmap_len),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
            slot_count,
            xres,
            yres,
        );
    }
}

/// Runs `Element::legacyUpdate` for every live particle.
///
/// TPT calls it after the element update and before movement whenever
/// `legacy_enable` is set (`Simulation.cpp:2398-2400`).
pub fn step_legacy_update_gpu(bufs: &mut SimBuffers, registry: &ElementRegistry, frame_seed: u32) {
    let capacity = bufs.capacity;
    let xres = bufs.config.xres;
    let yres = bufs.config.yres;
    let xcells = bufs.config.xcells();
    if capacity == 0 || xres == 0 || yres == 0 || xcells == 0 {
        return;
    }
    let slot_count = bufs.used_slot_bound() as u32;
    if slot_count == 0 {
        return;
    }
    let _ = registry;
    let pmap_len = (xres * yres) as usize;
    let cell_len = (xcells * bufs.config.ycells()) as usize;
    let blocks = slot_count.div_ceil(WORKGROUP_SIZE);
    let client = bufs.client.clone();
    unsafe {
        legacy_update_kernel::launch_unchecked(
            &client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pmap_len),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), cell_len),
            slot_count,
            xres,
            yres,
            xcells,
            frame_seed,
        );
    }
    bufs.gpu_particles_dirty = true;
    bufs.gpu_dirty = true;
}
