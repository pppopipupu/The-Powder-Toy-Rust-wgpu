//! Dynamic heat-display range (`Renderer::AdjustHdispLimit`,
//! `Renderer.cpp:1444-1512`).
//!
//! TPT re-derives both ends of the heat colour scale from the live simulation
//! on every frame: it scans the particles (skipping elements whose
//! `HeatConduct` is zero) and, when the ambient-heat overlay is on, the `hv`
//! grid, then eases the sampled extremes towards the previous frame's values
//! so a single hot spark cannot flash the whole canvas
//! (`Renderer.cpp:1488-1491`).
//!
//! The port's particle and air arrays live on the device, so the scan is a
//! CubeCL reduction: one thread per particle slot and one per air cell update
//! two `Atomic<u32>` slots holding the sampled temperatures as raw bit
//! patterns, and the host reads the four-word result back (the same pattern
//! as [`crate::live_count`]). Only the auto mode pays for the read-back.
//!
//! Temperatures sit in `MIN_TEMP..=MAX_TEMP`: `Air.cpp:186-187` clamps `hv`
//! and the particle kernels clamp `temp`, so the IEEE-754 bit pattern of a
//! non-negative `f32` orders like a `u32` and plain atomic
//! `fetch_min`/`fetch_max` implement the reference's `std::min`/`std::max`.
//! The easing itself is host-side; it is a straight port of the reference and
//! is unit-tested against a synthetic field.

use core::cell::{Cell, RefCell};
use core::mem::offset_of;

use cubecl::prelude::*;

use tpt_core::constants::*;
use tpt_core::elements::GpuElementProps;

use crate::buffers::SimBuffers;

use super::{CachedBuffer, cached_buffer};

/// `PT_NONE` is the free-slot marker; mirrored as a literal because a `#[cube]`
/// kernel cannot read host constants.
const C_PT_NONE: u32 = 0;
const _: () = assert!(PT_NONE == 0);

/// `PT_NUM` mirrored for the `ty < PT_NUM` bounds check of the reference.
const C_PT_NUM: u32 = PT_NUM as u32;

/// `CELL` mirrored as a `u32` for the cell-to-pixel arithmetic.
const C_CELL: u32 = CELL;

/// Words per element in the packed table (`render_kernel::ELEM_WORDS`), mirrored
/// because a `#[cube]` kernel cannot read a host constant through a path.
const C_ELEM_WORDS: u32 = super::ELEM_WORDS;

/// `ElementGraphics.h:53`: `DISPLAY_AIR` is the union of every air overlay.
const DISPLAY_AIR: u32 = DISPLAY_AIRC | DISPLAY_AIRP | DISPLAY_AIRV | DISPLAY_AIRH | DISPLAY_AIRW;
const _: () = assert!(DISPLAY_AIR == 0x8F);

/// Word offset of `GpuElementProps::heat_conduct` in the packed element table.
const ELEM_HEAT_CONDUCT: u32 = (offset_of!(GpuElementProps, heat_conduct) / 4) as u32;

/// Threads per reduction cube, matching [`crate::live_count::CUBE_DIM`].
const CUBE_DIM: u32 = 256;

/// Words of the reduction buffer: `[min_key, max_key, valid, pad]`.
const RANGE_WORDS: usize = 4;

/// `Renderer::AdjustHdispLimit`'s shrink factor (`Renderer.cpp:1489,1491`):
/// each bound moves 5 % of the remaining gap per frame, while an expansion
/// applies immediately.
const SHRINK_RATIO: f32 = 0.05;

/// Previous frame's resolved limits (`RendererFrame.h:13-15`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct HdispStats {
    /// Lower bound of the heat colour scale, in Kelvin.
    min: f32,
    /// Upper bound of the heat colour scale, in Kelvin.
    max: f32,
    /// Whether the auto scan found at least one sample in the area.
    valid: bool,
}

thread_local! {
    /// Device copy of the four-word reduction result.
    #[allow(clippy::missing_const_for_thread_local)]
    static HEAT_RANGE_GPU: RefCell<Option<CachedBuffer>> =
        const { RefCell::new(None::<CachedBuffer>) };
    /// Last resolved bounds, the port of `Renderer::stats.hdispLimit*`.
    ///
    /// The port has no `Renderer` object (the passes are free functions), and
    /// the device handles already live in thread-locals in this module, so the
    /// smoothing state follows the same rule. The UI thread is the only
    /// renderer caller.
    // The initialiser is already a `const` block; clippy 1.99 still reports
    // the statics, so the lint is silenced per static (see `render_kernel/mod.rs`).
    #[allow(clippy::missing_const_for_thread_local)]
    static HDISP_STATS: Cell<HdispStats> = const { Cell::new(HdispStats {
        // `RendererFrame.h:13-15`: both limits and the valid flag start at 0.
        min: 0.0,
        max: 0.0,
        valid: false,
    }) };
}

/// The last limits the renderer actually used, for the Render window.
///
/// Mirrors `GameView::NotifyRendererStatsChanged` reading
/// `rendererStats.hdispLimitMin/Max` (`GameView.cpp:2563-2569`): the third
/// value is `hdispLimitValid`, i.e. whether the auto scan found a sample.
pub fn current_hdisp_limits() -> (f32, f32, bool) {
    HDISP_STATS.with(|cell| {
        let stats = cell.get();
        (stats.min, stats.max, stats.valid)
    })
}

/// Resolve the heat-display limits for one frame.
///
/// `want_min`/`want_max` are the configuration's `HdispLimit` variants
/// (`None` is `HdispLimitAuto`). The sampling area and the ambient-heat gate
/// are read from the same configuration.
pub(crate) fn resolve_hdisp_limits(bufs: &SimBuffers) -> (f32, f32) {
    let cfg = &bufs.config;
    let want_min = cfg.heat_display_limit_min;
    let want_max = cfg.heat_display_limit_max;

    // `autoHdispLimitMin/Max` start at the extremes; a scan that finds nothing
    // leaves them there, exactly like the reference.
    let mut auto = (MAX_TEMP, MIN_TEMP);
    let mut valid = false;
    if want_min.is_none() || want_max.is_none() {
        // `colorMode & COLOUR_HEAT`; the element table is needed for the
        // `HeatConduct` filter, so an unpopulated table disables only the
        // particle half of the scan.
        let heat = cfg.colour_mode & COLOUR_HEAT != 0 && !bufs.elem_props_host.is_empty();
        let air = cfg.aheat_enable
            && cfg.display_mode & DISPLAY_AIR != 0
            && cfg.display_mode & DISPLAY_AIRH != 0;
        if (heat || air)
            && let Some((min, max)) = sample_range(bufs, heat, air)
        {
            auto = (min, max);
            valid = true;
        }
    }

    let prev = HDISP_STATS.with(Cell::get);
    let (min, max) = resolve_bounds(auto, want_min, want_max, (prev.min, prev.max));
    HDISP_STATS.with(|cell| {
        cell.set(HdispStats { min, max, valid });
    });
    (min, max)
}

/// `Renderer::AdjustHdispLimit`'s bound arithmetic (`Renderer.cpp:1488-1512`).
///
/// Split out from the device scan so the shrink/expand rules can be tested
/// against a synthetic field without a GPU.
fn resolve_bounds(
    auto: (f32, f32),
    want_min: Option<f32>,
    want_max: Option<f32>,
    prev: (f32, f32),
) -> (f32, f32) {
    let (mut min, mut max) = if want_min.is_none() || want_max.is_none() {
        smooth_auto_bounds(auto.0, auto.1, prev.0, prev.1)
    } else {
        // Both bounds are explicit: the reference still runs the smoother on
        // its local extremes and then overwrites them; the resolved values are
        // exactly the explicit pair.
        auto
    };
    if let Some(value) = want_min {
        min = value;
    }
    if let Some(value) = want_max {
        max = value;
    }
    // `if (std::isnan(...))` guards of `Renderer.cpp:1508-1509`.
    if min.is_nan() {
        min = MIN_TEMP;
    }
    if max.is_nan() {
        max = MAX_TEMP;
    }
    max = max.clamp(MIN_TEMP, MAX_TEMP);
    min = min.clamp(MIN_TEMP, max);
    (min, max)
}

/// One frame of `Renderer.cpp:1488-1495`.
///
/// `auto` is the scanned pair, `prev` the pair the previous frame displayed.
/// The maximum and minimum are smoothed independently: an expansion is taken
/// immediately, a contraction only covers 5 % of the gap. The final block
/// enforces the reference's one-degree gap so a flat field cannot divide by
/// zero in `heatTableAt`.
fn smooth_auto_bounds(auto_min: f32, auto_max: f32, prev_min: f32, prev_max: f32) -> (f32, f32) {
    // float maxGap = stats.hdispLimitMax - autoHdispLimitMax;
    // autoHdispLimitMax = std::max(autoHdispLimitMax, stats.hdispLimitMax - maxGap * 0.05f);
    let max_gap = prev_max - auto_max;
    let auto_max = auto_max.max(prev_max - max_gap * SHRINK_RATIO);
    // float minGap = autoHdispLimitMin - stats.hdispLimitMin;
    // autoHdispLimitMin = std::min(autoHdispLimitMin, stats.hdispLimitMin + minGap * 0.05f);
    let min_gap = auto_min - prev_min;
    let auto_min = auto_min.min(prev_min + min_gap * SHRINK_RATIO);
    // Ensure a 1C gap ... (`Renderer.cpp:1494-1495`).
    let auto_max = MAX_TEMP.min(auto_max.max(auto_min + 1.0));
    let auto_min = MIN_TEMP.max(auto_min.min(auto_max - 1.0));
    (auto_min, auto_max)
}

/// Zero the reduction slot.
#[cube(launch_unchecked)]
fn reset_heat_range_kernel(range: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0usize {
        range[0].store(u32::MAX);
        range[1].store(0u32);
        range[2].store(0u32);
    }
}

/// `Renderer.cpp:1457-1475`: scan the particle field.
#[cube(launch_unchecked)]
fn heat_range_particles_kernel(
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    part_temp: &[f32],
    elem_props: &[u32],
    range: &mut [Atomic<u32>],
    bound: u32,
    elem_rows: u32,
    area_x: i32,
    area_y: i32,
    area_w: i32,
    area_h: i32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < bound {
        let ty = part_ty[idx as usize];
        if ty != C_PT_NONE && ty < C_PT_NUM && ty < elem_rows {
            // `if (!sd.elements[t].HeatConduct) continue;`
            let heat_conduct = elem_props[(ty * C_ELEM_WORDS + ELEM_HEAT_CONDUCT) as usize];
            if heat_conduct != 0 {
                // `int(sim->parts[i].x + 0.5f)` truncates towards zero.
                let px = (part_x[idx as usize] + 0.5f32) as i32;
                let py = (part_y[idx as usize] + 0.5f32) as i32;
                if px >= area_x && px < area_x + area_w && py >= area_y && py < area_y + area_h {
                    let key = u32::reinterpret(part_temp[idx as usize]);
                    range[0].fetch_min(key);
                    range[1].fetch_max(key);
                    range[2].store(1u32);
                }
            }
        }
    }
}

/// `Renderer.cpp:1478-1485`: scan the ambient-heat grid.
#[cube(launch_unchecked)]
fn heat_range_air_kernel(
    air_hv: &[f32],
    range: &mut [Atomic<u32>],
    cells: u32,
    xcells: u32,
    area_x: i32,
    area_y: i32,
    area_w: i32,
    area_h: i32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < cells {
        // `for (auto p : CELLS.OriginRect()) visit(p * CELL, hv[p.Y][p.X]);`
        let cx = idx % xcells;
        let cy = idx / xcells;
        let px = (cx * C_CELL) as i32;
        let py = (cy * C_CELL) as i32;
        if px >= area_x && px < area_x + area_w && py >= area_y && py < area_y + area_h {
            let key = u32::reinterpret(air_hv[idx as usize]);
            range[0].fetch_min(key);
            range[1].fetch_max(key);
            range[2].store(1u32);
        }
    }
}

/// Run the device reduction and return the sampled `(min, max)` in Kelvin.
///
/// `None` means the read-back failed or no sample landed in the area; the
/// caller keeps the reference's `MAX_TEMP`/`MIN_TEMP` extremes in that case.
fn sample_range(bufs: &SimBuffers, heat: bool, air: bool) -> Option<(f32, f32)> {
    let range = cached_buffer(&HEAT_RANGE_GPU, &bufs.client, RANGE_WORDS, None);
    let cfg = &bufs.config;
    unsafe {
        reset_heat_range_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(range.clone(), RANGE_WORDS),
        );
        if heat {
            let bound = bufs.used_slot_bound().min(bufs.capacity) as u32;
            if bound > 0 {
                let cubes = bound.div_ceil(CUBE_DIM).max(1);
                heat_range_particles_kernel::launch_unchecked(
                    &bufs.client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(
                        bufs.gpu_elem_props.clone(),
                        bufs.elem_props_host.len() * C_ELEM_WORDS as usize,
                    ),
                    BufferArg::from_raw_parts(range.clone(), RANGE_WORDS),
                    bound,
                    bufs.elem_props_host.len() as u32,
                    cfg.heat_display_auto_x,
                    cfg.heat_display_auto_y,
                    cfg.heat_display_auto_w,
                    cfg.heat_display_auto_h,
                );
            }
        }
        if air {
            let cells = cfg.num_cells();
            let xcells = cfg.xcells();
            if cells > 0 && xcells > 0 {
                let cubes = (cells as u32).div_ceil(CUBE_DIM).max(1);
                heat_range_air_kernel::launch_unchecked(
                    &bufs.client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    BufferArg::from_raw_parts(bufs.gpu_air_hv.clone(), cells),
                    BufferArg::from_raw_parts(range.clone(), RANGE_WORDS),
                    cells as u32,
                    xcells,
                    cfg.heat_display_auto_x,
                    cfg.heat_display_auto_y,
                    cfg.heat_display_auto_w,
                    cfg.heat_display_auto_h,
                );
            }
        }
    }

    let readback = bufs.client.read(vec![range]).ok()?;
    let raw: &[u8] = readback.first()?;
    let words = raw.get(..RANGE_WORDS * 4)?;
    let mut keys = [0u32; RANGE_WORDS];
    for (slot, chunk) in keys.iter_mut().zip(words.as_chunks::<4>().0) {
        *slot = u32::from_le_bytes(*chunk);
    }
    let (min_key, max_key, valid) = (keys[0], keys[1], keys[2]);
    if valid == 0 {
        return None;
    }
    let min = f32::from_bits(min_key);
    let max = f32::from_bits(max_key);
    Some((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors the predicate of [`heat_range_particles_kernel`]: only slots in
    /// the area whose element has a non-zero `HeatConduct` contribute.
    fn scan_field(
        field: &[(i32, i32, f32, bool)],
        area: (i32, i32, i32, i32),
    ) -> Option<(f32, f32)> {
        let (ax, ay, aw, ah) = area;
        let mut min = MAX_TEMP;
        let mut max = MIN_TEMP;
        let mut found = false;
        for &(x, y, temp, heat_conduct) in field {
            if !heat_conduct {
                continue;
            }
            if x >= ax && x < ax + aw && y >= ay && y < ay + ah {
                min = min.min(temp);
                max = max.max(temp);
                found = true;
            }
        }
        found.then_some((min, max))
    }

    /// A synthetic field with a cold and a hot outlier, a non-conductive
    /// element that must be ignored, and a pixel outside the auto area.
    fn synthetic_field() -> Vec<(i32, i32, f32, bool)> {
        vec![
            (10, 10, 300.0, true),
            (11, 10, 320.0, true),
            (12, 10, 305.0, true),
            // `WALL`-like element: HeatConduct 0, must not move the bounds.
            (13, 10, 9999.0, false),
            // Outside the sampled rectangle.
            (40, 40, -100.0, true),
        ]
    }

    #[test]
    fn auto_range_ignores_non_conductive_and_out_of_area_pixels() {
        let field = synthetic_field();
        let raw = scan_field(&field, (0, 0, 20, 20)).expect("the area holds samples");
        assert_eq!(raw, (300.0, 320.0));

        // A field whose bounds expand immediately from the previous frame.
        let (min, max) = resolve_bounds(raw, None, None, (300.0, 300.0));
        assert_eq!((min, max), (300.0, 320.0));
    }

    #[test]
    fn auto_range_contracts_five_percent_of_the_gap_per_frame() {
        // Previous frame showed 100..300, the new field is 90..200.
        let (min, max) = smooth_auto_bounds(90.0, 200.0, 100.0, 300.0);
        // maxGap = 300 - 200 = 100, so the upper bound shrinks to 300 - 5.
        assert!((max - 295.0).abs() < 1e-4, "max was {max}");
        // minGap = 90 - 100 = -10, expansion applies immediately.
        assert!((min - 90.0).abs() < 1e-4, "min was {min}");
    }

    #[test]
    fn auto_range_keeps_a_one_degree_gap() {
        let (min, max) = smooth_auto_bounds(500.0, 500.0, 500.0, 500.0);
        assert_eq!(min, 500.0);
        assert_eq!(max, 501.0);
    }

    #[test]
    fn auto_range_clamps_into_the_temperature_limits() {
        let (min, max) = resolve_bounds((MAX_TEMP, MAX_TEMP), None, None, (MAX_TEMP, MAX_TEMP));
        assert_eq!(max, MAX_TEMP);
        assert_eq!(min, MAX_TEMP - 1.0);

        let (min, max) = resolve_bounds((MIN_TEMP, MIN_TEMP), None, None, (MIN_TEMP, MIN_TEMP));
        assert_eq!(min, MIN_TEMP);
        assert_eq!(max, 1.0);
    }

    #[test]
    fn explicit_bounds_override_the_auto_scan() {
        let (min, max) = resolve_bounds((300.0, 320.0), Some(50.0), Some(900.0), (0.0, 9999.0));
        assert_eq!((min, max), (50.0, 900.0));
    }

    #[test]
    fn explicit_nan_falls_back_to_the_reference_defaults() {
        let (min, max) = resolve_bounds(
            (300.0, 320.0),
            Some(f32::NAN),
            Some(f32::NAN),
            (0.0, 9999.0),
        );
        assert_eq!((min, max), (MIN_TEMP, MAX_TEMP));
    }

    #[test]
    fn non_negative_temperature_bits_order_like_the_values() {
        // The device reduction relies on this for its atomic min/max.
        assert!(20.0f32.to_bits() < 300.0f32.to_bits());
        assert!(MIN_TEMP.to_bits() < MAX_TEMP.to_bits());
    }
}
