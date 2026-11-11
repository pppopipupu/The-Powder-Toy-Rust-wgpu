//! HUD inspector probes and wall samples.
//!
//! The inspector runs every frame, so device probes are cached here instead of
//! stalling the render loop. `AppState`'s probe methods live in this module.

use std::cell::RefCell;
use std::time::{Duration, Instant};

use tpt_core::constants::CELL;
use tpt_core::particle::GpuParticle;
use tpt_sim::buffers::PixelProbe;

use crate::i18n::{t, t_fmt};

use super::AppState;
use super::property::{temperature_suffix, temperature_to_scale};

/// Shortest gap between two inspector device probes while the cursor moves.
const INSPECTOR_PROBE_MIN_INTERVAL: Duration = Duration::from_millis(40);
/// A probe older than this is always refreshed, even for a stationary cursor.
const INSPECTOR_PROBE_MAX_AGE: Duration = Duration::from_millis(150);

/// Last device probe rendered by the HUD inspector.
struct InspectorCache {
    x: i32,
    y: i32,
    cell: (u32, u32),
    sampled_at: Instant,
    probe: Option<PixelProbe>,
}

// The initialiser is already a `const` block; clippy 1.99 reports it anyway.
thread_local! {
    /// The inspector is drawn from this cache instead of reading the device
    /// once per frame; the simulation thread owns it for the whole UI frame.
    #[allow(clippy::missing_const_for_thread_local)]
    static INSPECTOR_CACHE: RefCell<Option<InspectorCache>> =
        const { RefCell::new(None::<InspectorCache>) };
}

/// Coarse `CELL x CELL` block of pixel `(x, y)`.
fn inspector_cell(x: i32, y: i32) -> (u32, u32) {
    (
        x.div_euclid(CELL as i32) as u32,
        y.div_euclid(CELL as i32) as u32,
    )
}

/// Drops the cached probe (canvas changed, save loaded, simulation cleared).
pub(super) fn clear_inspector_cache() {
    INSPECTOR_CACHE.with(|cache| *cache.borrow_mut() = None);
}

/// Wall report for the HUD inspector and the property window's sample row.
///
/// Mirrors `SimulationSample`'s wall fields (`Simulation::GetSample` in
/// `Editing.cpp` sets `WallType` from `bmap` and keeps the block's air state
/// next to it) and the way `GameView::Draw` renders them: the wall's name
/// followed by the block's pressure.
///
/// The reference stopped giving walls a temperature, so there is no wall
/// temperature to report; [`Self::air_temperature`] is the temperature of the
/// `CELL x CELL` block the wall occupies (`SimulationSample::AirTemperature`).
#[derive(Clone, Debug, PartialEq)]
pub struct WallSample {
    /// Wall id (`WL_*`).
    pub id: u8,
    /// Name from the wall table (`GameController::WallName`).
    pub name: String,
    /// Base colour of the wall type, `0x00RRGGBB`.
    pub colour: u32,
    /// Electrical state of the block (`wall_emap`); non-zero when the wall is
    /// carrying a charge.
    pub emap: u8,
    /// Air temperature of the block, in Kelvin.
    pub air_temperature: f32,
    /// Air pressure of the block.
    pub pressure: f32,
}

impl WallSample {
    /// One-line report, in the style of the HUD's sample readout.
    ///
    /// `temperature_scale` is a `TEMPSCALE_*`; the same scale the inspector uses
    /// for particle temperatures.
    pub fn summary(&self, temperature_scale: u32) -> String {
        let temperature = temperature_to_scale(self.air_temperature, temperature_scale);
        let mut text = t_fmt(
            "app.wall_summary",
            &[
                &self.name,
                &format!("{:06X}", self.colour & 0x00FF_FFFF),
                &format!("{:.1}", temperature),
                temperature_suffix(temperature_scale),
            ],
        );
        if self.emap != 0 {
            text.push_str(&t_fmt("app.wall_electrified", &[&self.emap.to_string()]));
        }
        text.push_str(&t_fmt(
            "app.wall_pressure",
            &[&format!("{:+.2}", self.pressure)],
        ));
        text
    }
}

impl AppState {
    /// Returns the particle occupying the given simulation pixel, if any.
    ///
    /// The HUD runs every frame, so the device probe is cached: it refreshes at
    /// most every 150 ms, and at most every 40 ms while the cursor crosses
    /// cells. A blocking probe per frame stalled the render loop.
    pub fn particle_at(&self, x: i32, y: i32) -> Option<GpuParticle> {
        self.probe_at(x, y).and_then(|probe| probe.particle)
    }

    /// Cached pixel probe (particle, wall, electrical state and pressure).
    ///
    /// Out-of-canvas coordinates return `None` without touching the device.
    pub fn probe_at(&self, x: i32, y: i32) -> Option<PixelProbe> {
        let cell = inspector_cell(x, y);
        INSPECTOR_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(cached) = cache.as_ref() {
                let age = cached.sampled_at.elapsed();
                let same_pixel = cached.x == x && cached.y == y;
                let same_cell = cached.cell == cell;
                let reuse = age < INSPECTOR_PROBE_MAX_AGE
                    && (same_pixel || (same_cell && age < INSPECTOR_PROBE_MIN_INTERVAL));
                if reuse {
                    return cached.probe.clone();
                }
            }
            let probe = self.engine.bufs.probe_pixel(x, y);
            *cache = Some(InspectorCache {
                x,
                y,
                cell,
                sampled_at: Instant::now(),
                probe: probe.clone(),
            });
            probe
        })
    }

    /// Air pressure of the cell containing pixel `(x, y)`, from the same cached
    /// probe the inspector uses (`0` outside the canvas).
    pub fn cell_pressure_at(&self, x: i32, y: i32) -> f32 {
        if let Some(probe) = self.probe_at(x, y) {
            return probe.pressure;
        }
        let config = &self.engine.bufs.config;
        if x < 0 || y < 0 || x as u32 >= config.xres || y as u32 >= config.yres {
            return 0.0;
        }
        let index =
            (y as usize / CELL as usize) * config.xcells() as usize + (x as usize / CELL as usize);
        self.engine.bufs.air_pv.get(index).copied().unwrap_or(0.0)
    }

    /// Returns the wall id occupying the given simulation pixel, if any.
    pub fn wall_at(&self, x: i32, y: i32) -> Option<u8> {
        self.engine.bufs.wall_at(x, y)
    }

    /// Wall occupying `(x, y)` together with the air state of its block.
    ///
    /// The wall counterpart of [`Self::particle_at`]: the sample tool reports
    /// walls this way (`SimulationSample::WallType`), and the property window
    /// shows the report when a sample lands on a wall instead of a particle.
    ///
    /// Read from the host shadow (`wall_bmap`, `wall_emap`, `air_hv`, `air_pv`),
    /// the same source [`Self::cell_pressure_at`] falls back to, so the HUD can
    /// call it every frame without a device read-back.
    pub fn wall_sample_at(&self, x: i32, y: i32) -> Option<WallSample> {
        let id = self.wall_at(x, y)?;
        let bufs = &self.engine.bufs;
        let cell = bufs.cell_index(x, y)?;
        let (name, colour) = self
            .engine
            .walls
            .iter()
            .find(|wall| wall.id == id)
            .map(|wall| (wall.name.clone(), wall.colour))
            .unwrap_or_else(|| (t("app.wall_fallback").to_string(), 0x00A0_A0A0));
        Some(WallSample {
            id,
            name,
            colour,
            emap: (bufs.wall_emap.get(cell).copied().unwrap_or(0) & 0xFF) as u8,
            air_temperature: bufs
                .air_hv
                .get(cell)
                .copied()
                .unwrap_or(bufs.config.ambient_air_temp),
            pressure: bufs.air_pv.get(cell).copied().unwrap_or(0.0),
        })
    }

    /// Resolves the effective heat display limits (min, max) in Kelvin.
    /// If explicit, returns the explicit values. If "auto", calculates the min/max
    /// of active particles inside the configured auto area.
    pub fn resolve_heat_display_limits(&self) -> (f32, f32) {
        let cfg = &self.engine.bufs.config;
        let mut min = cfg.heat_display_limit_min;
        let mut max = cfg.heat_display_limit_max;
        if min.is_none() || max.is_none() {
            let ax = cfg.heat_display_auto_x;
            let ay = cfg.heat_display_auto_y;
            let aw = cfg.heat_display_auto_w;
            let ah = cfg.heat_display_auto_h;
            let mut auto_min = 9999.0f32;
            let mut auto_max = 0.0f32;
            let mut found = false;
            let bufs = &self.engine.bufs;
            // Live slots are sparse, so the scan bound is the slot watermark,
            // not the live count.
            let active = bufs.used_slot_bound();
            for i in 0..active.min(bufs.part_ty.len()) {
                if bufs.part_ty[i] != 0 {
                    let px = bufs.part_x[i] as i32;
                    let py = bufs.part_y[i] as i32;
                    if px >= ax && px < ax + aw && py >= ay && py < ay + ah {
                        let temp = bufs.part_temp[i];
                        if temp < auto_min {
                            auto_min = temp;
                        }
                        if temp > auto_max {
                            auto_max = temp;
                        }
                        found = true;
                    }
                }
            }
            if !found {
                auto_min = 0.0;
                auto_max = 9999.0;
            }
            if min.is_none() {
                min = Some(auto_min);
            }
            if max.is_none() {
                max = Some(auto_max);
            }
        }
        let eff_min = min.unwrap_or(0.0).clamp(0.0, 9999.0);
        let eff_max = max.unwrap_or(9999.0).clamp(0.0, 9999.0).max(eff_min + 1.0);
        (eff_min, eff_max)
    }
}
