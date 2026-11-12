//! HUD inspector probes and wall samples.
//!
//! The inspector runs every frame, so device probes are cached here instead of
//! stalling the render loop. `AppState`'s probe methods live in this module.

use std::cell::RefCell;
use std::time::{Duration, Instant};

use tpt_core::constants::CELL;
use tpt_core::particle::GpuParticle;
use tpt_sim::buffers::PixelProbe;
use tpt_sim::live_count::PendingRead;

use crate::i18n::{t, t_fmt};

use super::AppState;
use super::property::{temperature_suffix, temperature_to_scale};

/// Shortest gap between two inspector device probes, whatever the pointer does.
///
/// A probe is a blocking `Client::read` (two of them when a particle occupies
/// the pixel), i.e. a full queue drain. The top-bar inspector asks for the
/// pixel under the cursor on every frame, and a cursor crossing `CELL`-sized
/// cells asks for a *different* pixel on nearly every frame: throttling only
/// same-cell refreshes let a fast drag issue one device sync per frame.
const INSPECTOR_PROBE_MIN_INTERVAL: Duration = Duration::from_millis(40);
/// A probe older than this is always refreshed, even for a stationary cursor.
const INSPECTOR_PROBE_MAX_AGE: Duration = Duration::from_millis(150);

/// Last device probe rendered by the HUD inspector.
struct InspectorCache {
    x: i32,
    y: i32,
    sampled_at: Instant,
    probe: Option<PixelProbe>,
}

struct PendingProbe {
    x: i32,
    y: i32,
    handle: PendingRead<Option<PixelProbe>>,
}

/// Whether the cached probe may stand in for a request on another pixel.
///
/// The interval floor applies to every request, not just the same cell: the
/// top bar asks for a new pixel as soon as the cursor crosses a `CELL` block,
/// so a per-cell rule still allowed one blocking device read per frame while
/// the mouse moved. A sample up to [`INSPECTOR_PROBE_MIN_INTERVAL`] old is
/// invisible next to the 150 ms stationary refresh.
fn probe_cache_reuse(age: Duration, same_pixel: bool) -> bool {
    age < INSPECTOR_PROBE_MIN_INTERVAL || (same_pixel && age < INSPECTOR_PROBE_MAX_AGE)
}

// The initialiser is already a `const` block; clippy 1.99 reports it anyway.
thread_local! {
    /// The inspector is drawn from this cache instead of reading the device
    /// once per frame; the simulation thread owns it for the whole UI frame.
    #[allow(clippy::missing_const_for_thread_local)]
    static INSPECTOR_CACHE: RefCell<Option<InspectorCache>> =
        const { RefCell::new(None::<InspectorCache>) };
    #[allow(clippy::missing_const_for_thread_local)]
    static PENDING_PROBE: RefCell<Option<PendingProbe>> =
        const { RefCell::new(None::<PendingProbe>) };
}

/// Drops the cached probe (canvas changed, save loaded, simulation cleared).
pub(super) fn clear_inspector_cache() {
    INSPECTOR_CACHE.with(|cache| *cache.borrow_mut() = None);
    PENDING_PROBE.with(|pending| *pending.borrow_mut() = None);
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
    /// most every 40 ms, and at most every 150 ms for a stationary pixel. A
    /// blocking probe per frame stalled the render loop.
    pub fn particle_at(&self, x: i32, y: i32) -> Option<GpuParticle> {
        self.probe_at(x, y).and_then(|probe| probe.particle)
    }

    /// Pixel probe that must match the device *now*, for click actions.
    ///
    /// The middle-click sampler, the `Sample` tool and the shift-click property
    /// seeding all act on the pixel the player clicked, so they cannot reuse a
    /// probe that was taken for a neighbouring pixel up to
    /// [`INSPECTOR_PROBE_MIN_INTERVAL`] ago. Storing the fresh result keeps the
    /// per-frame HUD cache coherent with the click.
    pub fn probe_at_now(&self, x: i32, y: i32) -> Option<PixelProbe> {
        let probe = self.engine.bufs.probe_pixel(x, y);
        PENDING_PROBE.with(|pending| *pending.borrow_mut() = None);
        INSPECTOR_CACHE.with(|cache| {
            *cache.borrow_mut() = Some(InspectorCache {
                x,
                y,
                sampled_at: Instant::now(),
                probe: probe.clone(),
            });
        });
        probe
    }

    /// Uncached counterpart of [`Self::particle_at`].
    pub fn particle_at_now(&self, x: i32, y: i32) -> Option<GpuParticle> {
        self.probe_at_now(x, y).and_then(|probe| probe.particle)
    }

    /// Cached non-blocking pixel probe (particle, wall, electrical state and
    /// pressure).
    ///
    /// Refreshes asynchronously through `probe_pixel_async` so moving or
    /// dragging across a large canvas never stalls the render loop waiting for
    /// the GPU queue to drain. Out-of-canvas coordinates return `None` without
    /// touching the device.
    pub fn probe_at(&self, x: i32, y: i32) -> Option<PixelProbe> {
        let mut completed: Option<InspectorCache> = None;
        let mut in_flight = false;
        PENDING_PROBE.with(|pending| {
            let mut pending = pending.borrow_mut();
            if let Some(job) = pending.as_ref() {
                if let Some(probe) = job.handle.try_take() {
                    completed = Some(InspectorCache {
                        x: job.x,
                        y: job.y,
                        sampled_at: Instant::now(),
                        probe,
                    });
                    *pending = None;
                } else if !job.handle.is_done() {
                    in_flight = true;
                } else {
                    *pending = None;
                }
            }
        });

        INSPECTOR_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(fresh) = completed {
                *cache = Some(fresh);
            }
            let needs_refresh = match cache.as_ref() {
                Some(cached) => {
                    !probe_cache_reuse(cached.sampled_at.elapsed(), cached.x == x && cached.y == y)
                }
                None => true,
            };
            if needs_refresh
                && !in_flight
                && let Some(handle) = self.engine.bufs.probe_pixel_async(x, y)
            {
                if let Some(probe) = handle.try_take() {
                    *cache = Some(InspectorCache {
                        x,
                        y,
                        sampled_at: Instant::now(),
                        probe,
                    });
                } else {
                    PENDING_PROBE.with(|pending| {
                        *pending.borrow_mut() = Some(PendingProbe { x, y, handle });
                    });
                }
            }
            cache.as_ref().and_then(|cached| cached.probe.clone())
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cursor crossing cells must not issue a device probe per frame: the
    /// floor applies to a *different* pixel as well.
    #[test]
    fn probe_cache_throttles_a_moving_cursor() {
        assert!(probe_cache_reuse(Duration::from_millis(1), false));
        assert!(probe_cache_reuse(Duration::from_millis(39), false));
        assert!(!probe_cache_reuse(Duration::from_millis(41), false));
        assert!(!probe_cache_reuse(Duration::from_millis(149), false));
    }

    /// A stationary pixel still refreshes so the readout follows the sim.
    #[test]
    fn probe_cache_refreshes_a_stationary_pixel() {
        assert!(probe_cache_reuse(Duration::from_millis(41), true));
        assert!(probe_cache_reuse(Duration::from_millis(149), true));
        assert!(!probe_cache_reuse(Duration::from_millis(151), true));
    }
}
