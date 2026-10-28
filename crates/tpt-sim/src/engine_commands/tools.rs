//! Simulation tool application commands (heat, air, wind, mix, cyclone, gravity).

use tpt_core::commands::SimToolKind;
use tpt_core::constants::*;

use crate::SimulationEngine;
use crate::buffers::CreationSource;

impl SimulationEngine {
    /// Apply a simulation tool (heat, cool, air, vacuum, wind, mix, cyclone, ambient).
    ///
    /// One call is one `SimTool::Perform` pass over the brush footprint: the
    /// reference calls `Perform` once per pixel too (`SimTool.cpp:20-30`
    /// iterates the brush and calls `CallPerform` for every offset), so every
    /// per-pixel body below runs per point, not per stroke.
    ///
    /// The edits are applied by the device kernels in [`crate::tool_kernel`]
    /// against the live simulation. The host shadow is a snapshot from the last
    /// read-back, so editing it and re-uploading rewound the device: `HEAT`
    /// pinned the temperature at `snapshot + 2 K`, and holding `AIR` overwrote
    /// the evolved air field on every tick.
    ///
    /// `dir` is the stroke step in simulation pixels (`position2 - position1`,
    /// only the wind tool uses it) and `center` is the brush centre TPT passes
    /// as `brushX`/`brushY` (`SimTool.cpp:16`; the cyclone tool rotates around
    /// it).
    pub(super) fn apply_sim_tool(
        &mut self,
        points: &[(i32, i32)],
        tool: SimToolKind,
        strength: f32,
        dir: (f32, f32),
        center: (i32, i32),
        dragging: bool,
    ) {
        if matches!(tool, SimToolKind::PosGrav | SimToolKind::NegGrav) {
            self.apply_gravity_tool(points, tool == SimToolKind::PosGrav, strength);
            return;
        }

        let xres = self.bufs.config.xres as i32;
        let yres = self.bufs.config.yres as i32;
        let inside: Vec<(i32, i32)> = points
            .iter()
            .copied()
            .filter(|&(x, y)| x >= 0 && y >= 0 && x < xres && y < yres)
            .collect();
        if inside.is_empty() {
            return;
        }

        match tool {
            SimToolKind::Heat | SimToolKind::Cool => {
                // `HEAT.cpp:18-21`: pumps move twenty times slower; the clamp
                // is `0..=MAX_TEMP` (`HEAT.cpp:23-26`).
                let sign = if matches!(tool, SimToolKind::Heat) {
                    1.0
                } else {
                    -1.0
                };
                crate::tool_kernel::heat_cool(
                    &mut self.bufs,
                    &inside,
                    sign * strength * 2.0,
                    sign * strength * 0.1,
                );
            }
            SimToolKind::Air | SimToolKind::Vac => {
                // `AIR.cpp:17-23`: `pv += ±strength * 0.05` per brush pixel.
                let sign = if matches!(tool, SimToolKind::Air) {
                    1.0
                } else {
                    -1.0
                };
                let cells = self.cell_counts(&inside);
                crate::tool_kernel::air_tool(
                    &mut self.bufs,
                    &cells,
                    sign * strength * 0.05,
                    0.0,
                    0.0,
                    0.0,
                    true,
                );
            }
            SimToolKind::Wind => {
                // `WIND.cpp:20-22`: `(position2 - position1) * strength`, with
                // `0.01` on the held drag path (`DrawLine(..., true)`) and
                // `0.002` for `PerformDrag`, the line/rect gesture.
                let scale = if dragging { 0.01 } else { 0.002 } * strength;
                let cells = self.cell_counts(&inside);
                // The reference add is unbounded (`WIND.cpp:24-28`).
                crate::tool_kernel::air_tool(
                    &mut self.bufs,
                    &cells,
                    0.0,
                    0.0,
                    dir.0 * scale,
                    dir.1 * scale,
                    false,
                );
            }
            SimToolKind::Cycl => {
                // `CYCL.cpp:24-51`: one hit per cell, around the brush centre.
                let cells = self.cell_aligned(&inside);
                crate::tool_kernel::cyclone(&mut self.bufs, &cells, center, strength / 16.0);
            }
            SimToolKind::AmbHeatPlus | SimToolKind::AmbHeatMinus => {
                // `AMBP.cpp:16-24` / `AMBM.cpp:16-24`: no effect unless the
                // ambient heat simulation is enabled.
                if self.bufs.config.aheat_enable {
                    let sign = if matches!(tool, SimToolKind::AmbHeatPlus) {
                        1.0
                    } else {
                        -1.0
                    };
                    let cells = self.cell_counts(&inside);
                    crate::tool_kernel::air_tool(
                        &mut self.bufs,
                        &cells,
                        0.0,
                        sign * strength * 2.0,
                        0.0,
                        0.0,
                        true,
                    );
                }
            }
            SimToolKind::Mix => {
                // `MIX.cpp:28-31`: `(int)(pow(strength, .5f) * 10)`.
                let distance = (strength.max(0.0).sqrt() * 10.0) as i32;
                if distance <= 0 {
                    return;
                }
                // Three rolls per pixel in brush order (`MIX.cpp:25,36-38`).
                let mut rolls = Vec::with_capacity(inside.len() * 3);
                for _ in 0..inside.len() * 3 {
                    rolls.push(rng_next(&mut self.rng_state) as u32);
                }
                crate::tool_kernel::mix(&mut self.bufs, &inside, &rolls, distance);
            }
            // Handled above.
            SimToolKind::PosGrav | SimToolKind::NegGrav => {}
        }
    }

    /// Deduplicates the brush footprint into `(air cell, pixel count)` pairs.
    ///
    /// Every tool page adds once per brush pixel (`for (ui::Point off : brush)`),
    /// and a brush stamp covers several pixels of the same `CELL`-sized air cell,
    /// so the device kernel needs the multiplicity instead of one write per pixel
    /// (concurrent float read-modify-write would lose updates).
    fn cell_counts(&self, points: &[(i32, i32)]) -> Vec<(u32, u32)> {
        let xcells = self.bufs.config.xcells();
        let mut index: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut cells: Vec<(u32, u32)> = Vec::new();
        for &(x, y) in points {
            let cell = (y as u32 / CELL) * xcells + (x as u32 / CELL);
            let slot = *index.entry(cell).or_insert_with(|| {
                cells.push((cell, 0));
                cells.len() - 1
            });
            cells[slot].1 += 1;
        }
        cells
    }

    /// Cells whose top-left pixel is under the brush, in brush order.
    ///
    /// `CYCL.cpp:24-27` only fires on pixels that are cell corners, so this is
    /// the one-hit-per-cell list the cyclone kernel expects.
    fn cell_aligned(&self, points: &[(i32, i32)]) -> Vec<u32> {
        let xcells = self.bufs.config.xcells();
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut cells = Vec::new();
        for &(x, y) in points {
            if x % CELL as i32 != 0 || y % CELL as i32 != 0 {
                continue;
            }
            let cell = (y as u32 / CELL) * xcells + (x as u32 / CELL);
            if seen.insert(cell) {
                cells.push(cell);
            }
        }
        cells
    }

    /// Write the positive (`PGRV`) or negative (`NGRV`) gravity well into the
    /// per-cell mass grid, exactly like the reference tools.
    ///
    /// `PGRV.cpp` does `sim->gravIn.mass[Vec2{x, y} / CELL] = strength * 5.0f`
    /// (`NGRV.cpp` negates it). TPT clears `gravIn.mass` right after each
    /// gravity dispatch, so a painted well affects exactly one solve; the port
    /// keeps the writes in [`crate::SimBuffers::grav_tool_mass`], which
    /// `step_gravity_gpu` uploads and folds into the device accumulator before
    /// the FFT runs.
    pub(super) fn apply_gravity_tool(
        &mut self,
        points: &[(i32, i32)],
        attract: bool,
        strength: f32,
    ) {
        if points.is_empty() {
            return;
        }
        let xres = self.bufs.config.xres as i32;
        let yres = self.bufs.config.yres as i32;
        let xcells = self.bufs.config.xcells() as i32;
        let ycells = self.bufs.config.ycells() as i32;
        let mass = tool_gravity_mass(strength, attract);
        let mut wrote = false;
        for &(x, y) in points {
            let Some(cell) = gravity_tool_cell(x, y, xres, yres, xcells, ycells) else {
                continue;
            };
            self.bufs.grav_tool_mass[cell] = mass;
            wrote = true;
        }
        if wrote {
            self.bufs.grav_tool_mass_dirty = true;
        }
    }

    /// Spark the conductor under a pixel (`create_part(-1, x, y, PT_SPRK)`,
    /// `Simulation.cpp:1851-1878`), used by a `{b|..}` button sign
    /// (`GameController.cpp:583-585`).
    ///
    /// Nothing is created on an empty, walled, non-conductive or already
    /// charged pixel: the reference's spark branch takes over the occupant
    /// instead of allocating a new particle, so this routes through
    /// [`crate::SimBuffers::create_part`] rather than `spawn_particle`.
    pub(super) fn spark_at(&mut self, x: i32, y: i32) {
        if self.bufs.gpu_dirty {
            self.bufs.read_map_rect(x, y, x, y);
        }
        let _ = self
            .bufs
            .create_part(x, y, PT_SPRK, CreationSource::Program, 0, &self.registry);
    }
}

/// `RNG::next` (`tpt-rand.cpp:12-23`), the xoroshiro128+ step behind
/// `Simulation::rng`.
///
/// [`SimulationEngine::rng_state`] holds the two 64-bit state words the same
/// way Lua's `sim.randomSeed` exposes them (`LuaSimulation.cpp:1992-2002`):
/// the low and high halves of `s[0]`, then those of `s[1]`.
#[inline]
pub(super) fn rng_next(state: &mut [u32; 4]) -> u64 {
    let s0 = u64::from(state[0]) | (u64::from(state[1]) << 32);
    let s1 = u64::from(state[2]) | (u64::from(state[3]) << 32);

    // `RNG::next`: result = s0 + s1; s1 ^= s0; s[0] = rotl(s0, 55) ^ s1 ^
    // (s1 << 14); s[1] = rotl(s1, 36).
    let result = s0.wrapping_add(s1);
    let s1 = s1 ^ s0;
    let next_s0 = s0.rotate_left(55) ^ s1 ^ (s1 << 14);
    let next_s1 = s1.rotate_left(36);

    state[0] = next_s0 as u32;
    state[1] = (next_s0 >> 32) as u32;
    state[2] = next_s1 as u32;
    state[3] = (next_s1 >> 32) as u32;
    result
}

/// Mass a `PGRV`/`NGRV` stroke writes into a coarse cell.
///
/// `PGRV.cpp` / `NGRV.cpp` write `strength * ±5.0f`; the UI tools use
/// `Strength == 1.0`, and the parameter is kept so Lua-driven tools scale the
/// same way they do in TPT.
pub(super) fn tool_gravity_mass(strength: f32, attract: bool) -> f32 {
    let magnitude = strength.abs() * 5.0;
    if attract { magnitude } else { -magnitude }
}

/// Coarse cell index a pixel's gravity-tool write lands in.
///
/// Mirrors `gravIn.mass[Vec2{x, y} / CELL]`: integer division of the pixel
/// coordinate, rejecting pixels outside the canvas.
pub(super) fn gravity_tool_cell(
    x: i32,
    y: i32,
    xres: i32,
    yres: i32,
    xcells: i32,
    ycells: i32,
) -> Option<usize> {
    if x < 0 || y < 0 || x >= xres || y >= yres {
        return None;
    }
    let cx = x / CELL as i32;
    let cy = y / CELL as i32;
    if cx < 0 || cy < 0 || cx >= xcells || cy >= ycells {
        return None;
    }
    Some((cy * xcells + cx) as usize)
}
