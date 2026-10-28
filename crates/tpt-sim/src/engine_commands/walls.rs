//! Wall placement and flood-fill commands.

use std::collections::VecDeque;

use tpt_core::constants::*;

use crate::SimulationEngine;

impl SimulationEngine {
    /// Paint wall cells (and optionally set the fan velocity of those cells).
    ///
    /// `cells` are coarse `CELL x CELL` block coordinates in `0..xcells` /
    /// `0..ycells`, exactly as documented on [`tpt_core::commands::SimCommand::PlaceWalls`]: every
    /// caller (the brush stroke code, Lua's `sim.wallProperty`, save restores)
    /// has already divided the pixel position by `CELL`. Dividing again here
    /// would paint a block a quarter of the way towards the origin, so the
    /// coordinates are used as-is.
    pub(super) fn place_walls(
        &mut self,
        cells: &[(i32, i32)],
        wall_id: u8,
        fan_vel: Option<(f32, f32)>,
    ) {
        let xcells = self.bufs.config.xcells() as i32;
        let ycells = self.bufs.config.ycells() as i32;
        if xcells <= 0 || ycells <= 0 {
            return;
        }
        for &(cx, cy) in cells {
            if cx < 0 || cy < 0 || cx >= xcells || cy >= ycells {
                continue;
            }
            let idx = cy as usize * xcells as usize + cx as usize;
            // `Editing.cpp:209-219`: `WL_ERASEALL` wipes the whole block -
            // particles and sign anchors included - and leaves no wall, unlike
            // `WL_ERASE` which only clears the wall layer.
            let stored_wall = if wall_id == WL_ERASEALL {
                let x0 = cx * CELL as i32;
                let y0 = cy * CELL as i32;
                for y in y0..y0 + CELL as i32 {
                    for x in x0..x0 + CELL as i32 {
                        self.bufs.kill_at(x, y);
                    }
                }
                let signs = self.signs().to_vec();
                let kept: Vec<_> = signs
                    .into_iter()
                    .filter(|sign| {
                        sign.x < x0
                            || sign.x >= x0 + CELL as i32
                            || sign.y < y0
                            || sign.y >= y0 + CELL as i32
                    })
                    .collect();
                self.set_signs(kept);
                0u32
            } else {
                wall_id as u32
            };
            self.bufs.wall_bmap[idx] = stored_wall;
            self.bufs.wall_emap[idx] = 0;
            match fan_vel {
                Some((fx, fy)) => {
                    self.bufs.fan_vx[idx] = fx;
                    self.bufs.fan_vy[idx] = fy;
                }
                None => {
                    if wall_id != WL_FAN {
                        self.bufs.fan_vx[idx] = 0.0;
                        self.bufs.fan_vy[idx] = 0.0;
                    }
                }
            }
        }
        self.bufs.rebuild_wall_masks();
    }

    /// Flood-fill walls starting at (x, y) with wall_id.
    pub fn flood_walls(&mut self, x: i32, y: i32, wall_id: u8, replace_wall: Option<u8>) -> usize {
        let xcells = self.bufs.config.xcells() as i32;
        let ycells = self.bufs.config.ycells() as i32;
        if xcells <= 0 || ycells <= 0 {
            return 0;
        }
        let (cx, cy) = if x >= xcells || y >= ycells {
            (x / CELL as i32, y / CELL as i32)
        } else {
            (x, y)
        };
        if cx < 0 || cy < 0 || cx >= xcells || cy >= ycells {
            return 0;
        }
        let start_idx = cy as usize * xcells as usize + cx as usize;
        let target = replace_wall.unwrap_or(self.bufs.wall_bmap[start_idx] as u8);
        if target == wall_id {
            return 0;
        }
        let mut queue = VecDeque::new();
        queue.push_back((cx, cy));
        let mut count = 0;
        let mut visited = vec![false; (xcells * ycells) as usize];
        visited[start_idx] = true;

        while let Some((c_x, c_y)) = queue.pop_front() {
            let idx = c_y as usize * xcells as usize + c_x as usize;
            if self.bufs.wall_bmap[idx] as u8 == target {
                self.bufs.wall_bmap[idx] = wall_id as u32;
                self.bufs.wall_emap[idx] = 0;
                count += 1;
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (c_x + dx, c_y + dy);
                    if nx >= 0 && ny >= 0 && nx < xcells && ny < ycells {
                        let nidx = ny as usize * xcells as usize + nx as usize;
                        if !visited[nidx] {
                            visited[nidx] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }
        }
        if count > 0 {
            self.bufs.rebuild_wall_masks();
            self.bufs.host_walls_dirty = true;
            self.bufs.host_dirty = true;
        }
        count
    }
}
