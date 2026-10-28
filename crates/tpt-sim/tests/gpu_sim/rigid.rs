use super::*;

/// Walls off the `CELL`-wide canvas border with `WL_WALL`.
///
/// `Simulation.cpp:2313-2317` kills every particle inside the border band
/// (`x < CELL || y < CELL || x >= XRES-CELL || y >= YRES-CELL`), and a loose
/// pile thins out sideways until grains reach it. Walls live on the wall grid,
/// not in the particle pool, so the kill cannot touch them, and `eval_move`
/// (`Simulation.cpp:1024-1050`) refuses to move a particle into a `WL_WALL`
/// cell, which keeps every grain clear of the band.
fn wall_off_canvas_border(engine: &mut SimulationEngine) {
    let xcells = engine.config().xres.div_euclid(CELL) as i32;
    let ycells = engine.config().yres.div_euclid(CELL) as i32;
    let mut cells = Vec::with_capacity(2 * (xcells + ycells) as usize);
    for cy in 0..ycells {
        cells.push((0, cy));
        cells.push((xcells - 1, cy));
    }
    for cx in 0..xcells {
        cells.push((cx, 0));
        cells.push((cx, ycells - 1));
    }
    engine.apply_command(SimCommand::PlaceWalls {
        cells,
        wall_id: WL_WALL,
        fan_vel: None,
    });
}

#[test]
fn test_rigid_body_physics_and_melting() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(320, 200);

    let body_id = engine.spawn_rigid_body(160.0, 60.0, 6, 4, RigidSpawnShape::Box, PT_WOOD);
    assert_eq!(engine.rigid_body_count(), 1);
    assert_eq!(engine.bufs.rigid_bodies[0].id, body_id);

    let pixels0 = engine.bufs.rigid_bodies[0].pixels.len();
    let mass0 = engine.bufs.rigid_bodies[0].mass;
    assert_eq!(pixels0, 13 * 9, "13x9 pixel box");
    assert!(
        (mass0 - 100.0 * pixels0 as f32).abs() < 1.0,
        "mass must be the sum of the pixel weights ({mass0})"
    );

    // Give the body a spin so the rasteriser has to rotate the cluster.
    engine.bufs.rigid_bodies[0].omega = 0.05;
    let cm_y0 = engine.bufs.rigid_bodies[0].cm_y;
    let angle0 = engine.bufs.rigid_bodies[0].angle;

    for _ in 0..25 {
        step(&mut engine);
    }

    engine.sync_rigid_bodies_to_host();
    let body = engine
        .bufs
        .rigid_bodies
        .first()
        .cloned()
        .expect("the rigid body must survive the drop");
    assert_eq!(
        body.pixels.len(),
        pixels0,
        "an intact body keeps every pixel while falling"
    );
    assert!(
        body.cm_y > cm_y0 + 5.0,
        "the body must fall under gravity: cm_y {cm_y0} -> {}",
        body.cm_y
    );
    assert!(
        (body.angle - angle0).abs() > 0.2,
        "the body must rotate as a rigid cluster: angle {angle0} -> {}",
        body.angle
    );
    assert!(
        count_type(&engine, PT_WOOD) >= pixels0,
        "every body pixel must be materialised as a wood particle ({} < {pixels0})",
        count_type(&engine, PT_WOOD)
    );

    // The cluster stays welded: no pixel may drift away from the centre of mass.
    let max_radius = body
        .pixels
        .iter()
        .map(|px| {
            let dx = px.grid_x as f32 - body.cm_x;
            let dy = px.grid_y as f32 - body.cm_y;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(
        max_radius < 12.0,
        "the rigid cluster must stay compact, max radius = {max_radius}"
    );

    engine.config_mut().gravity_mode = GravityMode::Off;
    {
        let resting = &mut engine.bufs.rigid_bodies[0];
        resting.vx = 0.0;
        resting.vy = 0.0;
        resting.omega = 0.0;
    }
    let body = engine.bufs.rigid_bodies[0].clone();
    let (_, min_y, _, _) = body_bounds(&body);
    // Target the two topmost pixel rows so the lava is always inside the
    // combustion kernel's 5x5 neighbourhood.
    let mut top_xs: Vec<i32> = body
        .pixels
        .iter()
        .filter(|px| px.grid_y <= min_y + 1)
        .map(|px| px.grid_x)
        .collect();
    top_xs.sort_unstable();
    let damage_x0 = top_xs[0];
    let damage_x1 = top_xs[top_xs.len() / 2];
    // A 3-row lava reservoir over the left half stays hot long enough to ignite
    // the wood instead of freezing on the first conduction.
    let lava_y0 = (min_y - 3).max(0);
    fill_rect(
        &mut engine,
        damage_x0,
        lava_y0,
        damage_x1,
        min_y - 1,
        PT_LAVA,
    );
    set_temp_of_type(&mut engine, PT_LAVA, 1800.0);
    let mass_before_damage = body.mass;

    // Wood only ignites through its `HighTemperatureTransition`, i.e. once
    // conduction has carried a pixel past 973 K; twelve frames sat close enough
    // to the threshold that GPU float ordering decided the outcome.
    for _ in 0..30 {
        // Keep the reservoir at its spawn temperature: the test is about the
        // rigid-body fracture path, not about lava thermodynamics.
        set_temp_of_type(&mut engine, PT_LAVA, 1800.0);
        step(&mut engine);
    }

    engine.sync_rigid_bodies_to_host();
    let damaged = engine
        .bufs
        .rigid_bodies
        .first()
        .cloned()
        .expect("the partially burned body must still exist");
    let rigid_after_damage = count_rigid_any(&engine);
    assert!(
        damaged.pixels.len() < pixels0,
        "lava must burn pixels off the cluster ({} of {pixels0} left)",
        damaged.pixels.len()
    );
    assert!(
        rigid_after_damage >= damaged.pixels.len(),
        "every surviving body pixel stays welded ({rigid_after_damage} welded for {} pixels)",
        damaged.pixels.len()
    );
    assert!(
        count_rigid_type(&engine, PT_WOOD) <= rigid_after_damage,
        "the wood subset cannot outnumber the welded particles"
    );
    // The global count of `FLAG_RIGID_PIXEL` wood is deliberately not asserted
    // to shrink: interior filler pixels carry the same flag, and a welded pixel
    // that melts and resolidifies in place nets out to no change. The body's
    // own pixel list and mass are the deterministic signals.
    assert!(
        damaged.mass < mass_before_damage,
        "the mass must shrink with the cluster ({mass_before_damage} -> {})",
        damaged.mass
    );
    assert!(
        (damaged.mass - 100.0 * damaged.pixels.len() as f32).abs() < 1.0,
        "mass must track the surviving pixels ({} pixels, mass {})",
        damaged.pixels.len(),
        damaged.mass
    );
    assert!(
        (damaged.cm_x - body.cm_x).abs() + (damaged.cm_y - body.cm_y).abs() > 0.1,
        "the centroid must be recomputed after the fracture ({}, {}) -> ({}, {})",
        body.cm_x,
        body.cm_y,
        damaged.cm_x,
        damaged.cm_y
    );

    // Detached pixels become ordinary, free-flowing particles.
    let mut free_fire = 0usize;
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] == PT_FIRE && engine.bufs.part_flags[id] & FLAG_RIGID_PIXEL == 0
        {
            free_fire += 1;
        }
    }
    assert!(
        free_fire > 0,
        "burned pixels must detach into free-flowing particles"
    );
}

/// Contact with a dense loose-particle pile must stay in normal frame time.
///
/// The contact pass walks one body's pixels sequentially in a single GPU
/// thread, once per collision sub-step. A brush-radius cluster (the UI caps
/// `MAX_BRUSH_RADIUS` at 100, i.e. 201x201 pixels) therefore used to run up to
/// 12 passes over 40401 pixels per frame while it ploughed through sand, which
/// cost hundreds of milliseconds per frame - long enough for the GUI's TPS
/// counter to read 0.
///
/// The test measures the same body twice: once falling through empty space and
/// once ploughing into a sand/water pile, and compares the per-frame averages.
/// A ratio keeps the assertion independent of how loaded the shared GPU is
/// (the full suite leaks one device per test, which slows every frame by the
/// same factor), while the fix moved the collision cost from ~13x the free
/// fall to ~3x. The one-word map read forces the queue to drain, so the timings
/// are real device wall time and not submission time.
#[test]
fn test_rigid_body_particle_contact_stays_fast() {
    use std::time::{Duration, Instant};

    let _guard = gpu_lock();
    let mut engine = create_test_engine(612, 384);

    // Warm-up body: compiles the rigid kernels (and every other pass) on a
    // handful of pixels, so no measured frame pays a shader compile.
    engine.spawn_rigid_body(80.0, 120.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    for _ in 0..30 {
        engine.step();
    }

    // The largest body the spawn tool can build: a 201x201 stone cluster.
    let body_id = engine.spawn_rigid_body(300.0, 100.0, 100, 100, RigidSpawnShape::Box, PT_STNE);
    let spawn_pixels = engine
        .bufs
        .rigid_bodies
        .iter()
        .find(|b| b.id == body_id)
        .map(|b| b.pixels.len());
    assert_eq!(spawn_pixels, Some(201 * 201));

    let anchor = engine.bufs.gpu_pmap.clone();
    let trim = (engine.bufs.pmap.len() * 4 - 4) as u64;
    // One frame: step, then block on a one-word map read so every kernel this
    // frame submitted has completed before the timer stops.
    let frame = |engine: &mut SimulationEngine| -> Duration {
        let started = Instant::now();
        engine.step();
        engine
            .bufs
            .client
            .read(vec![anchor.clone().offset_start(0).offset_end(trim)])
            .expect("the rigid frame must not lose the device");
        started.elapsed()
    };

    // The first step after a spawn runs the host initialiser (full read-back,
    // 40k `create_part` calls, upload) and returns without running the device
    // passes; the second allocates the owner / claim grids for the new body.
    // Both are one-time costs no steady-state frame pays.
    frame(&mut engine);
    frame(&mut engine);

    // Phase A: the same body in empty space, no contact but the canvas edge.
    let mut free_total = Duration::ZERO;
    for _ in 0..10 {
        free_total += frame(&mut engine);
    }

    // Phase B: a deep, loose pile below the body. Sand carries the solid
    // contact branch, the water layer above it the fluid drag / displacement
    // branch. It is painted after phase A so the free-fall frames cannot
    // touch it.
    fill_rect(&mut engine, 40, 250, 570, 360, PT_SAND);
    fill_rect(&mut engine, 40, 230, 570, 249, PT_WATR);
    // Absorb the paint upload so it does not land in a measured frame.
    frame(&mut engine);
    frame(&mut engine);

    let mut worst = Duration::ZERO;
    let mut worst_frame = 0usize;
    let mut hit_total = Duration::ZERO;
    for index in 0..40 {
        let elapsed = frame(&mut engine);
        hit_total += elapsed;
        if elapsed > worst {
            worst = elapsed;
            worst_frame = index;
        }
        // A single frame over a second is the stall that reads as 0 TPS in the
        // GUI; the ratio below is the actual regression guard.
        assert!(
            elapsed < Duration::from_secs(1),
            "collision frame {index} took {elapsed:?} with a {}-pixel body in \
             the pile (worst so far {worst:?} at frame {worst_frame})",
            201 * 201
        );
    }

    // Per-frame averages, so the two phases are comparable.
    let free_avg = free_total.as_secs_f64() / 10.0;
    let hit_avg = hit_total.as_secs_f64() / 40.0;
    // Free fall samples almost no contacts, so a ratio against it tracks
    // machine noise (and the contact-free passes keep getting cheaper) rather
    // than a regression. The absolute cost of the largest spawnable body
    // ploughing a sand pile is what a frame actually pays here.
    assert!(
        hit_avg < 0.25,
        "colliding frames averaged {hit_avg:.3} s against {free_avg:.3} s in \
         free fall (worst {worst:?} at frame {worst_frame})"
    );

    // The body must stay one consistent cluster across the impact.
    engine.ensure_host_current();
    engine.sync_rigid_bodies_to_host();
    let body = engine
        .bufs
        .rigid_bodies
        .iter()
        .find(|b| b.id == body_id)
        .cloned()
        .expect("the body must survive the impact");
    assert!(body.active, "the body must stay active");
    assert_eq!(body.pixels.len(), 201 * 201, "stone keeps every pixel");
    assert!(
        body.cm_y > 160.0,
        "the body must have fallen through the pile (cm_y {})",
        body.cm_y
    );
    assert!(
        body.cm_x.is_finite() && body.cm_y.is_finite(),
        "the body pose must stay finite: ({}, {})",
        body.cm_x,
        body.cm_y
    );
    assert!(
        body.mass.is_finite() && body.mass > 0.0,
        "the body mass must stay finite and positive ({})",
        body.mass
    );
    for px in &body.pixels {
        assert!(
            px.grid_x >= 0
                && px.grid_x < engine.config().xres as i32
                && px.grid_y >= 0
                && px.grid_y < engine.config().yres as i32,
            "a body pixel escaped the canvas at ({}, {})",
            px.grid_x,
            px.grid_y
        );
    }
}

/// The steady-state rigid pass must run entirely on the device: a body falls,
/// lands on a wall floor and comes to rest without the host ever syncing
/// particles or maps, and the host description stays consistent once an
/// explicit `ensure_host_current()` asks for it.
#[test]
fn test_rigid_body_device_steady_state_falls_and_settles() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(320, 200);

    let ycells = engine.config().yres.div_euclid(CELL);
    let xcells = engine.config().xres.div_euclid(CELL);
    let mut floor_cells = Vec::new();
    for cx in 0..xcells as i32 {
        floor_cells.push((cx, ycells as i32 - 1));
    }
    engine.apply_command(SimCommand::PlaceWalls {
        cells: floor_cells,
        wall_id: WL_WALL,
        fan_vel: None,
    });

    let body_id = engine.spawn_rigid_body(160.0, 150.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    assert_eq!(engine.rigid_body_count(), 1);
    let spawn_pixels = engine.bufs.rigid_bodies[0].pixels.len();
    assert_eq!(spawn_pixels, 9 * 7);
    let start_y = engine.bufs.rigid_bodies[0].cm_y;
    let floor_top = engine.config().yres as f32 - 1.0;

    // 60 device ticks. `step` performs the explicit read-back the save/Lua
    // paths use; the per-frame rigid work itself must not transfer anything.
    // `ensure_host_current` only refreshes the particle shadows, so the body
    // description lags by the adoption step; tick 59 therefore reports the
    // transform of tick 58, one pair of frames away from the final state.
    let mut resting_y = f32::NAN;
    for tick in 0..60 {
        engine.step();
        engine.ensure_host_current();
        if tick == 59 {
            resting_y = engine
                .bufs
                .rigid_bodies
                .first()
                .map(|b| b.cm_y)
                .unwrap_or(f32::NAN);
        }
    }

    assert!(
        engine.rigid_gpu.is_device_authoritative(),
        "the device must own the rigid bodies in the steady state"
    );
    engine.sync_rigid_bodies_to_host();
    let body = engine
        .bufs
        .rigid_bodies
        .first()
        .cloned()
        .expect("the rigid body must survive the drop");
    assert_eq!(body.id, body_id);
    assert!(body.active, "the settled body must stay active");
    assert_eq!(
        body.pixels.len(),
        spawn_pixels,
        "an intact body keeps every pixel while falling and landing"
    );

    let (_, _, _, max_y) = body_bounds(&body);
    assert!(
        body.cm_y > start_y + 20.0,
        "the body must fall under gravity: cm_y {start_y} -> {}",
        body.cm_y
    );
    assert!(
        max_y >= 188 && (max_y as f32) <= floor_top,
        "the body must rest on the wall floor (bottom row {max_y})"
    );
    assert!(
        (body.cm_y - resting_y).abs() < 0.75,
        "the landed body must stop moving: cm_y {resting_y} -> {}",
        body.cm_y
    );
    assert!(
        body.vy.abs() < 0.5,
        "the resting body must lose its vertical velocity ({})",
        body.vy
    );
    assert!(
        body.omega.abs() < 0.25,
        "the resting body must stop spinning ({})",
        body.omega
    );

    // The rasterised pixels stay welded to the body in the host read-back.
    assert_eq!(
        count_rigid_type(&engine, PT_WOOD),
        body.pixels.len(),
        "every surviving body pixel must still carry FLAG_RIGID_PIXEL"
    );
    let rigid_body_pixels = engine
        .bufs
        .part_flags
        .iter()
        .zip(engine.bufs.part_ty.iter())
        .filter(|(flags, ty)| **ty == PT_WOOD && **flags & FLAG_RIGID_PIXEL != 0)
        .count();
    assert_eq!(rigid_body_pixels, body.pixels.len());
}

/// A fast body must never step over a one-pixel-thick terrain floor.
///
/// The contact pass samples one cell per sub-step, so the sub-step length is
/// the whole collision guarantee: before the fix the count was capped at 12
/// while the step target was 0.5 cells, so a clamped 16 px/frame body moved
/// 1.33 px per sub-step and the destination cell could jump from `y` to
/// `y + 2` - over the floor. The chunk now sizes its sub-steps from the
/// per-frame surface travel (at most one cell each, acceleration included) and
/// stops at the face it crosses instead of pushing from the cell centre.
#[test]
fn test_rigid_body_fast_drop_does_not_tunnel_through_one_pixel_floor() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(208, 300);
    // `Simulation.cpp:2313-2317` kills every particle inside the 4-pixel canvas
    // border (`x < CELL || y < CELL || x >= XRES-CELL || y >= YRES-CELL`), so
    // the one-pixel floor is kept one row above the bottom band, stops short of
    // the side bands, and rests on a wall bed (walls are not particles and are
    // not subject to the kill). The 208-wide canvas keeps the original 200
    // floor stones once the 4-pixel side bands are excluded.
    let xcells = engine.config().xres.div_euclid(CELL);
    let ycells = engine.config().yres.div_euclid(CELL);
    let cells: Vec<(i32, i32)> = (0..xcells as i32)
        .map(|cx| (cx, ycells as i32 - 1))
        .collect();
    engine.apply_command(SimCommand::PlaceWalls {
        cells,
        wall_id: WL_WALL,
        fan_vel: None,
    });
    let floor_y = (ycells as i32 - 1) * CELL as i32 - 1;
    let floor_x0 = CELL as i32;
    let floor_x1 = engine.config().xres as i32 - CELL as i32 - 1;
    fill_rect(&mut engine, floor_x0, floor_y, floor_x1, floor_y, PT_STNE);
    let floor_count = count_type(&engine, PT_STNE);
    assert_eq!(floor_count, (floor_x1 - floor_x0 + 1) as usize);

    let body_id = engine.spawn_rigid_body(100.0, 50.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    // The solver clamps the linear speed to 16 px/frame; 40 only pins that the
    // drop is a fast one.
    engine.bufs.rigid_bodies[0].vy = 40.0;

    let mut worst_bottom = i32::MIN;
    for _ in 0..150 {
        step(&mut engine);
        engine.sync_rigid_bodies_to_host();
        let body = engine
            .bufs
            .rigid_bodies
            .iter()
            .find(|b| b.id == body_id)
            .cloned()
            .expect("the body must survive the impact");
        let (_, _, _, max_y) = body_bounds(&body);
        worst_bottom = worst_bottom.max(max_y);
    }

    assert!(
        worst_bottom < floor_y,
        "the body entered the one-pixel floor: worst bottom row {worst_bottom}, floor row {floor_y}"
    );

    let body = engine
        .bufs
        .rigid_bodies
        .iter()
        .find(|b| b.id == body_id)
        .cloned()
        .expect("the body must survive the impact");
    assert!(body.active, "the body must stay active");
    assert_eq!(body.pixels.len(), 9 * 7, "wood keeps every pixel");
    assert!(
        body.vy.abs() < 0.25,
        "the body must come to rest on the floor, vy = {}",
        body.vy
    );
    // The floor is not deleted, overwritten or displaced: every stone is still
    // in its own cell and still mapped.
    assert_eq!(
        count_type(&engine, PT_STNE),
        floor_count,
        "the floor lost grains"
    );
    for slot in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[slot] == PT_STNE {
            let gx = engine.bufs.part_x[slot].round() as i32;
            let gy = engine.bufs.part_y[slot].round() as i32;
            assert_eq!(gy, floor_y, "a floor grain moved to ({gx}, {gy})");
            let idx = gy as usize * engine.config().xres as usize + gx as usize;
            let want = (((slot as u32) + 1) << 9) | (PT_STNE & 0x1FF);
            assert_eq!(
                engine.bufs.pmap[idx], want,
                "a floor grain lost its pmap cell"
            );
        }
    }
}

/// A fast body must not tunnel through another body, and the two clusters must
/// never share a cell or a bounding box.
///
/// The old contact resolved against the destination cell with the axis taken
/// from the previous separation. Once a fast pixel's centre passed the
/// obstacle pixel's centre, that axis pointed *into* the resting body and the
/// positional correction shoved the moving body forward - the harder the
/// impact, the further it was launched through. The entry face is now derived
/// from the crossed face of the swept segment, and the deeper of the two cells
/// cannot be reached in one sub-step.
#[test]
fn test_rigid_body_fast_hit_does_not_interpenetrate_resting_body() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(400, 200);
    engine.config_mut().gravity_mode = GravityMode::Off;

    let a_id = engine.spawn_rigid_body(60.0, 100.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    let b_id = engine.spawn_rigid_body(200.0, 100.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    engine.bufs.rigid_bodies[0].vx = 40.0;

    for _ in 0..30 {
        step(&mut engine);
        engine.sync_rigid_bodies_to_host();
    }

    let a = engine
        .bufs
        .rigid_bodies
        .iter()
        .find(|b| b.id == a_id)
        .cloned()
        .expect("the moving body must survive the impact");
    let b = engine
        .bufs
        .rigid_bodies
        .iter()
        .find(|b| b.id == b_id)
        .cloned()
        .expect("the resting body must survive the impact");
    assert_eq!(a.pixels.len(), 9 * 7, "the moving body must stay welded");
    assert_eq!(b.pixels.len(), 9 * 7, "the resting body must stay welded");

    let cells_a: std::collections::HashSet<(i32, i32)> =
        a.pixels.iter().map(|p| (p.grid_x, p.grid_y)).collect();
    let cells_b: std::collections::HashSet<(i32, i32)> =
        b.pixels.iter().map(|p| (p.grid_x, p.grid_y)).collect();
    let shared: Vec<(i32, i32)> = cells_a.intersection(&cells_b).copied().collect();
    assert!(
        shared.is_empty(),
        "the bodies share {} cells, e.g. {shared:?}",
        shared.len()
    );

    let (ax0, ay0, ax1, ay1) = body_bounds(&a);
    let (bx0, by0, bx1, by1) = body_bounds(&b);
    let overlap_w = (ax1.min(bx1) - ax0.max(bx0) + 1).max(0);
    let overlap_h = (ay1.min(by1) - ay0.max(by0) + 1).max(0);
    assert_eq!(
        overlap_w * overlap_h,
        0,
        "the bounding boxes overlap: A = ({ax0},{ay0})-({ax1},{ay1}), B = ({bx0},{by0})-({bx1},{by1})"
    );
    assert!(
        ax1 < bx0 && a.cm_x < b.cm_x,
        "the moving body ended up past the resting one: A max_x {ax1}, B min_x {bx0}, \
         A cm_x {}, B cm_x {} (vx {})",
        a.cm_x,
        b.cm_x,
        a.vx
    );
    assert!(
        a.vx.abs() < 6.0,
        "the impact must stop the moving body instead of letting it through (vx {})",
        a.vx
    );
}

/// A body dropped at full speed onto a wall floor must stop on its surface, not
/// sink into the wall band.
///
/// The old penetration was measured from the blocked cell's centre
/// (`1 - |offset|` is at least 0.5 for any point inside the cell), so the
/// correction could never fully separate the pixel and the rasteriser kept
/// rounding it back into the wall. The measurement is now taken from the face
/// the pixel crossed, plus a slop so `round` cannot land inside again.
#[test]
fn test_rigid_body_fast_drop_onto_wall_stops_on_the_surface() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(200, 300);
    let ycells = engine.config().yres.div_euclid(CELL);
    let xcells = engine.config().xres.div_euclid(CELL);
    let cells: Vec<(i32, i32)> = (0..xcells as i32)
        .map(|cx| (cx, ycells as i32 - 1))
        .collect();
    engine.apply_command(SimCommand::PlaceWalls {
        cells,
        wall_id: WL_WALL,
        fan_vel: None,
    });
    let wall_top = (ycells as i32 - 1) * 4;

    let body_id = engine.spawn_rigid_body(100.0, 50.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    engine.bufs.rigid_bodies[0].vy = 40.0;

    let mut worst_bottom = i32::MIN;
    for _ in 0..150 {
        step(&mut engine);
        engine.sync_rigid_bodies_to_host();
        let body = engine
            .bufs
            .rigid_bodies
            .iter()
            .find(|b| b.id == body_id)
            .cloned()
            .expect("the body must survive the impact");
        let (_, _, _, max_y) = body_bounds(&body);
        worst_bottom = worst_bottom.max(max_y);
    }

    assert!(
        worst_bottom < wall_top,
        "the body entered the wall band: worst bottom row {worst_bottom}, wall top {wall_top}"
    );
    let body = engine
        .bufs
        .rigid_bodies
        .iter()
        .find(|b| b.id == body_id)
        .cloned()
        .expect("the body must survive the impact");
    assert_eq!(body.pixels.len(), 9 * 7);
    assert!(
        body.vy.abs() < 0.25,
        "the body must rest on the wall floor (vy {})",
        body.vy
    );
    assert!(
        body.omega.abs() < 0.25,
        "the body must stop spinning on the wall floor (omega {})",
        body.omega
    );
}

/// A body resting on a powder pile must be supported by it: it settles on the
/// surface instead of sinking through the pile, and the grains under it stay in
/// the map.
///
/// A loose powder used to be handled by the fluid branch only (buoyancy and a
/// drag term normalised by the *whole* pixel count), so a 63-pixel cluster sank
/// at free-fall speed and the rasteriser overwrote the grains it replaced. A
/// powder whose overburden pressure (`unit_weight * radius`) is below its yield
/// strength is now a hard support; a boulder still ploughs (see the contact
/// cost test).
#[test]
fn test_rigid_body_rests_on_sand_without_sinking() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(208, 300);
    // Confine the pile: `Simulation.cpp:2313-2317` kills every particle inside
    // the 4-pixel canvas border, and a loose powder spreads sideways until it
    // reaches that band. The `WL_WALL` ring over the border cells is not a
    // particle, so the kill cannot erase it, and it blocks movement into the
    // band. The pile then rests on its wall bed (walls are not particles)
    // instead of bleeding grains off the canvas edge. The 80-row depth, the
    // pile top and the 208-wide canvas keep the original 16000 grains.
    wall_off_canvas_border(&mut engine);
    let xres = engine.config().xres as i32;
    let yres = engine.config().yres as i32;
    let x0 = CELL as i32;
    let x1 = xres - CELL as i32 - 1;
    let pile_top = 120;
    let pile_bottom = pile_top + 79;
    let bed_top = pile_bottom + 1;
    assert_eq!(
        bed_top % CELL as i32,
        0,
        "the wall bed must be cell aligned"
    );
    let xcells = engine.config().xres.div_euclid(CELL);
    let cells: Vec<(i32, i32)> = (0..xcells as i32)
        .map(|cx| (cx, bed_top / CELL as i32))
        .collect();
    engine.apply_command(SimCommand::PlaceWalls {
        cells,
        wall_id: WL_WALL,
        fan_vel: None,
    });
    fill_rect(&mut engine, x0, pile_top, x1, pile_bottom, PT_SAND);
    let sand_count = count_type(&engine, PT_SAND);
    assert_eq!(
        sand_count,
        ((pile_bottom - pile_top + 1) * (x1 - x0 + 1)) as usize
    );

    let body_id = engine.spawn_rigid_body(100.0, 90.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    engine.bufs.rigid_bodies[0].vy = 0.0;

    let mut worst_bottom = i32::MIN;
    let mut last_cm = f32::NAN;
    let mut settled_cm = f32::NAN;
    for tick in 0..120 {
        step(&mut engine);
        engine.sync_rigid_bodies_to_host();
        let body = engine
            .bufs
            .rigid_bodies
            .iter()
            .find(|b| b.id == body_id)
            .cloned()
            .expect("the body must survive the landing");
        let (_, _, _, max_y) = body_bounds(&body);
        worst_bottom = worst_bottom.max(max_y);
        if tick == 100 {
            settled_cm = body.cm_y;
        }
        last_cm = body.cm_y;
    }

    assert!(
        worst_bottom <= pile_top,
        "the body sank below the pile surface: worst bottom row {worst_bottom}, pile top {pile_top}"
    );
    assert!(
        (last_cm - settled_cm).abs() < 0.25,
        "the body must stop moving on the pile: cm_y {settled_cm} -> {last_cm}"
    );
    let body = engine
        .bufs
        .rigid_bodies
        .iter()
        .find(|b| b.id == body_id)
        .cloned()
        .expect("the body must survive the landing");
    assert!(
        body.vy.abs() < 0.25,
        "the resting body must lose its vertical velocity ({})",
        body.vy
    );

    // The pile is neither consumed nor overwritten: every grain is still live
    // and still owns the pmap cell it stands on, and the surface row under the
    // body is intact.
    assert_eq!(
        count_type(&engine, PT_SAND),
        sand_count,
        "the pile lost grains"
    );
    let mut surface = vec![false; xres as usize];
    for slot in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[slot] == PT_SAND {
            let gx = engine.bufs.part_x[slot].round() as i32;
            let gy = engine.bufs.part_y[slot].round() as i32;
            assert!(
                (0..xres).contains(&gx) && (0..yres).contains(&gy),
                "a grain escaped the canvas at ({gx}, {gy})"
            );
            let idx = gy as usize * xres as usize + gx as usize;
            let want = (((slot as u32) + 1) << 9) | (PT_SAND & 0x1FF);
            assert_eq!(
                engine.bufs.pmap[idx], want,
                "the grain at ({gx}, {gy}) was overwritten out of the map"
            );
            if gy == pile_top {
                surface[gx as usize] = true;
            }
        }
    }
    assert!(
        surface[x0 as usize..=x1 as usize]
            .iter()
            .all(|&grain| grain),
        "the body ate into the pile's surface row"
    );
}

/// A body too heavy for the powder's yield strength must displace the grains it
/// sinks through, never overwrite them out of the map.
///
/// A brush-sized cluster exceeds the yield strength (`unit_weight * radius`),
/// so the contact keeps only the fluid buoyancy/drag response and the chunk
/// sinks. The rasteriser's displacement search used to stop at a 3x3 ring: a
/// grain under the middle of the chunk had no free neighbour, was left out of
/// `pmap`, and the rigid pixel took its cell. The search now covers a 17x17
/// window with a distance-weighted outward score.
#[test]
fn test_rigid_body_sinking_through_sand_displaces_grains() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(208, 300);
    // Confine the bed: the `CELL`-wide canvas border is lethal
    // (`Simulation.cpp:2313-2317`), and grains pushed aside by the plank would
    // otherwise thin out into it. The `WL_WALL` ring is immune to the kill and
    // blocks movement into the band, so the displaced grains stay in the pool
    // and the bed rests on its wall floor well clear of the band.
    wall_off_canvas_border(&mut engine);
    let xres = engine.config().xres as i32;
    let yres = engine.config().yres as i32;
    let x0 = CELL as i32;
    let x1 = xres - CELL as i32 - 1;
    let pile_top = 194;
    let pile_bottom = pile_top + 5;
    let bed_top = pile_bottom + 1;
    assert_eq!(
        bed_top % CELL as i32,
        0,
        "the wall bed must be cell aligned"
    );
    let xcells = engine.config().xres.div_euclid(CELL);
    let cells: Vec<(i32, i32)> = (0..xcells as i32)
        .map(|cx| (cx, bed_top / CELL as i32))
        .collect();
    engine.apply_command(SimCommand::PlaceWalls {
        cells,
        wall_id: WL_WALL,
        fan_vel: None,
    });
    fill_rect(&mut engine, x0, pile_top, x1, pile_bottom, PT_SAND);
    let sand_count = count_type(&engine, PT_SAND);
    assert_eq!(
        sand_count,
        ((pile_bottom - pile_top + 1) * (x1 - x0 + 1)) as usize
    );

    // A tall, thin stone plank: its radius puts it above the powder yield
    // strength, while its 3-pixel footprint keeps the displaced grains within
    // the rasteriser's search window.
    let body_id = engine.spawn_rigid_body(100.0, 60.0, 1, 40, RigidSpawnShape::Box, PT_STNE);
    assert_eq!(engine.bufs.rigid_bodies[0].pixels.len(), 3 * 81);

    for _ in 0..90 {
        step(&mut engine);
        engine.sync_rigid_bodies_to_host();
    }

    let body = engine
        .bufs
        .rigid_bodies
        .iter()
        .find(|b| b.id == body_id)
        .cloned()
        .expect("the plank must survive the pile");
    let (_, _, _, max_y) = body_bounds(&body);
    assert!(
        max_y > pile_top,
        "the heavy body must sink into the pile (bottom {max_y}, pile top {pile_top})"
    );

    assert_eq!(
        count_type(&engine, PT_SAND),
        sand_count,
        "sinking consumed grains"
    );
    // Displaced, not overwritten: no live grain may still stand in a cell the
    // body occupies. (Two grains can transiently share a cell through the
    // movement pass's own arbitration; that is a CA concern, not the rigid
    // rasteriser's, so the check is specifically against the body's cells.)
    let body_cells: std::collections::HashSet<(i32, i32)> =
        body.pixels.iter().map(|p| (p.grid_x, p.grid_y)).collect();
    for slot in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[slot] == PT_SAND {
            let gx = engine.bufs.part_x[slot].round() as i32;
            let gy = engine.bufs.part_y[slot].round() as i32;
            assert!(
                (0..xres).contains(&gx) && (0..yres).contains(&gy),
                "a grain escaped the canvas at ({gx}, {gy})"
            );
            assert!(
                !body_cells.contains(&(gx, gy)),
                "the grain at ({gx}, {gy}) was overwritten into a body cell"
            );
        }
    }
}

/// A falling body must hand its momentum to the body it lands on.
///
/// Contact used to be one-sided: the moving body resolved against the resting
/// body's rasterised cells (plus the resting body's surface velocity) and
/// stopped, but the resting body was never given the equal-and-opposite
/// impulse. The pair is now resolved with both inverse masses, and the
/// reaction is queued on the receiver's record as a fixed-point velocity
/// delta, so the total linear momentum is conserved across the impact.
#[test]
fn test_rigid_body_drop_transfers_momentum_to_resting_body() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(200, 900);
    engine.config_mut().gravity_mode = GravityMode::Off;

    let a_id = engine.spawn_rigid_body(100.0, 100.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    let b_id = engine.spawn_rigid_body(100.0, 300.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    engine.bufs.rigid_bodies[0].vy = 12.0;

    let mut shared_cells = 0usize;
    let mut prev_momentum = f32::NAN;
    let mut impact_momentum = f32::NAN;
    let mut pre_impact_momentum = f32::NAN;
    let mut b_start = f32::NAN;
    for _ in 0..40 {
        step(&mut engine);
        engine.sync_rigid_bodies_to_host();
        let a = engine
            .bufs
            .rigid_bodies
            .iter()
            .find(|b| b.id == a_id)
            .cloned()
            .expect("the falling body must survive the impact");
        let b = engine
            .bufs
            .rigid_bodies
            .iter()
            .find(|b| b.id == b_id)
            .cloned()
            .expect("the resting body must survive the impact");
        if b_start.is_nan() {
            b_start = b.cm_y;
        }
        let cells_a: std::collections::HashSet<(i32, i32)> =
            a.pixels.iter().map(|p| (p.grid_x, p.grid_y)).collect();
        let cells_b: std::collections::HashSet<(i32, i32)> =
            b.pixels.iter().map(|p| (p.grid_x, p.grid_y)).collect();
        shared_cells += cells_a.intersection(&cells_b).count();

        // No gravity, no world contact: apart from the solver's linear damping
        // (a fraction of a percent per frame) the only vertical force is the
        // pair impulse, so the impact frame must conserve the total. Measure
        // the frame the resting body starts moving to isolate the impulse from
        // the long free-flight damping.
        if b.vy > 0.5 && impact_momentum.is_nan() {
            pre_impact_momentum = prev_momentum;
            impact_momentum = a.mass * a.vy + b.mass * b.vy;
        }
        prev_momentum = a.mass * a.vy + b.mass * b.vy;
    }

    assert_eq!(
        shared_cells, 0,
        "the two bodies shared a cell during the impact"
    );
    assert!(
        !impact_momentum.is_nan(),
        "the resting body never started moving: the impulse was not transferred"
    );
    let error = (impact_momentum - pre_impact_momentum).abs();
    assert!(
        error <= pre_impact_momentum.abs() * 0.05,
        "the pair must conserve linear momentum: {pre_impact_momentum} -> {impact_momentum}"
    );

    let b = engine
        .bufs
        .rigid_bodies
        .iter()
        .find(|b| b.id == b_id)
        .cloned()
        .expect("the resting body must survive the impact");
    let a = engine
        .bufs
        .rigid_bodies
        .iter()
        .find(|b| b.id == a_id)
        .cloned()
        .expect("the falling body must survive the impact");
    assert!(
        b.vy > 0.5,
        "the resting body never received the impulse (vy {})",
        b.vy
    );
    assert!(
        b.cm_y > b_start + 100.0,
        "the resting body must be pushed down (cm_y {b_start} -> {})",
        b.cm_y
    );
    assert!(
        a.vy < 6.0,
        "the falling body must lose momentum to the other body (vy {})",
        a.vy
    );
}

/// A three-body stack must settle without jitter, without sharing cells and
/// without crushing the bottom body into the wall.
///
/// The load path is: top pushes the middle (queued reaction), middle pushes the
/// bottom (queued reaction), bottom is stopped by the wall. With one-sided
/// contacts the middle and bottom never felt the load. With the symmetric
/// exchange the stack has to converge to rest instead of oscillating.
///
/// The stack is confined in a vertical `WL_WALL` channel and starts at its
/// rest pose. A free-sided stack of equal boxes rocks itself apart on the
/// contact restitution and the test would then measure a fallen stack instead
/// of the load path; the channel pins the horizontal pose while the vertical
/// load path stays open.
#[test]
fn test_rigid_body_three_body_stack_transfers_load() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(200, 300);
    let ycells = engine.config().yres.div_euclid(CELL);
    let xcells = engine.config().xres.div_euclid(CELL);
    let cells: Vec<(i32, i32)> = (0..xcells as i32)
        .map(|cx| (cx, ycells as i32 - 1))
        .collect();
    engine.apply_command(SimCommand::PlaceWalls {
        cells,
        wall_id: WL_WALL,
        fan_vel: None,
    });
    let wall_top = (ycells as i32 - 1) * 4;

    // Confine the stack in a vertical `WL_WALL` channel. A pair contact stops
    // the sender's pose at the receiver's rasterised face, but with an open
    // side the top box can rock off the middle one and land beside it; the
    // stack-order assertion would then measure a fallen stack instead of the
    // load path. The channel's block cells pin the horizontal pose while the
    // vertical load path stays open. Walls are `CELL`-pixel blocks, so the
    // body (pixels 96..104 at x = 100.0) occupies cells 24..26 and the
    // flanking columns are 23 and 27.
    let column = (100.0f32 / CELL as f32) as i32;
    let mut side: Vec<(i32, i32)> = Vec::with_capacity(2 * ycells as usize);
    for cy in 0..ycells as i32 {
        side.push((column - 2, cy));
        side.push((column + 2, cy));
    }
    engine.apply_command(SimCommand::PlaceWalls {
        cells: side,
        wall_id: WL_WALL,
        fan_vel: None,
    });

    // Spawn the stack already at its rest pose. `Simulation.cpp:2313-2317`
    // kills every particle that ends up inside the border band, so the bottom
    // body must stop on the last free row above the wall (`max_y ==
    // wall_top - 1`, i.e. its centre at `wall_top - 4` for a 9x7 box). The
    // middle and top bodies start one body height higher each, so the three
    // boxes touch without sharing a cell - the resting invariant of the rigid
    // solver. Starting at rest isolates the load path: a body that misses the
    // symmetric pair reaction keeps accumulating the gravity velocity instead
    // of settling, which the rest assertions below catch.
    let bottom_y = wall_top as f32 - 4.0;
    let middle_y = bottom_y - 7.0;
    let top_y = middle_y - 7.0;
    let bottom_id = engine.spawn_rigid_body(100.0, bottom_y, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    let middle_id = engine.spawn_rigid_body(100.0, middle_y, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    let top_id = engine.spawn_rigid_body(100.0, top_y, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    let ids = [top_id, middle_id, bottom_id];

    let mut worst_bottom = i32::MIN;
    let mut shared_cells = 0usize;
    let mut late = [0.0f32; 3];
    for tick in 0..150 {
        step(&mut engine);
        engine.sync_rigid_bodies_to_host();
        let mut set: Vec<std::collections::HashSet<(i32, i32)>> = Vec::new();
        for id in ids {
            let body = engine
                .bufs
                .rigid_bodies
                .iter()
                .find(|b| b.id == id)
                .cloned()
                .expect("the stacked body must survive");
            set.push(body.pixels.iter().map(|p| (p.grid_x, p.grid_y)).collect());
            let (_, _, _, max_y) = body_bounds(&body);
            if id == bottom_id {
                worst_bottom = worst_bottom.max(max_y);
            }
        }
        shared_cells += set[0].intersection(&set[1]).count()
            + set[1].intersection(&set[2]).count()
            + set[0].intersection(&set[2]).count();
        if tick == 110 {
            for (i, id) in ids.iter().enumerate() {
                late[i] = engine
                    .bufs
                    .rigid_bodies
                    .iter()
                    .find(|b| b.id == *id)
                    .expect("body")
                    .cm_y;
            }
        }
    }

    assert_eq!(shared_cells, 0, "the stack shared cells while settling");
    assert!(
        worst_bottom < wall_top,
        "the bottom body entered the wall band: worst bottom row {worst_bottom}, wall top {wall_top}"
    );
    let bodies: Vec<_> = ids
        .iter()
        .map(|id| {
            engine
                .bufs
                .rigid_bodies
                .iter()
                .find(|b| b.id == *id)
                .cloned()
                .expect("the stacked body must survive")
        })
        .collect();
    for (i, body) in bodies.iter().enumerate() {
        assert_eq!(body.pixels.len(), 9 * 7, "body {i} must stay welded");
        assert!(
            body.vy.abs() < 0.3 && body.omega.abs() < 0.3,
            "body {i} must come to rest (vy {}, omega {})",
            body.vy,
            body.omega
        );
        assert!(
            (body.cm_y - late[i]).abs() < 0.25,
            "body {i} must stop jittering: cm_y {} -> {}",
            late[i],
            body.cm_y
        );
    }
    // The stack must keep its order: top over middle over bottom.
    let (_, _, _, top_bottom) = body_bounds(&bodies[0]);
    let (_, middle_top, _, _) = body_bounds(&bodies[1]);
    let (_, _, _, middle_bottom) = body_bounds(&bodies[1]);
    let (_, bottom_top, _, _) = body_bounds(&bodies[2]);
    assert!(
        top_bottom < middle_top,
        "the top body sank into the middle (top bottom {top_bottom}, middle top {middle_top})"
    );
    assert!(
        middle_bottom < bottom_top,
        "the middle body sank into the bottom (middle bottom {middle_bottom}, bottom top {bottom_top})"
    );
}

/// Two bodies resting on each other must settle without sharing a cell.
///
/// This is the two-body half of the three-body stack: the lower body publishes
/// its upward support, the upper body's pair contact stops its pose at the
/// lower one's rasterised face, and the claim grid keeps the two pixel sets
/// disjoint. Without the positional half of the pair contact the upper body
/// ends the frame rounded into the lower one and the rasteriser has to displace
/// a pixel that has nowhere to go.
#[test]
fn test_rigid_body_pair_stack_rests_without_sharing_cells() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(200, 300);
    let ycells = engine.config().yres.div_euclid(CELL);
    let xcells = engine.config().xres.div_euclid(CELL);
    let cells: Vec<(i32, i32)> = (0..xcells as i32)
        .map(|cx| (cx, ycells as i32 - 1))
        .collect();
    engine.apply_command(SimCommand::PlaceWalls {
        cells,
        wall_id: WL_WALL,
        fan_vel: None,
    });
    let wall_top = (ycells as i32 - 1) * 4;

    let lower_id = engine.spawn_rigid_body(100.0, 289.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    let upper_id = engine.spawn_rigid_body(100.0, 282.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    let ids = [upper_id, lower_id];

    let mut shared_cells = 0usize;
    let mut worst_bottom = i32::MIN;
    let mut late = [0.0f32; 2];
    for tick in 0..150 {
        step(&mut engine);
        engine.sync_rigid_bodies_to_host();
        let bodies: Vec<_> = ids
            .iter()
            .map(|id| {
                engine
                    .bufs
                    .rigid_bodies
                    .iter()
                    .find(|b| b.id == *id)
                    .cloned()
                    .expect("the stacked body must survive")
            })
            .collect();
        let upper: std::collections::HashSet<(i32, i32)> = bodies[0]
            .pixels
            .iter()
            .map(|p| (p.grid_x, p.grid_y))
            .collect();
        let lower: std::collections::HashSet<(i32, i32)> = bodies[1]
            .pixels
            .iter()
            .map(|p| (p.grid_x, p.grid_y))
            .collect();
        shared_cells += upper.intersection(&lower).count();
        worst_bottom = worst_bottom.max(body_bounds(&bodies[1]).3);
        if tick == 110 {
            for (i, body) in bodies.iter().enumerate() {
                late[i] = body.cm_y;
            }
        }
    }

    assert_eq!(shared_cells, 0, "the pair shared cells while settling");
    assert!(
        worst_bottom < wall_top,
        "the lower body entered the wall band: worst bottom row {worst_bottom}, wall top {wall_top}"
    );
    let bodies: Vec<_> = ids
        .iter()
        .map(|id| {
            engine
                .bufs
                .rigid_bodies
                .iter()
                .find(|b| b.id == *id)
                .cloned()
                .expect("the stacked body must survive")
        })
        .collect();
    for (i, body) in bodies.iter().enumerate() {
        assert_eq!(body.pixels.len(), 9 * 7, "body {i} must stay welded");
        assert!(
            body.vy.abs() < 0.25 && body.omega.abs() < 0.25,
            "body {i} must come to rest (vy {}, omega {})",
            body.vy,
            body.omega
        );
        assert!(
            (body.cm_y - late[i]).abs() < 0.25,
            "body {i} must stop jittering: cm_y {} -> {}",
            late[i],
            body.cm_y
        );
    }
    let (_, _, _, upper_bottom) = body_bounds(&bodies[0]);
    let (_, lower_top, _, _) = body_bounds(&bodies[1]);
    assert!(
        upper_bottom < lower_top,
        "the upper body sank into the lower one (upper bottom {upper_bottom}, lower top {lower_top})"
    );
}

/// A spinning cluster must keep every pixel in a cell of its own.
///
/// A rotation maps lattice points to non-lattice positions, so two pixels can
/// round into the same cell. The claim grid has to catch that inside one body
/// too: the loser takes a free neighbour and the cluster stays welded instead
/// of two particles claiming one `pmap` word.
#[test]
fn test_rigid_body_rotation_keeps_cells_distinct() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(320, 200);
    engine.config_mut().gravity_mode = GravityMode::Off;
    let body_id = engine.spawn_rigid_body(160.0, 100.0, 6, 4, RigidSpawnShape::Box, PT_WOOD);
    engine.bufs.rigid_bodies[0].omega = 0.35;
    let pixel_count = engine.bufs.rigid_bodies[0].pixels.len();
    assert_eq!(pixel_count, 13 * 9);

    let mut worst_duplicates = 0usize;
    let mut max_radius = 0.0f32;
    for _ in 0..90 {
        step(&mut engine);
        engine.sync_rigid_bodies_to_host();
        let body = engine
            .bufs
            .rigid_bodies
            .iter()
            .find(|b| b.id == body_id)
            .cloned()
            .expect("the spinning body must survive");
        assert_eq!(
            body.pixels.len(),
            pixel_count,
            "a spin must not fracture the body"
        );
        let cells: std::collections::HashSet<(i32, i32)> =
            body.pixels.iter().map(|p| (p.grid_x, p.grid_y)).collect();
        worst_duplicates = worst_duplicates.max(body.pixels.len() - cells.len());
        let radius = body
            .pixels
            .iter()
            .map(|p| {
                let dx = p.grid_x as f32 - body.cm_x;
                let dy = p.grid_y as f32 - body.cm_y;
                (dx * dx + dy * dy).sqrt()
            })
            .fold(0.0f32, f32::max);
        max_radius = max_radius.max(radius);
    }
    assert_eq!(
        worst_duplicates, 0,
        "two pixels of the spinning cluster shared a cell"
    );
    assert!(
        max_radius < 9.5,
        "the rotating cluster must stay welded, max radius {max_radius}"
    );
}

/// Two equal bodies moving into each other must exchange momentum symmetrically.
///
/// A one-sided contact stopped the mover as if the other body were a wall and
/// never moved the receiver; with both bodies movable the pair's total
/// horizontal momentum has to stay at zero (up to the solver's damping) and
/// each body has to reverse away from the other.
#[test]
fn test_rigid_body_head_on_pair_conserves_momentum() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(400, 200);
    engine.config_mut().gravity_mode = GravityMode::Off;
    let a_id = engine.spawn_rigid_body(80.0, 100.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    let b_id = engine.spawn_rigid_body(320.0, 100.0, 4, 3, RigidSpawnShape::Box, PT_WOOD);
    engine.bufs.rigid_bodies[0].vx = 8.0;
    engine.bufs.rigid_bodies[1].vx = -8.0;
    let mass = engine.bufs.rigid_bodies[0].mass;
    assert_eq!(mass, engine.bufs.rigid_bodies[1].mass);

    let mut shared_cells = 0usize;
    let mut bounced = false;
    let mut impact_momentum = f32::NAN;
    for _ in 0..40 {
        step(&mut engine);
        engine.sync_rigid_bodies_to_host();
        let a = engine
            .bufs
            .rigid_bodies
            .iter()
            .find(|b| b.id == a_id)
            .cloned()
            .expect("the moving body must survive the impact");
        let b = engine
            .bufs
            .rigid_bodies
            .iter()
            .find(|b| b.id == b_id)
            .cloned()
            .expect("the moving body must survive the impact");
        let cells_a: std::collections::HashSet<(i32, i32)> =
            a.pixels.iter().map(|p| (p.grid_x, p.grid_y)).collect();
        let cells_b: std::collections::HashSet<(i32, i32)> =
            b.pixels.iter().map(|p| (p.grid_x, p.grid_y)).collect();
        shared_cells += cells_a.intersection(&cells_b).count();
        // The first frame both bodies have reversed is the impact frame; the
        // later frames include the canvas-edge bounces, which reverse them
        // again and would hide a one-sided contact.
        if !bounced && a.vx < 0.0 && b.vx > 0.0 {
            bounced = true;
            impact_momentum = mass * (a.vx + b.vx);
        }
    }

    let a = engine
        .bufs
        .rigid_bodies
        .iter()
        .find(|b| b.id == a_id)
        .cloned()
        .expect("the moving body must survive the impact");
    let b = engine
        .bufs
        .rigid_bodies
        .iter()
        .find(|b| b.id == b_id)
        .cloned()
        .expect("the moving body must survive the impact");
    assert_eq!(shared_cells, 0, "the two bodies shared a cell");
    assert_eq!(a.pixels.len(), 9 * 7, "the left body must stay welded");
    assert_eq!(b.pixels.len(), 9 * 7, "the right body must stay welded");
    assert!(
        a.cm_x < b.cm_x,
        "the bodies passed through each other (cm_x {} vs {})",
        a.cm_x,
        b.cm_x
    );
    assert!(
        bounced,
        "both bodies must bounce back instead of one stopping as if the other were a wall"
    );
    assert!(
        impact_momentum.abs() < 0.05 * mass * 16.0,
        "the pair must cancel its momentum: {impact_momentum} left of {}",
        mass * 16.0
    );
}

/// Newtonian gravity accelerates a rigid body.
///
/// `GetGravityField` adds `newtonGrav * force[cell]` on top of the mode's own
/// field (`Simulation.cpp:2120-2124`). Rigid pixels never reach the movement
/// pass's velocity update, so the rigid solver has to sample that field itself;
/// before it did, a rigid body ignored Newtonian gravity entirely.
#[test]
fn test_rigid_body_follows_newtonian_gravity() {
    let _guard = gpu_lock();

    // A heavy tool well to the right of the body (`PGRV` writes `gravIn.mass`),
    // re-armed every frame so the field persists for the whole run.
    let xcells = 32usize;
    let well = |engine: &mut SimulationEngine| {
        let mut mass = vec![0.0f32; engine.bufs.grav_tool_mass.len()];
        mass[16 * xcells + 24] = 400.0;
        engine.bufs.grav_tool_mass = mass;
        engine.bufs.grav_tool_mass_dirty = true;
    };
    // The spawn itself initialises the body through the host solver, so the
    // first tick proves nothing; three ticks leave the device path in charge.
    let run = |newton: bool| {
        let mut engine = create_test_engine(128, 128);
        assert_eq!(engine.bufs.config.xcells() as usize, xcells);
        engine.bufs.config.newton_grav = newton;
        engine.spawn_rigid_body(30.0, 64.0, 4, 4, RigidSpawnShape::Box, PT_WOOD);
        for _ in 0..3 {
            well(&mut engine);
            step(&mut engine);
        }
        engine.sync_rigid_bodies_to_host();
        engine.bufs.rigid_bodies[0].vx
    };

    let control_vx = run(false);
    assert_eq!(
        control_vx, 0.0f32,
        "without Newtonian gravity the body must not drift sideways"
    );

    // The well sits to the right, so the field pulls the body towards +x.
    let vx = run(true);
    assert!(
        vx > 0.1f32,
        "the body must accelerate towards the mass (vx {vx})"
    );
}
