use super::*;

#[test]
fn test_variable_canvas_sizes_and_unbounded_particles() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(320, 200);

    assert_eq!(engine.config().xres, 320);
    assert_eq!(engine.config().yres, 200);
    let initial_capacity = engine.bufs.capacity;
    assert_eq!(
        initial_capacity,
        tpt_sim::SimBuffers::initial_capacity(engine.config()),
        "a fresh canvas starts with the documented pool size"
    );
    assert_eq!(initial_capacity, 320 * 200 * 2);

    fill_rect(&mut engine, 40, 40, 79, 59, PT_SAND);
    fill_rect(&mut engine, 100, 40, 139, 59, PT_WATR);
    assert_eq!(engine.particle_count(), 2 * 40 * 20);

    for _ in 0..3 {
        step(&mut engine);
    }
    assert_eq!(
        engine.particle_count(),
        1600,
        "nothing may vanish on a 320x200 canvas"
    );

    let (len, non_black) = render_stats(&mut engine);
    assert_eq!(len, 320 * 200 * 4, "RGBA buffer must match the canvas size");
    assert!(
        non_black > 0,
        "a populated canvas must not render as pure black"
    );

    engine.resize_canvas(800, 600);
    assert_eq!(engine.config().xres, 800);
    assert_eq!(engine.config().yres, 600);
    assert_eq!(engine.particle_count(), 0, "resizing clears the simulation");
    assert_eq!(
        engine.bufs.capacity,
        tpt_sim::SimBuffers::initial_capacity(engine.config()),
        "resizing reallocates the pool for the new pixel count"
    );
    assert_eq!(engine.bufs.capacity, 800 * 600 * 2);

    let (len, _) = render_stats(&mut engine);
    assert_eq!(
        len,
        800 * 600 * 4,
        "the resized canvas renders at its own size"
    );

    fill_rect(&mut engine, 200, 200, 299, 259, PT_SAND);
    assert_eq!(engine.particle_count(), 100 * 60);
    for _ in 0..3 {
        step(&mut engine);
    }
    assert_eq!(
        engine.particle_count(),
        6000,
        "the resized canvas simulates normally"
    );

    // A save restore (or Lua `elem.allocate`) may hand out slots past the end
    // of the current pool; `import_particle` must grow the pool instead of
    // dropping the particle.
    let capacity_before_growth = engine.bufs.capacity;
    let first_high_slot = capacity_before_growth;
    let extra = 64usize;
    for k in 0..extra {
        let particle = GpuParticle {
            ty: PT_STNE,
            x: 400.0 + k as f32,
            y: 100.0,
            temp: DEFAULT_TEMP_K,
            ..Default::default()
        };
        engine.bufs.import_particle(first_high_slot + k, &particle);
    }
    assert!(
        engine.bufs.capacity > capacity_before_growth,
        "the pool must grow past its initial capacity ({} slots)",
        capacity_before_growth
    );
    assert_eq!(
        engine.bufs.capacity,
        capacity_before_growth * 2,
        "the pool doubles when it runs out of slots"
    );
    assert_eq!(engine.particle_count(), 6000 + extra);

    engine.bufs.rebuild_spatial_maps();
    engine.bufs.sync_host_to_gpu();
    for _ in 0..2 {
        step(&mut engine);
    }
    assert_eq!(
        engine.particle_count(),
        6000 + extra,
        "particles above the original capacity must survive reallocation and stepping"
    );
    assert_eq!(
        count_type(&engine, PT_STNE),
        extra,
        "the imported high slots are simulated like any other particle"
    );
}

#[test]
fn test_powder_and_liquid_lockfree_movement() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(320, 200);

    fill_rect(&mut engine, 100, 20, 104, 39, PT_SAND);
    fill_rect(&mut engine, 200, 20, 204, 39, PT_WATR);

    let sand = count_type(&engine, PT_SAND);
    let water = count_type(&engine, PT_WATR);
    assert_eq!(sand, 5 * 20, "sand brush stamp");
    assert_eq!(water, 5 * 20, "water brush stamp");
    assert_eq!(engine.particle_count(), sand + water);

    let sand_y0 = mean_y(&engine, PT_SAND);
    let water_y0 = mean_y(&engine, PT_WATR);

    for tick in 0..15 {
        step(&mut engine);
        assert_eq!(
            engine.particle_count(),
            sand + water,
            "particle count changed on tick {tick} (lost or duplicated particles)"
        );
        assert_eq!(
            count_type(&engine, PT_SAND),
            sand,
            "sand disappeared on tick {tick}"
        );
        assert_eq!(
            count_type(&engine, PT_WATR),
            water,
            "water disappeared on tick {tick}"
        );
    }

    let sand_y1 = mean_y(&engine, PT_SAND);
    let water_y1 = mean_y(&engine, PT_WATR);
    assert!(
        sand_y1 > sand_y0 + 2.0,
        "sand must fall under gravity: mean y {sand_y0} -> {sand_y1}"
    );
    assert!(
        water_y1 > water_y0 + 2.0,
        "water must fall under gravity: mean y {water_y0} -> {water_y1}"
    );

    // Every live particle owns exactly one spatial-map entry: no duplicates.
    let mut mapped = 0usize;
    for id in 0..engine.bufs.capacity {
        let ty = engine.bufs.part_ty[id];
        if ty == PT_NONE {
            continue;
        }
        let x = engine.bufs.part_x[id].round() as i32;
        let y = engine.bufs.part_y[id].round() as i32;
        let Some(pixel) = engine.bufs.pixel_index(x, y) else {
            panic!("particle {id} escaped the canvas at ({x}, {y})");
        };
        let mapped_id = pmap_id(engine.bufs.pmap[pixel])
            .unwrap_or_else(|| panic!("pixel ({x}, {y}) has no owner, particle {id} was lost"));
        assert_eq!(
            mapped_id as usize, id,
            "pixel ({x}, {y}) is not owned by particle {id}"
        );
        mapped += 1;
    }
    assert_eq!(
        mapped,
        sand + water,
        "spatial map must cover every live particle"
    );
}

/// Paints one grain per tick on a running simulation.
///
/// Before the slot-scoped upload, `place_particles` marked the whole host SoA
/// dirty and the next step re-uploaded it from the mouse-down snapshot, so every
/// painted frame rolled the device pool back to the click state: the grains
/// never fell and the painted slots were overwritten by the pre-stroke values.
#[test]
fn test_painting_while_stepping_does_not_rewind_particles() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(320, 200);

    // Baseline before the stroke: an empty pool. A stale full upload would keep
    // restoring exactly this (zero particles) every frame.
    let before = engine.particle_count();
    assert_eq!(before, 0, "fresh engine starts empty");

    let ticks = 20;
    for tick in 0..ticks {
        // One grain per tick, at a fixed row but a distinct column so the
        // non-replace brush never hits a cached occupant.
        let x = 140 + tick as i32;
        engine.apply_command(SimCommand::PlaceParticles {
            points: vec![(x, 20)],
            elem_ty: PT_SAND,
            replace_mode: false,
            dcolour: 0,
        });
        // No read-back between paint and step: this is the interactive path.
        engine.step();
    }

    engine.ensure_host_current();

    let painted = count_type(&engine, PT_SAND);
    assert_eq!(
        painted, ticks,
        "every painted grain must survive the following steps (got {painted})"
    );
    assert_ne!(
        engine.particle_count(),
        before,
        "the particle pool must not be reset to its pre-stroke state"
    );
    let mean = mean_y(&engine, PT_SAND);
    assert!(
        mean > 22.0,
        "painted grains must keep falling while the stroke continues: mean y {mean} (placed at 20)"
    );
}

/// A 2048x2048 canvas must stay under the 65 535 workgroup-per-dimension cap.
///
/// The movement pass used 64-wide cubes, so 2048^2 pixels dispatched 65 536
/// groups and wgpu aborted the process; the pool was also sized to 2^23 slots,
/// one past the 23-bit `pmap` id field. A 2048^2 canvas saturates the pool at
/// `MAX_PARTICLE_CAPACITY`, so every slot-parallel pass (movement, reactions,
/// rendering) dispatches from that count; the render pass originally used
/// 128-wide cubes, which is `65 536` slots-cubes and a device-validation panic
/// that only the interactive path reached, because headless smoke never
/// rendered at that size.
#[test]
fn test_2048_canvas_step_stays_within_dispatch_limits() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(2048, 2048);
    assert_eq!(
        engine.bufs.capacity,
        tpt_sim::SimBuffers::MAX_PARTICLE_CAPACITY,
        "a 2048x2048 canvas must saturate the pool so slot-parallel passes are covered"
    );

    fill_rect(&mut engine, 1000, 1000, 1039, 1039, PT_SAND);
    step(&mut engine);
    assert!(
        count_type(&engine, PT_SAND) > 0,
        "painted sand vanished after one 2048x2048 tick"
    );

    // The GPU render passes are slot- and pixel-parallel over the same pool and
    // canvas, so they must survive the identical dispatch budget. `render_rgba`
    // runs the whole fancy-renderer chain and reads the image back, so a
    // validation fault inside any render dispatch aborts here.
    let rgba = engine.render_rgba();
    let pixels = (2048 * 2048) as usize;
    assert_eq!(
        rgba.len(),
        pixels * 4,
        "the 2048x2048 framebuffer has the wrong size"
    );
    let mut painted = 0usize;
    for chunk in rgba.as_chunks::<4>().0.iter() {
        if chunk[3] != 0 {
            painted += 1;
        }
    }
    assert!(
        painted > 0,
        "the 2048x2048 render produced a fully transparent framebuffer"
    );
}

#[test]
fn test_gravity_tools_feed_the_newtonian_mass_grid() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(128, 128);
    engine.config_mut().newton_grav = true;
    // Isolate the tool well: no uniform gravity, no air motion.
    engine.config_mut().gravity_mode = GravityMode::Off;

    let sand_velocity = |engine: &SimulationEngine| {
        let id = (0..engine.bufs.capacity)
            .find(|&i| engine.bufs.part_ty[i] == PT_SAND)
            .expect("sand particle missing");
        (engine.bufs.part_vx[id], engine.bufs.part_vy[id])
    };

    // `PGRV.cpp`: `gravIn.mass[Vec2{x, y} / CELL] = strength * 5.0f`. The well
    // sits two cells to the bottom-right of the particle, so attraction must
    // produce a positive x/y velocity.
    fill_rect(&mut engine, 64, 64, 64, 64, PT_SAND);
    engine.apply_command(SimCommand::ApplySimTool {
        points: vec![(72, 72)],
        tool: SimToolKind::PosGrav,
        strength: 1.0,
        dir: (0.0, 0.0),
        center: (0, 0),
        dragging: true,
    });
    assert!(
        engine.bufs.grav_tool_mass_dirty,
        "the tool must queue a mass-grid upload"
    );
    let cell = (18 * engine.config().xcells() + 18) as usize;
    assert_eq!(
        engine.bufs.grav_tool_mass[cell], 5.0,
        "PGRV must write strength * 5 into the coarse cell"
    );

    step(&mut engine);
    let (vx, vy) = sand_velocity(&engine);
    assert!(
        vx > 0.05 && vy > 0.05,
        "PGRV must pull the particle towards the well: v = ({vx}, {vy})"
    );
    assert!(
        !engine.bufs.grav_tool_mass_dirty,
        "the mass grid must be consumed by the gravity step"
    );

    // The well lasts exactly one solve, like TPT's post-dispatch clear: the
    // second step must not add any acceleration on top of the first.
    let (vx1, vy1) = (vx, vy);
    step(&mut engine);
    let (vx2, vy2) = sand_velocity(&engine);
    assert!(
        vx2 <= vx1 + 1e-3 && vy2 <= vy1 + 1e-3,
        "the well must not persist past one solve: v {vx1}/{vy1} -> {vx2}/{vy2}"
    );

    // `NGRV.cpp` negates the mass, so the same geometry pushes away.
    engine.apply_command(SimCommand::ClearAll);
    fill_rect(&mut engine, 64, 64, 64, 64, PT_SAND);
    engine.apply_command(SimCommand::ApplySimTool {
        points: vec![(72, 72)],
        tool: SimToolKind::NegGrav,
        strength: 1.0,
        dir: (0.0, 0.0),
        center: (0, 0),
        dragging: true,
    });
    assert_eq!(
        engine.bufs.grav_tool_mass[cell], -5.0,
        "NGRV must write the negated mass"
    );
    step(&mut engine);
    let (vx, vy) = sand_velocity(&engine);
    assert!(
        vx < -0.05 && vy < -0.05,
        "NGRV must push the particle away from the well: v = ({vx}, {vy})"
    );
}

/// A `FRAY` beam longer than the default 10 pixels still reaches its target.
///
/// The gather is bounded by the frame's longest emitter beam
/// (`fray_range_kernel`), so a `tmp` above 10 must widen the walk instead of
/// being cut at the old fixed window.
#[test]
fn test_fray_beam_beyond_the_default_window() {
    let _guard = gpu_lock();

    let run = |tmp: i64| {
        let mut engine = create_test_engine(96, 48);
        // Emitter at (32, 24) with a spark at (33, 24): the beam travels left.
        fill_rect(&mut engine, 32, 24, 32, 24, PT_FRAY);
        spark_at(&mut engine, 33, 24);
        // Target 20 pixels to the left, well past the 11-pixel default beam.
        fill_rect(&mut engine, 12, 24, 12, 24, PT_DUST);
        if tmp > 0 {
            // `FRAY.cpp:55-57`: a positive `tmp` is the beam length.
            engine.apply_command(SimCommand::SetParticleFieldAt {
                points: vec![(32, 24)],
                field: ParticleField::Tmp,
                value: tmp,
            });
        }
        step(&mut engine);
        // The dust is the only movable particle; it may have fallen one row,
        // so read the smallest x velocity among the live slots.
        engine
            .bufs
            .part_vx
            .iter()
            .zip(&engine.bufs.part_ty)
            .filter(|(_, ty)| **ty != PT_NONE)
            .map(|(vx, _)| *vx)
            .fold(0.0f32, f32::min)
    };

    assert_eq!(
        run(0),
        0.0f32,
        "a default 10-pixel beam must not reach a pixel 20 cells away"
    );
    let vx = run(20);
    assert!(
        vx < -0.5f32,
        "a 20-pixel beam pushes the target away from the emitter (vx {vx})"
    );
}
