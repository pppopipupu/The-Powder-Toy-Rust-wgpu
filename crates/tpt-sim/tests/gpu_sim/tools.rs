//! `SimToolKind` ports of `src/simulation/simtools/*.cpp`.
//!
//! Every expected number below is the literal arithmetic of the reference
//! file quoted next to it. The edits run on the device against the live
//! simulation (`crate::tool_kernel`), so [`apply_tool`] reads the result back
//! before the assertions inspect the host shadow; the simulation is never
//! stepped.

use super::*;

/// Applies one tool over `points` with the reference brush centre at `center`.
fn apply_tool(
    engine: &mut SimulationEngine,
    points: Vec<(i32, i32)>,
    tool: SimToolKind,
    strength: f32,
    dir: (f32, f32),
    center: (i32, i32),
) {
    apply_tool_with_drag(engine, points, tool, strength, dir, center, false);
}

/// [`apply_tool`] with TPT's `dragging` flag: `false` is the click stamp and the
/// line/rect gesture (`PerformDrag`), `true` the held drag path.
fn apply_tool_with_drag(
    engine: &mut SimulationEngine,
    points: Vec<(i32, i32)>,
    tool: SimToolKind,
    strength: f32,
    dir: (f32, f32),
    center: (i32, i32),
    dragging: bool,
) {
    engine.apply_command(SimCommand::ApplySimTool {
        points,
        tool,
        strength,
        dir,
        center,
        dragging,
    });
    // The tool kernels run on the device; refresh the host shadow so the
    // assertions read what they just did.
    engine.bufs.sync_gpu_to_host();
}

/// `RNG::next` (`tpt-rand.cpp:12-23`, xoroshiro128+). The `MIX` tests need to
/// predict the two offset rolls, so they carry a copy of the generator.
fn next_u64(state: &mut [u32; 4]) -> u64 {
    let s0 = u64::from(state[0]) | (u64::from(state[1]) << 32);
    let s1 = u64::from(state[2]) | (u64::from(state[3]) << 32);
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

/// `sim.randomSeed` word layout of a seed: `[s0.lo, s0.hi, s1.lo, s1.hi]`.
fn seed_state(seed: u32) -> [u32; 4] {
    [seed, 0x9abc_def0, 0x0fed_cba9, 0x8765_4321]
}

/// Finds a seed whose `MIX.cpp` rolls fire on the pixel (`rng() % 100 == 0`)
/// and pick `offset` out of the `distance = 2` window (`rng() % 2 - 1`).
fn mix_seed(offset: (i32, i32)) -> [u32; 4] {
    for seed in 0u32..2_000_000 {
        let mut state = seed_state(seed);
        if !(next_u64(&mut state) as u32).is_multiple_of(100) {
            continue;
        }
        let dx = (next_u64(&mut state) as u32 % 2) as i32 - 1;
        let dy = (next_u64(&mut state) as u32 % 2) as i32 - 1;
        if (dx, dy) == offset {
            return seed_state(seed);
        }
    }
    panic!("no seed produces the MIX offset {offset:?}");
}

/// `AIR.cpp:17-23` / `VAC.cpp:17-23`: `pv += ±strength * 0.05` per brush pixel.
#[test]
fn test_air_and_vacuum_pressure_step() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);

    // Four brush pixels, two of them inside cell (0, 0).
    let cell_a = engine.bufs.cell_index(0, 0).unwrap();
    apply_tool(
        &mut engine,
        vec![(0, 0), (1, 0)],
        SimToolKind::Air,
        1.0,
        (0.0, 0.0),
        (0, 0),
    );
    assert!(
        (engine.bufs.air_pv[cell_a] - 0.1).abs() < 1e-6,
        "AIR must add strength * 0.05 per pixel: {}",
        engine.bufs.air_pv[cell_a]
    );

    // `VAC` is the same subtraction on another cell: 2 * 0.05 * 2 = 0.2.
    let cell_b = engine.bufs.cell_index(4, 0).unwrap();
    apply_tool(
        &mut engine,
        vec![(4, 0), (5, 0)],
        SimToolKind::Vac,
        2.0,
        (0.0, 0.0),
        (4, 0),
    );
    assert!(
        (engine.bufs.air_pv[cell_b] + 0.2).abs() < 1e-6,
        "VAC must subtract strength * 0.05 per pixel: {}",
        engine.bufs.air_pv[cell_b]
    );

    // Both clamp to MIN/MAX_PRESSURE; one pixel at strength 10000 would be 500.
    let cell_c = engine.bufs.cell_index(8, 0).unwrap();
    apply_tool(
        &mut engine,
        vec![(8, 0)],
        SimToolKind::Air,
        10_000.0,
        (0.0, 0.0),
        (8, 0),
    );
    assert_eq!(engine.bufs.air_pv[cell_c], MAX_PRESSURE);
    let cell_d = engine.bufs.cell_index(12, 0).unwrap();
    apply_tool(
        &mut engine,
        vec![(12, 0)],
        SimToolKind::Vac,
        10_000.0,
        (0.0, 0.0),
        (12, 0),
    );
    assert_eq!(engine.bufs.air_pv[cell_d], MIN_PRESSURE);
}

/// `HEAT.cpp:16-28` (and `COOL.cpp`): only the particle under the cursor moves,
/// `+= strength * 2.0` (`* 0.1` for `PUMP`/`GPMP`), clamped `0 ..= MAX_TEMP`.
#[test]
fn test_heat_touches_only_the_particle() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);

    fill_rect(&mut engine, 10, 10, 10, 10, PT_SAND);
    let sand = engine.bufs.particle_index_at(10, 10).expect("sand missing");
    engine.bufs.part_temp[sand] = 300.0;
    engine.bufs.mark_particle_slot_dirty(sand);
    // The tool reads the live device value, so push the host write first.
    engine.bufs.sync_host_to_gpu();
    let cell = engine.bufs.cell_index(10, 10).unwrap();
    let ambient_before = engine.bufs.air_hv[cell];

    apply_tool(
        &mut engine,
        vec![(10, 10)],
        SimToolKind::Heat,
        1.0,
        (0.0, 0.0),
        (10, 10),
    );

    assert!(
        (engine.bufs.part_temp[sand] - 302.0).abs() < 1e-4,
        "HEAT must add 2.0 K: {}",
        engine.bufs.part_temp[sand]
    );
    assert_eq!(
        engine.bufs.air_hv[cell], ambient_before,
        "HEAT must not write the air heat field"
    );

    // `HEAT.cpp:18-21`: a pump only moves a tenth of that.
    fill_rect(&mut engine, 30, 30, 30, 30, PT_PUMP);
    let pump = engine.bufs.particle_index_at(30, 30).expect("pump missing");
    engine.bufs.part_temp[pump] = 300.0;
    engine.bufs.mark_particle_slot_dirty(pump);
    engine.bufs.sync_host_to_gpu();
    apply_tool(
        &mut engine,
        vec![(30, 30)],
        SimToolKind::Heat,
        1.0,
        (0.0, 0.0),
        (30, 30),
    );
    assert!(
        (engine.bufs.part_temp[pump] - 300.1).abs() < 1e-4,
        "HEAT must add 0.1 K to PUMP: {}",
        engine.bufs.part_temp[pump]
    );

    // `HEAT.cpp:16-17`: an empty pixel is a no-op (and `COOL` mirrors it).
    let empty = engine.bufs.cell_index(50, 50).unwrap();
    let before = engine.bufs.air_hv[empty];
    apply_tool(
        &mut engine,
        vec![(50, 50)],
        SimToolKind::Cool,
        1.0,
        (0.0, 0.0),
        (50, 50),
    );
    assert_eq!(engine.bufs.air_hv[empty], before);
}

/// `AMBP.cpp:16-24` / `AMBM.cpp:16-24`: the cell's ambient heat field moves by
/// `±strength * 2.0`, only when `aheat_enable`; the global ambient temperature
/// is untouched.
#[test]
fn test_ambient_heat_paints_the_cell_field() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().aheat_enable = true;

    let cell = engine.bufs.cell_index(10, 10).unwrap();
    engine.bufs.air_hv[cell] = 300.0;
    engine.bufs.mark_air_dirty();
    engine.bufs.sync_host_to_gpu();
    let ambient_before = engine.config().ambient_air_temp;

    apply_tool(
        &mut engine,
        vec![(10, 10)],
        SimToolKind::AmbHeatPlus,
        1.0,
        (0.0, 0.0),
        (10, 10),
    );
    assert!(
        (engine.bufs.air_hv[cell] - 302.0).abs() < 1e-4,
        "AMBP must add strength * 2.0: {}",
        engine.bufs.air_hv[cell]
    );

    apply_tool(
        &mut engine,
        vec![(10, 10)],
        SimToolKind::AmbHeatMinus,
        2.0,
        (0.0, 0.0),
        (10, 10),
    );
    assert!(
        (engine.bufs.air_hv[cell] - 298.0).abs() < 1e-4,
        "AMBM must subtract strength * 2.0: {}",
        engine.bufs.air_hv[cell]
    );
    assert_eq!(
        engine.config().ambient_air_temp,
        ambient_before,
        "the ambient tools must not move the global ambient temperature"
    );

    // `AMBP.cpp:16-18`: both tools are gated on ambient heat being enabled.
    engine.config_mut().aheat_enable = false;
    engine.bufs.air_hv[cell] = 300.0;
    engine.bufs.mark_air_dirty();
    engine.bufs.sync_host_to_gpu();
    apply_tool(
        &mut engine,
        vec![(10, 10)],
        SimToolKind::AmbHeatPlus,
        1.0,
        (0.0, 0.0),
        (10, 10),
    );
    assert_eq!(
        engine.bufs.air_hv[cell], 300.0,
        "AMBP must do nothing while aheat is disabled"
    );
}

/// `CYCL.cpp:24-51`: a `strength / 16` tangential kick around the brush centre,
/// gated on cell-aligned pixels and skipping the centre itself.
#[test]
fn test_cyclone_rotates_about_the_brush_centre() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);

    // Centre (8, 4), pixel (4, 4): dv = (4, 0), |dv| = 4, strength 16.
    // `vx -= (16/16) * (-0) * 0.25` = 0, `vy -= (16/16) * 4 * 0.25` = -1.
    let cell = engine.bufs.cell_index(4, 4).unwrap();
    apply_tool(
        &mut engine,
        vec![(4, 4)],
        SimToolKind::Cycl,
        16.0,
        (0.0, 0.0),
        (8, 4),
    );
    assert!(
        engine.bufs.air_vx[cell].abs() < 1e-6,
        "CYCL must not push along the radius: {}",
        engine.bufs.air_vx[cell]
    );
    assert!(
        (engine.bufs.air_vy[cell] + 1.0).abs() < 1e-6,
        "CYCL must kick vy by -(strength/16) * dvx / |dv|: {}",
        engine.bufs.air_vy[cell]
    );

    // Pixel (4, 8): dv = (4, -4), |dv| = sqrt(32). Both components gain
    // `-1 / sqrt(2)` because `vx -= s * (-dvy) / |dv|` and `vy -= s * dvx / |dv|`.
    let cell_diag = engine.bufs.cell_index(4, 8).unwrap();
    apply_tool(
        &mut engine,
        vec![(4, 8)],
        SimToolKind::Cycl,
        16.0,
        (0.0, 0.0),
        (8, 4),
    );
    let diag = -1.0 / 2.0f32.sqrt();
    assert!(
        (engine.bufs.air_vx[cell_diag] - diag).abs() < 1e-6,
        "CYCL vx = {diag}: {}",
        engine.bufs.air_vx[cell_diag]
    );
    assert!(
        (engine.bufs.air_vy[cell_diag] - diag).abs() < 1e-6,
        "CYCL vy = {diag}: {}",
        engine.bufs.air_vy[cell_diag]
    );

    // `CYCL.cpp:24`: only cell-aligned pixels fire. (13, 5) is inside cell
    // (3, 1), which nothing above touched.
    let unaligned = engine.bufs.cell_index(13, 5).unwrap();
    apply_tool(
        &mut engine,
        vec![(13, 5)],
        SimToolKind::Cycl,
        16.0,
        (0.0, 0.0),
        (8, 4),
    );
    assert_eq!(
        (engine.bufs.air_vx[unaligned], engine.bufs.air_vy[unaligned]),
        (0.0, 0.0),
        "CYCL must only run once per cell, on the cell origin"
    );

    // `CYCL.cpp:26-27`: the brush centre is skipped (invsqr would be infinite).
    let centre_cell = engine.bufs.cell_index(8, 4).unwrap();
    apply_tool(
        &mut engine,
        vec![(8, 4)],
        SimToolKind::Cycl,
        16.0,
        (0.0, 0.0),
        (8, 4),
    );
    assert_eq!(
        (
            engine.bufs.air_vx[centre_cell],
            engine.bufs.air_vy[centre_cell]
        ),
        (0.0, 0.0),
        "CYCL must skip the brush centre"
    );
}

/// `WIND.cpp:20-29`: `vx/vy += delta * 0.002 * strength` for every brush pixel.
#[test]
fn test_wind_scales_the_stroke_delta() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);

    let cell = engine.bufs.cell_index(0, 0).unwrap();
    apply_tool(
        &mut engine,
        vec![(0, 0), (1, 0)],
        SimToolKind::Wind,
        1.0,
        (3.0, -4.0),
        (0, 0),
    );
    // Two pixels: 2 * 3 * 0.002 = 0.012 and 2 * -4 * 0.002 = -0.016.
    assert!(
        (engine.bufs.air_vx[cell] - 0.012).abs() < 1e-6,
        "WIND vx: {}",
        engine.bufs.air_vx[cell]
    );
    assert!(
        (engine.bufs.air_vy[cell] + 0.016).abs() < 1e-6,
        "WIND vy: {}",
        engine.bufs.air_vy[cell]
    );

    // `WIND.cpp:20`: the held drag path uses `0.01`, five times the gesture.
    apply_tool_with_drag(
        &mut engine,
        vec![(0, 0)],
        SimToolKind::Wind,
        1.0,
        (1.0, 0.0),
        (0, 0),
        true,
    );
    assert!(
        (engine.bufs.air_vx[cell] - 0.022).abs() < 1e-6,
        "held WIND must add delta * 0.01: {}",
        engine.bufs.air_vx[cell]
    );
}

/// `MIX.cpp:21-57`: a 1-in-100 roll swaps the cursor particle with a particle of
/// the same `STATE_FLAGS` within `(int)(sqrt(strength) * 10)` pixels.
#[test]
fn test_mix_swaps_same_state_particles() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;

    fill_rect(&mut engine, 9, 10, 9, 10, PT_SAND);
    fill_rect(&mut engine, 10, 10, 10, 10, PT_SAND);

    let this_pixel = engine.bufs.pixel_index(10, 10).unwrap();
    let that_pixel = engine.bufs.pixel_index(9, 10).unwrap();
    let this_id = pmap_id(engine.bufs.pmap[this_pixel]).expect("no particle at 10,10");
    let that_id = pmap_id(engine.bufs.pmap[that_pixel]).expect("no particle at 9,10");
    assert_ne!(this_id, that_id);

    // `MIX.cpp:28`: `(int)(sqrt(0.05) * 10) = 2`, so `rng() % 2 - 1` picks
    // offsets in `{-1, 0}`. Seed the rolls to land on (9, 10).
    engine.rng_state = mix_seed((-1, 0));

    apply_tool(
        &mut engine,
        vec![(10, 10)],
        SimToolKind::Mix,
        0.05,
        (0.0, 0.0),
        (10, 10),
    );

    // `MIX.cpp:49-56`: both map entries and both positions are exchanged.
    assert_eq!(pmap_id(engine.bufs.pmap[this_pixel]), Some(that_id));
    assert_eq!(pmap_id(engine.bufs.pmap[that_pixel]), Some(this_id));
    assert_eq!(
        (
            engine.bufs.part_x[this_id as usize],
            engine.bufs.part_y[this_id as usize]
        ),
        (9.0, 10.0),
        "the cursor particle must move to the partner's pixel"
    );
    assert_eq!(
        (
            engine.bufs.part_x[that_id as usize],
            engine.bufs.part_y[that_id as usize]
        ),
        (10.0, 10.0),
        "the partner must move to the cursor's pixel"
    );

    // `MIX.cpp:46-47`: identical `STATE_FLAGS` are required; `BRCK` is
    // `TYPE_SOLID` where `SAND` is `TYPE_PART`, so the same roll is a no-op.
    fill_rect(&mut engine, 19, 10, 19, 10, PT_BRCK);
    fill_rect(&mut engine, 20, 10, 20, 10, PT_SAND);
    let solid_pixel = engine.bufs.pixel_index(19, 10).unwrap();
    let solid_id = pmap_id(engine.bufs.pmap[solid_pixel]).expect("no brick");
    let sand_pixel = engine.bufs.pixel_index(20, 10).unwrap();
    let sand_id = pmap_id(engine.bufs.pmap[sand_pixel]).expect("no sand");
    engine.rng_state = mix_seed((-1, 0));

    apply_tool(
        &mut engine,
        vec![(20, 10)],
        SimToolKind::Mix,
        0.05,
        (0.0, 0.0),
        (20, 10),
    );

    assert_eq!(pmap_id(engine.bufs.pmap[sand_pixel]), Some(sand_id));
    assert_eq!(pmap_id(engine.bufs.pmap[solid_pixel]), Some(solid_id));
}
