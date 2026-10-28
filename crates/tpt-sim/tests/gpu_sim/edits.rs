use super::*;

/// The decoration and property brushes must edit the device world directly.
///
/// Regression guard: both tools used to pull the whole particle SoA back to the
/// host (`sync_particles_gpu_to_host`) before every stroke segment, edit the
/// host shadow and upload the touched slots again. That copy ran once per
/// frame while dragging a brush and rewound the rest of the pool to pre-step
/// values. The kernels now do the read-modify-write in place, so a stroke
/// applied after a tick must land on the device state.
#[test]
fn test_brush_edits_write_the_device_world() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    // `BRCK` is a `TYPE_SOLID` that never falls, so the row stays at y = 20
    // across the tick and the assertions address the pixels they painted.
    fill_rect(&mut engine, 10, 20, 29, 20, PT_BRCK);
    // The tick makes the device authoritative; the host shadow is now stale.
    engine.step();

    let points: Vec<(i32, i32)> = (10..30).map(|x| (x, 20)).collect();
    engine.apply_command(SimCommand::DecorateAt {
        points: points.clone(),
        colour: 0xFF11_2233,
        mode: DecoMode::Draw,
    });
    engine.apply_command(SimCommand::SetParticleFieldAt {
        points: points.clone(),
        field: ParticleField::Tmp,
        value: 77,
    });

    engine.ensure_host_current();
    for &(x, y) in &points {
        let p = engine
            .bufs
            .particle_at(x, y)
            .unwrap_or_else(|| panic!("no particle at ({x}, {y})"));
        assert_eq!(
            p.ty, PT_BRCK,
            "brush must not change the element at ({x}, {y})"
        );
        assert_eq!(
            p.dcolour, 0xFF11_2233,
            "decoration not applied at ({x}, {y})"
        );
        assert_eq!(p.tmp, 77, "field write not applied at ({x}, {y})");
    }
}

/// A smudge stroke averages the decorated neighbours and skips empty areas.
#[test]
fn test_brush_smudge_blends_neighbours() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    fill_rect(&mut engine, 10, 10, 40, 40, PT_BRCK);
    // Two colours side by side, so the pixel between them averages to a value
    // that is neither one.
    let blue: Vec<(i32, i32)> = (10..25).map(|x| (x, 10)).collect();
    let red: Vec<(i32, i32)> = (25..40).map(|x| (x, 10)).collect();
    engine.apply_command(SimCommand::DecorateAt {
        points: blue,
        colour: 0xFF00_00FF,
        mode: DecoMode::Draw,
    });
    engine.apply_command(SimCommand::DecorateAt {
        points: red,
        colour: 0xFFFF_0000,
        mode: DecoMode::Draw,
    });
    engine.step();

    // `ApplyDecoration`'s smudge window is `|rx| + |ry| <= 2`, so this pixel
    // sees x = 24 (blue) and x = 25..26 (red) on the decorated row.
    engine.apply_command(SimCommand::DecorateAt {
        points: vec![(25, 11)],
        colour: 0,
        mode: DecoMode::Smudge,
    });
    engine.ensure_host_current();
    let p = engine.bufs.particle_at(25, 11).expect("brick");
    let blue = p.dcolour & 0xFF;
    let red = (p.dcolour >> 16) & 0xFF;
    assert!(
        blue > 0 && blue < 0xFF && red > 0 && red < 0xFF,
        "smudge must blend towards the neighbour colours, got {:#010x}",
        p.dcolour
    );
}

/// The HUD particle counter must come from the device reduction.
///
/// The old implementation read the whole `part_ty` array back and counted on
/// the CPU; the device kernel must report the same number.
#[test]
fn test_live_count_matches_the_pool() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    fill_rect(&mut engine, 5, 5, 24, 24, PT_SAND);
    engine.step();
    let live = engine
        .bufs
        .refresh_active_count_device()
        .expect("device live count");
    engine.ensure_host_current();
    let counted = (0..engine.bufs.capacity)
        .filter(|&i| engine.bufs.part_ty[i] != PT_NONE)
        .count();
    assert_eq!(live, counted, "device reduction must match the host scan");
}
