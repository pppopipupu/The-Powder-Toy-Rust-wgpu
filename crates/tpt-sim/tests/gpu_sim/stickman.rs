use super::*;

/// Creating a stickman also creates its spawn point, and a dead player
/// reappears there.
///
/// TPT's `Element_STKM_create` (`STKM.cpp:75-79`) creates a `PT_SPAWN` at the
/// stickman's pixel and records it in `player.spawnID`; `Simulation.cpp:3911`
/// recreates the player at that pixel whenever `spwn` is clear. The port used
/// to kill the player for good.
#[test]
fn test_stickman_respawns_at_its_spawn_point() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;

    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(30, 30)],
        elem_ty: PT_STKM,
        replace_mode: false,
        dcolour: 0,
    });
    step(&mut engine);
    assert_eq!(
        count_type(&engine, PT_STKM),
        1,
        "the placed stickman must exist"
    );
    assert_eq!(
        count_type(&engine, PT_SPAWN),
        1,
        "a stickman must create its spawn point (`STKM.cpp:75-79`)"
    );

    // Kill the player and let the spawn point bring it back. The death and the
    // respawn can land in the same dispatch (both are threads of the same
    // kernel), so the test only checks the end state and the new slot.
    engine.apply_command(SimCommand::SetParticleFieldAt {
        points: vec![(30, 30)],
        field: tpt_core::commands::ParticleField::Life,
        value: 0,
    });
    for _ in 0..3 {
        step(&mut engine);
    }
    assert_eq!(
        count_type(&engine, PT_STKM),
        1,
        "the spawn point must recreate the player (`Simulation.cpp:3911-3914`)"
    );
    let respawned = stickman_slot(&engine, PT_STKM).expect("the respawned stickman must exist");
    assert_eq!(
        engine.bufs.part_life[respawned], 100,
        "the player must be recreated from `DefaultProperties`, not kept alive"
    );
    let (x, y) = stickman_position(&engine, PT_STKM).expect("the respawned stickman must exist");
    // The stickman keeps falling under its own gravity after the respawn, so
    // the horizontal position and the short drop are what pin the origin.
    assert_eq!(x, 30, "the player must respawn at its spawn point's column");
    assert!(
        (30..=42).contains(&y),
        "the player must respawn at its spawn point and fall from there, got ({x}, {y})"
    );
}

/// Slot of the first particle of `ty`.
fn stickman_slot(engine: &SimulationEngine, ty: u32) -> Option<usize> {
    let bufs = &engine.bufs;
    (0..bufs.capacity).find(|&id| bufs.part_ty[id] == ty)
}

/// Resolves the pixel of the first particle of `ty`.
fn stickman_position(engine: &SimulationEngine, ty: u32) -> Option<(i32, i32)> {
    let bufs = &engine.bufs;
    stickman_slot(engine, ty).map(|id| {
        (
            bufs.part_x[id].round() as i32,
            bufs.part_y[id].round() as i32,
        )
    })
}

/// `GameController.cpp:611-730` -> `STKM.cpp:277-287`: a command bit must
/// actually move the stickman. The port used to leave `player.comm` at zero
/// forever, so arrow keys/WASD did nothing.
#[test]
fn test_stickman_responds_to_player_command() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;

    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(30, 30)],
        elem_ty: PT_STKM,
        replace_mode: false,
        dcolour: 0,
    });
    step(&mut engine);
    let (start, _) = stickman_position(&engine, PT_STKM).expect("the stickman must exist");

    engine.apply_command(SimCommand::SetStickmanControl {
        player: 0,
        comm: 0x02,
        pcomm: 0x02,
    });
    // `rig.rs` also keeps the rig falling, so stop before it reaches the
    // off-screen kill margin of this tiny 64x64 canvas.
    for _ in 0..4 {
        step(&mut engine);
    }
    let (end, _) = stickman_position(&engine, PT_STKM).expect("the stickman must survive");
    assert!(
        end > start + 2,
        "the right command must move the stickman rightwards, got {start} -> {end}"
    );
}
