//! Integration tests for the `.tptrs` save format and the undo/redo history.
//!
//! A save is only meaningful if it round-trips through a real, GPU-backed
//! [`SimulationEngine`]: `SaveData::capture_from` reads the device buffers back
//! into the host shadows, and `SaveData::restore_into` resizes the canvas and
//! re-uploads everything. These tests therefore drive a headless CubeCL/wgpu
//! device instead of fabricating a `SaveData` by hand, so a mismatch between the
//! save layer and the simulation buffers shows up here.
//!
//! The tests never call `SimulationEngine::step`, so no kernels are compiled:
//! only buffer allocation, upload and read-back are exercised.

use std::sync::LazyLock;

use cubecl::client::Client;

use tpt_core::constants::{
    CELL, PT_DMND, PT_FIRE, PT_GLAS, PT_SAND, PT_STNE, PT_WATR, WL_DETECT, WL_FAN, WL_WALL,
};
use tpt_core::{
    AirMode, EdgeMode, GpuParticle, GravityMode, RigidBody2D, RigidSpawnShape, SimCommand,
    SimulationConfig,
};
use tpt_save::{SAVE_VERSION, SaveData, TPT_RS_MAGIC, UndoHistory};
use tpt_sim::SimulationEngine;

/// The device every test engine runs on.
///
/// `WgpuDevice::default()` probes every enabled backend, prefers a real GPU and
/// falls back to a software rasteriser, so the suite also runs on machines
/// without a display. CubeCL registers one runtime per device id and serialises
/// concurrent creation, so the (slow) adapter/device acquisition happens once
/// per process even though the test harness runs tests in parallel.
static DEVICE: LazyLock<cubecl::wgpu::WgpuDevice> =
    LazyLock::new(cubecl::wgpu::WgpuDevice::default);

/// Creates a headless [`Client`], bringing the wgpu device up on first use.
fn headless_client() -> Client {
    use cubecl_runtime::runtime::Runtime;
    cubecl::wgpu::WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(&*DEVICE)
}

/// Builds an empty engine of the requested (cell-aligned) size.
fn headless_engine(xres: u32, yres: u32) -> SimulationEngine {
    SimulationEngine::new(headless_client(), SimulationConfig::new(xres, yres))
}

/// A particle with every field at a neutral value; tests tweak what they need.
fn particle(ty: u32, x: f32, y: f32, temp: f32, life: i32, ctype: u32) -> GpuParticle {
    GpuParticle {
        ty,
        life,
        ctype,
        x,
        y,
        vx: 0.0,
        vy: 0.0,
        temp,
        flags: 0,
        tmp: 0,
        tmp2: 0,
        tmp3: 0,
        tmp4: 0,
        dcolour: 0,
        _pad0: 0,
        _pad1: 0,
    }
}

/// Six particles that touch every field the save format persists.
fn seeded_particles() -> Vec<GpuParticle> {
    let mut sand = particle(PT_SAND, 10.0, 12.0, 512.75, 137, PT_GLAS);
    sand.vx = 0.25;
    sand.vy = -0.5;
    sand.flags = 0x0000_0004;
    sand.tmp = 1;
    sand.tmp2 = -2;
    sand.tmp3 = 0x1234_5678;
    sand.tmp4 = -0x0BAD_F00D;
    sand.dcolour = 0x0011_2233;

    let mut water = particle(PT_WATR, 11.5, 12.25, 283.5, 0, PT_STNE);
    water.vx = -1.0;
    water.vy = 2.5;
    water.tmp = -7;
    water.tmp2 = 9;

    let stone = particle(PT_STNE, 300.75, 200.5, 1200.0, -3, 0);

    let mut fire = particle(PT_FIRE, 30.0, 40.0, 900.25, 42, PT_SAND);
    fire.vx = 0.5;
    fire.vy = -0.75;
    fire.tmp = 3;
    fire.tmp2 = 4;

    let mut diamond = particle(PT_DMND, 31.0, 41.0, 300.125, 5, PT_WATR);
    diamond.dcolour = 0x0000_00FF;

    let mut glass = particle(PT_GLAS, 250.0, 64.0, 77.0, 9, PT_DMND);
    glass.vx = 1.5;
    glass.vy = -1.5;
    glass.tmp = 11;
    glass.tmp2 = 12;

    vec![sand, water, stone, fire, diamond, glass]
}

/// Writes distinctive values into the air, fan and wall grids.
fn seed_grids(engine: &mut SimulationEngine) {
    let cells = engine.bufs.config.num_cells();
    assert!(cells > 16, "test canvas is too small for the seeded cells");

    engine.bufs.air_vx[0] = 1.25;
    engine.bufs.air_vy[cells / 2] = -2.5;
    engine.bufs.air_pv[0] = 1.5;
    engine.bufs.air_pv[cells / 2] = -3.75;
    engine.bufs.air_pv[cells - 1] = 0.125;
    engine.bufs.air_hv[3] = 400.5;

    engine.bufs.wall_bmap[0] = WL_WALL as u32;
    engine.bufs.wall_bmap[cells / 2] = WL_FAN as u32;
    engine.bufs.wall_bmap[cells - 1] = WL_DETECT as u32;
    engine.bufs.wall_emap[cells / 2] = 9;

    engine.bufs.fan_vx[cells / 2] = 0.5;
    engine.bufs.fan_vy[cells / 2] = -0.25;

    // Direct host writes bypass `apply_command`, so mark every shadow dirty for
    // the next `sync_host_to_gpu` (otherwise the capture would read the stale
    // device contents back over these values).
    engine.bufs.mark_all_dirty();
}

/// `RigidBody2D` has no `PartialEq`; compare every field explicitly.
fn assert_rigid_bodies_match(expected: &[RigidBody2D], actual: &[RigidBody2D]) {
    assert_eq!(expected.len(), actual.len(), "rigid body count changed");
    for (want, got) in expected.iter().zip(actual) {
        assert_eq!(got.id, want.id, "rigid body id changed");
        assert_eq!(got.active, want.active, "rigid body activity changed");
        assert_eq!(got.cm_x, want.cm_x, "rigid body cm_x changed");
        assert_eq!(got.cm_y, want.cm_y, "rigid body cm_y changed");
        assert_eq!(got.vx, want.vx, "rigid body vx changed");
        assert_eq!(got.vy, want.vy, "rigid body vy changed");
        assert_eq!(got.angle, want.angle, "rigid body angle changed");
        assert_eq!(got.omega, want.omega, "rigid body omega changed");
        assert_eq!(got.mass, want.mass, "rigid body mass changed");
        assert_eq!(got.inv_mass, want.inv_mass, "rigid body inv_mass changed");
        assert_eq!(got.inertia, want.inertia, "rigid body inertia changed");
        assert_eq!(
            got.inv_inertia, want.inv_inertia,
            "rigid body inv_inertia changed"
        );
        assert_eq!(got.pixels, want.pixels, "rigid body pixels changed");
    }
}

/// A save carries its own canvas size, so loading it into a differently sized
/// engine must resize the engine and re-fit every grid to the saved cell count.
#[test]
fn restore_resizes_engine_to_the_saved_canvas() {
    const SAVED_W: u32 = 512;
    const SAVED_H: u32 = 320;
    const ENGINE_W: u32 = 612;
    const ENGINE_H: u32 = 384;

    let mut source = headless_engine(SAVED_W, SAVED_H);
    assert_eq!(
        (source.bufs.config.xres, source.bufs.config.yres),
        (SAVED_W, SAVED_H)
    );

    source.apply_command(SimCommand::PlaceParticles {
        points: vec![(40, 30), (41, 30), (200, 150)],
        elem_ty: PT_SAND,
        replace_mode: false,
        dcolour: 0x0000_FF00,
    });
    source.apply_command(SimCommand::PlaceWalls {
        cells: vec![(4, 4), (8, 4)],
        wall_id: WL_WALL,
        fan_vel: None,
    });

    let xcells = (SAVED_W / CELL) as usize;
    let pressure_cell = 5 * xcells + 7;
    source.bufs.air_pv[pressure_cell] = 2.5;
    source.bufs.mark_air_dirty();

    let save = SaveData::capture_from(&mut source);
    assert_eq!((save.config.xres, save.config.yres), (SAVED_W, SAVED_H));
    assert_eq!(save.particles.len(), 3);
    assert_eq!(save.air_pv[pressure_cell], 2.5);

    // Locate the painted wall cells in the saved grid instead of assuming how
    // the command maps its coordinates onto the block grid.
    let painted_walls: Vec<usize> = save
        .wall_bmap
        .iter()
        .enumerate()
        .filter(|(_, wall)| **wall == WL_WALL as u32)
        .map(|(cell, _)| cell)
        .collect();
    assert_eq!(painted_walls.len(), 2, "expected exactly two wall cells");

    let bytes = save.to_bytes().expect("serialize save");
    assert_eq!(&bytes[..TPT_RS_MAGIC.len()], TPT_RS_MAGIC);
    let decoded = SaveData::from_bytes(&bytes).expect("deserialize save");
    assert_eq!(decoded.version, SAVE_VERSION);
    assert_eq!(
        (decoded.config.xres, decoded.config.yres),
        (SAVED_W, SAVED_H)
    );

    // Target engine starts at the app's default canvas and already holds a
    // particle that the load must wipe out.
    let mut engine = headless_engine(ENGINE_W, ENGINE_H);
    assert_eq!(
        (engine.bufs.config.xres, engine.bufs.config.yres),
        (ENGINE_W, ENGINE_H)
    );
    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(600, 370)],
        elem_ty: PT_STNE,
        replace_mode: false,
        dcolour: 0,
    });
    assert_eq!(engine.particle_count(), 1);

    decoded.restore_into(&mut engine);

    // The engine followed the save's dimensions, not its own.
    assert_eq!(engine.bufs.config.xres, SAVED_W);
    assert_eq!(engine.bufs.config.yres, SAVED_H);
    assert_eq!(engine.bufs.config.num_cells(), save.config.num_cells());

    // Grids were re-fitted to the new cell count and hold the saved values.
    assert_eq!(engine.bufs.air_pv.len(), save.config.num_cells());
    assert_eq!(engine.bufs.air_hv.len(), save.config.num_cells());
    assert_eq!(engine.bufs.wall_bmap.len(), save.config.num_cells());
    assert_eq!(engine.bufs.wall_emap.len(), save.config.num_cells());
    assert_eq!(engine.bufs.air_pv, save.air_pv);
    assert_eq!(engine.bufs.air_hv, save.air_hv);
    assert_eq!(engine.bufs.wall_bmap, save.wall_bmap);
    assert_eq!(
        engine.bufs.pressure_at(7 * CELL as i32, 5 * CELL as i32),
        2.5
    );

    // The pre-load particle is gone and the saved ones landed on their pixels.
    assert_eq!(engine.particle_count(), save.particles.len());
    assert_eq!(engine.bufs.particle_at(600, 370), None);
    for saved in &save.particles {
        let restored = engine
            .bufs
            .particle_at(saved.x as i32, saved.y as i32)
            .expect("restored particle missing at its saved pixel");
        assert_eq!(restored.ty, saved.ty);
        assert_eq!(restored.temp, saved.temp);
        assert_eq!(restored.dcolour, saved.dcolour);
    }
    for &cell in &painted_walls {
        let cx = (cell % xcells) as i32;
        let cy = (cell / xcells) as i32;
        assert_eq!(
            engine.bufs.wall_at(cx * CELL as i32, cy * CELL as i32),
            Some(WL_WALL),
            "wall missing from restored cell {cell}"
        );
    }
}

/// Every byte of a snapshot must survive `to_bytes` -> `from_bytes` ->
/// `restore_into`, including particle fields, the wall/air grids and rigid
/// bodies.
#[test]
fn roundtrip_preserves_particles_walls_air_and_rigid_bodies() {
    const W: u32 = 320;
    const H: u32 = 256;

    let mut source = headless_engine(W, H);

    let expected_particles = seeded_particles();
    for (slot, saved) in expected_particles.iter().enumerate() {
        source.bufs.import_particle(slot, saved);
    }
    source.bufs.set_particle_count(expected_particles.len());
    source.bufs.rebuild_spatial_maps();
    seed_grids(&mut source);

    // Non-default modes must travel with the snapshot as well.
    source.bufs.config.paused = true;
    source.bufs.config.aheat_enable = true;
    source.bufs.config.pretty_powder = true;
    source.bufs.config.edge_mode = EdgeMode::Loop;
    source.bufs.config.gravity_mode = GravityMode::Radial;
    source.bufs.config.air_mode = AirMode::NoUpdate;
    source.bufs.config.ambient_air_temp = 310.5;
    source.bufs.config.edge_pressure = 0.75;
    source.bufs.config.display_mode = 0b1011;
    source.bufs.config.colour_mode = 3;
    source.bufs.mark_all_dirty();

    let box_id = source.spawn_rigid_body(60.0, 60.0, 3, 2, RigidSpawnShape::Box, PT_STNE);
    let disc_id = source.spawn_rigid_body(200.0, 120.0, 4, 4, RigidSpawnShape::Circle, PT_DMND);
    assert_eq!(source.rigid_body_count(), 2);

    let before = SaveData::capture_from(&mut source);
    assert_eq!(before.version, SAVE_VERSION);
    assert_eq!((before.config.xres, before.config.yres), (W, H));
    assert_eq!(before.particles, expected_particles);
    assert_eq!(before.rigid_bodies.len(), 2);
    assert_eq!(before.rigid_bodies[0].id, box_id);
    assert_eq!(before.rigid_bodies[1].id, disc_id);

    let bytes = before.to_bytes().expect("serialize save");
    assert_eq!(&bytes[..TPT_RS_MAGIC.len()], TPT_RS_MAGIC);

    let decoded = SaveData::from_bytes(&bytes).expect("deserialize save");
    assert_eq!(decoded.version, SAVE_VERSION);
    assert_eq!(decoded.particles, before.particles);
    assert_eq!(decoded.air_pv, before.air_pv);
    assert_eq!(decoded.wall_bmap, before.wall_bmap);
    assert_eq!(decoded.wall_emap, before.wall_emap);
    assert_rigid_bodies_match(&before.rigid_bodies, &decoded.rigid_bodies);

    // Restore into a *fresh* engine: only the save may define the result.
    let mut target = headless_engine(W, H);
    assert_eq!(target.particle_count(), 0);
    assert_eq!(target.rigid_body_count(), 0);

    decoded.restore_into(&mut target);

    assert_eq!(target.bufs.config.xres, W);
    assert_eq!(target.bufs.config.yres, H);
    assert_eq!(target.particle_count(), expected_particles.len());
    assert_eq!(target.rigid_body_count(), 2);
    assert_eq!(target.bufs.next_rigid_id, disc_id + 1);

    // Physics / display modes.
    assert!(target.bufs.config.paused);
    assert!(target.bufs.config.aheat_enable);
    assert!(target.bufs.config.pretty_powder);
    assert_eq!(target.bufs.config.edge_mode, EdgeMode::Loop);
    assert_eq!(target.bufs.config.gravity_mode, GravityMode::Radial);
    assert_eq!(target.bufs.config.air_mode, AirMode::NoUpdate);
    assert_eq!(target.bufs.config.ambient_air_temp, 310.5);
    assert_eq!(target.bufs.config.edge_pressure, 0.75);
    assert_eq!(target.bufs.config.display_mode, 0b1011);
    assert_eq!(target.bufs.config.colour_mode, 3);

    // Re-capture the restored engine and require the same snapshot back.
    let after = SaveData::capture_from(&mut target);
    assert_eq!(after.particles, before.particles);
    assert_eq!(after.air_vx, before.air_vx);
    assert_eq!(after.air_vy, before.air_vy);
    assert_eq!(after.air_pv, before.air_pv);
    assert_eq!(after.air_hv, before.air_hv);
    assert_eq!(after.fan_vx, before.fan_vx);
    assert_eq!(after.fan_vy, before.fan_vy);
    assert_eq!(after.wall_bmap, before.wall_bmap);
    assert_eq!(after.wall_emap, before.wall_emap);
    assert_rigid_bodies_match(&before.rigid_bodies, &after.rigid_bodies);
    assert_eq!(
        after.to_bytes().expect("re-serialize save"),
        bytes,
        "restored engine does not reproduce the saved payload"
    );

    // The spatial map was rebuilt for the restored particles.
    for saved in &before.particles {
        let at = target
            .bufs
            .particle_at(saved.x.round() as i32, saved.y.round() as i32);
        assert!(
            at.is_some(),
            "no particle at the restored pixel ({}, {})",
            saved.x,
            saved.y
        );
    }
}

/// Undo/redo must walk the history of brush strokes, wall edits and rigid body
/// spawns, and a fresh edit after an undo must drop the redo branch.
#[test]
fn undo_history_replays_brush_wall_and_rigid_body_edits() {
    let mut engine = headless_engine(192, 128);
    let mut history = UndoHistory::new(8);

    assert!(!history.undo(&mut engine));
    assert!(!history.redo(&mut engine));

    history.push_snapshot(&mut engine);
    assert!(history.can_undo());
    assert!(!history.can_redo());

    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(20, 20), (21, 20), (20, 21), (21, 21)],
        elem_ty: PT_SAND,
        replace_mode: false,
        dcolour: 0,
    });
    assert_eq!(engine.particle_count(), 4);
    history.push_snapshot(&mut engine);

    engine.apply_command(SimCommand::PlaceWalls {
        cells: vec![(12, 8)],
        wall_id: WL_WALL,
        fan_vel: None,
    });
    // Derive the painted block from the grid: the command's coordinate mapping
    // is an engine detail, the history only has to restore the same block.
    let painted: Vec<usize> = engine
        .bufs
        .wall_bmap
        .iter()
        .enumerate()
        .filter(|(_, wall)| **wall == WL_WALL as u32)
        .map(|(cell, _)| cell)
        .collect();
    assert_eq!(painted.len(), 1, "expected exactly one painted wall cell");
    let wall_cell = painted[0];
    history.push_snapshot(&mut engine);

    engine.apply_command(SimCommand::SpawnRigidBody {
        x: 96.0,
        y: 64.0,
        half_w: 3,
        half_h: 2,
        shape: RigidSpawnShape::Box,
        elem_ty: PT_STNE,
    });
    assert_eq!(engine.rigid_body_count(), 1);
    let box_id = engine.bufs.rigid_bodies[0].id;
    history.push_snapshot(&mut engine);

    engine.apply_command(SimCommand::SpawnRigidBody {
        x: 48.0,
        y: 32.0,
        half_w: 2,
        half_h: 2,
        shape: RigidSpawnShape::Circle,
        elem_ty: PT_DMND,
    });
    assert_eq!(engine.rigid_body_count(), 2);
    let disc_id = engine.bufs.rigid_bodies[1].id;
    history.push_snapshot(&mut engine);

    // An edit made after the last snapshot is not an undo point of its own; the
    // first undo pushes it onto the redo stack.
    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(40, 40)],
        elem_ty: PT_WATR,
        replace_mode: false,
        dcolour: 0,
    });
    assert_eq!(engine.particle_count(), 5);

    assert!(history.undo(&mut engine));
    assert_eq!(engine.particle_count(), 4);
    assert_eq!(engine.rigid_body_count(), 2);
    assert_eq!(engine.bufs.rigid_bodies[0].id, box_id);
    assert_eq!(engine.bufs.rigid_bodies[1].id, disc_id);
    assert_eq!(engine.bufs.wall_bmap[wall_cell], WL_WALL as u32);
    assert!(history.can_redo());

    assert!(history.undo(&mut engine));
    assert_eq!(engine.particle_count(), 4);
    assert_eq!(engine.rigid_body_count(), 1);
    assert_eq!(engine.bufs.rigid_bodies[0].id, box_id);
    assert_eq!(engine.bufs.wall_bmap[wall_cell], WL_WALL as u32);

    assert!(history.undo(&mut engine));
    assert_eq!(engine.particle_count(), 4);
    assert_eq!(engine.rigid_body_count(), 0);
    assert_eq!(engine.bufs.wall_bmap[wall_cell], WL_WALL as u32);

    assert!(history.undo(&mut engine));
    assert_eq!(engine.particle_count(), 4);
    assert!(engine.bufs.wall_bmap.iter().all(|&wall| wall == 0));

    assert!(history.undo(&mut engine));
    assert_eq!(engine.particle_count(), 0);
    assert_eq!(engine.rigid_body_count(), 0);
    assert!(!history.can_undo());
    assert!(!history.undo(&mut engine));

    // Redo back up through every state.
    assert!(history.redo(&mut engine));
    assert_eq!(engine.particle_count(), 4);
    assert_eq!(engine.rigid_body_count(), 0);
    assert!(engine.bufs.wall_bmap.iter().all(|&wall| wall == 0));

    assert!(history.redo(&mut engine));
    assert_eq!(engine.bufs.wall_bmap[wall_cell], WL_WALL as u32);
    assert_eq!(engine.rigid_body_count(), 0);

    assert!(history.redo(&mut engine));
    assert_eq!(engine.rigid_body_count(), 1);
    assert_eq!(engine.bufs.rigid_bodies[0].id, box_id);

    assert!(history.redo(&mut engine));
    assert_eq!(engine.rigid_body_count(), 2);
    assert_eq!(engine.bufs.rigid_bodies[1].id, disc_id);

    // The un-snapshotted water pixel comes back with the final redo.
    assert!(history.redo(&mut engine));
    assert_eq!(engine.particle_count(), 5);
    assert!(!history.can_redo());
    assert!(!history.redo(&mut engine));

    // A new edit after an undo discards the redo branch.
    assert!(history.undo(&mut engine));
    assert!(history.can_redo());
    engine.apply_command(SimCommand::KillAt { x: 20, y: 20 });
    assert_eq!(engine.particle_count(), 3);
    history.push_snapshot(&mut engine);
    assert!(!history.can_redo());
    assert!(history.redo_stack.is_empty());

    // The bound is enforced by `push_snapshot`, and zero is clamped to one.
    let mut bounded = UndoHistory::new(2);
    for _ in 0..4 {
        bounded.push_snapshot(&mut engine);
    }
    assert_eq!(bounded.max_snapshots, 2);
    assert_eq!(bounded.undo_stack.len(), 2);

    let mut clamped = UndoHistory::new(0);
    assert_eq!(clamped.max_snapshots, 1);
    clamped.push_snapshot(&mut engine);
    clamped.push_snapshot(&mut engine);
    assert_eq!(clamped.undo_stack.len(), 1);
}

/// The UI offers a 2048x2048 canvas, which saturates the particle pool at
/// `SimBuffers::MAX_PARTICLE_CAPACITY` (a 2^23 slot request clamped to the
/// 23-bit id space). A capture at that size reads back the full device state,
/// so an index or size assumption that only breaks at the cap would crash the
/// application's `Save` button. The restore must then hand the same particles
/// back without dropping or duplicating any of them.
#[test]
fn save_round_trip_at_the_saturated_pool_canvas() {
    let mut engine = headless_engine(2048, 2048);
    assert_eq!(
        engine.bufs.capacity,
        tpt_sim::SimBuffers::MAX_PARTICLE_CAPACITY,
        "2048x2048 must saturate the pool for this test to cover the cap"
    );
    // Half the canvas is enough to prove the whole-pool read-back and the
    // per-particle serialisation stay consistent at the cap.
    let mut points = Vec::new();
    for y in 0..1024i32 {
        for x in 0..2048i32 {
            points.push((x, y));
        }
    }
    engine.apply_command(SimCommand::PlaceParticles {
        points,
        elem_ty: PT_SAND,
        replace_mode: false,
        dcolour: 0,
    });
    let placed = engine.particle_count();
    assert_eq!(placed, 1024 * 2048);

    let save = SaveData::capture_from(&mut engine);
    assert_eq!(save.particles.len(), placed, "capture dropped particles");
    let bytes = save.to_bytes().expect("serialize save");
    let decoded = SaveData::from_bytes(&bytes).expect("deserialize save");
    assert_eq!(
        decoded.particles.len(),
        placed,
        "serialisation dropped particles"
    );
    assert_eq!((decoded.config.xres, decoded.config.yres), (2048, 2048));

    decoded.restore_into(&mut engine);
    assert_eq!(engine.particle_count(), placed, "restore lost particles");
}
