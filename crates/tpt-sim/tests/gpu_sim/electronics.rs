//! GPU tests for the electrical element updates: `BTRY.cpp:48-76`,
//! `WIRE.cpp:50-94`, `PUMP.cpp:53-111` and `THDR.cpp:51-93`.

use super::*;

/// Slot of the particle occupying pixel `(x, y)` in the host shadow.
fn slot_at(engine: &SimulationEngine, x: i32, y: i32) -> usize {
    engine
        .bufs
        .particle_index_at(x, y)
        .unwrap_or_else(|| panic!("no particle at ({x}, {y})"))
}

/// `ctype` of the particle at `(x, y)`.
fn ctype_at(engine: &SimulationEngine, x: i32, y: i32) -> u32 {
    engine.bufs.part_ctype[slot_at(engine, x, y)]
}

/// `life` of the particle at `(x, y)`.
fn life_at(engine: &SimulationEngine, x: i32, y: i32) -> i32 {
    engine.bufs.part_life[slot_at(engine, x, y)]
}

/// `BTRY.cpp:48-76`: the battery sparks every idle conductor in its radius-2
/// diamond, records the conductor type in `ctype` and gives the spark a life of
/// four. The `(2, 2)` corner has Manhattan distance four and is outside the
/// diamond.
#[test]
fn battery_sparks_a_conductor_in_its_radius_two_diamond() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(32, 32);

    fill_rect(&mut engine, 8, 8, 8, 8, PT_BTRY);
    fill_rect(&mut engine, 10, 8, 10, 8, PT_METL); // distance 2
    fill_rect(&mut engine, 10, 10, 10, 10, PT_METL); // distance 4

    step(&mut engine);

    assert_eq!(count_type(&engine, PT_BTRY), 1);
    assert_eq!(
        count_type(&engine, PT_SPRK),
        1,
        "the radius-2 pixel sparks, the (2, 2) corner does not"
    );
    assert_eq!(
        engine.bufs.part_ty[slot_at(&engine, 10, 8)],
        PT_SPRK,
        "the conductor two pixels away became a spark"
    );
    assert_eq!(
        life_at(&engine, 10, 8),
        4,
        "`BTRY.cpp:67` gives the new spark a life of 4"
    );
    assert_eq!(
        ctype_at(&engine, 10, 8),
        PT_METL,
        "`BTRY.cpp:68` records the conductor type in `ctype`"
    );
    assert_eq!(
        engine.bufs.part_ty[slot_at(&engine, 10, 10)],
        PT_METL,
        "`abs(rx)+abs(ry) < 4` excludes the diagonal corner"
    );
}

/// `Simulation::set_emap` (`Simulation.cpp:928-983`) is triggered by a
/// particle standing in an uncharged `WL_DETECT` cell and must continue from
/// the horizontal span into the vertical wire tree above and below it. The
/// reference recursion is replaced by `emap_closure_kernel`, so this test
/// exercises the actual device pass: four detector cells form a vertical line,
/// the particle sits in the bottom one, and every cell of the line must be
/// charged to 16.
#[test]
fn detect_set_emap_charges_a_vertical_wire_span() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);

    let cells: Vec<(i32, i32)> = (5..=8).map(|cy| (8, cy)).collect();
    engine.apply_command(SimCommand::PlaceWalls {
        cells,
        wall_id: WL_DETECT,
        fan_vel: None,
    });
    // One pixel in the bottom detector cell (cell (8, 8) covers pixels
    // 32..=35); `WL_DETECT` does not kill solids, so it survives the pre-move
    // gates long enough to fire the detector.
    fill_rect(&mut engine, 32, 32, 32, 32, PT_DUST);

    step(&mut engine);
    engine.bufs.sync_walls_gpu_to_host();

    let xcells = engine.bufs.config.xcells();
    for cy in 5..=8 {
        let emap = engine.bufs.wall_emap[cy as usize * xcells as usize + 8] & 0xFF;
        assert!(emap >= 8, "cell (8, {cy}) must be charged, got emap {emap}");
    }
    assert_eq!(
        engine.bufs.wall_emap[8 * xcells as usize + 8] & 0xFF,
        16,
        "set_emap writes 16 into the triggered span"
    );
}

/// `WIRE.cpp:50-94` together with `Simulation.cpp:3862-3876`: a spark head
/// travels one pixel per frame along a straight wire, leaving a tail behind.
#[test]
fn wire_chain_advances_one_stage_per_frame() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(32, 32);

    // Five wire pixels at x = 6..=10, y = 10.
    fill_rect(&mut engine, 6, 10, 10, 10, PT_WIRE);
    assert_eq!(count_type(&engine, PT_WIRE), 5);

    // Seed the head by hand, the way a `PSCN` spark would (`WIRE.cpp:79-83`).
    let head = slot_at(&engine, 6, 10);
    engine.bufs.part_ctype[head] = 1;
    engine.bufs.mark_particles_dirty();

    step(&mut engine);
    assert_eq!(
        [
            ctype_at(&engine, 6, 10),
            ctype_at(&engine, 7, 10),
            ctype_at(&engine, 8, 10),
            ctype_at(&engine, 9, 10),
            ctype_at(&engine, 10, 10),
        ],
        [2, 1, 0, 0, 0],
        "the head becomes a tail and lights exactly one neighbour"
    );

    step(&mut engine);
    assert_eq!(
        [
            ctype_at(&engine, 6, 10),
            ctype_at(&engine, 7, 10),
            ctype_at(&engine, 8, 10),
            ctype_at(&engine, 9, 10),
            ctype_at(&engine, 10, 10),
        ],
        [0, 2, 1, 0, 0],
        "the front advances exactly one pixel per frame"
    );

    step(&mut engine);
    assert_eq!(
        [
            ctype_at(&engine, 6, 10),
            ctype_at(&engine, 7, 10),
            ctype_at(&engine, 8, 10),
            ctype_at(&engine, 9, 10),
            ctype_at(&engine, 10, 10),
        ],
        [0, 0, 2, 1, 0],
        "the third generation is one more pixel along"
    );
}

/// `WIRE.cpp:79-83`: a `PSCN` spark at the third frame of its life lights an
/// adjacent wire.
#[test]
fn wire_lights_from_a_pscn_spark_head() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(32, 32);

    fill_rect(&mut engine, 6, 10, 6, 10, PT_WIRE);
    fill_rect(&mut engine, 7, 10, 7, 10, PT_PSCN);
    spark_at(&mut engine, 7, 10);
    let spark = slot_at(&engine, 7, 10);
    assert_eq!(engine.bufs.part_life[spark], 4, "a fresh spark starts at 4");

    // The life pass drops the spark to 3 before the wire update runs, which is
    // exactly the `parts[ID(r)].life == 3` gate in `WIRE.cpp:79`.
    step(&mut engine);
    assert_eq!(
        ctype_at(&engine, 6, 10),
        1,
        "the PSCN spark head lights the wire"
    );
}

/// `WIRE.cpp:84-85`: a lit wire sparks an adjacent `NSCN` through
/// `create_part`, so the scanner becomes a four-frame spark of its own type.
#[test]
fn wire_sparks_an_adjacent_nscn() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(32, 32);

    fill_rect(&mut engine, 6, 10, 6, 10, PT_WIRE);
    fill_rect(&mut engine, 7, 10, 7, 10, PT_NSCN);
    let wire = slot_at(&engine, 6, 10);
    engine.bufs.part_ctype[wire] = 1;
    engine.bufs.mark_particles_dirty();

    step(&mut engine);

    let scanner = slot_at(&engine, 7, 10);
    assert_eq!(engine.bufs.part_ty[scanner], PT_SPRK);
    assert_eq!(engine.bufs.part_ctype[scanner], PT_NSCN);
    assert_eq!(engine.bufs.part_life[scanner], 4);
    assert!(
        (engine.bufs.part_temp[scanner] - 305.15).abs() < 1e-3,
        "`create_part` heats the good conductors by 10 K below 673 K, found {}",
        engine.bufs.part_temp[scanner]
    );
}

/// `PUMP.cpp:67-75`: an active pump pulls the five non-diagonal cells of its
/// 3x3 window a tenth of the way towards `temp - 273.15` each frame.
#[test]
fn pump_pulls_pressure_towards_its_temperature() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    // Freeze the air solver so only the pump writes the pressure.
    engine.config_mut().air_mode = AirMode::NoUpdate;

    // Pixel (6, 6) sits in the middle of air cell (1, 1), so all five writes
    // land on the same cell.
    fill_rect(&mut engine, 6, 6, 6, 6, PT_PUMP);
    set_temp_of_type(&mut engine, PT_PUMP, 273.15 + 50.0);

    step(&mut engine);

    let cell = engine.bufs.cell_index(6, 6).expect("cell in range");
    // Five successive `pv += 0.1*(50 - pv)` steps: 50*(1 - 0.9^5).
    let expected = 50.0 * (1.0 - 0.9f32.powi(5));
    assert!(
        (engine.bufs.air_pv[cell] - expected).abs() < 1e-3,
        "pump target 50 over five writes: expected {expected}, got {}",
        engine.bufs.air_pv[cell]
    );
}

/// `PUMP.cpp:76-86`: a pump whose `tmp == 1` writes the pressure serialised in
/// an adjacent `FILT`'s `ctype` (`0x10000000 + pressure`) straight into the
/// filter's air cell.
#[test]
fn pump_writes_a_filt_pressure_setting() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().air_mode = AirMode::NoUpdate;

    fill_rect(&mut engine, 6, 6, 6, 6, PT_PUMP);
    fill_rect(&mut engine, 7, 6, 7, 6, PT_FILT);

    let pump = slot_at(&engine, 6, 6);
    let filt = slot_at(&engine, 7, 6);
    engine.bufs.part_tmp[pump] = 1; // filter mode
    engine.bufs.part_ctype[filt] = 0x1000_0000 + 123;
    engine.bufs.mark_particles_dirty();

    step(&mut engine);

    let cell = engine.bufs.cell_index(7, 6).expect("cell in range");
    assert_eq!(
        engine.bufs.air_pv[cell],
        123.0 + MIN_PRESSURE,
        "`newPressure + MIN_PRESSURE` is 123 - 256"
    );
}

/// `THDR.cpp:66-72`: lightning turns an idle conductor into a four-frame spark
/// that remembers the conductor type, and `THDR.cpp:88-91` then kills it.
#[test]
fn thunder_sparks_a_conductor_and_dies() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(32, 32);

    fill_rect(&mut engine, 8, 8, 8, 8, PT_THDR);
    fill_rect(&mut engine, 9, 8, 9, 8, PT_METL);
    // The default lightning is 9273 K; a single conduction step would melt the
    // metal before the reaction pass runs, so isolate the spark rule.
    set_temp_of_type(&mut engine, PT_THDR, 300.0);

    step(&mut engine);

    assert_eq!(count_type(&engine, PT_THDR), 0, "`kill_part` removes it");
    assert_eq!(
        engine.bufs.part_ty[slot_at(&engine, 9, 8)],
        PT_SPRK,
        "the adjacent metal became a spark"
    );
    assert_eq!(
        ctype_at(&engine, 9, 8),
        PT_METL,
        "`parts[ID(r)].ctype = parts[ID(r)].type` keeps the conductor type"
    );
    assert_eq!(life_at(&engine, 9, 8), 4, "`THDR.cpp:70` sets life 4");
}

/// `THDR.cpp:73-84`: every non-conductor neighbour adds 100 pressure to the
/// lightning's own air cell, and the lightning dies as well.
#[test]
fn thunder_adds_100_pressure_and_dies() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(32, 32);
    // Freeze the air solver so the lightning's burst is the only pressure
    // change in the frame.
    engine.config_mut().air_mode = AirMode::NoUpdate;

    fill_rect(&mut engine, 8, 8, 8, 8, PT_THDR);
    fill_rect(&mut engine, 9, 8, 9, 8, PT_DUST);
    set_temp_of_type(&mut engine, PT_THDR, 300.0);

    step(&mut engine);

    let cell = engine.bufs.cell_index(8, 8).expect("cell in range");
    assert_eq!(count_type(&engine, PT_THDR), 0, "the lightning is consumed");
    assert_eq!(
        engine.bufs.air_pv[cell], 100.0,
        "one qualifying neighbour adds exactly 100 pressure"
    );
}

/// `HSWC.cpp:66-102`: a latched heat switch (`life == 10`) keeps its life and
/// scans its radius-2 ring. With `tmp == 1` an adjacent `FILT`'s
/// `0x10000000 + temperature` is deserialised into the switch, and an idle
/// heat switch within the ring follows the latch.
#[test]
fn hswc_latch_deserialises_a_filt_temperature_and_relays() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(32, 32);
    engine.config_mut().air_mode = AirMode::NoUpdate;

    fill_rect(&mut engine, 10, 10, 10, 10, PT_HSWC);
    fill_rect(&mut engine, 11, 10, 11, 10, PT_FILT);
    fill_rect(&mut engine, 12, 10, 12, 10, PT_HSWC);

    let latched = slot_at(&engine, 10, 10);
    engine.bufs.part_life[latched] = 10;
    engine.bufs.part_tmp[latched] = 1; // temperature deserialisation mode
    let filter = slot_at(&engine, 11, 10);
    engine.bufs.part_ctype[filter] = 0x1000_0000 + 400;
    engine.bufs.part_temp[filter] = 400.0;
    let quiet = slot_at(&engine, 12, 10);
    engine.bufs.part_life[quiet] = 0;
    engine.bufs.mark_particles_dirty();

    step(&mut engine);

    assert_eq!(
        engine.bufs.part_temp[latched], 400.0,
        "`HSWC.cpp:82-87` copies the FILT's serialised temperature"
    );
    assert_eq!(
        engine.bufs.part_life[latched], 10,
        "a latched heat switch must not count down"
    );
    assert_eq!(
        engine.bufs.part_life[quiet], 10,
        "`HSWC.cpp:73-74`: an idle neighbour within radius two follows the latch"
    );
}
