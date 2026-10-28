//! GPU tests for the partial element branches completed in this pass:
//! photoelectric sparking, the Tesla coil's per-neighbour bolts, the `PCLN`
//! `PHOT`/`LIFE` clone shapes, the `ETRD` pool-scan fallback and the `DEUT`
//! neighbour fill.

use super::*;

/// Octant angle stored in `LIGH.tmp`, mirroring `atan2(-ry, rx) * 360 / pi`
/// for the eight primary directions (`SPRK.cpp:153`).
fn octant(dx: i32, dy: i32) -> i32 {
    if dx > 0 {
        if dy < 0 {
            45
        } else if dy > 0 {
            315
        } else {
            0
        }
    } else if dx < 0 {
        if dy < 0 {
            135
        } else if dy > 0 {
            225
        } else {
            180
        }
    } else if dy < 0 {
        90
    } else {
        270
    }
}

/// `Simulation.cpp:1568-1580` `photoelectric_effect`: a photon stopped by an
/// idle `PSCN` sparks it when an `NSCN` touches one of its orthogonal sides.
#[test]
fn test_photoelectric_effect_sparks_pscn_beside_nscn() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(96, 96);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    // Target electrode with n-type silicon below it, and a control electrode
    // without any `NSCN` neighbour.
    fill_rect(&mut engine, 60, 30, 60, 30, PT_PSCN);
    fill_rect(&mut engine, 60, 31, 60, 31, PT_NSCN);
    fill_rect(&mut engine, 60, 60, 60, 60, PT_PSCN);

    // Two white photons three pixels to the left, travelling +x at 3 px/frame:
    // the intent pass reports the blocking boundary pixel (`Simulation.cpp:
    // 3044-3056`).
    fill_rect(&mut engine, 57, 30, 57, 30, PT_PHOT);
    fill_rect(&mut engine, 57, 60, 57, 60, PT_PHOT);
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] == PT_PHOT {
            engine.bufs.part_vx[id] = 3.0;
            engine.bufs.part_vy[id] = 0.0;
            engine.bufs.part_ctype[id] = 0x3FFF_FFFF;
            engine.bufs.mark_particle_slot_dirty(id);
        }
    }

    step(&mut engine);

    let sparked = engine
        .bufs
        .particle_index_at(60, 30)
        .expect("the blocked PSCN pixel");
    assert_eq!(
        engine.bufs.part_ty[sparked], PT_SPRK,
        "`Simulation.cpp:1577`: the PSCN turns into a spark"
    );
    assert_eq!(
        engine.bufs.part_ctype[sparked], PT_PSCN,
        "`Simulation.cpp:1576`: the spark remembers the electrode type"
    );
    assert_eq!(
        engine.bufs.part_life[sparked], 4,
        "`Simulation.cpp:1580`: the spark keeps four frames of life"
    );

    let control = engine
        .bufs
        .particle_index_at(60, 60)
        .expect("the control PSCN pixel");
    assert_eq!(
        engine.bufs.part_ty[control], PT_PSCN,
        "an electrode without an NSCN neighbour must not spark"
    );
}

/// `SPRK.cpp:130-170` `case PT_TESC`: one independent chance roll per empty
/// neighbour, each success creating a bolt at twice the neighbour offset,
/// while the coil self-cools and damps its air cell once per bolt.
#[test]
fn test_tesla_coil_fires_one_bolt_per_successful_neighbour() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);

    // Power 5 gives `chance(1, 5*5/20 + 6) = 1/7` per neighbour: a frame with
    // two or more successes is common but seed dependent, so search the frame
    // counter the same way TPT's RNG would advance.
    let mut found = false;
    for attempt in 0..64u64 {
        engine.clear();
        engine.config_mut().gravity_mode = GravityMode::Off;
        engine.config_mut().air_mode = AirMode::NoUpdate;
        engine.config_mut().aheat_enable = false;
        engine.frame_count = attempt;

        fill_rect(&mut engine, 32, 32, 32, 32, PT_TESC);
        spark_at(&mut engine, 32, 32);
        let coil = engine
            .bufs
            .particle_index_at(32, 32)
            .expect("the sparked coil");
        engine.bufs.part_tmp[coil] = 5;
        engine.bufs.part_temp[coil] = 300.0;
        engine.bufs.mark_particle_slot_dirty(coil);

        let xcells = engine.bufs.config.xcells();
        let cell = ((32u32 / CELL) * xcells + 32u32 / CELL) as usize;
        engine.bufs.air_pv[cell] = 2.0;
        engine.bufs.mark_air_dirty();

        step(&mut engine);

        let bolts: Vec<usize> = (0..engine.bufs.capacity)
            .filter(|&id| engine.bufs.part_ty[id] == PT_LIGH)
            .collect();
        if bolts.len() < 2 {
            continue;
        }
        found = true;
        let n = bolts.len() as i32;

        // `SPRK.cpp:157`: `temp -= tmp*2 + temp/5`, once per created bolt.
        let mut expected_temp = 300.0f32;
        // `SPRK.cpp:26` gives a spark `HotAir = 0.001`; the pre-reaction
        // `hot_air_pressure_kernel` (`Simulation.cpp:2347-2358`) adds
        // `4 * HotAir = 0.004` to the coil's cell before the arc pass runs.
        // `SPRK.cpp:158-164`: `pv` then loses half a unit per bolt, and a cell
        // at or below half a unit snaps to zero.
        let mut expected_pv = 2.0f32 + 4.0f32 * 0.001f32;
        for _ in 0..n {
            expected_temp -= 10.0 + expected_temp / 5.0;
            if expected_pv.abs() <= 0.5 {
                expected_pv = 0.0;
            } else if expected_pv > 0.0 {
                expected_pv -= 0.5;
            } else {
                expected_pv += 0.5;
            }
        }
        assert!(
            (engine.bufs.part_temp[coil] - expected_temp).abs() < 0.01,
            "coil self-cooling after {n} bolts: expected {expected_temp}, got {}",
            engine.bufs.part_temp[coil]
        );
        assert!(
            (engine.bufs.air_pv[cell] - expected_pv).abs() < 1e-6,
            "pressure damping after {n} bolts: expected {expected_pv}, got {}",
            engine.bufs.air_pv[cell]
        );

        let mut seen: Vec<(i32, i32)> = Vec::new();
        for id in bolts {
            let bx = engine.bufs.part_x[id].round() as i32;
            let by = engine.bufs.part_y[id].round() as i32;
            let rx = bx - 32;
            let ry = by - 32;
            assert!(
                rx % 2 == 0
                    && ry % 2 == 0
                    && rx.abs() <= 2
                    && ry.abs() <= 2
                    && (rx != 0 || ry != 0),
                "`create_part(-1, x+rx*2, y+ry*2)`: bolt at ({bx},{by}) is not a double offset"
            );
            let dir = (rx / 2, ry / 2);
            assert!(
                !seen.contains(&dir),
                "two bolts share the {dir:?} neighbour"
            );
            seen.push(dir);
            assert_eq!(
                engine.bufs.part_tmp[id],
                octant(dir.0, dir.1),
                "bolt angle for neighbour {dir:?}"
            );
            assert_eq!(
                engine.bufs.part_tmp2[id], 1,
                "`SPRK.cpp:154`: the bolt uses the deferred bending mode"
            );
        }
        break;
    }
    assert!(
        found,
        "no frame in 64 produced two Tesla bolts; the per-neighbour loop is not running"
    );
}

/// `PCLN.cpp:121-152`: a `PHOT` payload clones into eight directed photons,
/// a `LIFE` payload into nine genome copies. The single `FLAG_SPAWN_CLONE`
/// path must not collapse them into one.
#[test]
fn test_pcln_photon_and_life_clone_shapes() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(96, 96);
    // Keep the GoL generation away from the freshly created `LIFE` ring.
    engine.gspeed = 1000;
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    // --- LIFE payload -----------------------------------------------------
    fill_rect(&mut engine, 40, 40, 40, 40, PT_PCLN);
    let clone = engine.bufs.particle_index_at(40, 40).expect("PCLN");
    engine.bufs.part_ctype[clone] = PT_LIFE;
    engine.bufs.part_tmp[clone] = 7;
    engine.bufs.part_life[clone] = 10;
    engine.bufs.mark_particle_slot_dirty(clone);

    step(&mut engine);

    assert_eq!(
        count_type(&engine, PT_LIFE),
        8,
        "`PCLN.cpp:142-152`: the 3x3 block minus the occupied clone pixel"
    );
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] == PT_LIFE {
            assert_eq!(
                engine.bufs.part_ctype[id], 7,
                "`create_part(-1, ..., PT_LIFE, tmp)` puts the genome in ctype"
            );
        }
    }

    // --- PHOT payload -----------------------------------------------------
    engine.clear();
    engine.gspeed = 1000;
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 40, 40, 40, 40, PT_PCLN);
    let clone = engine.bufs.particle_index_at(40, 40).expect("PCLN");
    engine.bufs.part_ctype[clone] = PT_PHOT;
    engine.bufs.part_life[clone] = 10;
    engine.bufs.mark_particle_slot_dirty(clone);

    step(&mut engine);

    assert_eq!(
        count_type(&engine, PT_PHOT),
        8,
        "`PCLN.cpp:121-141`: one photon per neighbour, not a single clone"
    );
    let mut dirs: Vec<(i32, i32)> = Vec::new();
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] == PT_PHOT {
            let vx = engine.bufs.part_vx[id];
            let vy = engine.bufs.part_vy[id];
            let dir = ((vx / 3.0).round() as i32, (vy / 3.0).round() as i32);
            assert!(
                (dir.0 != 0 || dir.1 != 0)
                    && dir.0.abs() <= 1
                    && dir.1.abs() <= 1
                    && (vx - (dir.0 as f32) * 3.0).abs() < 1e-3
                    && (vy - (dir.1 as f32) * 3.0).abs() < 1e-3,
                "photon velocity must be (rx*3, ry*3), got ({vx}, {vy})"
            );
            assert!(
                !dirs.contains(&dir),
                "two photons carry the same direction {dir:?}"
            );
            dirs.push(dir);
        }
    }
    assert_eq!(dirs.len(), 8, "all eight directions must be represented");
}

/// `ETRD.cpp:93-160`: the ordered 12 px neighbour table runs first, then the
/// pool scan finds an electrode anywhere on the canvas (bounded by `tmp2`).
#[test]
fn test_etrd_pool_scan_reaches_beyond_twelve_pixels() {
    let _guard = gpu_lock();

    // --- a target outside the 12 px table is reached ----------------------
    let mut engine = create_test_engine(128, 160);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 64, 40, 64, 40, PT_ETRD);
    fill_rect(&mut engine, 64, 100, 64, 100, PT_ETRD);
    spark_at(&mut engine, 64, 40);
    let source = engine
        .bufs
        .particle_index_at(64, 40)
        .expect("the sparked source");
    // `life_decay_kernel` runs before the reactions, so a stored 2 is the
    // `life == 1` window `SPRK.cpp:101` gates on.
    engine.bufs.part_life[source] = 2;
    engine.bufs.mark_particle_slot_dirty(source);

    step(&mut engine);

    assert_eq!(
        engine.bufs.part_ty[source], PT_ETRD,
        "`SPRK.cpp:110`: the source recovers after arcing"
    );
    assert_eq!(
        engine.bufs.part_life[source], 20,
        "`SPRK.cpp:110`: the source takes the 20 frame cooldown"
    );
    let target = engine
        .bufs
        .particle_index_at(64, 100)
        .expect("the far electrode");
    assert_eq!(
        engine.bufs.part_ty[target], PT_SPRK,
        "a 60 px distant electrode must be found by the pool scan"
    );
    assert_eq!(
        engine.bufs.part_ctype[target], PT_ETRD,
        "`SPRK.cpp:112`: the target spark remembers ETRD"
    );
    assert_eq!(
        engine.bufs.part_life[target], 9,
        "`SPRK.cpp:113`: the target spark starts with nine frames"
    );

    // --- `tmp2` still caps the search ------------------------------------
    let mut capped = create_test_engine(128, 160);
    capped.config_mut().gravity_mode = GravityMode::Off;
    capped.config_mut().air_mode = AirMode::NoUpdate;
    capped.config_mut().aheat_enable = false;

    fill_rect(&mut capped, 64, 40, 64, 40, PT_ETRD);
    fill_rect(&mut capped, 64, 100, 64, 100, PT_ETRD);
    spark_at(&mut capped, 64, 40);
    let source = capped
        .bufs
        .particle_index_at(64, 40)
        .expect("the capped source");
    capped.bufs.part_life[source] = 2;
    capped.bufs.part_tmp2[source] = 20;
    capped.bufs.mark_particle_slot_dirty(source);

    step(&mut capped);

    assert_eq!(
        capped.bufs.part_ty[source], PT_SPRK,
        "`ETRD.cpp:101-102` + `tmp2 = 20`: a 60 px target is out of range"
    );
    let target = capped
        .bufs
        .particle_index_at(64, 100)
        .expect("the capped target");
    assert_eq!(
        capped.bufs.part_ty[target], PT_ETRD,
        "the out-of-range electrode must stay idle"
    );
}

/// `DEUT.cpp:88-113` `tryExpand`: every empty neighbour receives a new `DEUT`
/// (life 0, the parent's temperature) and the parent spends one life per
/// created pixel.
#[test]
fn test_deut_fills_every_empty_neighbour_with_one_life() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(96, 96);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 40, 40, 40, 40, PT_DEUT);
    let parent = engine
        .bufs
        .particle_index_at(40, 40)
        .expect("the deuterium pixel");
    // `10000 % 250 == 0`, so the fractional carry is never rolled, and
    // gravity is zero: `maxlife = int(10000/251) - 1 = 38` (`DEUT.cpp:56-66`).
    engine.bufs.part_temp[parent] = 249.0;
    engine.bufs.part_life[parent] = 46;
    engine.bufs.mark_particle_slot_dirty(parent);

    step(&mut engine);

    assert_eq!(
        count_type(&engine, PT_DEUT),
        9,
        "`DEUT.cpp:92-113`: the parent plus all eight empty neighbours"
    );
    assert_eq!(
        engine.bufs.part_life[parent], 38,
        "one life is spent per created pixel, stopping at maxlife"
    );
    for id in 0..engine.bufs.capacity {
        if id != parent && engine.bufs.part_ty[id] == PT_DEUT {
            assert_eq!(
                engine.bufs.part_life[id], 0,
                "`DEUT.cpp:109`: a new pixel starts with life 0"
            );
            assert!(
                (engine.bufs.part_temp[id] - 249.0).abs() < 1e-3,
                "`DEUT.cpp:108`: the child inherits the parent temperature"
            );
        }
    }
}
