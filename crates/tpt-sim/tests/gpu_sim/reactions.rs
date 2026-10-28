use super::*;
use cubecl::bytes::Bytes;

#[test]
fn test_element_reactions_fire_water_acid_c4_clne_sprk() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(320, 200);

    fill_rect(&mut engine, 140, 140, 149, 149, PT_GUNP);
    fill_rect(&mut engine, 160, 140, 169, 149, PT_C4);
    fill_rect(&mut engine, 140, 139, 149, 139, PT_FIRE);
    fill_rect(&mut engine, 160, 139, 169, 139, PT_FIRE);

    let gunp0 = count_type(&engine, PT_GUNP);
    let c4_0 = count_type(&engine, PT_C4);
    let fire0 = count_type(&engine, PT_FIRE);
    assert_eq!((gunp0, c4_0, fire0), (100, 100, 20));

    let mut peak_pressure = 0.0f32;
    let mut peak_fire = fire0;
    for _ in 0..60 {
        step(&mut engine);
        peak_pressure = peak_pressure.max(max_air_pressure(&engine));
        peak_fire = peak_fire.max(count_type(&engine, PT_FIRE));
    }

    assert!(
        peak_fire > fire0,
        "fire must spread into the explosives: {fire0} -> {peak_fire} fire pixels"
    );
    assert!(
        count_type(&engine, PT_GUNP) < gunp0,
        "gunpowder must ignite ({} of {gunp0} left)",
        count_type(&engine, PT_GUNP)
    );
    assert!(
        count_type(&engine, PT_C4) < c4_0,
        "C-4 must detonate ({} of {c4_0} left)",
        count_type(&engine, PT_C4)
    );
    assert!(
        peak_pressure > 0.5,
        "explosions must raise air pressure, peak pv = {peak_pressure}"
    );

    engine.clear();
    engine.config_mut().gravity_mode = GravityMode::Vertical;
    engine.config_mut().air_mode = AirMode::On;

    // Four identical slabs, each with a 4-row acid pool on top. The pool is
    // walled in on both sides (WL_WALL) so the acid stays in contact instead of
    // spreading across the canvas. `PlaceWalls` takes coarse `CELL x CELL` block
    // coordinates, so the pool bounds are divided down to block indices.
    // The wood slab sits in its own column so the fire it produces cannot drift
    // into the immune controls or the metal pool.
    let slabs: [(i32, i32, i32, u32); 4] = [
        (12, 67, 120, PT_GLAS),
        (12, 67, 140, PT_DMND),
        (12, 67, 160, PT_METL),
        (120, 175, 160, PT_WOOD),
    ];
    let cell = CELL as i32;
    let mut wall_cells = Vec::new();
    for (x0, x1, y, ty) in slabs {
        fill_rect(&mut engine, x0, y, x1, y + 5, ty);
        fill_rect(&mut engine, x0, y - 4, x1, y - 1, PT_ACID);
        let pool_cy = (y - 4).div_euclid(cell);
        wall_cells.push((x0.div_euclid(cell) - 1, pool_cy));
        wall_cells.push((x1.div_euclid(cell) + 1, pool_cy));
    }
    engine.apply_command(SimCommand::PlaceWalls {
        cells: wall_cells,
        wall_id: WL_WALL,
        fan_vel: None,
    });

    let wood0 = count_type(&engine, PT_WOOD);
    let metl0 = count_type(&engine, PT_METL);
    let glas0 = count_type(&engine, PT_GLAS);
    let dmnd0 = count_type(&engine, PT_DMND);
    assert_eq!(wood0, 56 * 6);
    assert_eq!(metl0, 56 * 6);
    assert_eq!(glas0, 56 * 6);
    assert_eq!(dmnd0, 56 * 6);

    for _ in 0..150 {
        step(&mut engine);
    }

    assert!(
        count_type(&engine, PT_WOOD) < wood0,
        "acid must corrode wood ({} of {wood0} left)",
        count_type(&engine, PT_WOOD)
    );
    assert!(
        count_type(&engine, PT_METL) < metl0,
        "acid must corrode metal ({} of {metl0} left)",
        count_type(&engine, PT_METL)
    );
    assert_eq!(
        count_type(&engine, PT_GLAS),
        glas0,
        "glass must be immune to acid"
    );
    assert_eq!(
        count_type(&engine, PT_DMND),
        dmnd0,
        "diamond must be immune to acid"
    );

    engine.clear();
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::Off;

    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(100, 100)],
        elem_ty: PT_CLNE,
        replace_mode: true,
        dcolour: 0,
    });
    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(101, 100)],
        elem_ty: PT_WOOD,
        replace_mode: true,
        dcolour: 0,
    });
    let clone_id = engine
        .bufs
        .particle_index_at(100, 100)
        .expect("the clone particle must exist");
    assert_eq!(engine.bufs.part_ty[clone_id], PT_CLNE);
    let wood_before_clone = count_type(&engine, PT_WOOD);
    assert_eq!(wood_before_clone, 1);

    for _ in 0..12 {
        step(&mut engine);
    }

    assert_eq!(
        engine.bufs.part_ctype[clone_id], PT_WOOD,
        "the clone must capture the adjacent element into ctype"
    );
    assert!(
        count_type(&engine, PT_WOOD) > wood_before_clone,
        "the clone must spawn copies ({} wood particles)",
        count_type(&engine, PT_WOOD)
    );

    engine.clear();
    fill_rect(&mut engine, 100, 100, 119, 100, PT_METL);
    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(100, 100)],
        elem_ty: PT_SPRK,
        replace_mode: true,
        dcolour: 0,
    });
    let spark_id = engine
        .bufs
        .particle_index_at(100, 100)
        .expect("the spark particle must exist");
    assert_eq!(engine.bufs.part_ty[spark_id], PT_SPRK);
    // `create_part`'s spark branch assigns the life and the conductor itself.
    assert_eq!(
        engine.bufs.part_life[spark_id], 4,
        "a fresh spark lives 4 frames"
    );
    assert_eq!(
        engine.bufs.part_ctype[spark_id], PT_METL,
        "the spark must remember the conductor it replaced"
    );
    let wire0 = count_type(&engine, PT_METL);
    assert_eq!(wire0, 19, "the spark replaced one wire pixel");

    let mut peak_sparks = 0usize;
    let mut settled = false;
    for _ in 0..200 {
        step(&mut engine);
        let sparks = count_type(&engine, PT_SPRK);
        peak_sparks = peak_sparks.max(sparks);
        if peak_sparks > 5 && sparks == 0 && count_type(&engine, PT_METL) == wire0 + 1 {
            settled = true;
            break;
        }
    }

    assert!(
        peak_sparks > 5,
        "the spark must conduct along the metal wire, peak = {peak_sparks} sparks"
    );
    assert!(
        settled,
        "the spark wave must die out and the wire revert to metal \
         ({} sparks, {} metal pixels left)",
        count_type(&engine, PT_SPRK),
        count_type(&engine, PT_METL)
    );
}

/// `(x, y)` of every live `LIFE` particle on the canvas, sorted.
fn life_cells(engine: &SimulationEngine) -> Vec<(i32, i32)> {
    let mut cells = Vec::new();
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] != PT_LIFE {
            continue;
        }
        cells.push((
            engine.bufs.part_x[id].round() as i32,
            engine.bufs.part_y[id].round() as i32,
        ));
    }
    cells.sort_unstable();
    cells
}

/// A `LIFE` glider under the builtin GOL rule advances exactly one generation.
///
/// The painted pattern is the classic B3/S23 glider
///
/// ```text
///  . X .
///  . . X
///  X X X
/// ```
///
/// anchored at `(20, 22)`. One `Simulation::SimulateGoL` generation leaves the
/// three survivors `(22, 21)`, `(21, 22)`, `(22, 22)` and births `(20, 21)` and
/// `(21, 23)`, i.e. the rotated glider phase.
#[test]
fn test_life_glider_advances_one_generation() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(128, 128);

    // `ctype` 0 = builtin GOL (B3/S23), exactly what the `SC_LIFE` palette
    // binds (`ElementTool(PMAP(0, PT_LIFE))`).
    let glider = [(21, 20), (22, 21), (20, 22), (21, 22), (22, 22)];
    tpt_sim::gol_kernel::paint_life(&mut engine, &glider, 0, true);
    assert_eq!(
        life_cells(&engine),
        {
            let mut painted = glider.to_vec();
            painted.sort_unstable();
            painted
        },
        "the GOL brush must paint exactly the requested cells"
    );
    assert_eq!(
        engine.bufs.part_ctype[engine
            .bufs
            .particle_index_at(21, 20)
            .expect("glider cell missing")],
        0,
        "painted LIFE must carry the configured ruleset in ctype"
    );

    let generations = engine.gol.generations;
    step(&mut engine);
    assert_eq!(
        engine.gol.generations,
        generations + 1,
        "exactly one generation per step"
    );

    let mut expected = vec![(20, 21), (22, 21), (21, 22), (22, 22), (21, 23)];
    expected.sort_unstable();
    assert_eq!(
        life_cells(&engine),
        expected,
        "the glider must advance one B3/S23 generation"
    );
    // A generation that only moves cells leaves five live particles behind.
    assert_eq!(count_type(&engine, PT_LIFE), 5);
    assert!(
        engine.gol.is_active(),
        "the pass must stay armed while LIFE lives"
    );
}

#[test]
fn test_create_hooks_seed_cflm_and_tron() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);

    fill_rect(&mut engine, 10, 10, 10, 10, PT_HFLM);
    fill_rect(&mut engine, 20, 20, 20, 20, PT_TRON);

    let cflm = engine
        .bufs
        .particle_index_at(10, 10)
        .expect("painted CFLM missing");
    let life = engine.bufs.part_life[cflm];
    assert!(
        (50..=199).contains(&life),
        "CFLM create hook must roll life in 50..=199, got {life}"
    );

    let tron = engine
        .bufs
        .particle_index_at(20, 20)
        .expect("painted TRON missing");
    assert_eq!(engine.bufs.part_life[tron], 5, "TRON create life");
    assert_eq!(engine.bufs.part_tmp2[tron], 4, "TRON create tail length");
    assert_eq!(
        engine.bufs.part_tmp[tron] & 1,
        1,
        "TRON create must mark the head bit"
    );
    let hue = engine.bufs.part_tmp[tron] >> 7;
    assert!(
        (0..360).contains(&hue),
        "TRON create hue out of range: {hue}"
    );

    // `PROP_LIFE_KILL` plus a zero life would delete the painted particle on
    // the first tick; the create hook is what keeps it alive.
    step(&mut engine);
    assert!(
        count_type(&engine, PT_TRON) >= 1,
        "painted TRON must survive its first tick"
    );
}

/// The rest of the `Create` hooks and the `create_part` guards the brush
/// depends on (`Simulation.cpp:1843-1985`).
#[test]
fn test_create_hooks_and_brush_creation_rules() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(128, 128);
    engine.config_mut().gravity_mode = GravityMode::Off;

    fill_rect(&mut engine, 10, 10, 10, 10, PT_FIRE);
    let fire = engine.bufs.particle_index_at(10, 10).expect("FIRE placed");
    let life = engine.bufs.part_life[fire];
    assert!(
        (120..=169).contains(&life),
        "FIRE create hook must roll life in 120..=169, got {life}"
    );

    // `NEUT.cpp:252-258`: a painted neutron starts with its 480..959 timer.
    fill_rect(&mut engine, 20, 20, 20, 20, PT_NEUT);
    let neut = engine.bufs.particle_index_at(20, 20).expect("NEUT placed");
    let life = engine.bufs.part_life[neut];
    assert!(
        (480..=959).contains(&life),
        "NEUT create hook must roll life in 480..=959, got {life}"
    );

    // `ELEC.cpp:44-50`: 680 life and a 2 px/frame kick.
    fill_rect(&mut engine, 30, 30, 30, 30, PT_ELEC);
    let elec = engine.bufs.particle_index_at(30, 30).expect("ELEC placed");
    assert_eq!(engine.bufs.part_life[elec], 680, "ELEC create life");
    let speed = engine.bufs.part_vx[elec].hypot(engine.bufs.part_vy[elec]);
    assert!((speed - 2.0).abs() < 0.01, "ELEC create speed {speed}");

    // `PHOT.cpp:176-184`: 3 px/frame on the eight-direction lattice.
    fill_rect(&mut engine, 40, 40, 40, 40, PT_PHOT);
    let phot = engine.bufs.particle_index_at(40, 40).expect("PHOT placed");
    let speed = engine.bufs.part_vx[phot].hypot(engine.bufs.part_vy[phot]);
    assert!((speed - 3.0).abs() < 0.01, "PHOT create speed {speed}");
    let axis = engine.bufs.part_vx[phot]
        .abs()
        .max(engine.bufs.part_vy[phot].abs());
    assert!(
        (axis - 3.0).abs() < 0.01 || (axis - 2.1213).abs() < 0.01,
        "a photon leaves along one of eight directions, got vx {} vy {}",
        engine.bufs.part_vx[phot],
        engine.bufs.part_vy[phot]
    );

    // `Simulation.cpp:1852-1878`: a spark never allocates a slot, so painting
    // it on empty space does nothing at all.
    let before = engine.bufs.active_count;
    fill_rect(&mut engine, 50, 50, 50, 50, PT_SPRK);
    assert_eq!(
        engine.bufs.active_count, before,
        "painting SPRK on empty space must not create a particle"
    );
    assert!(engine.bufs.particle_index_at(50, 50).is_none());

    // ... and on a conductor it takes the conductor over, remembering it.
    fill_rect(&mut engine, 60, 60, 60, 60, PT_METL);
    fill_rect(&mut engine, 60, 60, 60, 60, PT_SPRK);
    let spark = engine
        .bufs
        .particle_index_at(60, 60)
        .expect("spark created");
    assert_eq!(engine.bufs.part_ty[spark], PT_SPRK);
    assert_eq!(engine.bufs.part_life[spark], 4);
    assert_eq!(engine.bufs.part_ctype[spark], PT_METL);

    // `Simulation.cpp:1875-1877`: the good conductors heat up by 10 K.
    fill_rect(&mut engine, 70, 70, 70, 70, PT_IRON);
    let iron = engine.bufs.particle_index_at(70, 70).expect("IRON placed");
    engine.bufs.part_temp[iron] = 300.0;
    engine.bufs.mark_particle_slot_dirty(iron);
    fill_rect(&mut engine, 70, 70, 70, 70, PT_SPRK);
    let spark = engine
        .bufs
        .particle_index_at(70, 70)
        .expect("spark created");
    assert!(
        (engine.bufs.part_temp[spark] - 310.0).abs() < 0.01,
        "a charged IRON wire heats by 10 K, got {}",
        engine.bufs.part_temp[spark]
    );

    // `Element::ctypeDrawVInCtype`: painting FILT over a particle writes it into
    // `ctype` instead of replacing it.
    fill_rect(&mut engine, 80, 80, 80, 80, PT_WOOD);
    let wood = engine.bufs.particle_index_at(80, 80).expect("WOOD placed");
    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(80, 80)],
        elem_ty: PT_FILT,
        replace_mode: false,
        dcolour: 0,
    });
    assert_eq!(
        engine.bufs.particle_index_at(80, 80),
        Some(wood),
        "the painted-over particle must survive"
    );
    assert_eq!(engine.bufs.part_ty[wood], PT_WOOD);
}

/// Place an energy particle over an existing matter pixel without disturbing
/// it: `create_part(-3, ...)`, the "skip pmap checks" form TPT's own photon
/// spawners use (`Simulation.cpp:2036`). The brush (`-2`) refuses a pixel that
/// already holds matter.
fn place_energy(engine: &mut SimulationEngine, points: Vec<(i32, i32)>, ty: u32) {
    for (x, y) in points {
        engine.bufs.create_part(
            x,
            y,
            ty,
            tpt_sim::buffers::CreationSource::SkipChecks,
            0,
            &engine.registry,
        );
    }
    engine.bufs.sync_host_to_gpu();
}

/// `GOLD.cpp:91-97` shields the neutron standing on it, and `POLO.cpp:58-108`
/// absorbs a `PROT` into `tmp2` and decays into `PLUT` once `tmp2` reaches 10.
#[test]
fn test_gold_shields_neutrons_and_polo_decays() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(200, 160);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 20, 20, 20, 20, PT_POLO);
    let polo = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_POLO)
        .expect("POLO placed");
    engine.bufs.part_tmp2[polo] = 9;
    engine.bufs.part_temp[polo] = 300.0;
    engine.bufs.mark_particle_slot_dirty(polo);
    place_energy(&mut engine, vec![(20, 20)], PT_PROT);
    assert_eq!(count_type(&engine, PT_PROT), 1, "the proton must be placed");

    step(&mut engine);
    assert_eq!(
        count_type(&engine, PT_PROT),
        0,
        "POLO must absorb the proton on its own pixel"
    );
    let polo = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_POLO)
        .expect("POLO survives the absorption");
    assert_eq!(
        engine.bufs.part_tmp2[polo], 10,
        "the absorbed proton must land in tmp2"
    );

    step(&mut engine);
    assert_eq!(
        count_type(&engine, PT_POLO),
        0,
        "POLO must decay once tmp2 reaches its limit"
    );
    assert_eq!(count_type(&engine, PT_PLUT), 1, "the decay product is PLUT");

    let mut shielded = create_test_engine(200, 160);
    shielded.config_mut().gravity_mode = GravityMode::Off;
    shielded.config_mut().air_mode = AirMode::NoUpdate;
    shielded.config_mut().aheat_enable = false;
    fill_rect(&mut shielded, 40, 40, 45, 45, PT_GOLD);
    let mut points = Vec::new();
    for y in 40..=45 {
        for x in 40..=45 {
            points.push((x, y));
        }
    }
    place_energy(&mut shielded, points.clone(), PT_NEUT);
    assert_eq!(
        count_type(&shielded, PT_NEUT),
        36,
        "one neutron per gold pixel"
    );
    for _ in 0..6 {
        step(&mut shielded);
    }
    let gold_kept = count_type(&shielded, PT_NEUT);
    assert!(
        gold_kept < 36,
        "gold must absorb neutrons standing on it ({gold_kept} of 36 kept)"
    );

    // Control: the same geometry on a solid without the rule keeps every
    // neutron, so the decrease above is GOLD's own rule and not a reaction
    // pass eating the energy layer.
    let mut control = create_test_engine(200, 160);
    control.config_mut().gravity_mode = GravityMode::Off;
    control.config_mut().air_mode = AirMode::NoUpdate;
    control.config_mut().aheat_enable = false;
    fill_rect(&mut control, 40, 40, 45, 45, PT_BRCK);
    place_energy(&mut control, points, PT_NEUT);
    for _ in 0..6 {
        step(&mut control);
    }
    let control_kept = count_type(&control, PT_NEUT);
    assert_eq!(
        control_kept, 36,
        "a brick must not absorb neutrons ({control_kept} of 36 kept)"
    );
    assert!(
        gold_kept < control_kept,
        "gold must keep fewer neutrons than brick ({gold_kept} vs {control_kept})"
    );
}

/// `PLNT.cpp:61-262` tree growth and `PLNT.cpp:292-300` smoke feeding.
///
/// A tree-mode plant (`ctype & 1`) grows new plant, wood or goo pixels into
/// free neighbours, and a usual plant eats the `CO2`/`SMKE` next to it.
#[test]
fn test_plant_grows_and_eats_smoke() {
    let _guard = gpu_lock();

    let mut engine = create_test_engine(160, 160);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 80, 80, 80, 80, PT_PLNT);
    let plant = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_PLNT)
        .expect("PLNT placed");
    // Genome (`PLNT.h:11-14`): bit 0 = growth bit, bits 1-2 phase, bits 3-5
    // direction, bits 6-11 colour, bits 12-19 vigor. A seed that drank water
    // before germinating carries its water count as vigor; 20 is a stem.
    // `SEED.cpp:110-113` hands a germinating seed `life = 15 * water` and keeps
    // the water count as the genome's vigor, so a painted tree starts the same
    // way: growth bit set, vigor 20, and a 300-frame stem countdown.
    engine.bufs.part_ctype[plant] = 1 | (20 << 12);
    engine.bufs.part_life[plant] = 15 * 20;
    engine.bufs.part_temp[plant] = 293.15;
    engine.bufs.mark_particle_slot_dirty(plant);

    for _ in 0..120 {
        step(&mut engine);
    }
    let grown =
        count_type(&engine, PT_PLNT) + count_type(&engine, PT_WOOD) + count_type(&engine, PT_GOO);
    assert!(
        grown > 1,
        "tree-mode PLNT must grow into free space ({grown} plant pixels)"
    );

    let mut fed = create_test_engine(160, 160);
    fed.config_mut().gravity_mode = GravityMode::Off;
    fed.config_mut().air_mode = AirMode::NoUpdate;
    fed.config_mut().aheat_enable = false;

    fill_rect(&mut fed, 80, 80, 80, 80, PT_PLNT);
    let plant = (0..fed.bufs.capacity)
        .find(|&id| fed.bufs.part_ty[id] == PT_PLNT)
        .expect("PLNT placed");
    fed.bufs.part_ctype[plant] = 0;
    fed.bufs.mark_particle_slot_dirty(plant);

    // `PLNT.cpp:292-300`: a plant next to `CO2`/`SMKE` eats it and refills its
    // life to 60..119. Gases drift, so one `CO2` is kept on the pixel above.
    let mut fed_ok = false;
    for _ in 0..300 {
        if fed.bufs.part_ty[plant] != PT_PLNT {
            break;
        }
        if fed.bufs.particle_index_at(80, 79).is_none() {
            fill_rect(&mut fed, 80, 79, 80, 79, PT_CO2);
        }
        step(&mut fed);
        if fed.bufs.part_life[plant] > 0 {
            fed_ok = true;
            break;
        }
    }
    assert!(
        fed_ok,
        "PLNT must consume an adjacent CO2 pixel and refill its life"
    );
}

/// `NEUT.cpp:56-244`: the neutron transmutation table, `PLUT` fission and the
/// `DEUT` detonation.
///
/// The fission roll is `chance(3 + pv, 1000)` with an **unsigned** sum
/// (`NEUT.cpp:60`), so a strong vacuum makes it certain; the air field is frozen
/// (`AirMode::NoUpdate`) so the vacuum survives the step.
#[test]
fn test_neutron_transmutations_fission_and_deut_burst() {
    let _guard = gpu_lock();

    // --- YEST -> DYST is unconditional (`NEUT.cpp:138-141`) ---------------
    let mut engine = create_test_engine(200, 160);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 20, 20, 20, 20, PT_YEST);
    place_energy(&mut engine, vec![(20, 20)], PT_NEUT);
    step(&mut engine);
    assert_eq!(
        count_type(&engine, PT_YEST),
        0,
        "a neutron must transmute the YEST it touches"
    );
    assert_eq!(count_type(&engine, PT_DYST), 1, "YEST becomes DYST");

    let mut fission = create_test_engine(200, 160);
    fission.config_mut().gravity_mode = GravityMode::Off;
    fission.config_mut().air_mode = AirMode::NoUpdate;
    fission.config_mut().aheat_enable = false;

    fill_rect(&mut fission, 60, 60, 65, 65, PT_PLUT);
    let mut points = Vec::new();
    for y in 60..=65 {
        for x in 60..=65 {
            points.push((x, y));
        }
    }
    place_energy(&mut fission, points, PT_NEUT);
    assert_eq!(
        count_type(&fission, PT_NEUT),
        36,
        "one neutron per PLUT pixel"
    );
    for pv in fission.bufs.air_pv.iter_mut() {
        *pv = -50.0;
    }
    fission.bufs.mark_air_dirty();

    step(&mut fission);
    let plut_left = count_type(&fission, PT_PLUT);
    assert!(
        plut_left < 36,
        "neutrons must fission PLUT under vacuum ({plut_left} of 36 left)"
    );
    assert!(
        count_type(&fission, PT_NEUT) > 36,
        "the 2-in-3 branch leaves a second neutron ({} neutrons)",
        count_type(&fission, PT_NEUT)
    );
    assert!(
        count_type(&fission, PT_URAN) + count_type(&fission, PT_LAVA) > 0,
        "the 1-in-3 branch leaves URAN or LAVA(PLUT)"
    );

    // --- PLUT spontaneous fission (`PLUT.cpp:58-65`) ----------------------
    let mut vent = create_test_engine(200, 160);
    vent.config_mut().gravity_mode = GravityMode::Off;
    vent.config_mut().air_mode = AirMode::NoUpdate;
    vent.config_mut().aheat_enable = false;
    fill_rect(&mut vent, 40, 40, 59, 54, PT_PLUT);
    // `int(5 * 200) == 1000` makes the second `chance` always hit, so only the
    // one-in-a-hundred gate is left and 300 pixels cannot all miss.
    for pv in vent.bufs.air_pv.iter_mut() {
        *pv = 200.0;
    }
    vent.bufs.mark_air_dirty();

    let mut born: Vec<(usize, i32)> = Vec::new();
    for _ in 0..30 {
        step(&mut vent);
        for id in 0..vent.bufs.capacity {
            if vent.bufs.part_ty[id] == PT_NEUT && !born.iter().any(|&(b, _)| b == id) {
                born.push((id, vent.bufs.part_life[id]));
            }
        }
    }
    assert!(
        !born.is_empty(),
        "plutonium under pressure must release neutrons"
    );
    for &(id, life) in &born {
        // The movement pass may already have decremented the fresh timer.
        assert!(
            (470..=959).contains(&life),
            "released neutron {id} must carry the NEUT create hook's life, got {life}"
        );
    }

    // --- DEUT detonation (`NEUT.cpp:106-112`, `DeutExplosion`) -------------
    let mut det = create_test_engine(200, 160);
    det.config_mut().gravity_mode = GravityMode::Off;
    det.config_mut().air_mode = AirMode::NoUpdate;
    det.config_mut().aheat_enable = false;

    fill_rect(&mut det, 40, 40, 40, 40, PT_DEUT);
    let deut = (0..det.bufs.capacity)
        .find(|&id| det.bufs.part_ty[id] == PT_DEUT)
        .expect("DEUT placed");
    det.bufs.part_life[deut] = 200;
    det.bufs.part_temp[deut] = 300.0;
    det.bufs.mark_particle_slot_dirty(deut);
    place_energy(&mut det, vec![(40, 40)], PT_NEUT);
    for pv in det.bufs.air_pv.iter_mut() {
        *pv = -50.0;
    }
    det.bufs.mark_air_dirty();

    step(&mut det);
    assert_eq!(
        count_type(&det, PT_DEUT),
        0,
        "the neutron must detonate DEUT"
    );
    assert_eq!(
        count_type(&det, PT_NEUT),
        5,
        "DeutExplosion: the source neutron plus life/50 = 4 products"
    );
    assert!(
        max_temp_of_type(&det, PT_NEUT) > 9000.0,
        "the burst is born at temp + life*500, clamped to MAX_TEMP"
    );

    // --- PROT + DEUT implosion (`PROT.cpp:78-85`, `DeutImplosion`) ---------
    let mut implo = create_test_engine(200, 160);
    implo.config_mut().gravity_mode = GravityMode::Off;
    implo.config_mut().air_mode = AirMode::NoUpdate;
    implo.config_mut().aheat_enable = false;

    fill_rect(&mut implo, 40, 40, 40, 40, PT_DEUT);
    let deut = (0..implo.bufs.capacity)
        .find(|&id| implo.bufs.part_ty[id] == PT_DEUT)
        .expect("DEUT placed");
    implo.bufs.part_life[deut] = 200;
    // `DEUT.cpp:56-66`: keep `life < maxlife`. At 47 K `maxlife` is 203, so
    // the `tryExpand` branch stays out of this implosion test; at the DEUT
    // default temperature the particle would fill its eight neighbours first
    // (`DEUT.cpp:88-113`) and those fresh pixels would outlive the parent.
    implo.bufs.part_temp[deut] = 47.0;
    implo.bufs.mark_particle_slot_dirty(deut);
    place_energy(&mut implo, vec![(40, 40)], PT_PROT);
    let trigger = (0..implo.bufs.capacity)
        .find(|&id| implo.bufs.part_ty[id] == PT_PROT)
        .expect("the triggering proton was placed");
    // `chance(4 - pv + life/100, 200)` always hits at this pressure.
    for pv in implo.bufs.air_pv.iter_mut() {
        *pv = -200.0;
    }
    implo.bufs.mark_air_dirty();

    step(&mut implo);
    step(&mut implo);
    assert_eq!(
        count_type(&implo, PT_DEUT),
        0,
        "the proton must detonate the deuterium"
    );
    let protons: Vec<usize> = (0..implo.bufs.capacity)
        .filter(|&id| implo.bufs.part_ty[id] == PT_PROT)
        .collect();
    // `Element_PROT::create` gives every proton 680 life, the trigger
    // included, so the burst is "every proton but the one that detonated it".
    let burst: Vec<usize> = protons
        .iter()
        .copied()
        .filter(|&id| id != trigger)
        .collect();
    assert!(
        burst.len() >= 3,
        "`DeutImplosion` creates life/50 = 4 protons in the same pixel (got {} of 4, proton-proton fusion may merge one)",
        burst.len()
    );
    assert!(
        burst.iter().all(|&id| implo.bufs.part_temp[id] > 9000.0),
        "every proton of the burst is born at the implosion temperature"
    );
}

/// `VIBR.cpp:118-142`: a charged vibranium detonates on its last countdown
/// frame into `EXOT` plus `ELEC`/`PHOT`/`BREC` at 7000 K.
///
/// The countdown is `PROP_LIFE_DEC`, and the port's life pass decrements before
/// the special pass, so the detonation lands one frame before TPT's.
#[test]
fn test_vibranium_detonates_into_exot_and_energy() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(120, 120);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 60, 60, 60, 60, PT_VIBR);
    let vibr = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_VIBR)
        .expect("VIBR placed");
    engine.bufs.part_life[vibr] = 3;
    engine.bufs.part_tmp[vibr] = 0;
    engine.bufs.part_tmp2[vibr] = 0;
    engine.bufs.mark_particle_slot_dirty(vibr);

    step(&mut engine);
    step(&mut engine);
    let exot = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_EXOT)
        .expect("the charged vibranium turns into EXOT");
    assert!(
        (engine.bufs.part_temp[exot] - 9000.0).abs() < 100.0,
        "`VIBR.cpp:133` seeds 9000 K, found {}",
        engine.bufs.part_temp[exot]
    );
    assert_eq!(engine.bufs.part_life[exot], 1000, "EXOT's default life");
    assert!(
        (0..1000).contains(&engine.bufs.part_tmp2[exot]),
        "`VIBR.cpp:132`: tmp2 = rnd % 1000"
    );

    // The creation queue drains at the next reaction step.
    step(&mut engine);
    let energy = (0..engine.bufs.capacity)
        .any(|id| matches!(engine.bufs.part_ty[id], PT_ELEC | PT_PHOT | PT_BREC));
    assert!(energy, "the detonation emits ELEC/PHOT/BREC");
}

/// `QRTZ.cpp:93-141`: a crystal grows into the sampled free pixel, the child
/// inherits the parent's `temp` and `tmp2`, and one child in two is born dead
/// (`tmp = -1`).
///
/// The port used to materialise growth through the host spawn request, which
/// always produced a default-state `QRTZ` in a free neighbour.
#[test]
fn test_quartz_growth_inherits_parent_state() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(120, 120);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 60, 60, 60, 60, PT_QRTZ);
    let parent = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_QRTZ)
        .expect("QRTZ placed");
    engine.bufs.part_tmp[parent] = 20;
    engine.bufs.part_tmp2[parent] = 7;
    engine.bufs.part_temp[parent] = 500.0;
    engine.bufs.mark_particle_slot_dirty(parent);

    for _ in 0..40 {
        step(&mut engine);
    }

    let children: Vec<usize> = (0..engine.bufs.capacity)
        .filter(|&id| id != parent && engine.bufs.part_ty[id] == PT_QRTZ)
        .collect();
    assert!(
        !children.is_empty(),
        "a crystal with `tmp` left must grow (`QRTZ.cpp:106`)"
    );
    assert!(
        children.iter().any(|&c| engine.bufs.part_temp[c] > 400.0),
        "the child inherits the parent's temperature, not the default"
    );
    assert!(
        children.iter().any(|&c| engine.bufs.part_tmp[c] == -1),
        "one child in two is born dead (`tmp = -1`)"
    );
    assert!(
        children
            .iter()
            .all(|&c| (0..=10).contains(&engine.bufs.part_tmp2[c])),
        "the inherited `tmp2` stays inside the clamp"
    );
}

#[test]
fn test_glow_pressure_water_and_gel() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(200, 160);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    // --- `GLOW + GEL -> RSST` (`GLOW.cpp:72-80`) ---------------------------
    fill_rect(&mut engine, 60, 60, 61, 61, PT_GLOW);
    fill_rect(&mut engine, 62, 60, 62, 61, PT_GEL);
    assert_eq!(count_type(&engine, PT_GLOW), 4);

    step(&mut engine);
    assert_eq!(count_type(&engine, PT_GEL), 0, "gel must turn into residue");
    assert_eq!(count_type(&engine, PT_RSST), 2, "one RSST per gel pixel");
    assert_eq!(
        count_type(&engine, PT_GLOW),
        2,
        "only the glow pixels touching the gel die"
    );

    // --- `ctype` / `tmp` track the surrounding air (`GLOW.cpp:82-90`) -----
    engine.clear();
    fill_rect(&mut engine, 100, 100, 100, 100, PT_GLOW);
    let cell_x = 100u32 / CELL;
    let cell_y = 100u32 / CELL;
    let cell = (cell_y * 200u32.div_ceil(CELL) + cell_x) as usize;
    engine.bufs.air_pv[cell] = 20.0;
    engine.bufs.air_vx[cell] = 0.5;
    engine.bufs.air_vy[cell] = -0.25;
    engine.bufs.mark_air_dirty();

    step(&mut engine);
    let glow = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_GLOW)
        .expect("the glow pixel survived");
    assert_eq!(
        engine.bufs.part_ctype[glow], 320,
        "ctype = int(pv * 16) = int(20 * 16)"
    );
    // |(0.5 - 0.25) * 16| + |(vx + vy) * 64|, and the glow has no velocity of
    // its own with gravity off.
    assert_eq!(
        engine.bufs.part_tmp[glow],
        (0.25f32 * 16.0) as i32,
        "tmp sums the truncated air and own velocity terms"
    );

    // --- `GLOW + WATR -> DEUT` (`GLOW.cpp:64-71`) -------------------------
    engine.clear();
    let mut glow_points = Vec::new();
    let mut watr_points = Vec::new();
    for y in 60..80 {
        for x in 60..80 {
            if (x + y) % 2 == 0 {
                glow_points.push((x, y));
            } else {
                watr_points.push((x, y));
            }
        }
    }
    for (points, ty) in [(glow_points, PT_GLOW), (watr_points, PT_WATR)] {
        engine.apply_command(SimCommand::PlaceParticles {
            points,
            elem_ty: ty,
            replace_mode: true,
            dcolour: 0,
        });
    }
    assert_eq!(count_type(&engine, PT_GLOW), 200);
    for _ in 0..60 {
        step(&mut engine);
        if count_type(&engine, PT_DEUT) > 0 {
            break;
        }
    }
    assert!(
        count_type(&engine, PT_DEUT) > 0,
        "a glow in contact with water must leave DEUT with life = 10"
    );
    let deut = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_DEUT)
        .expect("DEUT exists");
    assert_eq!(
        engine.bufs.part_life[deut], 10,
        "DEUT keeps the 10-frame life"
    );
}

#[test]
fn test_mort_puffs_smoke_into_the_pixel_above() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(200, 160);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 80, 100, 83, 103, PT_MORT);
    assert_eq!(count_type(&engine, PT_SMKE), 0);

    step(&mut engine);
    assert_eq!(
        count_type(&engine, PT_SMKE),
        4,
        "only the top row of the mortar block has a free pixel above"
    );
    // MORT is a diffusing `TYPE_GAS`, so its pixels shuffle in the movement
    // pass that follows the reaction; only the count is stable.

    // An occupied pixel above blocks the puff (`create_part(-1, ...)` refuses).
    engine.clear();
    fill_rect(&mut engine, 80, 100, 83, 103, PT_MORT);
    fill_rect(&mut engine, 80, 99, 83, 99, PT_BRCK);
    step(&mut engine);
    assert_eq!(
        count_type(&engine, PT_SMKE),
        0,
        "a blocked pixel above the mortar creates nothing"
    );
}

#[test]
fn test_temperature_sensor_sparks_a_conductor() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(200, 160);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 100, 100, 100, 100, PT_TSNS);
    fill_rect(&mut engine, 101, 100, 101, 100, PT_METL);
    fill_rect(&mut engine, 100, 102, 100, 102, PT_GLAS);
    let glas = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_GLAS)
        .expect("glass placed");
    engine.bufs.part_temp[glas] = 2000.0;
    engine.bufs.mark_particle_slot_dirty(glas);
    let tsns = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_TSNS)
        .expect("sensor placed");
    engine.bufs.part_temp[tsns] = 300.0;
    engine.bufs.mark_particle_slot_dirty(tsns);

    // Frame 1 arms `life`; frame 2 consumes it and sparks the conductor.
    step(&mut engine);
    assert_eq!(
        engine.bufs.part_life[tsns], 1,
        "a hotter particle inside the radius must arm the sensor"
    );
    step(&mut engine);
    assert_eq!(
        engine.bufs.part_life[tsns], 1,
        "the scan re-arms the latch while the glass stays hot"
    );
    assert_eq!(
        count_type(&engine, PT_SPRK),
        1,
        "the armed sensor sparks its conducting neighbour"
    );
    let spark = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_SPRK)
        .expect("spark exists");
    assert_eq!(
        engine.bufs.part_ctype[spark], PT_METL,
        "the spark remembers METL"
    );
    assert_eq!(engine.bufs.part_life[spark], 4, "SPARK_LIFE");
}

#[test]
fn test_velocity_sensor_sparks_on_a_fast_neighbour() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(200, 160);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 100, 100, 100, 100, PT_VSNS);
    fill_rect(&mut engine, 101, 100, 101, 100, PT_METL);
    fill_rect(&mut engine, 101, 101, 101, 101, PT_WATR);
    let vsns = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_VSNS)
        .expect("sensor placed");
    engine.bufs.part_temp[vsns] = 274.0;
    engine.bufs.mark_particle_slot_dirty(vsns);
    let watr = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_WATR)
        .expect("water placed");
    engine.bufs.part_vx[watr] = 2.0;
    engine.bufs.part_vy[watr] = 0.0;
    engine.bufs.mark_particle_slot_dirty(watr);

    step(&mut engine);
    assert_eq!(
        engine.bufs.part_life[vsns], 1,
        "2 px/frame exceeds temp - 273.15 = 0.85"
    );
    step(&mut engine);
    assert_eq!(
        count_type(&engine, PT_SPRK),
        1,
        "the armed sensor sparks its conducting neighbour"
    );
}

/// `CRAY.cpp:84-140`: a spark with `life == 3` makes the laser cast a line of
/// its stored element away from the spark, and a `FILT` in the path colours
/// everything created after it.
///
/// The spark is placed with `life = 4` because the life pass decrements it
/// before the reaction passes run, which is the same frame TPT's generic
/// `PROP_LIFE_DEC` gives the laser a spark at `life == 3`.
#[test]
fn test_cray_beam_creates_a_coloured_line() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;

    fill_rect(&mut engine, 32, 32, 32, 32, PT_CRAY);
    engine.apply_command(SimCommand::SetParticleFieldAt {
        points: vec![(32, 32)],
        field: tpt_core::commands::ParticleField::Ctype,
        value: PT_BRCK as i64,
    });
    fill_rect(&mut engine, 35, 32, 35, 32, PT_FILT);
    spark_at(&mut engine, 31, 32);

    step(&mut engine);

    let bricks = count_type(&engine, PT_BRCK);
    assert!(
        bricks >= 5,
        "the laser must emit a line of BRICK away from the spark, got {bricks}"
    );

    // The beam colours only what it creates after the filter.
    let mut before_filter = 0usize;
    let mut after_filter = 0usize;
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] != PT_BRCK || engine.bufs.part_y[id] != 32.0 {
            continue;
        }
        let x = engine.bufs.part_x[id] as i32;
        if x < 35 {
            before_filter += 1;
            assert_eq!(
                engine.bufs.part_dcolour[id], 0,
                "the beam must not colour a BRICK created before the filter"
            );
        } else if x > 35 {
            after_filter += 1;
            assert_ne!(
                engine.bufs.part_dcolour[id], 0,
                "a BRICK created after the filter must carry its colour"
            );
        }
    }
    assert!(
        before_filter >= 2 && after_filter >= 2,
        "the beam must create on both sides of the filter ({before_filter} before, {after_filter} after)"
    );

    // The beam travels along +x, so the cells behind the laser stay empty.
    let xres = engine.config().xres as usize;
    let rgba = engine.render_rgba();
    let left = (32 * xres + 20) * 4;
    assert_eq!(
        (rgba[left], rgba[left + 1], rgba[left + 2]),
        (0, 0, 0),
        "the beam must not fire towards the spark"
    );
}

/// `PLUT.cpp:58-65`: plutonium emits a neutron when the air pressure is
/// positive (`chance(1, 100) && chance(int(5 * pv), 1000)`), and never in a
/// vacuum. Each emission replaces the plutonium pixel with the neutron.
#[test]
fn test_plut_emits_neutrons_under_pressure_only() {
    let _guard = gpu_lock();

    let build = |pv: f32| {
        let mut engine = create_test_engine(64, 64);
        engine.config_mut().gravity_mode = GravityMode::Off;
        engine.config_mut().air_mode = AirMode::NoUpdate;
        engine.config_mut().aheat_enable = false;
        fill_rect(&mut engine, 32, 10, 32, 49, PT_PLUT);
        for cell in engine.bufs.air_pv.iter_mut() {
            *cell = pv;
        }
        engine.bufs.mark_air_dirty();
        engine
    };

    // `int(5 * 500) = 2500` overshoots the 1000 denominator, so the second
    // chance always hits and the emission rate is the flat one in a hundred.
    let mut pressured = build(500.0);
    let mut peak_neutrons = 0usize;
    for _ in 0..600 {
        step(&mut pressured);
        peak_neutrons = peak_neutrons.max(count_type(&pressured, PT_NEUT));
    }
    assert!(peak_neutrons > 0, "PLUT under pressure must emit neutrons");
    assert!(
        count_type(&pressured, PT_PLUT) < 40,
        "each emission consumes the plutonium pixel"
    );

    let mut vacuum = build(0.0);
    for _ in 0..600 {
        step(&mut vacuum);
        assert_eq!(
            count_type(&vacuum, PT_NEUT),
            0,
            "without pressure a plutonium pixel never emits"
        );
    }
    assert_eq!(
        count_type(&vacuum, PT_PLUT),
        40,
        "the vacuum column must be untouched"
    );
}

/// `Simulation.cpp:3044-3056`: a photon that bounces off a particle keeps only
/// the wavelengths that particle reflects (`PhotonReflectWavelengths`), so the
/// colours it carries after the bounce are the ones the surface did not absorb.
#[test]
fn test_photon_reflection_filters_wavelengths() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(120, 80);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    // `GOLD.cpp:30` reflects `0x3C038100`; nothing else is on the photon's row.
    fill_rect(&mut engine, 60, 38, 60, 42, PT_GOLD);
    engine.apply_command(SimCommand::SpawnParticle {
        x: 59,
        y: 40,
        elem_ty: PT_PHOT,
        vx: 1.0,
        vy: 0.0,
    });
    step(&mut engine);

    let slot = (0..engine.bufs.capacity)
        .find(|&s| engine.bufs.part_ty[s] == PT_PHOT)
        .expect("the photon must survive bouncing off gold");
    eprintln!(
        "DEBUG photon slot={slot} x={} y={} vx={} vy={} ctype={:#x} pmap60={:#x} gold_ty={}",
        engine.bufs.part_x[slot],
        engine.bufs.part_y[slot],
        engine.bufs.part_vx[slot],
        engine.bufs.part_vy[slot],
        engine.bufs.part_ctype[slot],
        engine.bufs.pmap[40 * 120 + 60],
        PT_GOLD
    );
    assert_eq!(
        engine.bufs.part_ctype[slot], 0x3C03_8100,
        "the reflected wavelength must be filtered by gold's reflect mask"
    );
    assert_eq!(
        engine.bufs.part_x[slot] as i32, 59,
        "a photon blocked on its first step must not move"
    );
}

/// `Simulation::try_move`'s shared-pixel block (`e == 2`): a photon entering a
/// `FILT` is masked in the same frame, `INVIS` scatters it into a neutron while
/// the air pressure stays inside the wall's resistance, and `VOID` admits only
/// the types its `ctype`/`tmp` filter names.
#[test]
fn test_photon_filter_invisible_wall_and_void_filter() {
    let _guard = gpu_lock();

    let mut engine = create_test_engine(120, 80);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 60, 40, 60, 40, PT_FILT);
    let filt = (0..engine.bufs.capacity)
        .find(|&s| engine.bufs.part_ty[s] == PT_FILT)
        .expect("FILT placed");
    engine.bufs.part_ctype[filt] = 0x00FF_0000;
    engine.bufs.part_tmp[filt] = 1;
    engine.bufs.part_temp[filt] = 293.15;
    engine.bufs.mark_particle_slot_dirty(filt);

    engine.apply_command(SimCommand::SpawnParticle {
        x: 59,
        y: 40,
        elem_ty: PT_PHOT,
        vx: 1.0,
        vy: 0.0,
    });
    let phot = (0..engine.bufs.capacity)
        .find(|&s| engine.bufs.part_ty[s] == PT_PHOT)
        .expect("photon placed");
    engine.bufs.part_ctype[phot] = 0x00FF_00FF;
    engine.bufs.part_life[phot] = 200;
    engine.bufs.mark_particle_slot_dirty(phot);

    let mut filtered = false;
    for _ in 0..4 {
        step(&mut engine);
        if engine.bufs.part_ty[phot] == PT_PHOT && engine.bufs.part_ctype[phot] == 0x00FF_0000 {
            filtered = true;
            break;
        }
    }
    assert!(
        filtered,
        "a photon crossing FILT must be masked to the filter's band (ctype {:#x}, type {})",
        engine.bufs.part_ctype[phot], engine.bufs.part_ty[phot]
    );

    let mut invis_engine = create_test_engine(120, 80);
    invis_engine.config_mut().gravity_mode = GravityMode::Off;
    invis_engine.config_mut().air_mode = AirMode::NoUpdate;
    invis_engine.config_mut().aheat_enable = false;

    fill_rect(&mut invis_engine, 60, 40, 60, 40, PT_INVIS);
    invis_engine.apply_command(SimCommand::SpawnParticle {
        x: 59,
        y: 40,
        elem_ty: PT_PHOT,
        vx: 1.0,
        vy: 0.0,
    });
    for _ in 0..3 {
        step(&mut invis_engine);
    }
    assert_eq!(
        count_type(&invis_engine, PT_PHOT),
        0,
        "a photon inside an invisible wall must not stay a photon"
    );
    assert!(
        count_type(&invis_engine, PT_NEUT) >= 1,
        "`Simulation.cpp:1315-1326`: the photon becomes a neutron"
    );

    let mut void_other = create_test_engine(120, 80);
    void_other.config_mut().gravity_mode = GravityMode::Off;
    void_other.config_mut().air_mode = AirMode::NoUpdate;
    void_other.config_mut().aheat_enable = false;
    fill_rect(&mut void_other, 60, 40, 60, 40, PT_VOID);
    let void = (0..void_other.bufs.capacity)
        .find(|&s| void_other.bufs.part_ty[s] == PT_VOID)
        .expect("VOID placed");
    // `tmp == 0`: admit `ctype` only. A dust-only void rejects the photon.
    void_other.bufs.part_ctype[void] = PT_DUST;
    void_other.bufs.part_tmp[void] = 0;
    void_other.bufs.mark_particle_slot_dirty(void);
    void_other.apply_command(SimCommand::SpawnParticle {
        x: 59,
        y: 40,
        elem_ty: PT_PHOT,
        vx: 1.0,
        vy: 0.0,
    });
    for _ in 0..3 {
        step(&mut void_other);
    }
    assert_eq!(
        count_type(&void_other, PT_PHOT),
        1,
        "a dust-only VOID must not absorb a photon"
    );

    let mut void_phot = create_test_engine(120, 80);
    void_phot.config_mut().gravity_mode = GravityMode::Off;
    void_phot.config_mut().air_mode = AirMode::NoUpdate;
    void_phot.config_mut().aheat_enable = false;
    fill_rect(&mut void_phot, 60, 40, 60, 40, PT_VOID);
    let void = (0..void_phot.bufs.capacity)
        .find(|&s| void_phot.bufs.part_ty[s] == PT_VOID)
        .expect("VOID placed");
    void_phot.bufs.part_ctype[void] = PT_PHOT;
    void_phot.bufs.part_tmp[void] = 0;
    void_phot.bufs.mark_particle_slot_dirty(void);
    void_phot.apply_command(SimCommand::SpawnParticle {
        x: 59,
        y: 40,
        elem_ty: PT_PHOT,
        vx: 1.0,
        vy: 0.0,
    });
    for _ in 0..3 {
        step(&mut void_phot);
    }
    assert_eq!(
        count_type(&void_phot, PT_PHOT),
        0,
        "a photon-only VOID must absorb the photon"
    );

    // Gases random-walk, so the hydrogen is a thick cloud: the photon is bound
    // to enter one of its pixels within a few frames.
    let mut h2_engine = create_test_engine(120, 80);
    h2_engine.config_mut().gravity_mode = GravityMode::Off;
    h2_engine.config_mut().air_mode = AirMode::NoUpdate;
    h2_engine.config_mut().aheat_enable = false;
    fill_rect(&mut h2_engine, 60, 34, 72, 46, PT_H2);
    h2_engine.apply_command(SimCommand::SpawnParticle {
        x: 59,
        y: 40,
        elem_ty: PT_PHOT,
        vx: 1.0,
        vy: 0.0,
    });
    let h2_before = count_type(&h2_engine, PT_H2);
    for _ in 0..10 {
        step(&mut h2_engine);
        if count_type(&h2_engine, PT_PROT) > 0 {
            break;
        }
    }
    assert_eq!(
        count_type(&h2_engine, PT_PROT),
        1,
        "the photon must turn into a proton"
    );
    assert!(
        count_type(&h2_engine, PT_ELEC) >= 1,
        "`Simulation.cpp:1336`: the hydrogen atom is rebuilt as the electron"
    );
    assert!(
        count_type(&h2_engine, PT_H2) < h2_before,
        "one hydrogen atom must be consumed"
    );
}

/// `SPRK.cpp:62-88`: a spark that runs out of life settles back into the
/// conductor it stored in `ctype`, and a spark with no valid target becomes
/// `METL` instead of disappearing.
#[test]
fn test_spark_discharge_restores_its_conductor() {
    let _guard = gpu_lock();

    // A bare spark (brush-placed, no `ctype`) must become metal.
    let mut engine = create_test_engine(128, 128);
    engine.config_mut().gravity_mode = GravityMode::Off;
    spark_at(&mut engine, 40, 40);
    let spark = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_SPRK)
        .expect("the spark must be placed");
    engine.bufs.part_life[spark] = 1;
    engine.bufs.part_ctype[spark] = PT_NONE;
    engine.bufs.mark_particle_slot_dirty(spark);
    step(&mut engine);
    assert_eq!(
        engine.bufs.part_ty[spark], PT_METL,
        "an empty ctype must fall back to METL, not kill the spark"
    );
    assert_eq!(
        engine.bufs.part_life[spark], 4,
        "METL comes back with life 4"
    );

    // A spark over water restores water at room temperature with life 64.
    let mut wet = create_test_engine(128, 128);
    wet.config_mut().gravity_mode = GravityMode::Off;
    spark_at(&mut wet, 60, 60);
    let spark = (0..wet.bufs.capacity)
        .find(|&id| wet.bufs.part_ty[id] == PT_SPRK)
        .expect("the spark must be placed");
    wet.bufs.part_life[spark] = 1;
    wet.bufs.part_ctype[spark] = PT_WATR;
    wet.bufs.part_temp[spark] = 900.0;
    wet.bufs.mark_particle_slot_dirty(spark);
    step(&mut wet);
    assert_eq!(
        wet.bufs.part_ty[spark], PT_WATR,
        "`SPRK.cpp:64-70`: a water spark returns to water"
    );
    assert_eq!(
        wet.bufs.part_life[spark], 64,
        "water comes back with life 64"
    );
    assert!(
        (wet.bufs.part_temp[spark] - 293.15).abs() < 0.5,
        "`SPRK.cpp:64`: the water family returns at room temperature ({} K)",
        wet.bufs.part_temp[spark]
    );
}

/// `QRTZ.cpp:66-67`: the crack timer only clamps at five; the countdown is the
/// generic `PROP_LIFE_DEC` pass. The port decremented it a second time inside
/// the quartz kernel, so a crystal recovered twice as fast as the reference.
#[test]
fn test_quartz_crack_timer_ticks_once_per_frame() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    fill_rect(&mut engine, 30, 30, 30, 30, PT_QRTZ);
    let quartz = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_QRTZ)
        .expect("the quartz must be placed");
    engine.bufs.part_life[quartz] = 5;
    engine.bufs.mark_particle_slot_dirty(quartz);
    step(&mut engine);
    assert_eq!(
        engine.bufs.part_life[quartz], 4,
        "one frame must spend exactly one tick of the crack timer"
    );

    // Above five the reference clamps instead of counting down.
    engine.bufs.part_life[quartz] = 9;
    engine.bufs.mark_particle_slot_dirty(quartz);
    step(&mut engine);
    assert_eq!(engine.bufs.part_life[quartz], 5, "`QRTZ.cpp:68-69` clamps");
}

/// `PROT.cpp:180-200`: the collision energy turns the proton into a heavier
/// element *next to* it (`create_part(-1, x + rng(-1..1), ...)`) and kills the
/// proton; the port used to transmute in place and keep the slot.
#[test]
fn test_proton_transmutation_lands_next_to_the_proton() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(128, 128);
    engine.config_mut().gravity_mode = GravityMode::Off;
    fill_rect(&mut engine, 50, 50, 50, 50, PT_DUST);
    place_energy(&mut engine, vec![(50, 50)], PT_PROT);
    let proton = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_PROT)
        .expect("the proton must be placed");
    // `tmp > 50` transmutes into `CO2` on the next frame.
    engine.bufs.part_tmp[proton] = 60;
    engine.bufs.mark_particle_slot_dirty(proton);
    step(&mut engine);

    assert_eq!(
        count_type(&engine, PT_PROT),
        0,
        "the proton must be consumed"
    );
    assert_eq!(
        count_type(&engine, PT_CO2),
        1,
        "`tmp = 60` transmutes to CO2"
    );
    let co2 = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_CO2)
        .expect("the product must exist");
    assert_ne!(
        co2, proton,
        "the product is a fresh particle; the proton's own slot must be freed"
    );
    // The product is a gas, so the movement pass of the same frame drifts it a
    // little; what matters is that it started on the proton or its ring.
    let dx = (engine.bufs.part_x[co2] - 50.0).abs();
    let dy = (engine.bufs.part_y[co2] - 50.0).abs();
    assert!(
        dx <= 1.0 && dy <= 4.0,
        "the product must land on the proton or a neighbour, got ({dx}, {dy})"
    );
    assert!(
        (engine.bufs.part_temp[co2] - 6000.0).abs() < 1.0,
        "`create_part`: the product inherits `temp = 100 * tmp` ({})",
        engine.bufs.part_temp[co2]
    );
}

/// `PROT.cpp:74-85`: a proton passing over an active spark discharges it back
/// into the conductor in its `ctype` and the restored particle keeps the
/// `life + 44` the `part_change_type` caller adds, so the spark stays live.
#[test]
fn test_proton_discharges_a_spark_without_killing_it() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(128, 128);
    engine.config_mut().gravity_mode = GravityMode::Off;
    spark_at(&mut engine, 70, 70);
    let spark = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_SPRK)
        .expect("the spark must be placed");
    engine.bufs.part_ctype[spark] = PT_WATR;
    engine.bufs.part_life[spark] = 40;
    engine.bufs.mark_particle_slot_dirty(spark);
    place_energy(&mut engine, vec![(70, 70)], PT_PROT);
    step(&mut engine);

    assert_eq!(
        engine.bufs.part_ty[spark], PT_WATR,
        "`part_change_type` restores the conductor stored in `ctype`"
    );
    assert_eq!(engine.bufs.part_ctype[spark], 0, "`ctype` is cleared");
    assert!(
        engine.bufs.part_life[spark] >= 80,
        "the caller adds 44 frames of life to the restored conductor ({})",
        engine.bufs.part_life[spark]
    );
}

#[test]
fn tmp_probe_phot_look() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(160, 96);
    engine.config_mut().gravity_mode = GravityMode::Off;
    // White photon (brush default) and a red one.
    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(40, 40)],
        elem_ty: PT_PHOT,
        replace_mode: true,
        dcolour: 0,
    });
    let id = (0..engine.bufs.capacity)
        .find(|&i| engine.bufs.part_ty[i] == PT_PHOT)
        .unwrap();
    println!(
        "phot default: life={} ctype=0x{:08X} temp={}",
        engine.bufs.part_life[id], engine.bufs.part_ctype[id], engine.bufs.part_temp[id]
    );
    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(80, 40)],
        elem_ty: PT_PHOT,
        replace_mode: true,
        dcolour: 0,
    });
    let red = (0..engine.bufs.capacity)
        .find(|&i| engine.bufs.part_ty[i] == PT_PHOT && i != id)
        .unwrap();
    engine.bufs.part_ctype[red] = 0x3FFC_0000; // red band only
    engine.bufs.mark_particle_slot_dirty(red);
    let rgba = engine.render_rgba();
    let at = |x: usize, y: usize| {
        let i = (y * 160 + x) * 4;
        (rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3])
    };
    println!("white pixel (40,40) = {:?}", at(40, 40));
    println!("red pixel   (80,40) = {:?}", at(80, 40));
    println!("bg          (10,10) = {:?}", at(10, 10));
}

/// `O2.cpp:80-123`: white-hot, over-pressured oxygen in a strong Newtonian
/// gravity field collapses into `BRMT` and sprays `NEUT`/`PHOT`/`GRVT` out of
/// the same pixel, with a `tmp |= 4` plasma in a random neighbour and the
/// cell pressure capped at `MAX_PRESSURE`.
#[test]
fn test_o2_hot_graviton_burst() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(120, 80);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    // A dense oxygen pool: every pixel sits far above 9973.15 K and every
    // cell above 250 pressure, so the 1-in-5 gate fires for each pixel.
    fill_rect(&mut engine, 56, 36, 64, 44, PT_O2);
    // `newton_grav` stays off, so the solve never runs and the device force
    // grids can be seeded directly (the same hook the warp-render test uses).
    // `21^2 = 441 > 400` is the `O2.cpp:82` gate.
    let cells = engine.bufs.config.num_cells();
    engine.bufs.client.write(
        &engine.bufs.gpu_grav_fx,
        Bytes::from_elems(vec![21.0f32; cells]),
    );
    engine.bufs.client.write(
        &engine.bufs.gpu_grav_fy,
        Bytes::from_elems(vec![0.0f32; cells]),
    );

    for _ in 0..16 {
        // A cooled or drifted pixel must not hide the reaction: refresh the
        // pool before every attempt.
        for id in 0..engine.bufs.capacity {
            if engine.bufs.part_ty[id] == PT_O2 {
                engine.bufs.part_temp[id] = 9998.0;
            }
        }
        engine.bufs.mark_particles_dirty();
        for pv in engine.bufs.air_pv.iter_mut() {
            *pv = 255.0;
        }
        engine.bufs.mark_air_dirty();
        step(&mut engine);
        if count_type(&engine, PT_BRMT) > 0 {
            break;
        }
    }

    assert!(
        count_type(&engine, PT_BRMT) > 0,
        "hot, compressed oxygen in a strong gravity field must collapse into BRMT"
    );
    // Every burst queues the same-pixel energy products. The photons layer
    // holds one word per pixel, so the last creation (GRVT) owns the map
    // entry while the NEUT and PHOT slots stay alive in the pool — the same
    // model the hydrogen fusion burst uses.
    assert!(
        count_type(&engine, PT_NEUT) >= 1,
        "the burst must emit a neutron"
    );
    assert!(
        count_type(&engine, PT_PHOT) >= 1,
        "the burst must emit a photon"
    );
    assert!(
        count_type(&engine, PT_GRVT) >= 1,
        "the burst must emit a graviton"
    );
    assert!(
        count_type(&engine, PT_PLSM) >= 1,
        "a random neighbour must receive the tmp|=4 plasma"
    );
    assert!(
        engine.bufs.air_pv.contains(&MAX_PRESSURE),
        "a burst must cap its cell pressure at MAX_PRESSURE"
    );
}

/// `LITH.cpp:104-121`: a spark sitting on `PSCN` next to lithium stores one
/// unit and arms the 8 frame burn timer; a charged particle beside an idle
/// `NSCN` spends one unit and turns that silicon into a spark.
///
/// The two branches only meet a spark that reaches `life == 3` on the tick the
/// lithium pass runs, so this drives a real wire spark along a `PSCN` span with
/// the lithium hanging off one segment.
#[test]
fn test_lith_battery_charges_from_pscn_and_discharges_into_nscn() {
    let _guard = gpu_lock();

    let mut charge = create_test_engine(64, 64);
    charge.config_mut().gravity_mode = GravityMode::Off;
    fill_rect(&mut charge, 20, 10, 26, 10, PT_PSCN);
    fill_rect(&mut charge, 22, 11, 22, 11, PT_LITH);
    let lith = (0..charge.bufs.capacity)
        .find(|&id| charge.bufs.part_ty[id] == PT_LITH)
        .expect("the lithium must be placed");
    charge.bufs.part_ctype[lith] = 0;
    charge.bufs.part_life[lith] = 0;
    charge.bufs.part_temp[lith] = 300.0;
    charge.bufs.mark_particle_slot_dirty(lith);
    spark_at(&mut charge, 20, 10);
    step(&mut charge);
    assert_eq!(
        charge.bufs.part_ctype[lith], 1,
        "`LITH.cpp:110-114`: a spark on an adjacent PSCN stores one unit"
    );
    assert_eq!(
        charge.bufs.part_life[lith], 8,
        "`LITH.cpp:203-206`: charging arms the 8 frame burn timer"
    );

    let mut drain = create_test_engine(64, 64);
    drain.config_mut().gravity_mode = GravityMode::Off;
    fill_rect(&mut drain, 10, 10, 10, 10, PT_LITH);
    fill_rect(&mut drain, 11, 10, 11, 10, PT_NSCN);
    fill_rect(&mut drain, 12, 10, 12, 10, PT_METL);
    let lith = (0..drain.bufs.capacity)
        .find(|&id| drain.bufs.part_ty[id] == PT_LITH)
        .expect("the lithium must be placed");
    drain.bufs.part_ctype[lith] = 5;
    drain.bufs.part_life[lith] = 0;
    drain.bufs.part_temp[lith] = 300.0;
    drain.bufs.mark_particle_slot_dirty(lith);
    step(&mut drain);
    assert_eq!(
        drain.bufs.part_ctype[lith], 4,
        "`LITH.cpp:121-131`: the discharge spends one stored unit"
    );
    assert_eq!(drain.bufs.part_life[lith], 8, "discharging arms the timer");
    let nscn = drain
        .bufs
        .particle_index_at(11, 10)
        .expect("the n-type silicon must still exist");
    assert_eq!(
        drain.bufs.part_ty[nscn], PT_SPRK,
        "`LITH.cpp:124-127`: the n-type silicon turns into a spark"
    );
    assert_eq!(drain.bufs.part_ctype[nscn], PT_NSCN);
    assert_eq!(drain.bufs.part_life[nscn], 4);
}

/// `LITH.cpp:86-91`: while the burn timer runs above 1012 a burning particle
/// spits `FIRE` into the empty cells of its 5x5 ring, one in ten.
#[test]
fn test_burning_lith_spits_fire_into_empty_air() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    fill_rect(&mut engine, 32, 32, 32, 32, PT_LITH);
    let lith = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_LITH)
        .expect("the lithium must be placed");
    engine.bufs.part_life[lith] = 1020;
    engine.bufs.part_temp[lith] = 300.0;
    engine.bufs.mark_particle_slot_dirty(lith);

    let mut fires = 0;
    for _ in 0..3 {
        step(&mut engine);
        fires = count_type(&engine, PT_FIRE);
        if fires > 0 {
            break;
        }
    }
    assert!(fires > 0, "a burning lithium must spit fire into open air");
    assert_eq!(
        engine.bufs.part_ty[lith], PT_LITH,
        "the lithium itself keeps burning"
    );
}

/// `LITH.cpp:230-232`: at or above 100 stored units the battery arms a
/// 24 frame fuse (`burnTimer = 1024`, one decrement per frame) and detonates
/// into `LAVA`, whose `ctype` records the absorbed element: `LITH` below three
/// carbonation units and `GLAS` at or above them.
///
/// The lithium is packed in `INSL` so the burning particle has no empty cell to
/// spit `FIRE` into (`LITH.cpp:86-91`); an escaping flame would heat the
/// particle past its 453.65 K melting point and the melt transition would beat
/// the fuse.
#[test]
fn test_lith_overcharge_fuse_turns_it_into_lava() {
    let _guard = gpu_lock();

    for (carbonation, expected) in [(0, PT_LITH), (3, PT_GLAS)] {
        let mut engine = create_test_engine(64, 64);
        engine.config_mut().gravity_mode = GravityMode::Off;
        fill_rect(&mut engine, 30, 30, 34, 34, PT_INSL);
        fill_rect(&mut engine, 32, 32, 32, 32, PT_LITH);
        let lith = (0..engine.bufs.capacity)
            .find(|&id| engine.bufs.part_ty[id] == PT_LITH)
            .expect("the lithium must be placed");
        engine.bufs.part_ctype[lith] = 100;
        engine.bufs.part_life[lith] = 0;
        engine.bufs.part_tmp2[lith] = carbonation;
        engine.bufs.part_temp[lith] = 300.0;
        engine.bufs.mark_particle_slot_dirty(lith);

        step(&mut engine);
        assert_eq!(
            engine.bufs.part_life[lith], 1024,
            "`LITH.cpp:230-231`: 100 stored units arm the fuse"
        );
        for _ in 0..24 {
            step(&mut engine);
        }
        assert_eq!(
            engine.bufs.part_ty[lith], PT_LAVA,
            "`LITH.cpp:232-243`: the fuse detonates at life 1000"
        );
        assert_eq!(
            engine.bufs.part_ctype[lith], expected,
            "carbonation {carbonation}: `LITH.cpp:236-242` picks the melt's ctype"
        );
    }
}

/// `LITH.cpp:203-219`: adjacent lithiums trade half the difference of their
/// stored energy, one successful trade per particle per frame. The reference
/// runs the trade serially, so the pile conserves its total charge exactly;
/// the port hands the transfer to atomics because every lithium trades from a
/// parallel lane, and a plain store into the neighbour's `ctype` dropped
/// updates (a charged row leaked ~5% of its charge in ten frames).
#[test]
fn test_lith_charge_spreads_and_conserves_energy() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    fill_rect(&mut engine, 20, 20, 40, 20, PT_LITH);
    let mut ids: Vec<usize> = (0..engine.bufs.capacity)
        .filter(|&id| engine.bufs.part_ty[id] == PT_LITH)
        .collect();
    ids.sort_unstable();
    assert_eq!(ids.len(), 21, "a 21 pixel lithium row");
    for &id in &ids {
        engine.bufs.part_temp[id] = 300.0;
        engine.bufs.part_life[id] = 0;
    }
    let mid = ids[ids.len() / 2];
    engine.bufs.part_ctype[mid] = 90;
    for &id in &ids {
        engine.bufs.mark_particle_slot_dirty(id);
    }

    for round in 0..20 {
        step(&mut engine);
        let total: i32 = ids
            .iter()
            .map(|&id| engine.bufs.part_ctype[id] as i32)
            .sum();
        assert_eq!(total, 90, "round {round}: the trade loop conserves charge");
    }
    let charged = ids
        .iter()
        .filter(|&&id| engine.bufs.part_ctype[id] > 0)
        .count();
    assert!(
        charged >= ids.len() / 2,
        "the charge must diffuse through the row, reached {charged} of {}",
        ids.len()
    );
}

/// `CBNW.cpp:55-64`: in a cell at or below 3 pressure soda water outgasses
/// into `CO2` (badge `ctype = 5`) and adds half a pressure unit. A vacuum
/// (`pv <= -0.5`) takes the branch every frame, with no random roll.
#[test]
fn test_carbonated_water_degasses_in_low_pressure() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 32, 32, 32, 32, PT_CBNW);
    let xcells = engine.bufs.config.xcells();
    let cell = ((32u32 / CELL) * xcells + 32u32 / CELL) as usize;
    engine.bufs.air_pv[cell] = -1.0;
    engine.bufs.mark_air_dirty();

    step(&mut engine);

    assert_eq!(
        count_type(&engine, PT_CBNW),
        0,
        "`CBNW.cpp:57-60`: a vacuum must outgas the soda water"
    );
    let co2 = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_CO2)
        .expect("the soda water becomes CO2");
    assert_eq!(
        engine.bufs.part_ctype[co2], 5,
        "`CBNW.cpp:61`: the gas is tagged ctype = 5"
    );
    assert!(
        (engine.bufs.air_pv[cell] - (-0.5)).abs() < 1e-6,
        "`CBNW.cpp:62`: the cell gains 0.5 pressure, found {}",
        engine.bufs.air_pv[cell]
    );
    // `CBNW.cpp:65-67`: the rest of the update still runs on the new CO2, so
    // the speckle timer takes its step toward 20.
    assert_eq!(
        engine.bufs.part_tmp2[co2], 1,
        "`CBNW.cpp:65-67`: tmp2 drifts one unit per frame"
    );
}

/// `GLAS.cpp:52-64`: the pane remembers `int(pv * 64)` in `tmp3` and shatters
/// into `BGLA` when the jump exceeds its strength (16 for unstrengthened
/// glass). A small change of the same cell survives, a large one does not.
#[test]
fn test_glass_shatters_on_pressure_jump() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 32, 32, 32, 32, PT_GLAS);
    let xcells = engine.bufs.config.xcells();
    let cell = ((32u32 / CELL) * xcells + 32u32 / CELL) as usize;
    let glas = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_GLAS)
        .expect("glass placed");
    assert_eq!(
        engine.bufs.part_tmp3[glas], 0,
        "the paint path seeds tmp3 from the ambient pressure (`GLAS.cpp:94`)"
    );

    // int(0.1 * 64) = 6: below the base strength of 16.
    engine.bufs.air_pv[cell] = 0.1;
    engine.bufs.mark_air_dirty();
    step(&mut engine);
    assert_eq!(
        count_type(&engine, PT_GLAS),
        1,
        "a 6/64 pressure step must not shatter the pane"
    );
    assert_eq!(engine.bufs.part_tmp3[glas], 6, "`GLAS.cpp:64` remembers it");

    // int(1.0 * 64) = 64, a jump of 58 against the 16 strength.
    engine.bufs.air_pv[cell] = 1.0;
    engine.bufs.mark_air_dirty();
    step(&mut engine);
    assert_eq!(
        count_type(&engine, PT_GLAS),
        0,
        "`GLAS.cpp:59-63`: a 58 unit jump must shatter the glass"
    );
    assert_eq!(
        count_type(&engine, PT_BGLA),
        1,
        "the shattered pane becomes BGLA"
    );
}

/// `GLAS.cpp:66-87`: strengthened glass (`strength > 200`) in a cold,
/// pressurised cell condenses liquid nitrogen into an empty neighbour once
/// every 100 frames, spending one pressure unit and heating the pane by 200 K.
#[test]
fn test_strengthened_glass_condenses_liquid_nitrogen() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(128, 128);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    // Ten panes in ten different pressure cells, so the 1-in-100 roll has ten
    // independent chances each frame.
    let panes: Vec<i32> = (0..10).map(|k| 8 + 12 * k).collect();
    for &x in &panes {
        fill_rect(&mut engine, x, 64, x, 64, PT_GLAS);
    }
    for pv in engine.bufs.air_pv.iter_mut() {
        *pv = 6.0;
    }
    for hv in engine.bufs.air_hv.iter_mut() {
        *hv = 0.0;
    }
    engine.bufs.mark_air_dirty();
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] == PT_GLAS {
            // 22200 / 120 + 16 = 201 > 200; 70 K is below the 77 K limit.
            engine.bufs.part_life[id] = 22_200;
            engine.bufs.part_temp[id] = 70.0;
            // The jump must stay at zero so the pane does not shatter before
            // the condensation roll fires.
            engine.bufs.part_tmp3[id] = (6.0f32 * 64.0) as i32;
            engine.bufs.mark_particle_slot_dirty(id);
        }
    }

    let mut created = false;
    for _ in 0..300 {
        step(&mut engine);
        if count_type(&engine, PT_LN2) > 0 {
            created = true;
            break;
        }
    }
    assert!(
        created,
        "`GLAS.cpp:66-87`: strengthened glass must condense LNTG under pressure"
    );
    // The pane that condensed the nitrogen takes the 200 K; the others stay
    // at 70 K.
    let heated = (0..engine.bufs.capacity)
        .filter(|&id| engine.bufs.part_ty[id] == PT_GLAS)
        .any(|id| engine.bufs.part_temp[id] >= 270.0);
    assert!(
        heated,
        "`GLAS.cpp:84`: the pane that condensed the nitrogen gains 200 K"
    );
    assert!(
        engine.bufs.air_pv.contains(&5.0),
        "`GLAS.cpp:83`: the condensation spends one pressure unit"
    );
}

/// `VIBR.cpp:70-80`: a cell above 2.5 pressure charges 7 `tmp` and spends one
/// pressure unit; the vacuum mirror drains 2 and returns one unit.
#[test]
fn test_vibranium_absorbs_pressure() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 32, 32, 32, 32, PT_VIBR);
    let id = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_VIBR)
        .expect("vibranium placed");
    // 273.15 K is inside the dead band, so only the pressure branch runs.
    engine.bufs.part_temp[id] = 273.15;
    engine.bufs.part_tmp[id] = 0;
    engine.bufs.mark_particle_slot_dirty(id);
    let xcells = engine.bufs.config.xcells();
    let cell = ((32u32 / CELL) * xcells + 32u32 / CELL) as usize;
    engine.bufs.air_pv[cell] = 3.0;
    engine.bufs.mark_air_dirty();

    step(&mut engine);
    assert_eq!(
        engine.bufs.part_tmp[id], 7,
        "`VIBR.cpp:71-73`: pv > 2.5 charges 7"
    );
    assert!(
        (engine.bufs.air_pv[cell] - 2.0).abs() < 1e-6,
        "`VIBR.cpp:74`: the crystal spends one pressure unit, found {}",
        engine.bufs.air_pv[cell]
    );

    // Negative mirror: tmp -= 2 and pv += 1.
    engine.bufs.part_tmp[id] = 10;
    engine.bufs.part_temp[id] = 273.15;
    engine.bufs.mark_particle_slot_dirty(id);
    engine.bufs.air_pv[cell] = -3.0;
    engine.bufs.mark_air_dirty();
    step(&mut engine);
    assert_eq!(
        engine.bufs.part_tmp[id], 8,
        "`VIBR.cpp:76-79`: pv < -2.5 drains 2"
    );
    assert!(
        (engine.bufs.air_pv[cell] - (-2.0)).abs() < 1e-6,
        "`VIBR.cpp:79`: the cell regains one unit, found {}",
        engine.bufs.air_pv[cell]
    );
}

/// `(x, y)` of every live particle of `ty` on the canvas, sorted.
fn element_cells(engine: &SimulationEngine, ty: u32) -> Vec<(i32, i32)> {
    let mut cells = Vec::new();
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] != ty {
            continue;
        }
        cells.push((
            engine.bufs.part_x[id].round() as i32,
            engine.bufs.part_y[id].round() as i32,
        ));
    }
    cells.sort_unstable();
    cells
}

/// `Element_LOVE_RuleTable[nnx][nny]` (`LOVE.cpp:45-58`).
const LOVE_RULE_TABLE: [[u32; 9]; 9] = [
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
];

/// `Element_LOLZ_RuleTable[nny][nnx]` (`LOLZ.cpp:46-58`).
const LOLZ_RULE_TABLE: [[u32; 9]; 9] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
];

/// A single `LOVE` cell seeds the block and the table stamps every `1` cell.
///
/// TPT's rule loops start at `(9, 9)` (`Simulation.cpp:3817-3819`), and
/// `Element_LOVE_RuleTable[0][0] == 0`, so that one painted cell marks the
/// block and is then culled; the block is left with exactly the table's `1`
/// cells, which the next frame must leave untouched.
#[test]
fn test_love_rule_table_stamps_one_generation() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(128, 128);

    fill_rect(&mut engine, 9, 9, 9, 9, PT_LOVE);
    assert_eq!(count_type(&engine, PT_LOVE), 1);

    let mut expected = Vec::new();
    for (nnx, row) in LOVE_RULE_TABLE.iter().enumerate() {
        for (nny, &bit) in row.iter().enumerate() {
            if bit != 0 {
                expected.push((9 + nnx as i32, 9 + nny as i32));
            }
        }
    }
    expected.sort_unstable();
    assert_eq!(expected.len(), 20, "the LOVE table has twenty live cells");

    step(&mut engine);
    assert_eq!(
        element_cells(&engine, PT_LOVE),
        expected,
        "`Simulation.cpp:3821-3838` must stamp Element_LOVE_RuleTable[nnx][nny]"
    );

    // Every `1` cell holds LOVE and every other cell of the block is empty, so
    // another frame is a no-op.
    step(&mut engine);
    assert_eq!(
        element_cells(&engine, PT_LOVE),
        expected,
        "the stamped LOVE block is a fixed point"
    );
}

/// The `LOLZ` mirror case: the pass reads `Element_LOLZ_RuleTable[nny][nnx]`.
///
/// The two tables are not transposes of each other, so a port that indexed
/// `[nnx][nny]` would stamp a different 24-cell block. Like the LOVE case, the
/// seed sits on a `0` cell and is therefore culled.
#[test]
fn test_lolz_rule_table_stamps_the_mirrored_pattern() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(128, 128);

    fill_rect(&mut engine, 9, 9, 9, 9, PT_LOLZ);
    assert_eq!(count_type(&engine, PT_LOLZ), 1);

    let mut expected = Vec::new();
    for (nny, row) in LOLZ_RULE_TABLE.iter().enumerate() {
        for (nnx, &bit) in row.iter().enumerate() {
            if bit != 0 {
                expected.push((9 + nnx as i32, 9 + nny as i32));
            }
        }
    }
    expected.sort_unstable();
    assert_eq!(
        expected.len(),
        24,
        "the LOLZ table has twenty-four live cells"
    );

    step(&mut engine);
    assert_eq!(
        element_cells(&engine, PT_LOLZ),
        expected,
        "`Simulation.cpp:3847` must index Element_LOLZ_RuleTable[nny][nnx]"
    );

    step(&mut engine);
    assert_eq!(
        element_cells(&engine, PT_LOLZ),
        expected,
        "the stamped LOLZ block is a fixed point"
    );
}

/// `VIRS.cpp:48`: every virus starts with `DefaultProperties.tmp4 = 250`.
/// The host brush allocates from the element row and the device creation queue
/// (`FloodFillParticles`) goes through `create_fields_kernel`, so both paths
/// must seed the full death timer; the save primitive exports/imports it.
#[test]
fn virus_creation_applies_the_tmp4_default() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 10, 10, 10, 10, PT_VIRS);
    let brushed = engine
        .bufs
        .particle_index_at(10, 10)
        .expect("the brushed virus");
    assert_eq!(
        engine.bufs.part_tmp4[brushed], 250,
        "`allocate_particle` copies the virus `DefaultProperties.tmp4`"
    );

    engine.apply_command(SimCommand::FloodFillParticles {
        x: 40,
        y: 40,
        elem_ty: PT_VIRS,
    });
    engine.ensure_host_current();
    let queued = engine
        .bufs
        .particle_index_at(40, 40)
        .expect("the flood-filled virus");
    assert_eq!(
        engine.bufs.part_tmp4[queued], 250,
        "the device creation drain copies the virus `DefaultProperties.tmp4`"
    );

    let saved = engine.bufs.export_particle(queued);
    assert_eq!(saved.tmp4, 250);
    engine.bufs.clear_all_particles();
    engine.bufs.import_particle(0, &saved);
    assert_eq!(
        engine.bufs.part_tmp4[0], 250,
        "a save restore must keep the exported timer"
    );
}

/// `STOR.cpp:70-83` calls `Element_SOAP_detach` (`SOAP.cpp:73-88`) before it
/// captures a soap bubble: both reciprocal link bits on the mate are cleared
/// and the stored `ctype` word is zero.
#[test]
fn stor_capture_detaches_soap_mates() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 10, 10, 10, 10, PT_SOAP);
    fill_rect(&mut engine, 11, 10, 11, 10, PT_SOAP);
    let a = engine.bufs.particle_index_at(10, 10).expect("soap A");
    let b = engine.bufs.particle_index_at(11, 10).expect("soap B");
    // A live two-bubble link in both directions: A -> B through `tmp`,
    // B -> A through `tmp2`, with bits 2 and 4 set on both ends so the bubble
    // pass does not attach anything new. Both bubbles are "dead" (life 0) but
    // below the freezing point, so the pop branch stays asleep.
    engine.bufs.part_ctype[a] = 1 | 2 | 4;
    engine.bufs.part_tmp[a] = b as i32;
    engine.bufs.part_tmp2[a] = b as i32;
    engine.bufs.part_ctype[b] = 1 | 2 | 4;
    engine.bufs.part_tmp[b] = a as i32;
    engine.bufs.part_tmp2[b] = a as i32;
    engine.bufs.part_life[a] = 0;
    engine.bufs.part_life[b] = 0;
    engine.bufs.part_temp[a] = 150.0;
    engine.bufs.part_temp[b] = 150.0;
    engine.bufs.mark_particles_dirty();

    fill_rect(&mut engine, 9, 10, 9, 10, PT_STOR);
    let stor = engine.bufs.particle_index_at(9, 10).expect("the storer");

    step(&mut engine);

    assert_eq!(
        engine.bufs.part_ty[a], PT_NONE,
        "`STOR.cpp:83` kills the captured particle"
    );
    assert_eq!(
        engine.bufs.part_tmp[stor], PT_SOAP as i32,
        "the storer records the captured type"
    );
    assert_eq!(
        engine.bufs.part_ctype[b] & 6,
        0,
        "`Element_SOAP_detach` clears both reciprocal mate bits"
    );
    assert_eq!(
        engine.bufs.part_ctype[b] & 1,
        1,
        "the mate is still a bubble"
    );
    assert_eq!(
        engine.bufs.part_tmp4[stor], 0,
        "the detached `ctype` word (zero) is what the storer captures"
    );
}

/// `LIGH.cpp:165-178`: a bolt's segments count `tmp2` down and kill
/// themselves, so a placed bolt leaves no permanent fragments behind.
/// Regression guard for the creation queue replaying stale records and
/// rewriting `part_tmp2` every frame.
#[test]
fn ligh_bolt_segments_expire_without_fragments() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.apply_command(SimCommand::CreateLightning {
        x: 32,
        y: 32,
        radius_x: 20,
        radius_y: 20,
    });

    // The first frame rasterises a line of segments; every one of them then
    // counts `tmp2` down to 5 and is killed a few frames later.
    let mut saw_segments = false;
    for _ in 0..16 {
        step(&mut engine);
        saw_segments |= count_type(&engine, PT_LIGH) > 1;
    }
    assert!(saw_segments, "the bolt must draw a line of LIGH segments");
    assert_eq!(
        count_type(&engine, PT_LIGH),
        0,
        "LIGH segments must expire, leaving no permanent fragments"
    );
}

/// A stationary hold dispatches a bolt on every frame
/// (`GameView.cpp:1739-1756`), throttled to `power / 4` ticks
/// (`Editing.cpp:782-793`). The creation queue's per-frame record count then
/// changes from frame to frame, which is where stale records used to be
/// replayed and rewrite `part_tmp2`. Every bolt must still expire.
#[test]
fn ligh_continuous_placement_leaves_no_fragments() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;
    engine.config_mut().gravity_mode = GravityMode::Off;

    let mut saw_segments = false;
    for _ in 0..24 {
        engine.apply_command(SimCommand::CreateLightning {
            x: 32,
            y: 32,
            radius_x: 20,
            radius_y: 20,
        });
        step(&mut engine);
        saw_segments |= count_type(&engine, PT_LIGH) > 1;
    }
    assert!(saw_segments, "continuous placement must draw LIGH segments");

    for _ in 0..16 {
        step(&mut engine);
    }
    assert_eq!(
        count_type(&engine, PT_LIGH),
        0,
        "continuous placement must leave no permanent fragments"
    );
}
