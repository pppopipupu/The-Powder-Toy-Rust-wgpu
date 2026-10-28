use super::*;

/// Walls off the `CELL`-wide canvas border with `WL_WALL`.
///
/// `Simulation.cpp:2313-2317` kills every particle inside the border band
/// (`x < CELL || y < CELL || x >= XRES-CELL || y >= YRES-CELL`). A rising gas
/// reaches that band, so the band is filled with walls: walls live on the wall
/// grid, not in the particle pool, so the kill cannot touch them, and
/// `eval_move` (`Simulation.cpp:1024-1050`) refuses to move a particle into a
/// `WL_WALL` cell.
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
fn test_hot_air_raises_air_pressure() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(160, 128);
    engine.config_mut().air_mode = AirMode::On;
    engine.config_mut().heat_enable = false;
    engine.config_mut().aheat_enable = false;

    // Dense gas block: every pixel adds `4 * HotAir * (3.5 - pv)` to the
    // pressure of its 4x4 air cell (`Simulation.cpp:2347`).
    fill_rect(&mut engine, 40, 40, 119, 87, PT_GAS);
    assert_eq!(max_air_pressure(&engine), 0.0);

    step(&mut engine);
    let pressure = max_air_pressure(&engine);
    assert!(
        pressure > 0.0,
        "HotAir did not raise the air pressure (got {pressure})"
    );
    assert!(
        pressure < 3.5,
        "GAS pressure must approach 3.5 from below, got {pressure}"
    );

    // A cold solid has negative HotAir and lowers the pressure instead.
    let mut cold = create_test_engine(160, 128);
    cold.config_mut().air_mode = AirMode::On;
    cold.config_mut().heat_enable = false;
    cold.config_mut().aheat_enable = false;
    fill_rect(&mut cold, 40, 40, 119, 87, PT_ICEI);
    step(&mut cold);
    assert!(
        max_air_pressure(&cold) <= 0.0,
        "ICEI has negative HotAir and must not raise the pressure"
    );
}

#[test]
fn test_phase_transition_pressure_shift_only_for_boiling_liquids() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(160, 120);
    // Freeze the air field so the pressure written below survives the step
    // instead of being re-solved from the velocity divergence.
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().aheat_enable = false;
    // The steam rises to the canvas top; keep it out of the lethal border band
    // (`Simulation.cpp:2313-2317`) so the 25 grains stay countable.
    wall_off_canvas_border(&mut engine);

    fill_rect(&mut engine, 20, 20, 24, 24, PT_SALT);
    fill_rect(&mut engine, 60, 20, 64, 24, PT_WATR);
    set_temp_of_type(&mut engine, PT_SALT, 1000.0);
    set_temp_of_type(&mut engine, PT_WATR, 350.0);

    // Ambient pressure: neither block has reached its own transition point.
    step(&mut engine);
    assert_eq!(count_type(&engine, PT_SALT), 25, "SALT melts at 1173 K");
    assert_eq!(count_type(&engine, PT_WATR), 25, "WATR boils at 373 K");

    // A 150-unit vacuum moves the boiling point to `373 - 2 * (-150) = 673 K`,
    // so the 350 K water is above it. `Simulation.cpp:2452` gates every phase
    // transition behind the per-particle `HeatConduct`/250 roll, so only about
    // `29/250` of the `WATR` pixels transition per frame (WATR's HeatConduct is
    // 29); a single step therefore leaves most of the pool liquid. Run enough
    // frames for the whole block to boil. `Simulation.cpp:2511-2519` only
    // shifts liquids boiling into gases (and gases condensing into liquids), so
    // the powder's 1000 K stays below its raw 1173 K melting point even though
    // the old global shift would have read it as 1300 K; the steam also stays
    // gaseous because its pressure-shifted condensation point remains far
    // above the temperature it boiled at.
    for pv in engine.bufs.air_pv.iter_mut() {
        *pv = -150.0;
    }
    engine.bufs.mark_air_dirty();
    for _ in 0..120 {
        step(&mut engine);
    }

    assert_eq!(
        count_type(&engine, PT_SALT),
        25,
        "a vacuum must not melt SALT: its melting point does not shift"
    );
    assert_eq!(count_type(&engine, PT_LAVA), 0, "no lava may appear");
    assert_eq!(
        count_type(&engine, PT_WATR),
        0,
        "the water must boil off under vacuum"
    );
    assert_eq!(
        count_type(&engine, PT_WTRV),
        25,
        "boiled water must leave steam behind"
    );
}

#[test]
fn test_thermal_conduction_and_phase_transitions() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(320, 200);

    // Isolate conduction from advection so the three layers stay in contact.
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::Off;
    // Ambient heat exchange keeps the cooler air coupled to the lava; without
    // it an isolated liquid pixel (liquids can be pushed one row up by the
    // diagonal velocity move) would never lose heat.
    engine.config_mut().aheat_enable = true;

    let x0 = 100;
    let x1 = 159;
    fill_rect(&mut engine, x0, 90, x1, 90, PT_LAVA);
    fill_rect(&mut engine, x0, 91, x1, 94, PT_ICEI);
    fill_rect(&mut engine, x0, 95, x1, 98, PT_WATR);

    let lava = count_type(&engine, PT_LAVA);
    let ice = count_type(&engine, PT_ICEI);
    let water = count_type(&engine, PT_WATR);
    assert_eq!(lava, 60);
    assert_eq!(ice, 60 * 4);
    assert_eq!(water, 60 * 4);

    set_temp_of_type(&mut engine, PT_LAVA, 1800.0);
    assert!(
        max_temp_of_type(&engine, PT_ICEI) < 250.0,
        "ice must start at its 245 K spawn temperature"
    );

    let mut max_ice_temp = f32::NEG_INFINITY;
    let mut saw_steam = false;
    let mut saw_stone = false;
    let mut ticks = 0usize;
    for _ in 0..300 {
        step(&mut engine);
        ticks += 1;
        max_ice_temp = max_ice_temp.max(max_temp_of_type(&engine, PT_ICEI));
        saw_steam |= count_type(&engine, PT_WTRV) > 0;
        saw_stone |= count_type(&engine, PT_STNE) > 0;
        if count_type(&engine, PT_LAVA) == 0 && saw_steam && saw_stone {
            break;
        }
    }

    assert!(
        max_ice_temp > 250.0,
        "heat must conduct into the ice (hottest ice = {max_ice_temp} K after {ticks} ticks)"
    );
    assert!(
        count_type(&engine, PT_ICEI) < ice,
        "ice must melt: {} of {ice} pixels left after {ticks} ticks",
        count_type(&engine, PT_ICEI)
    );
    assert!(
        saw_steam,
        "boiling water/ice must produce WTRV within {ticks} ticks"
    );
    assert!(
        saw_stone,
        "lava cooled below its 973 K solidification point must become STNE"
    );
    assert_eq!(
        count_type(&engine, PT_LAVA),
        0,
        "every lava pixel must freeze after {ticks} ticks"
    );
    assert!(
        count_type(&engine, PT_STNE) > 0,
        "solidified lava must leave stone behind"
    );

    // The melt water is still on the canvas as water, steam or condensate.
    let melt_products =
        count_type(&engine, PT_WATR) + count_type(&engine, PT_WTRV) + count_type(&engine, PT_DSTW);
    assert!(
        melt_products > 0,
        "melted ice must leave water, steam or distilled water behind"
    );
}

#[test]
fn test_black_hole_folds_eaten_heat_into_its_air_cell() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(128, 96);
    // TPT's default configuration (`Simulation.h:112`): air on, heat
    // simulation on, ambient heat off, vertical gravity. Ambient heat stays off
    // while the particle falls so the air cell is still at `ambient_air_temp`
    // when the hole swallows it.
    engine.config_mut().aheat_enable = false;
    engine.config_mut().air_mode = AirMode::On;
    engine.config_mut().heat_enable = true;
    engine.config_mut().gravity_mode = GravityMode::Vertical;

    // `VACU`/`BHOL` spawns at `R_TEMP + 70 C` (`BHOL.cpp:29`).
    fill_rect(&mut engine, 64, 52, 64, 52, PT_BHOL);
    fill_rect(&mut engine, 64, 49, 64, 49, PT_SAND);
    set_temp_of_type(&mut engine, PT_SAND, 1500.0);

    let hole_id = (0..engine.bufs.capacity)
        .find(|&id| engine.bufs.part_ty[id] == PT_BHOL)
        .expect("the black hole is on the canvas");
    let spawn_temp = engine.bufs.part_temp[hole_id];
    assert!(
        (spawn_temp - 365.15).abs() < 0.5,
        "VACU spawns at R_TEMP + 70 C, got {spawn_temp} K"
    );

    let xcells = engine.bufs.config.xcells() as usize;
    let cell = (52 / 4) * xcells + 64 / 4;

    // Let the powder fall into the hole. On the contact frame the 255
    // HeatConduct of the hole also equilibrates both particles, so only the
    // direction of the temperature change is asserted.
    for _ in 0..120 {
        step(&mut engine);
        if count_type(&engine, PT_SAND) == 0 {
            break;
        }
    }
    assert_eq!(
        count_type(&engine, PT_SAND),
        0,
        "the black hole must swallow the particle that moves into it"
    );

    // `try_move` folds half the eaten particle's temperature into the hole
    // (`Simulation.cpp:1393-1401`).
    let hole_temp = engine.bufs.part_temp[hole_id];
    assert!(
        hole_temp > spawn_temp + 100.0,
        "eating a 1500 K particle must heat the hole: {spawn_temp} K -> {hole_temp} K"
    );

    // Ambient heat off pins the air field to `ambient_air_temp` every frame, so
    // the hole's own heat is the only source once it is enabled.
    let ambient = engine.config().ambient_air_temp;
    assert!((engine.bufs.air_hv[cell] - ambient).abs() < 0.01);
    engine.config_mut().aheat_enable = true;
    step(&mut engine);

    // Heat transfer with air (`Simulation.cpp:2455-2463`): the air cell loses
    // exactly `alpha * dtemp` with `alpha = min(0.04, 0.4 * hc) = 0.04`, the
    // same magnitude the particle gains. The port used to apply only 1/16 of
    // that to the air, which left a freshly fed hole's heat invisible.
    let expected = ambient + 0.04 * (hole_temp - ambient);
    let hv = engine.bufs.air_hv[cell];
    assert!(
        (hv - expected).abs() < 2.0,
        "the {hole_temp:.1} K hole must heat its air cell to {expected:.1} K, got {hv:.1} K"
    );
    assert!(
        hv > ambient + 10.0,
        "the hole's absorbed heat must reach the air field, got {hv:.1} K"
    );

    // The white hole is the mirror case: `WHOL`/`VENT` only eats `ANAR`, and
    // folds `-(MAX_TEMP - anar.temp) / 2` into its own temperature
    // (`Simulation.cpp:1402-1413`). A 3000 K anti-air pixel therefore drives
    // the vent to `MIN_TEMP`, and the same ambient exchange has to cool its
    // air cell.
    let mut vent = create_test_engine(128, 96);
    vent.config_mut().aheat_enable = false;
    vent.config_mut().air_mode = AirMode::On;
    vent.config_mut().heat_enable = true;
    vent.config_mut().gravity_mode = GravityMode::Vertical;

    fill_rect(&mut vent, 64, 52, 64, 52, PT_WHOL);
    // `ANAR` has `Gravity = -0.1` and negative advection (`ANAR.cpp:11-24`), so
    // it rises into the vent from the pixel below.
    fill_rect(&mut vent, 64, 53, 64, 53, PT_ANAR);
    set_temp_of_type(&mut vent, PT_ANAR, 3000.0);

    let vent_id = (0..vent.bufs.capacity)
        .find(|&id| vent.bufs.part_ty[id] == PT_WHOL)
        .expect("the white hole is on the canvas");
    let vent_xcells = vent.bufs.config.xcells() as usize;
    let vent_cell = (52 / 4) * vent_xcells + 64 / 4;
    let vent_ambient = vent.config().ambient_air_temp;
    for _ in 0..240 {
        step(&mut vent);
        if count_type(&vent, PT_ANAR) == 0 {
            break;
        }
    }
    assert_eq!(
        count_type(&vent, PT_ANAR),
        0,
        "the white hole must swallow the anti-air particle"
    );

    let vent_temp = vent.bufs.part_temp[vent_id];
    assert!(
        vent_temp < 50.0,
        "eating hot anti-air must cool the vent towards MIN_TEMP, got {vent_temp} K"
    );
    assert!((vent.bufs.air_hv[vent_cell] - vent_ambient).abs() < 0.01);
    vent.config_mut().aheat_enable = true;
    step(&mut vent);

    let expected_vent = vent_ambient + 0.04 * (vent_temp - vent_ambient);
    let hv_vent = vent.bufs.air_hv[vent_cell];
    assert!(
        (hv_vent - expected_vent).abs() < 2.0,
        "the {vent_temp:.1} K vent must cool its air cell to {expected_vent:.1} K, got {hv_vent:.1} K"
    );
    assert!(
        hv_vent < vent_ambient - 5.0,
        "the vent's absorbed cold must reach the air field, got {hv_vent:.1} K"
    );
}
