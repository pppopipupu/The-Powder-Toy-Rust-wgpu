//! `try_move`'s shared-pixel (`e == 2`) and half-silvered-mirror rules, plus
//! the `WOOD -> SAWD` probe.
//!
//! These drive the real pipeline through `step()`, so the assertions cover the
//! pair pass running between the intent and commit launches, not just the
//! helper functions: `Simulation.cpp:1205-1218,1258-1381,1529-1534`.

use std::collections::HashSet;

use super::*;

/// Finds the single live slot of `ty`, panicking when it is missing.
fn single_slot(engine: &SimulationEngine, ty: u32) -> usize {
    let mut found = None;
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] == ty {
            assert!(found.is_none(), "more than one {ty} particle");
            found = Some(id);
        }
    }
    found.unwrap_or_else(|| panic!("no particle of type {ty}"))
}

fn pixel_of(engine: &SimulationEngine, id: usize) -> (i32, i32) {
    (
        engine.bufs.part_x[id].round() as i32,
        engine.bufs.part_y[id].round() as i32,
    )
}

/// `Simulation.cpp:1333-1343`: `PHOT` + `H2` -> `PROT` + `ELEC`.
///
/// The reference changes the photon into a proton in place, then rebuilds the
/// hydrogen slot as a fresh electron on the *photon's origin*
/// (`create_part(ID(r), x, y, PT_ELEC)`), and returns 1 so the pair does not
/// move. The port resolves the pair before the commit launch: the electron is
/// relocated to the mover's origin and the mover still commits its intent.
/// Both particles therefore end on distinct pixels, and the electron occupies
/// the old mover pixel rather than the hydrogen's pixel.
#[test]
fn test_photon_hydrogen_pair_keeps_one_particle_per_pixel() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(120, 80);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 60, 40, 60, 40, PT_H2);
    let h2 = single_slot(&engine, PT_H2);
    // Freeze the gas so the photon, not the hydrogen's own random walk,
    // decides the pair.
    engine.bufs.part_flags[h2] |= FLAG_SKIPMOVE;
    engine.bufs.mark_particle_slot_dirty(h2);

    engine.apply_command(SimCommand::SpawnParticle {
        x: 59,
        y: 40,
        elem_ty: PT_PHOT,
        vx: 1.0,
        vy: 0.0,
    });

    step(&mut engine);

    assert_eq!(
        engine.bufs.part_ty[h2], PT_ELEC,
        "`part_change_type` targets the hydrogen's slot"
    );
    assert_eq!(
        engine.bufs.part_x[h2], 59.0,
        "the electron must be rebuilt on the mover's origin x"
    );
    assert_eq!(
        engine.bufs.part_y[h2], 40.0,
        "the electron must be rebuilt on the mover's origin y"
    );

    assert_eq!(count_type(&engine, PT_H2), 0, "the hydrogen is consumed");
    assert_eq!(count_type(&engine, PT_PHOT), 0, "the photon is consumed");
    assert_eq!(
        count_type(&engine, PT_PROT),
        1,
        "the photon becomes a proton"
    );
    assert_eq!(
        count_type(&engine, PT_ELEC),
        1,
        "the hydrogen becomes an electron"
    );

    let prot = single_slot(&engine, PT_PROT);
    assert_eq!(
        pixel_of(&engine, prot),
        (60, 40),
        "the proton keeps the photon's move into the hydrogen's pixel"
    );

    // Exactly one particle per pixel: the pair pass must not leave the proton
    // and the new electron sharing the origin, nor the electron under the
    // hydrogen's old pixel.
    let mut pixels = HashSet::new();
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] == PT_NONE {
            continue;
        }
        let pixel = pixel_of(&engine, id);
        assert!(
            pixels.insert(pixel),
            "two particles share pixel {pixel:?} after the PHOT+H2 pair"
        );
    }
    assert_eq!(pixels.len(), 2);
}

/// `Simulation.cpp:1261-1276`: `PHOT` + `C5` records the photon's sub-pixel
/// entry offset in the C5's `tmp2`.
///
/// `tmp2 = (0xFFFF & int((x - int(x)) * 255)) | (0xFFFF0000 & int((y - int(y)) * 16711680))`.
/// The origin must be read *before* the commit writes the photon's position,
/// otherwise the photon is already standing on the C5 pixel and the offset is
/// always zero.
#[test]
fn test_photon_c5_records_pre_move_subpixel_origin() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(120, 80);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    fill_rect(&mut engine, 60, 40, 60, 40, PT_C5);
    let c5 = single_slot(&engine, PT_C5);
    engine.bufs.part_life[c5] = 0;
    engine.bufs.part_ctype[c5] = 0;
    engine.bufs.part_tmp2[c5] = 0;
    engine.bufs.mark_particle_slot_dirty(c5);

    engine.apply_command(SimCommand::SpawnParticle {
        x: 59,
        y: 40,
        elem_ty: PT_PHOT,
        vx: 1.0,
        vy: 0.0,
    });
    let phot = single_slot(&engine, PT_PHOT);
    // A non-integral origin: the photon occupies pixel 59 but its float centre
    // is 0.4 px to the right, exactly the value the reference stores.
    let origin_x = 59.4f32;
    let origin_y = 40.0f32;
    engine.bufs.part_x[phot] = origin_x;
    engine.bufs.part_y[phot] = origin_y;
    engine.bufs.part_ctype[phot] = 0x00FF_00FF;
    engine.bufs.mark_particle_slot_dirty(phot);

    step(&mut engine);

    assert_eq!(count_type(&engine, PT_PHOT), 0, "the C5 absorbs the photon");
    let origin_pixel_x = (origin_x + 0.5f32) as i32;
    let expected_x = (((origin_x - origin_pixel_x as f32) * 255.0f32) as i32) & 0xFFFF;
    let expected_y = (((origin_y - (origin_y + 0.5f32) as i32 as f32) * 16_711_680.0f32) as i32)
        & 0xFFFF_0000u32 as i32;
    assert_ne!(expected_x, 0, "fixture must exercise a non-zero offset");
    assert_eq!(
        engine.bufs.part_tmp2[c5],
        expected_x | expected_y,
        "C5's tmp2 must carry the mover's pre-move sub-pixel origin"
    );
}

/// `Simulation.cpp:1205-1206`: the half-silvered `BMTL` mirror lets a blocked
/// photon in only when its 1-in-2 roll passes; otherwise the photon reflects
/// and never enters the mirror pixel.
///
/// The roll is seeded from the frame counter, so a single frame cannot choose
/// an outcome from the host. Repeating fresh attempts exercises both branches
/// deterministically for a given run: a passing photon shares the mirror pixel
/// and keeps moving right, a reflected one stays on the incoming side and
/// reverses `vx`.
#[test]
fn test_bmtl_half_mirror_reflects_and_passes() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(80, 60);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    let attempts = 64usize;
    let mut reflected = 0usize;
    let mut passed = 0usize;
    for _ in 0..attempts {
        engine.apply_command(SimCommand::ClearAll);
        fill_rect(&mut engine, 40, 28, 40, 32, PT_BMTL);
        engine.apply_command(SimCommand::SpawnParticle {
            x: 39,
            y: 30,
            elem_ty: PT_PHOT,
            vx: 1.0,
            vy: 0.0,
        });

        step(&mut engine);

        let phot = single_slot(&engine, PT_PHOT);
        let x = engine.bufs.part_x[phot];
        if x >= 40.0 {
            passed += 1;
            assert_eq!(
                x, 40.0,
                "a photon that the mirror lets through shares the BMTL pixel"
            );
            assert!(
                engine.bufs.part_vx[phot] > 0.0,
                "a passing photon keeps its incoming direction"
            );
        } else {
            reflected += 1;
            assert!(
                x <= 39.0,
                "a reflected photon must not enter the mirror pixel (x = {x})"
            );
            assert!(
                engine.bufs.part_vx[phot] < 0.0,
                "the mirror must reverse the reflected photon's velocity"
            );
        }
    }

    assert!(
        reflected > 0,
        "the half-silvered mirror must reflect the failed 1-in-2 rolls \
         (reflected {reflected}/{attempts})"
    );
    assert!(
        passed > 0,
        "the half-silvered mirror must also let photons pass \
         (passed {passed}/{attempts})"
    );
}

/// `Simulation.cpp:1211-1218`: a blocked probe against `WOOD` saws it into
/// `SAWD` when the mover's speed exceeds 5, and leaves it untouched below the
/// threshold.
///
/// `PlanMove` samples the velocity path in 2 px steps, so a one-pixel obstacle
/// can be jumped over. The wall therefore covers every sample of both probes;
/// the sampled stop pixel decides `saw_wood_at`'s speed test.
#[test]
fn test_wood_becomes_sawdust_above_speed_five_only() {
    let _guard = gpu_lock();

    let blocked_by_wood = |vx0: f32| -> (usize, usize) {
        let mut engine = create_test_engine(120, 80);
        engine.config_mut().gravity_mode = GravityMode::Off;
        engine.config_mut().air_mode = AirMode::NoUpdate;
        engine.config_mut().aheat_enable = false;

        fill_rect(&mut engine, 60, 40, 68, 40, PT_WOOD);
        engine.apply_command(SimCommand::SpawnParticle {
            x: 59,
            y: 40,
            elem_ty: PT_DUST,
            vx: vx0,
            vy: 0.0,
        });

        step(&mut engine);
        (count_type(&engine, PT_WOOD), count_type(&engine, PT_SAWD))
    };

    // DUST integrates `vx * Loss` with `Loss = 0.80`, so these spawn speeds
    // become roughly 6.5 and 1.7 px/frame in the movement pass.
    let (fast_wood, fast_sawd) = blocked_by_wood(8.0);
    assert!(
        fast_sawd > 0,
        "a probe moving faster than 5 px/frame must saw the wood"
    );
    assert_eq!(
        fast_wood + fast_sawd,
        9,
        "the wood pixels only change type, they are not consumed"
    );

    let (slow_wood, slow_sawd) = blocked_by_wood(2.0);
    assert_eq!(
        slow_sawd, 0,
        "a probe moving slower than 5 px/frame must leave the wood intact"
    );
    assert_eq!(slow_wood, 9, "the slow probe must not consume the wood");
}
