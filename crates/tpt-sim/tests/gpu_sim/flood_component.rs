//! One-call 4-connected floods: a run taller/longer than any per-frame round
//! budget still reaches its whole component inside the frame.
//!
//! The per-tick element floods (`LED`, `BANG`) run on the device drain in
//! `reaction_kernel::flood_component`; the editor floods (`PropertyField` fill
//! and `FloodFillParticles`) run until their own fixed point. Each test builds
//! a run whose 4-connected distance exceeds the old per-frame bound
//! (`32` rounds for the element floods, `xres + yres` for the editor floods).

use super::*;

/// `LED.cpp:78-81`: a sparked conductor latches the whole LED run in the
/// trigger frame, however long the run is.
///
/// The run starts at `x = 4`: `Simulation.cpp`'s margin gate kills any particle
/// with `x < CELL` (`CELL == 4`) on the next tick, so a border-hugging run
/// cannot survive to be observed.
#[test]
fn test_led_flood_lights_a_long_run_in_one_tick() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(256, 64);
    fill_rect(&mut engine, 4, 20, 4, 20, PT_PSCN);
    fill_rect(&mut engine, 5, 20, 84, 20, PT_LED);
    spark_at(&mut engine, 4, 20);

    engine.step();
    engine.ensure_host_current();

    // The spark itself is a travelling conductor: it takes over the LEDs it
    // touches, so only the pixels that are still `LED` can carry the latch.
    let mut lit = 0;
    for slot in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[slot] == PT_LED {
            assert_eq!(
                engine.bufs.part_life[slot], 8,
                "LED at slot {slot} was not latched on the trigger frame"
            );
            lit += 1;
        }
    }
    for x in 5i32..=84 {
        let slot = engine
            .bufs
            .particle_index_at(x, 20)
            .unwrap_or_else(|| panic!("pixel ({x}, 20) of the run is empty"));
        let ty = engine.bufs.part_ty[slot];
        if ty == PT_LED {
            assert_eq!(engine.bufs.part_life[slot], 8, "LED at ({x}, 20) is unlit");
        } else {
            // The spark eats the first conductor it walks into; nothing else
            // may replace a run pixel.
            assert_eq!(ty, PT_SPRK, "run pixel ({x}, 20) became {ty}");
        }
    }
    assert!(lit >= 78, "only {lit} of the 80 LEDs stayed LED");
}

/// `BANG.cpp:74-80`: `flood_prop(x, y, FIELD_TMP, 2)` arms the entire charge
/// in the frame the seed sees fire, not one 16-pixel layer per frame.
#[test]
fn test_tnt_run_arms_in_one_tick() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(256, 64);
    fill_rect(&mut engine, 4, 20, 64, 20, PT_BANG);
    let seed = engine.bufs.particle_index_at(4, 20).expect("seed slot");
    engine.apply_command(SimCommand::SetParticleProperty {
        part_id: seed as u32,
        field: "tmp".to_string(),
        value: 1.0,
    });

    engine.step();
    engine.ensure_host_current();

    let mut armed = 0;
    for slot in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[slot] == PT_BANG {
            assert_eq!(
                engine.bufs.part_tmp[slot], 2,
                "BANG at slot {slot} missed the mass arming wave"
            );
            armed += 1;
        }
    }
    assert_eq!(armed, 61, "the whole 61-pixel charge must still be BANG");
}

/// `Simulation::flood_prop` has no distance cap: the property fill must reach a
/// run whose winding length exceeds `xres + yres`.
#[test]
fn test_property_flood_walks_a_winding_run() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(256, 64);
    // Serpentine of 625 connected pixels; the straight-line distance is 20.
    fill_rect(&mut engine, 10, 10, 210, 10, PT_DUST);
    fill_rect(&mut engine, 210, 10, 210, 20, PT_DUST);
    fill_rect(&mut engine, 10, 20, 210, 20, PT_DUST);
    fill_rect(&mut engine, 10, 20, 10, 30, PT_DUST);
    fill_rect(&mut engine, 10, 30, 210, 30, PT_DUST);

    let touched = tpt_sim::flood_kernel::flood_property(
        &mut engine.bufs,
        10,
        10,
        tpt_sim::flood_kernel::PropertyField::Tmp,
        tpt_sim::flood_kernel::PropertyValue::Int(7),
    );
    assert!(
        touched > 300,
        "the walk stopped after {touched} pixels, before the far row"
    );

    engine.ensure_host_current();
    let far = engine.bufs.particle_index_at(210, 30).expect("far end");
    assert_eq!(
        engine.bufs.part_tmp[far], 7,
        "the far end of the winding run was not reached"
    );
}

/// `FloodParts` walks the empty region without a Manhattan-distance cap.
#[test]
fn test_flood_fill_walks_a_serpentine_corridor() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(160, 48);
    // Full-width barriers with alternating 12-pixel gaps: the corridor from the
    // seed to the bottom region is ~600 pixels, while `xres + yres` is 208.
    fill_rect(&mut engine, 0, 8, 147, 8, PT_DMRN);
    fill_rect(&mut engine, 12, 16, 159, 16, PT_DMRN);
    fill_rect(&mut engine, 0, 24, 147, 24, PT_DMRN);
    fill_rect(&mut engine, 12, 32, 159, 32, PT_DMRN);
    fill_rect(&mut engine, 0, 40, 147, 40, PT_DMRN);

    engine.apply_command(SimCommand::FloodFillParticles {
        x: 5,
        y: 4,
        elem_ty: PT_DUST,
    });
    engine.ensure_host_current();

    let far = engine
        .bufs
        .particle_index_at(150, 44)
        .expect("the bottom region was never reached");
    assert_eq!(engine.bufs.part_ty[far], PT_DUST);
}
