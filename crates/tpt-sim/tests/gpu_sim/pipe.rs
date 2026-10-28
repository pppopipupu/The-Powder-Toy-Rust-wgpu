//! `PIPE` / `PPIP` state tests.
//!
//! The pipe transport itself is covered by the reaction suites; these tests pin
//! the two pieces of `Element_PIPE_update` that talk to the *border* rather than
//! to the cargo: the stasis marker a pause flip writes into the neighbouring
//! `BRCK` (`PIPE.cpp:126-145`) and the glowing border a `PPIP` is born with
//! (`PIPE.cpp:279-285`).

use super::*;

// The bits TPT writes into the pipe's own `tmp` (`PIPE.h:13-25`,
// `PIPE.cpp:57-62`). The pipe has no command that pokes these, so the tests set
// them exactly where TPT's flood would leave them.
/// `PFLAG_COLOR_RED` (`PIPE.h:22`).
const PFLAG_COLOR_RED: i32 = 0x0004_0000;
/// `PFLAG_COLOR_BLUE` (`PIPE.h:24`).
const PFLAG_COLOR_BLUE: i32 = 0x000C_0000;
/// `PFLAG_COLORS` (`PIPE.h:25`).
const PFLAG_COLORS: i32 = 0x000C_0000;
/// `PPIP_TMPFLAG_PAUSED` (`PIPE.cpp:58`).
const PPIP_PAUSED: i32 = 0x0200_0000;
/// `PPIP_TMPFLAG_TRIGGER_OFF` (`PIPE.cpp:60`).
const PPIP_TRIGGER_OFF: i32 = 0x0800_0000;
/// `PPIP_TMPFLAG_TRIGGER_ON` (`PIPE.cpp:61`).
const PPIP_TRIGGER_ON: i32 = 0x1000_0000;

/// `PIPE.cpp:111-147`: a `PPIP` whose trigger flips the pause bit writes
/// `tmp = 0` into every `BRCK` within two pixels while it is paused and
/// `tmp = 1` when it runs again, and the reference pushes the cargo in the very
/// frame the resume trigger lands (`PIPE.cpp:114-118` clears `PAUSED` before the
/// colour branch at `PIPE.cpp:165-232` runs).
#[test]
fn test_paused_ppip_marks_the_border_and_resumes_in_the_same_frame() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    // A red `PPIP` holding water feeds a blue `PIPE` (`nextColor(red) == blue`,
    // `PIPE.cpp:93-103`). The bricks cover all three distances the reference's
    // `rx`/`ry` box distinguishes: one neighbour, two pixels away on an axis,
    // two pixels away diagonally, and one out of reach at three pixels.
    fill_rect(&mut engine, 20, 20, 20, 20, PT_PPIP);
    fill_rect(&mut engine, 21, 20, 21, 20, PT_PIPE);
    fill_rect(&mut engine, 21, 19, 21, 19, PT_BRCK);
    fill_rect(&mut engine, 20, 22, 20, 22, PT_BRCK);
    fill_rect(&mut engine, 22, 22, 22, 22, PT_BRCK);
    fill_rect(&mut engine, 17, 20, 17, 20, PT_BRCK);

    engine.apply_command(SimCommand::SetParticleFieldAt {
        points: vec![(20, 20)],
        field: ParticleField::Tmp,
        value: (PFLAG_COLOR_RED | PPIP_TRIGGER_OFF) as i64,
    });
    engine.apply_command(SimCommand::SetParticleFieldAt {
        points: vec![(21, 20)],
        field: ParticleField::Tmp,
        value: PFLAG_COLOR_BLUE as i64,
    });
    engine.apply_command(SimCommand::SetParticleFieldAt {
        points: vec![(20, 20)],
        field: ParticleField::Ctype,
        value: PT_WATR as i64,
    });
    // A value the reference would never write, so "untouched" is observable.
    engine.apply_command(SimCommand::SetParticleFieldAt {
        points: vec![(17, 20)],
        field: ParticleField::Tmp,
        value: 7,
    });

    // Frame 1: `TRIGGER_OFF` pauses the pipe and dims its border.
    step(&mut engine);

    let sender = engine.bufs.particle_at(20, 20).expect("PPIP placed");
    assert_eq!(sender.ty, PT_PPIP);
    assert_ne!(
        sender.tmp & PPIP_PAUSED,
        0,
        "TRIGGER_OFF must set the pause bit (`PIPE.cpp:120-125`)"
    );
    assert_eq!(
        sender.tmp & PPIP_TRIGGER_OFF,
        0,
        "the trigger bits are consumed after they are applied (`PIPE.cpp:163`)"
    );
    assert_eq!(
        sender.tmp & PFLAG_COLORS,
        PFLAG_COLOR_RED,
        "pausing must not disturb the flow colour (`PIPE.cpp:114-125`)"
    );
    assert_eq!(
        sender.ctype, PT_WATR,
        "a paused pipe keeps the cargo it holds"
    );
    for (x, y) in [(21, 19), (20, 22), (22, 22)] {
        let brick = engine
            .bufs
            .particle_at(x, y)
            .unwrap_or_else(|| panic!("no brick at ({x}, {y})"));
        assert_eq!(brick.ty, PT_BRCK);
        assert_eq!(
            brick.tmp, 0,
            "the pause marker must dim the brick at ({x}, {y}) (`PIPE.cpp:137-138`)"
        );
    }
    assert_eq!(
        engine.bufs.particle_at(17, 20).expect("far brick").tmp,
        7,
        "a brick more than two pixels away is outside the 5x5 box (`PIPE.cpp:128-131`)"
    );

    // Frame 2: `TRIGGER_ON` clears the pause bit and the same update pushes the
    // cargo, so the resume lands on this tick and not the next.
    engine.apply_command(SimCommand::SetParticleFieldAt {
        points: vec![(20, 20)],
        field: ParticleField::Tmp,
        value: (sender.tmp | PPIP_TRIGGER_ON) as i64,
    });
    step(&mut engine);

    let sender = engine.bufs.particle_at(20, 20).expect("PPIP placed");
    assert_eq!(
        sender.tmp & PPIP_PAUSED,
        0,
        "TRIGGER_ON resumes the pipe (`PIPE.cpp:114-118`)"
    );
    for (x, y) in [(21, 19), (20, 22), (22, 22)] {
        let brick = engine
            .bufs
            .particle_at(x, y)
            .unwrap_or_else(|| panic!("no brick at ({x}, {y})"));
        assert_eq!(
            brick.tmp, 1,
            "the resume marker must light the brick at ({x}, {y}) (`PIPE.cpp:139-140`)"
        );
    }
    assert_eq!(
        engine.bufs.particle_at(17, 20).expect("far brick").tmp,
        7,
        "the 5x5 box must not reach three pixels"
    );

    assert_eq!(
        sender.ctype, 0,
        "the cargo leaves on the resume tick (`PIPE.cpp:165-232` runs unpaused)"
    );
    let receiver = engine.bufs.particle_at(21, 20).expect("PIPE placed");
    assert_eq!(receiver.ty, PT_PIPE);
    assert_eq!(
        receiver.ctype, PT_WATR,
        "the cargo reached the next pipe on the resume tick"
    );
}

/// `PIPE.cpp:279-285`: the `BRCK` border of a `PPIP` is created glowing
/// (`parts[index].tmp = 1`) while a plain `PIPE`'s border keeps `tmp = 0`.
#[test]
fn test_ppip_border_bricks_are_created_glowing() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;

    // Uncoloured pipes at `life <= 10` seal themselves with `BRCK`
    // (`PIPE.cpp:268-290`). `FLAG_SPAWN_CLONE` carries one pixel per frame, so
    // each pipe creates exactly one brick in the first tick.
    fill_rect(&mut engine, 20, 20, 20, 20, PT_PPIP);
    fill_rect(&mut engine, 40, 20, 40, 20, PT_PIPE);
    engine.apply_command(SimCommand::SetParticleFieldAt {
        points: vec![(20, 20), (40, 20)],
        field: ParticleField::Life,
        value: 10,
    });

    step(&mut engine);

    assert_eq!(
        count_type(&engine, PT_BRCK),
        2,
        "each initialising pipe must create one border brick per frame"
    );
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] != PT_BRCK {
            continue;
        }
        let x = engine.bufs.part_x[id] as i32;
        let y = engine.bufs.part_y[id] as i32;
        if (x - 20).abs() <= 1 && (y - 20).abs() <= 1 {
            assert_eq!(
                engine.bufs.part_tmp[id], 1,
                "the PPIP's border brick at ({x}, {y}) must glow (`PIPE.cpp:283-284`)"
            );
        } else if (x - 40).abs() <= 1 && (y - 20).abs() <= 1 {
            assert_eq!(
                engine.bufs.part_tmp[id], 0,
                "a plain PIPE's border brick at ({x}, {y}) must not glow (`PIPE.cpp:283`)"
            );
        } else {
            panic!("unexpected brick at ({x}, {y})");
        }
    }
}
