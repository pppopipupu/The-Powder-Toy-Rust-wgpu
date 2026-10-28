use super::*;
use cubecl::bytes::Bytes;

/// Default colour mode must paint particles, not just the background.
///
/// Regression guard: an all-black framebuffer still satisfies the blit parity
/// and "non-zero bytes" checks, so the interactive viewport went black without
/// any test failing.
#[test]
fn test_default_colour_mode_paints_sand() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(160, 120);
    fill_rect(&mut engine, 40, 40, 79, 79, PT_SAND);
    engine.config_mut().display_mode = 0;
    engine.config_mut().colour_mode = COLOUR_DEFAULT;

    let rgba = engine.render_rgba();
    let bright = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] as u32 + px[1] as u32 + px[2] as u32 > 150)
        .count();
    assert!(
        bright > 200,
        "default colour mode must draw the 40x40 sand block, bright pixels = {bright}"
    );
}

/// A canvas resize must not leak allocator garbage into device-only buffers.
///
/// Regression guard: `fire_r/g/b`, the gravity field, the blast grid and the
/// framebuffer are never uploaded by a host pass, so allocating them with
/// `empty` left whatever the driver returned. Starting a smaller canvas from
/// the title screen (612x384 -> 256x256) then rendered a saturated red wash
/// over the whole viewport.
#[test]
fn test_canvas_resize_resets_device_only_buffers() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(1024, 768);
    engine.resize_canvas(256, 256);

    let floor: Vec<(i32, i32)> = (244..256)
        .flat_map(|y| (0..256).map(move |x| (x, y)))
        .collect();
    engine.apply_command(SimCommand::PlaceParticles {
        points: floor,
        elem_ty: PT_STNE,
        replace_mode: true,
        dcolour: 0,
    });
    let water: Vec<(i32, i32)> = (216..244)
        .flat_map(|y| (8..248).map(move |x| (x, y)))
        .collect();
    engine.apply_command(SimCommand::PlaceParticles {
        points: water,
        elem_ty: PT_WATR,
        replace_mode: true,
        dcolour: 0,
    });
    for _ in 0..10 {
        step(&mut engine);
    }

    engine.config_mut().display_mode = 0;
    engine.config_mut().colour_mode = COLOUR_DEFAULT;
    let rgba = engine.render_rgba();
    let px = |x: usize, y: usize| {
        let i = (y * 256 + x) * 4;
        (rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3])
    };
    assert_eq!(
        px(128, 4),
        (0, 0, 0, 255),
        "air must render black after a resize, not a stale fire glow"
    );
    let water_px = px(128, 226);
    assert!(
        water_px.2 > water_px.0 && water_px.2 > 100,
        "water must stay blue after a resize, got {water_px:?}"
    );
}

#[test]
fn test_gpu_render_rgba() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(320, 200);

    fill_rect(&mut engine, 120, 120, 139, 139, PT_SAND);
    fill_rect(&mut engine, 160, 120, 169, 129, PT_LAVA);
    set_temp_of_type(&mut engine, PT_LAVA, 1800.0);
    for _ in 0..2 {
        step(&mut engine);
    }

    let stroke: Vec<(i32, i32)> = (100..220)
        .flat_map(|x| (60..140).map(move |y| (x, y)))
        .collect();
    engine.apply_command(SimCommand::ApplySimTool {
        points: stroke.clone(),
        tool: SimToolKind::Air,
        strength: 20.0,
        dir: (0.0, 0.0),
        center: (0, 0),
        dragging: true,
    });
    engine.apply_command(SimCommand::ApplySimTool {
        points: stroke,
        tool: SimToolKind::Wind,
        strength: 5.0,
        dir: (1.0, 0.0),
        center: (0, 0),
        dragging: true,
    });

    let expected_len = (engine.config().xres * engine.config().yres * 4) as usize;

    for mode in [DISPLAY_AIRP, DISPLAY_AIRV, DISPLAY_AIRH] {
        engine.config_mut().display_mode = mode;
        engine.config_mut().colour_mode = COLOUR_DEFAULT;
        let (len, non_black) = render_stats(&mut engine);
        assert_eq!(
            len, expected_len,
            "display mode {mode:#x} must render exactly xres * yres * 4 bytes"
        );
        assert!(
            non_black > 0,
            "display mode {mode:#x} must produce non-zero RGBA output"
        );
    }

    engine.config_mut().display_mode = 0;
    engine.config_mut().colour_mode = COLOUR_HEAT;
    let (len, non_black) = render_stats(&mut engine);
    assert_eq!(
        len, expected_len,
        "COLOUR_HEAT must render exactly xres * yres * 4 bytes"
    );
    assert!(non_black > 0, "COLOUR_HEAT must colour the hot lava pixels");

    // Alpha is always opaque, which is what `queue.write_texture` expects.
    let rgba = engine.render_rgba();
    assert_eq!(rgba.len(), expected_len);
    assert!(
        rgba.as_chunks::<4>().0.iter().all(|px| px[3] == 0xFF),
        "every rendered pixel must be fully opaque"
    );
}

#[test]
fn test_graphics_rules_colour_particles() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;

    // `GOLD.cpp` graphics: a per-particle +-5 channel jitter. Four adjacent
    // pixels must therefore not all render with the same colour.
    fill_rect(&mut engine, 10, 10, 13, 10, PT_GOLD);
    step(&mut engine);
    let xres = engine.config().xres as usize;
    let rgba = engine.render_rgba();
    let pixel = |x: usize, y: usize| {
        let i = (y * xres + x) * 4;
        (rgba[i], rgba[i + 1], rgba[i + 2])
    };
    let colours: Vec<_> = (10..=13).map(|x| pixel(x, 10)).collect();
    assert!(
        colours.windows(2).any(|w| w[0] != w[1]),
        "GOLD graphics jitter must vary between particles: {colours:?}"
    );

    // `STOR.cpp` graphics: an empty storer keeps its base colour, one that
    // holds cargo (`tmp != 0`) renders brighter.
    engine.apply_command(SimCommand::ClearAll);
    fill_rect(&mut engine, 20, 20, 20, 20, PT_STOR);
    let uncharged = {
        let rgba = engine.render_rgba();
        let i = (20 * xres + 20) * 4;
        (rgba[i], rgba[i + 1], rgba[i + 2])
    };
    engine.apply_command(SimCommand::SetParticleFieldAt {
        points: vec![(20, 20)],
        field: tpt_core::commands::ParticleField::Tmp,
        value: 5,
    });
    let charged = {
        let rgba = engine.render_rgba();
        let i = (20 * xres + 20) * 4;
        (rgba[i], rgba[i + 1], rgba[i + 2])
    };
    assert_ne!(
        uncharged, charged,
        "a loaded STOR must render differently from an empty one"
    );
}

/// `Element::defaultGraphics` drives the render mode of every element without
/// its own `Graphics` function: radioactive elements glow, liquids blur and
/// gases render through the fire buffer.
///
/// Regression guard: the element table used to store `PMODE_FLAT` for all
/// three groups, so `PLUT`/`URAN` had no glow at all and gases were drawn with
/// their full flat colour.
#[test]
fn test_default_graphics_glow_blur_and_gas_fire() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(96, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().display_mode = 0;
    engine.config_mut().colour_mode = COLOUR_DEFAULT;
    engine.config_mut().render_mode = RENDER_FANCY;

    fill_rect(&mut engine, 60, 20, 79, 29, PT_PLUT);
    fill_rect(&mut engine, 20, 20, 39, 39, PT_WATR);
    fill_rect(&mut engine, 60, 45, 79, 54, PT_H2);

    let xres = engine.config().xres as usize;
    let rgba = engine.render_rgba();
    let pixel = |x: usize, y: usize| {
        let i = (y * xres + x) * 4;
        (rgba[i], rgba[i + 1], rgba[i + 2])
    };
    let sum = |p: (u8, u8, u8)| p.0 as u32 + p.1 as u32 + p.2 as u32;

    assert_eq!(
        sum(pixel(5, 5)),
        0,
        "the empty background must stay black for the halo checks"
    );
    let halo = pixel(59, 24);
    assert!(
        sum(halo) > 0,
        "a radioactive element must glow outside its pixels (`Element.cpp:236`): {halo:?}"
    );
    // The glow gather reaches five pixels, and the effect-flag gate skips a
    // pixel whose 5x5 cell window holds no glow source, so the last lit pixel
    // and the first dark one pin that window.
    let far = pixel(55, 24);
    assert!(
        sum(far) > 0,
        "the glow must reach five pixels from the source: {far:?}"
    );
    let beyond = pixel(52, 24);
    assert_eq!(
        sum(beyond),
        0,
        "the glow must not reach eight pixels from the source: {beyond:?}"
    );
    let spill = pixel(19, 30);
    assert!(
        sum(spill) > 0,
        "a liquid must blur outside its pixels (`Element.cpp:238-241`): {spill:?}"
    );
    let gas = pixel(70, 50);
    assert!(
        gas.2 < 0x80,
        "a gas must render through the fire buffer, not flat (`Element.cpp:243-249`): {gas:?}"
    );
}

/// `Renderer::draw_other` tints the whole canvas while `emp_decor > 0`.
///
/// `SPRK` -> `EMP` is the only way TPT raises `emp_decor`, so the port raises it
/// when an `EMP` is placed. The flash must survive into the render pass and
/// must disappear again once it decays.
#[test]
fn test_emp_flash_tints_the_canvas() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().render_mode = RENDER_EFFE | RENDER_BASC;
    let xres = engine.config().xres as usize;
    let pixel = |rgba: &[u8], x: usize, y: usize| {
        let i = (y * xres + x) * 4;
        (rgba[i], rgba[i + 1], rgba[i + 2])
    };

    let before = pixel(engine.render_rgba(), 4, 4);
    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(32, 32)],
        elem_ty: PT_EMP,
        replace_mode: false,
        dcolour: 0,
    });
    // `SPRK.cpp:281-284` adds 3 per trigger, so one `EMP` is a faint tint
    // (`alpha = 3 / 110 * 255 = 6`), not a white-out.
    let flashed = pixel(engine.render_rgba(), 4, 4);
    assert!(
        flashed.2 > before.2 + 3 && flashed.2 < 40,
        "EMP flash must tint the canvas blue: before {before:?} after {flashed:?}"
    );

    // `emp_decor` decays by `emp_decor/25 + 2` per tick, so 40 -> 37 -> 34 ...
    // and reaches 0 after a few dozen frames.
    for _ in 0..40 {
        step(&mut engine);
    }
    let settled = pixel(engine.render_rgba(), 4, 4);
    assert!(
        settled.2 <= before.2 + 4,
        "EMP flash must decay away: before {before:?} settled {settled:?}"
    );
}

/// `Renderer::draw_grav_zones` marks every cell the gravity mask covers.
///
/// `UpdateGravityMask` floods from the border through non-`WL_GRAV` cells, so an
/// empty canvas is fully masked and a gravity-wall enclosure is not.
#[test]
fn test_gravity_zones_overlay_follows_the_mask() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().render_mode = RENDER_BASC;
    let xres = engine.config().xres as usize;
    let pixel = |rgba: &[u8], x: usize, y: usize| {
        let i = (y * xres + x) * 4;
        (rgba[i], rgba[i + 1], rgba[i + 2])
    };

    let before = pixel(engine.render_rgba(), 2, 2);
    engine.config_mut().gravity_zones_enabled = true;
    let marked = pixel(engine.render_rgba(), 2, 2);
    assert_ne!(
        before, marked,
        "enabling gravity zones must draw the checker overlay"
    );
}

/// `PROT.cpp:232-241`: a proton is a solid dark-red pixel (`PMODE_FLAT`) with a
/// 7-alpha fire blend on top, not a gas. Rendering it with `FIRE_BLEND` alone
/// drops the solid pixel and leaves a faint haze, which is what made protons
/// look like a gas.
#[test]
fn test_proton_renders_a_solid_pixel_not_a_gas() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().colour_mode = COLOUR_DEFAULT;

    fill_rect(&mut engine, 32, 32, 32, 32, PT_PROT);
    let xres = engine.config().xres as usize;
    let rgba = engine.render_rgba();
    let i = (32 * xres + 32) * 4;
    assert_eq!(rgba[i + 3], 255, "the proton pixel must be opaque");
    assert!(
        rgba[i] > 120 && rgba[i + 1] < 80 && rgba[i + 2] < 80,
        "PROT is 0x990000 with a 7-alpha fire blend, got {:?}",
        &rgba[i..i + 4]
    );
    // A neighbouring empty pixel must stay background: the fire blend is far
    // too weak to bleed into the cell.
    let j = (32 * xres + 30) * 4;
    assert!(
        rgba[j] < 40 && rgba[j + 1] < 40 && rgba[j + 2] < 40,
        "the proton glow must not wash out its neighbours, got {:?}",
        &rgba[j..j + 4]
    );
}

/// `DEUT.cpp:148-157`: deuterium above life 240 brightens its own pixel by 255
/// per channel and feeds the fire buffer at alpha 60; below that it only blurs.
#[test]
fn test_hot_deuterium_brightens_and_glows() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().colour_mode = COLOUR_DEFAULT;

    fill_rect(&mut engine, 20, 20, 20, 20, PT_DEUT);
    let xres = engine.config().xres as usize;
    let cold = {
        let rgba = engine.render_rgba();
        let i = (20 * xres + 20) * 4;
        (rgba[i], rgba[i + 1], rgba[i + 2])
    };
    engine.apply_command(SimCommand::SetParticleFieldAt {
        points: vec![(20, 20)],
        field: tpt_core::commands::ParticleField::Life,
        value: 250,
    });
    let hot = {
        let rgba = engine.render_rgba();
        let i = (20 * xres + 20) * 4;
        (rgba[i], rgba[i + 1], rgba[i + 2])
    };
    assert!(
        hot.0 > 200 && hot.1 > 200 && hot.2 > 200,
        "a deuterium at life 250 is its base colour + 255 per channel, got {hot:?}"
    );
    assert_ne!(cold, hot, "the cold and hot deuterium must differ");
}

/// `(best fire_r cell, lava pixels)` per canvas half, split at `split` pixels.
///
/// The fire cells are the device-side bloom source (`render_fire_accum_*`), so
/// they isolate the emissive contribution of an element from the pixel colour
/// the base pass draws. Only cells that actually hold a `LAVA` pixel are
/// considered, because the sum of the neighbouring cells' bloom would otherwise
/// mask a missing emitter.
fn lava_fire_red(engine: &mut SimulationEngine, split: usize) -> [(u32, usize); 2] {
    let xres = engine.config().xres as usize;
    let yres = engine.config().yres as usize;
    let xcells = xres / 4;
    let ycells = yres / 4;
    engine
        .bufs
        .read_map_rect(0, 0, xres as i32 - 1, yres as i32 - 1);
    let readback = engine
        .bufs
        .client
        .read(vec![engine.bufs.gpu_fire_r.clone()])
        .expect("fire buffer readback");
    let red = |index: usize| -> u32 {
        let bytes: &[u8] = &readback[0];
        let base = index * 4;
        u32::from_le_bytes([
            bytes[base],
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
        ])
    };
    let mut halves = [(0u32, 0usize); 2];
    for cy in 0..ycells {
        for cx in 0..xcells {
            let mut lava = 0usize;
            for dy in 0..4 {
                for dx in 0..4 {
                    let w = engine.bufs.pmap[(cy * 4 + dy) * xres + cx * 4 + dx];
                    if w != 0 && w & 0x1FF == PT_LAVA {
                        lava += 1;
                    }
                }
            }
            if lava == 0 {
                continue;
            }
            let half = usize::from(cx * 4 >= split);
            halves[half].1 += lava;
            halves[half].0 = halves[half].0.max(red(cy * xcells + cx));
        }
    }
    halves
}

/// `NEUT + PLUT -> LAVA(PLUT)` must feed the fire bloom exactly like a LAVA
/// pixel painted by the brush.
///
/// Regression guard: the fission product records `ctype = PT_PLUT`, and the
/// pipe-cargo pixel-mode lookup used to run for *every* element, not just
/// `PIPE`/`PPIP`. LAVA therefore rendered with PLUT's `PMODE_FLAT | PMODE_GLOW`
/// row instead of its own `PMODE_FLAT | FIRE_ADD | PMODE_BLUR`: the base pass
/// still drew the pixel lava-orange (the colour rule dispatches on the element
/// type), but `render_fire_accum_matter_kernel` saw no `FIRE_ADD` bit and never
/// added the bloom, so fission lava did not glow.
#[test]
fn test_fission_lava_glows_like_painted_lava() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(96, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().air_mode = AirMode::NoUpdate;
    engine.config_mut().aheat_enable = false;
    engine.config_mut().display_mode = 0;
    engine.config_mut().colour_mode = COLOUR_DEFAULT;
    engine.config_mut().render_mode = RENDER_FANCY;

    // Brush-painted LAVA on the left.
    fill_rect(&mut engine, 18, 18, 23, 23, PT_LAVA);
    // PLUT block with one neutron per pixel. The hard vacuum makes the
    // `chance(3 + pv, 1000)` fission gate land on every pair (`NEUT.cpp:78`),
    // and the two-in-three branch turns the PLUT into `LAVA(PLUT)`.
    fill_rect(&mut engine, 62, 18, 67, 23, PT_PLUT);
    for y in 18..=23 {
        for x in 62..=67 {
            engine.bufs.create_part(
                x,
                y,
                PT_NEUT,
                tpt_sim::buffers::CreationSource::SkipChecks,
                0,
                &engine.registry,
            );
        }
    }
    for pv in engine.bufs.air_pv.iter_mut() {
        *pv = -50.0;
    }
    engine.bufs.mark_air_dirty();
    engine.bufs.sync_host_to_gpu();

    step(&mut engine);

    let fission_lava = (0..engine.bufs.capacity)
        .filter(|&id| engine.bufs.part_ty[id] == PT_LAVA && engine.bufs.part_x[id] > 48.0)
        .count();
    assert!(
        fission_lava >= 4,
        "the fission branch must leave LAVA(PLUT) behind ({fission_lava} found)"
    );

    // One presented frame resolves the fire accumulation and its glow layer.
    let _ = engine.render_rgba();

    let [(painted_red, painted_pixels), (fission_red, fission_pixels)] =
        lava_fire_red(&mut engine, 48);
    assert!(
        painted_pixels > 0 && painted_red > 0,
        "painted LAVA must feed the fire buffer (pixels {painted_pixels}, red {painted_red})"
    );
    assert!(fission_pixels > 0, "the fission half must hold LAVA pixels");
    assert!(
        fission_red > 0,
        "fission-created LAVA must feed the fire buffer like painted LAVA \
         (painted red {painted_red} over {painted_pixels} pixels, \
         fission red {fission_red} over {fission_pixels} pixels)"
    );
}

/// `Renderer::RenderSimulation` runs the gravity-lensing pass whenever
/// `DISPLAY_WARP` is set, even with Newtonian gravity off (`Renderer.cpp:42-47`);
/// the pass itself samples the same pixel back for a zero field
/// (`Renderer.cpp:60-80`).
///
/// Regression guards: the port used to gate the pass on `newton_grav`, so the
/// fancy preset rendered without lensing on the default (gravity-off) canvas.
/// The device force grids are `empty` allocations, so an unseeded grid would
/// also have bent the default view with allocator garbage.
#[test]
fn test_fancy_warp_gravity_lensing() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(64, 64);
    engine.config_mut().gravity_mode = GravityMode::Off;
    engine.config_mut().colour_mode = COLOUR_DEFAULT;
    engine.config_mut().render_mode = RENDER_BASC;
    engine.config_mut().display_mode = 0;
    assert!(!engine.config().newton_grav);

    // Block A touches the left edge (out-of-canvas samples); block B starts at
    // x = 20 so the background between the blocks pins the channel offsets.
    fill_rect(&mut engine, 0, 30, 15, 33, PT_SAND);
    fill_rect(&mut engine, 20, 30, 35, 33, PT_SAND);

    let xres = engine.config().xres as usize;
    let pixel = |rgba: &[u8], x: usize, y: usize| {
        let i = (y * xres + x) * 4;
        (rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3])
    };
    let black = (0u8, 0u8, 0u8, 0xFFu8);

    let baseline = engine.render_rgba().to_vec();
    let sand = pixel(&baseline, 30, 31);
    assert!(
        sand.0 > 0 || sand.1 > 0 || sand.2 > 0,
        "the sand block must be visible: {sand:?}"
    );
    assert_eq!(pixel(&baseline, 23, 31), sand);
    assert_eq!(pixel(&baseline, 18, 31), black);

    // The model is off and the force grids have never been solved, so the
    // lensing pass must be an exact identity, not a bend by stale memory.
    engine.config_mut().display_mode = DISPLAY_WARP;
    let zero_force = engine.render_rgba().to_vec();
    assert_eq!(
        zero_force, baseline,
        "a zero gravity field must leave the frame untouched"
    );

    // A non-zero field must bend the channels with Newtonian gravity still off:
    // red samples `force * 0.75` left, green `* 0.875`, blue `* 1.0`.
    // `forceX = 4` shifts red/green by 3 pixels and blue by 4.
    let cells = engine.bufs.config.num_cells();
    engine.bufs.client.write(
        &engine.bufs.gpu_grav_fx,
        Bytes::from_elems(vec![4.0f32; cells]),
    );
    let lensed = engine.render_rgba().to_vec();
    assert_ne!(lensed, baseline, "a force field must bend the frame");
    assert_eq!(
        pixel(&lensed, 0, 31),
        black,
        "a sample left of the canvas stays black (`Renderer.cpp:45,68-78`)"
    );
    assert_eq!(pixel(&lensed, 1, 31), black);
    assert_eq!(pixel(&lensed, 2, 31), black);
    assert_eq!(
        pixel(&lensed, 3, 31),
        sand,
        "all three samples land on x = 0, still inside block A"
    );
    assert_eq!(
        pixel(&lensed, 22, 31),
        black,
        "block B's left edge samples the background gap at x = 17..19"
    );
    assert_eq!(
        pixel(&lensed, 23, 31),
        (sand.0, sand.1, 0, 0xFF),
        "red/green sample block B at x = 20, blue samples the background"
    );
    assert_eq!(pixel(&lensed, 24, 31), sand);
    assert_eq!(pixel(&lensed, 30, 31), sand);

    // `Simulation::EnableNewtonianGravity(false)` resets `gravOut`
    // (`Simulation.cpp:3975-3983`): after a solve, disabling the model must
    // erase the field so the fancy view returns to the identity instead of
    // lensing with the previous solve's force.
    engine.bufs.grav_tool_mass.fill(0.0);
    let xcells = engine.bufs.config.xcells() as usize;
    engine.bufs.grav_tool_mass[4 * xcells + 4] = 50.0;
    engine.bufs.grav_tool_mass_dirty = true;
    engine.config_mut().newton_grav = true;
    step(&mut engine);
    assert!(
        engine.bufs.grav_force_stale,
        "the solve must leave a force field behind"
    );

    engine.config_mut().newton_grav = false;
    engine.config_mut().display_mode = 0;
    let settled = engine.render_rgba().to_vec();
    assert!(
        !engine.bufs.grav_force_stale,
        "the renderer must zero the force grid once the model is off"
    );
    assert!(
        settled
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[0] != 0 || px[1] != 0 || px[2] != 0)
            .count()
            > 64,
        "the sand blocks must survive the test frame"
    );
    engine.config_mut().display_mode = DISPLAY_WARP;
    let relensed = engine.render_rgba().to_vec();
    assert_eq!(
        relensed, settled,
        "a disabled gravity model must lens with the zeroed field"
    );
}
