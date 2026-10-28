//! Integration tests for the Lua scripting layer.
//!
//! Every test builds a real [`SimulationEngine`] on a headless CubeCL/wgpu
//! device and drives it through [`LuaScriptingEngine`], so the tests exercise
//! the same bridge path the application uses: Lua callbacks read and write the
//! live simulation, and the host flushes the bridge after each call.

use tpt_core::constants::{
    BUILTIN_ELEMENT_COUNT, CELL, PT_DUST, PT_NONE, PT_NUM, PT_STNE, PT_WATR,
};
use tpt_core::particle::SimulationConfig;
use tpt_lua::LuaScriptingEngine;
use tpt_sim::SimulationEngine;

/// Canvas edge length used by every test, in pixels.
const CANVAS: u32 = 64;

/// Air/wall cells along one axis of a [`CANVAS`] canvas.
const CELLS: usize = (CANVAS / CELL) as usize;

/// Builds a fresh simulation on the default CubeCL device.
fn make_engine() -> SimulationEngine {
    SimulationEngine::new(
        cubecl::Device::default().client(),
        SimulationConfig::new(CANVAS, CANVAS),
    )
}

/// Builds a fresh Lua state with the TPT API installed.
fn make_script() -> LuaScriptingEngine {
    LuaScriptingEngine::new().expect("the Lua API installs")
}

/// Linear index of an air/wall cell.
fn cell_index(cx: usize, cy: usize) -> usize {
    cy * CELLS + cx
}

#[test]
fn print_and_tpt_log_capture_console_output() {
    let mut engine = make_engine();
    let mut script = make_script();

    let output = script
        .exec_script(
            r#"
            print("hello", 42, true)
            tpt.log("from tpt.log")
            print()
            "#,
            &mut engine,
        )
        .expect("the script runs");

    assert_eq!(output, "hello\t42\ttrue\nfrom tpt.log\n");
    assert_eq!(
        script.console_log,
        vec![
            String::from("hello\t42\ttrue"),
            String::from("from tpt.log"),
            String::new(),
        ]
    );

    // The log is cumulative across calls and `tpt.log` shares `print`'s buffer.
    script
        .exec_script("print('second call')", &mut engine)
        .expect("the second script runs");
    assert_eq!(script.console_log.len(), 4);
    assert_eq!(script.console_log[3], "second call");
}

#[test]
fn signs_table_adds_edits_and_deletes() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            assert(#sim.signs == 0)

            local first = sim.signs.new("hello", 10, 12)
            assert(first == 1)
            local second = sim.signs.new("world", 20, 30, sim.JUSTMODE_RIGHT)
            assert(second == 2)
            assert(#sim.signs == 2)

            assert(sim.signs[1].text == "hello")
            assert(sim.signs[1].x == 10 and sim.signs[1].y == 12)
            -- `Sign_new` defaults the justification to `Middle` (1).
            assert(sim.signs[1].ju == sim.JUSTMODE_MIDDLE)
            assert(sim.signs[2].ju == sim.JUSTMODE_RIGHT)

            -- Out-of-range indices read as nil.
            assert(sim.signs[3] == nil)

            sim.signs[1].text = "changed"
            sim.signs[1].x = 11
            assert(sim.signs[1].text == "changed")
            assert(sim.signs[1].x == 11)

            -- Coordinates outside the canvas are rejected.
            assert(not pcall(function() sim.signs.new("bad", -1, 5) end))
            assert(not pcall(function() sim.signs.new("bad", 5, 500) end))

            sim.signs.delete(1)
            assert(#sim.signs == 1)
            assert(sim.signs[1].text == "world")
            "#,
            &mut engine,
        )
        .expect("the sign script runs");

    assert_eq!(engine.signs().len(), 1);
    assert_eq!(engine.signs()[0].text, "world");
    assert_eq!(engine.signs()[0].x, 20);
}

#[test]
fn particle_create_count_id_position_and_kill() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            assert(sim.partCount() == 0)

            local id = sim.partCreate(10, 10, elem.DEFAULT_PT_DUST)
            assert(id ~= nil)
            assert(sim.partCount() == 1)
            assert(sim.partID(10, 10) == id)

            local x, y = sim.partPosition(id)
            assert(x == 10 and y == 10)

            -- A stale id has no position and no readable properties.
            local sx, sy = sim.partPosition(id + 1000)
            assert(sx == nil and sy == nil)
            assert(sim.partProperty(id + 1000, "type") == nil)

            -- Out-of-bounds positions and unknown element ids are rejected.
            assert(sim.partCreate(-1, 5, elem.DEFAULT_PT_DUST) == nil)
            assert(sim.partCreate(1000, 5, elem.DEFAULT_PT_DUST) == nil)
            assert(sim.partCreate(5, 5, 60000) == nil)
            assert(sim.partID(1000, 5) == nil)

            part_id = id

            sim.partKill(10, 10)
            assert(sim.partCount() == 0)
            assert(sim.partID(10, 10) == nil)
            assert(sim.partPosition(id) == nil)
            assert(sim.partProperty(id, "type") == nil)
            "#,
            &mut engine,
        )
        .expect("the particle lifecycle script runs");

    let part_id: u32 = script
        .lua
        .globals()
        .get("part_id")
        .expect("part_id global is set");
    assert_eq!(engine.bufs.part_ty[part_id as usize], PT_NONE);
    assert_eq!(engine.bufs.count_active_particles(), 0);
    assert_eq!(engine.particle_count(), 0);
    assert!(engine.bufs.particle_index_at(10, 10).is_none());
}

#[test]
fn particle_properties_round_trip() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            local id = sim.partCreate(20, 20, elem.DEFAULT_PT_DUST)
            assert(id ~= nil)
            part_id = id

            assert(sim.partProperty(id, "type") == elem.DEFAULT_PT_DUST)
            assert(sim.partProperty(id, "temp") > 0)
            assert(sim.partProperty(id, "life") >= 0)
            assert(sim.partProperty(id, "vx") == 0)
            assert(sim.partProperty(id, "vy") == 0)

            -- Every writable field can be written.
            sim.partProperty(id, "temp", 512.5)
            sim.partProperty(id, "life", 1234)
            sim.partProperty(id, "vx", 1.5)
            sim.partProperty(id, "vy", -2.25)
            sim.partProperty(id, "ctype", elem.DEFAULT_PT_WATR)
            sim.partProperty(id, "tmp3", 7)
            sim.partProperty(id, "tmp4", -9)
            -- Reads see the writes.
            assert(sim.partProperty(id, "temp") == 512.5)
            assert(sim.partProperty(id, "life") == 1234)
            assert(sim.partProperty(id, "vx") == 1.5)
            assert(sim.partProperty(id, "vy") == -2.25)
            assert(sim.partProperty(id, "ctype") == elem.DEFAULT_PT_WATR)
            assert(sim.partProperty(id, "tmp3") == 7)
            assert(sim.partProperty(id, "tmp4") == -9)
            "#,
            &mut engine,
        )
        .expect("the particle property script runs");

    let part_id: u32 = script
        .lua
        .globals()
        .get("part_id")
        .expect("part_id global is set");
    let id = part_id as usize;
    assert_eq!(engine.bufs.part_ty[id], PT_DUST);
    assert_eq!(engine.bufs.part_temp[id], 512.5);
    assert_eq!(engine.bufs.part_life[id], 1234);
    assert_eq!(engine.bufs.part_vx[id], 1.5);
    assert_eq!(engine.bufs.part_vy[id], -2.25);
    assert_eq!(engine.bufs.part_ctype[id], PT_WATR);
}

#[test]
fn air_pressure_velocity_and_ambient_heat() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            assert(sim.pressure(1, 2) == 0)
            sim.pressure(1, 2, 12.5)
            assert(sim.pressure(1, 2) == 12.5)
            -- Writes are clamped to the pressure range.
            sim.pressure(0, 0, 1e6)
            assert(sim.pressure(0, 0) == 256)
            sim.pressure(0, 0, -1e6)
            assert(sim.pressure(0, 0) == -256)

            local vx, vy = sim.velocity(3, 4)
            assert(vx == 0 and vy == 0)
            local wx, wy = sim.velocity(3, 4, 5, -6)
            assert(wx == 5 and wy == -6)
            local ax, ay = sim.velocity(3, 4)
            assert(ax == 5 and ay == -6)

            local heat = sim.ambientHeat(2, 2)
            assert(math.abs(heat - 295.15) < 0.01)
            sim.ambientHeat(2, 2, 400)
            assert(sim.ambientHeat(2, 2) == 400)
            -- Writes are clamped to the temperature range.
            sim.ambientHeat(0, 0, -50)
            assert(sim.ambientHeat(0, 0) == 0)
            sim.ambientHeat(0, 0, 1e6)
            assert(sim.ambientHeat(0, 0) == 9999)

            -- Cells outside the canvas throw out of range errors.
            assert(not pcall(sim.pressure, 999, 999))
            assert(not pcall(sim.pressure, 999, 999, 10))
            assert(not pcall(sim.velocity, 999, 999))
            assert(not pcall(sim.ambientHeat, 999, 999))
            "#,
            &mut engine,
        )
        .expect("the air script runs");

    assert_eq!(engine.bufs.air_pv[cell_index(1, 2)], 12.5);
    assert_eq!(engine.bufs.air_pv[cell_index(0, 0)], -256.0);
    assert_eq!(engine.bufs.air_vx[cell_index(3, 4)], 5.0);
    assert_eq!(engine.bufs.air_vy[cell_index(3, 4)], -6.0);
    assert_eq!(engine.bufs.air_hv[cell_index(2, 2)], 400.0);
    assert_eq!(engine.bufs.air_hv[cell_index(0, 0)], 9999.0);
}

#[test]
fn spawn_rigid_box_creates_body() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            local id = sim.spawnRigidBox(32, 32, 2, 2, elem.DEFAULT_PT_STNE)
            assert(id ~= nil)
            rigid_id = id

            -- Ids are handed out sequentially.
            local second = sim.spawnRigidBox(16, 16, 1, 1, elem.DEFAULT_PT_STNE)
            assert(second == id + 1)
            "#,
            &mut engine,
        )
        .expect("the rigid body script runs");

    let rigid_id: u32 = script
        .lua
        .globals()
        .get("rigid_id")
        .expect("rigid_id global is set");

    assert_eq!(engine.bufs.rigid_bodies.len(), 2);
    let body = &engine.bufs.rigid_bodies[0];
    assert_eq!(body.id, rigid_id);
    assert!(body.active);
    assert_eq!(body.pixels.len(), 25);
    assert!(body.pixels.iter().all(|pixel| pixel.elem_ty == PT_STNE));
    assert_eq!(engine.bufs.rigid_bodies[1].pixels.len(), 9);
}

#[test]
fn custom_element_allocation_and_properties() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            -- "AURUM", not "GOLD": GOLD is a built-in TPT element, and the
            -- lookup table is shared with the built-in namespace.
            local gold = elem.allocate("CUSTOM", "AURUM")
            assert(gold ~= nil)
            gold_id = gold

            -- The identifier is derived from the group and name.
            assert(elem.getByName("AURUM") == gold)
            assert(elem.getByName("CUSTOM_PT_AURUM") == gold)
            -- Allocating the same element twice is rejected.
            assert(elem.allocate("CUSTOM", "AURUM") == nil)

            assert(elem.property(gold, "name") == "AURUM")
            assert(elem.property(gold, "colour", 0x123456) == 0x123456)
            assert(elem.property(gold, "weight", 77) == 77)
            assert(elem.property(gold, "gravity", 0.5) == 0.5)
            assert(elem.property(gold, "flammable", 42) == 42)
            assert(elem.property(gold, "defaulttemp", 320) == 320)
            assert(elem.property(gold, "description", "A custom test element")
                == "A custom test element")

            -- The edited definition is what new particles spawn with.
            local part = sim.partCreate(6, 6, gold)
            assert(part ~= nil)
            part_id = part
            assert(sim.partProperty(part, "type") == gold)
            assert(sim.partProperty(part, "temp") == 320)
            "#,
            &mut engine,
        )
        .expect("the custom element script runs");

    let gold: u32 = script
        .lua
        .globals()
        .get("gold_id")
        .expect("gold_id global is set");
    let part_id: u32 = script
        .lua
        .globals()
        .get("part_id")
        .expect("part_id global is set");

    // Custom elements are allocated from the reserved slots below `PT_NUM`.
    assert!((gold as usize) >= BUILTIN_ELEMENT_COUNT);
    assert!((gold as usize) < PT_NUM);

    let definition = engine.registry.get(gold);
    assert!(definition.enabled);
    assert_eq!(definition.identifier, "CUSTOM_PT_AURUM");
    assert_eq!(definition.name, "AURUM");
    assert_eq!(definition.description, "A custom test element");
    assert_eq!(definition.colour, 0x123456);
    assert_eq!(definition.weight, 77);
    assert_eq!(definition.gravity, 0.5);
    assert_eq!(definition.flammable, 42);
    assert_eq!(definition.default_temp, 320.0);
    assert_eq!(engine.registry.find_by_name("AURUM"), Some(gold));

    assert_eq!(engine.bufs.part_ty[part_id as usize], gold);
    assert_eq!(engine.bufs.part_temp[part_id as usize], 320.0);
}

#[test]
fn tick_handlers_run_across_fire_tick() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            local part = sim.partCreate(12, 12, elem.DEFAULT_PT_DUST)
            assert(part ~= nil)
            part_id = part

            tick_count = 0
            step_count = 0

            evt.register(evt.TICK, function()
                tick_count = tick_count + 1
                sim.partProperty(part, "life", 42)
                print("tick", tick_count)
            end)

            tpt.register_step(function()
                step_count = step_count + 1
            end)
            "#,
            &mut engine,
        )
        .expect("the callback registration script runs");

    // Both callbacks were moved into the engine's tick handler list by the
    // bridge flush, and registering them produced no console output.
    assert_eq!(script.tick_handlers.len(), 2);
    assert!(script.console_log.is_empty());

    script.fire_tick(&mut engine);
    script.fire_tick(&mut engine);

    let tick_count: i64 = script
        .lua
        .globals()
        .get("tick_count")
        .expect("tick_count global is set");
    let step_count: i64 = script
        .lua
        .globals()
        .get("step_count")
        .expect("step_count global is set");
    assert_eq!(tick_count, 2);
    assert_eq!(step_count, 2);

    assert_eq!(
        script.console_log,
        vec![String::from("tick\t1"), String::from("tick\t2")]
    );

    let part_id: u32 = script
        .lua
        .globals()
        .get("part_id")
        .expect("part_id global is set");
    assert_eq!(engine.bufs.part_life[part_id as usize], 42);

    assert_eq!(script.tick_handlers.len(), 2);
}

#[test]
fn creation_functions_create_box_and_walls() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            assert(sim.partCount() == 0)
            -- Create a 5x5 box of dust
            sim.createBox(10, 10, 14, 14, elem.DEFAULT_PT_DUST)
            local parts_after = sim.partCount()
            assert(parts_after == 25, "expected 25 particles created, got " .. tostring(parts_after))

            -- Create walls
            local wall_cell = sim.walls.wall
            sim.createWallBox(0, 0, 8, 8, wall_cell)
            assert(sim.wallMap(1, 1) == wall_cell)
            "#,
            &mut engine,
        )
        .expect("the creation function script runs");
}

#[test]
fn grid_getter_setter_round_trip() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            -- Pressure grid round-trip
            sim.pressure(4, 4, 15.5)
            local p = sim.pressure(4, 4)
            assert(math.abs(p - 15.5) < 0.1, "pressure round-trip")

            -- Ambient heat round-trip
            sim.ambientHeat(4, 4, 350.0)
            local h = sim.ambientHeat(4, 4)
            assert(math.abs(h - 350.0) < 0.1, "ambient heat round-trip")

            -- Wall map round-trip
            sim.wallMap(2, 2, 3)
            assert(sim.wallMap(2, 2) == 3, "wallMap round-trip")

            -- Velocity X / Y round-trip
            sim.velocityX(4, 4, 2.5)
            sim.velocityY(4, 4, -1.5)
            assert(math.abs(sim.velocityX(4, 4) - 2.5) < 0.1, "velocityX round-trip")
            assert(math.abs(sim.velocityY(4, 4) - (-1.5)) < 0.1, "velocityY round-trip")
            "#,
            &mut engine,
        )
        .expect("the grid round-trip script runs");
}

#[test]
fn brush_function_and_tools() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            -- Test brush iterator
            local count = 0
            for x, y in sim.brush(10, 10, 2, 2, 1) do
                count = count + 1
                assert(x >= 8 and x <= 12)
                assert(y >= 8 and y <= 12)
            end
            assert(count == 25, "5x5 square brush produces 25 points")

            -- Test decoBrush tool
            local id = sim.partCreate(15, 15, elem.DEFAULT_PT_DUST)
            assert(id ~= nil)
            sim.decoBrush(15, 15, 1, 1, 255, 128, 64, 255, sim.DECO_DRAW)
            local dcol = sim.partProperty(id, "dcolour")
            assert(dcol ~= nil and dcol ~= 0, "decoBrush painted particle")
            "#,
            &mut engine,
        )
        .expect("the brush and tools script runs");
}

#[test]
fn save_and_stamp_round_trip() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            -- Create particle to stamp
            local id = sim.partCreate(10, 10, elem.DEFAULT_PT_DUST)
            assert(id ~= nil)
            assert(sim.partCount() == 1)

            -- Save stamp
            local stamp_name = sim.saveStamp(0, 0, 20, 20)
            assert(stamp_name ~= nil, "stamp name created")

            local stamps = sim.listStamps()
            assert(#stamps >= 1, "at least one stamp listed")

            -- Clear simulation
            sim.clearSim()
            assert(sim.partCount() == 0)

            -- Load stamp back
            local ok, err = sim.loadStamp(stamp_name, 0, 0)
            assert(ok == 1, "stamp load succeeded: " .. tostring(err))
            assert(sim.partCount() == 1, "stamp restored particle")

            -- Delete stamp
            local deleted = sim.deleteStamp(stamp_name)
            assert(deleted == 0, "stamp deleted successfully")
            "#,
            &mut engine,
        )
        .expect("the save/stamp script runs");
}

#[test]
fn ren_toggles_getter_setter_shape() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            -- Defaults
            assert(ren.decorations() == true)
            assert(ren.grid() == false)
            assert(ren.gridCheckerboard() == false)
            assert(ren.hud() == true)
            assert(ren.debugHud() == false)
            assert(ren.showBrush() == true)
            assert(ren.zoomEnabled() == false)

            -- Test getters/setters on ren
            assert(ren.decorations(false) == nil)
            assert(ren.decorations() == false)
            ren.decorations(true)
            assert(ren.decorations() == true)

            assert(ren.grid(true) == nil)
            assert(ren.grid() == true)
            ren.grid(false)
            assert(ren.grid() == false)

            assert(ren.gridCheckerboard(true) == nil)
            assert(ren.gridCheckerboard() == true)
            ren.gridCheckerboard(false)
            assert(ren.gridCheckerboard() == false)

            assert(ren.hud(false) == nil)
            assert(ren.hud() == false)
            ren.hud(true)
            assert(ren.hud() == true)

            assert(ren.debugHud(true) == nil)
            assert(ren.debugHud() == true)
            ren.debugHud(false)
            assert(ren.debugHud() == false)

            assert(ren.showBrush(false) == nil)
            assert(ren.showBrush() == false)
            ren.showBrush(true)
            assert(ren.showBrush() == true)

            assert(ren.zoomEnabled(true) == nil)
            assert(ren.zoomEnabled() == true)
            ren.zoomEnabled(false)
            assert(ren.zoomEnabled() == false)

            -- Zoom window & scope
            -- Canvas size in make_engine is CANVAS = 64
            -- Set zoomWindow first with factor 2, so s=16 gives window size 32 <= 64
            assert(ren.zoomWindow(10, 10, 1) == nil)
            assert(ren.zoomScope(5, 6, 16) == nil)
            assert(ren.zoomWindow(10, 10, 2) == nil)
            local sx, sy, ss = ren.zoomScope()
            assert(sx == 5 and sy == 6 and ss == 16, "zoomScope values match")
            local wx, wy, wf, ws = ren.zoomWindow()
            assert(wx == 10 and wy == 10 and wf == 2 and ws == 32, "zoomWindow values match")

            -- Heat display limits & auto area
            assert(ren.heatDisplayLimits(100.5, 500.0) == nil)
            local hmin, hmax = ren.heatDisplayLimits()
            assert(math.abs(hmin - 100.5) < 1e-4 and math.abs(hmax - 500.0) < 1e-4)

            assert(ren.heatDisplayLimits("auto", "auto") == nil)
            local hmin2, hmax2 = ren.heatDisplayLimits()
            assert(hmin2 == "auto" and hmax2 == "auto")

            assert(ren.heatDisplayAutoArea(2, 3, 20, 25) == nil)
            local ax, ay, aw, ah = ren.heatDisplayAutoArea()
            assert(ax == 2 and ay == 3 and aw == 20 and ah == 25)

            -- renderer alias check
            assert(renderer.grid(true) == nil)
            assert(renderer.grid() == true)
            "#,
            &mut engine,
        )
        .expect("the ren toggles script runs");
}

#[test]
fn ren_separate_thread_and_depth3d_store_values() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            -- `GameModel::threadedRendering` defaults to the
            -- `Renderer.SeparateThread` preference (true) and depth3d is off.
            assert(ren.separateThread() == true)
            assert(ren.depth3d() == false)

            assert(ren.separateThread(false) == nil)
            assert(ren.separateThread() == false)
            assert(ren.depth3d(true) == nil)
            assert(ren.depth3d() == true)

            -- The classic alias exposes the same state.
            assert(renderer.separateThread() == false)
            assert(renderer.depth3d() == true)
            "#,
            &mut engine,
        )
        .expect("the ren state script runs");

    assert!(!script.render_lua_state().separate_thread());
    assert!(script.render_lua_state().depth3d());
}

#[test]
fn bit_library_round_trips() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            assert(bit.tobit(0xFFFFFFFF) == -1)
            assert(bit.tobit(2^32 + 5) == 5)
            assert(bit.bnot(0) == -1)
            assert(bit.band(0xF0, 0x3C) == 0x30)
            assert(bit.bor(1, 2, 4) == 7)
            assert(bit.bxor(0xF0, 0xFF) == 0x0F)
            assert(bit.lshift(1, 31) == -2147483648)
            assert(bit.lshift(1, 32) == 1)          -- count masked to 5 bits
            assert(bit.rshift(-1, 1) == 0x7FFFFFFF)
            assert(bit.arshift(-8, 2) == -2)
            assert(bit.rol(0x12345678, 8) == 0x34567812)
            assert(bit.ror(0x12345678, 8) == 0x78123456)
            assert(bit.bswap(0x12345678) == 0x78563412)
            assert(bit.tohex(0x1234) == "00001234")
            assert(bit.tohex(0x1234, 2) == "34")
            assert(bit.tohex(0xABCD, 4) == "abcd")
            assert(bit.tohex(0xABCD, -4) == "ABCD")
            assert(bit.tohex(0x123456789) == "23456789") -- only 32 bits survive

            -- Argument conversion: numeric strings work, a missing argument
            -- raises like `luaL_checknumber`.
            assert(bit.band("0xF0", "60") == 0x30)
            local ok, err = pcall(bit.band)
            assert(not ok and string.find(tostring(err), "number expected") ~= nil)
            "#,
            &mut engine,
        )
        .expect("the bit library script runs");
}

#[test]
fn bz2_compress_and_decompress_round_trip() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            local compressed = bz2.compress("hello hello hello")
            assert(type(compressed) == "string" and #compressed > 0)
            assert(bz2.decompress(compressed) == "hello hello hello")

            local value, code, message = bz2.decompress("not a bzip2 stream")
            assert(value == nil)
            assert(code == bz2.DECOMPRESS_TYPE)
            assert(message == "corrupted stream")

            local limited, limit_code, limit_message = bz2.compress("hello hello hello", 2)
            assert(limited == nil)
            assert(limit_code == bz2.COMPRESS_LIMIT)
            assert(limit_message == "size limit exceeded")

            local short, short_code = bz2.decompress(compressed, 4)
            assert(short == nil and short_code == bz2.DECOMPRESS_LIMIT)
            "#,
            &mut engine,
        )
        .expect("the bz2 script runs");
}

#[test]
fn tools_allocate_and_property() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            local index = tools.allocate("mygroup", "mytool")
            assert(type(index) == "number")
            assert(tools.exists(index))
            assert(tools.isCustom(index))
            assert(tools.index.MYGROUP_TOOL_MYTOOL == index)
            assert(tools.property(index, "Identifier") == "MYGROUP_TOOL_MYTOOL")
            assert(tools.property(index, "Name") == "NULL")
            assert(tools.property(index, "Description") == "NULL Tool, does NOTHING")
            assert(tools.property(index, "Colour") == 0xFFFFFF)
            assert(tools.property(index, "MenuVisible") == true)
            assert(tools.property(index, "MenuSort") == 0)

            tools.property(index, "Name", "My Tool")
            tools.property(index, "Description", "Does something.")
            tools.property(index, "Colour", 0x123456)
            tools.property(index, "Color", 0x654321)
            tools.property(index, "MenuSection", 3)
            tools.property(index, "MenuVisible", false)
            tools.property(index, "MenuSort", 7)
            assert(tools.property(index, "Name") == "My Tool")
            assert(tools.property(index, "Description") == "Does something.")
            assert(tools.property(index, "Colour") == 0x654321)
            assert(tools.property(index, "MenuSection") == 3)
            assert(tools.property(index, "MenuVisible") == false)
            assert(tools.property(index, "MenuSort") == 7)

            performed = false
            tools.property(index, "Perform", function() performed = true end)
            tools.property(index, "Perform", false)   -- clears the callback
            tools.property(index, "Perform", function() performed = true end)
            tool_index = index

            -- Built-in tools exist but cannot be changed or freed.
            assert(tools.exists(0))
            assert(not tools.isCustom(0))
            local ok = pcall(function() tools.property(0, "Name", "X") end)
            assert(not ok)
            local free_ok = pcall(function() tools.free(0) end)
            assert(not free_ok)

            -- Validation mirrors `LuaTools.cpp:9-66`.
            local bad1 = pcall(function() tools.allocate("DEFAULT", "X") end)
            assert(not bad1)
            local bad2 = pcall(function() tools.allocate("G", "A_B") end)
            assert(not bad2)
            local bad3 = pcall(function() tools.allocate("G_X", "A") end)
            assert(not bad3)
            local bad4 = pcall(function() tools.allocate("MYGROUP", "MYTOOL") end)
            assert(not bad4)
            local bad5 = pcall(function() tools.property(index, "Nope") end)
            assert(not bad5)

            -- Freeing clears the stored callbacks (`LuaTools.cpp:48-66`), so the
            -- host entry point below must target a tool that still exists. A
            -- second custom tool covers the free path.
            local freed = tools.allocate("FREED", "TOOL")
            tools.free(freed)
            assert(not tools.exists(freed))
            assert(tools.index.FREED_TOOL_TOOL == nil)
            "#,
            &mut engine,
        )
        .expect("the tools script runs");

    // `tools.property` stores the Lua callback; the host entry point fires it.
    let index: i64 = script
        .lua
        .globals()
        .get("tool_index")
        .expect("the tool index global");
    tpt_lua::api::tools::call_tool_callback(
        &script.lua,
        index as usize,
        "Perform",
        mlua::MultiValue::new(),
    )
    .expect("the perform callback runs");
    let performed: bool = script
        .lua
        .globals()
        .get("performed")
        .expect("the performed global");
    assert!(performed);
}

#[test]
fn graphics_records_overlay_state() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            assert(gfx == graphics)
            assert(graphics.WIDTH == 629)
            assert(graphics.HEIGHT == 424)

            local w, h = graphics.textSize("abc")
            assert(w == 21 and h == 10)
            local w2, h2 = graphics.textSize("ab\ncd")
            assert(w2 == 14 and h2 == 22)
            local w3, h3 = graphics.textSize("")
            assert(w3 == 0 and h3 == 10)

            graphics.drawPixel(1, 2, 3, 4, 5, 6)
            graphics.drawLine(1, 2, 3, 4, 255, 0, 0)
            graphics.drawRect(1, 2, 3, 4)
            graphics.fillRect(5, 6, 7, 8, 10, 20, 30, 40)
            graphics.drawCircle(9, 10, -3, -4)
            graphics.fillCircle(11, 12, 5, 6)
            graphics.drawText(13, 14, "hi", 1, 2, 3, 4)

            local r, g, b, a = graphics.getColors(0xFF112233)
            assert(r == 0x11 and g == 0x22 and b == 0x33 and a == 0xFF)
            assert(graphics.getHexColor(0x11, 0x22, 0x33, 0x44) == 0x44112233)
            assert(graphics.getHexColor(0x11, 0x22, 0x33) == 0x112233)

            local cx, cy, cw, ch = graphics.setClipRect(1, 2, 3, 4)
            assert(cx == 0 and cy == 0 and cw == 629 and ch == 424)
            local ox, oy, ow, oh = graphics.setClipRect()
            assert(ox == 1 and oy == 2 and ow == 3 and oh == 4)
            "#,
            &mut engine,
        )
        .expect("the graphics script runs");

    let graphics = script.graphics();
    let overlay = graphics.borrow();
    assert_eq!(overlay.commands.len(), 7);
    assert_eq!(
        overlay.commands[0],
        tpt_lua::api::graphics::OverlayCommand::Pixel {
            x: 1,
            y: 2,
            colour: [3, 4, 5, 6]
        }
    );
    assert_eq!(
        overlay.commands[2],
        tpt_lua::api::graphics::OverlayCommand::Rect {
            x: 1,
            y: 2,
            w: 3,
            h: 4,
            colour: [255, 255, 255, 255],
            filled: false
        }
    );
    assert_eq!(
        overlay.commands[3],
        tpt_lua::api::graphics::OverlayCommand::Rect {
            x: 5,
            y: 6,
            w: 7,
            h: 8,
            colour: [10, 20, 30, 40],
            filled: true
        }
    );
    assert_eq!(
        overlay.commands[4],
        tpt_lua::api::graphics::OverlayCommand::Ellipse {
            x: 9,
            y: 10,
            rx: 3,
            ry: 4,
            colour: [255, 255, 255, 255],
            filled: false
        }
    );
    assert_eq!(
        overlay.commands[5],
        tpt_lua::api::graphics::OverlayCommand::Ellipse {
            x: 11,
            y: 12,
            rx: 5,
            ry: 6,
            colour: [255, 255, 255, 255],
            filled: true
        }
    );
    assert_eq!(
        overlay.commands[6],
        tpt_lua::api::graphics::OverlayCommand::Text {
            x: 13,
            y: 14,
            text: String::from("hi"),
            colour: [1, 2, 3, 4]
        }
    );
    assert_eq!(
        overlay.clip_rect,
        tpt_lua::api::graphics::OverlayRect {
            x: 0,
            y: 0,
            w: 629,
            h: 424
        }
    );
    drop(overlay);
    graphics.borrow_mut().clear();
    assert!(graphics.borrow().commands.is_empty());
}

#[test]
fn interface_window_components_and_settings() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            local w = interface.window(10, 20, 100, 80)
            local wx, wy = w:position()
            assert(wx == 10 and wy == 20)
            w:position(30, 40)
            wx, wy = w:position()
            assert(wx == 30 and wy == 40)
            w:size(120, 90)
            local ww, wh = w:size()
            assert(ww == 120 and wh == 90)
            local bad = pcall(function() w:position(0, 5) end)
            assert(not bad)
            local bad_size = pcall(function() w:size(5, 90) end)
            assert(not bad_size)

            local b = interface.button(1, 2, 30, 40, "OK", "tip")
            assert(b:text() == "OK")
            b:text("Go")
            assert(b:text() == "Go")
            assert(b:visible() == true)
            b:visible(false)
            assert(b:visible() == false)
            assert(b:enabled() == true)
            b:enabled(false)
            assert(b:enabled() == false)
            b:action(function(self) end)

            local l = interface.label(0, 0, 50, 12, "hello")
            assert(l:text() == "hello")
            l:text("bye")
            assert(l:text() == "bye")

            local tb = interface.textbox(0, 0, 60, 12, "text", "placeholder")
            assert(tb:text() == "text")
            tb:text("changed")
            assert(tb:text() == "changed")
            assert(tb:readonly() == false)
            tb:readonly(true)
            assert(tb:readonly() == true)
            tb:onTextChanged(function(self) end)

            local cb = interface.checkbox(0, 0, 20, 20, "check")
            assert(cb:text() == "check")
            assert(cb:checked() == false)
            cb:checked(false)
            assert(cb:checked() == false)
            cb:action(function(self) end)

            local sl = interface.slider(0, 0, 50, 12, 20)
            assert(sl:steps() == 20)
            sl:steps(5)
            assert(sl:steps() == 5)
            assert(sl:value() == 0)
            sl:value(3)
            assert(sl:value() == 3)
            sl:onValueChanged(function(self) end)

            local pb = interface.progressBar(0, 0, 50, 12, 42, "loading")
            assert(pb:progress() == 42)
            assert(pb:status() == "loading")
            pb:progress(80)
            pb:status("done")
            assert(pb:progress() == 80)
            assert(pb:status() == "done")

            w:addComponent(b)
            w:addComponent(l)
            w:addComponent(tb)
            -- Focus is a window property: `LuaTextbox::focus` consults
            -- `GetParentWindow()->IsFocused` (`LuaTextbox.cpp:52-69`).
            assert(tb:focus() == false)
            tb:focus(true)
            assert(tb:focus() == true)
            tb:focus(false)
            assert(tb:focus() == false)
            w:addComponent(cb)
            w:addComponent(sl)
            w:addComponent(pb)
            interface.showWindow(w)
            w:removeComponent(b)
            interface.closeWindow(w)

            interface.grabTextInput()
            interface.grabTextInput()
            interface.dropTextInput()
            interface.textInputRect(1, 2, 3, 4)

            interface.beginMessageBox("T", "M", true)
            interface.beginThrowError("boom")
            interface.beginConfirm("T", "M", "OK")
            interface.beginInput("T", "P", "x", "s")

            assert(interface.numMenus() == 16)
            assert(interface.numMenus(false) == 16)
            assert(interface.activeMenu() == 8)
            interface.activeMenu(3)
            assert(interface.activeMenu() == 3)
            assert(interface.menuEnabled(3) == true)
            interface.menuEnabled(3, false)
            assert(interface.menuEnabled(3) == false)
            assert(interface.numMenus() == 15)
            local bad_menu = pcall(function() interface.activeMenu(99) end)
            assert(not bad_menu)

            assert(interface.console() == false)
            interface.console(true)
            assert(interface.console() == true)
            local scale, fullscreen = interface.windowSize()
            assert(scale == 1 and fullscreen == false)
            interface.windowSize(3, true)
            scale, fullscreen = interface.windowSize()
            assert(scale == 3 and fullscreen == true)
            interface.windowSize(99, false)
            scale = interface.windowSize()
            assert(scale == 1)

            assert(interface.brushID() == 0)
            interface.brushID(2)
            assert(interface.brushID() == 2)
            local bad_brush = pcall(function() interface.brushID(99) end)
            assert(not bad_brush)
            local rx, ry = interface.brushRadius()
            assert(rx == 4 and ry == 4)
            interface.brushRadius(7, 8)
            rx, ry = interface.brushRadius()
            assert(rx == 7 and ry == 8)

            assert(interface.perfectCircleBrush() == false)
            interface.perfectCircleBrush(true)
            assert(interface.perfectCircleBrush() == true)

            local mx, my = interface.mousePosition()
            assert(mx == 0 and my == 0)

            assert(interface.MOUSEUP_NORMAL == 0)
            assert(interface.MOUSEUP_BLUR == 1)
            assert(interface.MOUSEUP_DRAWEND == 2)
            assert(interface.NUM_TOOLINDICES == 4)
            assert(ui == interface)

            assert(type(interface.activeTool(0)) == "string")
            interface.activeTool(0, "DEFAULT_TOOL_HEAT")
            assert(interface.activeTool(0) == "DEFAULT_TOOL_HEAT")
            local bad_tool = pcall(function() interface.activeTool(0, "NOPE") end)
            assert(not bad_tool)
            "#,
            &mut engine,
        )
        .expect("the interface script runs");

    let interface = script.interface();
    let state = interface.borrow();
    assert_eq!(state.windows.len(), 1);
    assert_eq!(state.components.len(), 6);
    assert_eq!(state.shown_window, None);
    assert_eq!(state.text_input_refcount, 1);
    assert_eq!(state.text_input_rect, Some((1, 2, 3, 4)));
    assert!(state.message_box.is_some());
    assert!(state.error_message.is_some());
    assert!(state.confirm.is_some());
    assert!(state.input_prompt.is_some());
    assert_eq!(state.active_menu, 3);
    assert!(!state.menus[3].enabled);
    assert_eq!(state.brush_id, 2);
    assert_eq!(state.brush_radius, (7, 8));
    assert_eq!(state.window_scale, 1);
    assert!(state.perfect_circle);
    let window = &state.windows[0];
    assert_eq!(window.position, (30, 40));
    assert_eq!(window.size, (120, 90));
    assert_eq!(window.components.len(), 5);
    assert!(!window.visible);
    drop(state);
    drop(interface);

    let tools = script.tools();
    assert_eq!(
        tools.borrow().active[0].as_deref(),
        Some("DEFAULT_TOOL_HEAT")
    );
}

#[test]
fn interface_dialog_callbacks_run_on_dismiss() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            box = "unset"
            confirmed = "unset"
            typed = "unset"
            errored = "unset"
            interface.beginMessageBox("T", "M", false, function() box = "closed" end)
            interface.beginConfirm("T", "M", "OK", function(result) confirmed = result end)
            interface.beginInput("T", "P", "x", "s", function(value) typed = value end)
            interface.beginThrowError("boom", function() errored = "closed" end)
            "#,
            &mut engine,
        )
        .expect("the dialog script runs");

    let interface = script.interface();
    interface
        .dismiss_message_box(&script.lua)
        .expect("the message box callback runs");
    interface
        .dismiss_confirm(&script.lua, true)
        .expect("the confirm callback runs");
    interface
        .dismiss_input(&script.lua, Some("typed text"))
        .expect("the input callback runs");
    interface
        .dismiss_error_message(&script.lua)
        .expect("the error callback runs");

    let globals = script.lua.globals();
    assert_eq!(globals.get::<String>("box").unwrap(), "closed");
    assert!(globals.get::<bool>("confirmed").unwrap());
    assert_eq!(globals.get::<String>("typed").unwrap(), "typed text");
    assert_eq!(globals.get::<String>("errored").unwrap(), "closed");

    let state = interface.borrow();
    assert!(state.message_box.is_none());
    assert!(state.confirm.is_none());
    assert!(state.input_prompt.is_none());
    assert!(state.error_message.is_none());
}

#[test]
fn platform_clipboard_restart_and_tpt_helpers() {
    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            r#"
            assert(plat == platform)
            platform.clipboardPaste("tpt-lua clipboard round trip")
            assert(platform.clipboardCopy() == "tpt-lua clipboard round trip")

            assert(platform.restart() == nil)
            assert(__tpt_restart_request == true)

            -- `openLink` launches the desktop handler; the port still checks
            -- its argument like `tpt_lua_checkByteString`.
            local ok = pcall(function() platform.openLink() end)
            assert(not ok)

            assert(type(tpt.getUserName()) == "string")
            assert(tpt.debug() == 0)
            assert(tpt.debug(9) == nil)
            assert(tpt.debug() == 9)
            "#,
            &mut engine,
        )
        .expect("the platform script runs");

    assert_eq!(script.debug_flags().get(), 9);
}

#[test]
fn socket_udp_send_and_receive() {
    // A bound peer makes the receive path deterministic: no ICMP unreachable,
    // so an empty queue reports the reference's "timeout".
    let peer = std::net::UdpSocket::bind("127.0.0.1:0").expect("the peer socket binds");
    peer.set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .expect("the peer timeout is set");
    let port = peer.local_addr().expect("the peer address").port();

    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            &format!(
                r#"
                assert(type(socket.getTime()) == "number")
                socket.sleep(0)

                sock = socket.tcp()
                assert(sock:status() == "ready")
                assert(sock:connect("127.0.0.1", {port}, false) == 1)
                assert(sock:status() == "connected")
                sock:settimeout(0.01)
                local data, reason = sock:receive()
                assert(data == nil and reason == "timeout")

                assert(sock:send("ping") == 4)
                "#,
            ),
            &mut engine,
        )
        .expect("the socket script runs");

    let mut buffer = [0u8; 64];
    let (received, client) = peer.recv_from(&mut buffer).expect("the peer receives");
    assert_eq!(&buffer[..received], b"ping");
    peer.send_to(b"pong", client).expect("the peer replies");

    script
        .exec_script(
            r#"
            sock:settimeout(5)
            assert(sock:receive("*a") == "pong")
            sock:close()
            assert(sock:status() == "dead")
            "#,
            &mut engine,
        )
        .expect("the socket reply script runs");
}

#[test]
fn http_get_against_a_local_server_and_unreachable_host() {
    use std::io::{Read, Write};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("the test server binds");
    let port = listener.local_addr().expect("the test address").port();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("the test server accepts");
        let mut request = [0u8; 1024];
        let _ = stream.read(&mut request);
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nX-Test: yes\r\nConnection: close\r\n\r\nhello",
            )
            .expect("the test server replies");
    });

    let mut engine = make_engine();
    let mut script = make_script();

    script
        .exec_script(
            &format!(
                r#"
                local handle = http.get("http://127.0.0.1:{port}/test")
                assert(handle:status() == "done")
                local total, done = handle:progress()
                assert(total == 5 and done == 5)
                local body, code, headers = handle:finish()
                assert(body == "hello")
                assert(code == 200)
                assert(headers[1][1] == "Content-Length" and headers[1][2] == "5")
                assert(headers[2][1] == "X-Test" and headers[2][2] == "yes")
                assert(handle:status() == "dead")
                assert(handle:finish() == nil)

                local dead = http.get("http://127.0.0.1:1/")
                assert(dead:status() == "done")
                local ok, err = pcall(function() dead:finish() end)
                assert(not ok)
                assert(string.find(tostring(err), "Couldn't connect to server", 1, true) ~= nil)

                local tls = http.get("https://example.com/")
                local tls_ok, tls_err = pcall(function() tls:finish() end)
                assert(not tls_ok)
                assert(string.find(tostring(tls_err), "Couldn't connect to server", 1, true) ~= nil)

                local token, token_error = http.getAuthToken("test")
                assert(token == nil and token_error == "not authenticated")
                "#,
            ),
            &mut engine,
        )
        .expect("the http script runs");

    server.join().expect("the test server finishes");
}
