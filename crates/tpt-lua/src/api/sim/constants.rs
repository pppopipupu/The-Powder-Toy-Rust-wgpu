//! Constants registered on the `sim` table.

use mlua::{Lua, Table};
use tpt_core::MAX_SIGNS;
use tpt_core::constants::{WL_NUM, *};
use tpt_core::walls::default_walls;

pub(crate) fn register_constants(lua: &Lua, sim: &Table) -> mlua::Result<()> {
    sim.set("CELL", CELL as i64)?;
    sim.set("XRES", DEFAULT_XRES as i64)?;
    sim.set("YRES", DEFAULT_YRES as i64)?;
    sim.set("XCELLS", (DEFAULT_XRES / CELL) as i64)?;
    sim.set("YCELLS", (DEFAULT_YRES / CELL) as i64)?;
    sim.set(
        "NCELL",
        ((DEFAULT_XRES / CELL) * (DEFAULT_YRES / CELL)) as i64,
    )?;
    sim.set("XCNTR", (DEFAULT_XRES / 2) as i64)?;
    sim.set("YCNTR", (DEFAULT_YRES / 2) as i64)?;
    sim.set("MAX_PARTS", 65536i64)?;

    sim.set("NT", NT as i64)?;
    sim.set("ST", ST as i64)?;
    sim.set("ITH", ITH as f64)?;
    sim.set("ITL", ITL as f64)?;
    sim.set("IPH", IPH as f64)?;
    sim.set("IPL", IPL as f64)?;
    sim.set("PT_NUM", PT_NUM as i64)?;
    sim.set("R_TEMP", R_TEMP as f64)?;
    sim.set("MAX_TEMP", MAX_TEMP as f64)?;
    sim.set("MIN_TEMP", MIN_TEMP as f64)?;
    sim.set("MAX_PRESSURE", MAX_PRESSURE as f64)?;
    sim.set("MIN_PRESSURE", MIN_PRESSURE as f64)?;
    sim.set("ISTP", 0i64)?;
    sim.set("CFDS", 4.0f64)?;
    sim.set("MAX_VELOCITY", 16.0f64)?;

    sim.set("DECO_DRAW", 0i64)?;
    sim.set("DECO_CLEAR", 1i64)?;
    sim.set("DECO_ADD", 2i64)?;
    sim.set("DECO_SUBTRACT", 3i64)?;
    sim.set("DECO_MULTIPLY", 4i64)?;
    sim.set("DECO_DIVIDE", 5i64)?;
    sim.set("DECO_SMUDGE", 6i64)?;

    sim.set("FLAG_STAGNANT", FLAG_STAGNANT as i64)?;
    sim.set("FLAG_SKIPMOVE", FLAG_SKIPMOVE as i64)?;
    sim.set("FLAG_MOVABLE", FLAG_MOVABLE as i64)?;
    sim.set("FLAG_PHOTDECO", 8i64)?;

    sim.set("PMAPBITS", 8i64)?;
    sim.set("PMAPMASK", 0xFFi64)?;

    sim.set("BRUSH_CIRCLE", 0i64)?;
    sim.set("BRUSH_SQUARE", 1i64)?;
    sim.set("BRUSH_TRIANGLE", 2i64)?;
    sim.set("NUM_DEFAULTBRUSHES", 3i64)?;
    sim.set("NUM_BRUSHES", 3i64)?;

    sim.set("EDGE_VOID", 0i64)?;
    sim.set("EDGE_SOLID", 1i64)?;
    sim.set("EDGE_LOOP", 2i64)?;
    sim.set("NUM_EDGEMODES", 3i64)?;

    sim.set("AIR_ON", 0i64)?;
    sim.set("AIR_PRESSUREOFF", 1i64)?;
    sim.set("AIR_VELOCITYOFF", 2i64)?;
    sim.set("AIR_OFF", 3i64)?;
    sim.set("AIR_NOUPDATE", 4i64)?;
    sim.set("NUM_AIRMODES", 5i64)?;

    sim.set("AIRC_NONE", 0i64)?;
    sim.set("AIRC_LEGACY", 1i64)?;
    sim.set("AIRC_BOUSSINESQ", 2i64)?;
    sim.set("NUM_CONVMODES", 3i64)?;

    sim.set("GRAV_VERTICAL", 0i64)?;
    sim.set("GRAV_OFF", 1i64)?;
    sim.set("GRAV_RADIAL", 2i64)?;
    sim.set("GRAV_CUSTOM", 3i64)?;
    sim.set("NUM_GRAVMODES", 4i64)?;

    sim.set("DECOSPACE_SRGB", 0i64)?;
    sim.set("DECOSPACE_LINEAR", 1i64)?;
    sim.set("DECOSPACE_GAMMA22", 2i64)?;
    sim.set("DECOSPACE_GAMMA18", 3i64)?;
    sim.set("NUM_DECOSPACES", 4i64)?;

    sim.set("TEMPSCALE_KELVIN", 0i64)?;
    sim.set("TEMPSCALE_CELSIUS", 1i64)?;
    sim.set("TEMPSCALE_FAHRENHEIT", 2i64)?;
    sim.set("NUM_TEMPSCALES", 3i64)?;

    sim.set("CANMOVE_BOUNCE", 0i64)?;
    sim.set("CANMOVE_SWAP", 1i64)?;
    sim.set("CANMOVE_ENTER", 2i64)?;
    sim.set("CANMOVE_BUILTIN", 3i64)?;
    sim.set("NUM_CANMOVEMODES", 4i64)?;

    sim.set("NUM_WALLS", WL_NUM as i64)?;

    // Walls bidirectional table
    let walls_table = lua.create_table()?;
    for wall in default_walls() {
        walls_table.set(wall.identifier.clone(), wall.id as i64)?;
        if let Some(short) = wall.identifier.strip_prefix("DEFAULT_WL_") {
            walls_table.set(short.to_lowercase(), wall.id as i64)?;
            walls_table.set(short, wall.id as i64)?;
        }
        walls_table.set(wall.id as i64, wall.identifier)?;
    }
    sim.set("walls", walls_table)?;

    // Sign justification modes
    sim.set("JUSTMODE_LEFT", 0i64)?;
    sim.set("JUSTMODE_MIDDLE", 1i64)?;
    sim.set("JUSTMODE_RIGHT", 2i64)?;
    sim.set("JUSTMODE_NONE", 3i64)?;
    sim.set("NUM_JUSTMODES", 4i64)?;
    sim.set("MAX_SIGNS", MAX_SIGNS as i64)?;

    Ok(())
}
