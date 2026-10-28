//! The `sim` table (and the classic `simulation` alias).

mod constants;
mod geom;
mod gol;
mod grid;
mod lifecycle;
mod properties;
mod saves;
mod signs;
mod state;
mod tools;

use mlua::{Lua, Table};
use tpt_core::constants::PT_NONE;
use tpt_sim::SimulationEngine;
use tpt_sim::buffers::CreationSource;

use crate::bridge::Bridge;

/// Makes the host pixel map, and the slot occupying `(x, y)`, current.
///
/// Creation and coordinate-addressed kills inspect the occupancy of a single
/// pixel before acting. Without the blanket read-back that `run_attached` used
/// to perform, only that pixel and its occupant are pulled from the device;
/// the rest of the pool stays untouched.
pub(crate) fn refresh_pixel_occupant(engine: &mut SimulationEngine, x: i32, y: i32) {
    crate::api::refresh_map_rect(&mut engine.bufs, x, y, x, y);
    if let Some(slot) = engine.bufs.particle_index_at(x, y) {
        engine.bufs.read_particle_slot(slot);
    }
}

/// Allocates a particle, validating the element id and canvas bounds first.
pub(crate) fn allocate_particle(
    engine: &mut SimulationEngine,
    x: i32,
    y: i32,
    ty: u32,
) -> Option<usize> {
    let xres = engine.bufs.config.xres as i32;
    let yres = engine.bufs.config.yres as i32;
    if x < 0 || y < 0 || x >= xres || y >= yres {
        return None;
    }
    if ty == PT_NONE || ty as usize >= engine.registry.elements.len() {
        return None;
    }
    // The occupancy check and the map-word update need the destination pixel
    // current; only that pixel (and its occupant) is read back.
    refresh_pixel_occupant(engine, x, y);
    // `sim.partCreate` is `CreateParts(-1, ...)` (`LuaSimulation.cpp:518`), so
    // the element's `Create` hook and `CreateAllowed` run exactly as they do
    // for a reaction's creation.
    engine
        .bufs
        .create_part(x, y, ty, CreationSource::Program, 0, &engine.registry)
}

/// Builds the `sim` table.
pub(super) fn build_sim_table(lua: &Lua, bridge: &Bridge) -> mlua::Result<Table> {
    let sim = lua.create_table()?;
    constants::register_constants(lua, &sim)?;
    lifecycle::register_lifecycle(lua, bridge, &sim)?;
    properties::register_properties(lua, bridge, &sim)?;
    grid::register_grid(lua, bridge, &sim)?;
    state::register_state(lua, bridge, &sim)?;
    tools::register_tools(lua, bridge, &sim)?;
    saves::register_saves(lua, bridge, &sim)?;
    gol::register_gol(lua, bridge, &sim)?;
    signs::register_signs(lua, bridge, &sim)?;
    Ok(sim)
}
