//! Block map grid getters and setters (Eulerian air, walls, gravity, pmap).

use mlua::{Lua, MultiValue, Table, Value};
use std::rc::Rc;
use tpt_core::constants::{MAX_PRESSURE, MAX_TEMP, MIN_PRESSURE, MIN_TEMP, WL_NUM, pmap_id};

use crate::api::{pair, refresh_air, refresh_map_rect, value_to_f64, value_to_i64};
use crate::bridge::{Bridge, borrow_bridge};

pub(crate) fn register_grid(lua: &Lua, bridge: &Bridge, sim: &Table) -> mlua::Result<()> {
    // pressure(x, y, [val]) or (x, y, w, h, val)
    register_block_map_f32(
        lua,
        bridge,
        sim,
        "pressure",
        Some((MIN_PRESSURE, MAX_PRESSURE)),
        |bufs, idx| {
            // One air-grid read per Lua call, not a whole-pool read-back.
            refresh_air(bufs);
            bufs.air_pv[idx]
        },
        |bufs, idx, val| {
            bufs.air_pv[idx] = val;
            bufs.host_air_dirty = true;
        },
    )?;

    // ambientHeat(x, y, [val]) or (x, y, w, h, val)
    register_block_map_f32(
        lua,
        bridge,
        sim,
        "ambientHeat",
        Some((MIN_TEMP, MAX_TEMP)),
        |bufs, idx| {
            refresh_air(bufs);
            bufs.air_hv[idx]
        },
        |bufs, idx, val| {
            bufs.air_hv[idx] = val;
            bufs.host_air_dirty = true;
        },
    )?;

    // velocityX(x, y, [val]) or (x, y, w, h, val)
    register_block_map_f32(
        lua,
        bridge,
        sim,
        "velocityX",
        Some((MIN_PRESSURE, MAX_PRESSURE)),
        |bufs, idx| {
            refresh_air(bufs);
            bufs.air_vx[idx]
        },
        |bufs, idx, val| {
            bufs.air_vx[idx] = val;
            bufs.host_air_dirty = true;
        },
    )?;

    // velocityY(x, y, [val]) or (x, y, w, h, val)
    register_block_map_f32(
        lua,
        bridge,
        sim,
        "velocityY",
        Some((MIN_PRESSURE, MAX_PRESSURE)),
        |bufs, idx| {
            refresh_air(bufs);
            bufs.air_vy[idx]
        },
        |bufs, idx, val| {
            bufs.air_vy[idx] = val;
            bufs.host_air_dirty = true;
        },
    )?;

    // wallMap(x, y, [val]) or (x, y, w, h, val)
    register_block_map_u32(
        lua,
        bridge,
        sim,
        "wallMap",
        Some((0, (WL_NUM as u32).saturating_sub(1))),
        |bufs, idx| bufs.wall_bmap[idx],
        |bufs, idx, val| {
            bufs.wall_bmap[idx] = val;
            bufs.wall_emap[idx] = 0;
            bufs.rebuild_wall_masks();
            bufs.host_walls_dirty = true;
        },
    )?;

    // elecMap(x, y, [val]) or (x, y, w, h, val)
    register_block_map_u32(
        lua,
        bridge,
        sim,
        "elecMap",
        Some((0, 255)),
        |bufs, idx| bufs.wall_emap[idx],
        |bufs, idx, val| {
            bufs.wall_emap[idx] = val;
            bufs.host_wall_state_dirty = true;
        },
    )?;

    // fanVelocityX(x, y, [val]) or (x, y, w, h, val)
    register_block_map_f32(
        lua,
        bridge,
        sim,
        "fanVelocityX",
        None,
        |bufs, idx| bufs.fan_vx[idx],
        |bufs, idx, val| {
            bufs.fan_vx[idx] = val;
            bufs.host_air_dirty = true;
        },
    )?;

    // fanVelocityY(x, y, [val]) or (x, y, w, h, val)
    register_block_map_f32(
        lua,
        bridge,
        sim,
        "fanVelocityY",
        None,
        |bufs, idx| bufs.fan_vy[idx],
        |bufs, idx, val| {
            bufs.fan_vy[idx] = val;
            bufs.host_air_dirty = true;
        },
    )?;

    // gravityMass(x, y, [val]) or (x, y, w, h, val)
    register_block_map_f32(
        lua,
        bridge,
        sim,
        "gravityMass",
        None,
        |bufs, idx| {
            // The Poisson solve is device-resident; refresh the shadow on demand.
            bufs.sync_gravity_gpu_to_host();
            bufs.grav_mass[idx]
        },
        |bufs, idx, val| {
            bufs.grav_mass[idx] = val;
            bufs.grav_mass_dirty = true;
        },
    )?;

    // gravityMask(x, y, [val]) or (x, y, w, h, val)
    register_block_map_u32(
        lua,
        bridge,
        sim,
        "gravityMask",
        None,
        |bufs, idx| bufs.grav_mask[idx],
        |bufs, idx, val| {
            bufs.grav_mask[idx] = val;
            bufs.grav_mask_dirty = true;
        },
    )?;

    // sim.gravityField(x, y) -> fx, fy
    sim.set(
        "gravityField",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (x, y): (i32, i32)| -> mlua::Result<MultiValue> {
                let mut state = borrow_bridge(&bridge)?;

                state.with_engine(|engine| {
                    let xcells = engine.bufs.config.xcells() as i32;
                    let ycells = engine.bufs.config.ycells() as i32;
                    if x < 0 || y < 0 || x >= xcells || y >= ycells {
                        return Err(mlua::Error::runtime(format!(
                            "Coordinates ({x}, {y}) out of range"
                        )));
                    }
                    let idx = (y * xcells + x) as usize;
                    engine.bufs.sync_gravity_gpu_to_host();
                    let fx = engine.bufs.grav_fx.get(idx).copied().unwrap_or(0.0);
                    let fy = engine.bufs.grav_fy.get(idx).copied().unwrap_or(0.0);
                    Ok(pair(fx, fy))
                })?
            }
        })?,
    )?;

    // sim.pmap(x, y) -> id | nil
    sim.set(
        "pmap",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (x, y): (i32, i32)| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;

                state.with_engine(|engine| {
                    let xres = engine.bufs.config.xres as i32;
                    let yres = engine.bufs.config.yres as i32;
                    if x < 0 || y < 0 || x >= xres || y >= yres {
                        return Err(mlua::Error::runtime(format!(
                            "coordinates out of range ({x},{y})"
                        )));
                    }
                    let pixel = (y * xres + x) as usize;
                    refresh_map_rect(&mut engine.bufs, x, y, x, y);
                    let word = engine.bufs.pmap[pixel];
                    Ok(match pmap_id(word) {
                        Some(id) => Value::Integer(id as i64),
                        None => Value::Nil,
                    })
                })?
            }
        })?,
    )?;

    // sim.photons(x, y) -> id | nil
    sim.set(
        "photons",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (x, y): (i32, i32)| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;

                state.with_engine(|engine| {
                    let xres = engine.bufs.config.xres as i32;
                    let yres = engine.bufs.config.yres as i32;
                    if x < 0 || y < 0 || x >= xres || y >= yres {
                        return Err(mlua::Error::runtime(format!(
                            "coordinates out of range ({x},{y})"
                        )));
                    }
                    let pixel = (y * xres + x) as usize;
                    refresh_map_rect(&mut engine.bufs, x, y, x, y);
                    let word = engine.bufs.photons[pixel];
                    Ok(match pmap_id(word) {
                        Some(id) => Value::Integer(id as i64),
                        None => Value::Nil,
                    })
                })?
            }
        })?,
    )?;

    // sim.resetPressure([x, y, w, h])
    sim.set(
        "resetPressure",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    let xcells = engine.bufs.config.xcells() as i32;
                    let ycells = engine.bufs.config.ycells() as i32;
                    let (x, y, w, h) = if vec.len() >= 4 {
                        (
                            value_to_i64(&vec[0]).unwrap_or(0) as i32,
                            value_to_i64(&vec[1]).unwrap_or(0) as i32,
                            value_to_i64(&vec[2]).unwrap_or(xcells as i64) as i32,
                            value_to_i64(&vec[3]).unwrap_or(ycells as i64) as i32,
                        )
                    } else if vec.len() >= 2 {
                        (
                            value_to_i64(&vec[0]).unwrap_or(0) as i32,
                            value_to_i64(&vec[1]).unwrap_or(0) as i32,
                            1,
                            1,
                        )
                    } else {
                        (0, 0, xcells, ycells)
                    };
                    engine.reset_pressure(x, y, w, h);
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.resetVelocity([x, y, w, h])
    sim.set(
        "resetVelocity",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    let xcells = engine.bufs.config.xcells() as i32;
                    let ycells = engine.bufs.config.ycells() as i32;
                    let (x, y, w, h) = if vec.len() >= 4 {
                        (
                            value_to_i64(&vec[0]).unwrap_or(0) as i32,
                            value_to_i64(&vec[1]).unwrap_or(0) as i32,
                            value_to_i64(&vec[2]).unwrap_or(xcells as i64) as i32,
                            value_to_i64(&vec[3]).unwrap_or(ycells as i64) as i32,
                        )
                    } else {
                        (0, 0, xcells, ycells)
                    };
                    engine.reset_velocity(x, y, w, h);
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.velocity(x, y, [vx, vy]) helper/compatibility
    sim.set(
        "velocity",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<MultiValue> {
                let vec = args.into_vec();
                if vec.len() < 2 {
                    return Err(mlua::Error::runtime("expected at least x and y"));
                }
                let x = value_to_i64(&vec[0])? as i32;
                let y = value_to_i64(&vec[1])? as i32;

                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    let xcells = engine.bufs.config.xcells() as i32;
                    let ycells = engine.bufs.config.ycells() as i32;
                    if x < 0 || y < 0 || x >= xcells || y >= ycells {
                        return Err(mlua::Error::runtime(format!(
                            "Coordinates ({x}, {y}) out of range"
                        )));
                    }
                    let idx = (y * xcells + x) as usize;
                    if vec.len() >= 4 {
                        let vx = value_to_f64(&vec[2])? as f32;
                        let vy = value_to_f64(&vec[3])? as f32;
                        engine.bufs.air_vx[idx] = vx;
                        engine.bufs.air_vy[idx] = vy;
                        engine.bufs.host_air_dirty = true;
                        engine.bufs.host_dirty = true;
                        let mut vals = MultiValue::new();
                        vals.push_back(Value::Number(vx as f64));
                        vals.push_back(Value::Number(vy as f64));
                        Ok(vals)
                    } else {
                        refresh_air(&mut engine.bufs);
                        let vx = engine.bufs.air_vx[idx];
                        let vy = engine.bufs.air_vy[idx];
                        let mut vals = MultiValue::new();
                        vals.push_back(Value::Number(vx as f64));
                        vals.push_back(Value::Number(vy as f64));
                        Ok(vals)
                    }
                })?
            }
        })?,
    )?;

    Ok(())
}

fn register_block_map_f32<FGet, FSet>(
    lua: &Lua,
    bridge: &Bridge,
    sim: &Table,
    name: &str,
    clamp: Option<(f32, f32)>,
    get: FGet,
    set: FSet,
) -> mlua::Result<()>
where
    FGet: Fn(&mut tpt_sim::buffers::SimBuffers, usize) -> f32 + 'static,
    FSet: Fn(&mut tpt_sim::buffers::SimBuffers, usize, f32) + 'static,
{
    let bridge = Rc::clone(bridge);
    sim.set(
        name,
        lua.create_function(move |_, args: MultiValue| -> mlua::Result<Value> {
            let vec = args.into_vec();
            if vec.len() < 2 {
                return Err(mlua::Error::runtime("expected at least x and y"));
            }
            let x = value_to_i64(&vec[0])? as i32;
            let y = value_to_i64(&vec[1])? as i32;

            let mut state = borrow_bridge(&bridge)?;
            state.with_engine(|engine| {
                let xcells = engine.bufs.config.xcells() as i32;
                let ycells = engine.bufs.config.ycells() as i32;
                if x < 0 || y < 0 || x >= xcells || y >= ycells {
                    return Err(mlua::Error::runtime(format!(
                        "Coordinates ({x}, {y}) out of range"
                    )));
                }

                if vec.len() == 2 {
                    let idx = (y * xcells + x) as usize;
                    Ok(Value::Number(get(&mut engine.bufs, idx) as f64))
                } else {
                    let (w, h, val_idx) = if vec.len() > 3 {
                        let w = value_to_i64(&vec[2])? as i32;
                        let h = value_to_i64(&vec[3])? as i32;
                        (w, h, 4)
                    } else {
                        (1, 1, 2)
                    };
                    let mut val = value_to_f64(&vec[val_idx])? as f32;
                    if let Some((min_val, max_val)) = clamp {
                        val = val.clamp(min_val, max_val);
                    }

                    for cy in y..(y + h).min(ycells) {
                        for cx in x..(x + w).min(xcells) {
                            if cx >= 0 && cy >= 0 {
                                let idx = (cy * xcells + cx) as usize;
                                set(&mut engine.bufs, idx, val);
                            }
                        }
                    }
                    engine.bufs.host_dirty = true;
                    Ok(Value::Nil)
                }
            })?
        })?,
    )?;
    Ok(())
}

fn register_block_map_u32<FGet, FSet>(
    lua: &Lua,
    bridge: &Bridge,
    sim: &Table,
    name: &str,
    clamp: Option<(u32, u32)>,
    get: FGet,
    set: FSet,
) -> mlua::Result<()>
where
    FGet: Fn(&tpt_sim::buffers::SimBuffers, usize) -> u32 + 'static,
    FSet: Fn(&mut tpt_sim::buffers::SimBuffers, usize, u32) + 'static,
{
    let bridge = Rc::clone(bridge);
    sim.set(
        name,
        lua.create_function(move |_, args: MultiValue| -> mlua::Result<Value> {
            let vec = args.into_vec();
            if vec.len() < 2 {
                return Err(mlua::Error::runtime("expected at least x and y"));
            }
            let x = value_to_i64(&vec[0])? as i32;
            let y = value_to_i64(&vec[1])? as i32;

            let mut state = borrow_bridge(&bridge)?;
            state.with_engine(|engine| {
                let xcells = engine.bufs.config.xcells() as i32;
                let ycells = engine.bufs.config.ycells() as i32;
                if x < 0 || y < 0 || x >= xcells || y >= ycells {
                    return Err(mlua::Error::runtime(format!(
                        "Coordinates ({x}, {y}) out of range"
                    )));
                }

                if vec.len() == 2 {
                    let idx = (y * xcells + x) as usize;
                    Ok(Value::Integer(get(&engine.bufs, idx) as i64))
                } else {
                    let (w, h, val_idx) = if vec.len() > 3 {
                        let w = value_to_i64(&vec[2])? as i32;
                        let h = value_to_i64(&vec[3])? as i32;
                        (w, h, 4)
                    } else {
                        (1, 1, 2)
                    };
                    let mut val = value_to_i64(&vec[val_idx])? as u32;
                    if let Some((min_val, max_val)) = clamp {
                        val = val.clamp(min_val, max_val);
                    }

                    for cy in y..(y + h).min(ycells) {
                        for cx in x..(x + w).min(xcells) {
                            if cx >= 0 && cy >= 0 {
                                let idx = (cy * xcells + cx) as usize;
                                set(&mut engine.bufs, idx, val);
                            }
                        }
                    }
                    engine.bufs.host_dirty = true;
                    Ok(Value::Nil)
                }
            })?
        })?,
    )?;
    Ok(())
}
