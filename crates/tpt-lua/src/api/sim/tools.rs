//! Creation, brush, tool, wall and decoration commands.

use mlua::{Lua, MultiValue, Table, Value};
use std::cell::RefCell;
use std::rc::Rc;
use tpt_core::commands::{BrushShape, DecoMode, SimCommand, SimToolKind};
use tpt_core::constants::{PT_LIGH, WL_NUM};

use super::geom::*;

use crate::api::{value_to_f64, value_to_i64};
use crate::bridge::{Bridge, borrow_bridge};

pub(crate) fn register_tools(lua: &Lua, bridge: &Bridge, sim: &Table) -> mlua::Result<()> {
    // Current decoration colour stored for sim.decoColor
    let current_deco_color = Rc::new(RefCell::new(0xFFFF_0000u32)); // Default red

    // sim.brush(x, y, [rx, ry], [brushID]) -> iterator
    sim.set(
        "brush",
        lua.create_function({
            move |lua, args: MultiValue| -> mlua::Result<mlua::Function> {
                let vec = args.into_vec();
                if vec.len() < 2 {
                    return Err(mlua::Error::runtime("expected x, y"));
                }
                let cx = value_to_i64(&vec[0])? as i32;
                let cy = value_to_i64(&vec[1])? as i32;
                let (rx, ry, brush_id) = if vec.len() >= 4 {
                    let rx = value_to_i64(&vec[2])? as i32;
                    let ry = value_to_i64(&vec[3])? as i32;
                    let b = if vec.len() >= 5 {
                        value_to_i64(&vec[4])? as i32
                    } else {
                        0
                    };
                    (rx, ry, b)
                } else {
                    let b = if vec.len() >= 3 {
                        value_to_i64(&vec[2])? as i32
                    } else {
                        0
                    };
                    (5, 5, b)
                };

                let shape = match brush_id {
                    1 => BrushShape::Square,
                    2 => BrushShape::Triangle,
                    _ => BrushShape::Circle,
                };
                let points = generate_brush_points(cx, cy, rx, ry, shape);
                let cursor = Rc::new(RefCell::new(0usize));

                lua.create_function(move |_, ()| -> mlua::Result<MultiValue> {
                    let mut cur = cursor.borrow_mut();
                    if *cur < points.len() {
                        let (px, py) = points[*cur];
                        *cur += 1;
                        let mut vals = MultiValue::new();
                        vals.push_back(Value::Integer(px as i64));
                        vals.push_back(Value::Integer(py as i64));
                        Ok(vals)
                    } else {
                        Ok(MultiValue::new())
                    }
                })
            }
        })?,
    )?;

    // sim.createParts(x, y, rx, ry, [c], [brushID], [flags]) -> int
    sim.set(
        "createParts",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<i64> {
                let vec = args.into_vec();
                let x = if !vec.is_empty() {
                    value_to_i64(&vec[0])? as i32
                } else {
                    -1
                };
                let y = if vec.len() >= 2 {
                    value_to_i64(&vec[1])? as i32
                } else {
                    -1
                };
                let rx = if vec.len() >= 3 {
                    value_to_i64(&vec[2])? as i32
                } else {
                    5
                };
                let ry = if vec.len() >= 4 {
                    value_to_i64(&vec[3])? as i32
                } else {
                    5
                };
                let elem = if vec.len() >= 5 {
                    value_to_i64(&vec[4])? as u32
                } else {
                    1
                }; // Default DUST
                let brush_id = if vec.len() >= 6 {
                    value_to_i64(&vec[5])? as i32
                } else {
                    0
                };
                let flags = if vec.len() >= 7 {
                    value_to_i64(&vec[6])? as u32
                } else {
                    0
                };

                let shape = match brush_id {
                    1 => BrushShape::Square,
                    2 => BrushShape::Triangle,
                    _ => BrushShape::Circle,
                };
                let points = generate_brush_points(x, y, rx, ry, shape);
                let replace_mode = (flags & 0x01) != 0;

                let mut state = borrow_bridge(&bridge)?;
                // `CreateParts`'s `PT_LIGH` special case: one bolt at the
                // brush centre, powered by the radius.
                if elem == PT_LIGH {
                    state.dispatch(SimCommand::CreateLightning {
                        x,
                        y,
                        radius_x: rx,
                        radius_y: ry,
                    })?;
                    state.particles_dirty = true;
                    return Ok(1);
                }
                let count = points.len() as i64;
                state.dispatch(SimCommand::PlaceParticles {
                    points,
                    elem_ty: elem,
                    replace_mode,
                    dcolour: 0,
                })?;
                state.particles_dirty = true;
                Ok(count)
            }
        })?,
    )?;

    // sim.createLine(x1, y1, x2, y2, rx, ry, [c], [brushID], [flags])
    sim.set(
        "createLine",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                if vec.len() < 4 {
                    return Err(mlua::Error::runtime("expected x1, y1, x2, y2"));
                }
                let x1 = value_to_i64(&vec[0])? as i32;
                let y1 = value_to_i64(&vec[1])? as i32;
                let x2 = value_to_i64(&vec[2])? as i32;
                let y2 = value_to_i64(&vec[3])? as i32;
                let rx = if vec.len() >= 5 {
                    value_to_i64(&vec[4])? as i32
                } else {
                    0
                };
                let ry = if vec.len() >= 6 {
                    value_to_i64(&vec[5])? as i32
                } else {
                    0
                };
                let elem = if vec.len() >= 7 {
                    value_to_i64(&vec[6])? as u32
                } else {
                    1
                };
                let brush_id = if vec.len() >= 8 {
                    value_to_i64(&vec[7])? as i32
                } else {
                    0
                };
                let flags = if vec.len() >= 9 {
                    value_to_i64(&vec[8])? as u32
                } else {
                    0
                };

                let shape = match brush_id {
                    1 => BrushShape::Square,
                    2 => BrushShape::Triangle,
                    _ => BrushShape::Circle,
                };
                let points = generate_line_points(x1, y1, x2, y2, rx, ry, shape);
                let replace_mode = (flags & 0x01) != 0;

                let mut state = borrow_bridge(&bridge)?;
                // `CreateParts`'s `PT_LIGH` special case: the throttle admits
                // the first pixel of the line and refuses the rest.
                if elem == PT_LIGH {
                    state.dispatch(SimCommand::CreateLightning {
                        x: x1,
                        y: y1,
                        radius_x: rx,
                        radius_y: ry,
                    })?;
                    state.particles_dirty = true;
                    return Ok(());
                }
                state.dispatch(SimCommand::PlaceParticles {
                    points,
                    elem_ty: elem,
                    replace_mode,
                    dcolour: 0,
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.createBox(x1, y1, x2, y2, [c], [flags])
    sim.set(
        "createBox",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                if vec.len() < 4 {
                    return Err(mlua::Error::runtime("expected x1, y1, x2, y2"));
                }
                let x1 = value_to_i64(&vec[0])? as i32;
                let y1 = value_to_i64(&vec[1])? as i32;
                let x2 = value_to_i64(&vec[2])? as i32;
                let y2 = value_to_i64(&vec[3])? as i32;
                let elem = if vec.len() >= 5 {
                    value_to_i64(&vec[4])? as u32
                } else {
                    1
                };
                let flags = if vec.len() >= 6 {
                    value_to_i64(&vec[5])? as u32
                } else {
                    0
                };

                let points = generate_box_points(x1, y1, x2, y2);
                let replace_mode = (flags & 0x01) != 0;

                let mut state = borrow_bridge(&bridge)?;
                // `CreateBox` walks the rectangle with a zero radius, so the
                // throttle admits one bolt at the first pixel.
                if elem == PT_LIGH {
                    state.dispatch(SimCommand::CreateLightning {
                        x: x1,
                        y: y1,
                        radius_x: 0,
                        radius_y: 0,
                    })?;
                    state.particles_dirty = true;
                    return Ok(());
                }
                state.dispatch(SimCommand::PlaceParticles {
                    points,
                    elem_ty: elem,
                    replace_mode,
                    dcolour: 0,
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.floodParts(x, y, [c], [cm], [flags]) -> int
    sim.set(
        "floodParts",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<i64> {
                let vec = args.into_vec();
                if vec.len() < 2 {
                    return Err(mlua::Error::runtime("expected x, y"));
                }
                let x = value_to_i64(&vec[0])? as i32;
                let y = value_to_i64(&vec[1])? as i32;
                let elem = if vec.len() >= 3 {
                    value_to_i64(&vec[2])? as u32
                } else {
                    1
                };

                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    let xres = engine.bufs.config.xres as i32;
                    let yres = engine.bufs.config.yres as i32;
                    if x < 0 || y < 0 || x >= xres || y >= yres {
                        return Err(mlua::Error::runtime(format!(
                            "coordinates out of range ({x},{y})"
                        )));
                    }
                    Ok(())
                })??;

                state.dispatch(SimCommand::FloodFillParticles {
                    x,
                    y,
                    elem_ty: elem,
                })?;
                // `flood_fill` updates the map words of the slots it creates
                // and uploads them; a full rebuild would remap the pool from a
                // host shadow that may lag the device.
                state.particles_dirty = true;
                Ok(1)
            }
        })?,
    )?;

    // sim.createWalls(x, y, rx, ry, [c]) -> int
    sim.set(
        "createWalls",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<i64> {
                let vec = args.into_vec();
                if vec.len() < 2 {
                    return Err(mlua::Error::runtime("expected x, y"));
                }
                let x = value_to_i64(&vec[0])? as i32;
                let y = value_to_i64(&vec[1])? as i32;
                let rx = if vec.len() >= 3 {
                    value_to_i64(&vec[2])? as i32
                } else {
                    0
                };
                let ry = if vec.len() >= 4 {
                    value_to_i64(&vec[3])? as i32
                } else {
                    0
                };
                let wall_id = if vec.len() >= 5 {
                    value_to_i64(&vec[4])? as u8
                } else {
                    8
                }; // WL_WALL

                if wall_id >= WL_NUM as u8 {
                    return Err(mlua::Error::runtime(format!(
                        "Unrecognised wall id '{wall_id}'"
                    )));
                }

                let cells = generate_wall_brush_cells(x, y, rx, ry);
                let count = cells.len() as i64;
                let mut state = borrow_bridge(&bridge)?;
                state.dispatch(SimCommand::PlaceWalls {
                    cells,
                    wall_id,
                    fan_vel: None,
                })?;
                state.particles_dirty = true;
                Ok(count)
            }
        })?,
    )?;

    // sim.createWallLine(x1, y1, x2, y2, rx, ry, [c])
    sim.set(
        "createWallLine",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                if vec.len() < 4 {
                    return Err(mlua::Error::runtime("expected x1, y1, x2, y2"));
                }
                let x1 = value_to_i64(&vec[0])? as i32;
                let y1 = value_to_i64(&vec[1])? as i32;
                let x2 = value_to_i64(&vec[2])? as i32;
                let y2 = value_to_i64(&vec[3])? as i32;
                let rx = if vec.len() >= 5 {
                    value_to_i64(&vec[4])? as i32
                } else {
                    0
                };
                let ry = if vec.len() >= 6 {
                    value_to_i64(&vec[5])? as i32
                } else {
                    0
                };
                let wall_id = if vec.len() >= 7 {
                    value_to_i64(&vec[6])? as u8
                } else {
                    8
                };

                if wall_id >= WL_NUM as u8 {
                    return Err(mlua::Error::runtime(format!(
                        "Unrecognised wall id '{wall_id}'"
                    )));
                }

                let cells = generate_wall_line_cells(x1, y1, x2, y2, rx, ry);
                let mut state = borrow_bridge(&bridge)?;
                state.dispatch(SimCommand::PlaceWalls {
                    cells,
                    wall_id,
                    fan_vel: None,
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.createWallBox(x1, y1, x2, y2, [c])
    sim.set(
        "createWallBox",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                if vec.len() < 4 {
                    return Err(mlua::Error::runtime("expected x1, y1, x2, y2"));
                }
                let x1 = value_to_i64(&vec[0])? as i32;
                let y1 = value_to_i64(&vec[1])? as i32;
                let x2 = value_to_i64(&vec[2])? as i32;
                let y2 = value_to_i64(&vec[3])? as i32;
                let wall_id = if vec.len() >= 5 {
                    value_to_i64(&vec[4])? as u8
                } else {
                    8
                };

                if wall_id >= WL_NUM as u8 {
                    return Err(mlua::Error::runtime(format!(
                        "Unrecognised wall id '{wall_id}'"
                    )));
                }

                let cells = generate_wall_box_cells(x1, y1, x2, y2);
                let mut state = borrow_bridge(&bridge)?;
                state.dispatch(SimCommand::PlaceWalls {
                    cells,
                    wall_id,
                    fan_vel: None,
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.floodWalls(x, y, [c], [bm]) -> int
    sim.set(
        "floodWalls",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<i64> {
                let vec = args.into_vec();
                if vec.len() < 2 {
                    return Err(mlua::Error::runtime("expected x, y"));
                }
                let x = value_to_i64(&vec[0])? as i32;
                let y = value_to_i64(&vec[1])? as i32;
                let wall_id = if vec.len() >= 3 {
                    value_to_i64(&vec[2])? as u8
                } else {
                    8
                };
                let replace_wall = if vec.len() >= 4 {
                    let bm = value_to_i64(&vec[3])? as i32;
                    if bm >= 0 { Some(bm as u8) } else { None }
                } else {
                    None
                };

                if wall_id >= WL_NUM as u8 {
                    return Err(mlua::Error::runtime(format!(
                        "Unrecognised wall id '{wall_id}'"
                    )));
                }

                let mut state = borrow_bridge(&bridge)?;
                let count = state.with_engine(|engine| {
                    let xres = engine.bufs.config.xres as i32;
                    let yres = engine.bufs.config.yres as i32;
                    if x < 0 || y < 0 || x >= xres || y >= yres {
                        return Err(mlua::Error::runtime(format!(
                            "coordinates out of range ({x},{y})"
                        )));
                    }
                    Ok(engine.flood_walls(x, y, wall_id, replace_wall) as i64)
                })??;

                state.particles_dirty = true;
                Ok(count)
            }
        })?,
    )?;

    // sim.toolBrush(x, y, rx, ry, tool, [brushID], [strength])
    sim.set(
        "toolBrush",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                if vec.len() < 5 {
                    return Err(mlua::Error::runtime("expected x, y, rx, ry, tool"));
                }
                let x = value_to_i64(&vec[0])? as i32;
                let y = value_to_i64(&vec[1])? as i32;
                let rx = value_to_i64(&vec[2])? as i32;
                let ry = value_to_i64(&vec[3])? as i32;
                let tool_id = value_to_i64(&vec[4])? as u32;
                let brush_id = if vec.len() >= 6 {
                    value_to_i64(&vec[5])? as i32
                } else {
                    0
                };
                let strength = if vec.len() >= 7 {
                    value_to_f64(&vec[6])? as f32
                } else {
                    1.0
                };

                let Some(tool) = SimToolKind::from_index(tool_id) else {
                    return Err(mlua::Error::runtime(format!("Invalid tool id '{tool_id}'")));
                };

                let shape = match brush_id {
                    1 => BrushShape::Square,
                    2 => BrushShape::Triangle,
                    _ => BrushShape::Circle,
                };
                let points = generate_brush_points(x, y, rx, ry, shape);

                let mut state = borrow_bridge(&bridge)?;
                state.dispatch(SimCommand::ApplySimTool {
                    points,
                    tool,
                    strength,
                    dir: (0.0, 0.0),
                    // `SimTool.cpp:16`: a brush stamp is centred on its origin.
                    center: (x, y),
                    dragging: true,
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.toolLine(x1, y1, x2, y2, rx, ry, tool, [brushID], [strength])
    sim.set(
        "toolLine",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                if vec.len() < 7 {
                    return Err(mlua::Error::runtime(
                        "expected x1, y1, x2, y2, rx, ry, tool",
                    ));
                }
                let x1 = value_to_i64(&vec[0])? as i32;
                let y1 = value_to_i64(&vec[1])? as i32;
                let x2 = value_to_i64(&vec[2])? as i32;
                let y2 = value_to_i64(&vec[3])? as i32;
                let rx = value_to_i64(&vec[4])? as i32;
                let ry = value_to_i64(&vec[5])? as i32;
                let tool_id = value_to_i64(&vec[6])? as u32;
                let brush_id = if vec.len() >= 8 {
                    value_to_i64(&vec[7])? as i32
                } else {
                    0
                };
                let strength = if vec.len() >= 9 {
                    value_to_f64(&vec[8])? as f32
                } else {
                    1.0
                };

                let Some(tool) = SimToolKind::from_index(tool_id) else {
                    return Err(mlua::Error::runtime(format!("Invalid tool id '{tool_id}'")));
                };

                let shape = match brush_id {
                    1 => BrushShape::Square,
                    2 => BrushShape::Triangle,
                    _ => BrushShape::Circle,
                };
                let points = generate_line_points(x1, y1, x2, y2, rx, ry, shape);

                // `WIND.cpp:28-29` scales the pixel step, so pass the raw delta.
                let dx = (x2 - x1) as f32;
                let dy = (y2 - y1) as f32;
                let dir = (dx, dy);

                let mut state = borrow_bridge(&bridge)?;
                state.dispatch(SimCommand::ApplySimTool {
                    points,
                    tool,
                    strength,
                    dir,
                    // A single centre cannot express the per-sample centres of
                    // `defaultPerformDrawLine`; the stroke midpoint is closest.
                    center: ((x1 + x2) / 2, (y1 + y2) / 2),
                    dragging: true,
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.toolBox(x1, y1, x2, y2, tool, [strength], [brushID], [rx], [ry])
    sim.set(
        "toolBox",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                if vec.len() < 5 {
                    return Err(mlua::Error::runtime("expected x1, y1, x2, y2, tool"));
                }
                let x1 = value_to_i64(&vec[0])? as i32;
                let y1 = value_to_i64(&vec[1])? as i32;
                let x2 = value_to_i64(&vec[2])? as i32;
                let y2 = value_to_i64(&vec[3])? as i32;
                let tool_id = value_to_i64(&vec[4])? as u32;
                let strength = if vec.len() >= 6 {
                    value_to_f64(&vec[5])? as f32
                } else {
                    1.0
                };

                let Some(tool) = SimToolKind::from_index(tool_id) else {
                    return Err(mlua::Error::runtime(format!("Invalid tool id '{tool_id}'")));
                };

                let points = generate_box_points(x1, y1, x2, y2);
                let mut state = borrow_bridge(&bridge)?;
                state.dispatch(SimCommand::ApplySimTool {
                    points,
                    tool,
                    strength,
                    dir: (0.0, 0.0),
                    // `SimTool.cpp:88-89`: the rect gesture rotates about the
                    // box centre for every pixel.
                    center: ((x1 + x2) / 2, (y1 + y2) / 2),
                    dragging: true,
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.decoBrush(x, y, rx, ry, r, g, b, a, [tool], [brushID])
    sim.set(
        "decoBrush",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                if vec.len() < 8 {
                    return Err(mlua::Error::runtime("expected x, y, rx, ry, r, g, b, a"));
                }
                let x = value_to_i64(&vec[0])? as i32;
                let y = value_to_i64(&vec[1])? as i32;
                let rx = value_to_i64(&vec[2])? as i32;
                let ry = value_to_i64(&vec[3])? as i32;
                let r = (value_to_i64(&vec[4])? as u32).clamp(0, 255);
                let g = (value_to_i64(&vec[5])? as u32).clamp(0, 255);
                let b = (value_to_i64(&vec[6])? as u32).clamp(0, 255);
                let a = (value_to_i64(&vec[7])? as u32).clamp(0, 255);
                let tool = if vec.len() >= 9 {
                    value_to_i64(&vec[8])? as u32
                } else {
                    0
                };
                let brush_id = if vec.len() >= 10 {
                    value_to_i64(&vec[9])? as i32
                } else {
                    0
                };

                let shape = match brush_id {
                    1 => BrushShape::Square,
                    2 => BrushShape::Triangle,
                    _ => BrushShape::Circle,
                };
                let points = generate_brush_points(x, y, rx, ry, shape);
                let mode = DecoMode::from_index(tool);
                let colour = (a << 24) | (r << 16) | (g << 8) | b;

                let mut state = borrow_bridge(&bridge)?;
                state.dispatch(SimCommand::DecorateAt {
                    points,
                    colour,
                    mode,
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.decoLine(x1, y1, x2, y2, rx, ry, r, g, b, a, [tool], [brushID])
    sim.set(
        "decoLine",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                if vec.len() < 10 {
                    return Err(mlua::Error::runtime(
                        "expected x1, y1, x2, y2, rx, ry, r, g, b, a",
                    ));
                }
                let x1 = value_to_i64(&vec[0])? as i32;
                let y1 = value_to_i64(&vec[1])? as i32;
                let x2 = value_to_i64(&vec[2])? as i32;
                let y2 = value_to_i64(&vec[3])? as i32;
                let rx = value_to_i64(&vec[4])? as i32;
                let ry = value_to_i64(&vec[5])? as i32;
                let r = (value_to_i64(&vec[6])? as u32).clamp(0, 255);
                let g = (value_to_i64(&vec[7])? as u32).clamp(0, 255);
                let b = (value_to_i64(&vec[8])? as u32).clamp(0, 255);
                let a = (value_to_i64(&vec[9])? as u32).clamp(0, 255);
                let tool = if vec.len() >= 11 {
                    value_to_i64(&vec[10])? as u32
                } else {
                    0
                };
                let brush_id = if vec.len() >= 12 {
                    value_to_i64(&vec[11])? as i32
                } else {
                    0
                };

                let shape = match brush_id {
                    1 => BrushShape::Square,
                    2 => BrushShape::Triangle,
                    _ => BrushShape::Circle,
                };
                let points = generate_line_points(x1, y1, x2, y2, rx, ry, shape);
                let mode = DecoMode::from_index(tool);
                let colour = (a << 24) | (r << 16) | (g << 8) | b;

                let mut state = borrow_bridge(&bridge)?;
                state.dispatch(SimCommand::DecorateAt {
                    points,
                    colour,
                    mode,
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.decoBox(x1, y1, x2, y2, r, g, b, a, [tool])
    sim.set(
        "decoBox",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                if vec.len() < 8 {
                    return Err(mlua::Error::runtime("expected x1, y1, x2, y2, r, g, b, a"));
                }
                let x1 = value_to_i64(&vec[0])? as i32;
                let y1 = value_to_i64(&vec[1])? as i32;
                let x2 = value_to_i64(&vec[2])? as i32;
                let y2 = value_to_i64(&vec[3])? as i32;
                let r = (value_to_i64(&vec[4])? as u32).clamp(0, 255);
                let g = (value_to_i64(&vec[5])? as u32).clamp(0, 255);
                let b = (value_to_i64(&vec[6])? as u32).clamp(0, 255);
                let a = (value_to_i64(&vec[7])? as u32).clamp(0, 255);
                let tool = if vec.len() >= 9 {
                    value_to_i64(&vec[8])? as u32
                } else {
                    0
                };

                let points = generate_box_points(x1, y1, x2, y2);
                let mode = DecoMode::from_index(tool);
                let colour = (a << 24) | (r << 16) | (g << 8) | b;

                let mut state = borrow_bridge(&bridge)?;
                state.dispatch(SimCommand::DecorateAt {
                    points,
                    colour,
                    mode,
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.decoColor([color] | [r, g, b, a]) -> int | ()
    sim.set(
        "decoColor",
        lua.create_function({
            let deco_col = Rc::clone(&current_deco_color);
            move |_, args: MultiValue| -> mlua::Result<Value> {
                let vec = args.into_vec();
                if vec.is_empty() {
                    let cur = *deco_col.borrow();
                    Ok(Value::Integer(cur as i64))
                } else if vec.len() == 1 {
                    let col = value_to_i64(&vec[0])? as u32;
                    *deco_col.borrow_mut() = col;
                    Ok(Value::Nil)
                } else {
                    let r = (value_to_i64(&vec[0])? as u32).clamp(0, 255);
                    let g = if vec.len() >= 2 {
                        (value_to_i64(&vec[1])? as u32).clamp(0, 255)
                    } else {
                        255
                    };
                    let b = if vec.len() >= 3 {
                        (value_to_i64(&vec[2])? as u32).clamp(0, 255)
                    } else {
                        255
                    };
                    let a = if vec.len() >= 4 {
                        (value_to_i64(&vec[3])? as u32).clamp(0, 255)
                    } else {
                        255
                    };
                    let col = (a << 24) | (r << 16) | (g << 8) | b;
                    *deco_col.borrow_mut() = col;
                    Ok(Value::Nil)
                }
            }
        })?,
    )?;

    // sim.floodDeco(x, y, r, g, b, a)
    sim.set(
        "floodDeco",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (x, y, r, g, b, a): (i32, i32, i64, i64, i64, i64)| -> mlua::Result<()> {
                let r = (r as u32).clamp(0, 255);
                let g = (g as u32).clamp(0, 255);
                let b = (b as u32).clamp(0, 255);
                let a = (a as u32).clamp(0, 255);
                let colour = (a << 24) | (r << 16) | (g << 8) | b;

                let mut state = borrow_bridge(&bridge)?;
                state.dispatch(SimCommand::FloodDecorate { x, y, colour })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    Ok(())
}
