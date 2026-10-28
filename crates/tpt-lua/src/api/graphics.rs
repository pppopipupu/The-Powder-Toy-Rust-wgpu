//! The `graphics` table (and the classic `gfx` alias): `LuaGraphics.cpp`.
//!
//! TPT draws these primitives straight into the renderer's overlay surface.
//! This port records them in [`GraphicsOverlay`], which the application paints
//! over the viewport each frame; [`GraphicsOverlay::clear`] is the per-frame
//! reset. `graphics.WIDTH`/`HEIGHT` are the reference window size
//! (`SimulationConfig.h:5-27`: `RES = 612x384` plus `BARSIZE 17` and
//! `MENUSIZE 40`).

use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use mlua::{Lua, MultiValue, Table, Value};

/// `WINDOWW` (`SimulationConfig.h:27`).
pub const WINDOW_WIDTH: i32 = 612 + 17;
/// `WINDOWH` (`SimulationConfig.h:27`).
pub const WINDOW_HEIGHT: i32 = 384 + 40;

/// `FONT_H` (`graphics/FontReader.h:6`).
const FONT_HEIGHT: i32 = 12;
/// Advance per printable character.
///
/// TPT measures text with its proportional bitmap font (`FontReader`). This
/// port renders text with egui and has no copy of the bitmap table, so
/// `textSize` uses a fixed 6 px glyph plus the reference's 1 px spacing. The
/// value matches the average advance of the TPT font.
const GLYPH_ADVANCE: i32 = 7;

/// Axis-aligned rectangle in window pixels (`RectSized`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OverlayRect {
    /// Left edge.
    pub x: i32,
    /// Top edge.
    pub y: i32,
    /// Width.
    pub w: i32,
    /// Height.
    pub h: i32,
}

impl OverlayRect {
    /// Intersection with the window bounds (`Graphics::SwapClipRect`'s
    /// `clipRect &= video.Size().OriginRect()`, `Graphics.cpp:467-471`).
    fn clamped(self) -> Self {
        let x0 = self.x.max(0);
        let y0 = self.y.max(0);
        let x1 = (self.x + self.w).min(WINDOW_WIDTH);
        let y1 = (self.y + self.h).min(WINDOW_HEIGHT);
        Self {
            x: x0,
            y: y0,
            w: (x1 - x0).max(0),
            h: (y1 - y0).max(0),
        }
    }
}

/// One recorded drawing primitive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OverlayCommand {
    /// `BlendText`: text at the top-left corner, RGBA colour.
    Text {
        /// X position.
        x: i32,
        /// Y position.
        y: i32,
        /// Text, including TPT colour codes.
        text: String,
        /// RGBA colour.
        colour: [u8; 4],
    },
    /// `BlendPixel`.
    Pixel {
        /// X position.
        x: i32,
        /// Y position.
        y: i32,
        /// RGBA colour.
        colour: [u8; 4],
    },
    /// `DrawLine` (`alpha == 255`) or `BlendLine`.
    Line {
        /// Start X.
        x1: i32,
        /// Start Y.
        y1: i32,
        /// End X.
        x2: i32,
        /// End Y.
        y2: i32,
        /// RGBA colour.
        colour: [u8; 4],
    },
    /// `DrawRect`/`BlendRect` (`filled == false`) or
    /// `DrawFilledRect`/`BlendFilledRect` (`filled == true`).
    Rect {
        /// Left edge.
        x: i32,
        /// Top edge.
        y: i32,
        /// Width.
        w: i32,
        /// Height.
        h: i32,
        /// RGBA colour.
        colour: [u8; 4],
        /// Fill the rectangle instead of stroking its outline.
        filled: bool,
    },
    /// `BlendEllipse` (`filled == false`) or `BlendFilledEllipse`.
    Ellipse {
        /// Centre X.
        x: i32,
        /// Centre Y.
        y: i32,
        /// Horizontal radius (non-negative).
        rx: i32,
        /// Vertical radius (non-negative).
        ry: i32,
        /// RGBA colour.
        colour: [u8; 4],
        /// Fill the ellipse instead of stroking its outline.
        filled: bool,
    },
}

/// Recorded overlay layer for one Lua state.
#[derive(Clone, Debug)]
pub struct GraphicsOverlay {
    /// Primitives in call order.
    pub commands: Vec<OverlayCommand>,
    /// Current clip rectangle (`Graphics::clipRect`), clamped to the window.
    pub clip_rect: OverlayRect,
    /// Window width in pixels (`graphics.WIDTH`).
    pub width: i32,
    /// Window height in pixels (`graphics.HEIGHT`).
    pub height: i32,
}

impl Default for GraphicsOverlay {
    fn default() -> Self {
        Self {
            commands: Vec::new(),
            clip_rect: OverlayRect {
                x: 0,
                y: 0,
                w: WINDOW_WIDTH,
                h: WINDOW_HEIGHT,
            },
            width: WINDOW_WIDTH,
            height: WINDOW_HEIGHT,
        }
    }
}

impl GraphicsOverlay {
    /// Drops every recorded primitive. The application calls this at the start
    /// of each overlay pass; the clip rectangle is part of the graphics state
    /// and survives (TPT's `clipRect` is a member of `Graphics`, not per-frame
    /// state).
    pub fn clear(&mut self) {
        self.commands.clear();
    }
}

/// Shared handle to the overlay layer.
#[derive(Clone)]
pub struct GraphicsHandle(Rc<RefCell<GraphicsOverlay>>);

impl GraphicsHandle {
    /// Borrows the overlay.
    pub fn borrow(&self) -> Ref<'_, GraphicsOverlay> {
        self.0.borrow()
    }

    /// Borrows the overlay mutably.
    pub fn borrow_mut(&self) -> RefMut<'_, GraphicsOverlay> {
        self.0.borrow_mut()
    }
}

impl crate::LuaScriptingEngine {
    /// Returns the `graphics` overlay recorded by this Lua state.
    ///
    /// The application draws [`GraphicsOverlay::commands`] over the viewport
    /// and calls [`GraphicsOverlay::clear`] once per frame.
    pub fn graphics(&self) -> GraphicsHandle {
        self.lua
            .app_data_ref::<GraphicsHandle>()
            .expect("the graphics table is installed with the Lua state")
            .clone()
    }
}

/// Reads the overlay handle from the Lua app data.
fn graphics_handle(lua: &Lua) -> mlua::Result<GraphicsHandle> {
    lua.app_data_ref::<GraphicsHandle>()
        .map(|handle| handle.clone())
        .ok_or_else(|| mlua::Error::runtime("the graphics table is not installed"))
}

/// `luaL_optinteger`: a missing or nil argument uses `default`.
fn opt_int(value: Option<&Value>, default: i64) -> mlua::Result<i64> {
    match value {
        None | Some(Value::Nil) => Ok(default),
        Some(value) => super::value_to_i64(value),
    }
}

/// Clamps one colour channel to `0..=255` (`LuaGraphics.cpp:65-72`).
fn channel(value: Option<&Value>) -> mlua::Result<u8> {
    Ok(opt_int(value, 255)?.clamp(0, 255) as u8)
}

/// Reads the RGBA tail of a graphics call.
fn colour(values: &[Option<Value>]) -> mlua::Result<[u8; 4]> {
    Ok([
        channel(values.first().and_then(|value| value.as_ref()))?,
        channel(values.get(1).and_then(|value| value.as_ref()))?,
        channel(values.get(2).and_then(|value| value.as_ref()))?,
        channel(values.get(3).and_then(|value| value.as_ref()))?,
    ])
}

/// `graphics.textSize(text)` (`LuaGraphics.cpp:50-58`).
///
/// Mirrors `RasterDrawMethods::TextSize` (`RasterDrawMethodsImpl.h:342-374`):
/// the first line is `FONT_H - 2` tall, every newline adds `FONT_H`, colour
/// codes (`\x0F` + RGB) and the toggle bytes are skipped, and `\b` consumes
/// the following byte.
fn text_size(text: &str) -> (i32, i32) {
    let bytes = text.as_bytes();
    let mut size = (0, FONT_HEIGHT - 2);
    let mut current = 0;
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'\n' => {
                size.0 = size.0.max(current);
                size.1 += FONT_HEIGHT;
                current = 0;
            }
            0x0F => {
                if bytes.len() <= index + 3 {
                    break;
                }
                index += 3;
            }
            0x0E | 0x01 => {}
            0x08 => {
                if bytes.len() <= index + 1 {
                    break;
                }
                index += 1;
            }
            _ => current += GLYPH_ADVANCE,
        }
        index += 1;
    }
    size.0 = size.0.max(current);
    size
}

/// `textSize` (`LuaGraphics.cpp:50-58`).
fn text_size_lua(lua: &Lua, text: Option<String>) -> mlua::Result<(i64, i64)> {
    let _ = lua;
    let (width, height) = text_size(&text.unwrap_or_default());
    Ok((width as i64, height as i64))
}

/// `drawText(x, y, text [, r, g, b, a])` (`LuaGraphics.cpp:60-84`).
fn draw_text(lua: &Lua, args: MultiValue) -> mlua::Result<()> {
    let mut values = args.into_iter();
    let x = values
        .next()
        .map(|value| super::value_to_i64(&value))
        .transpose()?
        .unwrap_or(0) as i32;
    let y = values
        .next()
        .map(|value| super::value_to_i64(&value))
        .transpose()?
        .unwrap_or(0) as i32;
    let text = match values.next() {
        Some(value) => super::value_to_string(&value)?,
        None => String::new(),
    };
    let tail: Vec<Option<Value>> = values.map(Some).collect();
    let colour = colour(&tail)?;
    graphics_handle(lua)?.borrow_mut().commands.push(OverlayCommand::Text {
        x,
        y,
        text,
        colour,
    });
    Ok(())
}

/// `drawPixel([x, y, r, g, b, a])` (`LuaGraphics.cpp:86-105`).
fn draw_pixel(lua: &Lua, args: MultiValue) -> mlua::Result<()> {
    let values: Vec<Option<Value>> = args.into_iter().map(Some).collect();
    let get = |index: usize, default: i64| -> mlua::Result<i32> {
        Ok(opt_int(values.get(index).and_then(|value| value.as_ref()), default)? as i32)
    };
    let x = get(0, 0)?;
    let y = get(1, 0)?;
    let colour = colour(&values[2..])?;
    graphics_handle(lua)?.borrow_mut().commands.push(OverlayCommand::Pixel {
        x,
        y,
        colour,
    });
    Ok(())
}

/// `drawLine(x1, y1, x2, y2 [, r, g, b, a])` (`LuaGraphics.cpp:107-139`).
fn draw_line(lua: &Lua, args: MultiValue) -> mlua::Result<()> {
    let values: Vec<Option<Value>> = args.into_iter().map(Some).collect();
    let get = |index: usize| -> mlua::Result<i32> {
        Ok(opt_int(values.get(index).and_then(|value| value.as_ref()), 0)? as i32)
    };
    let (x1, y1, x2, y2) = (get(0)?, get(1)?, get(2)?, get(3)?);
    let colour = colour(&values[4..])?;
    graphics_handle(lua)?.borrow_mut().commands.push(OverlayCommand::Line {
        x1,
        y1,
        x2,
        y2,
        colour,
    });
    Ok(())
}

/// `drawRect`/`fillRect` (`LuaGraphics.cpp:141-211`).
fn rectangle(lua: &Lua, args: MultiValue, filled: bool) -> mlua::Result<()> {
    let values: Vec<Option<Value>> = args.into_iter().map(Some).collect();
    let get = |index: usize| -> mlua::Result<i32> {
        Ok(opt_int(values.get(index).and_then(|value| value.as_ref()), 0)? as i32)
    };
    let (x, y, w, h) = (get(0)?, get(1)?, get(2)?, get(3)?);
    let colour = colour(&values[4..])?;
    graphics_handle(lua)?.borrow_mut().commands.push(OverlayCommand::Rect {
        x,
        y,
        w,
        h,
        colour,
        filled,
    });
    Ok(())
}

/// `drawCircle`/`fillCircle` (`LuaGraphics.cpp:213-274`).
fn circle(lua: &Lua, args: MultiValue, filled: bool) -> mlua::Result<()> {
    let values: Vec<Option<Value>> = args.into_iter().map(Some).collect();
    let get = |index: usize| -> mlua::Result<i32> {
        Ok(opt_int(values.get(index).and_then(|value| value.as_ref()), 0)? as i32)
    };
    let (x, y) = (get(0)?, get(1)?);
    let rx = get(2)?.abs();
    let ry = get(3)?.abs();
    let colour = colour(&values[4..])?;
    graphics_handle(lua)?.borrow_mut().commands.push(OverlayCommand::Ellipse {
        x,
        y,
        rx,
        ry,
        colour,
        filled,
    });
    Ok(())
}

/// `graphics.getColors(color)` (`LuaGraphics.cpp:276-290`).
///
/// Returns `r, g, b, a` (the reference's order).
fn get_colors(lua: &Lua, value: Value) -> mlua::Result<(i64, i64, i64, i64)> {
    let number = match value {
        Value::Nil => 0.0,
        value => super::value_to_f64(&value)?,
    };
    // `int32Truncate` (`LuaGraphics.cpp:40-47`): truncate and wrap values at or
    // above 2^31 into the signed range.
    let mut number = number.trunc();
    if number >= 0x8000_0000i64 as f64 {
        number -= 0x1_0000_0000i64 as f64;
    }
    let word = number as i64 as u32;
    let _ = lua;
    Ok((
        ((word >> 16) & 0xFF) as i64,
        ((word >> 8) & 0xFF) as i64,
        (word & 0xFF) as i64,
        (word >> 24) as i64,
    ))
}

/// `graphics.getHexColor(r, g, b [, a])` (`LuaGraphics.cpp:292-304`).
fn get_hex_color(lua: &Lua, args: MultiValue) -> mlua::Result<i64> {
    let _ = lua;
    let mut values = args.into_iter();
    let get = |value: Option<Value>| -> mlua::Result<i64> {
        opt_int(value.as_ref(), 0)
    };
    let r = get(values.next())?;
    let g = get(values.next())?;
    let b = get(values.next())?;
    let a = get(values.next())?;
    Ok((a << 24).wrapping_add(r << 16).wrapping_add(g << 8).wrapping_add(b))
}

/// `graphics.setClipRect([x, y, w, h])` (`LuaGraphics.cpp:306-332`).
///
/// `Graphics::SwapClipRect` swaps the new rectangle with the current one and
/// clamps the result to the window, then returns the *old* rectangle.
fn set_clip_rect(lua: &Lua, args: MultiValue) -> mlua::Result<(i64, i64, i64, i64)> {
    let mut values = args.into_iter();
    let get = |value: Option<Value>, default: i32| -> mlua::Result<i32> {
        Ok(opt_int(value.as_ref(), default as i64)? as i32)
    };
    let x = get(values.next(), 0)?;
    let y = get(values.next(), 0)?;
    let w = get(values.next(), WINDOW_WIDTH)?;
    let h = get(values.next(), WINDOW_HEIGHT)?;
    let handle = graphics_handle(lua)?;
    let mut overlay = handle.borrow_mut();
    let old = overlay.clip_rect;
    overlay.clip_rect = OverlayRect { x, y, w, h }.clamped();
    Ok((old.x as i64, old.y as i64, old.w as i64, old.h as i64))
}

/// Builds the `graphics` table (`LuaGraphics::Open`, `LuaGraphics.cpp:334-357`).
pub(crate) fn build_graphics_table(lua: &Lua) -> mlua::Result<Table> {
    let handle = GraphicsHandle(Rc::new(RefCell::new(GraphicsOverlay::default())));
    lua.set_app_data(handle);

    let graphics = lua.create_table()?;
    graphics.set("textSize", lua.create_function(text_size_lua)?)?;
    graphics.set("drawText", lua.create_function(draw_text)?)?;
    graphics.set("drawPixel", lua.create_function(draw_pixel)?)?;
    graphics.set("drawLine", lua.create_function(draw_line)?)?;
    graphics.set(
        "drawRect",
        lua.create_function(|lua, args: MultiValue| rectangle(lua, args, false))?,
    )?;
    graphics.set(
        "fillRect",
        lua.create_function(|lua, args: MultiValue| rectangle(lua, args, true))?,
    )?;
    graphics.set(
        "drawCircle",
        lua.create_function(|lua, args: MultiValue| circle(lua, args, false))?,
    )?;
    graphics.set(
        "fillCircle",
        lua.create_function(|lua, args: MultiValue| circle(lua, args, true))?,
    )?;
    graphics.set("getColors", lua.create_function(get_colors)?)?;
    graphics.set("getHexColor", lua.create_function(get_hex_color)?)?;
    graphics.set("setClipRect", lua.create_function(set_clip_rect)?)?;
    graphics.set("WIDTH", WINDOW_WIDTH)?;
    graphics.set("HEIGHT", WINDOW_HEIGHT)?;
    Ok(graphics)
}
