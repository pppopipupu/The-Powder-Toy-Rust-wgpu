//! Reading and writing `prefs.json`, plus the per-section key mapping.

use tpt_core::commands::DecoMode;
use tpt_core::constants::{MAX_PRESSURE, MAX_TEMP, MIN_PRESSURE, MIN_TEMP};

use crate::app_state::PropertyKind;

use super::schema::{
    get_bool, get_f32, get_f32_in_range, get_f64, get_i32, get_string, get_u8, get_u32,
    get_u32_in_range, keys, set_bool, set_f32, set_f64, set_i32, set_string, set_u32,
};
use super::{
    FpsLimit, MAX_SANE_DRAW_FPS, MAX_UNDO_HISTORY_LIMIT, MAX_VOXEL_AIR_ITERATIONS,
    MAX_WINDOW_SCALE, MIN_SANE_DRAW_FPS, MIN_VOXEL_AIR_ITERATIONS, MIN_WINDOW_SCALE, PREFS_FILE,
    Prefs, snap_voxel_ao_rays, snap_voxel_render_scale,
};

/// Largest absolute edge velocity accepted by the air solver.
const MAX_EDGE_VELOCITY: f32 = 1.0e4;
/// Largest saved window dimension, in physical pixels. Anything bigger is a
/// hand-edited file; the platform layer clamps it to the monitor anyway.
const MAX_WINDOW_DIMENSION: u32 = 32_768;
/// Guard against pathological nesting in a hand-edited prefs file.
const MAX_JSON_DEPTH: usize = 32;

impl Prefs {
    /// Loads `prefs.json` from the working directory.
    ///
    /// A missing or malformed file yields [`Self::default`]; individual
    /// out-of-range values are ignored in favour of their defaults.
    pub fn load() -> Self {
        let mut prefs = Self::default();
        let text = match std::fs::read_to_string(PREFS_FILE) {
            Ok(text) => text,
            Err(err) => {
                if err.kind() != std::io::ErrorKind::NotFound {
                    eprintln!("[prefs] could not read '{PREFS_FILE}': {err}");
                }
                return prefs;
            }
        };

        let Some(JsonValue::Object(entries)) =
            JsonParser::new(text.strip_prefix('\u{feff}').unwrap_or(text.as_str()))
                .parse_document()
        else {
            eprintln!("[prefs] ignoring malformed '{PREFS_FILE}'");
            return prefs;
        };

        prefs.read_entries(&entries);
        // Retired keys are dropped instead of being preserved verbatim, so a
        // stale setting cannot outlive the code that read it.
        prefs.entries = entries
            .into_iter()
            .filter(|(key, _)| !keys::RETIRED.contains(&key.as_str()))
            .collect();
        prefs
    }

    /// Writes the current preferences to `prefs.json`.
    ///
    /// The write goes through a temporary file so a crash mid-save cannot
    /// truncate an existing configuration. Failures are logged, never fatal.
    pub fn save(&self) {
        let mut entries = self.entries.clone();
        self.write_entries(&mut entries);

        let mut text = String::with_capacity(2048);
        write_json(&JsonValue::Object(entries), &mut text, 0);
        text.push('\n');

        let temp = format!("{PREFS_FILE}.tmp");
        if std::fs::write(&temp, text.as_bytes()).is_ok() {
            if std::fs::rename(&temp, PREFS_FILE).is_ok() {
                return;
            }
            let _ = std::fs::remove_file(&temp);
        }
        if let Err(err) = std::fs::write(PREFS_FILE, text.as_bytes()) {
            eprintln!("[prefs] could not write '{PREFS_FILE}': {err}");
        }
    }
    fn read_entries(&mut self, entries: &[(String, JsonValue)]) {
        self.read_simulation_entries(entries);
        self.read_renderer_entries(entries);
        self.read_tool_entries(entries);
        self.read_application_entries(entries);
    }

    /// `Simulation.*` keys (heat, air, gravity, edge and decoration space).
    fn read_simulation_entries(&mut self, entries: &[(String, JsonValue)]) {
        self.newtonian_gravity = get_bool(entries, keys::NEWTONIAN_GRAVITY, self.newtonian_gravity);
        self.ambient_heat = get_bool(entries, keys::AMBIENT_HEAT, self.ambient_heat);
        self.ambient_air_temp = get_f32_in_range(
            entries,
            keys::AMBIENT_AIR_TEMP,
            self.ambient_air_temp,
            MIN_TEMP,
            MAX_TEMP,
        );
        self.edge_pressure = get_f32_in_range(
            entries,
            keys::EDGE_PRESSURE,
            self.edge_pressure,
            MIN_PRESSURE,
            MAX_PRESSURE,
        );
        self.edge_velocity_x = get_f32_in_range(
            entries,
            keys::EDGE_VELOCITY_X,
            self.edge_velocity_x,
            -MAX_EDGE_VELOCITY,
            MAX_EDGE_VELOCITY,
        );
        self.edge_velocity_y = get_f32_in_range(
            entries,
            keys::EDGE_VELOCITY_Y,
            self.edge_velocity_y,
            -MAX_EDGE_VELOCITY,
            MAX_EDGE_VELOCITY,
        );
        self.vorticity_coeff = get_f32_in_range(
            entries,
            keys::VORTICITY_COEFF,
            self.vorticity_coeff,
            0.0,
            1.0,
        );
        self.convection_mode =
            get_u32_in_range(entries, keys::CONVECTION_MODE, self.convection_mode, 0, 2);
        self.edge_mode = get_u32_in_range(entries, keys::EDGE_MODE, self.edge_mode, 0, 2);
        self.include_pressure = get_bool(entries, keys::INCLUDE_PRESSURE, self.include_pressure);
        self.undo_history_limit = get_u32_in_range(
            entries,
            keys::UNDO_HISTORY_LIMIT,
            self.undo_history_limit,
            1,
            MAX_UNDO_HISTORY_LIMIT,
        );
        self.pretty_powder = get_bool(entries, keys::PRETTY_POWDER, self.pretty_powder);
        self.mod_elements_enabled = get_bool(
            entries,
            keys::ENABLE_MOD_ELEMENTS,
            self.mod_elements_enabled,
        );

        // Not persisted by the C++ game, but useful for this front-end.
        self.heat_enable = get_bool(entries, keys::HEAT_SIMULATION, self.heat_enable);
        self.water_equal_test = get_bool(entries, keys::WATER_EQUALISATION, self.water_equal_test);
        self.air_mode = get_u32_in_range(entries, keys::AIR_MODE, self.air_mode, 0, 4);
        self.gravity_mode = get_u32_in_range(entries, keys::GRAVITY_MODE, self.gravity_mode, 0, 3);
        self.custom_gravity_x = get_f32_in_range(
            entries,
            keys::CUSTOM_GRAVITY_X,
            self.custom_gravity_x,
            -MAX_EDGE_VELOCITY,
            MAX_EDGE_VELOCITY,
        );
        self.custom_gravity_y = get_f32_in_range(
            entries,
            keys::CUSTOM_GRAVITY_Y,
            self.custom_gravity_y,
            -MAX_EDGE_VELOCITY,
            MAX_EDGE_VELOCITY,
        );
    }

    /// `Renderer.*` keys (display, render, colour and fire settings).
    fn read_renderer_entries(&mut self, entries: &[(String, JsonValue)]) {
        self.display_mode = get_u32(entries, keys::DISPLAY_MODE, self.display_mode);
        self.render_mode = get_u32(entries, keys::RENDER_MODE, self.render_mode);
        self.colour_mode = get_u32(entries, keys::COLOUR_MODE, self.colour_mode);
        self.decorations = get_bool(entries, keys::DECORATIONS, self.decorations);
        self.gravity_field = get_bool(entries, keys::GRAVITY_FIELD, self.gravity_field);
        self.grid_checkerboard = get_bool(entries, keys::GRID_CHECKERBOARD, self.grid_checkerboard);
        self.temperature_scale = get_u32_in_range(
            entries,
            keys::TEMPERATURE_SCALE,
            self.temperature_scale,
            0,
            2,
        );
        self.fire_intensity =
            get_f32_in_range(entries, keys::FIRE_INTENSITY, self.fire_intensity, 0.0, 4.0);
    }

    /// Tool keys (`MouseClickRequired`, brush and decoration colour).
    fn read_tool_entries(&mut self, entries: &[(String, JsonValue)]) {
        self.mouse_click_required = get_bool(
            entries,
            keys::MOUSE_CLICK_REQUIRED,
            self.mouse_click_required,
        );
        self.deco_red = get_u8(entries, keys::DECORATION_RED, self.deco_red);
        self.deco_green = get_u8(entries, keys::DECORATION_GREEN, self.deco_green);
        self.deco_blue = get_u8(entries, keys::DECORATION_BLUE, self.deco_blue);
        self.deco_alpha = get_u8(entries, keys::DECORATION_ALPHA, self.deco_alpha);
        self.deco_mode = get_u32_in_range(
            entries,
            keys::DECORATION_MODE,
            self.deco_mode,
            0,
            DecoMode::ALL.len() as u32 - 1,
        );

        // Property tool and spawn tool configuration, plus the two bound tools
        // (`Default*Tool` in the reference game is session-only; the port keeps
        // the selection across restarts).
        self.prop_field = get_u32_in_range(
            entries,
            keys::PROP_TYPE,
            self.prop_field,
            0,
            PropertyKind::MAX_INDEX,
        );
        self.prop_value = get_f32(entries, keys::PROP_VALUE, self.prop_value);
        self.prop_fill = get_bool(entries, keys::PROP_FILL, self.prop_fill);
        self.spawn_element = get_u32(entries, keys::TOOL_SPAWN_ELEMENT, self.spawn_element);
        self.language = crate::i18n::Language::from_code(&get_string(
            entries,
            keys::LANGUAGE,
            self.language.code(),
        ));
        self.primary_tool = get_string(entries, keys::TOOL_PRIMARY, &self.primary_tool);
        self.secondary_tool = get_string(entries, keys::TOOL_SECONDARY, &self.secondary_tool);
    }

    /// Application keys (framerate caps and window-level settings).
    fn read_application_entries(&mut self, entries: &[(String, JsonValue)]) {
        self.fps_limit = match get_f64(entries, keys::FPS_LIMIT) {
            Some(value) => FpsLimit::from_pref_value(value),
            None => self.fps_limit,
        };
        // TPT stores "follow display" as -1; the port has no display-derived
        // mode, so anything below the sane floor keeps the 240 fps default.
        self.draw_fps = match get_f64(entries, keys::DRAW_LIMIT) {
            Some(value) if value >= MIN_SANE_DRAW_FPS as f64 => {
                (value.round() as u32).min(MAX_SANE_DRAW_FPS)
            }
            Some(_) | None => self.draw_fps,
        };
        self.blurry_scaling = get_bool(entries, keys::BLURRY_SCALING, self.blurry_scaling);
        self.resizable = get_bool(entries, keys::RESIZABLE, self.resizable);
        self.fullscreen = get_bool(entries, keys::FULLSCREEN, self.fullscreen);
        self.native_clipboard =
            get_bool(entries, keys::NATIVE_CLIPBOARD, self.native_clipboard);
        self.grave_exits_console =
            get_bool(entries, keys::GRAVE_EXITS_CONSOLE, self.grave_exits_console);
        self.game_mode = crate::app_state::GameMode::from_key(&get_string(
            entries,
            keys::GAME_MODE,
            self.game_mode.key(),
        ));
        // `PowderToy.cpp:44-45`: the position saved on the previous clean exit.
        self.window_x = get_i32(entries, keys::WINDOW_X).or(self.window_x);
        self.window_y = get_i32(entries, keys::WINDOW_Y).or(self.window_y);
        // `PowderToySDL.cpp:212-216`: a resizable window keeps its previous
        // size. Zero means "no size saved yet".
        let saved_width = get_u32_in_range(entries, keys::WINDOW_WIDTH, 0, 1, MAX_WINDOW_DIMENSION);
        if saved_width != 0 {
            self.window_width = Some(saved_width);
        }
        let saved_height =
            get_u32_in_range(entries, keys::WINDOW_HEIGHT, 0, 1, MAX_WINDOW_DIMENSION);
        if saved_height != 0 {
            self.window_height = Some(saved_height);
        }
        // `OptionsView.cpp:340-376`: the window scale and integer-scaling
        // switches; both drive the simulation viewport blit.
        self.scale = get_u32_in_range(
            entries,
            keys::SCALE,
            self.scale,
            MIN_WINDOW_SCALE,
            MAX_WINDOW_SCALE,
        );
        // `PowderToy.cpp:386-399` and `OptionsView.cpp:471-485`.
        self.voxel_render_scale = snap_voxel_render_scale(get_f32(
            entries,
            keys::VOXEL_RENDER_SCALE,
            self.voxel_render_scale,
        ));
        self.voxel_ao_rays =
            snap_voxel_ao_rays(get_u32(entries, keys::VOXEL_AO_RAYS, self.voxel_ao_rays));
        self.voxel_shadows = get_bool(entries, keys::VOXEL_SHADOWS, self.voxel_shadows);
        self.voxel_air_iterations = get_u32_in_range(
            entries,
            keys::VOXEL_AIR_ITERATIONS,
            self.voxel_air_iterations,
            MIN_VOXEL_AIR_ITERATIONS,
            MAX_VOXEL_AIR_ITERATIONS,
        );
    }

    fn write_entries(&self, entries: &mut Vec<(String, JsonValue)>) {
        set_bool(entries, keys::NEWTONIAN_GRAVITY, self.newtonian_gravity);
        set_bool(entries, keys::AMBIENT_HEAT, self.ambient_heat);
        set_f32(entries, keys::AMBIENT_AIR_TEMP, self.ambient_air_temp);
        set_f32(entries, keys::EDGE_PRESSURE, self.edge_pressure);
        set_f32(entries, keys::EDGE_VELOCITY_X, self.edge_velocity_x);
        set_f32(entries, keys::EDGE_VELOCITY_Y, self.edge_velocity_y);
        set_f32(entries, keys::VORTICITY_COEFF, self.vorticity_coeff);
        set_u32(entries, keys::CONVECTION_MODE, self.convection_mode);
        set_u32(entries, keys::EDGE_MODE, self.edge_mode);
        set_bool(entries, keys::INCLUDE_PRESSURE, self.include_pressure);
        set_u32(entries, keys::UNDO_HISTORY_LIMIT, self.undo_history_limit);
        set_bool(entries, keys::PRETTY_POWDER, self.pretty_powder);
        set_bool(
            entries,
            keys::ENABLE_MOD_ELEMENTS,
            self.mod_elements_enabled,
        );
        set_bool(entries, keys::HEAT_SIMULATION, self.heat_enable);
        set_bool(entries, keys::WATER_EQUALISATION, self.water_equal_test);
        set_u32(entries, keys::AIR_MODE, self.air_mode);
        set_u32(entries, keys::GRAVITY_MODE, self.gravity_mode);
        set_f32(entries, keys::CUSTOM_GRAVITY_X, self.custom_gravity_x);
        set_f32(entries, keys::CUSTOM_GRAVITY_Y, self.custom_gravity_y);

        set_u32(entries, keys::DISPLAY_MODE, self.display_mode);
        set_u32(entries, keys::RENDER_MODE, self.render_mode);
        set_u32(entries, keys::COLOUR_MODE, self.colour_mode);
        set_bool(entries, keys::DECORATIONS, self.decorations);
        set_bool(entries, keys::GRAVITY_FIELD, self.gravity_field);
        set_bool(entries, keys::GRID_CHECKERBOARD, self.grid_checkerboard);
        set_u32(entries, keys::TEMPERATURE_SCALE, self.temperature_scale);
        set_f32(entries, keys::FIRE_INTENSITY, self.fire_intensity);

        set_bool(
            entries,
            keys::MOUSE_CLICK_REQUIRED,
            self.mouse_click_required,
        );
        set_u32(entries, keys::DECORATION_RED, self.deco_red as u32);
        set_u32(entries, keys::DECORATION_GREEN, self.deco_green as u32);
        set_u32(entries, keys::DECORATION_BLUE, self.deco_blue as u32);
        set_u32(entries, keys::DECORATION_ALPHA, self.deco_alpha as u32);
        set_u32(entries, keys::DECORATION_MODE, self.deco_mode);
        set_u32(entries, keys::PROP_TYPE, self.prop_field);
        set_f32(entries, keys::PROP_VALUE, self.prop_value);
        set_bool(entries, keys::PROP_FILL, self.prop_fill);
        set_u32(entries, keys::TOOL_SPAWN_ELEMENT, self.spawn_element);
        set_string(entries, keys::LANGUAGE, self.language.code());
        set_string(entries, keys::TOOL_PRIMARY, &self.primary_tool);
        set_string(entries, keys::TOOL_SECONDARY, &self.secondary_tool);

        set_f64(entries, keys::FPS_LIMIT, self.fps_limit.to_pref_value());
        set_f64(entries, keys::DRAW_LIMIT, self.draw_fps as f64);
        set_bool(entries, keys::BLURRY_SCALING, self.blurry_scaling);
        set_bool(entries, keys::RESIZABLE, self.resizable);
        set_bool(entries, keys::FULLSCREEN, self.fullscreen);
        set_bool(entries, keys::NATIVE_CLIPBOARD, self.native_clipboard);
        set_bool(entries, keys::GRAVE_EXITS_CONSOLE, self.grave_exits_console);
        set_string(entries, keys::GAME_MODE, self.game_mode.key());
        if let Some(x) = self.window_x {
            set_i32(entries, keys::WINDOW_X, x);
        }
        if let Some(y) = self.window_y {
            set_i32(entries, keys::WINDOW_Y, y);
        }
        if let Some(width) = self.window_width {
            set_u32(entries, keys::WINDOW_WIDTH, width);
        }
        if let Some(height) = self.window_height {
            set_u32(entries, keys::WINDOW_HEIGHT, height);
        }
        set_u32(entries, keys::SCALE, self.scale);
        set_f32(entries, keys::VOXEL_RENDER_SCALE, self.voxel_render_scale);
        set_u32(entries, keys::VOXEL_AO_RAYS, self.voxel_ao_rays);
        set_bool(entries, keys::VOXEL_SHADOWS, self.voxel_shadows);
        set_u32(
            entries,
            keys::VOXEL_AIR_ITERATIONS,
            self.voxel_air_iterations,
        );
    }
}

/// One JSON value. Only the shapes needed by `prefs.json` are represented, but
/// arrays and nested objects are parsed so foreign prefs survive a rewrite.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

fn write_json(value: &JsonValue, out: &mut String, indent: usize) {
    match value {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(true) => out.push_str("true"),
        JsonValue::Bool(false) => out.push_str("false"),
        JsonValue::Number(number) => {
            if number.is_finite() {
                out.push_str(&format!("{number}"));
            } else {
                out.push('0');
            }
        }
        JsonValue::String(text) => write_json_string(text, out),
        JsonValue::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (index, item) in items.iter().enumerate() {
                push_indent(out, indent + 1);
                write_json(item, out, indent + 1);
                if index + 1 < items.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            push_indent(out, indent);
            out.push(']');
        }
        JsonValue::Object(entries) => {
            if entries.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (index, (key, entry)) in entries.iter().enumerate() {
                push_indent(out, indent + 1);
                write_json_string(key, out);
                out.push_str(": ");
                write_json(entry, out, indent + 1);
                if index + 1 < entries.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            push_indent(out, indent);
            out.push('}');
        }
    }
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("  ");
    }
}

fn write_json_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            ch if ch.is_control() => {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }

    fn parse_document(&mut self) -> Option<JsonValue> {
        self.skip_whitespace();
        let value = self.parse_value(0)?;
        self.skip_whitespace();
        if self.pos == self.bytes.len() {
            Some(value)
        } else {
            None
        }
    }

    fn parse_value(&mut self, depth: usize) -> Option<JsonValue> {
        if depth > MAX_JSON_DEPTH {
            return None;
        }
        self.skip_whitespace();
        match *self.bytes.get(self.pos)? {
            b'{' => self.parse_object(depth),
            b'[' => self.parse_array(depth),
            b'"' => self.parse_string().map(JsonValue::String),
            b't' => {
                self.expect_literal(b"true")?;
                Some(JsonValue::Bool(true))
            }
            b'f' => {
                self.expect_literal(b"false")?;
                Some(JsonValue::Bool(false))
            }
            b'n' => {
                self.expect_literal(b"null")?;
                Some(JsonValue::Null)
            }
            b'-' | b'0'..=b'9' => self.parse_number().map(JsonValue::Number),
            _ => None,
        }
    }

    fn parse_object(&mut self, depth: usize) -> Option<JsonValue> {
        self.pos += 1; // '{'
        let mut entries = Vec::new();
        self.skip_whitespace();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Some(JsonValue::Object(entries));
        }
        loop {
            self.skip_whitespace();
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.bytes.get(self.pos) != Some(&b':') {
                return None;
            }
            self.pos += 1;
            let value = self.parse_value(depth + 1)?;
            entries.push((key, value));
            self.skip_whitespace();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Some(JsonValue::Object(entries));
                }
                _ => return None,
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Option<JsonValue> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Some(JsonValue::Array(items));
        }
        loop {
            let value = self.parse_value(depth + 1)?;
            items.push(value);
            self.skip_whitespace();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Some(JsonValue::Array(items));
                }
                _ => return None,
            }
        }
    }

    fn parse_string(&mut self) -> Option<String> {
        if self.bytes.get(self.pos) != Some(&b'"') {
            return None;
        }
        self.pos += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let byte = *self.bytes.get(self.pos)?;
            self.pos += 1;
            match byte {
                b'"' => return String::from_utf8(out).ok(),
                b'\\' => {
                    let escape = *self.bytes.get(self.pos)?;
                    self.pos += 1;
                    match escape {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0C),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let first = self.parse_hex4()?;
                            let code = if (0xD800..=0xDBFF).contains(&first) {
                                if self.bytes.get(self.pos) != Some(&b'\\') {
                                    return None;
                                }
                                self.pos += 1;
                                if self.bytes.get(self.pos) != Some(&b'u') {
                                    return None;
                                }
                                self.pos += 1;
                                let second = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&second) {
                                    return None;
                                }
                                0x1_0000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                            } else {
                                first
                            };
                            let ch = char::from_u32(code)?;
                            let mut buffer = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes());
                        }
                        _ => return None,
                    }
                }
                0x00..=0x1F => return None,
                other => out.push(other),
            }
        }
    }

    fn parse_hex4(&mut self) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..4 {
            let byte = *self.bytes.get(self.pos)?;
            self.pos += 1;
            let digit = match byte {
                b'0'..=b'9' => (byte - b'0') as u32,
                b'a'..=b'f' => (byte - b'a' + 10) as u32,
                b'A'..=b'F' => (byte - b'A' + 10) as u32,
                _ => return None,
            };
            value = value * 16 + digit;
        }
        Some(value)
    }

    fn parse_number(&mut self) -> Option<f64> {
        let start = self.pos;
        while let Some(&byte) = self.bytes.get(self.pos) {
            match byte {
                b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9' => self.pos += 1,
                _ => break,
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).ok()?;
        text.parse::<f64>().ok().filter(|value| value.is_finite())
    }

    fn expect_literal(&mut self, literal: &[u8]) -> Option<()> {
        let end = self.pos + literal.len();
        if self.bytes.get(self.pos..end)? == literal {
            self.pos = end;
            Some(())
        } else {
            None
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(&byte) = self.bytes.get(self.pos) {
            if matches!(byte, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::schema::find_entry;
    use super::*;

    #[test]
    fn reads_known_keys_and_preserves_unknown_ones() {
        let text = r#"{
            "Simulation.NewtonianGravity": true,
            "Simulation.AmbientAirTemp": 300.5,
            "Renderer.DisplayMode": 4,
            "Decoration.Red": 12,
            "Prop.Type": 2,
            "Prop.Value": 5,
            "Prop.Value2": -3,
            "Tool.SpawnElement": 25,
            "Tool.Primary": "element:17",
            "Tool.Secondary": "sim:pgrv",
            "FpsLimit": 2,
            "DrawLimit": -1,
            "Something.Else": [1, 2, 3]
        }"#;

        let Some(JsonValue::Object(entries)) = JsonParser::new(text).parse_document() else {
            panic!("prefs sample did not parse");
        };
        let mut prefs = Prefs::default();
        prefs.read_entries(&entries);
        prefs.entries = entries;

        assert!(prefs.newtonian_gravity);
        assert!((prefs.ambient_air_temp - 300.5).abs() < 1e-3);
        assert_eq!(prefs.display_mode, 4);
        assert_eq!(prefs.deco_red, 12);
        assert_eq!(prefs.prop_field, 2);
        assert_eq!(prefs.prop_value, 5.0);
        assert_eq!(prefs.spawn_element, 25);
        assert_eq!(prefs.primary_tool, "element:17");
        assert_eq!(prefs.secondary_tool, "sim:pgrv");
        assert_eq!(prefs.fps_limit, FpsLimit::Uncapped);
        assert_eq!(prefs.draw_fps, crate::prefs::DEFAULT_DRAW_FPS);

        let mut rewritten = prefs.entries.clone();
        prefs.write_entries(&mut rewritten);
        let mut serialized = String::new();
        write_json(&JsonValue::Object(rewritten), &mut serialized, 0);

        let Some(JsonValue::Object(round_tripped)) = JsonParser::new(&serialized).parse_document()
        else {
            panic!("rewritten prefs did not parse");
        };
        assert!(
            round_tripped
                .iter()
                .any(|(key, _)| key.as_str() == "Something.Else")
        );
        match find_entry(&round_tripped, "Renderer.DisplayMode") {
            Some(JsonValue::Number(value)) => assert_eq!(*value, 4.0),
            other => panic!("unexpected display mode entry: {other:?}"),
        }
        assert_eq!(
            find_entry(&round_tripped, "Tool.Primary"),
            Some(&JsonValue::String(String::from("element:17")))
        );
        match find_entry(&round_tripped, "Prop.Value2") {
            Some(JsonValue::Number(value)) => assert_eq!(*value, -3.0),
            other => panic!("unexpected property value entry: {other:?}"),
        }
        // Out-of-range `Prop.Type` values fall back to `tmp`.
        let entries = vec![(String::from("Prop.Type"), JsonValue::Number(99.0))];
        let mut clamped = Prefs::default();
        clamped.read_entries(&entries);
        assert_eq!(clamped.prop_field, 0);
    }

    /// `Prop.Type` addresses every property the property window offers, and a
    /// value past the last one still falls back to `tmp`.
    #[test]
    fn property_field_round_trips_every_property_kind() {
        use crate::app_state::PropertyKind;

        for kind in PropertyKind::ALL {
            let entries = vec![(
                String::from("Prop.Type"),
                JsonValue::Number(kind.index() as f64),
            )];
            let mut prefs = Prefs::default();
            prefs.read_entries(&entries);
            assert_eq!(prefs.prop_field, kind.index(), "{kind:?}");
            assert_eq!(PropertyKind::from_index(prefs.prop_field), kind);
        }

        let out_of_range = vec![(
            String::from("Prop.Type"),
            JsonValue::Number((PropertyKind::MAX_INDEX + 1) as f64),
        )];
        let mut prefs = Prefs::default();
        prefs.read_entries(&out_of_range);
        assert_eq!(prefs.prop_field, 0);
    }

    #[test]
    fn corrupt_json_falls_back_to_defaults() {
        assert!(
            JsonParser::new("{ this is not json")
                .parse_document()
                .is_none()
        );
        assert!(JsonParser::new("{\"a\": }").parse_document().is_none());
    }

    /// Every config-backed key that the merged settings UI exposes must survive
    /// a write/read cycle unchanged (TPT's `OptionsView`/`RenderView` keys).
    #[test]
    fn round_trips_simulation_and_renderer_keys() {
        use tpt_core::constants::{
            COLOUR_DEST, DISPLAY_AIRP, DISPLAY_EFFE, RENDER_BASC, RENDER_BLOB, TEMPSCALE_KELVIN,
        };

        let prefs = Prefs {
            heat_enable: false,
            newtonian_gravity: true,
            ambient_heat: true,
            water_equal_test: true,
            air_mode: 4, // No update
            ambient_air_temp: 512.0,
            edge_pressure: -3.5,
            edge_velocity_x: 12.5,
            edge_velocity_y: -0.25,
            vorticity_coeff: 0.75,
            convection_mode: 1, // Legacy
            gravity_mode: 3,    // Custom
            custom_gravity_x: 0.5,
            custom_gravity_y: -1.0,
            edge_mode: 2, // Loop
            pretty_powder: true,
            mod_elements_enabled: false,
            display_mode: DISPLAY_EFFE | DISPLAY_AIRP,
            render_mode: RENDER_BASC | RENDER_BLOB,
            colour_mode: COLOUR_DEST,
            fire_intensity: 2.5,
            decorations: false,
            grid_checkerboard: true,
            temperature_scale: TEMPSCALE_KELVIN,
            ..Prefs::default()
        };

        let mut entries = Vec::new();
        prefs.write_entries(&mut entries);

        for key in [
            "Simulation.HeatSimulation",
            "Simulation.NewtonianGravity",
            "Simulation.AmbientHeat",
            "Simulation.WaterEqualisation",
            "Simulation.AirMode",
            "Simulation.GravityMode",
            "Simulation.CustomGravityX",
            "Simulation.CustomGravityY",
            "Simulation.EdgeMode",
            "Simulation.PrettyPowder",
            "Simulation.EnableModElements",
            "Renderer.DisplayMode",
            "Renderer.RenderMode",
            "Renderer.ColourMode",
            "Renderer.FireIntensity",
            "Renderer.Decorations",
            "Renderer.GridCheckerboard",
            "Renderer.TemperatureScale",
        ] {
            assert!(find_entry(&entries, key).is_some(), "missing key {key}");
        }

        let mut loaded = Prefs::default();
        loaded.read_entries(&entries);

        assert!(!loaded.heat_enable);
        assert!(loaded.newtonian_gravity);
        assert!(loaded.ambient_heat);
        assert!(loaded.water_equal_test);
        assert_eq!(loaded.air_mode, 4);
        assert!((loaded.ambient_air_temp - 512.0).abs() < 1e-3);
        assert!((loaded.edge_pressure + 3.5).abs() < 1e-3);
        assert!((loaded.edge_velocity_x - 12.5).abs() < 1e-3);
        assert!((loaded.edge_velocity_y + 0.25).abs() < 1e-3);
        assert!((loaded.vorticity_coeff - 0.75).abs() < 1e-6);
        assert_eq!(loaded.convection_mode, 1);
        assert_eq!(loaded.gravity_mode, 3);
        assert!((loaded.custom_gravity_x - 0.5).abs() < 1e-6);
        assert!((loaded.custom_gravity_y + 1.0).abs() < 1e-6);
        assert_eq!(loaded.edge_mode, 2);
        assert!(loaded.pretty_powder);
        assert!(!loaded.mod_elements_enabled);
        assert_eq!(loaded.display_mode, DISPLAY_EFFE | DISPLAY_AIRP);
        assert_eq!(loaded.render_mode, RENDER_BASC | RENDER_BLOB);
        assert_eq!(loaded.colour_mode, COLOUR_DEST);
        assert!((loaded.fire_intensity - 2.5).abs() < 1e-6);
        assert!(!loaded.decorations);
        assert!(loaded.grid_checkerboard);
        assert_eq!(loaded.temperature_scale, TEMPSCALE_KELVIN);
    }

    /// The persisted enum values must drive the live [`SimulationConfig`], so
    /// Options changes apply without a restart.
    #[test]
    fn applies_prefs_to_simulation_config() {
        use tpt_core::particle::{AirMode, EdgeMode, GravityMode, SimulationConfig};

        let prefs = Prefs {
            air_mode: 3,     // Off
            gravity_mode: 2, // Radial
            edge_mode: 1,    // Solid
            newtonian_gravity: true,
            water_equal_test: true,
            heat_enable: false,
            ..Prefs::default()
        };

        let mut config = SimulationConfig::new(128, 128);
        prefs.apply_to_config(&mut config);
        assert_eq!(config.air_mode, AirMode::Off);
        assert_eq!(config.gravity_mode, GravityMode::Radial);
        assert_eq!(config.edge_mode, EdgeMode::Solid);
        assert!(config.newton_grav);
        assert!(config.water_equal_test);
        assert!(!config.heat_enable);
    }

    /// `Decoration.Mode` round-trips and out-of-range values fall back to
    /// `Draw`.
    #[test]
    fn deco_mode_round_trips() {
        let prefs = Prefs {
            deco_mode: DecoMode::Smudge.index(),
            ..Prefs::default()
        };
        let mut entries = Vec::new();
        prefs.write_entries(&mut entries);

        let mut loaded = Prefs::default();
        loaded.read_entries(&entries);
        assert_eq!(loaded.deco_mode, 6);
        assert_eq!(loaded.deco_mode(), DecoMode::Smudge);

        let entries = vec![(String::from("Decoration.Mode"), JsonValue::Number(42.0))];
        let mut clamped = Prefs::default();
        clamped.read_entries(&entries);
        assert_eq!(clamped.deco_mode, DecoMode::Draw.index());
        assert_eq!(clamped.deco_mode(), DecoMode::Draw);
    }

    /// The 3D mode and render settings survive `prefs.json`, and invalid
    /// values fall back to the offered choices.
    #[test]
    fn round_trips_voxel_settings() {
        use crate::app_state::GameMode;

        let prefs = Prefs {
            game_mode: GameMode::Sandbox3D,
            voxel_render_scale: 0.75,
            voxel_ao_rays: 4,
            voxel_shadows: false,
            voxel_air_iterations: 20,
            ..Prefs::default()
        };
        let mut entries = Vec::new();
        prefs.write_entries(&mut entries);

        let mut loaded = Prefs::default();
        loaded.read_entries(&entries);
        assert_eq!(loaded.game_mode, GameMode::Sandbox3D);
        assert!((loaded.voxel_render_scale - 0.75).abs() < 1.0e-6);
        assert_eq!(loaded.voxel_ao_rays, 4);
        assert!(!loaded.voxel_shadows);
        assert_eq!(loaded.voxel_air_iterations, 20);

        let invalid = vec![
            (
                String::from("Game.Mode"),
                JsonValue::String(String::from("nope")),
            ),
            (
                String::from("Renderer3D.RenderScale"),
                JsonValue::Number(0.62),
            ),
            (String::from("Renderer3D.AORays"), JsonValue::Number(5.0)),
            (
                String::from("Renderer3D.AirIterations"),
                JsonValue::Number(900.0),
            ),
        ];
        let mut fallback = Prefs::default();
        fallback.read_entries(&invalid);
        assert_eq!(fallback.game_mode, GameMode::Powder2D);
        assert!((fallback.voxel_render_scale - 0.5).abs() < 1.0e-6);
        assert_eq!(fallback.voxel_ao_rays, 4);
        assert_eq!(
            fallback.voxel_air_iterations,
            crate::prefs::DEFAULT_VOXEL_AIR_ITERATIONS
        );
    }

    /// Window geometry and the scale factor (`WindowX/Y`, `Scale`) survive a
    /// write/read cycle.
    #[test]
    fn round_trips_window_and_application_keys() {
        let prefs = Prefs {
            window_x: Some(-120),
            window_y: Some(48),
            window_width: Some(1440),
            window_height: Some(900),
            scale: 3,
            ..Prefs::default()
        };

        let mut entries = Vec::new();
        prefs.write_entries(&mut entries);
        for key in [
            "WindowX",
            "WindowY",
            "WindowWidth",
            "WindowHeight",
            "Scale",
        ] {
            assert!(find_entry(&entries, key).is_some(), "missing key {key}");
        }

        let mut loaded = Prefs::default();
        loaded.read_entries(&entries);
        assert_eq!(loaded.window_x, Some(-120));
        assert_eq!(loaded.window_y, Some(48));
        assert_eq!(loaded.window_width, Some(1440));
        assert_eq!(loaded.window_height, Some(900));
        assert_eq!(loaded.scale, 3);

        // Absent keys keep the defaults instead of inventing a position/size.
        let mut absent = Prefs::default();
        absent.read_entries(&[]);
        assert_eq!(absent.window_x, None);
        assert_eq!(absent.window_y, None);
        assert_eq!(absent.window_width, None);
        assert_eq!(absent.window_height, None);
        assert_eq!(absent.scale, 1);

        // A hand-edited out-of-range scale cannot leave the offered range.
        let out_of_range = vec![(String::from("Scale"), JsonValue::Number(99.0))];
        let mut clamped = Prefs::default();
        clamped.read_entries(&out_of_range);
        assert_eq!(clamped.scale, 1);
    }

    /// `NativeClipboard.Enabled` survives a write/read cycle
    /// (`OptionsView.cpp:413-415`), defaults to the reference value, and
    /// accepts numeric JSON.
    #[test]
    fn round_trips_clipboard_key() {
        let prefs = Prefs {
            native_clipboard: true,
            ..Prefs::default()
        };
        let mut entries = Vec::new();
        prefs.write_entries(&mut entries);
        assert_eq!(
            find_entry(&entries, "NativeClipboard.Enabled"),
            Some(&JsonValue::Bool(true))
        );

        let mut loaded = Prefs::default();
        loaded.read_entries(&entries);
        assert!(loaded.native_clipboard);

        // The reference default is the platform clipboard off
        // (`Clipboard/Dynamic.cpp:104`).
        let mut defaults = Prefs::default();
        defaults.read_entries(&[]);
        assert!(!defaults.native_clipboard);

        // A hand-edited numeric value is read as a boolean.
        let numeric = vec![(
            String::from("NativeClipboard.Enabled"),
            JsonValue::Number(1.0),
        )];
        let mut coerced = Prefs::default();
        coerced.read_entries(&numeric);
        assert!(coerced.native_clipboard);
    }

    /// `Scale` multiplies the fit scale, which is what the viewport blit
    /// consumes. A fresh profile keeps the fit factor untouched so a small
    /// canvas fills the window.
    #[test]
    fn viewport_blit_scale_applies_the_window_scale() {
        let scaled = Prefs {
            scale: 2,
            ..Prefs::default()
        };
        assert!((scaled.viewport_blit_scale(1.75) - 3.5).abs() < 1.0e-6);

        let fresh = Prefs::default();
        assert!((fresh.viewport_blit_scale(1.42) - 1.42).abs() < 1.0e-6);
        assert!((fresh.viewport_blit_scale(0.4) - 0.4).abs() < 1.0e-6);
        assert_eq!(fresh.viewport_blit_scale(f32::NAN), 1.0);
        assert_eq!(fresh.viewport_blit_scale(f32::INFINITY), 1.0);
        assert_eq!(fresh.viewport_blit_scale(-3.0), 1.0);
    }

    /// Out-of-range enum values fall back to the defaults instead of being
    /// written into the live configuration.
    #[test]
    fn rejects_out_of_range_enum_values() {
        let entries = vec![
            (String::from("Simulation.AirMode"), JsonValue::Number(9.0)),
            (
                String::from("Simulation.GravityMode"),
                JsonValue::Number(-1.0),
            ),
            (String::from("Simulation.EdgeMode"), JsonValue::Number(7.0)),
        ];
        let mut prefs = Prefs::default();
        prefs.read_entries(&entries);
        assert_eq!(prefs.air_mode, Prefs::default().air_mode);
        assert_eq!(prefs.gravity_mode, Prefs::default().gravity_mode);
        assert_eq!(prefs.edge_mode, Prefs::default().edge_mode);
    }
}
