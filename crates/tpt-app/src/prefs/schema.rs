//! Preference keys, defaults and typed access to the parsed flat object.

use tpt_core::constants::{
    AIRC_BOUSSINESQ, DEFAULT_TEMP_K, DISPLAY_WARP, RENDER_FANCY, TEMPSCALE_KELVIN,
};
use tpt_core::particle::{AirMode, EdgeMode, GravityMode};

use super::io::JsonValue;
use super::{DEFAULT_DRAW_FPS, DEFAULT_UNDO_HISTORY_LIMIT, FpsLimit, Prefs};

/// Dotted keys used by `prefs.json` (same strings as the C++ game).
pub(super) mod keys {
    pub(in crate::prefs) const NEWTONIAN_GRAVITY: &str = "Simulation.NewtonianGravity";
    pub(in crate::prefs) const AMBIENT_HEAT: &str = "Simulation.AmbientHeat";
    pub(in crate::prefs) const AMBIENT_AIR_TEMP: &str = "Simulation.AmbientAirTemp";
    pub(in crate::prefs) const EDGE_PRESSURE: &str = "Simulation.EdgePressure";
    pub(in crate::prefs) const EDGE_VELOCITY_X: &str = "Simulation.EdgeVelocityX";
    pub(in crate::prefs) const EDGE_VELOCITY_Y: &str = "Simulation.EdgeVelocityY";
    pub(in crate::prefs) const VORTICITY_COEFF: &str = "Simulation.VorticityCoeff";
    pub(in crate::prefs) const CONVECTION_MODE: &str = "Simulation.ConvectionMode";
    pub(in crate::prefs) const EDGE_MODE: &str = "Simulation.EdgeMode";
    pub(in crate::prefs) const INCLUDE_PRESSURE: &str = "Simulation.IncludePressure";
    pub(in crate::prefs) const UNDO_HISTORY_LIMIT: &str = "Simulation.UndoHistoryLimit";
    pub(in crate::prefs) const PRETTY_POWDER: &str = "Simulation.PrettyPowder";
    pub(in crate::prefs) const ENABLE_MOD_ELEMENTS: &str = "Simulation.EnableModElements";
    pub(in crate::prefs) const HEAT_SIMULATION: &str = "Simulation.HeatSimulation";
    pub(in crate::prefs) const WATER_EQUALISATION: &str = "Simulation.WaterEqualisation";
    pub(in crate::prefs) const AIR_MODE: &str = "Simulation.AirMode";
    pub(in crate::prefs) const GRAVITY_MODE: &str = "Simulation.GravityMode";
    pub(in crate::prefs) const CUSTOM_GRAVITY_X: &str = "Simulation.CustomGravityX";
    pub(in crate::prefs) const CUSTOM_GRAVITY_Y: &str = "Simulation.CustomGravityY";
    pub(in crate::prefs) const DISPLAY_MODE: &str = "Renderer.DisplayMode";
    pub(in crate::prefs) const RENDER_MODE: &str = "Renderer.RenderMode";
    pub(in crate::prefs) const COLOUR_MODE: &str = "Renderer.ColourMode";
    pub(in crate::prefs) const DECORATIONS: &str = "Renderer.Decorations";
    pub(in crate::prefs) const GRID_CHECKERBOARD: &str = "Renderer.GridCheckerboard";
    pub(in crate::prefs) const GRAVITY_FIELD: &str = "Renderer.GravityField";
    pub(in crate::prefs) const TEMPERATURE_SCALE: &str = "Renderer.TemperatureScale";
    pub(in crate::prefs) const FIRE_INTENSITY: &str = "Renderer.FireIntensity";
    pub(in crate::prefs) const MOUSE_CLICK_REQUIRED: &str = "MouseClickRequired";
    pub(in crate::prefs) const DECORATION_RED: &str = "Decoration.Red";
    pub(in crate::prefs) const DECORATION_GREEN: &str = "Decoration.Green";
    pub(in crate::prefs) const DECORATION_BLUE: &str = "Decoration.Blue";
    pub(in crate::prefs) const DECORATION_ALPHA: &str = "Decoration.Alpha";
    pub(in crate::prefs) const DECORATION_MODE: &str = "Decoration.Mode";
    pub(in crate::prefs) const PROP_TYPE: &str = "Prop.Type";
    pub(in crate::prefs) const PROP_VALUE: &str = "Prop.Value";
    pub(in crate::prefs) const PROP_FILL: &str = "Prop.Fill";
    pub(in crate::prefs) const TOOL_SPAWN_ELEMENT: &str = "Tool.SpawnElement";
    pub(in crate::prefs) const TOOL_PRIMARY: &str = "Tool.Primary";
    pub(in crate::prefs) const TOOL_SECONDARY: &str = "Tool.Secondary";
    pub(in crate::prefs) const FPS_LIMIT: &str = "FpsLimit";
    pub(in crate::prefs) const DRAW_LIMIT: &str = "DrawLimit";
    pub(in crate::prefs) const BLURRY_SCALING: &str = "BlurryScaling";
    pub(in crate::prefs) const RESIZABLE: &str = "Resizable";
    pub(in crate::prefs) const FULLSCREEN: &str = "Fullscreen";
    pub(in crate::prefs) const NATIVE_CLIPBOARD: &str = "NativeClipboard.Enabled";
    pub(in crate::prefs) const GRAVE_EXITS_CONSOLE: &str = "GraveExitsConsole";
    pub(in crate::prefs) const LANGUAGE: &str = "Language";
    pub(in crate::prefs) const GAME_MODE: &str = "Game.Mode";
    pub(in crate::prefs) const WINDOW_X: &str = "WindowX";
    pub(in crate::prefs) const WINDOW_Y: &str = "WindowY";
    pub(in crate::prefs) const WINDOW_WIDTH: &str = "WindowWidth";
    pub(in crate::prefs) const WINDOW_HEIGHT: &str = "WindowHeight";
    pub(in crate::prefs) const SCALE: &str = "Scale";
    /// Keys the port retired. They are dropped from `prefs.json` on the next
    /// save so an old profile does not keep a setting no code reads:
    /// `ForceIntegerScaling` (`WindowFrameOps.h:10`) has no egui equivalent.
    pub(in crate::prefs) const RETIRED: &[&str] = &["ForceIntegerScaling"];
    pub(in crate::prefs) const VOXEL_RENDER_SCALE: &str = "Renderer3D.RenderScale";
    pub(in crate::prefs) const VOXEL_AO_RAYS: &str = "Renderer3D.AORays";
    pub(in crate::prefs) const VOXEL_SHADOWS: &str = "Renderer3D.Shadows";
    pub(in crate::prefs) const VOXEL_AIR_ITERATIONS: &str = "Renderer3D.AirIterations";
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            newtonian_gravity: false,
            ambient_heat: false,
            ambient_air_temp: DEFAULT_TEMP_K,
            edge_pressure: 0.0,
            edge_velocity_x: 0.0,
            edge_velocity_y: 0.0,
            vorticity_coeff: 0.1,
            convection_mode: AIRC_BOUSSINESQ,
            edge_mode: EdgeMode::Void as u32,
            include_pressure: true,
            undo_history_limit: DEFAULT_UNDO_HISTORY_LIMIT,
            pretty_powder: false,
            mod_elements_enabled: true,
            heat_enable: true,
            water_equal_test: false,
            air_mode: AirMode::On as u32,
            gravity_mode: GravityMode::Vertical as u32,
            custom_gravity_x: 0.0,
            custom_gravity_y: 1.0,

            display_mode: DISPLAY_WARP,
            render_mode: RENDER_FANCY,
            colour_mode: 0,
            decorations: true,
            gravity_field: false,
            grid_checkerboard: false,
            temperature_scale: TEMPSCALE_KELVIN,
            fire_intensity: 1.0,

            mouse_click_required: false,
            deco_red: 200,
            deco_green: 100,
            deco_blue: 50,
            deco_alpha: 255,
            deco_mode: 0,
            prop_field: 0,
            prop_value: 0.0,
            prop_fill: false,
            spawn_element: tpt_core::constants::PT_DUST,
            primary_tool: String::new(),
            secondary_tool: String::new(),

            fps_limit: FpsLimit::default(),
            draw_fps: DEFAULT_DRAW_FPS,
            blurry_scaling: false,
            resizable: false,
            fullscreen: false,
            native_clipboard: false,
            grave_exits_console: true,
            language: crate::i18n::Language::En,
            game_mode: super::GameMode::Powder2D,
            voxel_render_scale: super::DEFAULT_VOXEL_RENDER_SCALE,
            voxel_ao_rays: super::DEFAULT_VOXEL_AO_RAYS,
            voxel_shadows: true,
            voxel_air_iterations: super::DEFAULT_VOXEL_AIR_ITERATIONS,

            window_x: None,
            window_y: None,
            window_width: None,
            window_height: None,
            scale: 1,

            entries: Vec::new(),
        }
    }
}

pub(super) fn find_entry<'a>(
    entries: &'a [(String, JsonValue)],
    key: &str,
) -> Option<&'a JsonValue> {
    entries
        .iter()
        .find(|(name, _)| name.as_str() == key)
        .map(|(_, value)| value)
}

pub(super) fn value_as_f64(value: &JsonValue) -> Option<f64> {
    match value {
        JsonValue::Number(number) if number.is_finite() => Some(*number),
        JsonValue::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        JsonValue::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

pub(super) fn get_f64(entries: &[(String, JsonValue)], key: &str) -> Option<f64> {
    find_entry(entries, key).and_then(value_as_f64)
}

pub(super) fn get_bool(entries: &[(String, JsonValue)], key: &str, default: bool) -> bool {
    match find_entry(entries, key) {
        Some(JsonValue::Bool(flag)) => *flag,
        Some(value) => value_as_f64(value)
            .map(|number| number != 0.0)
            .unwrap_or(default),
        None => default,
    }
}

pub(super) fn get_f32(entries: &[(String, JsonValue)], key: &str, default: f32) -> f32 {
    get_f64(entries, key)
        .map(|number| number as f32)
        .filter(|number| number.is_finite())
        .unwrap_or(default)
}

pub(super) fn get_f32_in_range(
    entries: &[(String, JsonValue)],
    key: &str,
    default: f32,
    min: f32,
    max: f32,
) -> f32 {
    let value = get_f32(entries, key, default);
    if (min..=max).contains(&value) {
        value
    } else {
        default
    }
}

pub(super) fn get_u32(entries: &[(String, JsonValue)], key: &str, default: u32) -> u32 {
    match get_f64(entries, key) {
        Some(number) if number.is_finite() && number >= 0.0 && number <= u32::MAX as f64 => {
            number.round() as u32
        }
        _ => default,
    }
}

/// Reads a signed integer. Returns `None` for a missing or non-numeric key so
/// the caller can distinguish "not saved yet" from a real coordinate.
pub(super) fn get_i32(entries: &[(String, JsonValue)], key: &str) -> Option<i32> {
    match get_f64(entries, key) {
        Some(number)
            if number.is_finite() && (i32::MIN as f64..=i32::MAX as f64).contains(&number) =>
        {
            Some(number.round() as i32)
        }
        _ => None,
    }
}

pub(super) fn get_u32_in_range(
    entries: &[(String, JsonValue)],
    key: &str,
    default: u32,
    min: u32,
    max: u32,
) -> u32 {
    let value = get_u32(entries, key, default);
    if (min..=max).contains(&value) {
        value
    } else {
        default
    }
}

/// Reads a string preference; numbers and booleans are accepted and rendered
/// in their JSON form so a hand-edited file still round-trips.
pub(super) fn get_string(entries: &[(String, JsonValue)], key: &str, default: &str) -> String {
    match find_entry(entries, key) {
        Some(JsonValue::String(text)) => text.clone(),
        Some(JsonValue::Number(number)) => number.to_string(),
        Some(JsonValue::Bool(flag)) => flag.to_string(),
        _ => default.to_string(),
    }
}

pub(super) fn set_string(entries: &mut Vec<(String, JsonValue)>, key: &str, value: &str) {
    set_entry(entries, key, JsonValue::String(value.to_string()));
}

pub(super) fn get_u8(entries: &[(String, JsonValue)], key: &str, default: u8) -> u8 {
    match get_f64(entries, key) {
        Some(number) if number.is_finite() && (0.0..=255.0).contains(&number) => {
            number.round() as u8
        }
        _ => default,
    }
}

pub(super) fn set_entry(entries: &mut Vec<(String, JsonValue)>, key: &str, value: JsonValue) {
    if let Some((_, slot)) = entries.iter_mut().find(|(name, _)| name.as_str() == key) {
        *slot = value;
    } else {
        entries.push((key.to_string(), value));
    }
}

pub(super) fn set_bool(entries: &mut Vec<(String, JsonValue)>, key: &str, value: bool) {
    set_entry(entries, key, JsonValue::Bool(value));
}

pub(super) fn set_u32(entries: &mut Vec<(String, JsonValue)>, key: &str, value: u32) {
    set_entry(entries, key, JsonValue::Number(value as f64));
}

pub(super) fn set_i32(entries: &mut Vec<(String, JsonValue)>, key: &str, value: i32) {
    set_entry(entries, key, JsonValue::Number(value as f64));
}

pub(super) fn set_f32(entries: &mut Vec<(String, JsonValue)>, key: &str, value: f32) {
    set_entry(entries, key, JsonValue::Number(clean_f32(value)));
}

/// Converts an `f32` to the shortest stable `f64` representation so a
/// round-trip through the file does not store binary-float noise.
pub(super) fn clean_f32(value: f32) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    format!("{value}").parse::<f64>().unwrap_or(value as f64)
}

pub(super) fn set_f64(entries: &mut Vec<(String, JsonValue)>, key: &str, value: f64) {
    set_entry(entries, key, JsonValue::Number(value));
}
