use crate::i18n::t;
use tpt_core::constants::{
    MAX_TEMP, MIN_TEMP, PT_NUM, TEMPSCALE_CELSIUS, TEMPSCALE_FAHRENHEIT, TEMPSCALE_KELVIN,
};
use tpt_core::elements::ElementRegistry;

use super::kind::{PropertyKind, PropertyValue};

/// Formats a value the way the window's value box shows it.
///
/// Temperatures are shown in the configured `TEMPSCALE_*` with the reference
/// `RenderTemperature` suffix (`"300K"`, `"26.85C"`, `"80F"`); every other
/// property is plain decimal.
pub fn format_value(kind: PropertyKind, value: PropertyValue, temperature_scale: u32) -> String {
    match value {
        PropertyValue::Int(value) => value.to_string(),
        PropertyValue::Float(value) => {
            if kind == PropertyKind::Temp {
                format!(
                    "{:.2}{}",
                    temperature_to_scale(value, temperature_scale),
                    temperature_letter(temperature_scale)
                )
            } else {
                format!("{value}")
            }
        }
    }
}

/// Parses the window's value box for `kind`.
///
/// Mirrors `AccessProperty::Parse`: `0x`/`#` hexadecimal, `0b` binary, element
/// names for the integer fields, and a trailing `K`/`C`/`F` on temperatures.
pub fn parse_value(
    kind: PropertyKind,
    text: &str,
    elements: &ElementRegistry,
    temperature_scale: u32,
) -> Result<PropertyValue, &'static str> {
    let text = text.trim();
    if text.is_empty() {
        return Err(t("property.err_enter_value"));
    }
    match kind {
        PropertyKind::Type => {
            let id = parse_int(text, elements)?;
            let id = u32::try_from(id).map_err(|_| t("property.err_element_out_of_range"))?;
            if id as usize >= PT_NUM || !elements.get(id).enabled {
                return Err(t("property.err_unknown_element"));
            }
            Ok(PropertyValue::Int(id as i64))
        }
        PropertyKind::Ctype => Ok(PropertyValue::Int(parse_int(text, elements)?)),
        PropertyKind::Life
        | PropertyKind::Tmp
        | PropertyKind::Tmp2
        | PropertyKind::Tmp3
        | PropertyKind::Tmp4 => Ok(PropertyValue::Int(parse_int(text, elements)?)),
        PropertyKind::Temp => parse_temperature(text, temperature_scale).map(PropertyValue::Float),
        PropertyKind::Vx | PropertyKind::Vy => parse_float(text).map(PropertyValue::Float),
        PropertyKind::Dcolour => parse_uint(text).map(|value| PropertyValue::Int(value as i64)),
    }
}

/// Integer token: `0x`/`#` hexadecimal, `0b` binary, element name or decimal.
fn parse_int(text: &str, elements: &ElementRegistry) -> Result<i64, &'static str> {
    if let Some(digits) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        return i64::from_str_radix(digits, 16).map_err(|_| t("property.err_invalid_hex"));
    }
    if let Some(digits) = text.strip_prefix('#') {
        return i64::from_str_radix(digits, 16).map_err(|_| t("property.err_invalid_hex"));
    }
    if let Some(digits) = text.strip_prefix("0b").or_else(|| text.strip_prefix("0B")) {
        return i64::from_str_radix(digits, 2).map_err(|_| t("property.err_invalid_bin"));
    }
    if let Some(id) = elements.find_by_name(text) {
        return Ok(id as i64);
    }
    text.parse::<i64>()
        .map_err(|_| t("property.err_invalid_int"))
}

/// Unsigned token (`dcolour`): hexadecimal, binary or decimal, no names.
fn parse_uint(text: &str) -> Result<u32, &'static str> {
    let parsed = if let Some(digits) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u32::from_str_radix(digits, 16)
    } else if let Some(digits) = text.strip_prefix('#') {
        u32::from_str_radix(digits, 16)
    } else if let Some(digits) = text.strip_prefix("0b").or_else(|| text.strip_prefix("0B")) {
        u32::from_str_radix(digits, 2)
    } else {
        text.parse::<u32>()
    };
    parsed.map_err(|_| t("property.err_invalid_color"))
}

/// Finite floating-point token.
fn parse_float(text: &str) -> Result<f32, &'static str> {
    text.parse::<f32>()
        .ok()
        .filter(|value| value.is_finite())
        .ok_or_else(|| t("property.err_invalid_float"))
}

/// Temperature token in the given `TEMPSCALE_*`, returned in Kelvin.
fn parse_temperature(text: &str, default_scale: u32) -> Result<f32, &'static str> {
    let (body, scale) = split_temperature_suffix(text, default_scale);
    let value = parse_float(body.trim())?;
    Ok(temperature_from_scale(value, scale).clamp(MIN_TEMP, MAX_TEMP))
}

/// Splits a trailing `K`/`C`/`F` off a temperature, like
/// `format::StringToTemperature`.
fn split_temperature_suffix(text: &str, default_scale: u32) -> (&str, u32) {
    match text.chars().next_back() {
        Some('K' | 'k') => (&text[..text.len() - 1], TEMPSCALE_KELVIN),
        Some('C' | 'c') => (&text[..text.len() - 1], TEMPSCALE_CELSIUS),
        Some('F' | 'f') => (&text[..text.len() - 1], TEMPSCALE_FAHRENHEIT),
        _ => (text, default_scale),
    }
}

/// Converts a Kelvin temperature to the given `TEMPSCALE_*`.
pub fn temperature_to_scale(kelvin: f32, scale: u32) -> f32 {
    match scale {
        TEMPSCALE_KELVIN => kelvin,
        TEMPSCALE_FAHRENHEIT => (kelvin - 273.15) * 9.0 / 5.0 + 32.0,
        _ => kelvin - 273.15,
    }
}

/// Converts a temperature in the given `TEMPSCALE_*` back to Kelvin.
pub fn temperature_from_scale(value: f32, scale: u32) -> f32 {
    match scale {
        TEMPSCALE_KELVIN => value,
        TEMPSCALE_FAHRENHEIT => (value - 32.0) * 5.0 / 9.0 + 273.15,
        _ => value + 273.15,
    }
}

/// Unit suffix for a `TEMPSCALE_*` in a labelled input (`" °C"`).
pub fn temperature_suffix(scale: u32) -> &'static str {
    match scale {
        TEMPSCALE_KELVIN => " K",
        TEMPSCALE_FAHRENHEIT => " °F",
        _ => " °C",
    }
}

/// Compact `RenderTemperature` suffix used inside the property value box.
fn temperature_letter(scale: u32) -> &'static str {
    match scale {
        TEMPSCALE_KELVIN => "K",
        TEMPSCALE_FAHRENHEIT => "F",
        _ => "C",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_core::constants::PT_DUST;
    use tpt_core::elements::ElementRegistry;

    fn registry() -> ElementRegistry {
        ElementRegistry::new_default()
    }

    fn celsius() -> u32 {
        TEMPSCALE_CELSIUS
    }

    #[test]
    fn parses_integers_like_access_property() {
        let elements = registry();
        for (text, expected) in [
            ("42", 42),
            ("-7", -7),
            ("0x10", 16),
            ("0X10", 16),
            ("#1F", 31),
            ("0b101", 5),
            ("0B101", 5),
        ] {
            assert_eq!(
                parse_value(PropertyKind::Tmp, text, &elements, celsius()),
                Ok(PropertyValue::Int(expected)),
                "{text}"
            );
        }

        // Element names resolve for the integer fields, like
        // `SimulationData::GetParticleType`.
        assert_eq!(
            parse_value(PropertyKind::Ctype, "DUST", &elements, celsius()),
            Ok(PropertyValue::Int(PT_DUST as i64))
        );
        assert_eq!(
            parse_value(PropertyKind::Ctype, "default_pt_dust", &elements, celsius()),
            Ok(PropertyValue::Int(PT_DUST as i64))
        );

        for text in ["", "   ", "dusty", "0xZZ", "1.5"] {
            assert!(
                parse_value(PropertyKind::Tmp, text, &elements, celsius()).is_err(),
                "{text:?} should not parse"
            );
        }
    }

    #[test]
    fn type_values_must_be_enabled_elements() {
        let elements = registry();
        assert_eq!(
            parse_value(PropertyKind::Type, "WATR", &elements, celsius()),
            Ok(PropertyValue::Int(
                elements.find_by_name("WATR").unwrap() as i64
            ))
        );
        // `PT_NONE` (0) and unused slots are disabled.
        for text in ["0", "-1", "999"] {
            assert!(
                parse_value(PropertyKind::Type, text, &elements, celsius()).is_err(),
                "{text:?} should not parse as a type"
            );
        }
    }

    #[test]
    fn parses_colours_and_velocities() {
        let elements = registry();
        assert_eq!(
            parse_value(PropertyKind::Dcolour, "0xFF8000", &elements, celsius()),
            Ok(PropertyValue::Int(0xFF_8000))
        );
        assert_eq!(
            parse_value(PropertyKind::Dcolour, "#ff8000", &elements, celsius()),
            Ok(PropertyValue::Int(0xFF_8000))
        );
        for text in ["-1", "DUST", "0x1FFFFFFFF", "1.5"] {
            assert!(
                parse_value(PropertyKind::Dcolour, text, &elements, celsius()).is_err(),
                "{text:?} should not parse"
            );
        }

        assert_eq!(
            parse_value(PropertyKind::Vx, "-1.25", &elements, celsius()),
            Ok(PropertyValue::Float(-1.25))
        );
        for text in ["inf", "NaN", "left"] {
            assert!(
                parse_value(PropertyKind::Vx, text, &elements, celsius()).is_err(),
                "{text:?} should not parse"
            );
        }
    }

    #[test]
    fn temperatures_parse_in_the_selected_scale() {
        let elements = registry();
        for (text, default_scale, expected) in [
            ("300K", celsius(), 300.0),
            ("300", TEMPSCALE_KELVIN, 300.0),
            ("26.85C", celsius(), 300.0),
            ("80.33F", celsius(), 300.0),
            // A bare number is in the configured scale, like `Prop.Value`.
            ("300", celsius(), 573.15),
        ] {
            let Ok(PropertyValue::Float(kelvin)) =
                parse_value(PropertyKind::Temp, text, &elements, default_scale)
            else {
                panic!("{text:?} did not parse");
            };
            assert!(
                (kelvin - expected).abs() < 0.05,
                "{text:?} -> {kelvin} K, expected {expected} K"
            );
        }

        // The suffix overrides the configured scale.
        assert_eq!(
            parse_value(PropertyKind::Temp, "300K", &elements, TEMPSCALE_FAHRENHEIT),
            Ok(PropertyValue::Float(300.0))
        );
        // Out-of-range temperatures are clamped like the engine does.
        assert_eq!(
            parse_value(PropertyKind::Temp, "-500C", &elements, celsius()),
            Ok(PropertyValue::Float(MIN_TEMP))
        );
        assert!(parse_value(PropertyKind::Temp, "hot", &elements, celsius()).is_err());
    }

    #[test]
    fn formatted_values_parse_back() {
        let elements = registry();
        for (kind, value) in [
            (PropertyKind::Type, PropertyValue::Int(PT_DUST as i64)),
            (PropertyKind::Life, PropertyValue::Int(-3)),
            (PropertyKind::Ctype, PropertyValue::Int(0)),
            (PropertyKind::Tmp, PropertyValue::Int(12_345)),
            (PropertyKind::Dcolour, PropertyValue::Int(0x00FF_8000)),
            (PropertyKind::Vx, PropertyValue::Float(-1.5)),
            (PropertyKind::Vy, PropertyValue::Float(2.25)),
        ] {
            let text = format_value(kind, value, celsius());
            assert_eq!(
                parse_value(kind, &text, &elements, celsius()),
                Ok(value),
                "{kind:?} formatted as {text:?}"
            );
        }

        let text = format_value(PropertyKind::Temp, PropertyValue::Float(300.0), celsius());
        assert_eq!(text, "26.85C");
        let Ok(PropertyValue::Float(kelvin)) =
            parse_value(PropertyKind::Temp, &text, &elements, celsius())
        else {
            panic!("{text:?} did not parse");
        };
        assert!((kelvin - 300.0).abs() < 0.05, "{kelvin} K");
    }
}
