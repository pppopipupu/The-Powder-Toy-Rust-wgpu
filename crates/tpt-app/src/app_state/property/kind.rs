use tpt_core::particle::GpuParticle;

use crate::i18n::t;
use crate::prefs::Prefs;

/// One scalar particle property the property tool can rewrite.
///
/// Mirrors `Particle::GetProperties()` except `x`, `y` and `flags`, which the
/// port has no brush command for. The dropdown lists the properties in the
/// reference order; [`Self::index`] is the value stored in `Prop.Type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropertyKind {
    /// Element type (`PT_*`).
    Type,
    /// Remaining life.
    Life,
    /// Transition / clone element.
    Ctype,
    /// Horizontal velocity.
    Vx,
    /// Vertical velocity.
    Vy,
    /// Temperature in Kelvin.
    Temp,
    /// `tmp` scratch value.
    Tmp,
    /// `tmp2` scratch value.
    Tmp2,
    /// `tmp3` scratch value.
    Tmp3,
    /// `tmp4` scratch value.
    Tmp4,
    /// Decoration colour (`0x00RRGGBB`).
    Dcolour,
}

impl PropertyKind {
    /// Every property, in the order the window's dropdown lists them
    /// (`Particle::GetProperties` order).
    pub const ALL: [Self; 11] = [
        Self::Type,
        Self::Life,
        Self::Ctype,
        Self::Vx,
        Self::Vy,
        Self::Temp,
        Self::Tmp,
        Self::Tmp2,
        Self::Tmp3,
        Self::Tmp4,
        Self::Dcolour,
    ];

    /// Highest value [`Self::index`] returns; `Prop.Type` is clamped to it.
    pub const MAX_INDEX: u32 = Self::ALL.len() as u32 - 1;

    /// Property name, identical to the `AccessProperty`/Lua field name.
    pub fn label(self) -> &'static str {
        match self {
            Self::Type => "type",
            Self::Life => "life",
            Self::Ctype => "ctype",
            Self::Vx => "vx",
            Self::Vy => "vy",
            Self::Temp => "temp",
            Self::Tmp => "tmp",
            Self::Tmp2 => "tmp2",
            Self::Tmp3 => "tmp3",
            Self::Tmp4 => "tmp4",
            Self::Dcolour => "dcolour",
        }
    }

    /// Localized display name shown in the UI.
    pub fn display(self) -> &'static str {
        match self {
            Self::Type => t("property.field_type"),
            Self::Life => t("property.field_life"),
            Self::Ctype => t("property.field_ctype"),
            Self::Vx => t("property.field_vx"),
            Self::Vy => t("property.field_vy"),
            Self::Temp => t("property.field_temp"),
            Self::Tmp => t("property.field_tmp"),
            Self::Tmp2 => t("property.field_tmp2"),
            Self::Tmp3 => t("property.field_tmp3"),
            Self::Tmp4 => t("property.field_tmp4"),
            Self::Dcolour => t("property.field_dcolour"),
        }
    }

    /// Value stored in the `Prop.Type` preference.
    ///
    /// `0..=5` are the indices written before the window existed, so an old
    /// `prefs.json` keeps selecting the same field.
    pub fn index(self) -> u32 {
        match self {
            Self::Tmp => 0,
            Self::Life => 1,
            Self::Ctype => 2,
            Self::Tmp2 => 3,
            Self::Tmp3 => 4,
            Self::Tmp4 => 5,
            Self::Type => 6,
            Self::Temp => 7,
            Self::Vx => 8,
            Self::Vy => 9,
            Self::Dcolour => 10,
        }
    }

    /// Field selected by a `Prop.Type` preference value. Unknown indices fall
    /// back to `tmp`, the historical default.
    pub fn from_index(index: u32) -> Self {
        match index {
            1 => Self::Life,
            2 => Self::Ctype,
            3 => Self::Tmp2,
            4 => Self::Tmp3,
            5 => Self::Tmp4,
            6 => Self::Type,
            7 => Self::Temp,
            8 => Self::Vx,
            9 => Self::Vy,
            10 => Self::Dcolour,
            _ => Self::Tmp,
        }
    }

    /// Whether the property is stored as an integer (the value input is parsed
    /// as one and `Prop.Value` holds the rounded value).
    pub fn is_integer(self) -> bool {
        !matches!(self, Self::Temp | Self::Vx | Self::Vy)
    }
}

/// A parsed `AccessProperty` value: integers and floats are kept apart so an
/// integer field never round-trips through `f32` before it reaches the engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropertyValue {
    /// Integer property (`type`, `life`, `ctype`, `tmp*`, `dcolour`).
    Int(i64),
    /// Floating-point property (`temp`, `vx`, `vy`).
    Float(f32),
}

impl PropertyValue {
    /// Rounded integer form, used by the integer engine commands.
    pub fn as_i64(self) -> i64 {
        match self {
            Self::Int(value) => value,
            Self::Float(value) => value.round() as i64,
        }
    }

    /// Floating-point form, used by the per-particle engine command.
    pub fn as_f32(self) -> f32 {
        match self {
            Self::Int(value) => value as f32,
            Self::Float(value) => value,
        }
    }
}

/// Value of `kind` stored in the preferences.
///
/// `Prop.Value` is a number in the port, so integer fields are rounded and
/// float fields keep the stored value (temperatures are stored in Kelvin).
pub fn value_from_prefs(kind: PropertyKind, prefs: &Prefs) -> PropertyValue {
    if kind.is_integer() {
        PropertyValue::Int(prefs.prop_value.round() as i64)
    } else {
        PropertyValue::Float(prefs.prop_value)
    }
}

/// Value of `kind` on a particle, used by `PropertyWindow::TakePropertyFrom`.
pub fn value_of(particle: &GpuParticle, kind: PropertyKind) -> PropertyValue {
    match kind {
        PropertyKind::Type => PropertyValue::Int(particle.ty as i64),
        PropertyKind::Life => PropertyValue::Int(particle.life as i64),
        PropertyKind::Ctype => PropertyValue::Int(particle.ctype as i64),
        PropertyKind::Vx => PropertyValue::Float(particle.vx),
        PropertyKind::Vy => PropertyValue::Float(particle.vy),
        PropertyKind::Temp => PropertyValue::Float(particle.temp),
        PropertyKind::Tmp => PropertyValue::Int(particle.tmp as i64),
        PropertyKind::Tmp2 => PropertyValue::Int(particle.tmp2 as i64),
        PropertyKind::Tmp3 => PropertyValue::Int(particle.tmp3 as i64),
        PropertyKind::Tmp4 => PropertyValue::Int(particle.tmp4 as i64),
        PropertyKind::Dcolour => PropertyValue::Int(particle.dcolour as i64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_property_round_trips_through_its_pref_index() {
        for kind in PropertyKind::ALL {
            assert_eq!(PropertyKind::from_index(kind.index()), kind);
        }
        assert_eq!(PropertyKind::MAX_INDEX, 10);
        // The indices written before the window existed keep their meaning.
        assert_eq!(PropertyKind::from_index(0), PropertyKind::Tmp);
        assert_eq!(PropertyKind::from_index(1), PropertyKind::Life);
        assert_eq!(PropertyKind::from_index(2), PropertyKind::Ctype);
        assert_eq!(PropertyKind::from_index(3), PropertyKind::Tmp2);
        assert_eq!(PropertyKind::from_index(4), PropertyKind::Tmp3);
        assert_eq!(PropertyKind::from_index(5), PropertyKind::Tmp4);
        // Out-of-range values fall back to `tmp`.
        assert_eq!(PropertyKind::from_index(99), PropertyKind::Tmp);
    }

    #[test]
    fn every_property_has_display_and_label() {
        for kind in PropertyKind::ALL {
            assert!(!kind.label().is_empty());
            assert!(!kind.display().is_empty());
        }
    }
}
