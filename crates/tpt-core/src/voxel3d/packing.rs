//! Packing of a single 3D voxel into one `u32`.
//!
//! Bit layout (plan §3.2):
//!
//! ```text
//! bits  0..8   ty        element id, 0 = empty
//! bits  8..24  temp_q    temperature fixed point, 0.125 K
//! bits 24..32  life      0..=255
//! ```
//!
//! TPT's fire and smoke lifetimes are in the 120..269 range, so the old
//! six-bit life field could not represent them. The former `reacted` bit was
//! write-only (no pass ever read it), so its bit joins the life counter.

/// Fixed-point scale between Kelvin and [`Voxel::temp_q`].
pub const TEMP_Q_SCALE: f32 = 8.0;

/// Largest representable life value.
pub const LIFE_MAX: u8 = 0xFF;

const TY_BITS: u32 = 8;
const TY_MASK: u32 = 0xFF;
const TEMP_Q_SHIFT: u32 = TY_BITS;
const TEMP_Q_MASK: u32 = 0xFFFF;
const LIFE_SHIFT: u32 = 24;
const LIFE_MASK: u32 = 0xFF;

/// A single 3D voxel in its unpacked host representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Voxel {
    /// Element id; `0` means empty.
    pub ty: u8,
    /// Temperature in fixed-point Kelvin, `T * 8`.
    pub temp_q: u16,
    /// Life counter, `0..=255`.
    pub life: u8,
}

impl Voxel {
    /// An all-zero voxel, i.e. air at 0 K.
    pub const EMPTY: Self = Self {
        ty: 0,
        temp_q: 0,
        life: 0,
    };

    /// Creates a voxel from its fields and quantises `temp_k`.
    ///
    /// `life` is already a `u8`, so [`LIFE_MAX`] is its full range and no
    /// clamp is needed.
    #[inline]
    pub fn new(ty: u8, temp_k: f32, life: u8) -> Self {
        Self {
            ty,
            temp_q: temp_to_q(temp_k),
            life,
        }
    }

    /// Decodes a packed voxel word.
    #[inline]
    pub const fn from_word(word: u32) -> Self {
        Self {
            ty: (word & TY_MASK) as u8,
            temp_q: ((word >> TEMP_Q_SHIFT) & TEMP_Q_MASK) as u16,
            life: ((word >> LIFE_SHIFT) & LIFE_MASK) as u8,
        }
    }

    /// Encodes this voxel into a word.
    #[inline]
    pub const fn to_word(self) -> u32 {
        (self.ty as u32)
            | ((self.temp_q as u32) << TEMP_Q_SHIFT)
            | (((self.life as u32) & LIFE_MASK) << LIFE_SHIFT)
    }

    /// Decoded temperature in Kelvin.
    #[inline]
    pub fn temp_k(self) -> f32 {
        q_to_temp(self.temp_q)
    }

    /// Returns a copy with its temperature replaced, quantised to 0.125 K.
    #[inline]
    pub fn with_temp_k(mut self, temp_k: f32) -> Self {
        self.temp_q = temp_to_q(temp_k);
        self
    }

    /// Whether the voxel is empty. Only the element id is considered.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.ty == 0
    }
}

/// Quantises Kelvin to `round(T * 8)`, saturating to `0..=65535`.
///
/// `NaN` maps to 0; infinities saturate to the representable range.
#[inline]
pub fn temp_to_q(temp_k: f32) -> u16 {
    (temp_k * TEMP_Q_SCALE).round().clamp(0.0, u16::MAX as f32) as u16
}

/// Decodes fixed-point temperature to Kelvin.
#[inline]
pub fn q_to_temp(temp_q: u16) -> f32 {
    temp_q as f32 / TEMP_Q_SCALE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_round_trips_every_field_boundary() {
        for ty in [0u8, 1, 0x7F, 0xAB, 0xFF] {
            for temp_q in [0u16, 1, 0x00FF, 0xFF00, u16::MAX] {
                for life in [0u8, 1, 31, 192, LIFE_MAX] {
                    let voxel = Voxel { ty, temp_q, life };
                    assert_eq!(Voxel::from_word(voxel.to_word()), voxel);
                }
            }
        }
    }

    #[test]
    fn layout_uses_exactly_the_documented_bits() {
        let full = Voxel {
            ty: 0xFF,
            temp_q: 0xFFFF,
            life: 0xFF,
        };
        assert_eq!(full.to_word(), 0xFFFF_FFFF);

        let word = Voxel::new(0x12, 0.0, 3).to_word();
        assert_eq!(word & 0xFF, 0x12);
        assert_eq!((word >> 8) & 0xFFFF, 0);
        assert_eq!((word >> 24) & 0xFF, 3);
    }

    #[test]
    fn maximum_word_round_trips() {
        let voxel = Voxel::from_word(u32::MAX);
        assert_eq!(
            voxel,
            Voxel {
                ty: 0xFF,
                temp_q: 0xFFFF,
                life: 0xFF,
            }
        );
        assert_eq!(voxel.to_word(), u32::MAX);
        assert_eq!(Voxel::from_word(0).to_word(), 0);
    }

    #[test]
    fn temperature_boundaries_and_rounding() {
        assert_eq!(temp_to_q(-1.0), 0);
        assert_eq!(temp_to_q(-f32::MAX), 0);
        assert_eq!(temp_to_q(0.0), 0);
        assert_eq!(temp_to_q(0.0624), 0);
        assert_eq!(temp_to_q(0.0625), 1);
        assert_eq!(temp_to_q(1.0), 8);
        assert_eq!(temp_to_q(293.15), 2345);
        assert_eq!(temp_to_q(8191.875), u16::MAX);
        assert_eq!(temp_to_q(8191.9375), u16::MAX);
        assert_eq!(temp_to_q(9000.0), u16::MAX);
        assert_eq!(temp_to_q(f32::INFINITY), u16::MAX);
        assert_eq!(temp_to_q(f32::NEG_INFINITY), 0);
        assert_eq!(temp_to_q(f32::NAN), 0);

        assert_eq!(q_to_temp(0), 0.0);
        assert_eq!(q_to_temp(1), 0.125);
        assert_eq!(q_to_temp(8), 1.0);
        assert_eq!(q_to_temp(2345), 293.125);
        assert_eq!(q_to_temp(u16::MAX), 8191.875);
    }

    #[test]
    fn constructors_clamp_and_quantise() {
        let voxel = Voxel::new(7, 293.15, 255);
        assert_eq!(voxel.ty, 7);
        assert_eq!(voxel.temp_q, 2345);
        assert_eq!(voxel.life, LIFE_MAX);
        assert_eq!(voxel.temp_k(), 293.125);

        let cold = Voxel::new(1, -50.0, 0).with_temp_k(1000.0);
        assert_eq!(cold.temp_q, 8000);
        assert_eq!(cold.temp_k(), 1000.0);

        assert!(Voxel::EMPTY.is_empty());
        assert!(!voxel.is_empty());
        assert!(Voxel::default().is_empty());
    }
}
