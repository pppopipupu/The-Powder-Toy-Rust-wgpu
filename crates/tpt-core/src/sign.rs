//! Canvas text signs.
//!
//! Mirrors `sign` in `src/simulation/Sign.h` and the `sim->signs` vector that
//! owns it (`Simulation.h`): a sign is an anchor pixel plus a short text. Signs
//! are annotations, never physics, but they belong to the simulation state, not
//! to the application layer, because the save file, the undo history and the
//! Lua `sim.signs` table all read them.

/// `sign::Left`: the text starts at the anchor pixel.
pub const JUSTIFY_LEFT: u8 = 0;
/// `sign::Middle`: the text box is centred on the anchor pixel.
pub const JUSTIFY_MIDDLE: u8 = 1;
/// `sign::Right`: the text ends at the anchor pixel.
pub const JUSTIFY_RIGHT: u8 = 2;
/// `sign::None`: no box, the text is painted from the anchor.
pub const JUSTIFY_NONE: u8 = 3;
/// `sign::Max` / `NUM_JUSTMODES`.
pub const JUSTIFY_MODES: u8 = 4;

/// Maximum number of signs on one canvas (`MAXSIGNS` in `Config.h`).
pub const MAX_SIGNS: usize = 16;

/// Maximum sign text length in characters (`Sign::MAX_TEXT` in `Sign.h`).
pub const MAX_SIGN_TEXT: usize = 45;

/// One text sign placed on the canvas.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Sign {
    /// Canvas x coordinate of the anchor pixel.
    pub x: i32,
    /// Canvas y coordinate of the anchor pixel.
    pub y: i32,
    /// Displayed text, at most [`MAX_SIGN_TEXT`] characters.
    pub text: String,
    /// Alignment relative to the anchor pixel (`sign::Justification`).
    #[serde(default)]
    pub ju: u8,
}

impl Sign {
    /// Creates a sign, clamping `text` the way `Sign::setText` does
    /// (`Sign.h`): control characters are dropped and the rest is cut to
    /// [`MAX_SIGN_TEXT`] characters.
    pub fn new(x: i32, y: i32, text: &str) -> Self {
        Self {
            x,
            y,
            text: sanitise_text(text),
            ju: JUSTIFY_LEFT,
        }
    }

    /// Sets the justification, clamping out-of-range values to left.
    pub fn with_justification(mut self, ju: u8) -> Self {
        self.ju = clamp_justification(ju);
        self
    }
}

/// Clamps a raw `ju` to the four modes TPT defines.
pub fn clamp_justification(ju: u8) -> u8 {
    if ju < JUSTIFY_MODES { ju } else { JUSTIFY_LEFT }
}

/// Drops control characters and truncates to [`MAX_SIGN_TEXT`] characters.
///
/// The limit counts characters, not bytes, matching the reference's text box
/// (`Textbox::SetLimit`), so a Chinese sign gets as many glyphs as an ASCII one.
pub fn sanitise_text(text: &str) -> String {
    text.chars()
        .filter(|ch| !ch.is_control())
        .take(MAX_SIGN_TEXT)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_cleaned_and_cut() {
        assert_eq!(sanitise_text("a\nb\tc"), "abc");
        assert_eq!(
            sanitise_text(&"x".repeat(60)).chars().count(),
            MAX_SIGN_TEXT
        );
        // The limit counts characters, so 45 CJK glyphs all fit.
        assert_eq!(
            sanitise_text(&"水".repeat(50)).chars().count(),
            MAX_SIGN_TEXT
        );
    }

    #[test]
    fn new_clamps_the_text() {
        let sign = Sign::new(3, 4, "line\r\nbreak");
        assert_eq!((sign.x, sign.y, sign.text.as_str()), (3, 4, "linebreak"));
        assert_eq!(sign.ju, JUSTIFY_LEFT);
    }

    #[test]
    fn justification_is_clamped() {
        assert_eq!(Sign::new(0, 0, "a").with_justification(2).ju, JUSTIFY_RIGHT);
        assert_eq!(Sign::new(0, 0, "a").with_justification(9).ju, JUSTIFY_LEFT);
    }
}
