//! Canvas text signs (`Simulation::signs`) and their editor state.
//!
//! The sign list itself lives in [`SimulationEngine::signs`], because the save
//! file, the undo history and Lua's `sim.signs` all read it from there. This
//! module owns only the app-side editor: which sign the floating window is
//! rewriting and the text being typed. Nothing here touches the simulation
//! buffers.

use tpt_core::MAX_SIGNS;
use tpt_core::sign::JUSTIFY_MIDDLE;
use tpt_core::sign::sanitise_text;

use crate::i18n::t_fmt;

/// Floating sign text editor.
///
/// `index` distinguishes editing an existing sign (`Some`) from placing a new
/// one (`None`); `x`/`y` are the anchor pixel. `focus` asks the window to focus
/// the text box on its first frame.
#[derive(Clone, Debug, Default)]
pub struct SignEditor {
    /// Whether the editor window is shown.
    pub open: bool,
    /// Index into `SimulationEngine::signs`, if editing.
    pub index: Option<usize>,
    /// Canvas x coordinate of the anchor pixel.
    pub x: i32,
    /// Canvas y coordinate of the anchor pixel.
    pub y: i32,
    /// Text currently in the editor.
    pub text: String,
    /// Alignment chosen for the sign being written (`sign::Justification`).
    pub ju: u8,
    /// Whether the text box still needs keyboard focus.
    pub focus: bool,
    /// `SignWindow::signMoving`: the editor is hidden and the sign follows the
    /// pointer until the next click places it.
    pub moving: bool,
}

/// Truncates and sanitizes editor text to what TPT stores.
///
/// Matches `SignWindow`'s `Textbox::SetLimit(45)` plus `format::CleanString`:
/// control characters are dropped, the result is trimmed and never exceeds
/// [`MAX_SIGN_TEXT`] characters (not bytes, so multi-byte scripts are capped at
/// their own character count).
pub fn clamp_sign_text(text: &str) -> String {
    sanitise_text(text).trim().to_string()
}

impl super::AppState {
    /// Index of the sign anchored exactly at pixel `(x, y)`.
    pub fn sign_index_at(&self, x: i32, y: i32) -> Option<usize> {
        self.engine
            .signs()
            .iter()
            .position(|sign| sign.x == x && sign.y == y)
    }

    /// Opens the editor for a new sign at `(x, y)`, or for the existing sign
    /// anchored there.
    ///
    /// Returns `false` when the sign limit is reached and no editor opened.
    pub fn open_sign_editor_at(&mut self, x: i32, y: i32) -> bool {
        if let Some(index) = self.sign_index_at(x, y) {
            return self.open_sign_editor_for(index);
        }
        if self.engine.signs().len() >= MAX_SIGNS {
            self.set_status(
                t_fmt("signs.limit_reached", &[&MAX_SIGNS.to_string()]),
                true,
            );
            return false;
        }
        self.sign_editor = SignEditor {
            open: true,
            index: None,
            x,
            y,
            text: String::new(),
            // `SignTool.cpp:97` seeds the drop-down with `Middle`.
            ju: JUSTIFY_MIDDLE,
            focus: true,
            moving: false,
        };
        true
    }

    /// Opens the editor seeded with the sign at `index`.
    pub fn open_sign_editor_for(&mut self, index: usize) -> bool {
        let Some(sign) = self.engine.signs().get(index) else {
            return false;
        };
        self.sign_editor = SignEditor {
            open: true,
            index: Some(index),
            x: sign.x,
            y: sign.y,
            text: sign.text.clone(),
            ju: sign.ju,
            focus: true,
            moving: false,
        };
        true
    }

    /// `SignTool.cpp:122-131`: hides the editor and makes the sign follow the
    /// pointer until the next canvas click places it.
    pub fn start_sign_move(&mut self) {
        if self.sign_editor.index.is_none() {
            return;
        }
        self.sign_editor.moving = true;
        self.sign_editor.open = false;
    }

    /// Hides the editor without applying its text.
    pub fn close_sign_editor(&mut self) {
        self.sign_editor.open = false;
        self.sign_editor.focus = false;
    }

    /// Applies the editor: updates the edited sign or appends a new one.
    ///
    /// Empty text is ignored, like the reference window's OK button, which
    /// only creates or rewrites a sign when the text box is non-empty.
    pub fn commit_sign_editor(&mut self) {
        let text = clamp_sign_text(&self.sign_editor.text);
        if !text.is_empty() {
            match self.sign_editor.index {
                Some(index) => {
                    if let Some(sign) = self.engine.signs.get_mut(index) {
                        sign.text = text;
                        sign.ju = self.sign_editor.ju;
                    }
                }
                None => {
                    if self
                        .engine
                        .add_sign(
                            self.sign_editor.x,
                            self.sign_editor.y,
                            &text,
                            self.sign_editor.ju,
                        )
                        .is_none()
                    {
                        self.set_status(
                            t_fmt("signs.limit_reached", &[&MAX_SIGNS.to_string()]),
                            true,
                        );
                    }
                }
            }
        }
        self.close_sign_editor();
    }

    /// Removes the sign the editor is showing (the reference window's Delete
    /// button) and closes the editor.
    pub fn delete_edited_sign(&mut self) {
        if let Some(index) = self.sign_editor.index {
            self.engine.remove_sign(index);
        }
        self.close_sign_editor();
    }

    /// Removes every sign whose anchor pixel is covered by `points`.
    ///
    /// TPT's eraser and the `PT_NONE` element brush both clear signs this way.
    /// Returns whether anything was removed.
    pub fn erase_signs(&mut self, points: &[(i32, i32)]) -> bool {
        if points.is_empty() || self.engine.signs().is_empty() {
            return false;
        }
        let before = self.engine.signs().len();
        self.engine
            .signs
            .retain(|sign| !points.iter().any(|&(x, y)| x == sign.x && y == sign.y));
        if self.engine.signs().len() == before {
            return false;
        }
        // A removed sign shifts every later index; an open editor would then
        // commit the wrong entry, so drop it instead of retargeting it.
        if self.sign_editor.open {
            self.close_sign_editor();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_core::MAX_SIGN_TEXT;

    #[test]
    fn clamps_to_forty_five_characters() {
        let text = "x".repeat(80);
        assert_eq!(clamp_sign_text(&text).chars().count(), MAX_SIGN_TEXT);
        assert_eq!(MAX_SIGN_TEXT, 45);
    }

    #[test]
    fn drops_control_characters_and_trims() {
        assert_eq!(clamp_sign_text("  a\tb\nc  "), "abc");
        assert_eq!(clamp_sign_text("   "), "");
    }

    #[test]
    fn counts_unicode_characters_not_bytes() {
        let text = "中".repeat(50);
        let clamped = clamp_sign_text(&text);
        assert_eq!(clamped.chars().count(), MAX_SIGN_TEXT);
    }
}
