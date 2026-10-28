//! Colour picker state for the decoration colour (`ColourPickerActivity`)
//! and TPT's eight decoration presets.
//!
//! The reference opens the picker from the decoration menu
//! (`GameController::OpenColourPicker`, `GameController.cpp:1378-1382`) and
//! seeds it with the colour every `DECO` tool is currently painting
//! (`GameModel::GetColourSelectorColour`). Its window is an HSV field plus
//! hue/saturation/value sliders and RGBA text boxes
//! (`ColourPickerActivity.cpp:15-114`); `Done` hands the picked colour back
//! through `GameController::SetColour`, which stores it in `Decoration.*` and
//! fans it out to the decoration tools (`GameModel::SetColourSelectorColour`,
//! `GameModel.cpp:1098-1107`).
//!
//! The eight swatches are `GameModel::colourPresets` (`GameModel.cpp:197-204`);
//! picking one applies it immediately (`GameView::NotifyColourPresetsChanged`)
//! and editing the picker while a preset is active overwrites that preset
//! (`GameModel::SetPresetColour`, `GameModel.cpp:1070-1077`).
//!
//! The conversions are literal ports of `HSV_to_RGB` / `RGB_to_HSV`
//! (`Misc.cpp:7-68`), including the 0-359 hue and 0-255 saturation/value
//! ranges.

use super::AppState;

/// The eight decoration presets in TPT's order (`GameModel.cpp:197-204`):
/// white, cyan, magenta, yellow, red, green, blue, black. All opaque.
pub const PRESET_COLOURS: [[u8; 4]; 8] = [
    [255, 255, 255, 255],
    [0, 255, 255, 255],
    [255, 0, 255, 255],
    [255, 255, 0, 255],
    [255, 0, 0, 255],
    [0, 255, 0, 255],
    [0, 0, 255, 255],
    [0, 0, 0, 255],
];

/// `HSV_to_RGB` (`Misc.cpp:7-39`): hue in degrees (`0..=359`), saturation and
/// value in `0..=255`.
pub fn hsv_to_rgb(hue: u16, saturation: u8, value: u8) -> [u8; 3] {
    let hh = hue.min(359) as f32 / 60.0;
    let ss = saturation as f32 / 255.0;
    let vv = value as f32 / 255.0;
    let c = vv * ss;
    let x = c * (1.0 - (hh % 2.0 - 1.0).abs());
    let (red, green, blue) = if hh < 1.0 {
        (c * 255.0, x * 255.0, 0.0)
    } else if hh < 2.0 {
        (x * 255.0, c * 255.0, 0.0)
    } else if hh < 3.0 {
        (0.0, c * 255.0, x * 255.0)
    } else if hh < 4.0 {
        (0.0, x * 255.0, c * 255.0)
    } else if hh < 5.0 {
        (x * 255.0, 0.0, c * 255.0)
    } else {
        (c * 255.0, 0.0, x * 255.0)
    };
    // `m = (int)((vv-c)*255.0); *r += m; ...`
    let m = ((vv - c) * 255.0) as i32;
    let channel = |v: f32| (v as i32 + m).clamp(0, 255) as u8;
    [channel(red), channel(green), channel(blue)]
}

/// `RGB_to_HSV` (`Misc.cpp:41-68`): returns `(hue, saturation, value)` with
/// the reference's `0..=359` / `0..=255` ranges.
pub fn rgb_to_hsv(red: u8, green: u8, blue: u8) -> (u16, u8, u8) {
    let rr = red as f32 / 255.0;
    let gg = green as f32 / 255.0;
    let bb = blue as f32 / 255.0;
    let a = rr.min(gg).min(bb);
    let x = rr.max(gg).max(bb);
    if a == x {
        // Greyscale.
        return (0, 0, (a * 255.0) as u8);
    }
    let c = if rr == a {
        gg - bb
    } else if bb == a {
        rr - gg
    } else {
        bb - rr
    };
    let d = if rr == a {
        3.0
    } else if bb == a {
        1.0
    } else {
        5.0
    };
    let hue = (60.0 * (d - c / (x - a))) as i32;
    let saturation = (255.0 * ((x - a) / x)) as u8;
    let value = (255.0 * x) as u8;
    (hue.clamp(0, 359) as u16, saturation, value)
}

/// Floating colour picker state: the HSV triple the sliders own plus the
/// alpha channel (`ColourPickerActivity::currentHue/Saturation/Value` and
/// `currentAlpha`).
#[derive(Clone, Debug, PartialEq)]
pub struct ColourPicker {
    /// Whether the floating window is shown.
    pub open: bool,
    /// The eight editable swatches (`GameModel::colourPresets`).
    pub presets: [[u8; 4]; 8],
    /// Preset the window was opened from, if any; `Done` writes the picked
    /// colour back into it (`GameModel::SetPresetColour`).
    pub active_preset: Option<usize>,
    /// Hue in degrees, `0..=359`.
    pub hue: u16,
    /// Saturation, `0..=255`.
    pub saturation: u8,
    /// Value, `0..=255`.
    pub value: u8,
    /// Alpha, `0..=255`.
    pub alpha: u8,
}

impl Default for ColourPicker {
    fn default() -> Self {
        let mut picker = Self {
            open: false,
            presets: PRESET_COLOURS,
            active_preset: None,
            hue: 0,
            saturation: 255,
            value: 255,
            alpha: 255,
        };
        // TPT's persisted decoration default (`GameModel.cpp:190-193`).
        picker.set_rgba([200, 100, 50, 255]);
        picker
    }
}

impl ColourPicker {
    /// The RGB channels derived from the HSV triple, plus the alpha channel.
    pub fn rgba(&self) -> [u8; 4] {
        let [red, green, blue] = hsv_to_rgb(self.hue, self.saturation, self.value);
        [red, green, blue, self.alpha]
    }

    /// `colourChange` (`ColourPickerActivity.cpp:30-44`): an RGB text box edit
    /// recomputes the HSV triple.
    pub fn set_rgba(&mut self, [red, green, blue, alpha]: [u8; 4]) {
        self.set_rgb(red, green, blue);
        self.alpha = alpha;
    }

    /// RGB channels of [`Self::set_rgba`].
    pub fn set_rgb(&mut self, red: u8, green: u8, blue: u8) {
        let (hue, saturation, value) = rgb_to_hsv(red, green, blue);
        self.hue = hue;
        self.saturation = saturation;
        self.value = value;
    }

    /// `colourChangeSlider` (`ColourPickerActivity.cpp:46-56`): the HSV
    /// sliders own the triple, the RGB values derive from it.
    pub fn set_hsv(&mut self, hue: u16, saturation: u8, value: u8) {
        self.hue = hue.min(359);
        self.saturation = saturation;
        self.value = value;
    }

    /// Reference `hexValue` label (`Format::Hex`, width 2, `a, r, g, b`):
    /// `0xAARRGGBB`.
    pub fn hex(&self) -> String {
        let [red, green, blue, alpha] = self.rgba();
        format!("0x{alpha:02X}{red:02X}{green:02X}{blue:02X}")
    }

    /// Seed the window from the current decoration colour, optionally marking
    /// the preset it was opened from.
    pub fn open_from(&mut self, colour: [u8; 4], preset: Option<usize>) {
        self.set_rgba(colour);
        self.active_preset = preset;
        self.open = true;
    }
}

impl AppState {
    /// `GameController::OpenColourPicker` (`GameController.cpp:1378-1382`):
    /// the window starts from the colour the decoration tools paint with. A
    /// previously selected preset stays active, so `Done` overwrites it
    /// (`GameModel::SetPresetColour`).
    pub fn open_colour_picker(&mut self) {
        let colour = self.prefs.deco_colour();
        let preset = self.colour_picker.active_preset;
        self.colour_picker.open_from(colour, preset);
    }

    /// `GameController::SetColour` (`GameController.cpp:1116-1120`) followed by
    /// `GameModel::SetColourSelectorColour` (`GameModel.cpp:1098-1107`): the
    /// picked colour becomes the decoration colour every `DECO` tool blends
    /// over the element display.
    pub fn set_decoration_colour(&mut self, [red, green, blue, alpha]: [u8; 4]) {
        self.prefs.deco_red = red;
        self.prefs.deco_green = green;
        self.prefs.deco_blue = blue;
        self.prefs.deco_alpha = alpha;
        if let Some(index) = self.colour_picker.active_preset
            && let Some(slot) = self.colour_picker.presets.get_mut(index)
        {
            *slot = [red, green, blue, alpha];
        }
        self.colour_picker.set_rgba([red, green, blue, alpha]);
        self.apply_prefs();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_match_the_reference_swatches() {
        // `GameModel.cpp:197-204`.
        assert_eq!(PRESET_COLOURS[0], [255, 255, 255, 255]);
        assert_eq!(PRESET_COLOURS[1], [0, 255, 255, 255]);
        assert_eq!(PRESET_COLOURS[2], [255, 0, 255, 255]);
        assert_eq!(PRESET_COLOURS[3], [255, 255, 0, 255]);
        assert_eq!(PRESET_COLOURS[4], [255, 0, 0, 255]);
        assert_eq!(PRESET_COLOURS[5], [0, 255, 0, 255]);
        assert_eq!(PRESET_COLOURS[6], [0, 0, 255, 255]);
        assert_eq!(PRESET_COLOURS[7], [0, 0, 0, 255]);
    }

    #[test]
    fn hsv_to_rgb_matches_the_reference_knots() {
        assert_eq!(hsv_to_rgb(0, 255, 255), [255, 0, 0]);
        assert_eq!(hsv_to_rgb(120, 255, 255), [0, 255, 0]);
        assert_eq!(hsv_to_rgb(240, 255, 255), [0, 0, 255]);
        assert_eq!(hsv_to_rgb(0, 0, 255), [255, 255, 255]);
        assert_eq!(hsv_to_rgb(0, 0, 0), [0, 0, 0]);
    }

    #[test]
    fn rgb_to_hsv_recovers_the_preset_swatches() {
        assert_eq!(rgb_to_hsv(255, 255, 255), (0, 0, 255));
        assert_eq!(rgb_to_hsv(0, 0, 0), (0, 0, 0));
        assert_eq!(rgb_to_hsv(255, 0, 0), (0, 255, 255));
        assert_eq!(rgb_to_hsv(0, 255, 0), (120, 255, 255));
        assert_eq!(rgb_to_hsv(0, 0, 255), (240, 255, 255));
        assert_eq!(rgb_to_hsv(255, 0, 255), (300, 255, 255));
        assert_eq!(rgb_to_hsv(255, 255, 0), (60, 255, 255));
    }

    #[test]
    fn rgb_round_trips_through_hsv() {
        // HSV's primary and secondary colours are the exact fixed points of
        // the reference conversions.
        for rgb in [
            [255u8, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 255, 255],
            [0, 0, 0],
        ] {
            let (hue, saturation, value) = rgb_to_hsv(rgb[0], rgb[1], rgb[2]);
            assert_eq!(hsv_to_rgb(hue, saturation, value), rgb, "colour {rgb:?}");
        }
    }

    #[test]
    fn hex_label_is_argb() {
        let mut picker = ColourPicker {
            hue: 0,
            saturation: 0,
            value: 0,
            alpha: 255,
            ..ColourPicker::default()
        };
        picker.set_rgba([255, 0, 255, 0xFF]);
        assert_eq!(picker.hex(), "0xFFFF00FF");
    }

    #[test]
    fn editing_a_preset_swatch_keeps_the_alpha_channel() {
        let mut picker = ColourPicker::default();
        picker.set_rgba([255, 0, 255, 128]);
        assert_eq!(picker.rgba(), [255, 0, 255, 128]);
    }
}
