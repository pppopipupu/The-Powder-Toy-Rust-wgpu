//! Canvas sign overlay.
//!
//! TPT draws signs after the simulation (`Renderer::DrawSigns`) and this port
//! does the same at the egui layer: the text boxes are painted over the
//! simulation texture with the viewport's canvas-to-screen transform, so they
//! pan and zoom with the canvas. Nothing here reads the GPU.

use std::fmt::Write as _;
use std::sync::Arc;

use egui::text::Galley;
use egui::{Color32, CornerRadius, FontId, Pos2, Rect, Stroke, StrokeKind, Vec2};
use tpt_core::commands::SimCommand;
use tpt_core::constants::pmap_id;
use tpt_core::elements::ElementRegistry;
use tpt_core::particle::GpuParticle;
use tpt_core::sign::{JUSTIFY_MIDDLE, JUSTIFY_NONE, JUSTIFY_RIGHT};
use tpt_sim::SimBuffers;

use crate::app_state::AppState;
use crate::i18n::t_fmt;

use super::ViewTransform;

/// Inner padding of a sign box, the reference's `{ x + 3, y + 4 }` text origin
/// with the one-pixel border and a little breathing room (`Renderer.cpp:224`).
const SIGN_PADDING: Vec2 = Vec2::new(6.0, 4.0);

/// Origin the reference builds its online links against (`SERVER` in
/// `Config.template.h`, set to the full origin in release builds).
const SERVER: &str = "https://powdertoy.co.uk";

/// A drawn sign's screen rectangle, used to hit-test a click on the sign tool.
pub(super) struct SignBox {
    /// Index into `SimulationEngine::signs`.
    pub index: usize,
    /// Screen rectangle covered by the text box.
    pub rect: Rect,
}

/// `sign::Type` (`Sign.h:18-26`): the prefix of a special sign that decides
/// what a click on it does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum SignType {
    /// A plain text sign, possibly with `{placeholder}` expansion.
    #[default]
    Normal,
    /// `{c:<save id>|...}`: open the referenced save (`Type::Save`).
    Save,
    /// `{t:<thread id>|...}`: open the forum thread (`Type::Thread`).
    Thread,
    /// `{b|...}`: spark the particle under the anchor (`Type::Button`).
    Button,
    /// `{s:<query>|...}`: search the save browser (`Type::Search`).
    Search,
}

impl SignType {
    /// Colour `sign::getDisplayText` paints the label with (`Sign.cpp:134-145`).
    ///
    /// The reference prepends a font colour code to the drawable text - `\bt`
    /// `0x20AAFF` for a save, `\bl` `0xFF4B4B` for a thread, `\bo` `0xFFD820`
    /// for a button and `\bu` `0x9353D3` for a search (`RasterDrawMethodsImpl.h`
    /// parses those codes). An egui galley carries a single fallback colour, so
    /// the port tints the whole label instead of switching colour mid-string.
    fn colour(self) -> Color32 {
        match self {
            Self::Normal => Color32::WHITE,
            Self::Save => Color32::from_rgb(0x20, 0xAA, 0xFF),
            Self::Thread => Color32::from_rgb(0xFF, 0x4B, 0x4B),
            Self::Button => Color32::from_rgb(0xFF, 0xD8, 0x20),
            Self::Search => Color32::from_rgb(0x93, 0x53, 0xD3),
        }
    }
}

/// Splits a special sign (`sign::split`, `Sign.cpp:147-185`).
///
/// Returns the byte index of the `|` that separates the routing payload from
/// the displayed label and the sign's type, or `(0, Normal)` for a plain or
/// `{placeholder}` sign. The payload lives between the `:` and the `|`
/// (`str.Substr(3, si.first - 3)` in `GameController.cpp:568-581`), while
/// everything after the pipe is the label the sign shows.
pub(super) fn split_sign(text: &str) -> (usize, SignType) {
    let bytes = text.as_bytes();
    // `text.size() >= 4 && text.front() == '{' && text.back() == '}'`.
    if bytes.len() < 4 || bytes[0] != b'{' || bytes[bytes.len() - 1] != b'}' {
        return (0, SignType::Normal);
    }

    let pipe_from = |from: usize| -> Option<usize> {
        bytes
            .get(from..)?
            .iter()
            .position(|&byte| byte == b'|')
            .map(|offset| from + offset)
    };

    match bytes[1] {
        // `case 'c': case 't':` - a save/thread sign needs a `:`, a pipe at or
        // after index 4 and only digits in between (`Sign.cpp:152-166`).
        b'c' | b't' => {
            if bytes[2] != b':' {
                return (0, SignType::Normal);
            }
            let Some(pipe) = pipe_from(4) else {
                return (0, SignType::Normal);
            };
            if !bytes[3..pipe].iter().all(u8::is_ascii_digit) {
                return (0, SignType::Normal);
            }
            let kind = if bytes[1] == b'c' {
                SignType::Save
            } else {
                SignType::Thread
            };
            (pipe, kind)
        }
        // `case 'b': if (text[2] == '|')` (`Sign.cpp:169-174`).
        b'b' => {
            if bytes[2] == b'|' {
                (2, SignType::Button)
            } else {
                (0, SignType::Normal)
            }
        }
        // `case 's':` - the query may contain anything, only the pipe is
        // required (`Sign.cpp:176-180`).
        b's' => {
            if bytes[2] != b':' {
                return (0, SignType::Normal);
            }
            match pipe_from(3) {
                Some(pipe) => (pipe, SignType::Search),
                None => (0, SignType::Normal),
            }
        }
        _ => (0, SignType::Normal),
    }
}

/// One sign's laid-out label, the values `sign::getDisplayText` returns.
struct SignLabel {
    /// Index into `SimulationEngine::signs`.
    index: usize,
    /// Parsed special type, which picks the label colour.
    kind: SignType,
    /// Laid-out label text and its colour.
    galley: Arc<Galley>,
    /// Screen box the label fills, including the reference's padding.
    box_rect: Rect,
    /// Screen position of the anchor pixel (the little tick).
    anchor: Pos2,
}

/// Lays out every non-empty sign and returns the labels for painting and hit
/// testing (`sign::getDisplayText`, `Sign.cpp:15-145`).
///
/// A special sign shows only its label after the pipe and skips the anchor
/// sampling entirely (`Sign.cpp:24-30`); a normal sign expands its
/// `{t}`/`{p}`/`{aheat}` placeholders from the pixel it points at. The box sits
/// above the anchor when there is room and below it near the canvas top edge
/// (`y > 18`), and the horizontal placement follows the sign's `ju`.
fn layout_signs(painter: &egui::Painter, view: ViewTransform, state: &AppState) -> Vec<SignLabel> {
    let font_size = (10.0 * view.scale).clamp(9.0, 30.0);
    let font = FontId::proportional(font_size);

    let mut labels = Vec::with_capacity(state.engine.signs().len());
    for (index, sign) in state.engine.signs().iter().enumerate() {
        if sign.text.is_empty() {
            continue;
        }

        let (pipe, kind) = split_sign(&sign.text);
        let text = if pipe != 0 {
            // `text.Between(si.first + 1, text.size() - 1)`: the label after
            // the pipe, without the closing `}` (`Sign.cpp:26-30`).
            sign.text[pipe + 1..sign.text.len() - 1].to_owned()
        } else {
            let sample = sample_sign(&state.engine.bufs, sign.x, sign.y);
            expand_sign_text(&sign.text, &sample, &state.engine.registry)
        };

        let galley = painter.layout_no_wrap(text, font.clone(), kind.colour());
        let box_size = galley.size() + 2.0 * SIGN_PADDING;
        let anchor = view.to_screen(sign.x as f32, sign.y as f32);
        let left = if sign.ju == JUSTIFY_RIGHT {
            anchor.x - box_size.x
        } else if sign.ju == JUSTIFY_MIDDLE {
            anchor.x - box_size.x * 0.5
        } else {
            anchor.x
        };
        let min = if sign.y > 18 {
            egui::pos2(left, anchor.y - box_size.y - 4.0)
        } else {
            egui::pos2(left, anchor.y + 4.0)
        };
        labels.push(SignLabel {
            index,
            kind,
            galley,
            box_rect: Rect::from_min_size(min, box_size),
            anchor,
        });
    }
    labels
}

/// Draws every non-empty sign and returns the boxes for click hit-testing.
///
/// A special `{c:..}`/`{t:..}`/`{b|..}`/`{s:..}` sign is drawn in its type
/// colour, exactly like the reference's `\b` colour prefix (`Sign.cpp:134-145`).
/// The anchor pixel is marked like TPT's little pointer tick, and `None` paints
/// bare text from the anchor with no box. The font scales with the viewport but
/// is clamped so it stays legible when zoomed far out and does not become a
/// wall of text when zoomed in.
pub(super) fn draw_signs(
    painter: &egui::Painter,
    view: ViewTransform,
    state: &AppState,
) -> Vec<SignBox> {
    let labels = layout_signs(painter, view, state);
    let mut boxes = Vec::with_capacity(labels.len());
    for label in labels {
        let sign = &state.engine.signs()[label.index];
        let rect = label.box_rect;

        // `None` is TPT's bare-text mode: no backdrop, no border.
        if sign.ju != JUSTIFY_NONE {
            // Translucent black keeps the text readable over bright elements,
            // and the light border makes the box distinct from the simulation.
            painter.rect_filled(rect, CornerRadius::same(2), Color32::from_black_alpha(205));
            painter.rect_stroke(
                rect,
                CornerRadius::same(2),
                Stroke::new(1.0, Color32::from_rgb(0xC0, 0xC0, 0xC0)),
                StrokeKind::Inside,
            );
        }
        painter.galley(rect.min + SIGN_PADDING, label.galley, label.kind.colour());

        // Anchor tick, drawn after the box so it stays visible when the box
        // overlaps the pixel.
        let marker = Rect::from_center_size(
            label.anchor,
            Vec2::splat((view.scale * 2.0).clamp(2.0, 5.0)),
        );
        painter.rect_filled(
            marker,
            CornerRadius::ZERO,
            Color32::from_rgb(0xC0, 0xC0, 0xC0),
        );

        boxes.push(SignBox {
            index: label.index,
            rect,
        });
    }
    boxes
}

/// Index of the topmost special sign whose box contains `position`.
///
/// The hit test follows the paint order (`Sign.cpp:134-145` draws each sign as
/// one box), so the sign painted last wins when boxes overlap.
fn special_sign_at(
    painter: &egui::Painter,
    view: ViewTransform,
    state: &AppState,
    position: Pos2,
) -> Option<usize> {
    layout_signs(painter, view, state)
        .iter()
        .rev()
        .find(|label| label.kind != SignType::Normal && label.box_rect.contains(position))
        .map(|label| label.index)
}

/// State of a sign's anchor pixel, sampled once before its text is expanded.
#[derive(Clone, Debug, Default)]
struct SignSample {
    /// Particle occupying the pixel, if any (`sim->photons` / `sim->pmap`).
    particle: Option<GpuParticle>,
    /// Air pressure of the pixel's cell (`sim->pv`), `0` outside the canvas.
    pressure: f32,
    /// Ambient air heat in Celsius (`sim->hv - 273.15`), `0` outside the canvas.
    ambient_heat: f32,
}

/// Snapshots the pixel a sign points at from the host shadows.
///
/// `Sign::getDisplayText` reads `sim->photons`, `sim->pmap`, `sim->pv` and
/// `sim->hv` directly (`Sign.cpp:33-48`). The port draws from the host mirrors
/// because the canvas overlay runs every frame and must not stall on a device
/// read-back; energy particles win over matter in the same pixel, as in the
/// reference's `photons`-first check.
fn sample_sign(bufs: &SimBuffers, x: i32, y: i32) -> SignSample {
    // `Sign.cpp:36-48`: outside the canvas the particle stays null and both
    // measurements stay at their `0.0f` initialisers.
    let Some(pixel) = bufs.pixel_index(x, y) else {
        return SignSample::default();
    };

    let id = pmap_id(bufs.photons[pixel]).or_else(|| pmap_id(bufs.pmap[pixel]));
    let particle = id
        .filter(|&id| (id as usize) < bufs.capacity)
        .map(|id| bufs.export_particle(id as usize));

    SignSample {
        particle,
        pressure: bufs.pressure_at(x, y),
        ambient_heat: bufs.ambient_heat_at(x, y) - 273.15,
    }
}

/// Expands the `{...}` placeholders of `Sign::getDisplayText`
/// (`Sign.cpp:50-122`).
///
/// The reference formats temperatures through
/// `Format::Precision(Format::ShowPoint(v), 2)`; this port prints one decimal,
/// as specified for the port. `{type}` is the element's short registry name,
/// while `{ctype}`, `{life}`, `{tmp}` and `{tmp2}` print the particle's integer
/// values and fall back to `0` on an empty pixel. An unknown or unterminated
/// token is copied verbatim, matching the reference's literal `{token}` branch
/// and its `SplitBy('}')` failure path.
fn expand_sign_text(text: &str, sample: &SignSample, registry: &ElementRegistry) -> String {
    // `Sign.cpp:20-23`: no placeholder, no rewriting.
    if !text.contains('{') {
        return text.to_owned();
    }

    let mut out = String::with_capacity(text.len() + 16);
    let mut rest = text;
    while let Some(open) = rest.find('{') {
        let after_open = &rest[open + 1..];
        let Some(close) = after_open.find('}') else {
            // `Sign.cpp:117-120`: a missing `}` breaks the loop and the
            // remainder (including this `{`) is appended unchanged.
            break;
        };

        // `Sign.cpp:57`: everything before the `{` is copied first, so the
        // `Empty`/`empty` casing below sees the text produced so far.
        out.push_str(&rest[..open]);
        let token = &after_open[..close];
        let particle = sample.particle.as_ref();

        match token {
            // `Sign.cpp:60-69`: `{t}` / `{temp}`.
            "t" | "temp" => {
                let celsius = particle.map_or(0.0, |particle| particle.temp - 273.15);
                let _ = write!(out, "{celsius:.1}");
            }
            // `Sign.cpp:70-75`: `{p}` / `{pres}`.
            "p" | "pres" => {
                let _ = write!(out, "{:.1}", sample.pressure);
            }
            // `Sign.cpp:76-81`: `{a}` / `{aheat}`.
            "a" | "aheat" => {
                let _ = write!(out, "{:.1}", sample.ambient_heat);
            }
            // `Sign.cpp:82-87`: `{type}`.
            "type" => match particle {
                Some(particle) => out.push_str(&registry.get(particle.ty).name),
                None if out.is_empty() => out.push_str("Empty"),
                None => out.push_str("empty"),
            },
            // `Sign.cpp:88-93`: `{ctype}`, printed numerically by this port.
            "ctype" => {
                let _ = write!(out, "{}", particle.map_or(0, |particle| particle.ctype));
            }
            // `Sign.cpp:94-111`: `{life}`, `{tmp}`, `{tmp2}`.
            "life" => {
                let _ = write!(out, "{}", particle.map_or(0, |particle| particle.life));
            }
            "tmp" => {
                let _ = write!(out, "{}", particle.map_or(0, |particle| particle.tmp));
            }
            "tmp2" => {
                let _ = write!(out, "{}", particle.map_or(0, |particle| particle.tmp2));
            }
            // `Sign.cpp:112-115`: unknown placeholders stay literal.
            _ => {
                out.push('{');
                out.push_str(token);
                out.push('}');
            }
        }

        rest = &after_open[close + 1..];
    }
    // `Sign.cpp:122`: append whatever follows the last placeholder.
    out.push_str(rest);
    out
}

/// Screen position used as a sign's click target.
///
/// The box itself is the target, but a sign without text is never drawn, so
/// callers should fall back to the anchor pixel for those.
pub(super) fn box_hit(boxes: &[SignBox], position: Pos2) -> Option<usize> {
    boxes
        .iter()
        .rev()
        .find(|sign_box| sign_box.rect.contains(position))
        .map(|sign_box| sign_box.index)
}

/// egui memory key holding the special sign a mouse button pressed on.
fn pending_key(is_secondary: bool) -> egui::Id {
    egui::Id::unique(("tpt_sign_press", is_secondary))
}

/// Whether a press on a special sign is still waiting for its release.
///
/// The sign editor uses this to stay closed when a right click is routing a
/// special sign instead of editing it.
pub(super) fn press_pending(ctx: &egui::Context, is_secondary: bool) -> bool {
    ctx.data_mut(|data| data.get_temp::<usize>(pending_key(is_secondary)).is_some())
}

/// Routes a press/release on a special sign.
///
/// `GameController::MouseDown` (`GameController.cpp:531-540`) swallows a click
/// on a special sign before the tool sees it, and `MouseUp`
/// (`GameController.cpp:553-591`) then runs the sign's action when the release
/// is still over the same sign. The sign tool's left button is the documented
/// exception: it opens the editor instead of firing the action.
///
/// Returns `true` when the event belongs to the sign, so the caller must not
/// start or flush a paint stroke.
// The arguments are the caller's event-routing context; bundling them into a
// struct would only rename the same list.
#[allow(clippy::too_many_arguments)]
pub(super) fn handle_special_sign_click(
    ctx: &egui::Context,
    painter: &egui::Painter,
    view: ViewTransform,
    state: &mut AppState,
    frame: &super::PointerFrame,
    pointer: Option<Pos2>,
    is_secondary: bool,
    sign_tool: bool,
) -> bool {
    // `Identifier != "DEFAULT_UI_SIGN" || button != SDL_BUTTON_LEFT`.
    if sign_tool && !is_secondary {
        return false;
    }
    if !frame.pointer_in_view {
        return false;
    }
    let pressed = if is_secondary {
        frame.secondary_pressed
    } else {
        frame.primary_pressed
    };
    let released = if is_secondary {
        frame.secondary_released
    } else {
        frame.primary_released
    };

    if pressed {
        let Some(position) = pointer else {
            return false;
        };
        let Some(index) = special_sign_at(painter, view, state, position) else {
            return false;
        };
        // `GameController::MouseDown` runs the Lua button handler before the
        // sign check, and a handler that claims the click also suppresses the
        // sign action (`GameController.cpp:526-533`).
        if let Some((x, y)) = frame.sim_pos {
            let button = if is_secondary { 2 } else { 1 };
            if state.lua.fire_mouse_down(x, y, button, &mut state.engine) {
                return true;
            }
        }
        ctx.data_mut(|data| data.insert_temp(pending_key(is_secondary), index));
        return true;
    }

    if released {
        let pending = ctx.data_mut(|data| data.remove_temp::<usize>(pending_key(is_secondary)));
        let Some(index) = pending else {
            return false;
        };
        // `GameController.cpp:556-558`: the release must still be on the sign,
        // otherwise the click is dropped without running its action.
        if let Some(position) = pointer
            && special_sign_at(painter, view, state, position) == Some(index)
        {
            activate_sign(state, index);
        }
        return true;
    }

    // A press and its release can arrive in the same frame; drop the entry once
    // the button is up so it cannot fire on a later click.
    let down = if is_secondary {
        frame.secondary_down
    } else {
        frame.primary_down
    };
    if !down {
        ctx.data_mut(|data| data.remove_temp::<usize>(pending_key(is_secondary)));
    }
    false
}

/// Runs a special sign's click action (`GameController.cpp:564-587`).
fn activate_sign(state: &mut AppState, index: usize) {
    let Some(sign) = state.engine.signs().get(index) else {
        return;
    };
    let (x, y) = (sign.x, sign.y);
    let text = sign.text.clone();
    let (pipe, kind) = split_sign(&text);
    if pipe == 0 {
        return;
    }
    // `str.Substr(3, si.first - 3)`: the routing payload between `:` and `|`.
    // A `{b|..}` button has no payload and its pipe sits before index 3.
    let payload = match kind {
        SignType::Button | SignType::Normal => "",
        _ => &text[3..pipe],
    };

    let url = match kind {
        SignType::Save => {
            // `OpenSavePreview(saveID, ...)`: the port has no online save
            // browser, so the referenced save page opens in the system
            // browser. `ToNumber<int>(true)` turns a malformed id into 0,
            // which the reference ignores.
            let save_id = payload.parse::<u64>().ok().filter(|&id| id != 0);
            let Some(save_id) = save_id else {
                return;
            };
            format!("{SERVER}/Browse/View.html?ID={save_id}")
        }
        SignType::Thread => {
            // `Platform::OpenURI(SERVER + "/Discussions/Thread/View.html?Thread=" ...)`.
            format!("{SERVER}/Discussions/Thread/View.html?Thread={payload}")
        }
        SignType::Search => {
            // `OpenSearch(str.Substr(3, si.first - 3))`: the reference opens
            // its online save search; the port sends the same query to the
            // website.
            format!("{SERVER}/Browse.html?Search_Query={}", url_encode(payload))
        }
        SignType::Button => {
            // `create_part(-1, foundSign.x, foundSign.y, PT_SPRK)`.
            state.engine.apply_command(SimCommand::SparkAt { x, y });
            return;
        }
        SignType::Normal => return,
    };

    if !open_uri(&url) {
        state.set_status(t_fmt("signs.link_failed", &[&url]), true);
    }
}

/// Opens `uri` in the system browser, the desktop port of `Platform::OpenURI`
/// (`common/platform/Windows.cpp:28-34`, `Linux.cpp:59-62`).
fn open_uri(uri: &str) -> bool {
    #[cfg(target_os = "windows")]
    let child = std::process::Command::new("cmd")
        .args(["/C", "start", "", uri])
        .spawn();
    #[cfg(target_os = "macos")]
    let child = std::process::Command::new("open").arg(uri).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let child = std::process::Command::new("xdg-open").arg(uri).spawn();
    #[cfg(not(any(unix, windows)))]
    let child: std::io::Result<std::process::Child> =
        Err(std::io::Error::other("no URI opener on this platform"));
    child.is_ok()
}

/// Percent-encodes a search query for the `Search_Query` URL parameter.
///
/// Unreserved characters (`RFC 3986`) pass through, a space becomes `+`, and
/// every other UTF-8 byte is `%XX`-escaped.
fn url_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for &byte in text.as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else if byte == b' ' {
            out.push('+');
        } else {
            let _ = write!(out, "%{byte:02X}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_core::constants::PT_DUST;

    fn sample(particle: Option<GpuParticle>, pressure: f32, ambient_heat: f32) -> SignSample {
        SignSample {
            particle,
            pressure,
            ambient_heat,
        }
    }

    #[test]
    fn expands_particle_and_air_fields() {
        let registry = ElementRegistry::new_default();
        let sample = sample(
            Some(GpuParticle {
                ty: PT_DUST,
                temp: 373.15,
                ctype: 7,
                life: 42,
                tmp: -3,
                tmp2: 11,
                ..GpuParticle::default()
            }),
            4.2,
            20.0,
        );

        assert_eq!(
            expand_sign_text(
                "t={t} p={p} a={aheat} type={type} ctype={ctype} life={life} tmp={tmp} tmp2={tmp2}",
                &sample,
                &registry,
            ),
            "t=100.0 p=4.2 a=20.0 type=DUST ctype=7 life=42 tmp=-3 tmp2=11"
        );
        assert_eq!(
            expand_sign_text("{temp}/{pres}/{a}", &sample, &registry),
            "100.0/4.2/20.0"
        );
    }

    #[test]
    fn empty_pixel_uses_the_zero_guard() {
        let registry = ElementRegistry::new_default();
        let sample = sample(None, 4.2, 20.0);

        assert_eq!(
            expand_sign_text("{t}|{ctype}|{life}|{tmp}|{tmp2}", &sample, &registry),
            "0.0|0|0|0|0"
        );
        // `{type}` keeps the reference's `Empty`/`empty` wording.
        assert_eq!(expand_sign_text("{type}", &sample, &registry), "Empty");
        assert_eq!(expand_sign_text("x{type}", &sample, &registry), "xempty");
        // Pressure and ambient heat come from the air cell, not the particle.
        assert_eq!(
            expand_sign_text("{p}/{aheat}", &sample, &registry),
            "4.2/20.0"
        );
    }

    #[test]
    fn unknown_and_unterminated_tokens_stay_literal() {
        let registry = ElementRegistry::new_default();
        let sample = sample(None, 0.0, 0.0);

        assert_eq!(expand_sign_text("a{wat}b", &sample, &registry), "a{wat}b");
        assert_eq!(expand_sign_text("a{t", &sample, &registry), "a{t");
        assert_eq!(expand_sign_text("{t}x", &sample, &registry), "0.0x");
    }

    #[test]
    fn split_parses_every_special_sign_type() {
        // `Sign.cpp:147-185`: the returned index is the pipe, and the payload
        // (save id, thread id, search query) sits between `:` and `|`.
        assert_eq!(split_sign("{c:123|My save}"), (6, SignType::Save));
        assert_eq!(split_sign("{t:45|Discuss}"), (5, SignType::Thread));
        assert_eq!(split_sign("{b|Spark me}"), (2, SignType::Button));
        assert_eq!(split_sign("{s:water save|Search}"), (13, SignType::Search));

        // The label is everything after the pipe without the closing brace.
        let (pipe, kind) = split_sign("{c:123|My save}");
        assert_eq!(kind, SignType::Save);
        let text = "{c:123|My save}";
        assert_eq!(&text[pipe + 1..text.len() - 1], "My save");
        let (pipe, _) = split_sign("{s:water save|Search}");
        assert_eq!(&"{s:water save|Search}"[3..pipe], "water save");

        // Empty labels are legal: `{b|}` and `{c:1|}`.
        assert_eq!(split_sign("{b|}"), (2, SignType::Button));
        assert_eq!(split_sign("{c:1|}"), (4, SignType::Save));
    }

    #[test]
    fn split_rejects_malformed_special_signs() {
        // A plain sign and the `{t}` placeholder are normal signs.
        assert_eq!(split_sign("hello"), (0, SignType::Normal));
        assert_eq!(split_sign("{t}"), (0, SignType::Normal));
        assert_eq!(split_sign(""), (0, SignType::Normal));

        // Save/thread signs require `:` and digits before the pipe; the pipe
        // must start at index 4 or later (`text.find('|', 4)`).
        assert_eq!(split_sign("{c:12x|bad}"), (0, SignType::Normal));
        assert_eq!(split_sign("{t:1a|bad}"), (0, SignType::Normal));
        assert_eq!(split_sign("{c:|empty}"), (0, SignType::Normal));
        assert_eq!(split_sign("{c:12}"), (0, SignType::Normal));
        assert_eq!(split_sign("{t|no digits}"), (0, SignType::Normal));
        // `{b}` without the pipe is not a button, and `{s}` without `:`/`|`
        // is not a search.
        assert_eq!(split_sign("{b}"), (0, SignType::Normal));
        assert_eq!(split_sign("{bogus}"), (0, SignType::Normal));
        assert_eq!(split_sign("{s:no pipe}"), (0, SignType::Normal));
        assert_eq!(split_sign("{x:1|other}"), (0, SignType::Normal));
        // A sign must be closed and at least four bytes long.
        assert_eq!(split_sign("{c:1|open"), (0, SignType::Normal));
        assert_eq!(split_sign("abc"), (0, SignType::Normal));
    }

    #[test]
    fn special_signs_use_the_reference_colours() {
        // `Sign.cpp:134-145` / `RasterDrawMethodsImpl.h:293-301`.
        assert_eq!(SignType::Save.colour(), Color32::from_rgb(0x20, 0xAA, 0xFF));
        assert_eq!(
            SignType::Thread.colour(),
            Color32::from_rgb(0xFF, 0x4B, 0x4B)
        );
        assert_eq!(
            SignType::Button.colour(),
            Color32::from_rgb(0xFF, 0xD8, 0x20)
        );
        assert_eq!(
            SignType::Search.colour(),
            Color32::from_rgb(0x93, 0x53, 0xD3)
        );
        assert_eq!(SignType::Normal.colour(), Color32::WHITE);
    }

    #[test]
    fn search_queries_are_url_encoded() {
        assert_eq!(url_encode("water save"), "water+save");
        assert_eq!(url_encode("a&b=c/1"), "a%26b%3Dc%2F1");
        assert_eq!(url_encode("safe-._~09"), "safe-._~09");
    }
}
