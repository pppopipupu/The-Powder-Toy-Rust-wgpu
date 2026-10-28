//! Property tool window (`PropertyWindow`) and the `AccessProperty` mapping.
//!
//! The reference game edits one scalar `Particle` field through
//! `AccessProperty::Parse` / `AccessProperty::Set`, with the configured
//! `(property, value)` pair stored in the `Prop.Type` / `Prop.Value`
//! preferences. The port exposes the same surface with the commands it already
//! has:
//!
//! * a stroke uploads its packed pixel list and
//!   [`tpt_sim::flood_kernel::set_property_stroke`] writes the field of every
//!   particle under the brush on the device. No part of the particle pool is
//!   copied back to the host, and the integer and floating-point fields take
//!   the same path.
//!
//! `Prop.Type` stores [`PropertyKind::index`]; the six indices that existed
//! before this module (`tmp`, `life`, `ctype`, `tmp2`, `tmp3`, `tmp4`) keep
//! their old values so an existing `prefs.json` keeps its selection. The same
//! indices are the kernel field ids, asserted by `apply.rs`'s tests.
//!
//! # Fill mode (`DrawFill`)
//!
//! `PropertyTool::DrawFill` calls `Simulation::flood_prop`, which walks the
//! pixel map with a scanline flood and rewrites every particle of the clicked
//! element type. The window exposes that mode as [`PropertyWindow::fill`]; while
//! it is on, a click floods instead of stamping the brush.
//!
//! The walk and the write run on the device
//! ([`tpt_sim::flood_kernel::flood_property`]): it marks the connected pixels,
//! compares the field through a device-side snapshot, and rewrites the matching
//! particles in place. No part of the map or the particle pool is copied back.
//! Deliberate differences from the reference:
//!
//! * the connected set is the 4-neighbour region of particles that share the
//!   clicked particle's element type *and* its current property value. The
//!   reference matches the element type only; requiring the sampled value keeps
//!   a fill from rewriting particles the user did not point at. Float fields
//!   (`temp`, `vx`, `vy`) compare bit-exactly, so a fill only crosses particles
//!   that hold the very same value;
//! * the walk grows by one ring per device dispatch and stops after `xres +
//!   yres` rings (the largest Manhattan distance on the canvas). The reference
//!   has no distance cap; a corridor that winds for longer than the canvas
//!   diameter is truncated here;
//! * `DrawFill` is a single-shot action here: the press sample floods and the
//!   rest of the drag is ignored. The reference re-floods on every mouse-move
//!   sample.
//!
//! `Prop.Fill` lives in [`PropertyWindow`] for the lifetime of the tool
//! configuration. [`Prefs`] (outside this module) has no field for it, so the
//! checkbox does not survive a restart; adding `Prefs::prop_fill` would need a
//! change in `crates/tpt-app/src/prefs`.
//!
//! # Sampling walls
//!
//! `SampleTool`'s shift behaviour opens this window from the particle under the
//! cursor (`PropertyWindow::TakePropertyFrom`). A wall has no editable particle
//! property, so the port reports the wall instead
//! ([`AppState::wall_sample_at`](super::AppState::wall_sample_at)): the window
//! records it in [`PropertyWindow::seeded_wall`], and a plain `SMPL` click on a
//! wall prints the same report on the HUD status line
//! (`brushes::apply_tool_stroke`).

#[allow(unused_imports)]
use crate::prefs::Prefs;

mod apply;
mod format;
mod kind;
mod window;

#[allow(unused_imports)]
pub use apply::*;
pub use format::*;
pub use kind::*;
pub use window::*;
