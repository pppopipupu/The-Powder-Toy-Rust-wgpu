//! Restored-window geometry and monitor clamping.
//!
//! Ports `LoadWindowPosition` (`PowderToy.cpp:54-69`) and the resizable-window
//! size handling from `PowderToySDL.cpp:212-216`: a saved rectangle that no
//! longer intersects any display is centred on a visible monitor instead of
//! being restored where it cannot be seen.

use winit::event_loop::ActiveEventLoop;

/// Upper bound for a restored window dimension, in physical pixels. A larger
/// value in a hand-edited prefs file is clamped to the monitor anyway.
const MAX_RESTORED_WINDOW_DIMENSION: u32 = 32_768;
/// One monitor's bounds in physical pixels (`SDL_GetDisplayBounds`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MonitorRect {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

/// Bounds of every attached monitor, in winit's physical coordinate space.
pub(super) fn monitor_rects(event_loop: &ActiveEventLoop) -> Vec<MonitorRect> {
    event_loop
        .available_monitors()
        .map(|monitor| {
            let position = monitor.position();
            let size = monitor.size();
            MonitorRect {
                x: position.x,
                y: position.y,
                width: size.width,
                height: size.height,
            }
        })
        .collect()
}

/// Length in physical pixels of the overlap of two 1-D spans.
fn overlap_len(start: i32, len: u32, other_start: i32, other_len: u32) -> u32 {
    let end = i64::from(start) + i64::from(len);
    let other_end = i64::from(other_start) + i64::from(other_len);
    (end.min(other_end) - i64::from(start).max(i64::from(other_start))).max(0) as u32
}

/// Clamps a window edge inside a monitor span, keeping the title bar reachable.
fn clamp_axis(start: i32, len: u32, bounds_start: i32, bounds_len: u32) -> i32 {
    let furthest = i64::from(bounds_start) + i64::from(bounds_len.saturating_sub(len));
    i64::from(start).clamp(i64::from(bounds_start), furthest) as i32
}

/// A restored window rectangle in physical pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WindowGeometry {
    pub(super) position: (i32, i32),
    pub(super) size: (u32, u32),
}

/// Restores window geometry onto a visible monitor.
///
/// Ports `LoadWindowPosition` (`PowderToy.cpp:54-69`) and the resizable-window
/// size handling from `PowderToySDL.cpp:212-216`: a saved rectangle that no
/// longer intersects any display (monitor removed, resolution changed, window
/// minimised to an off-screen coordinate) is centred on the first monitor
/// instead of being restored where it cannot be seen. The size is clamped to
/// that monitor so a stale prefs file cannot create a window larger than the
/// screen. Returns `None` when the platform reports no monitors.
pub(super) fn clamp_window_geometry(
    position: (i32, i32),
    size: (u32, u32),
    monitors: &[MonitorRect],
) -> Option<WindowGeometry> {
    let first = *monitors.first()?;
    let width = size.0.clamp(1, MAX_RESTORED_WINDOW_DIMENSION);
    let height = size.1.clamp(1, MAX_RESTORED_WINDOW_DIMENSION);

    let mut best = first;
    let mut best_area = 0u64;
    for monitor in monitors {
        let area = u64::from(overlap_len(position.0, width, monitor.x, monitor.width))
            * u64::from(overlap_len(position.1, height, monitor.y, monitor.height));
        if area > best_area {
            best_area = area;
            best = *monitor;
        }
    }

    let width = width.min(best.width).max(1);
    let height = height.min(best.height).max(1);
    let (x, y) = if best_area == 0 {
        (
            best.x + best.width.saturating_sub(width) as i32 / 2,
            best.y + best.height.saturating_sub(height) as i32 / 2,
        )
    } else {
        position
    };
    Some(WindowGeometry {
        position: (
            clamp_axis(x, width, best.x, best.width),
            clamp_axis(y, height, best.y, best.height),
        ),
        size: (width, height),
    })
}

/// Clamps a restored size to the monitors without a saved position.
pub(super) fn clamp_window_size(size: (u32, u32), monitors: &[MonitorRect]) -> (u32, u32) {
    let width = size.0.clamp(1, MAX_RESTORED_WINDOW_DIMENSION);
    let height = size.1.clamp(1, MAX_RESTORED_WINDOW_DIMENSION);
    let Some(max_width) = monitors.iter().map(|monitor| monitor.width).max() else {
        return (width, height);
    };
    let max_height = monitors
        .iter()
        .map(|monitor| monitor.height)
        .max()
        .unwrap_or(height);
    (width.min(max_width).max(1), height.min(max_height).max(1))
}
#[cfg(test)]
mod tests {
    use super::{MonitorRect, WindowGeometry, clamp_window_geometry, clamp_window_size};

    const PRIMARY: MonitorRect = MonitorRect {
        x: 0,
        y: 0,
        width: 1920,
        height: 1080,
    };
    const SECONDARY: MonitorRect = MonitorRect {
        x: 1920,
        y: 0,
        width: 1280,
        height: 1024,
    };

    fn geometry(position: (i32, i32), size: (u32, u32)) -> Option<WindowGeometry> {
        Some(WindowGeometry { position, size })
    }

    /// `LoadWindowPosition` (`PowderToy.cpp:54-69`): a saved window on a
    /// monitor that is still attached is restored untouched.
    #[test]
    fn visible_window_geometry_is_kept() {
        let monitors = [PRIMARY, SECONDARY];
        assert_eq!(
            clamp_window_geometry((2000, 100), (1000, 700), &monitors),
            geometry((2000, 100), (1000, 700))
        );
    }

    /// A position outside every display is moved onto a monitor instead of
    /// leaving the window unreachable.
    #[test]
    fn off_screen_window_is_centred_on_a_monitor() {
        let monitors = [PRIMARY, SECONDARY];
        assert_eq!(
            clamp_window_geometry((9000, -4000), (1200, 800), &monitors),
            geometry((360, 140), (1200, 800))
        );
    }

    /// A stale size larger than the chosen monitor is reduced to fit.
    #[test]
    fn oversized_window_is_clamped_to_the_monitor() {
        let monitors = [PRIMARY];
        assert_eq!(
            clamp_window_geometry((100, 100), (4000, 3000), &monitors),
            geometry((0, 0), (1920, 1080))
        );
        assert_eq!(clamp_window_size((4000, 3000), &monitors), (1920, 1080));
        // Without a monitor list the saved size survives unchanged.
        assert_eq!(clamp_window_size((4000, 3000), &[]), (4000, 3000));
        assert_eq!(clamp_window_geometry((0, 0), (800, 600), &[]), None);
    }
}
