//! Stamp files (`.stm`): saved canvas regions that can be placed again.
//!
//! Ports `GameController::StampRegion` (`GameController.cpp:476-493`), the
//! stamp half of `Client` (`Client.cpp:287-328,350-380`) and the `L` / `K`
//! cases of `GameView::OnKeyPress` (`GameView.cpp:1583-1601`). `Ctrl+S` arms
//! the rubber band (`GameView::BeginStampSelection`, `GameView.cpp:1351-1360`)
//! and the captured rectangle is stored as `./stamps/<seconds><counter>.stm`
//! next to `./saves`. The payload is the same [`SaveData`] format a `.tptrs`
//! save uses, sized to the captured rectangle the way `Simulation::Save`
//! sizes a `GameSave` to its rectangle (`Simulation.cpp:341-459`); a stamp
//! only carries particles and wall cells because the placement path
//! ([`Region`]) places exactly those.
//!
//! TPT keeps the stamp list in `MostRecentlyUsedFirst` order
//! (`Client.cpp:439`); the port derives the order from the file timestamp,
//! newest first, like the `.tptrs` browser (`canvas_dialog`).

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use tpt_save::SaveData;

use crate::i18n::{t, t_fmt};

use super::selection::{Placement, Region};

/// Directory (relative to the process working directory) holding `.stm`
/// stamps. `Client.cpp:311` uses `STAMPS_DIR` beside the save directory.
pub const STAMPS_DIR: &str = "stamps";
/// File extension used by stamp files (`Client::RescanStamps`,
/// `Client.cpp:383`).
pub const STAMP_EXTENSION: &str = "stm";

/// One `.stm` file listed by the stamp browser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StampEntry {
    /// Path of the `.stm` file.
    pub path: PathBuf,
    /// Display name: the file stem, which `Client::GetStamp` reports as the
    /// stamp id (`Client.cpp:311-318`).
    pub name: String,
    /// Last modification time, used for the newest-first order.
    pub modified: Option<SystemTime>,
}

/// Builds the `.stm` payload for a captured region.
///
/// `Simulation::Save` (`Simulation.cpp:341-459`) writes a `GameSave` whose
/// dimensions are the saved rectangle and offsets every particle/sign into
/// it; the port stores the rectangle's particles, wall cells, air grids and
/// signs in a [`SaveData`] whose config carries the region size
/// ([`Region::to_save_data`]). Fans and rigid bodies are not part of a region,
/// so they stay empty.
pub fn stamp_save_data(region: &Region) -> SaveData {
    region.to_save_data()
}

/// Decodes a stamp file back into a [`Region`] for the placement path.
///
/// [`Region::from_save_data`] resizes the wall grid to `w * h` so a
/// hand-written file can never make [`super::selection::Placement`] index past
/// the end of its wall vector.
pub fn region_from_stamp(save: &SaveData) -> Region {
    Region::from_save_data(save)
}

/// `Client::AddStamp` (`Client.cpp:350-380`) names a stamp from the Unix
/// second and a collision counter, both lower-case hex, column widths 8 and 2:
/// `65a1b2c300.stm`.
pub fn stamp_file_name(seconds: u64, counter: u32) -> String {
    format!("{seconds:08x}{counter:02x}.{STAMP_EXTENSION}")
}

/// First free stamp name for `seconds`, starting at `first_counter`.
///
/// Mirrors the `while (true)` probe in `Client::AddStamp`
/// (`Client.cpp:364-374`): a taken name bumps the counter and retries.
pub fn next_stamp_name(seconds: u64, first_counter: u32, taken: impl Fn(&str) -> bool) -> String {
    let mut counter = first_counter;
    loop {
        let name = stamp_file_name(seconds, counter);
        if !taken(&name) {
            return name;
        }
        counter = counter.wrapping_add(1);
    }
}

/// Writes `region` into [`STAMPS_DIR`] and returns the created path.
///
/// Errors are user-facing strings, matching the rest of the save paths.
pub fn write_stamp(region: &Region, seconds: u64) -> Result<PathBuf, String> {
    let dir = Path::new(STAMPS_DIR);
    std::fs::create_dir_all(dir).map_err(|err| {
        t_fmt(
            "app.error_create_stamps_dir",
            &[STAMPS_DIR, &err.to_string()],
        )
    })?;
    let name = next_stamp_name(seconds, 0, |candidate| dir.join(candidate).exists());
    let path = dir.join(name);
    stamp_save_data(region)
        .save_to_path(&path)
        .map_err(|err| t_fmt("app.error_stamp_create", &[&err]))?;
    Ok(path)
}

/// Orders stamps newest first, with the name breaking ties so the order is
/// deterministic for files written within the same timestamp tick.
pub fn sort_stamps_newest_first(entries: &mut [StampEntry]) {
    entries.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.path.cmp(&b.path))
    });
}

/// Lists every `*.stm` file in `dir`, newest first.
///
/// A missing directory is an error here; [`StampBrowser::refresh`] treats
/// `NotFound` as an empty list because the directory is created by the first
/// stamp.
pub fn scan_stamps(dir: &Path) -> std::io::Result<Vec<StampEntry>> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        let is_stamp = path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case(STAMP_EXTENSION));
        if !is_stamp {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned());
        entries.push(StampEntry {
            path,
            name,
            modified: metadata.modified().ok(),
        });
    }
    sort_stamps_newest_first(&mut entries);
    Ok(entries)
}

/// State of the floating stamp browser (`GameController::OpenStamps`,
/// `GameController.cpp:1399-1410`).
///
/// The reference browser is a `LocalBrowserController` window backed by
/// `Client::stampIDs`; the port keeps the scanned list, one selected row and
/// the two-step delete confirmation, mirroring the `.tptrs` card
/// (`canvas_dialog::NewCanvasDialog`).
#[derive(Clone, Debug, Default)]
pub struct StampBrowser {
    /// Whether `K` opened the window.
    pub open: bool,
    /// `*.stm` files in [`STAMPS_DIR`], newest first.
    pub entries: Vec<StampEntry>,
    /// Row selected for deletion, if any.
    pub selected: Option<PathBuf>,
    /// Whether the in-window delete confirmation is showing.
    pub delete_confirm: bool,
    /// Error from the latest scan or delete attempt.
    pub error: Option<String>,
}

impl StampBrowser {
    /// Rescans [`STAMPS_DIR`], keeping the selection only while the file is
    /// still listed. A missing directory is not an error.
    pub fn refresh(&mut self) {
        self.error = None;
        match scan_stamps(Path::new(STAMPS_DIR)) {
            Ok(entries) => self.entries = entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => self.entries.clear(),
            Err(err) => {
                self.entries.clear();
                self.error = Some(t_fmt(
                    "app.error_scan_stamps",
                    &[STAMPS_DIR, &err.to_string()],
                ));
            }
        }
        if self
            .selected
            .as_deref()
            .is_some_and(|path| !self.entries.iter().any(|entry| entry.path == path))
        {
            self.selected = None;
            self.delete_confirm = false;
        }
    }

    /// Newest stamp, i.e. the entry `L` places (`GameView.cpp:1584-1586` reads
    /// `Client::GetStamps()[0]`).
    pub fn newest(&self) -> Option<&StampEntry> {
        self.entries.first()
    }

    /// Entry currently selected for deletion.
    pub fn selected(&self) -> Option<&StampEntry> {
        let path = self.selected.as_deref()?;
        self.entries.iter().find(|entry| entry.path == path)
    }

    /// Selects `path` and cancels any pending delete confirmation.
    pub fn select(&mut self, path: &Path) {
        self.selected = Some(path.to_path_buf());
        self.delete_confirm = false;
    }

    /// Deletes the selected stamp (`Client::DeleteStamp`,
    /// `Client.cpp:319-328`) and rescans the directory.
    pub fn delete_selected(&mut self) -> Result<(), String> {
        let Some(entry) = self.selected().cloned() else {
            return Ok(());
        };
        if let Err(err) = std::fs::remove_file(&entry.path) {
            let message = t_fmt("app.error_stamp_delete", &[&entry.name, &err.to_string()]);
            self.error = Some(message.clone());
            return Err(message);
        }
        self.selected = None;
        self.delete_confirm = false;
        self.refresh();
        Ok(())
    }
}

impl super::AppState {
    /// `GameView::BeginStampSelection` (`GameView.cpp:1351-1360`): `Ctrl+S`
    /// arms the stamp rubber band.
    pub fn begin_stamp_selection(&mut self) {
        self.begin_selection(super::selection::SelectMode::Stamp);
    }

    /// `GameController::StampRegion` (`GameController.cpp:476-493`): writes
    /// the captured rectangle to a new `.stm` file and reports the result on
    /// the HUD status line.
    ///
    /// A stamp capture never touches the canvas, unlike a cut. The port skips
    /// a selection that contains neither particles nor walls instead of
    /// writing a `.stm` that would place nothing.
    pub fn save_stamp_region(&mut self, region: &Region) -> Result<String, String> {
        if self.mode == super::GameMode::Sandbox3D {
            let message = t("app.error_stamp_3d").to_string();
            self.set_status(message.clone(), true);
            return Err(message);
        }
        let result = if region.is_empty() {
            Err(t("app.error_stamp_empty").to_string())
        } else {
            let seconds = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|elapsed| elapsed.as_secs())
                .unwrap_or(0);
            write_stamp(region, seconds).map(|path| {
                path.file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
        };
        match result {
            Ok(name) => {
                self.stamps.refresh();
                let message = t_fmt("app.status_stamp_created", &[&name]);
                self.set_status(message.clone(), false);
                Ok(message)
            }
            Err(message) => {
                self.set_status(message.clone(), true);
                Err(message)
            }
        }
    }

    /// `GameController::OpenStamps` (`GameController.cpp:1399-1410`): `K`
    /// rescans the stamps directory and opens the browser window.
    pub fn open_stamp_browser(&mut self) {
        self.cancel_selection();
        self.stamps.refresh();
        self.stamps.delete_confirm = false;
        self.stamps.open = true;
    }

    /// `GameView::OnKeyPress` `SDL_SCANCODE_L` (`GameView.cpp:1583-1592`):
    /// places the newest stamp under the cursor.
    ///
    /// Returns `false` only when there is no stamp at all — the reference case
    /// then falls through to `K` and opens the browser
    /// (`GameView.cpp:1594-1601`). A stamp that fails to load reports its
    /// error and still counts as handled.
    pub fn place_newest_stamp(&mut self) -> bool {
        self.stamps.refresh();
        let Some(entry) = self.stamps.newest().cloned() else {
            self.set_status(t("app.status_stamp_none"), true);
            return false;
        };
        let _ = self.load_stamp(&entry.path);
        true
    }

    /// `GameController::LoadStamp` (`GameController.cpp:430-433`) through
    /// `GameModel::SetPlaceSave`: reads the file back into the placement path
    /// so the next canvas click places it (`GameController::PlaceSave`,
    /// `GameController.cpp:253-263`).
    pub fn load_stamp(&mut self, path: &Path) -> Result<(), String> {
        let result = SaveData::load_from_path(path).and_then(|save| {
            let region = region_from_stamp(&save);
            if region.is_empty() {
                return Err(t("app.error_stamp_empty").to_string());
            }
            Ok(region)
        });
        match result {
            Ok(region) => {
                self.cancel_selection();
                let anchor = self.hover_pos.unwrap_or((0, 0));
                self.placement = Some(Placement { region, anchor });
                Ok(())
            }
            Err(err) => {
                let name = path
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string());
                let message = t_fmt("app.error_stamp_load", &[&name, &err]);
                self.set_status(message.clone(), true);
                Err(message)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use tpt_core::particle::GpuParticle;

    fn particle(x: f32, y: f32, ty: u32) -> GpuParticle {
        GpuParticle {
            ty,
            x,
            y,
            ..GpuParticle::default()
        }
    }

    fn sample_region() -> Region {
        Region {
            w: 3,
            h: 2,
            particles: vec![particle(0.0, 0.0, 1), particle(2.0, 1.0, 5)],
            walls: vec![0, 0, 0, 0, 7, 0],
            air: tpt_core::commands::RegionAir::default(),
            signs: Vec::new(),
        }
    }

    fn entry(name: &str, seconds: u64) -> StampEntry {
        StampEntry {
            path: PathBuf::from(format!("{name}.{STAMP_EXTENSION}")),
            name: name.to_string(),
            modified: Some(UNIX_EPOCH + Duration::from_secs(seconds)),
        }
    }

    /// A stamp written to `.stm` bytes reads back as the region that was
    /// captured: the same dimensions, particles and wall cells.
    #[test]
    fn stamp_file_round_trips_through_save_bytes() {
        let region = sample_region();
        let save = stamp_save_data(&region);
        let bytes = save.to_bytes().expect("serialize stamp");
        let decoded = SaveData::from_bytes(&bytes).expect("decode stamp");
        let back = region_from_stamp(&decoded);

        assert_eq!((back.w, back.h), (region.w, region.h));
        assert_eq!(back.particles, region.particles);
        assert_eq!(back.walls, region.walls);
        assert_eq!(back.is_empty(), region.is_empty());
    }

    /// The browser lists the newest file first and breaks same-second ties by
    /// name, so the order is stable across scans.
    #[test]
    fn stamps_are_ordered_newest_first_then_by_name() {
        let mut entries = vec![
            entry("b", 20),
            entry("a", 10),
            entry("aa", 30),
            entry("c", 30),
        ];
        sort_stamps_newest_first(&mut entries);
        let names: Vec<&str> = entries.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, ["aa", "c", "b", "a"]);
    }

    /// `Client::AddStamp` layout: 8 hex digits of Unix seconds then a 2-digit
    /// hex collision counter, and a taken name bumps the counter.
    #[test]
    fn stamp_names_match_the_reference_layout_and_skip_taken_ones() {
        assert_eq!(stamp_file_name(0x65a1_b2c3, 0), "65a1b2c300.stm");
        assert_eq!(stamp_file_name(0x65a1_b2c3, 0x1a), "65a1b2c31a.stm");
        let taken = stamp_file_name(0x65a1_b2c3, 0);
        let next = next_stamp_name(0x65a1_b2c3, 0, |candidate| candidate == taken);
        assert_eq!(next, "65a1b2c301.stm");
    }

    /// A hand-written payload with a short wall grid must not leave the
    /// placement path indexing past the end of the vector.
    #[test]
    fn region_from_stamp_resizes_a_short_wall_grid() {
        let mut save = stamp_save_data(&sample_region());
        save.wall_bmap.truncate(1);
        let region = region_from_stamp(&save);
        assert_eq!(
            region.walls.len(),
            (region.w as usize) * (region.h as usize)
        );
    }
}
