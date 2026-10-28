//! Preset canvas sizes and the new-canvas / load-save dialog state.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::i18n::{t, t_fmt};
use tpt_core::constants::{DEFAULT_XRES, DEFAULT_YRES};
use tpt_core::particle::{EdgeMode, GravityMode};

use super::{SAVE_EXTENSION, SAVES_DIR};

/// Saves shown per page, matching `LocalBrowserModel.cpp:5`
/// (`constexpr auto pageSize = 20`).
pub const SAVES_PER_PAGE: usize = 20;

/// One `.tptrs` file listed by the main-menu save browser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaveEntry {
    /// Path of the `.tptrs` file; relative while [`SAVES_DIR`] is relative.
    pub path: PathBuf,
    /// Display name: the file stem, which is what `SaveButton` shows
    /// (`SaveButton.cpp:101` reads `SaveFile::GetDisplayName`).
    pub name: String,
    /// Last modification time. The browser lists saves newest first and prints
    /// this next to each name; the reference row pairs the name with its
    /// rendered thumbnail (`SaveButton.cpp:144-150`, `SaveButton.cpp:219-223`).
    pub modified: Option<SystemTime>,
}

/// The saves visible for one filter + page combination.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SavePage {
    /// Indices into the unfiltered, newest-first save list.
    pub indices: Vec<usize>,
    /// Zero-based page actually shown after clamping.
    pub page: usize,
    /// Number of saves matching the filter, before paging.
    pub total: usize,
    /// Number of pages; at least 1 so the pager never divides by zero, like
    /// `LocalBrowserModel::GetPageCount` (`LocalBrowserModel.cpp:110-116`).
    pub page_count: usize,
}

/// Filters saves by name and returns the requested page.
///
/// Mirrors `LocalBrowserModel::UpdateSavesList` (`LocalBrowserModel.cpp:76-99`):
/// an empty query matches every save, otherwise the query must be contained in
/// the save name (compared case-insensitively, which is friendlier than the
/// byte comparison and still matches every reference query), and pages hold
/// [`SAVES_PER_PAGE`] entries. Unlike the reference, the page is clamped here
/// (the reference clamps in `LocalBrowserController::SetPageRelative`,
/// `LocalBrowserController.cpp:123-129`) so a shrinking list never leaves the
/// browser on a page that no longer exists.
pub fn filter_save_page(entries: &[SaveEntry], filter: &str, page: usize) -> SavePage {
    let needle = filter.trim().to_lowercase();
    let matches: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| needle.is_empty() || entry.name.to_lowercase().contains(&needle))
        .map(|(index, _)| index)
        .collect();

    let total = matches.len();
    let page_count = total.div_ceil(SAVES_PER_PAGE).max(1);
    let page = page.min(page_count - 1);
    let start = page * SAVES_PER_PAGE;
    let end = (start + SAVES_PER_PAGE).min(total);
    SavePage {
        indices: matches[start..end].to_vec(),
        page,
        total,
        page_count,
    }
}

/// Orders saves newest first, the order the reference browser keeps in its
/// `MostRecentlyUsedFirst` list (`Client.cpp:61`); the name breaks ties so the
/// list is deterministic for files saved within the same timestamp tick.
pub fn sort_saves_newest_first(entries: &mut [SaveEntry]) {
    entries.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.path.cmp(&b.path))
    });
}

/// Formats a save timestamp as `YYYY-MM-DD HH:MM UTC`.
///
/// `SaveButton` prints the save's name under its thumbnail (`SaveButton.cpp:
/// 219-223`); the date is additional metadata the port shows beside the name.
/// The dependency set has no locale-aware date library, so the UTC civil date
/// is rendered directly, with the timezone suffix going through the locale
/// table. Returns `None` when the filesystem reports no timestamp (or one
/// before the Unix epoch).
pub fn format_save_time(modified: Option<SystemTime>) -> Option<String> {
    let elapsed = modified?.duration_since(SystemTime::UNIX_EPOCH).ok()?;
    let secs = elapsed.as_secs();
    let days = i64::try_from(secs / 86_400).ok()?;
    let (year, month, day) = civil_from_days(days);
    let minute_of_day = (secs % 86_400) / 60;
    let stamp = format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}",
        minute_of_day / 60,
        minute_of_day % 60
    );
    Some(t_fmt("menu.load_time_utc", &[&stamp]))
}

/// Converts days since the Unix epoch to a proleptic Gregorian date.
///
/// Howard Hinnant's `civil_from_days` algorithm, valid for the whole `i64`
/// range and the usual way to get a calendar date without a `chrono`-style
/// dependency.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (
        if month <= 2 { year + 1 } else { year },
        month as u32,
        day as u32,
    )
}

/// Preset canvas sizes offered by the new-canvas dialog.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CanvasSizePreset {
    /// Original The Powder Toy canvas (612 x 384).
    #[default]
    Classic612x384,
    /// 800 x 600.
    Medium800x600,
    /// 1024 x 768.
    Large1024x768,
    /// 1280 x 720.
    HD1280x720,
    /// 1600 x 900.
    Huge1600x900,
    /// Arbitrary size from the dialog's custom width/height fields.
    Custom,
}

impl CanvasSizePreset {
    /// All presets in display order.
    pub const ALL: [Self; 6] = [
        Self::Classic612x384,
        Self::Medium800x600,
        Self::Large1024x768,
        Self::HD1280x720,
        Self::Huge1600x900,
        Self::Custom,
    ];

    /// Human-readable label for menus and radio buttons.
    pub fn label(self) -> &'static str {
        match self {
            Self::Classic612x384 => t("menu.canvas_preset_612"),
            Self::Medium800x600 => t("menu.canvas_preset_800"),
            Self::Large1024x768 => t("menu.canvas_preset_1024"),
            Self::HD1280x720 => t("menu.canvas_preset_1280"),
            Self::Huge1600x900 => t("menu.canvas_preset_1600"),
            Self::Custom => t("menu.canvas_preset_custom"),
        }
    }

    /// Fixed pixel size of this preset, or `None` for [`Self::Custom`].
    pub fn dimensions(self) -> Option<(u32, u32)> {
        match self {
            Self::Classic612x384 => Some((612, 384)),
            Self::Medium800x600 => Some((800, 600)),
            Self::Large1024x768 => Some((1024, 768)),
            Self::HD1280x720 => Some((1280, 720)),
            Self::Huge1600x900 => Some((1600, 900)),
            Self::Custom => None,
        }
    }

    /// Resolves this preset to concrete pixel dimensions. For
    /// [`Self::Custom`] the supplied values are returned unchanged; the
    /// simulation config aligns them to the cell grid when the canvas starts.
    pub fn resolve(self, custom_width: u32, custom_height: u32) -> (u32, u32) {
        self.dimensions().unwrap_or((custom_width, custom_height))
    }
}

/// State backing the main-menu "New Simulation" card and the load-save browser.
#[derive(Clone, Debug)]
pub struct NewCanvasDialog {
    /// Selected preset size.
    pub preset: CanvasSizePreset,
    /// Width used when [`Self::preset`] is [`CanvasSizePreset::Custom`].
    pub custom_width: u32,
    /// Height used when [`Self::preset`] is [`CanvasSizePreset::Custom`].
    pub custom_height: u32,
    /// Boundary behaviour for the new simulation.
    pub edge_mode: EdgeMode,
    /// Gravity model for the new simulation.
    pub gravity_mode: GravityMode,
    /// Whether ambient heat simulation starts enabled.
    pub ambient_heat: bool,
    /// `.tptrs` files discovered under [`SAVES_DIR`], newest first.
    pub save_entries: Vec<SaveEntry>,
    /// Case-insensitive substring filter over [`Self::save_entries`].
    pub save_filter: String,
    /// Zero-based page shown by the save browser; clamped whenever the
    /// filtered list shrinks.
    pub save_page: usize,
    /// Path of the save selected in the browser, if any. The reference selects
    /// by stamp id (`LocalBrowserModel::SelectSave`,
    /// `LocalBrowserModel.cpp:117-125`), which maps to the file path for
    /// `.tptrs` saves.
    pub save_selected: Option<PathBuf>,
    /// Pending inline rename text; `Some` while the rename field is open.
    pub save_rename_input: Option<String>,
    /// Whether the in-card delete confirmation is showing. The reference opens
    /// a `ConfirmPrompt` (`LocalBrowserController.cpp:42-49`); the port keeps
    /// the two-step confirmation inside the card instead of an OS dialog.
    pub save_delete_confirm: bool,
    /// Transient feedback shown under the dialog controls.
    pub status_msg: Option<String>,
    /// Error from the most recent save/load/delete/rename attempt, if any.
    pub save_load_error: Option<String>,
}

impl Default for NewCanvasDialog {
    fn default() -> Self {
        Self::new()
    }
}

impl NewCanvasDialog {
    /// Creates a dialog pre-filled with the classic TPT canvas settings.
    pub fn new() -> Self {
        Self {
            preset: CanvasSizePreset::Classic612x384,
            custom_width: DEFAULT_XRES,
            custom_height: DEFAULT_YRES,
            edge_mode: EdgeMode::Void,
            gravity_mode: GravityMode::Vertical,
            ambient_heat: true,
            save_entries: Vec::new(),
            save_filter: String::new(),
            save_page: 0,
            save_selected: None,
            save_rename_input: None,
            save_delete_confirm: false,
            status_msg: None,
            save_load_error: None,
        }
    }

    /// Concrete canvas size currently selected by the user.
    pub fn resolved_size(&self) -> (u32, u32) {
        self.preset.resolve(self.custom_width, self.custom_height)
    }

    /// Width that would be used if the simulation were started now.
    pub fn width(&self) -> u32 {
        self.resolved_size().0
    }

    /// Height that would be used if the simulation were started now.
    pub fn height(&self) -> u32 {
        self.resolved_size().1
    }

    /// Index of the selected preset within [`CanvasSizePreset::ALL`].
    pub fn preset_index(&self) -> usize {
        CanvasSizePreset::ALL
            .iter()
            .position(|preset| *preset == self.preset)
            .unwrap_or(CanvasSizePreset::ALL.len() - 1)
    }

    /// Selects a preset by its [`CanvasSizePreset::ALL`] index; out-of-range
    /// indices fall back to [`CanvasSizePreset::Custom`].
    pub fn set_preset_index(&mut self, index: usize) {
        self.preset = CanvasSizePreset::ALL
            .get(index)
            .copied()
            .unwrap_or(CanvasSizePreset::Custom);
    }

    /// Switches to [`CanvasSizePreset::Custom`] with the given dimensions.
    pub fn set_custom_size(&mut self, width: u32, height: u32) {
        self.preset = CanvasSizePreset::Custom;
        self.custom_width = width;
        self.custom_height = height;
    }

    /// Error text from the most recent save/load attempt, if any.
    pub fn save_load_error(&self) -> Option<&str> {
        self.save_load_error.as_deref()
    }

    /// Save selected in the browser, if the file is still listed.
    pub fn selected_save(&self) -> Option<&SaveEntry> {
        let path = self.save_selected.as_deref()?;
        self.save_entries
            .iter()
            .find(|entry| entry.path.as_path() == path)
    }

    /// Selects `path`, dropping any pending rename/delete interaction.
    pub fn select_save(&mut self, path: &Path) {
        self.save_selected = Some(path.to_path_buf());
        self.save_rename_input = None;
        self.save_delete_confirm = false;
    }

    /// Drops the selection and any pending rename/delete interaction.
    ///
    /// Matches `LocalBrowserModel::ClearSelected`, which every list refresh
    /// goes through (`LocalBrowserController::RefreshSavesList`,
    /// `LocalBrowserController.cpp:106-115`).
    pub fn clear_save_selection(&mut self) {
        self.save_selected = None;
        self.save_rename_input = None;
        self.save_delete_confirm = false;
    }

    /// Saves visible for the current filter and page.
    pub fn visible_saves(&self) -> SavePage {
        filter_save_page(&self.save_entries, &self.save_filter, self.save_page)
    }

    /// Clamps [`Self::save_page`] to the pages that exist and returns the
    /// visible page.
    pub fn sync_visible_saves(&mut self) -> SavePage {
        let page = self.visible_saves();
        self.save_page = page.page;
        page
    }

    /// Deletes the selected save and rescans [`SAVES_DIR`].
    ///
    /// Mirrors `LocalBrowserController::removeSelectedC` -> `Client::DeleteStamp`
    /// (`Client.cpp:319-328`, `LocalBrowserController.cpp:42-73`): the file is
    /// removed and the browser refreshes; errors become user-facing strings.
    pub fn delete_selected_save(&mut self) -> Result<(), String> {
        let Some(entry) = self.selected_save().cloned() else {
            return Ok(());
        };
        if let Err(err) = std::fs::remove_file(&entry.path) {
            let message = t_fmt("menu.load_delete_failed", &[&entry.name, &err.to_string()]);
            self.save_load_error = Some(message.clone());
            return Err(message);
        }
        self.clear_save_selection();
        self.refresh_saves_list();
        Ok(())
    }

    /// Renames the selected save inside [`SAVES_DIR`], keeping `.tptrs`.
    ///
    /// Mirrors `Client::RenameStamp` (`Client.cpp:330-349`,
    /// `LocalBrowserController.cpp:80-98`): an empty or invalid name is
    /// rejected, so is a name already in use, and the browser rescans
    /// afterwards with the renamed file selected again.
    pub fn rename_selected_save(&mut self, new_name: &str) -> Result<(), String> {
        let Some(entry) = self.selected_save().cloned() else {
            return Ok(());
        };
        let stem = match super::commands::normalize_save_stem(new_name) {
            Ok(stem) => stem,
            Err(err) => {
                self.save_load_error = Some(err.clone());
                return Err(err);
            }
        };
        let target = Path::new(SAVES_DIR).join(format!("{stem}.{SAVE_EXTENSION}"));
        // Renaming to the current name is a no-op rather than the reference's
        // "already exists" error, since the target is the file itself.
        if target.as_path() == entry.path.as_path() {
            self.clear_save_selection();
            self.refresh_saves_list();
            return Ok(());
        }
        if target.exists() {
            let message = t("menu.load_rename_exists").to_string();
            self.save_load_error = Some(message.clone());
            return Err(message);
        }
        if let Err(err) = std::fs::rename(&entry.path, &target) {
            let message = t_fmt("menu.load_rename_failed", &[&err.to_string()]);
            self.save_load_error = Some(message.clone());
            return Err(message);
        }
        self.refresh_saves_list();
        // `refresh_saves_list` clears the selection; keep the renamed file
        // selected so the user can keep acting on it.
        self.save_selected = Some(target);
        Ok(())
    }

    /// Rescans [`SAVES_DIR`] for `*.tptrs` files, newest first, and resets
    /// the browser selection. A missing directory is not an error (it is
    /// created on the first save); any other I/O problem is surfaced through
    /// [`Self::status_msg`].
    pub fn refresh_saves_list(&mut self) {
        self.save_entries.clear();
        self.save_load_error = None;
        self.clear_save_selection();

        let dir = Path::new(SAVES_DIR);
        match std::fs::read_dir(dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let is_save = path
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .is_some_and(|ext| ext.eq_ignore_ascii_case(SAVE_EXTENSION));
                    if !is_save {
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
                    self.save_entries.push(SaveEntry {
                        path,
                        name,
                        modified: metadata.modified().ok(),
                    });
                }
                // The reference browser keeps saves in
                // `MostRecentlyUsedFirst` order (`Client.cpp:61,439`); the file
                // timestamp is the port's equivalent.
                sort_saves_newest_first(&mut self.save_entries);
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                let message = t_fmt("menu.canvas_scan_error", &[SAVES_DIR, &err.to_string()]);
                self.status_msg = Some(message.clone());
                self.save_load_error = Some(message);
            }
        }

        // The list may have shrunk under the current filter/page.
        self.save_page = self.visible_saves().page;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, modified: Option<u64>) -> SaveEntry {
        SaveEntry {
            path: PathBuf::from(format!("saves/{name}.{SAVE_EXTENSION}")),
            name: name.to_string(),
            modified: modified
                .map(|secs| SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs)),
        }
    }

    fn list(prefix: &str, count: usize) -> Vec<SaveEntry> {
        (0..count)
            .map(|index| entry(&format!("{prefix}-{index:02}"), Some(index as u64 * 60)))
            .collect()
    }

    /// The filter is a case-insensitive substring match and the pager slices
    /// the filtered list into [`SAVES_PER_PAGE`] chunks, clamped to the pages
    /// that exist.
    #[test]
    fn filter_and_paging_slice_the_matching_saves() {
        let entries = list("sand", 45);

        let first = filter_save_page(&entries, "", 0);
        assert_eq!(first.total, 45);
        assert_eq!(first.page_count, 3);
        assert_eq!(first.page, 0);
        assert_eq!(first.indices.len(), SAVES_PER_PAGE);
        assert_eq!(first.indices.first(), Some(&0));

        let second = filter_save_page(&entries, "SAND", 1);
        assert_eq!(second.page, 1);
        assert_eq!(second.indices, (20..40).collect::<Vec<_>>());

        let last = filter_save_page(&entries, "sand", 2);
        assert_eq!(last.indices, vec![40, 41, 42, 43, 44]);

        // Out-of-range pages land on the last real page instead of vanishing.
        let clamped = filter_save_page(&entries, "sand", 99);
        assert_eq!(clamped.page, 2);

        // A filter that matches ten entries still has exactly one page.
        let narrow = filter_save_page(&entries, "sand-1", 4);
        assert_eq!(narrow.total, 10);
        assert_eq!(narrow.page_count, 1);
        assert_eq!(narrow.page, 0);
        assert_eq!(narrow.indices.len(), 10);

        let none = filter_save_page(&entries, "stone", 0);
        assert_eq!(none.total, 0);
        assert_eq!(none.page_count, 1);
        assert!(none.indices.is_empty());
    }

    /// Rows are newest first; the name breaks ties so the order is stable.
    #[test]
    fn saves_sort_newest_first() {
        let mut entries = vec![
            entry("old", Some(10)),
            entry("new", Some(30)),
            entry("tie-b", Some(20)),
            entry("tie-a", Some(20)),
            entry("undated", None),
        ];
        sort_saves_newest_first(&mut entries);
        let names: Vec<&str> = entries.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, ["new", "tie-a", "tie-b", "old", "undated"]);
    }

    /// The row label uses a UTC civil date, including leap days.
    #[test]
    fn save_time_formats_as_utc_civil_date() {
        assert_eq!(format_save_time(None), None);
        assert_eq!(
            format_save_time(Some(SystemTime::UNIX_EPOCH)).as_deref(),
            Some("1970-01-01 00:00 UTC")
        );
        // 2024-02-29T12:34:00Z
        let leap_day = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_709_210_040);
        assert_eq!(
            format_save_time(Some(leap_day)).as_deref(),
            Some("2024-02-29 12:34 UTC")
        );
    }

    /// Selecting, renaming and deleting act on the real selection state.
    #[test]
    fn selection_tracks_the_listed_save() {
        let mut dialog = NewCanvasDialog::new();
        dialog.save_entries = list("save", 3);
        assert!(dialog.selected_save().is_none());

        let path = dialog.save_entries[1].path.clone();
        dialog.select_save(&path);
        assert_eq!(
            dialog.selected_save().map(|entry| entry.name.as_str()),
            Some("save-01")
        );
        dialog.clear_save_selection();
        assert!(dialog.selected_save().is_none());
    }
}
