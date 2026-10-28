//! Floating element search window (TPT's `ElementSearchActivity`).
//!
//! The window is opened from the simulation view (`GameView.cpp:1458-1462`
//! binds a bare `E`; `GameView.cpp:325-328` has the "Search for elements"
//! button in the HUD) and lists every tool the palette can bind, ranked by how
//! well its name, description and menu section match the query.
//!
//! Ranking follows `ElementSearchActivity::searchTools`
//! (`ElementSearchActivity.cpp:60-138`): each tool is matched against its name
//! (haystack 0), its description (haystack 1) and its menu section description
//! (haystack 2); the best `(haystack, position)` pair per tool wins, where
//! position `0` is an exact match, `1` a prefix and `2` a substring
//! (`ElementSearchActivity.cpp:99-112`). The results are ordered by that pair
//! and then by the tool's position in the palette
//! (`ElementSearchActivity.cpp:74-84,138`).
//!
//! Activation binds the primary brush exactly like the palette does
//! (`ElementSearchActivity::SetActiveTool`, `ElementSearchActivity.cpp:
//! 192-210` → `GameController::SetActiveTool(0, tool)` →
//! `GameModel::SetActiveTool`, `GameModel.cpp:796-800`), and a pick closes the
//! window. `Enter` picks the selected row, which is the first ranked result
//! until the arrow keys move it (`ElementSearchActivity.cpp:240-258` picks
//! `firstResult`), and `Escape` closes the window.
//!
//! Port note: the window state (query, selected row, pending focus) is kept in
//! egui's temporary data instead of [`AppState`] because `AppState` has a
//! single constructor — the struct literal in `app_state/commands.rs` — which
//! this module must not touch. The open request travels through a process-wide
//! flag because [`open_element_search`] runs from the hotkey handler, before
//! an egui pass exists to store anything in.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};

use egui::{Color32, Key, Modifiers, RichText, Stroke};
use tpt_core::commands::RigidSpawnShape;
use tpt_core::constants::{PT_NONE, WL_ERASE};

use crate::app_state::{
    AppScreen, AppState, GameMode, SelectedTool, pack_deco_colour, persist_tool_settings,
};
use crate::i18n::t;

use super::bottom_bar::RIGID_MATERIALS;
use super::shared::{
    DECO_TOOL_BUTTONS, GUI_TOOL_BUTTONS, SIM_TOOL_BUTTONS, TEXT_DIM, contrasting_text,
    deco_tool_colour, element_color, element_name, element_rgb, menu_visible, rigid_shape_label,
};
use super::side_bar::category_label;

/// Height of one result row, in points.
const ROW_HEIGHT: f32 = 24.0;

/// Set by [`open_element_search`]; consumed by the next [`draw_element_search`].
static OPEN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Tool the palette can bind, exactly as [`super::bottom_bar`] enumerates it.
#[derive(Clone, Copy)]
enum Candidate {
    /// Element tool for the given element id.
    Element(u32),
    /// Wall tool for the given wall id.
    Wall(u8),
    /// Rigid-body spawn tool (port-only palette section).
    Rigid {
        /// Shape rasterized for the new rigid body.
        shape: RigidSpawnShape,
        /// Element the body is made of.
        elem_ty: u32,
    },
    /// A tool whose name, description and menu label are fixed strings.
    Named {
        /// Tool bound by a pick.
        tool: SelectedTool,
        /// Localized tool name.
        name: &'static str,
        /// Localized tool description.
        description: &'static str,
        /// Localized palette section label.
        menu: &'static str,
        /// Base colour of the tool's palette chip, `0x00RRGGBB`.
        colour: u32,
    },
}

/// Per-window UI state persisted between frames in egui's temporary data.
#[derive(Clone, Default)]
struct SearchUi {
    /// Current contents of the search box.
    query: String,
    /// Index into the ranked result list.
    selected: usize,
    /// Requests keyboard focus for the search box on the next frame.
    focus_pending: bool,
}

/// One row of the result list, resolved for display.
struct Row {
    /// Localized tool name.
    name: String,
    /// Localized or reference description.
    description: String,
    /// Base colour of the tool's palette chip, `0x00RRGGBB`.
    colour: u32,
}

/// Opens the element search window.
///
/// `GameView.cpp:1458-1462`: a bare `E` opens the search from the simulation
/// view; Ctrl+E cycles the edge mode instead. The window searches the 2D
/// palette, so the 3D sandbox never opens it.
pub(super) fn open_element_search(state: &mut AppState) {
    if state.screen == AppScreen::InGame && state.mode == GameMode::Powder2D {
        OPEN_REQUESTED.store(true, Ordering::Relaxed);
    }
}

/// Draws the floating element search window while it is open.
pub(super) fn draw_element_search(ctx: &egui::Context, state: &mut AppState) {
    // `ElementSearchActivity` is a `GameView` window: the title screen has no
    // element search.
    if state.screen != AppScreen::InGame || state.mode != GameMode::Powder2D {
        return;
    }

    let mut open = ctx.data_mut(|data| data.get_temp::<bool>(open_key()).unwrap_or(false));
    let mut search = ctx.data_mut(|data| data.get_temp::<SearchUi>(ui_key()).unwrap_or_default());
    if OPEN_REQUESTED.swap(false, Ordering::Relaxed) {
        open = true;
        // `ElementSearchActivity.cpp:37-57`: the window starts with an empty
        // query and focuses the search box.
        search = SearchUi {
            focus_pending: true,
            ..SearchUi::default()
        };
    }
    if !open {
        return;
    }

    // `ElementSearchActivity.cpp:240-258`: Enter activates the first result,
    // Escape closes the window. The search box owns the keyboard while it has
    // focus (`ElementSearchActivity` runs the text box as an input field), so
    // the list keys are only consumed then — otherwise typing in another
    // window would move the hidden selection. The arrow keys move the
    // selection (port addition — the reference has no list selection).
    let typing = ctx.memory(|memory| memory.has_focus(query_key()));
    let (enter, escape, up, down) = ctx.input_mut(|input| {
        let list_key =
            |input: &mut egui::InputState, key| typing && input.consume_key(Modifiers::NONE, key);
        (
            list_key(input, Key::Enter),
            input.consume_key(Modifiers::NONE, Key::Escape),
            list_key(input, Key::ArrowUp),
            list_key(input, Key::ArrowDown),
        )
    });

    let mut activate: Option<SelectedTool> = None;
    let mut close = escape;

    egui::Window::new(t("side_bar.element_search_title"))
        .id(egui::Id::unique("tpt_element_search_window"))
        .open(&mut open)
        .resizable(false)
        .collapsible(false)
        .default_pos(ctx.content_rect().center() - egui::vec2(150.0, 210.0))
        .default_width(300.0)
        .show(ctx, |ui| {
            let edit = ui.add(
                egui::TextEdit::singleline(&mut search.query)
                    .id(query_key())
                    .hint_text(t("side_bar.element_search_placeholder"))
                    .desired_width(f32::INFINITY),
            );
            if std::mem::take(&mut search.focus_pending) {
                edit.request_focus();
            }
            let query_changed = edit.changed();

            let candidates = candidates(state);
            let ranked = ranked_indices(&candidates, state, &search.query);
            if query_changed {
                search.selected = 0;
            }
            search.selected = search.selected.min(ranked.len().saturating_sub(1));
            if down && !ranked.is_empty() {
                search.selected = (search.selected + 1).min(ranked.len() - 1);
            }
            if up {
                search.selected = search.selected.saturating_sub(1);
            }

            ui.separator();
            if ranked.is_empty() {
                ui.label(RichText::new(t("side_bar.element_search_empty")).color(TEXT_DIM));
            } else {
                let mut results = egui::ScrollArea::vertical()
                    .id_salt("tpt_element_search_results")
                    .max_height(300.0)
                    .auto_shrink([false, true]);
                if up || down {
                    // Port addition: keep the keyboard selection inside the
                    // viewport, three rows below the top.
                    let stride = ROW_HEIGHT + ui.spacing().item_spacing.y;
                    let offset = (search.selected as f32 - 3.0) * stride;
                    results = results.vertical_scroll_offset(offset.max(0.0));
                }
                results.show_rows(ui, ROW_HEIGHT, ranked.len(), |ui, range| {
                    for rank in range {
                        let candidate = candidates[ranked[rank]];
                        let row = candidate.row(state);
                        let background = element_color(row.colour);
                        let text = RichText::new(&row.name)
                            .color(contrasting_text(background))
                            .strong();
                        let mut button = egui::Button::new(text).fill(background);
                        if rank == search.selected {
                            button = button.stroke(Stroke::new(2.0, Color32::WHITE));
                        }
                        // `ElementSearchActivity.cpp:138-181` draws one
                        // `ToolButton` per match; a click activates it.
                        let response = ui
                            .add_sized(egui::vec2(ui.available_width(), ROW_HEIGHT), button)
                            .on_hover_ui(|ui| {
                                ui.label(RichText::new(&row.name).strong());
                                ui.label(&row.description);
                            });
                        if response.clicked() {
                            activate = Some(candidate.tool());
                        }
                    }
                });
            }

            if enter && let Some(&index) = ranked.get(search.selected) {
                activate = Some(candidates[index].tool());
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button(t("options.ok")).clicked()
                    && let Some(&index) = ranked.get(search.selected)
                {
                    activate = Some(candidates[index].tool());
                }
                if ui.button(t("side_bar.element_search_close")).clicked() {
                    close = true;
                }
            });
        });

    if close {
        open = false;
    }
    if let Some(tool) = activate {
        bind_primary(state, tool);
        open = false;
    }

    if open {
        ctx.data_mut(|data| {
            data.insert_temp(open_key(), true);
            data.insert_temp(ui_key(), search);
        });
    } else {
        ctx.data_mut(|data| {
            data.remove_temp::<bool>(open_key());
            data.remove_temp::<SearchUi>(ui_key());
        });
    }
}

/// Binds `tool` to the primary mouse button.
///
/// `ElementSearchActivity::SetActiveTool` calls `GameController::SetActiveTool(
/// 0, tool)` (`ElementSearchActivity.cpp:192-210`), which forwards to
/// `GameModel::SetActiveTool` (`GameModel.cpp:796-800`). The palette applies
/// the same side effects when it binds a tool.
fn bind_primary(state: &mut AppState, tool: SelectedTool) {
    state.primary_tool = tool;
    // The single-particle tool spawns "the currently selected element", so
    // every element a pick binds becomes that element (as in
    // `bottom_bar::draw_tool_grid`).
    if let SelectedTool::Element(elem_ty) = tool
        && elem_ty != PT_NONE
    {
        state.prefs.spawn_element = elem_ty;
    }
    persist_tool_settings(state);
    // Binding `PROP` opens its window, like `PropertyTool::Select`.
    if tool == SelectedTool::Property {
        state.open_property_window();
    }
}

/// Builds the searchable tool list in palette order.
///
/// `GameController::OpenElementSearch` searches `GameModel::GetTools()`
/// (`GameController.cpp:1364-1372`), i.e. every tool the palette can bind: the
/// menu-visible elements, the walls and the simulation / GUI tool classes.
fn candidates(state: &AppState) -> Vec<Candidate> {
    let mut list = Vec::new();

    // Hidden elements never get a tool (`GameModel::InitTools` skips
    // `MenuVisible = 0`), so they are not searchable either.
    for definition in &state.engine.registry.elements {
        if definition.enabled && !definition.name.is_empty() && menu_visible(definition) {
            list.push(Candidate::Element(definition.id));
        }
    }

    for wall in &state.engine.walls {
        if wall.id != WL_ERASE {
            list.push(Candidate::Wall(wall.id));
        }
    }

    list.push(Candidate::Named {
        tool: SelectedTool::Eraser,
        name: "Eraser",
        description: t("bottom_bar.eraser_desc"),
        menu: t("side_bar.cat_walls"),
        colour: 0x0030_343C,
    });
    for (kind, label, colour, description) in SIM_TOOL_BUTTONS {
        list.push(Candidate::Named {
            tool: SelectedTool::SimTool(kind),
            name: label,
            description: t(description),
            menu: t("side_bar.cat_tools"),
            colour,
        });
    }
    for (tool, label, colour, description) in GUI_TOOL_BUTTONS {
        list.push(Candidate::Named {
            tool,
            name: label,
            description: t(description),
            menu: t("side_bar.cat_tools"),
            colour,
        });
    }
    let deco_colour = pack_deco_colour(state.prefs.deco_colour());
    for (mode, label, description) in DECO_TOOL_BUTTONS {
        list.push(Candidate::Named {
            tool: SelectedTool::Sign { mode },
            name: label,
            description: t(description),
            menu: t("side_bar.cat_tools"),
            colour: deco_tool_colour(mode, deco_colour),
        });
    }
    for material in RIGID_MATERIALS {
        for shape in [RigidSpawnShape::Box, RigidSpawnShape::Circle] {
            list.push(Candidate::Rigid {
                shape,
                elem_ty: material,
            });
        }
    }
    for (index, rule) in tpt_sim::gol_kernel::BUILTIN_GOL.iter().enumerate() {
        list.push(Candidate::Named {
            tool: SelectedTool::Gol { rule: index as u32 },
            name: rule.name,
            description: rule.description,
            menu: t("side_bar.cat_life"),
            colour: rule.colour,
        });
    }

    list
}

impl Candidate {
    /// Tool bound when the row is activated.
    fn tool(self) -> SelectedTool {
        match self {
            Self::Element(ty) => SelectedTool::Element(ty),
            Self::Wall(id) => SelectedTool::Wall(id),
            Self::Rigid { shape, elem_ty } => SelectedTool::RigidBody { shape, elem_ty },
            Self::Named { tool, .. } => tool,
        }
    }

    /// Name, description and menu section searched for this tool.
    ///
    /// Order matters: `ElementSearchActivity.cpp:125-132` pushes the name with
    /// relevance `0`, the description with `1` and the menu description with
    /// `2`.
    fn haystacks<'a>(self, state: &'a AppState) -> [Cow<'a, str>; 3] {
        match self {
            Self::Element(ty) => {
                let definition = state.engine.registry.get(ty);
                [
                    Cow::Borrowed(definition.name.as_str()),
                    Cow::Borrowed(definition.description.as_str()),
                    Cow::Borrowed(category_label(definition.category)),
                ]
            }
            Self::Wall(id) => match state.engine.walls.iter().find(|wall| wall.id == id) {
                Some(wall) => [
                    Cow::Borrowed(wall.name.as_str()),
                    Cow::Borrowed(wall.description.as_str()),
                    Cow::Borrowed(t("side_bar.cat_walls")),
                ],
                None => [
                    Cow::Borrowed(t("tools.tool_wall_fallback")),
                    Cow::Borrowed(t("tools.tool_wall_fallback")),
                    Cow::Borrowed(t("side_bar.cat_walls")),
                ],
            },
            Self::Rigid { shape, elem_ty } => [
                Cow::Owned(format!(
                    "{} {}",
                    rigid_shape_label(shape),
                    element_name(state, elem_ty)
                )),
                Cow::Borrowed(t("bottom_bar.rigid_body_desc")),
                Cow::Borrowed(t("side_bar.cat_rigid_bodies")),
            ],
            Self::Named {
                name,
                description,
                menu,
                ..
            } => [
                Cow::Borrowed(name),
                Cow::Borrowed(description),
                Cow::Borrowed(menu),
            ],
        }
    }

    /// Resolves the strings and colour shown for a result row.
    fn row(self, state: &AppState) -> Row {
        match self {
            Self::Element(ty) => {
                let definition = state.engine.registry.get(ty);
                Row {
                    name: definition.name.clone(),
                    description: definition.description.clone(),
                    colour: definition.colour,
                }
            }
            Self::Wall(id) => match state.engine.walls.iter().find(|wall| wall.id == id) {
                Some(wall) => Row {
                    name: wall.name.clone(),
                    description: wall.description.clone(),
                    colour: wall.colour,
                },
                None => Row {
                    name: t("tools.tool_wall_fallback").to_string(),
                    description: String::new(),
                    colour: 0x0080_8080,
                },
            },
            Self::Rigid { shape, elem_ty } => Row {
                name: format!(
                    "{} {}",
                    rigid_shape_label(shape),
                    element_name(state, elem_ty)
                ),
                description: t("bottom_bar.rigid_body_desc").to_string(),
                colour: element_rgb(state, elem_ty),
            },
            Self::Named {
                name,
                description,
                colour,
                ..
            } => Row {
                name: name.to_string(),
                description: description.to_string(),
                colour,
            },
        }
    }
}

/// Indices of the matching tools, best first.
fn ranked_indices(list: &[Candidate], state: &AppState, query: &str) -> Vec<usize> {
    let mut matches: Vec<(u8, u8, usize)> = Vec::new();
    for (index, candidate) in list.iter().enumerate() {
        let haystacks = candidate.haystacks(state);
        if let Some((origin, position)) = match_key(
            [
                haystacks[0].as_ref(),
                haystacks[1].as_ref(),
                haystacks[2].as_ref(),
            ],
            query,
        ) {
            matches.push((origin, position, index));
        }
    }
    // `ElementSearchActivity.cpp:74-84,138`: (haystack, needle position, tool
    // order), so a name match always outranks a description or menu match.
    matches.sort_unstable();
    matches.into_iter().map(|(_, _, index)| index).collect()
}

/// Best `(haystack, position)` pair of `haystacks` for `needle`.
fn match_key(haystacks: [&str; 3], needle: &str) -> Option<(u8, u8)> {
    let mut best: Option<(u8, u8)> = None;
    for (origin, haystack) in haystacks.iter().enumerate() {
        if let Some(position) = match_position(haystack, needle) {
            let key = (origin as u8, position);
            best = Some(best.map_or(key, |current| current.min(key)));
        }
    }
    best
}

/// Relevance of one haystack, mirroring `pushIfMatches`
/// (`ElementSearchActivity.cpp:99-112`): `0` exact, `1` prefix, `2`
/// substring.
///
/// Matching is ASCII case-insensitive: element names and descriptions are
/// ASCII, and the translated menu labels have no case to fold.
fn match_position(haystack: &str, needle: &str) -> Option<u8> {
    if haystack.eq_ignore_ascii_case(needle) {
        Some(0)
    } else if starts_with_ignore_ascii_case(haystack, needle) {
        Some(1)
    } else if contains_ignore_ascii_case(haystack, needle) {
        Some(2)
    } else {
        None
    }
}

/// ASCII case-insensitive [`str::starts_with`].
fn starts_with_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    haystack.len() >= needle.len()
        && haystack.as_bytes()[..needle.len()].eq_ignore_ascii_case(needle.as_bytes())
}

/// ASCII case-insensitive [`str::contains`].
fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    let haystack = haystack.as_bytes();
    let needle = needle.as_bytes();
    haystack.len() >= needle.len()
        && (0..=haystack.len() - needle.len())
            .any(|start| haystack[start..start + needle.len()].eq_ignore_ascii_case(needle))
}

/// egui id of the open flag in the context's temporary data.
fn open_key() -> egui::Id {
    egui::Id::unique("tpt_element_search_open")
}

/// egui id of the [`SearchUi`] in the context's temporary data.
fn ui_key() -> egui::Id {
    egui::Id::unique("tpt_element_search_ui")
}

/// egui id of the search text box, used to test whether it has keyboard focus.
fn query_key() -> egui::Id {
    egui::Id::unique("tpt_element_search_query")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_prefix_and_substring_rank_in_order() {
        assert_eq!(match_position("DUST", "dust"), Some(0));
        assert_eq!(match_position("DUST", "du"), Some(1));
        assert_eq!(match_position("SAND", "an"), Some(2));
        assert_eq!(match_position("SAND", "zz"), None);
    }

    #[test]
    fn name_matches_outrank_description_and_menu_matches() {
        assert_eq!(
            match_key(["DUST", "a powder", "Powders"], "dust"),
            Some((0, 0))
        );
        assert_eq!(
            match_key(["SAND", "made of dust", "Powders"], "dust"),
            Some((1, 2))
        );
        assert_eq!(
            match_key(["SAND", "a powder", "Dusty things"], "dust"),
            Some((2, 1))
        );
    }

    #[test]
    fn empty_query_matches_every_tool() {
        assert_eq!(match_key(["DUST", "a powder", "Powders"], ""), Some((0, 1)));
    }

    #[test]
    fn non_ascii_labels_match_bytewise() {
        assert_eq!(match_position("元素搜索", "元素"), Some(1));
        assert_eq!(match_position("元素搜索", "搜索"), Some(2));
        assert_eq!(match_position("元素搜索", "元素搜索"), Some(0));
    }
}
