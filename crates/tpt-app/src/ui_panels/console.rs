//! Floating Lua console: log output, script editor, execution, command history
//! and the plain `!` commands.
//!
//! Ports `ConsoleView.cpp:34-96` / `ConsoleController.cpp` / `ConsoleModel.cpp`:
//! Up/Down recall the session's command history, repeated lines collapse, the
//! log keeps a bounded scrollback, and Enter runs the line the cursor is on.
//! Lines starting with `!` are handled by the port instead of Lua, matching the
//! plain command set of `CommandInterface::PlainCommand`
//! (`CommandInterface.cpp:56-92`, dispatch table at `CommandInterface.cpp:94-104`):
//! `!quit`, `!reset`, `!clear`, `!save`, `!load` and `h`/`help`.

use std::path::{Path, PathBuf};

use egui::{Key, Modifiers, RichText, vec2};

use crate::app_state::{AppState, ModalOutcome, PendingConsoleAction, SAVE_EXTENSION, SAVES_DIR};
use crate::i18n::{t, t_fmt};

use super::shared::TEXT_DIM;

/// Maximum number of log lines kept in the console scrollback.
///
/// The C++ console draws as many past commands as fit in its 150px window
/// (`ConsoleView.cpp:73-96`); the port keeps a bounded buffer so a long Lua
/// session cannot grow the log without limit.
const CONSOLE_LOG_LIMIT: usize = 500;

/// Stable id of the console's script editor, used to detect keyboard focus
/// before drawing the widget so Up/Down/Enter can be intercepted.
const CONSOLE_INPUT_ID: &str = "tpt_console_input";

/// Lua snippets offered as one-click templates in the console.
const LUA_TEMPLATES: [(&str, &str); 3] = [
    (
        "console.template_particle_ring",
        r#"-- Spawn a ring of dust particles around the canvas centre.
local w, h = sim.canvasSize()
local cx, cy = math.floor(w / 2), math.floor(h / 2)
local radius = math.min(w, h) * 0.30
for i = 0, 95 do
    local angle = (i / 96) * math.pi * 2
    local x = math.floor(cx + math.cos(angle) * radius)
    local y = math.floor(cy + math.sin(angle) * radius)
    sim.partCreate(x, y, elem.DEFAULT_PT_DUST)
end
print("spawned a dust ring")
"#,
    ),
    (
        "console.template_custom_element",
        r#"-- Allocate a custom element and give it physics properties.
local id = elem.allocate("custom", "myelem")
if id then
    elem.property(id, "Description", "A scripted element")
    elem.property(id, "Color", 0x00FF88)
    elem.property(id, "Weight", 60)
    elem.property(id, "Falldown", 1)
    print("custom element id = " .. tostring(id))
end
"#,
    ),
    (
        "console.template_explosion_wave",
        r#"-- Expanding explosion wave made of fire shells.
local w, h = sim.canvasSize()
local cx, cy = math.floor(w / 2), math.floor(h / 2)
for r = 0, 48, 4 do
    for i = 0, 47 do
        local angle = (i / 48) * math.pi * 2
        local x = math.floor(cx + math.cos(angle) * r)
        local y = math.floor(cy + math.sin(angle) * r)
        sim.partCreate(x, y, elem.DEFAULT_PT_FIRE)
    end
end
print("explosion wave fired")
"#,
    ),
];

/// Plain console commands, parsed without side effects so they can be tested.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ConsoleCommand {
    /// `!quit`: ask for confirmation, then leave the game.
    Quit,
    /// `!reset`: clear particles, walls, air and rigid bodies.
    Reset,
    /// `!clear`: drop the console scrollback.
    Clear,
    /// `!save [name]`: save with `name`, or prompt for one.
    Save(Option<String>),
    /// `!load [name]`: load a `.tptrs` save, or prompt for its name.
    Load(Option<String>),
    /// `h` / `help` / `!help`: list the commands.
    Help,
    /// An unrecognized `!` command; carries the original line.
    Unknown(String),
    /// Anything else is Lua and goes to the scripting engine.
    Lua,
}

/// Parses one console line.
///
/// Only `!`-prefixed lines (plus the bare `h`/`help` aliases) are plain
/// commands; the rest is Lua, as in `CommandInterface::Command`
/// (`CommandInterface.cpp:497-500`) which falls through to the script engine.
pub(crate) fn parse_console_line(line: &str) -> ConsoleCommand {
    let line = line.trim();
    if line == "h" || line == "help" {
        return ConsoleCommand::Help;
    }
    let Some(rest) = line.strip_prefix('!') else {
        return ConsoleCommand::Lua;
    };
    let (name, args) = match rest.trim().split_once(char::is_whitespace) {
        Some((name, args)) => (name, args.trim()),
        None => (rest.trim(), ""),
    };
    let arg = parse_argument(args);
    match name {
        "quit" => ConsoleCommand::Quit,
        "reset" => ConsoleCommand::Reset,
        "clear" => ConsoleCommand::Clear,
        "save" => ConsoleCommand::Save(arg),
        "load" => ConsoleCommand::Load(arg),
        "help" => ConsoleCommand::Help,
        _ => ConsoleCommand::Unknown(line.to_string()),
    }
}

/// Returns the command argument, dropping one pair of matching quotes so
/// `!save "my world"` saves under the spaces instead of literal quote marks.
fn parse_argument(args: &str) -> Option<String> {
    let args = args.trim();
    if args.is_empty() {
        return None;
    }
    let unquoted = if let Some(inner) = args
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
    {
        inner
    } else if let Some(inner) = args
        .strip_prefix('\'')
        .and_then(|value| value.strip_suffix('\''))
    {
        inner
    } else {
        args
    };
    Some(unquoted.to_string())
}

/// Removes and returns the last line of the editor.
///
/// `ConsoleView::DoKeyPress` (`ConsoleView.cpp:36-41`) executes the command
/// field and clears it; the port's editor is multiline, so Enter consumes the
/// current (last) line and leaves the lines above it in place.
fn take_last_line(input: &mut String) -> String {
    let trimmed = input.trim_end_matches('\n');
    let start = trimmed.rfind('\n').map_or(0, |index| index + 1);
    let line = trimmed[start..].to_string();
    input.truncate(start);
    line
}

/// Appends a line to the scrollback, trimming the oldest entries when the cap
/// is exceeded.
fn append_log(log: &mut Vec<String>, line: String) {
    log.push(line);
    if log.len() > CONSOLE_LOG_LIMIT {
        let overflow = log.len() - CONSOLE_LOG_LIMIT;
        log.drain(..overflow);
    }
}

/// Appends a line to the console's scrollback.
fn push_log(state: &mut AppState, line: impl Into<String>) {
    append_log(&mut state.lua.console_log, line.into());
}

/// Logs a failing plain command and opens the reusable error box with the
/// failing message (`ErrorMessage.cpp`).
fn report_error(state: &mut AppState, message: String) {
    push_log(state, t_fmt("console.error_prefix", &[&message]));
    state
        .modals
        .open_error(t("modal.error_title").to_string(), message);
}

/// Runs a Lua chunk, mirroring the original console's engine call.
fn run_lua(state: &mut AppState, code: &str) {
    match state.lua.exec_script(code, &mut state.engine) {
        Ok(output) => {
            if !output.is_empty() {
                push_log(state, output);
            }
        }
        Err(error) => push_log(state, t_fmt("console.error_prefix", &[error.as_str()])),
    }
}

/// Executes one console line: a plain command or a Lua chunk.
fn run_console_line(state: &mut AppState, line: &str) {
    match parse_console_line(line) {
        ConsoleCommand::Lua => run_lua(state, line),
        command => run_plain_command(state, command),
    }
}

/// Runs a parsed plain command.
fn run_plain_command(state: &mut AppState, command: ConsoleCommand) {
    match command {
        ConsoleCommand::Quit => {
            let id = state.modals.open_confirm(
                t("modal.quit_title").to_string(),
                t("modal.quit_message").to_string(),
                t("modal.confirm").to_string(),
            );
            state.console.pending = Some((id, PendingConsoleAction::Quit));
        }
        ConsoleCommand::Reset => {
            state.clear_sim();
            push_log(state, t("app.status_canvas_cleared").to_string());
        }
        ConsoleCommand::Clear => state.lua.console_log.clear(),
        ConsoleCommand::Save(Some(name)) => save_named(state, &name),
        ConsoleCommand::Save(None) => {
            let id = state.modals.open_prompt(
                t("modal.save_title").to_string(),
                t("modal.save_message").to_string(),
                state.save_name_input.clone(),
                t("modal.save_hint").to_string(),
                t("modal.save_action").to_string(),
            );
            state.console.pending = Some((id, PendingConsoleAction::SaveAs));
        }
        ConsoleCommand::Load(Some(name)) => load_named(state, &name),
        ConsoleCommand::Load(None) => {
            let id = state.modals.open_prompt(
                t("modal.load_title").to_string(),
                t("modal.load_message").to_string(),
                String::new(),
                t("modal.load_hint").to_string(),
                t("modal.load_action").to_string(),
            );
            state.console.pending = Some((id, PendingConsoleAction::LoadAs));
        }
        ConsoleCommand::Help => push_log(state, t("console.help").to_string()),
        ConsoleCommand::Unknown(command) => report_error(
            state,
            t_fmt("console.error_unknown_command", &[&command]),
        ),
        ConsoleCommand::Lua => {}
    }
}

/// Saves the live simulation under `name`, reporting the result in the log.
fn save_named(state: &mut AppState, name: &str) {
    match state.save_current(name) {
        Ok(message) => push_log(state, message),
        Err(error) => report_error(state, error),
    }
}

/// Resolves a `!load` argument to a file under [`SAVES_DIR`].
fn save_path(name: &str) -> PathBuf {
    let name = name.trim();
    if name.ends_with(&format!(".{SAVE_EXTENSION}")) {
        Path::new(SAVES_DIR).join(name)
    } else {
        Path::new(SAVES_DIR).join(format!("{name}.{SAVE_EXTENSION}"))
    }
}

/// Loads `name` into the engine, reporting failures through the error box.
fn load_named(state: &mut AppState, name: &str) {
    let path = save_path(name);
    match state.load_from_file(&path) {
        Ok(_) => push_log(
            state,
            t_fmt("app.status_loaded", &[&path.display().to_string()]),
        ),
        Err(error) => report_error(state, error),
    }
}

/// Applies the outcome of the modal the console opened, if this frame owns it.
fn handle_modal_result(state: &mut AppState) {
    let Some(result) = state.modal_result.take() else {
        return;
    };
    let Some((id, action)) = state.console.pending else {
        return;
    };
    if result.id != id {
        // A different panel's dialog closed on top; the console is still
        // waiting for its own prompt.
        state.console.pending = Some((id, action));
        return;
    }
    state.console.pending = None;
    match (action, result.outcome) {
        (PendingConsoleAction::Quit, ModalOutcome::Confirmed) => {
            state.exit_requested = true;
        }
        (PendingConsoleAction::SaveAs, ModalOutcome::Submitted(name)) => {
            save_named(state, &name);
        }
        (PendingConsoleAction::LoadAs, ModalOutcome::Submitted(name)) => {
            load_named(state, &name);
        }
        // Cancel/Escape leaves the simulation untouched.
        _ => {}
    }
}

pub(super) fn draw_lua_console(ctx: &egui::Context, state: &mut AppState) {
    // A modal may have closed last frame; run its action before drawing the
    // next console frame so save/load feedback appears in the log immediately.
    handle_modal_result(state);

    // `GameView.cpp:1412-1414` opens the console with SDL_SCANCODE_GRAVE and
    // `ConsoleView.cpp:28-32` closes it with the same key when
    // `GraveExitsConsole` is on - unless Shift turns the key into '~', which
    // must keep reaching the script editor. `matches_logically` does not
    // compare Shift, so it is checked before the event is consumed. While the
    // console is closed the key is only taken when no text field owns the
    // keyboard, mirroring SDL delivering it to the focused game view. An open
    // modal owns the keyboard (`MakeActiveWindow`), so the console does not
    // toggle underneath it.
    let shifted = ctx.input(|input| input.modifiers.shift);
    let mut focus_input = false;
    if state.prefs.grave_exits_console
        && state.modals.is_empty()
        && !shifted
        && (state.show_lua_console || !ctx.text_edit_focused())
        && ctx.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Backtick))
    {
        state.show_lua_console = !state.show_lua_console;
        // `ConsoleView` focuses its command field when the window opens.
        focus_input = state.show_lua_console;
    }
    if !state.show_lua_console {
        return;
    }

    let input_id = egui::Id::unique(CONSOLE_INPUT_ID);
    let input_focused = ctx.memory(|memory| memory.has_focus(input_id));

    // `ConsoleView::DoKeyPress`: Up/Down walk the command history, Enter runs
    // the current line. The events are consumed before the TextEdit is drawn,
    // so egui does not also move the caret or insert a newline. Shift+Enter
    // stays available for a literal newline in a multi-line script.
    let (recall_previous, recall_next, run_last_line) =
        if input_focused && state.modals.is_empty() {
            ctx.input_mut(|input| {
                let up = input.consume_key(Modifiers::NONE, Key::ArrowUp);
                let down = input.consume_key(Modifiers::NONE, Key::ArrowDown);
                let enter =
                    !input.modifiers.shift && input.consume_key(Modifiers::NONE, Key::Enter);
                (up, down, enter)
            })
        } else {
            (false, false, false)
        };

    if recall_previous
        && let Some(command) = state.console.history.previous(&state.lua_input)
    {
        state.lua_input = command;
    }
    if recall_next
        && let Some(command) = state.console.history.next()
    {
        state.lua_input = command;
    }
    if run_last_line {
        let line = take_last_line(&mut state.lua_input);
        if !line.trim().is_empty() {
            state.console.history.record(&line);
            run_console_line(state, &line);
        }
    }

    let mut open = true;
    let mut execute = false;
    let mut clear_log = false;
    let mut insert: Option<&'static str> = None;

    egui::Window::new(t("console.title"))
        .open(&mut open)
        .resizable(true)
        .default_size(vec2(680.0, 480.0))
        .show(ctx, |ui| {
            ui.label(RichText::new(t("console.subtitle")).small().color(TEXT_DIM));
            ui.separator();

            egui::ScrollArea::vertical()
                .id_salt("tpt_lua_log")
                .stick_to_bottom(true)
                .max_height(200.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if state.lua.console_log.is_empty() {
                        ui.label(
                            RichText::new(t("console.no_output"))
                                .italics()
                                .color(TEXT_DIM),
                        );
                    }
                    for line in &state.lua.console_log {
                        ui.label(RichText::new(line).monospace().size(12.0));
                    }
                });

            ui.separator();
            let editor = ui.add(
                egui::TextEdit::multiline(&mut state.lua_input)
                    .id(input_id)
                    .code_editor()
                    .desired_rows(8)
                    .desired_width(f32::INFINITY)
                    .hint_text(t("console.input_hint")),
            );
            if focus_input {
                editor.request_focus();
            }
            ui.label(RichText::new(t("console.enter_hint")).small().color(TEXT_DIM));

            ui.horizontal_wrapped(|ui| {
                if ui.button(t("console.execute")).clicked() {
                    execute = true;
                }
                if ui.button(t("console.clear_log")).clicked() {
                    clear_log = true;
                }
                ui.separator();
                ui.label(
                    RichText::new(t("console.templates"))
                        .small()
                        .color(TEXT_DIM),
                );
                for (key, code) in LUA_TEMPLATES {
                    if ui.small_button(t(key)).clicked() {
                        insert = Some(code);
                    }
                }
            });
        });

    if clear_log {
        state.lua.console_log.clear();
    }
    if let Some(code) = insert {
        if !state.lua_input.is_empty() && !state.lua_input.ends_with('\n') {
            state.lua_input.push('\n');
        }
        state.lua_input.push_str(code);
    }
    if execute {
        let code = std::mem::take(&mut state.lua_input);
        let code = code.trim().to_string();
        if code.is_empty() {
            push_log(state, t("console.error_empty_script").to_string());
        } else {
            state.console.history.record(&code);
            // A single-line entry is a command; a multi-line editor buffer is
            // one Lua chunk, matching the C++ "Execute" action.
            if code.contains('\n') {
                run_lua(state, &code);
            } else {
                run_console_line(state, &code);
            }
        }
    }

    state.show_lua_console = open;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_quit() {
        assert_eq!(parse_console_line("!quit"), ConsoleCommand::Quit);
        assert_eq!(parse_console_line("  !quit  "), ConsoleCommand::Quit);
    }

    #[test]
    fn parses_reset() {
        assert_eq!(parse_console_line("!reset"), ConsoleCommand::Reset);
    }

    #[test]
    fn parses_clear() {
        assert_eq!(parse_console_line("!clear"), ConsoleCommand::Clear);
    }

    #[test]
    fn parses_save_with_and_without_a_name() {
        assert_eq!(parse_console_line("!save"), ConsoleCommand::Save(None));
        assert_eq!(
            parse_console_line("!save demo"),
            ConsoleCommand::Save(Some("demo".into()))
        );
        assert_eq!(
            parse_console_line("!save   my simulation  "),
            ConsoleCommand::Save(Some("my simulation".into()))
        );
        assert_eq!(
            parse_console_line("!save \"quoted name\""),
            ConsoleCommand::Save(Some("quoted name".into()))
        );
    }

    #[test]
    fn parses_load_with_and_without_a_name() {
        assert_eq!(parse_console_line("!load"), ConsoleCommand::Load(None));
        assert_eq!(
            parse_console_line("!load demo.tptrs"),
            ConsoleCommand::Load(Some("demo.tptrs".into()))
        );
    }

    #[test]
    fn parses_help_aliases() {
        assert_eq!(parse_console_line("h"), ConsoleCommand::Help);
        assert_eq!(parse_console_line("help"), ConsoleCommand::Help);
        assert_eq!(parse_console_line("!help"), ConsoleCommand::Help);
    }

    #[test]
    fn unknown_bang_command_reports_the_original_line() {
        assert_eq!(
            parse_console_line("!bogus"),
            ConsoleCommand::Unknown("!bogus".into())
        );
    }

    #[test]
    fn anything_else_is_lua() {
        assert_eq!(parse_console_line("print('hello')"), ConsoleCommand::Lua);
        assert_eq!(
            parse_console_line("!not-a-command-with-args 1"),
            ConsoleCommand::Unknown("!not-a-command-with-args 1".into())
        );
    }

    #[test]
    fn enter_takes_the_last_editor_line() {
        let mut input = String::from("print(1)\nprint(2)\n");
        assert_eq!(take_last_line(&mut input), "print(2)");
        assert_eq!(input, "print(1)\n");

        assert_eq!(take_last_line(&mut input), "print(1)");
        assert_eq!(input, "");

        let mut input = String::new();
        assert_eq!(take_last_line(&mut input), "");
    }

    #[test]
    fn log_is_capped_at_the_scrollback_limit() {
        let mut log = Vec::new();
        for index in 0..CONSOLE_LOG_LIMIT + 3 {
            append_log(&mut log, index.to_string());
        }
        assert_eq!(log.len(), CONSOLE_LOG_LIMIT);
        assert_eq!(log.first().map(String::as_str), Some("3"));
        assert_eq!(
            log.last().map(String::as_str),
            Some((CONSOLE_LOG_LIMIT + 2).to_string().as_str())
        );
    }
}
