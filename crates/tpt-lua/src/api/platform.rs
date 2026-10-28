//! The `platform` table (and the classic `plat` alias): `LuaPlatform.cpp`.
//!
//! `restart` asks the application to relaunch through a Lua global
//! ([`RESTART_REQUEST_GLOBAL`]) because the winit event loop owns process
//! lifetime; `openLink` uses the same shell/URL-handler invocation as
//! `ui_panels/viewport/signs.rs`; the clipboard calls go through `arboard`
//! (the cross-platform equivalent of TPT's SDL clipboard).

use arboard::Clipboard;
use mlua::{Lua, Table, Value};

/// Lua global the application drains to perform `platform.restart`.
pub const RESTART_REQUEST_GLOBAL: &str = "__tpt_restart_request";

/// `Platform::OpenURI` (`common/platform/Platform.cpp`): hands the URI to the
/// desktop's URL handler.
fn open_uri(uri: &str) -> bool {
    #[cfg(target_os = "windows")]
    let command = std::process::Command::new("cmd")
        .args(["/C", "start", "", uri])
        .spawn();
    #[cfg(target_os = "macos")]
    let command = std::process::Command::new("open").arg(uri).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let command = std::process::Command::new("xdg-open").arg(uri).spawn();
    #[cfg(not(any(target_os = "windows", target_os = "macos", unix)))]
    let command: std::io::Result<std::process::Child> = Err(std::io::Error::other(
        "opening links is not supported on this platform",
    ));
    command.is_ok()
}

/// `ClipboardPull` (`LuaPlatform.cpp:44-50`): returns the clipboard text, or an
/// empty string when the clipboard is unavailable.
fn clipboard_pull() -> String {
    Clipboard::new()
        .and_then(|mut clipboard| clipboard.get_text())
        .unwrap_or_default()
}

/// `ClipboardPush` (`LuaPlatform.cpp:52-57`): copies text, ignoring failures
/// like SDL's void return.
fn clipboard_push(text: &str) {
    if let Ok(mut clipboard) = Clipboard::new() {
        let _ = clipboard.set_text(text.to_owned());
    }
}

/// Builds the `platform` table (`LuaPlatform::Open`, `LuaPlatform.cpp:64-90`).
pub(super) fn build_platform_table(lua: &Lua) -> mlua::Result<Table> {
    let plat = lua.create_table()?;

    // platform.platform() -> string
    plat.set(
        "platform",
        lua.create_function(|_, ()| -> mlua::Result<&'static str> {
            Ok(if cfg!(target_os = "windows") {
                "WIN32"
            } else if cfg!(target_os = "macos") {
                "MACOSX"
            } else {
                "LIN64"
            })
        })?,
    )?;

    // platform.ident() -> string
    plat.set(
        "ident",
        lua.create_function(|_, ()| -> mlua::Result<String> {
            Ok(format!(
                "{}-{}-{}",
                std::env::consts::ARCH,
                std::env::consts::OS,
                "RUST"
            )
            .to_ascii_uppercase())
        })?,
    )?;

    // platform.releaseType() -> string
    plat.set(
        "releaseType",
        lua.create_function(|_, ()| -> mlua::Result<&'static str> { Ok("R") })?,
    )?;

    // platform.exeName() -> string
    plat.set(
        "exeName",
        lua.create_function(|_, ()| -> mlua::Result<String> {
            let exe = std::env::current_exe()
                .map_err(|e| mlua::Error::runtime(format!("could not get executable name: {e}")))?;
            Ok(exe.to_string_lossy().into_owned())
        })?,
    )?;

    // platform.restart() -> relaunch request (`LuaPlatform.cpp:36-41`).
    //
    // `Platform::DoRestart` replaces the process; the port cannot tear down the
    // winit loop from a Lua callback, so the request is published for the
    // application to service.
    plat.set(
        "restart",
        lua.create_function(|lua, ()| -> mlua::Result<()> {
            lua.globals().set(RESTART_REQUEST_GLOBAL, true)
        })?,
    )?;

    // platform.openLink(url) -> () (`LuaPlatform.cpp:43-49`).
    plat.set(
        "openLink",
        lua.create_function(|_, url: String| -> mlua::Result<()> {
            let _ = open_uri(&url);
            Ok(())
        })?,
    )?;

    // platform.clipboardCopy() -> string (`LuaPlatform.cpp:51-57`).
    plat.set(
        "clipboardCopy",
        lua.create_function(|lua, ()| -> mlua::Result<mlua::LuaString> {
            let text = clipboard_pull();
            lua.create_string(text)
        })?,
    )?;

    // platform.clipboardPaste(text) -> () (`LuaPlatform.cpp:59-65`).
    plat.set(
        "clipboardPaste",
        lua.create_function(|_, text: Option<String>| -> mlua::Result<Value> {
            clipboard_push(&text.unwrap_or_default());
            Ok(Value::Nil)
        })?,
    )?;

    Ok(plat)
}
