//! The `fileSystem` table (and the classic `fs` alias).
//!
//! All operations are sandboxed to TPT's `scripts/` data directory, matching
//! `LuaFileSystem.cpp`.

use std::path::{Component, Path, PathBuf};

use mlua::{Lua, Table};

const SCRIPTS_DIR: &str = "scripts";

/// Resolves `user_path` safely inside `root`, preventing directory traversal
/// out of the sandbox.
fn sandbox_path(root: &Path, user_path: &str) -> Option<PathBuf> {
    let mut clean = PathBuf::new();
    for comp in Path::new(user_path).components() {
        match comp {
            Component::Normal(c) => clean.push(c),
            Component::CurDir => {}
            Component::ParentDir => {
                if !clean.pop() {
                    return None;
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                // Strip absolute / drive roots; sandbox treats everything relative to root.
            }
        }
    }
    Some(root.join(clean))
}

fn scripts_root() -> PathBuf {
    let root = PathBuf::from(SCRIPTS_DIR);
    let _ = std::fs::create_dir_all(&root);
    root
}

/// Builds the `fileSystem` table.
pub(super) fn build_filesystem_table(lua: &Lua) -> mlua::Result<Table> {
    let fs = lua.create_table()?;

    // fileSystem.list(dir) -> table of file names (1-indexed)
    fs.set(
        "list",
        lua.create_function(|lua, dir: Option<String>| -> mlua::Result<Table> {
            let root = scripts_root();
            let dir_str = dir.unwrap_or_default();
            let tbl = lua.create_table()?;
            if let Some(path) = sandbox_path(&root, &dir_str)
                && let Ok(entries) = std::fs::read_dir(path)
            {
                let mut index = 1;
                for entry in entries.flatten() {
                    if let Ok(name) = entry.file_name().into_string()
                        && name != "."
                        && name != ".."
                    {
                        tbl.raw_set(index, name)?;
                        index += 1;
                    }
                }
            }
            Ok(tbl)
        })?,
    )?;

    // fileSystem.exists(path) -> bool
    fs.set(
        "exists",
        lua.create_function(|_, path: String| -> mlua::Result<bool> {
            let root = scripts_root();
            Ok(sandbox_path(&root, &path)
                .map(|p| p.exists())
                .unwrap_or(false))
        })?,
    )?;

    // fileSystem.isFile(path) -> bool
    fs.set(
        "isFile",
        lua.create_function(|_, path: String| -> mlua::Result<bool> {
            let root = scripts_root();
            Ok(sandbox_path(&root, &path)
                .map(|p| p.is_file())
                .unwrap_or(false))
        })?,
    )?;

    // fileSystem.isDirectory(path) -> bool
    fs.set(
        "isDirectory",
        lua.create_function(|_, path: String| -> mlua::Result<bool> {
            let root = scripts_root();
            Ok(sandbox_path(&root, &path)
                .map(|p| p.is_dir())
                .unwrap_or(false))
        })?,
    )?;

    // fileSystem.isLink(path) -> bool
    fs.set(
        "isLink",
        lua.create_function(|_, path: String| -> mlua::Result<bool> {
            let root = scripts_root();
            Ok(sandbox_path(&root, &path)
                .and_then(|p| {
                    p.symlink_metadata()
                        .ok()
                        .map(|m| m.file_type().is_symlink())
                })
                .unwrap_or(false))
        })?,
    )?;

    // fileSystem.makeDirectory(path) -> bool
    fs.set(
        "makeDirectory",
        lua.create_function(|_, path: String| -> mlua::Result<bool> {
            let root = scripts_root();
            Ok(sandbox_path(&root, &path)
                .map(|p| std::fs::create_dir_all(p).is_ok())
                .unwrap_or(false))
        })?,
    )?;

    // fileSystem.removeDirectory(path) -> bool
    fs.set(
        "removeDirectory",
        lua.create_function(|_, path: String| -> mlua::Result<bool> {
            let root = scripts_root();
            Ok(sandbox_path(&root, &path)
                .map(|p| std::fs::remove_dir(p).is_ok())
                .unwrap_or(false))
        })?,
    )?;

    // fileSystem.removeFile(path) -> bool
    fs.set(
        "removeFile",
        lua.create_function(|_, path: String| -> mlua::Result<bool> {
            let root = scripts_root();
            Ok(sandbox_path(&root, &path)
                .map(|p| std::fs::remove_file(p).is_ok())
                .unwrap_or(false))
        })?,
    )?;

    // fileSystem.move(old, new, [replace]) -> bool
    fs.set(
        "move",
        lua.create_function(
            |_,
             (old_name, new_name, replace): (String, String, Option<bool>)|
             -> mlua::Result<bool> {
                let root = scripts_root();
                let Some(old_path) = sandbox_path(&root, &old_name) else {
                    return Ok(false);
                };
                let Some(new_path) = sandbox_path(&root, &new_name) else {
                    return Ok(false);
                };
                let replace = replace.unwrap_or(false);
                if !replace && new_path.exists() {
                    return Ok(false);
                }
                if replace && new_path.exists() {
                    let _ = std::fs::remove_file(&new_path);
                }
                Ok(std::fs::rename(old_path, new_path).is_ok())
            },
        )?,
    )?;

    // fileSystem.copy(old, new) -> bool
    fs.set(
        "copy",
        lua.create_function(
            |_, (old_name, new_name): (String, String)| -> mlua::Result<bool> {
                let root = scripts_root();
                let Some(old_path) = sandbox_path(&root, &old_name) else {
                    return Ok(false);
                };
                let Some(new_path) = sandbox_path(&root, &new_name) else {
                    return Ok(false);
                };
                Ok(std::fs::copy(old_path, new_path).is_ok())
            },
        )?,
    )?;

    Ok(fs)
}
