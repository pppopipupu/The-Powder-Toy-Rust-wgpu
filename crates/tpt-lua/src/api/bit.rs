//! The `bit` table: Lua BitOp 1.0.4.
//!
//! Ports `src/lua/LuaBit.cpp` (Mike Pall's Lua BitOp, MIT). Every function
//! converts its arguments to a 32-bit word with [`to_ubits`] and returns the
//! result as a signed 32-bit Lua integer, exactly like the reference's `barg` /
//! `BRET` macros (`LuaBit.cpp:58-89`).

use mlua::{Lua, MultiValue, Table, Value};

/// `2^52 + 2^51`, the magic constant that makes a double-to-integer conversion
/// round to nearest even when the low 32 bits of the IEEE representation are
/// taken (`LuaBit.cpp:70`).
const MAGIC: f64 = 6755399441055744.0;

/// Converts one argument to a 32-bit word the way `barg` does
/// (`LuaBit.cpp:58-89`).
///
/// Lua 5.4's `lua_tointegerx` succeeds for any integer-valued number and for
/// numeric strings (including hex numerals such as `"0xF0"`), so
/// [`Lua::coerce_integer`] is tried first. Other numbers use the magic-double
/// conversion, which truncates modulo 2^32 with round-to-nearest even — the
/// reference relies on that instead of a plain C cast.
fn to_ubits(lua: &Lua, value: &Value) -> mlua::Result<u32> {
    if let Some(number) = lua.coerce_integer(value.clone())? {
        return Ok(number as u32);
    }
    let Some(number) = lua.coerce_number(value.clone())? else {
        return Err(mlua::Error::runtime(format!(
            "number expected, got {}",
            value.type_name()
        )));
    };
    let adjusted = number + MAGIC;
    Ok(adjusted.to_bits() as u32)
}

/// `BRET` (`LuaBit.cpp:91-95`): push the word back as a signed 32-bit integer.
fn bret(word: u32) -> i64 {
    word as i32 as i64
}

fn tobit(lua: &Lua, value: Value) -> mlua::Result<i64> {
    Ok(bret(to_ubits(lua, &value)?))
}

fn bnot(lua: &Lua, value: Value) -> mlua::Result<i64> {
    Ok(bret(!to_ubits(lua, &value)?))
}

/// `BIT_OP` (`LuaBit.cpp:98-102`): folds every argument into the first.
fn fold(lua: &Lua, args: &MultiValue, op: fn(u32, u32) -> u32) -> mlua::Result<i64> {
    let mut iter = args.iter();
    let Some(first) = iter.next() else {
        return Err(mlua::Error::runtime("number expected, got no value"));
    };
    let mut word = to_ubits(lua, first)?;
    for value in iter {
        word = op(word, to_ubits(lua, value)?);
    }
    Ok(bret(word))
}

/// `BIT_SH` (`LuaBit.cpp:106-112`): the shift count is masked to 5 bits.
fn lshift(lua: &Lua, args: (Value, Value)) -> mlua::Result<i64> {
    let word = to_ubits(lua, &args.0)?;
    let count = to_ubits(lua, &args.1)? & 31;
    Ok(bret(word << count))
}

fn rshift(lua: &Lua, args: (Value, Value)) -> mlua::Result<i64> {
    let word = to_ubits(lua, &args.0)?;
    let count = to_ubits(lua, &args.1)? & 31;
    Ok(bret(word >> count))
}

fn arshift(lua: &Lua, args: (Value, Value)) -> mlua::Result<i64> {
    let word = to_ubits(lua, &args.0)? as i32;
    let count = to_ubits(lua, &args.1)? & 31;
    Ok(bret((word >> count) as u32))
}

fn rol(lua: &Lua, args: (Value, Value)) -> mlua::Result<i64> {
    let word = to_ubits(lua, &args.0)?;
    let count = to_ubits(lua, &args.1)? & 31;
    Ok(bret(word.rotate_left(count)))
}

fn ror(lua: &Lua, args: (Value, Value)) -> mlua::Result<i64> {
    let word = to_ubits(lua, &args.0)?;
    let count = to_ubits(lua, &args.1)? & 31;
    Ok(bret(word.rotate_right(count)))
}

fn bswap(lua: &Lua, value: Value) -> mlua::Result<i64> {
    let word = to_ubits(lua, &value)?;
    Ok(bret(
        (word >> 24) | ((word >> 8) & 0xff00) | ((word & 0xff00) << 8) | (word << 24),
    ))
}

/// `bit.tohex(x [, n])` (`LuaBit.cpp:124-141`).
fn tohex(lua: &Lua, args: MultiValue) -> mlua::Result<mlua::LuaString> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Err(mlua::Error::runtime("number expected, got no value"));
    };
    let mut word = to_ubits(lua, &first)?;
    let mut count = match iter.next() {
        None => 8,
        Some(Value::Nil) => {
            return Err(mlua::Error::runtime("number expected, got nil"));
        }
        Some(value) => to_ubits(lua, &value)?,
    };
    let mut digits = b"0123456789abcdef";
    if (count as i32) < 0 {
        count = (!count).wrapping_add(1);
        digits = b"0123456789ABCDEF";
    }
    if count > 8 {
        count = 8;
    }
    let mut buffer = [0u8; 8];
    let mut index = count as i32;
    while index > 0 {
        index -= 1;
        buffer[index as usize] = digits[(word & 15) as usize];
        word >>= 4;
    }
    lua.create_string(&buffer[..count as usize])
}

/// Builds the `bit` table (`luaopen_bit`, `LuaBit.cpp:143-179`).
pub(crate) fn build_bit_table(lua: &Lua) -> mlua::Result<Table> {
    let bit = lua.create_table()?;
    bit.set("tobit", lua.create_function(tobit)?)?;
    bit.set("bnot", lua.create_function(bnot)?)?;
    bit.set(
        "band",
        lua.create_function(|lua, args: MultiValue| fold(lua, &args, |a, b| a & b))?,
    )?;
    bit.set(
        "bor",
        lua.create_function(|lua, args: MultiValue| fold(lua, &args, |a, b| a | b))?,
    )?;
    bit.set(
        "bxor",
        lua.create_function(|lua, args: MultiValue| fold(lua, &args, |a, b| a ^ b))?,
    )?;
    bit.set("lshift", lua.create_function(lshift)?)?;
    bit.set("rshift", lua.create_function(rshift)?)?;
    bit.set("arshift", lua.create_function(arshift)?)?;
    bit.set("rol", lua.create_function(rol)?)?;
    bit.set("ror", lua.create_function(ror)?)?;
    bit.set("bswap", lua.create_function(bswap)?)?;
    bit.set("tohex", lua.create_function(tohex)?)?;
    Ok(bit)
}
