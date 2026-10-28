//! The `bz2` table: bzip2 compression (`LuaBz2.cpp`).
//!
//! The reference wraps libbz2 (`src/bzip2/bz2wrap.cpp`); this port uses the
//! pure-Rust `bzip2` crate. Result codes and the `(nil, code, message)` error
//! shape match the reference: `compress` reports `NOMEM`/`LIMIT`, `decompress`
//! additionally reports `TYPE`/`BAD`/`EOF`, and every decompression failure
//! carries the same `"corrupted stream"` text (`LuaBz2.cpp:14-44`).

use std::io::{Read, Write};

use bzip2::Compression;
use bzip2::read::BzDecoder;
use bzip2::write::BzEncoder;
use mlua::{Lua, MultiValue, Table, Value};

/// `BZ2WCompressResult` (`bz2wrap.h:6-11`).
const COMPRESS_NOMEM: i64 = 1;
const COMPRESS_LIMIT: i64 = 2;

/// `BZ2WDecompressResult` (`bz2wrap.h:14-22`).
const DECOMPRESS_NOMEM: i64 = 1;
const DECOMPRESS_LIMIT: i64 = 2;
const DECOMPRESS_TYPE: i64 = 3;
const DECOMPRESS_BAD: i64 = 4;
const DECOMPRESS_EOF: i64 = 5;

/// Error marker used by [`CappedWriter`] when the size limit is reached.
const LIMIT_MARKER: &str = "tpt-lua bz2 size limit";

/// `(nil, code, message)` — the reference's `RETURN_ERR` macro
/// (`LuaBz2.cpp:18`).
fn error_values(lua: &Lua, code: i64, message: &str) -> mlua::Result<MultiValue> {
    Ok(MultiValue::from_vec(vec![
        Value::Nil,
        Value::Integer(code),
        Value::String(lua.create_string(message)?),
    ]))
}

/// Writer that refuses to grow past `limit` bytes (`0` = unlimited), which is
/// how the reference enforces `maxSize` while compressing
/// (`bz2wrap.cpp:31-38`).
struct CappedWriter {
    data: Vec<u8>,
    limit: usize,
}

impl Write for CappedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.limit != 0 && self.data.len() + buf.len() > self.limit {
            return Err(std::io::Error::other(LIMIT_MARKER));
        }
        self.data.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn compress(lua: &Lua, args: MultiValue) -> mlua::Result<MultiValue> {
    let mut iter = args.into_iter();
    let Some(Value::String(source)) = iter.next() else {
        return Err(mlua::Error::runtime("string expected"));
    };
    let max_size = match iter.next() {
        Some(value) => super::value_to_i64(&value)?.max(0) as usize,
        None => 0,
    };

    let writer = CappedWriter {
        data: Vec::new(),
        limit: max_size,
    };
    let mut encoder = BzEncoder::new(writer, Compression::best());
    if let Err(error) = encoder.write_all(&source.as_bytes()) {
        return if error.to_string() == LIMIT_MARKER {
            error_values(lua, COMPRESS_LIMIT, "size limit exceeded")
        } else {
            error_values(lua, COMPRESS_NOMEM, "out of memory")
        };
    }
    match encoder.finish() {
        Ok(writer) => Ok(MultiValue::from_vec(vec![Value::String(
            lua.create_string(&writer.data)?,
        )])),
        Err(error) if error.to_string() == LIMIT_MARKER => {
            error_values(lua, COMPRESS_LIMIT, "size limit exceeded")
        }
        Err(_) => error_values(lua, COMPRESS_NOMEM, "out of memory"),
    }
}

fn decompress(lua: &Lua, args: MultiValue) -> mlua::Result<MultiValue> {
    let mut iter = args.into_iter();
    let Some(Value::String(source)) = iter.next() else {
        return Err(mlua::Error::runtime("string expected"));
    };
    let max_size = match iter.next() {
        Some(value) => super::value_to_i64(&value)?.max(0) as usize,
        None => 0,
    };

    let bytes = source.as_bytes();
    let decoder = BzDecoder::new(std::io::Cursor::new(bytes));
    // `take(limit + 1)` lets the port detect a stream that would exceed the
    // limit without materialising it (`bz2wrap.cpp:73-79`).
    let mut reader = decoder.take(if max_size == 0 {
        u64::MAX
    } else {
        max_size as u64 + 1
    });
    let mut output = Vec::new();
    match reader.read_to_end(&mut output) {
        Ok(_) => {
            if max_size != 0 && output.len() > max_size {
                return error_values(lua, DECOMPRESS_LIMIT, "size limit exceeded");
            }
            Ok(MultiValue::from_vec(vec![Value::String(
                lua.create_string(&output)?,
            )]))
        }
        Err(error) => {
            let code = match error
                .get_ref()
                .and_then(|inner| inner.downcast_ref::<bzip2::Error>())
            {
                Some(bzip2::Error::DataMagic) => DECOMPRESS_TYPE,
                Some(bzip2::Error::Data) => DECOMPRESS_BAD,
                _ => DECOMPRESS_EOF,
            };
            let _ = DECOMPRESS_NOMEM;
            error_values(lua, code, "corrupted stream")
        }
    }
}

/// Builds the `bz2` table (`LuaBz2::Open`, `LuaBz2.cpp:46-77`).
pub(crate) fn build_bz2_table(lua: &Lua) -> mlua::Result<Table> {
    let bz2 = lua.create_table()?;
    bz2.set("compress", lua.create_function(compress)?)?;
    bz2.set("decompress", lua.create_function(decompress)?)?;
    bz2.set("COMPRESS_NOMEM", COMPRESS_NOMEM)?;
    bz2.set("COMPRESS_LIMIT", COMPRESS_LIMIT)?;
    bz2.set("DECOMPRESS_NOMEM", DECOMPRESS_NOMEM)?;
    bz2.set("DECOMPRESS_LIMIT", DECOMPRESS_LIMIT)?;
    bz2.set("DECOMPRESS_TYPE", DECOMPRESS_TYPE)?;
    bz2.set("DECOMPRESS_BAD", DECOMPRESS_BAD)?;
    bz2.set("DECOMPRESS_EOF", DECOMPRESS_EOF)?;
    Ok(bz2)
}
