//! The `http` table (`LuaHttp.cpp`): a documented local-only subset.
//!
//! TPT runs libcurl requests on a background thread and returns a handle with
//! `status`/`progress`/`cancel`/`finish`. This port performs a plain HTTP/1.x
//! request synchronously over [`std::net::TcpStream`] when the handle is
//! created; TLS is not supported, so `https://` URLs report the same error the
//! reference reports when the host is unreachable
//! (`CURLE_COULDNT_CONNECT` → `"Couldn't connect to server"`,
//! `client/http/Request.cpp:129-133`). The handle shape is unchanged, so a
//! script that polls `status()` sees `"done"` immediately and `finish()`
//! returns the response or raises the request error.

use std::cell::RefCell;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::rc::Rc;
use std::time::Duration;

use mlua::{AnyUserData, Lua, MultiValue, Table, Value};

/// `curl_easy_strerror(CURLE_COULDNT_CONNECT)`: the string TPT surfaces when
/// the host cannot be reached.
pub(crate) const UNREACHABLE: &str = "Couldn't connect to server";

/// Connect/read timeout for one request.
const TIMEOUT: Duration = Duration::from_secs(10);

/// Preference file the desktop front-end writes, in the working directory.
/// Mirrors `tpt_app::prefs::PREFS_FILE` (the two crates cannot share the
/// constant because `tpt-app` depends on `tpt-lua`).
const PREFS_FILE: &str = "prefs.json";

/// Preference key of the signed-in session (`Client::SaveAuthUser`,
/// `client/AuthUserCommon.cpp:18-34`). The reference sends this value as
/// `X-Auth-Session-Key` (`client/http/Request.cpp:60-75`).
const AUTH_TOKEN_KEY: &str = "User.SessionID";

/// The session token the platform has stored, when a user is signed in.
///
/// This front-end has no login dialog, but it shares TPT's preference file
/// layout, so a `User.SessionID` written by any instance is the stored token.
fn stored_auth_token() -> Option<String> {
    let text = std::fs::read_to_string(PREFS_FILE).ok()?;
    let prefs: serde_json::Value = serde_json::from_str(&text).ok()?;
    prefs
        .get(AUTH_TOKEN_KEY)?
        .as_str()
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
}

/// One HTTP response.
pub(crate) struct Response {
    /// Status code (`200`, ...).
    pub status: i32,
    /// Response headers in wire order.
    pub headers: Vec<(String, String)>,
    /// Response body.
    pub body: Vec<u8>,
}

/// Splits `http://host[:port][/path][?query]` into its parts.
fn parse_url(url: &str) -> Option<(String, u16, String)> {
    let rest = url.strip_prefix("http://")?;
    let (authority, path) = match rest.find('/') {
        Some(index) => (&rest[..index], &rest[index..]),
        None => (rest, "/"),
    };
    if authority.is_empty() {
        return None;
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) if port.chars().all(|c| c.is_ascii_digit()) => {
            (host, port.parse().ok()?)
        }
        _ => (authority, 80),
    };
    if host.is_empty() {
        return None;
    }
    Some((host.to_owned(), port, path.to_owned()))
}

/// Performs one plain HTTP request. `Err` carries the TPT error string.
pub(crate) fn request(
    method: &str,
    url: &str,
    body: Option<&[u8]>,
    headers: &[(String, String)],
) -> Result<Response, String> {
    let Some((host, port, path)) = parse_url(url) else {
        // `https://` and non-HTTP schemes are outside the local-only subset.
        return Err(UNREACHABLE.to_owned());
    };
    let address = format!("{host}:{port}");
    let Ok(mut addresses) = address.to_socket_addrs() else {
        return Err(UNREACHABLE.to_owned());
    };
    let Some(address) = addresses.next() else {
        return Err(UNREACHABLE.to_owned());
    };
    let mut stream =
        TcpStream::connect_timeout(&address, TIMEOUT).map_err(|_| UNREACHABLE.to_owned())?;
    let _ = stream.set_read_timeout(Some(TIMEOUT));
    let _ = stream.set_write_timeout(Some(TIMEOUT));

    let mut request = format!("{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n");
    for (name, value) in headers {
        request.push_str(name);
        request.push_str(": ");
        request.push_str(value);
        request.push_str("\r\n");
    }
    if let Some(body) = body {
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
        if !headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        {
            request.push_str("Content-Type: application/octet-stream\r\n");
        }
    }
    request.push_str("\r\n");
    stream
        .write_all(request.as_bytes())
        .map_err(|_| UNREACHABLE.to_owned())?;
    if let Some(body) = body {
        stream
            .write_all(body)
            .map_err(|_| UNREACHABLE.to_owned())?;
    }

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|_| UNREACHABLE.to_owned())?;
    parse_response(&raw)
}

/// Parses an HTTP/1.x response (`Request::ParseResponse`'s wire side).
fn parse_response(raw: &[u8]) -> Result<Response, String> {
    let split = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| "Could not read response: malformed HTTP response".to_owned())?;
    let head = std::str::from_utf8(&raw[..split])
        .map_err(|_| "Could not read response: malformed HTTP response".to_owned())?;
    let mut body = raw[split + 4..].to_vec();

    let mut lines = head.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| "Could not read response: malformed HTTP response".to_owned())?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<i32>().ok())
        .ok_or_else(|| "Could not read response: malformed HTTP response".to_owned())?;

    let mut headers = Vec::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_owned(), value.trim().to_owned()));
        }
    }

    let header = |name: &str| {
        headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    };
    if header("transfer-encoding").is_some_and(|value| value.to_ascii_lowercase().contains("chunked"))
    {
        body = decode_chunked(&body)?;
    } else if let Some(length) = header("content-length").and_then(|value| value.parse().ok()) {
        body.truncate(length);
    }
    Ok(Response {
        status,
        headers,
        body,
    })
}

/// Decodes a `Transfer-Encoding: chunked` body.
fn decode_chunked(raw: &[u8]) -> Result<Vec<u8>, String> {
    let mut decoded = Vec::new();
    let mut cursor = 0;
    loop {
        let line_end = raw[cursor..]
            .windows(2)
            .position(|window| window == b"\r\n")
            .ok_or_else(|| "Could not read response: truncated chunked body".to_owned())?;
        let line = std::str::from_utf8(&raw[cursor..cursor + line_end])
            .map_err(|_| "Could not read response: malformed chunked body".to_owned())?;
        let size = usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16)
            .map_err(|_| "Could not read response: malformed chunked body".to_owned())?;
        cursor += line_end + 2;
        if size == 0 {
            return Ok(decoded);
        }
        if cursor + size + 2 > raw.len() {
            return Err("Could not read response: truncated chunked body".to_owned());
        }
        decoded.extend_from_slice(&raw[cursor..cursor + size]);
        cursor += size + 2;
    }
}

/// State behind one `HTTPRequest` handle.
struct HttpState {
    /// `running`, `done` or `dead`.
    status: Status,
    /// Response status code.
    status_code: i32,
    /// Response headers.
    headers: Vec<(String, String)>,
    /// Response body.
    body: Vec<u8>,
    /// Request error (`Request::Finish` raises it).
    error: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Done,
    Dead,
}

/// `HTTPRequest` userdata (`LuaHttp.cpp:188-260`).
struct HttpRequest {
    state: Rc<RefCell<HttpState>>,
}

impl mlua::UserData for HttpRequest {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        // `HTTPRequest_status` (`LuaHttp.cpp:200-213`).
        methods.add_method("status", |_, this, ()| {
            let state = this.state.borrow();
            Ok(match state.status {
                Status::Done => "done",
                Status::Dead => "dead",
            })
        });

        // `HTTPRequest_progress` (`LuaHttp.cpp:223-234`).
        methods.add_method("progress", |_, this, ()| {
            let state = this.state.borrow();
            if state.status == Status::Dead {
                return Ok(MultiValue::new());
            }
            let total = state.body.len() as i64;
            Ok(MultiValue::from_vec(vec![
                Value::Integer(total),
                Value::Integer(total),
            ]))
        });

        // `HTTPRequest_cancel` (`LuaHttp.cpp:236-247`).
        methods.add_method("cancel", |_, this, ()| {
            let mut state = this.state.borrow_mut();
            if state.status != Status::Dead {
                state.status = Status::Dead;
                state.body.clear();
                state.headers.clear();
            }
            Ok(())
        });

        // `HTTPRequest_finish` (`LuaHttp.cpp:249-272`).
        methods.add_method("finish", |lua, this, ()| {
            let mut state = this.state.borrow_mut();
            if state.status != Status::Done {
                return Ok(MultiValue::new());
            }
            if let Some(error) = state.error.clone() {
                state.status = Status::Dead;
                return Err(mlua::Error::runtime(error));
            }
            let body = lua.create_string(&state.body)?;
            let status = state.status_code;
            let headers = lua.create_table()?;
            for (index, (name, value)) in state.headers.iter().enumerate() {
                let entry = lua.create_table()?;
                entry.raw_set(1, lua.create_string(name)?)?;
                entry.raw_set(2, lua.create_string(value)?)?;
                headers.raw_set(index + 1, entry)?;
            }
            state.status = Status::Dead;
            Ok(MultiValue::from_vec(vec![
                Value::String(body),
                Value::Integer(status as i64),
                Value::Table(headers),
            ]))
        });
    }
}

/// Builds a handle from a finished request.
fn make_handle(lua: &Lua, result: Result<Response, String>) -> mlua::Result<AnyUserData> {
    let state = match result {
        Ok(response) => HttpState {
            status: Status::Done,
            status_code: response.status,
            headers: response.headers,
            body: response.body,
            error: None,
        },
        Err(error) => HttpState {
            status: Status::Done,
            status_code: 0,
            headers: Vec::new(),
            body: Vec::new(),
            error: Some(error),
        },
    };
    lua.create_userdata(HttpRequest {
        state: Rc::new(RefCell::new(state)),
    })
}

/// Reads a `{ {name, value}, ... }` header table, also accepting the old
/// dictionary form (`LuaHttp.cpp:352-401`).
fn read_headers(value: Option<&Value>) -> mlua::Result<Vec<(String, String)>> {
    let Some(Value::Table(table)) = value else {
        return Ok(Vec::new());
    };
    let mut headers = Vec::new();
    let length = table.raw_len();
    if length > 0 {
        for index in 1..=length {
            let entry: Table = table.raw_get(index)?;
            let name: String = entry.raw_get(1)?;
            let value: String = entry.raw_get(2)?;
            headers.push((name, value));
        }
    } else {
        for pair in table.pairs::<String, String>() {
            let (name, value) = pair?;
            headers.push((name, value));
        }
    }
    Ok(headers)
}

/// Reads the optional POST body: a string, or a table encoded as
/// `application/x-www-form-urlencoded` (the reference sends multipart form
/// data; this port documents the urlencoded subset).
fn read_post_body(value: Option<&Value>) -> mlua::Result<(Option<Vec<u8>>, Option<String>)> {
    match value {
        Some(Value::String(text)) => Ok((Some(text.as_bytes().to_vec()), None)),
        Some(Value::Table(table)) => {
            let mut body = String::new();
            for pair in table.pairs::<String, String>() {
                let (name, value) = pair?;
                if !body.is_empty() {
                    body.push('&');
                }
                body.push_str(&percent_encode(&name));
                body.push('=');
                body.push_str(&percent_encode(&value));
            }
            Ok((
                Some(body.into_bytes()),
                Some("application/x-www-form-urlencoded".to_owned()),
            ))
        }
        None => Ok((None, None)),
        Some(other) => Err(mlua::Error::runtime(format!(
            "expected a string or table, got {}",
            other.type_name()
        ))),
    }
}

/// Minimal form-urlencoded escaping.
fn percent_encode(text: &str) -> String {
    let mut encoded = String::new();
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            b' ' => encoded.push('+'),
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

/// `http.get`/`http.post` (`LuaHttp.cpp:274-415`).
fn request_lua(lua: &Lua, args: MultiValue, is_post: bool) -> mlua::Result<AnyUserData> {
    let values: Vec<Value> = args.into_vec();
    let Some(url) = values.first() else {
        return Err(mlua::Error::runtime("string expected, got no value"));
    };
    let url = super::value_to_string(url)?;

    let mut index = 1;
    let mut body = None;
    let mut content_type = None;
    if is_post {
        if let Some(value) = values.get(index) {
            let (read_body, read_type) = read_post_body(Some(value))?;
            body = read_body;
            content_type = read_type;
        }
        index += 1;
    }
    let mut headers = read_headers(values.get(index))?;
    if let Some(content_type) = content_type {
        headers.push(("Content-Type".to_owned(), content_type));
    }
    index += 1;
    let method = match values.get(index) {
        Some(Value::String(verb)) => verb.to_string_lossy(),
        _ => {
            if is_post {
                "POST".to_owned()
            } else {
                "GET".to_owned()
            }
        }
    };
    let result = request(&method, &url, body.as_deref(), &headers);
    make_handle(lua, result)
}

/// Builds the `http` table (`LuaHttp::Open`, `LuaHttp.cpp:420-459`).
pub(crate) fn build_http_table(lua: &Lua) -> mlua::Result<Table> {
    let http = lua.create_table()?;
    http.set(
        "get",
        lua.create_function(|lua, args: MultiValue| request_lua(lua, args, false))?,
    )?;
    http.set(
        "post",
        lua.create_function(|lua, args: MultiValue| request_lua(lua, args, true))?,
    )?;
    // `getAuthToken` (`LuaHttp.cpp:408-414`). The reference builds a handle
    // for `<SERVER>/ExternalAuth.api?Action=Get&Audience=<audience>` with the
    // signed-in session's auth headers (`LuaHttp.cpp:55-92`) and returns
    // `nil, "not authenticated"` when no user is signed in (`:60-66`). This
    // front-end has no login, so a stored `User.SessionID` is returned
    // directly through the same handle shape; without one the reference's
    // unauthenticated result is unchanged.
    http.set(
        "getAuthToken",
        lua.create_function(|lua, _audience: String| -> mlua::Result<MultiValue> {
            if let Some(token) = stored_auth_token() {
                let handle = make_handle(
                    lua,
                    Ok(Response {
                        status: 200,
                        headers: Vec::new(),
                        body: token.into_bytes(),
                    }),
                )?;
                return Ok(MultiValue::from_vec(vec![Value::UserData(handle)]));
            }
            Ok(MultiValue::from_vec(vec![
                Value::Nil,
                Value::String(lua.create_string("not authenticated")?),
            ]))
        })?,
    )?;
    Ok(http)
}

/// Fetches a URL and returns its body when the response is `200`
/// (`LuaMisc.cpp:104-113`'s `installScriptManager` path).
pub(crate) fn fetch_text(url: &str) -> Result<String, String> {
    let response = request("GET", url, None, &[])?;
    if response.status != 200 {
        return Err(format!("HTTP Error {}: {}", response.status, status_text(response.status)));
    }
    String::from_utf8(response.body).map_err(|_| "Could not read response: invalid UTF-8".to_owned())
}

/// `http::StatusText` (`client/http/Request.cpp:156-237`), limited to the
/// codes the local-only subset can produce.
fn status_text(code: i32) -> &'static str {
    match code {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown Status Code",
    }
}
