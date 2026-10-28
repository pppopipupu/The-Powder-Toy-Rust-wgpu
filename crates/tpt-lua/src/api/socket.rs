//! The `socket` table (`LuaSocket.cpp`): a documented local-only subset.
//!
//! TPT exposes `socket.sleep`, `socket.getTime` and a curl-backed
//! `socket.tcp()`. This port keeps the timing helpers verbatim
//! (`LuaSocketWindows.cpp:10-35` / `LuaSocketDefault.cpp:12-37`) and backs
//! `socket.tcp()` with [`std::net::UdpSocket`]: `connect`/`send`/`receive`/
//! `close`/`status`/`lasterror`/`settimeout`/`getpeername`/`getsockname` are
//! supported, TLS is not, and an unreachable host returns TPT's
//! `"Couldn't connect to server"` string.

use std::cell::RefCell;
use std::io::ErrorKind;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::rc::Rc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use mlua::{AnyUserData, Lua, MultiValue, Table, Value};

use crate::api::http::UNREACHABLE;

/// Largest datagram the port accepts in one `receive` call.
const MAX_DATAGRAM: usize = 65536;

/// `LuaSocket::Status` (`LuaSocketTCPHttp.cpp:20-26`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Ready,
    Connected,
    Dead,
}

/// State behind one `TCPSocket` handle (UDP-backed in this port).
struct SocketState {
    socket: Option<UdpSocket>,
    status: Status,
    last_error: String,
    timeout: Option<Duration>,
    peer: Option<SocketAddr>,
}

impl SocketState {
    /// Applies the stored timeout to the socket (`SetTimeout`,
    /// `LuaSocketTCPHttp.cpp:531-550`).
    fn apply_timeout(&self) {
        if let Some(socket) = self.socket.as_ref() {
            let _ = socket.set_read_timeout(self.timeout);
            let _ = socket.set_write_timeout(self.timeout);
        }
    }
}

/// `TCPSocket` userdata (`LuaSocketTCPHttp.cpp:80-113`).
struct LuaSocket {
    state: Rc<RefCell<SocketState>>,
}

/// `LuaSocket::Now` (`LuaSocketWindows.cpp:10-18` /
/// `LuaSocketDefault.cpp:12-19`).
fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

/// `LuaSocket::Timeout` (`LuaSocketDefault.cpp:21-37`): clamps the interval and
/// sleeps.
fn timeout(seconds: f64) {
    let seconds = seconds.clamp(0.0, i32::MAX as f64);
    if seconds > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(seconds));
    }
}

/// Whether a socket error means the configured timeout elapsed.
///
/// `LuaSocketTCPHttp.cpp` only ever sees `CURLE_AGAIN`, but the port's native
/// sockets surface the platform's own timeout error: `WouldBlock` on Unix-like
/// targets and `TimedOut` (`WSAETIMEDOUT`) on Windows.
fn is_timeout(kind: ErrorKind) -> bool {
    matches!(kind, ErrorKind::WouldBlock | ErrorKind::TimedOut)
}

/// Builds the `(nil, message)` failure shape used by `connect`
/// (`LuaSocketTCPHttp.cpp:479-484`).
fn connect_error(lua: &Lua, message: &str) -> mlua::Result<MultiValue> {
    Ok(MultiValue::from_vec(vec![
        Value::Nil,
        Value::String(lua.create_string(message)?),
    ]))
}

impl mlua::UserData for LuaSocket {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        // `Connect` (`LuaSocketTCPHttp.cpp:398-499`).
        methods.add_method(
            "connect",
            |lua, this, (host, port, secure): (String, u16, Option<bool>)| {
                let mut state = this.state.borrow_mut();
                if state.status == Status::Dead {
                    return Err(mlua::Error::runtime("attempt to connect dead socket"));
                }
                if state.status == Status::Connected {
                    return Err(mlua::Error::runtime("attempt to connect connected socket"));
                }
                if secure.unwrap_or(false) {
                    // No TLS in the local-only subset.
                    state.last_error = UNREACHABLE.to_owned();
                    return connect_error(lua, UNREACHABLE);
                }
                let address = format!("{host}:{port}");
                let resolved = address
                    .to_socket_addrs()
                    .ok()
                    .and_then(|mut addresses| addresses.next());
                let Some(address) = resolved else {
                    state.last_error = UNREACHABLE.to_owned();
                    return connect_error(lua, UNREACHABLE);
                };
                match UdpSocket::bind(("0.0.0.0", 0)).and_then(|socket| {
                    socket.connect(address)?;
                    Ok(socket)
                }) {
                    Ok(socket) => {
                        state.socket = Some(socket);
                        state.peer = Some(address);
                        state.status = Status::Connected;
                        state.apply_timeout();
                        Ok(MultiValue::from_vec(vec![Value::Integer(1)]))
                    }
                    Err(_) => {
                        state.last_error = UNREACHABLE.to_owned();
                        connect_error(lua, UNREACHABLE)
                    }
                }
            },
        );

        // `Send` (`LuaSocketTCPHttp.cpp:137-212`).
        methods.add_method(
            "send",
            |lua, this, (data, first, last): (mlua::LuaString, Option<i64>, Option<i64>)| {
                let mut state = this.state.borrow_mut();
                if state.status != Status::Connected {
                    return Err(mlua::Error::runtime(
                        "attempt to send on socket while not connected",
                    ));
                }
                let bytes = data.as_bytes();
                let length = bytes.len() as i64;
                let mut first = first.unwrap_or(1);
                let mut last = last.unwrap_or(-1);
                if first < 0 {
                    first += length + 1;
                }
                if last < 0 {
                    last += length + 1;
                }
                let first = first.clamp(1, length.max(1));
                let last = last.clamp(1, length.max(1));
                let begin = (first - 1) as usize;
                let end = (last as usize).max(begin);
                let slice = &bytes[begin..end];

                let sent = match state.socket.as_ref() {
                    Some(socket) => socket.send(slice).map_err(|error| error.kind()),
                    None => {
                        return Ok(MultiValue::from_vec(vec![
                            Value::Nil,
                            Value::String(lua.create_string("closed")?),
                            Value::Integer(begin as i64),
                        ]));
                    }
                };
                match sent {
                    Ok(written) => Ok(MultiValue::from_vec(vec![Value::Integer(
                        begin as i64 + written as i64,
                    )])),
                    Err(kind) => {
                        // Windows reports `SO_RCVTIMEO`/`SO_SNDTIMEO` expiry as
                        // `WSAETIMEDOUT` (mapped to `TimedOut`); Unix reports
                        // `EAGAIN`/`EWOULDBLOCK` (`WouldBlock`).
                        let reason = if is_timeout(kind) {
                            "timeout"
                        } else {
                            "closed"
                        };
                        if reason == "closed" {
                            state.status = Status::Dead;
                            state.socket = None;
                        }
                        Ok(MultiValue::from_vec(vec![
                            Value::Nil,
                            Value::String(lua.create_string(reason)?),
                            Value::Integer(begin as i64),
                        ]))
                    }
                }
            },
        );

        // `Receive` (`LuaSocketTCPHttp.cpp:213-386`), one datagram per call.
        methods.add_method("receive", |lua, this, args: MultiValue| {
            let values: Vec<Value> = args.into_vec();
            let pattern = values.first().cloned();
            let prefix = match values.get(2) {
                Some(Value::String(text)) => Some(text.to_string_lossy()),
                _ => None,
            };
            let mut state = this.state.borrow_mut();
            if state.status != Status::Connected {
                return Err(mlua::Error::runtime(
                    "attempt to receive on socket while not connected",
                ));
            }
            let mut buffer = vec![0u8; MAX_DATAGRAM];
            let received = match state.socket.as_ref() {
                Some(socket) => socket.recv(&mut buffer).map_err(|error| error.kind()),
                None => return connect_error(lua, "closed"),
            };
            let received = match received {
                Ok(received) => received,
                Err(kind) if is_timeout(kind) => {
                    return connect_error(lua, "timeout");
                }
                Err(_) => {
                    state.status = Status::Dead;
                    state.socket = None;
                    return connect_error(lua, "closed");
                }
            };

            let mut data = &buffer[..received];
            if let Some(pattern) = pattern {
                match pattern {
                    Value::Integer(count) if count >= 0 => {
                        data = &data[..(count as usize).min(data.len())];
                    }
                    Value::Number(count) if count >= 0.0 => {
                        data = &data[..(count as usize).min(data.len())];
                    }
                    Value::String(pattern) => match pattern.to_string_lossy().as_str() {
                        "*a" => {}
                        "*l" | "*L" => {
                            if let Some(index) = data.iter().position(|byte| *byte == b'\n') {
                                let mut line = data[..=index].to_vec();
                                if pattern.to_string_lossy() == "*L" {
                                    line.retain(|byte| *byte != b'\r' && *byte != b'\n');
                                }
                                return match prefix {
                                    Some(prefix) => {
                                        let mut joined = prefix.into_bytes();
                                        joined.extend_from_slice(&line);
                                        Ok(MultiValue::from_vec(vec![Value::String(
                                            lua.create_string(&joined)?,
                                        )]))
                                    }
                                    None => Ok(MultiValue::from_vec(vec![Value::String(
                                        lua.create_string(&line)?,
                                    )])),
                                };
                            }
                        }
                        _ => return Err(mlua::Error::runtime("unsupported receive pattern")),
                    },
                    _ => {}
                }
            }
            match prefix {
                Some(prefix) => {
                    let mut joined = prefix.into_bytes();
                    joined.extend_from_slice(data);
                    Ok(MultiValue::from_vec(vec![Value::String(
                        lua.create_string(&joined)?,
                    )]))
                }
                None => Ok(MultiValue::from_vec(vec![Value::String(
                    lua.create_string(data)?,
                )])),
            }
        });

        // `Close` (`LuaSocketTCPHttp.cpp:129-135`).
        methods.add_method("close", |_, this, ()| {
            let mut state = this.state.borrow_mut();
            state.socket = None;
            state.status = Status::Dead;
            Ok(())
        });

        // `GetStatus` (`LuaSocketTCPHttp.cpp:501-511`).
        methods.add_method("status", |_, this, ()| {
            let state = this.state.borrow();
            Ok(match state.status {
                Status::Ready => "ready",
                Status::Connected => "connected",
                Status::Dead => "dead",
            })
        });

        // `LastError` (`LuaSocketTCPHttp.cpp:487-493`).
        methods.add_method("lasterror", |_, this, ()| {
            Ok(this.state.borrow().last_error.clone())
        });

        // `SetTimeout` (`LuaSocketTCPHttp.cpp:531-550`).
        methods.add_method("settimeout", |_, this, seconds: Option<f64>| {
            let mut state = this.state.borrow_mut();
            state.timeout = seconds.map(|seconds| {
                Duration::from_secs_f64(seconds.clamp(0.0, i32::MAX as f64))
            });
            state.apply_timeout();
            Ok(())
        });

        // `GetPeerName` (`LuaSocketTCPHttp.cpp:513-529`).
        methods.add_method("getpeername", |lua, this, ()| {
            let state = this.state.borrow();
            if state.status != Status::Connected {
                return Err(mlua::Error::runtime(
                    "attempt to get remote socket info while not connected",
                ));
            }
            let Some(peer) = state.peer else {
                return Err(mlua::Error::runtime(
                    "attempt to get remote socket info while not connected",
                ));
            };
            Ok(MultiValue::from_vec(vec![
                Value::String(lua.create_string(peer.ip().to_string())?),
                Value::Integer(peer.port() as i64),
            ]))
        });

        // `GetSockName` (`LuaSocketTCPHttp.cpp:531-570`).
        methods.add_method("getsockname", |lua, this, ()| {
            let state = this.state.borrow();
            if state.status != Status::Connected {
                return Err(mlua::Error::runtime(
                    "attempt to get local socket info while not connected",
                ));
            }
            let Some(local) = state
                .socket
                .as_ref()
                .and_then(|socket| socket.local_addr().ok())
            else {
                return Err(mlua::Error::runtime(
                    "attempt to get local socket info while not connected",
                ));
            };
            Ok(MultiValue::from_vec(vec![
                Value::String(lua.create_string(local.ip().to_string())?),
                Value::Integer(local.port() as i64),
            ]))
        });
    }
}

/// `socket.tcp()` (`LuaSocketTCPHttp.cpp:80-113`).
fn new_socket(lua: &Lua) -> mlua::Result<AnyUserData> {
    lua.create_userdata(LuaSocket {
        state: Rc::new(RefCell::new(SocketState {
            socket: None,
            status: Status::Ready,
            last_error: String::new(),
            timeout: None,
            peer: None,
        })),
    })
}

/// Builds the `socket` table (`LuaSocket::Open`, `LuaSocket.cpp:20-30`).
pub(crate) fn build_socket_table(lua: &Lua) -> mlua::Result<Table> {
    let socket = lua.create_table()?;
    socket.set(
        "sleep",
        lua.create_function(|_, seconds: f64| -> mlua::Result<()> {
            timeout(seconds);
            Ok(())
        })?,
    )?;
    socket.set(
        "getTime",
        lua.create_function(|_, ()| -> mlua::Result<f64> { Ok(now()) })?,
    )?;
    socket.set(
        "tcp",
        lua.create_function(|lua, ()| -> mlua::Result<AnyUserData> { new_socket(lua) })?,
    )?;
    Ok(socket)
}
