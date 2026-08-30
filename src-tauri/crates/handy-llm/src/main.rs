//! Inference sidecar for Handy's transcript enhancement layer.
//!
//! Reads line-delimited JSON requests on stdin and writes one JSON response per
//! line on stdout. Runs as a child process of the app rather than as a library
//! inside it: `transcribe-cpp` and `llama.cpp` each vendor a full ggml and
//! cannot share a binary, and keeping inference in its own address space means
//! an out-of-memory kill costs the user a fallback to raw dictation instead of
//! taking down the app mid-sentence.
//!
//! Diagnostics go to stderr so they can never corrupt the response stream.

mod engine;
mod protocol;

use engine::Engine;
use protocol::{Request, Response};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Cap on generated tokens when a request does not specify one.
const DEFAULT_MAX_TOKENS: usize = 512;

fn main() {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    let mut engine = match Engine::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("handy-llm: fatal: {e:#}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "handy-llm: ready (gpu backend: {})",
        if engine::HAS_GPU_BACKEND { "yes" } else { "no" }
    );

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("handy-llm: stdin closed: {e}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        let request: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                // No id is recoverable from an unparseable line, so answer on
                // id 0 rather than staying silent and hanging the caller.
                respond(
                    &mut stdout,
                    &Response::error(0, format!("bad request: {e}")),
                );
                continue;
            }
        };

        if let Request::Shutdown { id } = request {
            respond(&mut stdout, &Response::ok(id));
            break;
        }

        let response = handle(&mut engine, request);
        respond(&mut stdout, &response);
    }
}

/// Execute one request against the engine.
fn handle(engine: &mut Engine, request: Request) -> Response {
    match request {
        Request::Ping { id } => Response::ok(id),

        Request::Load {
            id,
            path,
            threads,
            ctx,
            gpu_layers,
        } => {
            let started = Instant::now();
            match engine.load(&PathBuf::from(path), threads, ctx, gpu_layers) {
                Ok(()) => Response {
                    elapsed_ms: Some(elapsed_ms(started)),
                    ..Response::ok(id)
                },
                Err(e) => Response::error(id, format!("{e:#}")),
            }
        }

        Request::Generate {
            id,
            system,
            user,
            max_tokens,
            no_think,
        } => {
            let started = Instant::now();
            let budget = max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
            match engine.generate(&system, &user, budget, no_think.unwrap_or(false)) {
                Ok(text) => Response::generated(id, text, elapsed_ms(started)),
                Err(e) => Response::error(id, format!("{e:#}")),
            }
        }

        Request::Unload { id } => {
            engine.unload();
            Response::ok(id)
        }

        // Handled by the caller so the loop can terminate.
        Request::Shutdown { id } => Response::ok(id),
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Write one response line, flushing so the host is not left waiting.
fn respond(out: &mut std::io::Stdout, response: &Response) {
    match serde_json::to_string(response) {
        Ok(json) => {
            if writeln!(out, "{json}").is_err() {
                return;
            }
        }
        Err(e) => {
            eprintln!("handy-llm: failed to serialise response: {e}");
            return;
        }
    }
    let _ = out.flush();
}
