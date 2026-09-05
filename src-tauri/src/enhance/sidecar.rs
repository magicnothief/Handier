//! Client for the `handy-llm` inference sidecar.
//!
//! Owns the child process and speaks the line-delimited JSON protocol described
//! in `crates/handy-llm/README.md`. One request is in flight at a time — a
//! dictation cannot start until the previous one has pasted — so the transport
//! is a plain mutex-guarded write-then-read rather than a correlation map.
//!
//! The process is spawned lazily and can be dropped to reclaim its memory, which
//! matters on the low-end machines this targets: an idle enhancement layer
//! should not hold a model resident.

use anyhow::{anyhow, bail, Context, Result};
use log::{debug, info, warn};
use serde::Serialize;
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Mutex;
use std::time::Duration;

use super::{Backend, GenParams, PromptStyle};

/// How long to wait for a reply before declaring the sidecar wedged.
///
/// Generous because a first load on a slow disk is slow, and the cost of being
/// wrong is killing a healthy process mid-dictation.
const REPLY_TIMEOUT: Duration = Duration::from_secs(120);

/// A running `handy-llm` process.
///
/// Replies arrive over a channel rather than being read inline, so a wedged
/// child can be given up on. A blocking `read_line` on the pipe would hang the
/// dictation thread with no way out.
struct Process {
    child: Child,
    stdin: ChildStdin,
    replies: Receiver<String>,
}

/// Handle to the inference sidecar.
pub struct SidecarClient {
    exe: PathBuf,
    process: Mutex<Option<Process>>,
    /// Path of the model currently loaded, if any.
    loaded: Mutex<Option<PathBuf>>,
    next_id: AtomicU64,
    /// Whether the loaded model deliberates and needs the soft switch.
    reasoning: Mutex<bool>,
    /// How the loaded model expects to be prompted.
    prompt_style: Mutex<PromptStyle>,
}

impl SidecarClient {
    /// Create a client for the binary at `exe`, without starting it.
    pub fn new(exe: PathBuf) -> Self {
        Self {
            exe,
            process: Mutex::new(None),
            loaded: Mutex::new(None),
            next_id: AtomicU64::new(1),
            reasoning: Mutex::new(false),
            prompt_style: Mutex::new(PromptStyle::Instructed),
        }
    }

    /// Locate the sidecar binary.
    ///
    /// Checks the bundled resource location first, then the places a developer
    /// build leaves it, so `bun run tauri dev` works without a packaging step.
    pub fn find_binary(resource_dir: Option<&Path>) -> Option<PathBuf> {
        let exe_name = if cfg!(windows) {
            "handy-llm.exe"
        } else {
            "handy-llm"
        };

        // Tauri strips the target triple when it bundles an `externalBin`, but
        // the staging directory keeps it, so both spellings are searched.
        let staged_name = format!(
            "handy-llm-{}{}",
            env!("HANDY_TARGET_TRIPLE"),
            Self::exe_suffix()
        );

        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(dir) = resource_dir {
            candidates.push(dir.join(exe_name));
            candidates.push(dir.join(&staged_name));
        }
        // Alongside the app binary, which is where `externalBin` lands it.
        if let Ok(current) = std::env::current_exe() {
            if let Some(dir) = current.parent() {
                candidates.push(dir.join(exe_name));
                candidates.push(dir.join(&staged_name));
            }
        }
        // An explicit override, mostly for tests and unusual layouts.
        if let Ok(path) = std::env::var("HANDY_LLM_BIN") {
            candidates.insert(0, PathBuf::from(path));
        }
        // Developer runs: `bun run build:sidecar` stages here, and the crate's
        // own target directory may be redirected by `CARGO_TARGET_DIR` (the
        // Vulkan build needs a short path on Windows), so the staged copy is
        // the reliable one to look for.
        for base in ["binaries", "src-tauri/binaries", "../src-tauri/binaries"] {
            candidates.push(PathBuf::from(base).join(&staged_name));
            candidates.push(PathBuf::from(base).join(exe_name));
        }
        for profile in ["release", "debug"] {
            candidates.push(
                PathBuf::from("crates/handy-llm/target")
                    .join(profile)
                    .join(exe_name),
            );
        }

        candidates.into_iter().find(|p| p.is_file())
    }

    /// Executable suffix for the host platform.
    fn exe_suffix() -> &'static str {
        if cfg!(windows) {
            ".exe"
        } else {
            ""
        }
    }

    /// Whether the process is running.
    pub fn is_running(&self) -> bool {
        self.process.lock().map(|p| p.is_some()).unwrap_or(false)
    }

    /// Path of the model currently loaded, if any.
    pub fn loaded_model(&self) -> Option<PathBuf> {
        self.loaded.lock().ok().and_then(|m| m.clone())
    }

    /// Start the process if it is not already running.
    fn ensure_started(&self) -> Result<()> {
        let mut guard = self
            .process
            .lock()
            .map_err(|_| anyhow!("sidecar lock poisoned"))?;
        if guard.is_some() {
            return Ok(());
        }

        if !self.exe.is_file() {
            bail!("sidecar binary not found at {}", self.exe.display());
        }

        let mut cmd = Command::new(&self.exe);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            // Without this the child briefly flashes a console window.
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to start sidecar at {}", self.exe.display()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("sidecar stdin unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("sidecar stdout unavailable"))?;

        // Read replies on a thread and hand them over a channel, so `request`
        // can wait with a deadline instead of blocking forever on the pipe.
        let (tx, replies) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break; // client went away
                }
            }
        });

        // llama.cpp is verbose on stderr. Left unread its pipe fills and the
        // child blocks forever, so drain it on a thread and forward to the log.
        if let Some(stderr) = child.stderr.take() {
            std::thread::spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    debug!("handy-llm: {line}");
                }
            });
        }

        info!("started enhancement sidecar: {}", self.exe.display());
        *guard = Some(Process {
            child,
            stdin,
            replies,
        });
        Ok(())
    }

    /// Stop the process and release the model's memory.
    pub fn shutdown(&self) {
        let mut guard = match self.process.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if let Some(mut proc) = guard.take() {
            // Ask nicely, then make sure. A sidecar that ignores shutdown must
            // not outlive the app and keep a model resident.
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let _ = writeln!(proc.stdin, r#"{{"cmd":"shutdown","id":{id}}}"#);
            let _ = proc.stdin.flush();
            drop(proc.stdin);
            let _ = proc.child.kill();
            let _ = proc.child.wait();
        }
        if let Ok(mut loaded) = self.loaded.lock() {
            *loaded = None;
        }
    }

    /// Send one request and read its reply.
    fn request(&self, body: &impl Serialize) -> Result<Value> {
        self.ensure_started()?;
        let mut guard = self
            .process
            .lock()
            .map_err(|_| anyhow!("sidecar lock poisoned"))?;
        let proc = guard
            .as_mut()
            .ok_or_else(|| anyhow!("sidecar is not running"))?;

        let line = serde_json::to_string(body).context("failed to encode request")?;
        writeln!(proc.stdin, "{line}").context("failed to write to sidecar")?;
        proc.stdin.flush().context("failed to flush to sidecar")?;

        let reply = match proc.replies.recv_timeout(REPLY_TIMEOUT) {
            Ok(line) => line,
            Err(RecvTimeoutError::Timeout) => {
                // Give up on the process rather than leaving the user's
                // dictation blocked. The next request starts a fresh one.
                warn!("sidecar did not reply within {REPLY_TIMEOUT:?}; restarting it");
                if let Some(mut dead) = guard.take() {
                    let _ = dead.child.kill();
                    let _ = dead.child.wait();
                }
                bail!("sidecar timed out after {REPLY_TIMEOUT:?}");
            }
            Err(RecvTimeoutError::Disconnected) => {
                // The process died. Drop it so the next call restarts it rather
                // than talking to a corpse forever.
                *guard = None;
                bail!("sidecar exited unexpectedly");
            }
        };

        let value: Value =
            serde_json::from_str(reply.trim()).context("sidecar sent malformed JSON")?;
        if value.get("ok").and_then(Value::as_bool) != Some(true) {
            let err = value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown sidecar error");
            bail!("{err}");
        }
        Ok(value)
    }

    /// Load `model` into the sidecar, replacing any model already resident.
    ///
    /// `reasoning` records whether this model deliberates, so generation can
    /// suppress it. Loading the model that is already loaded is a no-op.
    pub fn load_model(
        &self,
        model: &Path,
        reasoning: bool,
        prompt_style: PromptStyle,
        gpu_layers: Option<u32>,
    ) -> Result<()> {
        if self.loaded_model().as_deref() == Some(model) {
            return Ok(());
        }

        #[derive(Serialize)]
        struct Load<'a> {
            cmd: &'a str,
            id: u64,
            path: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            gpu_layers: Option<u32>,
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.request(&Load {
            cmd: "load",
            id,
            path: model.to_string_lossy().into_owned(),
            gpu_layers,
        })
        .with_context(|| format!("failed to load {}", model.display()))?;

        if let Ok(mut loaded) = self.loaded.lock() {
            *loaded = Some(model.to_path_buf());
        }
        if let Ok(mut p) = self.prompt_style.lock() {
            *p = prompt_style;
        }
        if let Ok(mut r) = self.reasoning.lock() {
            *r = reasoning;
        }
        info!("enhancement model loaded: {}", model.display());
        Ok(())
    }

    /// Drop the model but leave the process running.
    pub fn unload_model(&self) {
        if !self.is_running() {
            return;
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        #[derive(Serialize)]
        struct Unload<'a> {
            cmd: &'a str,
            id: u64,
        }
        if let Err(e) = self.request(&Unload { cmd: "unload", id }) {
            warn!("failed to unload enhancement model: {e}");
        }
        if let Ok(mut loaded) = self.loaded.lock() {
            *loaded = None;
        }
    }
}

impl Drop for SidecarClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl Backend for SidecarClient {
    fn generate(&self, system: &str, user: &str, params: &GenParams) -> Result<String> {
        #[derive(Serialize)]
        struct Generate<'a> {
            cmd: &'a str,
            id: u64,
            system: &'a str,
            user: &'a str,
            max_tokens: usize,
            no_think: bool,
        }

        let no_think = self.reasoning.lock().map(|r| *r).unwrap_or(false);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let reply = self.request(&Generate {
            cmd: "generate",
            id,
            system,
            user,
            max_tokens: params.max_tokens,
            no_think,
        })?;

        reply
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| anyhow!("sidecar reply had no text"))
    }

    fn is_ready(&self) -> bool {
        self.loaded_model().is_some()
    }

    fn prompt_style(&self) -> PromptStyle {
        self.prompt_style.lock().map(|p| *p).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_binary_is_reported_not_panicked() {
        let client = SidecarClient::new(PathBuf::from("no-such-sidecar-binary"));
        let err = client.ensure_started().expect_err("must fail");
        assert!(err.to_string().contains("not found"));
        assert!(!client.is_running());
    }

    #[test]
    fn an_unstarted_client_is_not_ready() {
        let client = SidecarClient::new(PathBuf::from("no-such-sidecar-binary"));
        assert!(!client.is_ready());
        assert!(client.loaded_model().is_none());
    }

    #[test]
    fn generate_fails_cleanly_without_a_process() {
        let client = SidecarClient::new(PathBuf::from("no-such-sidecar-binary"));
        let err = client
            .generate("s", "u", &GenParams::for_input(10))
            .expect_err("must fail without a sidecar");
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn request_ids_are_unique() {
        let client = SidecarClient::new(PathBuf::from("x"));
        let a = client.next_id.fetch_add(1, Ordering::Relaxed);
        let b = client.next_id.fetch_add(1, Ordering::Relaxed);
        assert_ne!(a, b);
    }

    #[test]
    fn shutdown_on_an_unstarted_client_is_harmless() {
        let client = SidecarClient::new(PathBuf::from("no-such-sidecar-binary"));
        client.shutdown();
        client.unload_model();
        assert!(!client.is_running());
    }

    #[test]
    fn find_binary_prefers_the_resource_directory() {
        // The bundled copy must win over any developer build lying around,
        // otherwise a packaged app could run a stale binary from a target dir.
        let dir = std::env::temp_dir().join("handy-flow-sidecar-probe");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let name = if cfg!(windows) {
            "handy-llm.exe"
        } else {
            "handy-llm"
        };
        let planted = dir.join(name);
        std::fs::write(&planted, b"not a real binary").expect("plant a file");

        let found = SidecarClient::find_binary(Some(&dir));
        assert_eq!(found.as_deref(), Some(planted.as_path()));

        std::fs::remove_file(&planted).ok();
    }
}

/// Tests that drive the real sidecar binary.
///
/// Ignored by default because they need `handy-llm` built and a GGUF on disk.
/// Run with the paths supplied:
///
/// ```text
/// HANDY_LLM_BIN=D:/hl/release/handy-llm.exe \
/// HANDY_LLM_MODEL=D:/dev/handy-models/Qwen3-0.6B-Q4_K_M.gguf \
/// cargo test --lib sidecar::live -- --ignored --nocapture
/// ```
#[cfg(test)]
mod live {
    use super::*;
    use crate::enhance::{AppContext, EnhanceOptions, VerifyMode};

    fn client_and_model() -> Option<(SidecarClient, PathBuf)> {
        let exe = std::env::var("HANDY_LLM_BIN").ok()?;
        let model = std::env::var("HANDY_LLM_MODEL").ok()?;
        let exe = PathBuf::from(exe);
        let model = PathBuf::from(model);
        if !exe.is_file() || !model.is_file() {
            return None;
        }
        Some((SidecarClient::new(exe), model))
    }

    #[test]
    #[ignore = "needs a built sidecar and a downloaded model"]
    fn drives_the_real_sidecar_end_to_end() {
        let (client, model) = match client_and_model() {
            Some(v) => v,
            None => {
                eprintln!("skipping: set HANDY_LLM_BIN and HANDY_LLM_MODEL");
                return;
            }
        };

        assert!(!client.is_ready(), "nothing loaded yet");
        client
            .load_model(&model, true, PromptStyle::Instructed, None)
            .expect("model should load");
        assert!(client.is_ready(), "should be ready after load");
        assert!(client.is_running(), "process should be up");

        // Loading the same model again must be a no-op, not a reload.
        client
            .load_model(&model, true, PromptStyle::Instructed, None)
            .expect("idempotent load");

        let opts = EnhanceOptions {
            // Keep the pass to a single inference so the test stays quick; the
            // verifier is exercised by its own eval.
            verify: VerifyMode::Off,
            ..Default::default()
        };
        let out = crate::enhance::enhance(
            &client,
            "um so the meeting is uh moved to friday at three",
            &opts,
            &AppContext::default(),
        );

        eprintln!("enhanced in {:?}: {}", out.elapsed, out.text);
        assert!(out.error.is_none(), "unexpected error: {:?}", out.error);
        assert!(!out.text.trim().is_empty(), "must produce text");
        // Whatever the model does, the raw transcript is always preserved.
        assert_eq!(
            out.original,
            "um so the meeting is uh moved to friday at three"
        );

        client.unload_model();
        assert!(!client.is_ready(), "unload should clear readiness");
        client.shutdown();
        assert!(!client.is_running(), "shutdown should stop the process");
    }

    #[test]
    #[ignore = "needs a built sidecar and a downloaded model"]
    fn a_killed_sidecar_is_reported_and_restartable() {
        let (client, model) = match client_and_model() {
            Some(v) => v,
            None => return,
        };
        client
            .load_model(&model, true, PromptStyle::Instructed, None)
            .expect("load");
        client.shutdown();
        // After a shutdown the next request must start a fresh process rather
        // than talking to the dead one.
        client
            .load_model(&model, true, PromptStyle::Instructed, None)
            .expect("should restart after shutdown");
        assert!(client.is_running());
        client.shutdown();
    }
}
