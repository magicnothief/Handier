//! Wire protocol between Handy and the inference sidecar.
//!
//! One JSON object per line in each direction. Line-delimited JSON is enough
//! for a request/response pair carrying a transcript, and it keeps the sidecar
//! debuggable by hand — you can drive it from a terminal without a client.
//!
//! Every response carries the `id` of the request that produced it so the host
//! can match them up, and failures are reported as `ok: false` rather than by
//! exiting, so one bad request does not cost the caller a warm model.

use serde::{Deserialize, Serialize};

/// A request from the host application.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Request {
    /// Liveness check that does not touch the model.
    Ping {
        /// Correlation id echoed back in the response.
        id: u64,
    },
    /// Load a GGUF from disk, replacing any currently loaded model.
    Load {
        id: u64,
        /// Absolute path to the `.gguf` file.
        path: String,
        /// Inference threads. Omitted means "pick something conservative".
        #[serde(default)]
        threads: Option<i32>,
        /// Context window in tokens. Omitted means the built-in default.
        #[serde(default)]
        ctx: Option<u32>,
        /// Transformer layers to offload to the GPU. Omitted means "all of
        /// them if this build has a GPU backend"; `0` forces CPU.
        #[serde(default)]
        gpu_layers: Option<u32>,
    },
    /// Run one enhancement pass.
    Generate {
        id: u64,
        /// System prompt describing the editing task.
        system: String,
        /// The raw transcript to edit.
        user: String,
        /// Hard ceiling on generated tokens.
        #[serde(default)]
        max_tokens: Option<usize>,
        /// Suppress reasoning output on models that deliberate by default.
        #[serde(default)]
        no_think: Option<bool>,
    },
    /// Drop the model and release its memory, staying alive for a later load.
    Unload { id: u64 },
    /// Exit the process.
    Shutdown { id: u64 },
}

impl Request {
    /// Correlation id, so the host can match a reply to its request.
    pub fn id(&self) -> u64 {
        match self {
            Self::Ping { id }
            | Self::Load { id, .. }
            | Self::Generate { id, .. }
            | Self::Unload { id }
            | Self::Shutdown { id } => *id,
        }
    }
}

/// A response to the host application.
#[derive(Debug, Clone, Serialize)]
pub struct Response {
    /// Correlation id of the originating request.
    pub id: u64,
    /// Whether the request succeeded.
    pub ok: bool,
    /// Generated text, present on a successful `generate`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Human-readable failure reason, present when `ok` is false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Milliseconds spent handling the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
}

impl Response {
    /// A bare success with no payload.
    pub fn ok(id: u64) -> Self {
        Self {
            id,
            ok: true,
            text: None,
            error: None,
            elapsed_ms: None,
        }
    }

    /// A successful generation.
    pub fn generated(id: u64, text: String, elapsed_ms: u64) -> Self {
        Self {
            id,
            ok: true,
            text: Some(text),
            error: None,
            elapsed_ms: Some(elapsed_ms),
        }
    }

    /// A failure carrying the reason.
    pub fn error(id: u64, error: impl std::fmt::Display) -> Self {
        Self {
            id,
            ok: false,
            text: None,
            error: Some(error.to_string()),
            elapsed_ms: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_command() {
        let cases = [
            (r#"{"cmd":"ping","id":1}"#, 1),
            (r#"{"cmd":"load","id":2,"path":"m.gguf"}"#, 2),
            (r#"{"cmd":"generate","id":3,"system":"s","user":"u"}"#, 3),
            (r#"{"cmd":"unload","id":4}"#, 4),
            (r#"{"cmd":"shutdown","id":5}"#, 5),
        ];
        for (json, want_id) in cases {
            let req: Request = serde_json::from_str(json)
                .unwrap_or_else(|e| panic!("failed to parse {json}: {e}"));
            assert_eq!(req.id(), want_id);
        }
    }

    #[test]
    fn load_accepts_optional_tuning() {
        let req: Request =
            serde_json::from_str(r#"{"cmd":"load","id":1,"path":"m.gguf","threads":2,"ctx":1024}"#)
                .expect("parses");
        match req {
            Request::Load { threads, ctx, .. } => {
                assert_eq!(threads, Some(2));
                assert_eq!(ctx, Some(1024));
            }
            other => panic!("expected Load, got {other:?}"),
        }
    }

    #[test]
    fn unknown_command_is_rejected_rather_than_ignored() {
        assert!(serde_json::from_str::<Request>(r#"{"cmd":"nope","id":1}"#).is_err());
    }

    #[test]
    fn success_response_omits_error_field() {
        let json =
            serde_json::to_string(&Response::generated(7, "hi".into(), 12)).expect("serialises");
        assert!(json.contains("\"text\":\"hi\""));
        assert!(!json.contains("error"));
    }

    #[test]
    fn error_response_omits_text_field() {
        let json = serde_json::to_string(&Response::error(7, "boom")).expect("serialises");
        assert!(json.contains("\"error\":\"boom\""));
        assert!(!json.contains("\"text\""));
    }

    #[test]
    fn responses_are_single_line() {
        // The host reads line-delimited JSON; an embedded newline would desync
        // the stream.
        let json =
            serde_json::to_string(&Response::generated(1, "a\nb".into(), 1)).expect("serialises");
        assert!(
            !json.contains('\n'),
            "serialised response must not contain a raw newline"
        );
    }
}
