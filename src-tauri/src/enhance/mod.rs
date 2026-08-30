//! Local transcript enhancement.
//!
//! Sits between transcription and paste: takes the final transcript, asks a
//! small locally-run model to edit it into what the speaker meant to write, and
//! hands back either the rewrite or — whenever anything looks wrong — the
//! original text untouched.
//!
//! The rule this module is built around is that a dictation app that sometimes
//! eats your sentence is worse than one that never edits it. Every failure path
//! here returns the raw transcript rather than propagating an error to the user.

pub mod catalog;
pub mod prompt;
pub mod sidecar;
pub mod verify;

use anyhow::Result;
use log::{debug, warn};
use std::time::{Duration, Instant};

pub use catalog::{CatalogModel, ModelRole, ModelTier};
pub use prompt::{Aggressiveness, AppContext, EnhanceOptions, Rejection};
pub use sidecar::SidecarClient;
pub use verify::{Verdict, VerifyMode};

/// Sampling parameters for an enhancement pass.
///
/// Editing is close to a deterministic task, so this is deliberately not
/// exposed as user-tunable: sampling temperature is a good way to turn a
/// working transcript into a creative one.
#[derive(Debug, Clone)]
pub struct GenParams {
    /// Hard ceiling on generated tokens, derived from the input length.
    pub max_tokens: usize,
    /// Abandon the pass if it outruns this, and paste the raw transcript.
    pub timeout: Duration,
}

impl GenParams {
    /// Budget a pass for a transcript of `chars` characters.
    ///
    /// An edit is roughly the length of its input, so the ceiling is the input
    /// plus generous headroom. This exists to bound a runaway generation, not
    /// to shape the output.
    pub fn for_input(chars: usize) -> Self {
        let approx_tokens = chars / 3 + 32;
        Self {
            max_tokens: approx_tokens.saturating_mul(2).clamp(64, 2048),
            timeout: Duration::from_secs(30),
        }
    }
}

/// A loaded model that can run an enhancement pass.
///
/// Exists so the orchestration below is independent of how inference is hosted.
#[allow(dead_code)]
pub trait Backend: Send + Sync {
    /// Run one generation and return the raw completion text.
    fn generate(&self, system: &str, user: &str, params: &GenParams) -> Result<String>;

    /// Whether a model is loaded and ready to serve a request.
    fn is_ready(&self) -> bool;
}

/// Outcome of an enhancement pass, including why it fell back when it did.
#[derive(Debug, Clone)]
pub struct Enhanced {
    /// Text to paste. Equals the input transcript on any fallback path.
    pub text: String,
    /// The untouched transcript, always preserved for history and undo.
    pub original: String,
    /// Set when the model ran but its output was refused.
    pub rejected: Option<Rejection>,
    /// Verdict from the second-pass meaning check, when one ran.
    pub verdict: Option<Verdict>,
    /// Set when the model could not be run at all.
    pub error: Option<String>,
    /// Wall-clock time spent in the pass.
    pub elapsed: Duration,
}

impl Enhanced {
    fn passthrough(original: String, elapsed: Duration) -> Self {
        Self {
            text: original.clone(),
            original,
            rejected: None,
            verdict: None,
            error: None,
            elapsed,
        }
    }

    /// True when the paste text differs from what was transcribed.
    pub fn changed(&self) -> bool {
        self.text != self.original
    }
}

/// Enhance `transcript` with `backend`, falling back to the raw text on any
/// failure.
///
/// Never returns `Err`: a broken enhancement layer degrades to plain dictation
/// rather than costing the user their words.
pub fn enhance(
    backend: &dyn Backend,
    transcript: &str,
    opts: &EnhanceOptions,
    ctx: &AppContext,
) -> Enhanced {
    enhance_with_verifier(backend, backend, transcript, opts, ctx)
}

/// Like [`enhance`], but runs the verification pass on a separate backend.
///
/// Passing the same backend for both is the memory-cheap default: one model
/// stays resident and answers both questions. A dedicated verifier is better at
/// the job — a reasoning model in particular — but costs a second model in RAM,
/// which is the wrong trade on the hardware this targets.
pub fn enhance_with_verifier(
    backend: &dyn Backend,
    verifier: &dyn Backend,
    transcript: &str,
    opts: &EnhanceOptions,
    ctx: &AppContext,
) -> Enhanced {
    let started = Instant::now();
    let original = transcript.to_string();

    let trimmed = transcript.trim();
    if trimmed.is_empty() || opts.is_noop() {
        return Enhanced::passthrough(original, started.elapsed());
    }

    if !backend.is_ready() {
        return Enhanced {
            error: Some("enhancement model is not loaded".to_string()),
            ..Enhanced::passthrough(original, started.elapsed())
        };
    }

    let system = prompt::build_system_prompt(opts, ctx);
    let user = prompt::build_user_prompt(trimmed);
    let params = GenParams::for_input(trimmed.chars().count());

    let raw = match backend.generate(&system, &user, &params) {
        Ok(text) => text,
        Err(e) => {
            warn!("enhancement generation failed, pasting raw transcript: {e}");
            return Enhanced {
                error: Some(e.to_string()),
                ..Enhanced::passthrough(original, started.elapsed())
            };
        }
    };

    let cleaned = strip_wrapping(&raw);

    if let Err(rejection) = prompt::sanity_check(trimmed, cleaned, opts.aggressiveness) {
        warn!("enhancement output rejected ({rejection:?}), pasting raw transcript");
        return Enhanced {
            rejected: Some(rejection),
            ..Enhanced::passthrough(original, started.elapsed())
        };
    }

    // Second pass: the mechanical guards compare words, so they cannot see a
    // reordering that preserves every word. Only a model reading both versions
    // can catch that.
    let edit_took = started.elapsed();
    let verdict = run_verification(verifier, opts.verify, trimmed, cleaned, edit_took);
    if verdict == Some(Verdict::Changed) {
        warn!("verifier judged the meaning changed, pasting raw transcript");
        return Enhanced {
            verdict,
            ..Enhanced::passthrough(original, started.elapsed())
        };
    }

    let elapsed = started.elapsed();
    debug!("enhancement pass completed in {elapsed:?}");
    Enhanced {
        text: cleaned.to_string(),
        original,
        rejected: None,
        verdict,
        error: None,
        elapsed,
    }
}

/// How long an edit pass may take before a second pass is not worth waiting for.
///
/// Verification roughly doubles the wait, and deliberation makes it longer
/// still, so a machine that took this long to edit would take far longer than a
/// dictation flow tolerates to also verify.
const VERIFY_BUDGET: Duration = Duration::from_millis(2500);

/// Whether a second pass is affordable, judged by how long the first one took.
///
/// This is how `Auto` adapts to the machine instead of to a hardware probe. On
/// a GPU an edit takes ~130 ms and verification ~1-2 s, which is fine. On a
/// low-end CPU the same edit can take seconds and verification tens of seconds,
/// which is not — and those are exactly the machines this layer is meant to
/// stay usable on. Users who want the check regardless can choose `Always`.
fn affordable(edit_took: Duration) -> bool {
    edit_took <= VERIFY_BUDGET
}

/// Run the meaning check, or decide it is not worth running.
///
/// Returns `None` when no verification happened. A verifier that errors is
/// reported as `Unclear` rather than failing the edit: the mechanical guards
/// already passed, and a broken safety net should not cost the user a rewrite
/// that looks fine.
fn run_verification(
    verifier: &dyn Backend,
    mode: VerifyMode,
    original: &str,
    edited: &str,
    edit_took: Duration,
) -> Option<Verdict> {
    let wanted = match mode {
        VerifyMode::Off => false,
        VerifyMode::Always => true,
        VerifyMode::Auto => verify::needs_verification(original, edited) && affordable(edit_took),
    };
    if !wanted || !verifier.is_ready() {
        return None;
    }

    let system = verify::build_verify_prompt();
    let input = verify::build_verify_input(original, edited);
    // The answer is one word, but the budget has to cover a reasoning model
    // thinking its way there first, and undersizing it silently destroys the
    // check. Measured across an 8-case set of deliberate meaning changes:
    //
    //   Qwen3 0.6B, no thinking          3/8  (answered SAME to everything)
    //   Qwen3 0.6B, thinking,  640 cap   6/8
    //   Qwen3 1.7B, thinking,  640 cap   6/8  (hardest case truncated mid-thought)
    //   Qwen3 1.7B, thinking, 1400 cap   7/8
    //
    // This is a ceiling, not a target: a model that does not deliberate emits
    // its word and stops at end-of-generation, so the headroom costs it nothing.
    let params = GenParams {
        max_tokens: 1400,
        timeout: Duration::from_secs(20),
    };

    match verifier.generate(&system, &input, &params) {
        Ok(reply) => Some(verify::parse_verdict(&reply)),
        Err(e) => {
            warn!("verification pass failed, keeping the edit: {e}");
            Some(Verdict::Unclear)
        }
    }
}

/// Strip decoration small models like to wrap their answer in.
///
/// Only removes quotes or a fenced block when they enclose the *entire* reply,
/// so a transcript that legitimately opens and closes with a quotation mark
/// survives intact.
fn strip_wrapping(raw: &str) -> &str {
    let mut s = raw.trim();

    if s.starts_with("```") {
        if let Some(rest) = s.strip_prefix("```") {
            // Drop an optional language tag on the opening fence.
            let rest = rest.split_once('\n').map(|(_, r)| r).unwrap_or(rest);
            if let Some(inner) = rest.rsplit_once("```") {
                s = inner.0.trim();
            }
        }
    }

    for (open, close) in [('"', '"'), ('\u{201c}', '\u{201d}'), ('\'', '\'')] {
        if s.chars().count() >= 2 && s.starts_with(open) && s.ends_with(close) {
            let inner = &s[open.len_utf8()..s.len() - close.len_utf8()];
            // Refuse to unwrap when the delimiter also occurs inside, which
            // means it is punctuation rather than a wrapper.
            if !inner.contains(open) && !inner.contains(close) {
                s = inner.trim();
            }
        }
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct StubBackend {
        reply: Result<String, String>,
        ready: bool,
        calls: Mutex<usize>,
    }

    impl StubBackend {
        fn replying(text: &str) -> Self {
            Self {
                reply: Ok(text.to_string()),
                ready: true,
                calls: Mutex::new(0),
            }
        }
        fn failing() -> Self {
            Self {
                reply: Err("model exploded".to_string()),
                ready: true,
                calls: Mutex::new(0),
            }
        }
        fn unloaded() -> Self {
            Self {
                reply: Ok(String::new()),
                ready: false,
                calls: Mutex::new(0),
            }
        }
        fn call_count(&self) -> usize {
            *self.calls.lock().expect("call counter poisoned")
        }
    }

    impl Backend for StubBackend {
        fn generate(&self, _system: &str, _user: &str, _params: &GenParams) -> Result<String> {
            *self.calls.lock().expect("call counter poisoned") += 1;
            self.reply.clone().map_err(|e| anyhow::anyhow!(e))
        }
        fn is_ready(&self) -> bool {
            self.ready
        }
    }

    fn opts() -> EnhanceOptions {
        EnhanceOptions::default()
    }

    #[test]
    fn returns_the_rewrite_when_it_passes_checks() {
        let b = StubBackend::replying("So the meeting is moved to Friday at three.");
        let out = enhance(
            &b,
            "um so the meeting is uh moved to friday at three",
            &opts(),
            &AppContext::default(),
        );
        assert_eq!(out.text, "So the meeting is moved to Friday at three.");
        assert!(out.changed());
        assert!(out.rejected.is_none());
    }

    #[test]
    fn falls_back_to_raw_transcript_when_generation_fails() {
        let b = StubBackend::failing();
        let raw = "the meeting is friday";
        let out = enhance(&b, raw, &opts(), &AppContext::default());
        assert_eq!(out.text, raw);
        assert!(out.error.is_some());
    }

    #[test]
    fn falls_back_when_output_is_rejected() {
        let b = StubBackend::replying("Friday.");
        let raw = "so I was thinking we could move the meeting to friday afternoon";
        let out = enhance(&b, raw, &opts(), &AppContext::default());
        assert_eq!(out.text, raw);
        assert_eq!(out.rejected, Some(Rejection::TooShort));
    }

    #[test]
    fn original_is_always_preserved() {
        let raw = "um the meeting is friday";
        for b in [
            StubBackend::replying("The meeting is Friday."),
            StubBackend::failing(),
            StubBackend::replying("Friday."),
        ] {
            let out = enhance(&b, raw, &opts(), &AppContext::default());
            assert_eq!(out.original, raw);
        }
    }

    #[test]
    fn skips_the_model_for_empty_input() {
        let b = StubBackend::replying("something");
        let out = enhance(&b, "   ", &opts(), &AppContext::default());
        assert_eq!(out.text, "   ");
        assert_eq!(b.call_count(), 0);
    }

    #[test]
    fn skips_the_model_when_every_feature_is_off() {
        let b = StubBackend::replying("something");
        let off = EnhanceOptions {
            remove_fillers: false,
            fix_self_corrections: false,
            fix_punctuation: false,
            ..Default::default()
        };
        let out = enhance(&b, "hello there", &off, &AppContext::default());
        assert_eq!(out.text, "hello there");
        assert_eq!(b.call_count(), 0);
    }

    #[test]
    fn reports_an_unloaded_model_without_losing_text() {
        let b = StubBackend::unloaded();
        let out = enhance(&b, "hello there", &opts(), &AppContext::default());
        assert_eq!(out.text, "hello there");
        assert!(out.error.is_some());
        assert_eq!(b.call_count(), 0);
    }

    /// A backend that answers edits with one script and verifications with
    /// another, so a test can drive both passes independently.
    struct ScriptedBackend {
        edit: String,
        verify_reply: String,
        verify_calls: Mutex<usize>,
    }

    impl ScriptedBackend {
        fn new(edit: &str, verify_reply: &str) -> Self {
            Self {
                edit: edit.to_string(),
                verify_reply: verify_reply.to_string(),
                verify_calls: Mutex::new(0),
            }
        }
        fn verify_calls(&self) -> usize {
            *self.verify_calls.lock().expect("counter poisoned")
        }
    }

    impl Backend for ScriptedBackend {
        fn generate(&self, system: &str, _user: &str, _p: &GenParams) -> Result<String> {
            // The verification prompt is the one that asks for a one-word answer.
            if system.contains("SAME if the meaning is preserved") {
                *self.verify_calls.lock().expect("counter poisoned") += 1;
                Ok(self.verify_reply.clone())
            } else {
                Ok(self.edit.clone())
            }
        }
        fn is_ready(&self) -> bool {
            true
        }
    }

    #[test]
    fn a_changed_verdict_falls_back_to_the_raw_transcript() {
        // The motivating failure: a word-for-word reordering that the
        // mechanical guards pass but that inverts the meaning.
        let raw = "let's ship it on tuesday actually no let's ship it on thursday so qa has time";
        let swapped =
            "Let's ship it on Thursday actually no let's ship it on Tuesday so QA has time.";
        let b = ScriptedBackend::new(swapped, "CHANGED");

        // Guards alone accept it — that is precisely why verification exists.
        assert_eq!(
            prompt::sanity_check(raw, swapped, Aggressiveness::Light),
            Ok(())
        );

        let out = enhance(&b, raw, &opts(), &AppContext::default());
        assert_eq!(out.text, raw, "a changed verdict must not reach the user");
        assert_eq!(out.verdict, Some(Verdict::Changed));
    }

    #[test]
    fn a_same_verdict_keeps_the_edit() {
        let raw = "send it to john no wait not john send it to jane instead please";
        let b = ScriptedBackend::new("Send it to Jane instead please.", "SAME");
        let out = enhance(&b, raw, &opts(), &AppContext::default());
        assert_eq!(out.text, "Send it to Jane instead please.");
        assert_eq!(out.verdict, Some(Verdict::Same));
    }

    #[test]
    fn an_unclear_verdict_keeps_the_edit() {
        // The guards already passed; an unreadable second opinion is no reason
        // to discard a rewrite that otherwise looks fine.
        let raw = "send it to john no wait not john send it to jane instead please";
        let b = ScriptedBackend::new("Send it to Jane instead please.", "I am not sure");
        let out = enhance(&b, raw, &opts(), &AppContext::default());
        assert_eq!(out.text, "Send it to Jane instead please.");
        assert_eq!(out.verdict, Some(Verdict::Unclear));
    }

    #[test]
    fn cosmetic_edits_do_not_spend_a_verification_pass() {
        // Auto mode exists so the common case stays at one inference pass.
        let raw = "um so the meeting is uh moved to friday at three";
        let b = ScriptedBackend::new("So the meeting is moved to Friday at three.", "SAME");
        let out = enhance(&b, raw, &opts(), &AppContext::default());
        assert_eq!(
            b.verify_calls(),
            0,
            "cosmetic edit should skip verification"
        );
        assert_eq!(out.verdict, None);
        assert!(out.changed());
    }

    #[test]
    fn verify_off_never_calls_the_verifier() {
        let raw = "send it to john no wait not john send it to jane instead please";
        let b = ScriptedBackend::new("Send it to Jane instead please.", "CHANGED");
        let off = EnhanceOptions {
            verify: VerifyMode::Off,
            ..Default::default()
        };
        let out = enhance(&b, raw, &off, &AppContext::default());
        assert_eq!(b.verify_calls(), 0);
        // Without verification the swap-prone edit is accepted, which is the
        // trade the user makes by turning it off.
        assert_eq!(out.text, "Send it to Jane instead please.");
    }

    #[test]
    fn a_slow_edit_skips_verification_in_auto_mode() {
        // On a machine slow enough that the edit itself took seconds, a second
        // pass would push the wait past what dictation tolerates.
        assert!(affordable(Duration::from_millis(130)), "GPU-speed edit");
        assert!(affordable(Duration::from_millis(800)), "fast CPU edit");
        assert!(
            !affordable(Duration::from_secs(6)),
            "a six-second edit means verification would cost far more"
        );
    }

    #[test]
    fn always_mode_verifies_regardless_of_how_slow_the_edit_was() {
        // The budget is an `Auto` heuristic. A user who explicitly chose
        // `Always` has accepted the cost.
        let b = ScriptedBackend::new("Send it to Jane instead please.", "CHANGED");
        let always = EnhanceOptions {
            verify: VerifyMode::Always,
            ..Default::default()
        };
        let verdict = run_verification(
            &b,
            always.verify,
            "send it to john no wait not john send it to jane instead please",
            "Send it to Jane instead please.",
            Duration::from_secs(30),
        );
        assert_eq!(verdict, Some(Verdict::Changed));
        assert_eq!(b.verify_calls(), 1);
    }

    #[test]
    fn verify_always_checks_even_a_cosmetic_edit() {
        let raw = "um so the meeting is uh moved to friday at three";
        let b = ScriptedBackend::new("So the meeting is moved to Friday at three.", "SAME");
        let always = EnhanceOptions {
            verify: VerifyMode::Always,
            ..Default::default()
        };
        enhance(&b, raw, &always, &AppContext::default());
        assert_eq!(b.verify_calls(), 1);
    }

    #[test]
    fn a_separate_verifier_backend_is_used_for_the_second_pass() {
        let raw = "send it to john no wait not john send it to jane instead please";
        let editor = ScriptedBackend::new("Send it to John instead please.", "SAME");
        let verifier = ScriptedBackend::new("unused", "CHANGED");
        let out = enhance_with_verifier(&editor, &verifier, raw, &opts(), &AppContext::default());
        assert_eq!(
            verifier.verify_calls(),
            1,
            "verifier backend should be asked"
        );
        assert_eq!(editor.verify_calls(), 0, "editor should not self-verify");
        assert_eq!(out.text, raw);
    }

    #[test]
    fn strips_quotes_that_wrap_the_whole_reply() {
        assert_eq!(
            strip_wrapping("\"The meeting is Friday.\""),
            "The meeting is Friday."
        );
        assert_eq!(strip_wrapping("  plain text  "), "plain text");
    }

    #[test]
    fn keeps_quotes_that_are_part_of_the_text() {
        let s = "\"Ship it\" is what he said, then \"no\".";
        assert_eq!(strip_wrapping(s), s);
    }

    #[test]
    fn strips_a_fenced_block() {
        assert_eq!(
            strip_wrapping("```\nThe meeting is Friday.\n```"),
            "The meeting is Friday."
        );
        assert_eq!(
            strip_wrapping("```text\nThe meeting is Friday.\n```"),
            "The meeting is Friday."
        );
    }

    #[test]
    fn token_budget_scales_with_input_and_stays_bounded() {
        assert_eq!(GenParams::for_input(0).max_tokens, 64);
        assert!(GenParams::for_input(300).max_tokens > 64);
        assert_eq!(GenParams::for_input(1_000_000).max_tokens, 2048);
    }
}
