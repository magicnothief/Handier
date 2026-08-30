//! GGUF inference for the enhancement layer.
//!
//! Deliberately small: load a model, run one prompt to completion, unload.
//! There is no streaming, no conversation state and no KV-cache reuse between
//! requests, because the caller's unit of work is a whole transcript and the
//! text is not shown until the pass finishes.

use anyhow::{anyhow, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::TokenToStringError;
use std::num::NonZeroU32;
use std::path::Path;

/// Context window used when the caller does not ask for one.
///
/// Dictation is short and KV-cache memory scales with this number, so the
/// default is sized for a long paragraph rather than for the model's maximum.
const DEFAULT_CTX: u32 = 2048;

/// Upper bound on inference threads.
///
/// Past a handful of threads a sub-1B model is memory-bound rather than
/// compute-bound, and on the low-core machines this targets, saturating every
/// core makes the rest of the desktop stutter while the user waits to paste.
const MAX_THREADS: i32 = 4;

/// A loaded model, ready to serve generation requests.
///
/// The llama backend is process-global and can only be initialised once, so
/// there is at most one useful `Engine` per process — a second [`Engine::new`]
/// fails with `BackendAlreadyInitialized`. That suits the sidecar, which is one
/// process serving one model, but it does mean the engine cannot be constructed
/// ad hoc; tests share a single instance.
pub struct Engine {
    backend: LlamaBackend,
    model: Option<LoadedModel>,
}

struct LoadedModel {
    model: LlamaModel,
    n_ctx: u32,
    n_threads: i32,
}

impl Engine {
    /// Initialise the llama backend without loading a model.
    pub fn new() -> Result<Self> {
        let backend = LlamaBackend::init().context("failed to initialise llama backend")?;
        Ok(Self {
            backend,
            model: None,
        })
    }

    /// Whether a model is currently resident.
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Load `path`, replacing any model already resident.
    ///
    /// The previous model is dropped first so peak memory is one model, not
    /// two — the difference between working and swapping on a 4 GB machine.
    ///
    /// `gpu_layers` is how many transformer layers to offload: `None` means
    /// "offload everything if this build has a GPU backend", and `Some(0)`
    /// forces CPU. A GPU attempt that fails falls back to CPU rather than
    /// erroring, because the same binary ships to machines with no usable
    /// device and a missing driver must not cost the user the feature.
    pub fn load(
        &mut self,
        path: &Path,
        threads: Option<i32>,
        ctx: Option<u32>,
        gpu_layers: Option<u32>,
    ) -> Result<()> {
        self.model = None;

        if !path.exists() {
            return Err(anyhow!("model file not found: {}", path.display()));
        }

        let requested = gpu_layers.unwrap_or(default_gpu_layers());

        let model = match self.try_load(path, requested) {
            Ok(m) => m,
            Err(e) if requested > 0 => {
                eprintln!(
                    "handy-llm: GPU load failed ({e:#}); retrying on CPU. \
                     Enhancement will be slower but still works."
                );
                self.try_load(path, 0)
                    .with_context(|| format!("failed to load model at {} on CPU", path.display()))?
            }
            Err(e) => return Err(e),
        };

        let n_threads = threads
            .unwrap_or_else(default_threads)
            .clamp(1, MAX_THREADS);
        let n_ctx = ctx.unwrap_or(DEFAULT_CTX).max(256);

        self.model = Some(LoadedModel {
            model,
            n_ctx,
            n_threads,
        });

        if requested > 0 {
            self.warm_up();
        }
        Ok(())
    }

    /// Run a throwaway generation so the first real request is not the one that
    /// pays for pipeline setup.
    ///
    /// Vulkan compiles its compute pipelines lazily, and which pipelines it
    /// needs depends on the shape of the batch. A token-sized warm-up therefore
    /// does not cover a real request: with one, the first genuine enhancement
    /// still took 13 s on an RTX 3060 against ~130 ms for later ones. The dummy
    /// prompt below is sized like an actual enhancement prompt so the pipelines
    /// that get compiled are the ones the next request will use.
    ///
    /// Failures are ignored: this is an optimisation, and a model that cannot
    /// warm up will surface its error on the real request.
    fn warm_up(&self) {
        let started = std::time::Instant::now();
        // Roughly the token count of a real system prompt plus a dictated
        // sentence, without being any particular prompt.
        let system = "You edit dictated speech into the text the speaker meant to write.              Remove filler words and hesitations. Fix sentence boundaries, capitalisation              and punctuation. If the speaker corrects themselves, delete the abandoned              wording and keep only what they settled on. Reply with the edited text and              nothing else, with no preamble, explanation or quotation marks around it.              Keep the speaker's own words, voice and language throughout the reply.";
        let user = "um so the quarterly planning meeting is uh moved to friday afternoon              instead of thursday morning because several people have a conflict";
        match self.generate(system, user, 16, true) {
            Ok(_) => eprintln!("handy-llm: warm-up took {:?}", started.elapsed()),
            Err(e) => eprintln!("handy-llm: warm-up failed (ignored): {e:#}"),
        }
    }

    fn try_load(&self, path: &Path, gpu_layers: u32) -> Result<LlamaModel> {
        let params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
        LlamaModel::load_from_file(&self.backend, path, &params)
            .with_context(|| format!("failed to load model at {}", path.display()))
    }

    /// Drop the model and release its memory.
    pub fn unload(&mut self) {
        self.model = None;
    }

    /// Run one editing pass and return the completion.
    ///
    /// `no_think` appends the soft switch that reasoning models (Qwen3 and
    /// relatives) understand as "answer directly". Left on, such a model spends
    /// its whole token budget deliberating and never emits an answer — measured
    /// at roughly four seconds of pure thinking for a one-line transcript,
    /// against one second for the same edit with thinking suppressed.
    pub fn generate(
        &self,
        system: &str,
        user: &str,
        max_tokens: usize,
        no_think: bool,
    ) -> Result<String> {
        let loaded = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("no model is loaded"))?;
        let model = &loaded.model;

        let user = if no_think {
            format!("{user}\n/no_think")
        } else {
            user.to_string()
        };
        let (prompt, templated) = build_prompt(model, system, &user)?;

        // A chat template that needs a BOS emits one as text, so adding another
        // would prepend a duplicate and measurably degrade small models.
        let add_bos = if templated {
            AddBos::Never
        } else {
            AddBos::Always
        };
        let tokens = model
            .str_to_token(&prompt, add_bos)
            .map_err(|e| anyhow!("failed to tokenize prompt: {e}"))?;

        if tokens.is_empty() {
            return Err(anyhow!("prompt tokenized to nothing"));
        }

        // Reserve room for the answer inside the window rather than letting the
        // prompt fill it and leaving nothing to generate into.
        let needed = tokens.len().saturating_add(max_tokens).saturating_add(8);
        let n_ctx = loaded.n_ctx.max(u32::try_from(needed).unwrap_or(u32::MAX));

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx)
            .with_n_threads(loaded.n_threads)
            .with_n_threads_batch(loaded.n_threads);

        let mut ctx = model
            .new_context(&self.backend, ctx_params)
            .context("failed to create inference context")?;

        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        let last = tokens.len() - 1;
        for (i, token) in tokens.iter().enumerate() {
            let pos = i32::try_from(i).context("prompt longer than i32 positions")?;
            // Only the final token needs logits; it is the one we sample from.
            batch
                .add(*token, pos, &[0], i == last)
                .map_err(|e| anyhow!("failed to build prompt batch: {e}"))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| anyhow!("failed to evaluate prompt: {e}"))?;

        // Greedy. Editing a transcript has a right answer; sampling would only
        // add a chance of inventing words the speaker never said.
        let mut sampler = LlamaSampler::greedy();

        let mut out = Vec::with_capacity(max_tokens);
        let mut n_cur = i32::try_from(tokens.len()).context("prompt too long")?;

        for _ in 0..max_tokens {
            let token = sampler.sample(&ctx, -1);
            if model.is_eog_token(token) {
                break;
            }
            out.push(token);
            sampler.accept(token);

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| anyhow!("failed to build decode batch: {e}"))?;
            ctx.decode(&mut batch)
                .map_err(|e| anyhow!("failed to decode token: {e}"))?;
            n_cur += 1;
        }

        // Collect raw bytes and decode once at the end. Two reasons: a
        // multi-byte character can span two tokens, so per-token decoding can
        // split it; and `tokens_to_str` is deprecated and asks the FFI for a
        // fixed 8-byte piece buffer without retrying, so any longer piece fails
        // with `InsufficientBufferSpace`. `token_to_bytes` does retry.
        let mut bytes = Vec::with_capacity(out.len() * 4);
        for token in &out {
            let piece = token_bytes(model, *token)?;
            bytes.extend_from_slice(&piece);
        }
        let text =
            String::from_utf8(bytes).map_err(|e| anyhow!("model produced invalid UTF-8: {e}"))?;
        Ok(strip_thinking(&text).to_string())
    }
}

/// Format the turns for the model, preferring its own chat template.
///
/// Returns the prompt and whether a template was applied. Using the template
/// baked into the GGUF means a new model in the catalog needs no code change,
/// and gets the exact turn markers it was trained on.
fn build_prompt(model: &LlamaModel, system: &str, user: &str) -> Result<(String, bool)> {
    let messages = [
        LlamaChatMessage::new("system".to_string(), system.to_string()),
        LlamaChatMessage::new("user".to_string(), user.to_string()),
    ];
    let messages: Result<Vec<_>, _> = messages.into_iter().collect();
    let messages = messages.map_err(|e| anyhow!("prompt contained a null byte: {e}"))?;

    if let Ok(template) = model.chat_template(None) {
        if let Ok(prompt) = model.apply_chat_template(&template, &messages, true) {
            return Ok((prompt, true));
        }
    }

    // Fall back to a plain layout for a GGUF with no template. Less reliable,
    // but better than refusing to run the model at all.
    Ok((
        format!("{system}\n\nTranscript:\n{user}\n\nEdited:\n"),
        false,
    ))
}

/// Remove a leading reasoning block from a completion.
///
/// Even with the soft switch applied, reasoning models still emit an empty
/// `<think></think>` pair before the answer. Passing that through would land it
/// in the user's document, and it also defeats the host's length checks.
///
/// Only a block at the very start is removed, and only when it is closed: a
/// transcript that legitimately mentions `<think>` mid-sentence is left alone,
/// and an unterminated block means generation was cut off, in which case there
/// is no answer to salvage and the whole thing is discarded.
fn strip_thinking(text: &str) -> &str {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";

    let trimmed = text.trim_start();
    let Some(rest) = trimmed.strip_prefix(OPEN) else {
        return text.trim();
    };
    match rest.find(CLOSE) {
        Some(end) => rest[end + CLOSE.len()..].trim(),
        None => "",
    }
}

/// Raw bytes for one token, growing the buffer if the first guess is too small.
///
/// The convenience wrappers in `llama-cpp-2` are unsuitable here: `tokens_to_str`
/// asks for a fixed 8-byte piece buffer and never retries, so every generation
/// containing a longer piece fails outright, and `token_to_bytes` is deprecated.
/// `token_to_piece_bytes` is the supported call but leaves buffer sizing to the
/// caller, which is what this does — the FFI reports the needed size as a
/// negative length, so a rejected guess tells us exactly what to allocate.
///
/// Callers concatenate these and decode UTF-8 once at the end, because a
/// multi-byte character can span two tokens.
fn token_bytes(model: &LlamaModel, token: LlamaToken) -> Result<Vec<u8>> {
    /// Covers virtually every piece in one attempt without over-allocating.
    const FIRST_GUESS: usize = 32;

    match model.token_to_piece_bytes(token, FIRST_GUESS, true, None) {
        Ok(bytes) => Ok(bytes),
        Err(TokenToStringError::InsufficientBufferSpace(needed)) => {
            let needed = usize::try_from(-needed)
                .map_err(|_| anyhow!("token piece reported a nonsensical size: {needed}"))?;
            model
                .token_to_piece_bytes(token, needed, true, None)
                .map_err(|e| anyhow!("failed to decode token after resize: {e}"))
        }
        Err(e) => Err(anyhow!("failed to decode token: {e}")),
    }
}

/// Whether this build has a GPU backend compiled in.
pub const HAS_GPU_BACKEND: bool =
    cfg!(any(feature = "vulkan", feature = "cuda", feature = "metal"));

/// Layers to offload when the caller expresses no preference.
///
/// A large number means "all of them"; llama.cpp clamps to the layer count.
/// These models are small enough that partial offload is rarely worth the
/// split, so it is all-or-nothing. On a CPU-only build this is zero, which
/// keeps the CPU path free of any GPU bookkeeping.
fn default_gpu_layers() -> u32 {
    if HAS_GPU_BACKEND {
        1_000
    } else {
        0
    }
}

/// Thread count to use when the caller does not specify one.
///
/// Leaves a core free so the UI stays responsive while a pass runs.
fn default_threads() -> i32 {
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(2);
    i32::try_from(cores.saturating_sub(1).max(1)).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_threads_is_at_least_one_and_leaves_headroom() {
        let t = default_threads();
        assert!(t >= 1, "must use at least one thread");
        let cores = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(2);
        if cores > 1 {
            assert!(
                (t as usize) < cores,
                "should leave a core free (got {t} of {cores})"
            );
        }
    }

    #[test]
    fn thread_count_is_clamped_to_the_useful_range() {
        assert_eq!(64_i32.clamp(1, MAX_THREADS), MAX_THREADS);
        assert_eq!(0_i32.clamp(1, MAX_THREADS), 1);
    }

    /// The one `Engine` for this test process.
    ///
    /// The backend is a process-global singleton, so tests must share an
    /// instance rather than each constructing one.
    fn shared_engine() -> &'static std::sync::Mutex<Engine> {
        static ENGINE: std::sync::OnceLock<std::sync::Mutex<Engine>> = std::sync::OnceLock::new();
        ENGINE.get_or_init(|| std::sync::Mutex::new(Engine::new().expect("backend init")))
    }

    #[test]
    fn loading_a_missing_file_is_an_error_not_a_panic() {
        let mut engine = shared_engine().lock().expect("engine poisoned");
        let err = engine
            .load(
                Path::new("definitely-not-a-real-model.gguf"),
                None,
                None,
                None,
            )
            .expect_err("missing file must fail");
        assert!(err.to_string().contains("not found"));
        assert!(!engine.is_loaded());
    }

    #[test]
    fn generating_without_a_model_is_an_error() {
        let mut engine = shared_engine().lock().expect("engine poisoned");
        engine.unload();
        let err = engine
            .generate("system", "user", 16, false)
            .expect_err("must refuse without a model");
        assert!(err.to_string().contains("no model"));
    }

    #[test]
    fn gpu_default_is_all_layers_only_when_a_backend_exists() {
        if HAS_GPU_BACKEND {
            assert!(
                default_gpu_layers() > 0,
                "GPU build should offload by default"
            );
        } else {
            assert_eq!(
                default_gpu_layers(),
                0,
                "CPU build must not request offload"
            );
        }
    }

    #[test]
    fn strips_an_empty_reasoning_block() {
        // What a reasoning model emits once the soft switch is applied.
        assert_eq!(
            strip_thinking("<think>\n\n</think>\n\nSend it to Jane."),
            "Send it to Jane."
        );
    }

    #[test]
    fn strips_a_populated_reasoning_block() {
        assert_eq!(
            strip_thinking("<think>The user said X, so I should Y.</think> The answer."),
            "The answer."
        );
    }

    #[test]
    fn leaves_text_without_a_reasoning_block_alone() {
        assert_eq!(strip_thinking("  Send it to Jane.  "), "Send it to Jane.");
    }

    #[test]
    fn ignores_a_think_tag_that_is_not_at_the_start() {
        // A transcript may legitimately contain the word in angle brackets;
        // only a leading block is reasoning output.
        let s = "I wrote <think> in the doc and it broke";
        assert_eq!(strip_thinking(s), s);
    }

    #[test]
    fn discards_an_unterminated_reasoning_block() {
        // Generation was cut off mid-thought, so there is no answer to keep.
        // Returning the reasoning text would paste the model's monologue.
        assert_eq!(strip_thinking("<think>I am still thinking about"), "");
    }

    #[test]
    fn backend_refuses_a_second_initialisation() {
        // Pins the singleton constraint the sidecar's design depends on.
        let _shared = shared_engine();
        assert!(
            Engine::new().is_err(),
            "a second backend init must fail, not silently succeed"
        );
    }
}
