//! Curated catalog of enhancement models.
//!
//! Deliberately a short, hand-picked list rather than open Hugging Face search.
//! The enhancement layer only works if the model is small enough to run
//! alongside a transcription model on a weak machine *and* obedient enough to
//! return an edit rather than a conversation, and most models on the Hub fail
//! one of those tests. Users who want something else can still point the layer
//! at any local GGUF file.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// What a model is offered for.
///
/// Editing and verifying want different things: an editor must rewrite text
/// obediently, while a verifier only answers one question about two passages.
/// The smallest models can do the first acceptably but are too unreliable to
/// be trusted as a safety net, so they are not offered as verifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "lowercase")]
pub enum ModelRole {
    /// Rewrites the transcript.
    Editor,
    /// Checks that a rewrite preserved the meaning.
    Verifier,
}

/// How the host should prompt a model.
///
/// This is a property of the weights, not a preference. Getting it wrong is not
/// a small regression: a fine-tune given the instruction prompt starts copying
/// the prompt's own rules and few-shot examples into the user's document.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "lowercase")]
pub enum PromptStyle {
    /// Send the shipped instruction prompt. Correct for a stock
    /// instruction-tuned model, which has to be told the task.
    #[default]
    Instructed,
    /// Send an *empty* system turn. Correct for a model fine-tuned on this
    /// task: the behaviour is in the weights already. Measured on the first
    /// fine-tune, adding the prompt cost 57/68 -> 54/68.
    ///
    /// Empty, not absent. Measured on the current editor, dropping the system
    /// message instead of emptying it cost 66/68 -> 59/68, with the model
    /// answering "SAME" and "CHANGED" to editing requests — it stopped
    /// recognising the shape of its own input. So the sidecar passes the empty
    /// string straight through and lets the template decide; do not "tidy" the
    /// message away.
    ///
    /// The mechanism is *not* the model's own jinja template — rendering that
    /// directly drops an empty system block, giving identical text either way.
    /// `llama.cpp` ignores the jinja source and renders its own built-in chatml,
    /// which emits `<|im_start|>system\n<|im_end|>` even when the content is
    /// empty. So the two paths genuinely differ, and the empty block is the one
    /// that measures better. Reason about the built prompt, not the template:
    /// `HANDY_LLM_DEBUG_PROMPT=1` on the sidecar prints it.
    Tuned,
    /// Send the whole Alpaca prompt as one turn, built from
    /// [`ALPACA_INSTRUCTION`]. Correct for a model fine-tuned from a *base*
    /// checkpoint on an Alpaca-format corpus.
    ///
    /// The instruction is fixed and shared by file with the corpus builder, so
    /// the text at inference is byte-identical to the text seen in training.
    ///
    /// Assembled into a single turn rather than split across the system and user
    /// slots. A converted base-model GGUF often still carries a chat template,
    /// which then wraps a split prompt into a shape the fine-tune has never
    /// seen: measured, that made the model repeat itself until the token budget
    /// ran out and never emit a stop token, at six times the latency of the same
    /// weights given the assembled prompt.
    Alpaca,
}

/// The instruction an [`PromptStyle::Alpaca`] model is prompted with.
///
/// `include_str!` rather than a literal: the corpus builder reads the same file,
/// so the training and inference instruction cannot drift apart. A fine-tune
/// served an instruction a few words off the one it learned degrades quietly,
/// which is the worst way for this to break.
pub const ALPACA_INSTRUCTION: &str =
    include_str!("../../../scripts/enhance-train/alpaca_instruction.txt");

/// Rough capability/cost band, used to steer users to a sensible default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "lowercase")]
pub enum ModelTier {
    /// Runs on almost anything. Weak at spotting self-corrections unless it
    /// was trained for the task — the default editor is a 350M fine-tune and
    /// sits in this band.
    Ultralight,
    /// The recommended balance of size and edit quality.
    Balanced,
    /// Best rewrites, but wants real headroom.
    Quality,
}

/// One downloadable model in the catalog.
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "snake_case")]
pub struct CatalogModel {
    /// Stable identifier used in settings.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Hugging Face repository holding the GGUF.
    pub repo_id: String,
    /// File to fetch from `repo_id`.
    pub filename: String,
    /// Download size in bytes.
    pub size_bytes: u64,
    /// Quantisation of `filename`.
    pub quant: String,
    /// Parameter count, for display.
    pub parameters: String,
    /// SPDX-ish licence identifier, for display.
    pub license: String,
    /// Approximate release date (`YYYY-MM`), so users can weigh age.
    pub released: String,
    /// Whether the model deliberates before answering.
    ///
    /// Editing must suppress this — a reasoning model spends its whole token
    /// budget thinking and never emits an edit. Verification is the opposite:
    /// it is a judgement call on short input, so deliberation helps.
    pub reasoning: bool,
    /// How this model expects to be prompted.
    ///
    /// Defaults to [`PromptStyle::Instructed`], so every stock entry in the
    /// catalog is unaffected and only a fine-tune has to say so.
    #[serde(default)]
    pub prompt_style: PromptStyle,
    /// What this model is offered for.
    pub roles: Vec<ModelRole>,
    /// Approximate resident memory while loaded, in MiB.
    pub min_ram_mb: u64,
    /// Capability band.
    pub tier: ModelTier,
    /// Whether this is the out-of-the-box editing model.
    #[serde(default)]
    pub default_editor: bool,
    /// Whether this is the out-of-the-box verifying model.
    #[serde(default)]
    pub default_verifier: bool,
    /// One-line guidance shown next to the name.
    pub description: String,
}

/// Marks a model id that points at a file on disk rather than at the catalog.
pub const LOCAL_PREFIX: &str = "local:";

impl CatalogModel {
    /// Synthesise an entry for a GGUF the user picked off disk.
    ///
    /// The catalog is curated, but the point of this layer is that people can
    /// train their own editor, and requiring a Hugging Face upload before you
    /// can try your own weights would make every iteration a publishing step.
    /// Giving a local file a catalog-shaped entry means the download manager,
    /// the selector and the pipeline all need no special case: `model_path`
    /// joins `filename` onto the models directory, and joining an absolute path
    /// discards the prefix, so it resolves to the file itself.
    ///
    /// Defaults to [`PromptStyle::Tuned`], because the reason to load your own
    /// GGUF into this layer is that you fine-tuned it for the task. The setting
    /// overrides that for a local file that is really a stock download.
    pub fn from_local(path: &std::path::Path) -> Option<Self> {
        let size_bytes = std::fs::metadata(path).ok()?.len();
        let name = path.file_stem()?.to_string_lossy().to_string();
        let filename = path.to_string_lossy().to_string();
        let upper = filename.to_uppercase();
        let quant = [
            "Q2_K_L", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "IQ3_XXS",
        ]
        .into_iter()
        .find(|q| upper.contains(q))
        .unwrap_or("unknown")
        .to_string();
        // Weights plus a modest allowance for the context and compute buffers.
        let min_ram_mb = size_bytes / (1024 * 1024) + 320;
        Some(Self {
            id: format!("{LOCAL_PREFIX}{filename}"),
            name,
            repo_id: String::new(),
            filename,
            size_bytes,
            quant,
            parameters: String::new(),
            license: String::new(),
            released: String::new(),
            reasoning: false,
            prompt_style: PromptStyle::Tuned,
            roles: vec![ModelRole::Editor],
            min_ram_mb,
            tier: ModelTier::Ultralight,
            default_editor: false,
            default_verifier: false,
            // The path is the most useful thing to show, and it needs no
            // translation.
            description: path.display().to_string(),
        })
    }

    /// Whether this entry points at a file the user chose rather than a download.
    pub fn is_local(&self) -> bool {
        self.id.starts_with(LOCAL_PREFIX)
    }
}

#[derive(Debug, Deserialize)]
struct CatalogFile {
    #[allow(dead_code)]
    catalog_version: u32,
    models: Vec<CatalogModel>,
}

/// The catalog, parsed once from the file embedded at build time.
///
/// Embedding rather than downloading keeps first run fully offline: the user
/// can see and choose a model before the app has any network access.
pub fn catalog() -> &'static [CatalogModel] {
    static CATALOG: OnceLock<Vec<CatalogModel>> = OnceLock::new();
    CATALOG
        .get_or_init(|| {
            let raw = include_str!("models.json");
            match serde_json::from_str::<CatalogFile>(raw) {
                Ok(f) => f.models,
                // The file ships with the binary, so a parse failure is a build
                // error we want surfaced loudly in tests rather than a silent
                // empty catalog in production.
                Err(e) => {
                    debug_assert!(false, "embedded enhancement catalog is malformed: {e}");
                    Vec::new()
                }
            }
        })
        .as_slice()
}

/// Look up a model by its catalog id.
pub fn find(id: &str) -> Option<&'static CatalogModel> {
    catalog().iter().find(|m| m.id == id)
}

/// The editing model chosen when the user has not picked one.
pub fn default_editor() -> Option<&'static CatalogModel> {
    catalog()
        .iter()
        .find(|m| m.default_editor)
        .or_else(|| catalog().first())
}

/// The verifying model chosen when the user has not picked one.
pub fn default_verifier() -> Option<&'static CatalogModel> {
    catalog()
        .iter()
        .find(|m| m.default_verifier)
        .or_else(|| for_role(ModelRole::Verifier).into_iter().next())
}

/// Models offered for `role`, in catalog order.
pub fn for_role(role: ModelRole) -> Vec<&'static CatalogModel> {
    catalog()
        .iter()
        .filter(|m| m.roles.contains(&role))
        .collect()
}

/// Models that fit in `available_mb` of memory, smallest tier first.
///
/// Used to warn before a download that would not run on the machine doing the
/// downloading.
pub fn fitting(available_mb: u64) -> Vec<&'static CatalogModel> {
    catalog()
        .iter()
        .filter(|m| m.min_ram_mb <= available_mb)
        .collect()
}

impl CatalogModel {
    /// Whether this model is offered for `role`.
    pub fn supports(&self, role: ModelRole) -> bool {
        self.roles.contains(&role)
    }

    /// Whether this model plausibly runs given `available_mb` of free memory.
    pub fn fits_in(&self, available_mb: u64) -> bool {
        self.min_ram_mb <= available_mb
    }

    /// Download size in MiB, for display.
    pub fn size_mb(&self) -> u64 {
        self.size_bytes / (1024 * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_catalog_parses() {
        assert!(!catalog().is_empty(), "catalog should not be empty");
    }

    #[test]
    fn exactly_one_default_per_role_is_declared() {
        let editors: Vec<_> = catalog().iter().filter(|m| m.default_editor).collect();
        assert_eq!(editors.len(), 1, "expected exactly one default editor");
        let verifiers: Vec<_> = catalog().iter().filter(|m| m.default_verifier).collect();
        assert_eq!(verifiers.len(), 1, "expected exactly one default verifier");

        assert_eq!(
            default_editor().map(|m| m.id.as_str()),
            Some("magicnothief/handy-editor-350m-q8")
        );
        assert_eq!(
            default_verifier().map(|m| m.id.as_str()),
            Some("qwen/qwen3-1.7b")
        );
    }

    #[test]
    fn defaults_declare_the_role_they_default_to() {
        let e = default_editor().expect("an editor default exists");
        assert!(e.supports(ModelRole::Editor));
        let v = default_verifier().expect("a verifier default exists");
        assert!(v.supports(ModelRole::Verifier));
    }

    #[test]
    fn the_default_verifier_can_reason() {
        // Verification is a judgement call on short input, which is the one
        // place a thinking mode earns its latency.
        let v = default_verifier().expect("a verifier default exists");
        assert!(v.reasoning, "default verifier should be a reasoning model");
    }

    #[test]
    fn only_reasoning_models_are_offered_as_verifiers() {
        // Measured, not assumed: Qwen3 0.6B with thinking suppressed answered
        // SAME to all eight deliberate meaning changes in the verification set
        // — a rubber stamp. With thinking it caught six. A verifier that always
        // agrees is worse than none, because it looks like a safety net.
        for m in for_role(ModelRole::Verifier) {
            assert!(
                m.reasoning,
                "{} cannot deliberate and would rubber-stamp every edit",
                m.id
            );
        }
    }

    #[test]
    fn the_smallest_models_are_not_offered_as_verifiers() {
        // A safety net that is itself unreliable is worse than none, because it
        // silently discards good edits.
        for m in catalog().iter().filter(|m| m.tier == ModelTier::Ultralight) {
            assert!(
                !m.supports(ModelRole::Verifier),
                "{} is too small to be trusted as a verifier",
                m.id
            );
        }
    }

    #[test]
    fn every_model_is_offered_for_at_least_one_role() {
        for m in catalog() {
            assert!(!m.roles.is_empty(), "{} has no role", m.id);
        }
    }

    #[test]
    fn catalog_spans_the_full_size_range() {
        // The point of the list is choice across hardware; a catalog that
        // clustered in one band would not serve the low end or the high end.
        for tier in [
            ModelTier::Ultralight,
            ModelTier::Balanced,
            ModelTier::Quality,
        ] {
            assert!(
                catalog().iter().any(|m| m.tier == tier),
                "no model in tier {tier:?}"
            );
        }
    }

    #[test]
    fn ids_are_unique() {
        let mut ids: Vec<&str> = catalog().iter().map(|m| m.id.as_str()).collect();
        let total = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), total, "duplicate model ids in catalog");
    }

    #[test]
    fn the_default_editor_is_one_that_scored_full_marks() {
        // Only these have been measured at 68/68 on the self-correction suite.
        // Every stock model below 4B topped out at 66/68, and several inverted
        // meaning rather than merely missing an edit, so defaulting to an
        // unmeasured model would be shipping a known-wrong answer. The 350M
        // entries are not exceptions to that floor — they are a fine-tune, which
        // is the only way anything that small reaches full marks.
        const FULL_MARKS: &[&str] = &[
            "magicnothief/handy-editor-350m",
            "magicnothief/handy-editor-350m-q8",
            "qwen/qwen3-4b-instruct-iq3",
            "qwen/qwen3-4b-instruct",
        ];
        let e = default_editor().expect("an editor default exists");
        assert!(
            FULL_MARKS.contains(&e.id.as_str()),
            "{} has no measured 68/68; run scripts/enhance-eval before defaulting to it",
            e.id
        );
        assert!(e.supports(ModelRole::Editor));
    }

    #[test]
    fn the_low_tiers_fit_the_target_hardware() {
        // The layer promises to run alongside a transcription model on a 4 GB
        // machine. Only the explicitly-labelled quality tier may exceed that;
        // anything else has to fit a modest machine or it is mis-tiered.
        for m in catalog() {
            let ceiling = match m.tier {
                ModelTier::Ultralight => 800,
                ModelTier::Balanced => 1600,
                // Raised from 3072 when the default became a 4B: measurement
                // showed nothing below that size handles self-correction
                // reliably, so the quality tier now has to accommodate it.
                ModelTier::Quality => 3584,
            };
            assert!(
                m.min_ram_mb <= ceiling,
                "{} claims {} MiB, too large for tier {:?}",
                m.id,
                m.min_ram_mb,
                m.tier
            );
        }
    }

    #[test]
    fn lookup_finds_known_and_rejects_unknown() {
        assert!(find("qwen/qwen3-0.6b").is_some());
        assert!(find("nonexistent/model").is_none());
    }

    #[test]
    fn fitting_filters_by_available_memory() {
        let tiny = fitting(800);
        assert!(!tiny.is_empty(), "some model must fit in 800 MiB");
        assert!(tiny.iter().all(|m| m.min_ram_mb <= 800));

        // The largest entry should be excluded on a small machine but present
        // when there is headroom.
        let roomy = fitting(4096);
        assert!(roomy.len() > tiny.len());
    }

    #[test]
    fn size_mb_is_derived_from_bytes() {
        let m = find("qwen/qwen3-0.6b").expect("default editor present");
        assert_eq!(m.size_mb(), m.size_bytes / (1024 * 1024));
        assert!(m.fits_in(m.min_ram_mb));
        assert!(!m.fits_in(m.min_ram_mb - 1));
    }

    /// Write a stand-in GGUF and hand back its path.
    fn stub_gguf(dir: &tempfile::TempDir, name: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, vec![0u8; 4096]).expect("write stub model");
        path
    }

    #[test]
    fn a_local_file_becomes_a_catalog_shaped_entry() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = stub_gguf(&dir, "checkpoint-10000.Q4_K_M.gguf");
        let m = CatalogModel::from_local(&path).expect("readable file");

        assert!(m.is_local());
        assert_eq!(m.id, format!("{LOCAL_PREFIX}{}", path.display()));
        assert_eq!(m.name, "checkpoint-10000.Q4_K_M");
        assert_eq!(m.quant, "Q4_K_M");
        assert_eq!(m.size_bytes, 4096);
        assert!(m.supports(ModelRole::Editor));
        assert!(!m.supports(ModelRole::Verifier));
    }

    #[test]
    fn a_local_file_is_assumed_to_be_a_fine_tune() {
        // The only reason to point this layer at your own GGUF is that you
        // trained it for the task, and a fine-tune must not get the prompt.
        let dir = tempfile::tempdir().expect("temp dir");
        let m = CatalogModel::from_local(&stub_gguf(&dir, "mine.gguf")).expect("readable file");
        assert_eq!(m.prompt_style, PromptStyle::Tuned);
        assert_eq!(m.quant, "unknown", "an unrecognised name must not guess");
    }

    #[test]
    fn a_missing_local_file_has_no_entry() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert!(CatalogModel::from_local(&dir.path().join("absent.gguf")).is_none());
    }

    #[test]
    fn the_alpaca_instruction_is_a_single_usable_line() {
        // Shared with the corpus builder through the file system, so a bad edit
        // there reaches inference silently. Assert the shape the builder needs.
        let text = ALPACA_INSTRUCTION.trim();
        assert!(!text.is_empty(), "instruction file is empty");
        assert!(!text.contains('\n'), "must be one line, got: {text}");
        assert!(
            text.len() > 40,
            "suspiciously short for a task description: {text}"
        );
    }

    #[test]
    fn only_the_fine_tuned_editor_departs_from_the_instruction_prompt() {
        // The pipeline reads this field and not the name, so a stock model
        // wrongly marked `Tuned` would silently lose its instructions, and a
        // fine-tune wrongly marked `Instructed` would start copying the
        // prompt's own rules into the user's document. Pin both directions.
        for m in catalog() {
            let want = if m.repo_id == "MagicNoThief/handy-editor-lfm2.5-350m" {
                PromptStyle::Tuned
            } else {
                PromptStyle::Instructed
            };
            assert_eq!(m.prompt_style, want, "{} declares the wrong style", m.id);
        }
    }

    #[test]
    fn no_fine_tuned_editor_is_offered_as_a_verifier() {
        // A model trained only to rewrite has never been asked to judge a
        // rewrite. Measured over 400 live edits the trained editor's verify
        // pass caught 0 of 10 bad ones and rejected a good one, so the pipeline
        // skips verification for a non-`Instructed` model entirely. Offering
        // one as a verifier would present a check that silently does nothing.
        for m in catalog() {
            assert!(
                m.prompt_style == PromptStyle::Instructed || !m.supports(ModelRole::Verifier),
                "{} is a fine-tune and cannot verify",
                m.id
            );
        }
    }

    #[test]
    fn no_catalog_id_could_be_mistaken_for_a_local_path() {
        assert!(catalog().iter().all(|m| !m.is_local()));
    }

    #[test]
    fn joining_a_local_entry_onto_the_models_dir_yields_the_file_itself() {
        // The whole design rests on this: `EnhanceManager::model_path` joins
        // `filename` onto the models directory, and a local entry puts an
        // absolute path there. If joining ever stopped discarding the prefix,
        // every local model would resolve to a path that does not exist.
        let dir = tempfile::tempdir().expect("temp dir");
        let path = stub_gguf(&dir, "mine.gguf");
        let m = CatalogModel::from_local(&path).expect("readable file");
        let models_dir = std::path::Path::new("/some/other/models/dir");
        assert_eq!(models_dir.join(&m.filename), path);
    }
}
