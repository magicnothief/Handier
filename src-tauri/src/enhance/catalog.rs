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

/// Rough capability/cost band, used to steer users to a sensible default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "lowercase")]
pub enum ModelTier {
    /// Runs on almost anything; weakest at spotting self-corrections.
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
            Some("qwen/qwen3-0.6b")
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
    fn the_low_tiers_fit_the_target_hardware() {
        // The layer promises to run alongside a transcription model on a 4 GB
        // machine. Only the explicitly-labelled quality tier may exceed that;
        // anything else has to fit a modest machine or it is mis-tiered.
        for m in catalog() {
            let ceiling = match m.tier {
                ModelTier::Ultralight => 800,
                ModelTier::Balanced => 1600,
                ModelTier::Quality => 3072,
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
}
