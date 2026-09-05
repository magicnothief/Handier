//! Owns the enhancement sidecar and the models it can load.
//!
//! Downloads reuse the `hf-hub` client the transcription models already use, so
//! enhancement models land in the same cache and honour the same `HF_HOME`.

use anyhow::{anyhow, Context, Result};
use hf_hub::api::tokio::Progress;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};
use tauri_specta::Event as _;

use crate::enhance::catalog::{self, CatalogModel, ModelRole, PromptStyle};
use crate::enhance::{self, AppContext, EnhanceOptions, Enhanced, SidecarClient};

/// A catalog entry plus whether it is present on disk.
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct EnhanceModelInfo {
    /// The catalog entry.
    #[serde(flatten)]
    pub model: CatalogModel,
    /// Whether the GGUF has been downloaded.
    pub downloaded: bool,
    /// Whether this model is currently resident in the sidecar.
    pub loaded: bool,
    /// Whether a download is in flight right now.
    ///
    /// Read from the manager rather than remembered by the UI, so a settings
    /// page that was closed and reopened mid-download shows the download
    /// instead of an idle button.
    pub downloading: bool,
    /// Percentage complete, when a download is in flight.
    pub progress: Option<f64>,
}

/// Runtime state of the enhancement layer, for the settings UI.
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct EnhanceStatus {
    /// Whether the sidecar binary was found at all.
    pub available: bool,
    /// Whether the sidecar process is running.
    pub running: bool,
    /// Catalog id of the resident model, if any.
    pub loaded_model_id: Option<String>,
    /// Why the layer is unusable, when it is.
    pub error: Option<String>,
}

/// Where a download has got to.
///
/// The terminal states are carried on the same event as progress so a listener
/// cannot miss the end of a download by subscribing to the wrong channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "lowercase")]
pub enum DownloadState {
    /// Bytes are still arriving.
    Running,
    /// The file is on disk and passed its size check.
    Done,
    /// The download failed; `error` says why.
    Failed,
}

/// Progress of an enhancement model download, sent to the settings UI.
///
/// A typed `tauri_specta` event rather than a bare `emit`, so the frontend gets
/// a generated listener and the payload type reaches `bindings.ts`.
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type, tauri_specta::Event)]
#[serde(rename_all = "camelCase")]
pub struct EnhanceDownloadProgress {
    /// Catalog id of the model being fetched.
    pub model_id: String,
    /// Bytes received so far.
    pub downloaded: u64,
    /// Total bytes expected, or 0 before the size is known.
    pub total: u64,
    /// Convenience percentage, so the UI does not repeat the division.
    pub percentage: f64,
    /// Whether the download is still running, finished, or failed.
    pub state: DownloadState,
    /// Why it failed, when it did.
    pub error: Option<String>,
}

/// Reports download progress to the frontend as `enhance-download-progress`.
///
/// A separate event from the transcription models' so the two model pickers
/// cannot show each other's progress. hf-hub clones the reporter, so the
/// running totals live behind an `Arc`.
#[derive(Clone)]
struct DownloadReporter {
    app: AppHandle,
    model_id: String,
    state: Arc<Mutex<ReporterState>>,
    /// The manager's in-flight registry, so a reopened settings page can read
    /// the latest progress instead of waiting for the next event.
    registry: Downloads,
}

struct ReporterState {
    total: u64,
    downloaded: u64,
    last_emit: Instant,
}

/// In-flight downloads, keyed by catalog id.
type Downloads = Arc<Mutex<HashMap<String, EnhanceDownloadProgress>>>;

impl DownloadReporter {
    fn new(app: AppHandle, model_id: String, registry: Downloads) -> Self {
        Self {
            app,
            model_id,
            state: Arc::new(Mutex::new(ReporterState {
                total: 0,
                downloaded: 0,
                last_emit: Instant::now(),
            })),
            registry,
        }
    }

    fn emit(&self, downloaded: u64, total: u64) {
        let percentage = if total > 0 {
            (downloaded as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        publish(
            &self.app,
            &self.registry,
            EnhanceDownloadProgress {
                model_id: self.model_id.clone(),
                downloaded,
                total,
                percentage,
                state: DownloadState::Running,
                error: None,
            },
        );
    }
}

/// Record progress in the registry and send it to the frontend.
///
/// Both halves matter: the event drives a page that is already open, and the
/// registry answers a page that opens later. Keeping them in one function is
/// what stops the two from disagreeing.
fn publish(app: &AppHandle, registry: &Downloads, update: EnhanceDownloadProgress) {
    if let Ok(mut map) = registry.lock() {
        match update.state {
            DownloadState::Running => {
                map.insert(update.model_id.clone(), update.clone());
            }
            DownloadState::Done | DownloadState::Failed => {
                map.remove(&update.model_id);
            }
        }
    }
    let _ = update.emit(app);
}

impl Progress for DownloadReporter {
    async fn init(&mut self, size: usize, _filename: &str) {
        if let Ok(mut st) = self.state.lock() {
            st.total = size as u64;
            st.downloaded = 0;
            st.last_emit = Instant::now();
        }
        self.emit(0, size as u64);
    }

    async fn update(&mut self, size: usize) {
        let Ok(mut st) = self.state.lock() else {
            return;
        };
        st.downloaded = st.downloaded.saturating_add(size as u64);
        let now = Instant::now();
        // Throttle to ~10 updates a second so a fast link does not flood the
        // event channel, but never drop the final one.
        let done = st.total > 0 && st.downloaded >= st.total;
        if now.duration_since(st.last_emit) < Duration::from_millis(100) && !done {
            return;
        }
        st.last_emit = now;
        let (downloaded, total) = (st.downloaded, st.total);
        drop(st);
        self.emit(downloaded, total);
    }

    async fn finish(&mut self) {
        let (downloaded, total) = match self.state.lock() {
            Ok(st) => (st.downloaded.max(st.total), st.total),
            Err(_) => return,
        };
        self.emit(downloaded, total);
    }
}

/// Manages the enhancement sidecar's lifetime and model files.
pub struct EnhanceManager {
    client: Option<Arc<SidecarClient>>,
    models_dir: PathBuf,
    /// Handle used to report download progress to the frontend.
    app: AppHandle,
    /// Set when the sidecar binary could not be located at startup.
    unavailable_reason: Option<String>,
    /// Downloads currently running, so the state survives a UI that navigates
    /// away and comes back — and so a second request for the same model joins
    /// the running download instead of starting a competing one.
    downloads: Downloads,
}

impl EnhanceManager {
    /// Locate the sidecar and prepare the model directory.
    ///
    /// A missing sidecar is not fatal: the app runs with the enhancement layer
    /// disabled rather than refusing to start.
    pub fn new(app: &AppHandle) -> Result<Self> {
        let models_dir = crate::portable::app_data_dir(app)
            .map_err(|e| anyhow!("failed to resolve app data dir: {e}"))?
            .join("enhance-models");
        std::fs::create_dir_all(&models_dir)
            .with_context(|| format!("failed to create {}", models_dir.display()))?;

        let resource_dir = app.path().resource_dir().ok();
        match SidecarClient::find_binary(resource_dir.as_deref()) {
            Some(exe) => {
                info!("enhancement sidecar found at {}", exe.display());
                Ok(Self {
                    client: Some(Arc::new(SidecarClient::new(exe))),
                    models_dir,
                    app: app.clone(),
                    unavailable_reason: None,
                    downloads: Downloads::default(),
                })
            }
            None => {
                warn!("enhancement sidecar binary not found; the layer will stay disabled");
                Ok(Self {
                    client: None,
                    models_dir,
                    app: app.clone(),
                    unavailable_reason: Some(
                        "The enhancement engine was not found in this build.".to_string(),
                    ),
                    downloads: Downloads::default(),
                })
            }
        }
    }

    /// Where a catalog model's GGUF lives on disk.
    pub fn model_path(&self, model: &CatalogModel) -> PathBuf {
        self.models_dir.join(&model.filename)
    }

    /// Whether a model has been downloaded.
    pub fn is_downloaded(&self, model: &CatalogModel) -> bool {
        let path = self.model_path(model);
        // Compare against the catalog size so a half-finished download is not
        // mistaken for a usable model.
        std::fs::metadata(&path)
            .map(|m| m.len() == model.size_bytes)
            .unwrap_or(false)
    }

    /// The catalog, annotated with on-disk, loaded and downloading state.
    pub fn list_models(&self, role: Option<ModelRole>) -> Vec<EnhanceModelInfo> {
        catalog::catalog()
            .iter()
            .filter(|m| role.is_none_or(|r| m.supports(r)))
            .map(|m| self.describe(m))
            .collect()
    }

    /// Annotate one entry with the state the settings UI needs.
    ///
    /// Split out of [`Self::list_models`] so a model that is not in the catalog
    /// at all — one the user picked off disk — can be described the same way and
    /// rendered by the same card.
    pub fn describe(&self, model: &CatalogModel) -> EnhanceModelInfo {
        let loaded = self
            .client
            .as_ref()
            .and_then(|c| c.loaded_model())
            .unwrap_or_default();
        let progress = self
            .downloads
            .lock()
            .ok()
            .and_then(|d| d.get(&model.id).map(|p| p.percentage));
        EnhanceModelInfo {
            model: model.clone(),
            downloaded: self.is_downloaded(model),
            loaded: self.model_path(model) == loaded,
            downloading: progress.is_some(),
            progress,
        }
    }

    /// Current runtime state.
    pub fn status(&self) -> EnhanceStatus {
        let loaded_model_id = self
            .client
            .as_ref()
            .and_then(|c| c.loaded_model())
            .map(|path| {
                catalog::catalog()
                    .iter()
                    .find(|m| self.model_path(m) == path)
                    .map(|m| m.id.clone())
                    // Not in the catalog means the user loaded it off disk, and
                    // the path is exactly what its synthesised id is built from.
                    .unwrap_or_else(|| format!("{}{}", catalog::LOCAL_PREFIX, path.display()))
            });
        EnhanceStatus {
            available: self.client.is_some(),
            running: self.client.as_ref().is_some_and(|c| c.is_running()),
            loaded_model_id,
            error: self.unavailable_reason.clone(),
        }
    }

    /// Whether a download for `model_id` is already running.
    pub fn is_downloading(&self, model_id: &str) -> bool {
        self.downloads
            .lock()
            .map(|d| d.contains_key(model_id))
            .unwrap_or(false)
    }

    /// Download `model` from Hugging Face into the models directory.
    ///
    /// Idempotent while a download is running: asking again is a no-op rather
    /// than a second transfer of the same file. Without this, a settings page
    /// that re-issued the request on mount could stack several gigabyte-scale
    /// downloads over each other, all writing the same path.
    pub async fn download(&self, model: &CatalogModel) -> Result<PathBuf> {
        let target = self.model_path(model);
        if self.is_downloaded(model) {
            return Ok(target);
        }

        // Claim the slot before any await, so two callers cannot both find it
        // free and both start fetching.
        {
            let mut in_flight = self
                .downloads
                .lock()
                .map_err(|_| anyhow!("download registry is poisoned"))?;
            if in_flight.contains_key(&model.id) {
                debug!(
                    "download of {} already in flight; joining it rather than restarting",
                    model.id
                );
                return Ok(target);
            }
            in_flight.insert(
                model.id.clone(),
                EnhanceDownloadProgress {
                    model_id: model.id.clone(),
                    downloaded: 0,
                    total: model.size_bytes,
                    percentage: 0.0,
                    state: DownloadState::Running,
                    error: None,
                },
            );
        }

        info!(
            "downloading enhancement model {} ({} MiB)",
            model.id,
            model.size_mb()
        );
        let result = self.fetch(model, &target).await;

        let update = match &result {
            Ok(()) => {
                info!("enhancement model {} ready", model.id);
                EnhanceDownloadProgress {
                    model_id: model.id.clone(),
                    downloaded: model.size_bytes,
                    total: model.size_bytes,
                    percentage: 100.0,
                    state: DownloadState::Done,
                    error: None,
                }
            }
            Err(e) => {
                warn!("download of {} failed: {e:#}", model.id);
                EnhanceDownloadProgress {
                    model_id: model.id.clone(),
                    downloaded: 0,
                    total: model.size_bytes,
                    percentage: 0.0,
                    state: DownloadState::Failed,
                    error: Some(format!("{e:#}")),
                }
            }
        };
        // Releases the slot as well as telling the frontend.
        publish(&self.app, &self.downloads, update);

        result.map(|()| target)
    }

    /// Fetch and verify the file. Split out so [`download`] can release its
    /// slot and report the outcome on every path, including early returns.
    async fn fetch(&self, model: &CatalogModel, target: &PathBuf) -> Result<()> {
        let api = hf_hub::api::tokio::Api::new().context("failed to create Hugging Face client")?;
        let repo = api.model(model.repo_id.clone());
        let reporter = DownloadReporter::new(
            self.app.clone(),
            model.id.clone(),
            Arc::clone(&self.downloads),
        );
        let downloaded = repo
            .download_with_progress(&model.filename, reporter)
            .await
            .with_context(|| {
                format!(
                    "failed to download {} from {}",
                    model.filename, model.repo_id
                )
            })?;

        // hf-hub caches under HF_HOME; copy into the app's directory so the
        // model survives a cache clear and is visible next to the others.
        std::fs::copy(&downloaded, target).with_context(|| {
            format!(
                "failed to copy {} to {}",
                downloaded.display(),
                target.display()
            )
        })?;

        let size = std::fs::metadata(target).map(|m| m.len()).unwrap_or(0);
        if size != model.size_bytes {
            // A wrong size means a truncated or substituted file; refuse it
            // rather than loading something unexpected.
            let _ = std::fs::remove_file(target);
            return Err(anyhow!(
                "downloaded {} has {} bytes, expected {}",
                model.filename,
                size,
                model.size_bytes
            ));
        }
        Ok(())
    }

    /// Delete a downloaded model.
    pub fn delete(&self, model: &CatalogModel) -> Result<()> {
        let path = self.model_path(model);
        if self.client.as_ref().and_then(|c| c.loaded_model()) == Some(path.clone()) {
            if let Some(c) = &self.client {
                c.unload_model();
            }
        }
        if path.exists() {
            std::fs::remove_file(&path)
                .with_context(|| format!("failed to delete {}", path.display()))?;
        }
        Ok(())
    }

    /// Make `model` the resident one, starting the sidecar if needed.
    pub fn load(&self, model: &CatalogModel, use_gpu: bool) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow!("enhancement engine is not available in this build"))?;
        if !self.is_downloaded(model) {
            return Err(anyhow!("{} has not been downloaded", model.name));
        }
        // `None` lets the sidecar offload everything if its build supports it;
        // `Some(0)` pins it to CPU.
        let gpu_layers = if use_gpu { None } else { Some(0) };
        client.load_model(
            &self.model_path(model),
            model.reasoning,
            model.prompt_style,
            gpu_layers,
        )
    }

    /// Release the model's memory but leave the process alive.
    pub fn unload(&self) {
        if let Some(c) = &self.client {
            c.unload_model();
        }
    }

    /// Stop the sidecar entirely.
    pub fn shutdown(&self) {
        if let Some(c) = &self.client {
            c.shutdown();
        }
    }

    /// Run an enhancement pass, falling back to the raw transcript on failure.
    pub fn enhance(&self, transcript: &str, opts: &EnhanceOptions, ctx: &AppContext) -> Enhanced {
        match &self.client {
            Some(client) => enhance::enhance(client.as_ref(), transcript, opts, ctx),
            None => Enhanced {
                text: transcript.to_string(),
                original: transcript.to_string(),
                rejected: None,
                verdict: None,
                error: self.unavailable_reason.clone(),
                elapsed: std::time::Duration::ZERO,
            },
        }
    }
}

/// Resolve the model a setting refers to, falling back to the catalog default.
///
/// Returns an owned entry rather than a `&'static` one because a local file has
/// no entry in the embedded catalog to borrow from; it is synthesised on demand.
/// The clone is a handful of short strings and happens when a model is chosen,
/// not per request.
pub fn resolve_model(id: Option<&str>, role: ModelRole) -> Option<CatalogModel> {
    if let Some(id) = id {
        if let Some(path) = id.strip_prefix(catalog::LOCAL_PREFIX) {
            // A path that has gone missing falls through to the catalog default
            // rather than leaving the user with no working model.
            if let Some(m) = CatalogModel::from_local(std::path::Path::new(path)) {
                return Some(m);
            }
        } else if let Some(m) = catalog::find(id) {
            return Some(m.clone());
        }
    }
    match role {
        ModelRole::Editor => catalog::default_editor().cloned(),
        ModelRole::Verifier => catalog::default_verifier().cloned(),
    }
}

/// Resolve a model and apply the user's prompt-style override, if any.
///
/// The override only ever applies to a model loaded off disk. A catalog entry
/// is curated and states its own style, so letting a stale setting rewrite it
/// would break a stock model for a reason the user could not see.
pub fn resolve_model_with(
    id: Option<&str>,
    role: ModelRole,
    override_style: Option<PromptStyle>,
) -> Option<CatalogModel> {
    let mut model = resolve_model(id, role)?;
    if let Some(style) = override_style {
        if model.is_local() {
            model.prompt_style = style;
        }
    }
    Some(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_falls_back_to_the_role_default() {
        assert_eq!(
            resolve_model(None, ModelRole::Editor).map(|m| m.id),
            Some("magicnothief/handy-editor-350m-q8".to_string())
        );
        assert_eq!(
            resolve_model(None, ModelRole::Verifier).map(|m| m.id),
            Some("qwen/qwen3-1.7b".to_string())
        );
    }

    #[test]
    fn resolve_ignores_an_unknown_id() {
        // A settings file naming a model we no longer ship must not disable the
        // feature; falling back keeps it working across catalog changes.
        assert_eq!(
            resolve_model(Some("deleted/model"), ModelRole::Editor).map(|m| m.id),
            Some("magicnothief/handy-editor-350m-q8".to_string())
        );
    }

    #[test]
    fn resolve_honours_an_explicit_choice() {
        assert_eq!(
            resolve_model(Some("liquidai/lfm2.5-350m"), ModelRole::Editor).map(|m| m.id),
            Some("liquidai/lfm2.5-350m".to_string())
        );
    }

    /// Write a stand-in GGUF and hand back the id that selects it.
    fn local_id(dir: &tempfile::TempDir, name: &str) -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, vec![0u8; 2048]).expect("write stub model");
        format!("{}{}", catalog::LOCAL_PREFIX, path.display())
    }

    #[test]
    fn resolve_accepts_a_file_off_disk() {
        let dir = tempfile::tempdir().expect("temp dir");
        let id = local_id(&dir, "checkpoint-10000.Q4_K_M.gguf");
        let m = resolve_model(Some(&id), ModelRole::Editor).expect("local file resolves");
        assert_eq!(m.id, id);
        assert_eq!(m.prompt_style, PromptStyle::Tuned);
    }

    #[test]
    fn resolve_falls_back_when_the_local_file_is_gone() {
        // Deleting or moving the file must not leave the user with no editor.
        let id = format!("{}/nowhere/gone.gguf", catalog::LOCAL_PREFIX);
        assert_eq!(
            resolve_model(Some(&id), ModelRole::Editor).map(|m| m.id),
            Some("magicnothief/handy-editor-350m-q8".to_string())
        );
    }

    #[test]
    fn the_prompt_style_override_only_touches_a_local_model() {
        let dir = tempfile::tempdir().expect("temp dir");
        let id = local_id(&dir, "mine.gguf");
        let m = resolve_model_with(Some(&id), ModelRole::Editor, Some(PromptStyle::Instructed))
            .expect("local file resolves");
        assert_eq!(m.prompt_style, PromptStyle::Instructed);

        // A curated entry states its own style; a stale setting must not be able
        // to break a stock model in a way the user cannot see.
        let stock = resolve_model_with(
            Some("liquidai/lfm2.5-350m"),
            ModelRole::Editor,
            Some(PromptStyle::Tuned),
        )
        .expect("catalog entry resolves");
        assert_eq!(stock.prompt_style, PromptStyle::Instructed);
    }
}

#[cfg(test)]
mod progress_tests {
    #[test]
    fn percentage_is_derived_from_the_byte_counts() {
        // The UI shows this directly, so an off value is visible to the user.
        let cases = [(0u64, 100u64, 0.0), (50, 100, 50.0), (100, 100, 100.0)];
        for (done, total, want) in cases {
            let pct = if total > 0 {
                (done as f64 / total as f64) * 100.0
            } else {
                0.0
            };
            assert!((pct - want).abs() < f64::EPSILON, "{done}/{total}");
        }
    }

    #[test]
    fn unknown_total_reports_zero_rather_than_dividing_by_it() {
        let total = 0u64;
        let pct = if total > 0 { 100.0 / total as f64 } else { 0.0 };
        assert_eq!(pct, 0.0);
    }
}
