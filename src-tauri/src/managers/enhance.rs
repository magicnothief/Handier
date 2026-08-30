//! Owns the enhancement sidecar and the models it can load.
//!
//! Downloads reuse the `hf-hub` client the transcription models already use, so
//! enhancement models land in the same cache and honour the same `HF_HOME`.

use anyhow::{anyhow, Context, Result};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::{AppHandle, Manager};

use crate::enhance::catalog::{self, CatalogModel, ModelRole};
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

/// Manages the enhancement sidecar's lifetime and model files.
pub struct EnhanceManager {
    client: Option<Arc<SidecarClient>>,
    models_dir: PathBuf,
    /// Set when the sidecar binary could not be located at startup.
    unavailable_reason: Option<String>,
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
                    unavailable_reason: None,
                })
            }
            None => {
                warn!("enhancement sidecar binary not found; the layer will stay disabled");
                Ok(Self {
                    client: None,
                    models_dir,
                    unavailable_reason: Some(
                        "The enhancement engine was not found in this build.".to_string(),
                    ),
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

    /// The catalog, annotated with on-disk and loaded state.
    pub fn list_models(&self, role: Option<ModelRole>) -> Vec<EnhanceModelInfo> {
        let loaded = self
            .client
            .as_ref()
            .and_then(|c| c.loaded_model())
            .unwrap_or_default();
        catalog::catalog()
            .iter()
            .filter(|m| role.is_none_or(|r| m.supports(r)))
            .map(|m| EnhanceModelInfo {
                model: m.clone(),
                downloaded: self.is_downloaded(m),
                loaded: self.model_path(m) == loaded,
            })
            .collect()
    }

    /// Current runtime state.
    pub fn status(&self) -> EnhanceStatus {
        let loaded_model_id = self
            .client
            .as_ref()
            .and_then(|c| c.loaded_model())
            .and_then(|path| {
                catalog::catalog()
                    .iter()
                    .find(|m| self.model_path(m) == path)
                    .map(|m| m.id.clone())
            });
        EnhanceStatus {
            available: self.client.is_some(),
            running: self.client.as_ref().is_some_and(|c| c.is_running()),
            loaded_model_id,
            error: self.unavailable_reason.clone(),
        }
    }

    /// Download `model` from Hugging Face into the models directory.
    pub async fn download(&self, model: &CatalogModel) -> Result<PathBuf> {
        let target = self.model_path(model);
        if self.is_downloaded(model) {
            return Ok(target);
        }

        info!(
            "downloading enhancement model {} ({} MiB)",
            model.id,
            model.size_mb()
        );
        let api = hf_hub::api::tokio::Api::new().context("failed to create Hugging Face client")?;
        let repo = api.model(model.repo_id.clone());
        let downloaded = repo.get(&model.filename).await.with_context(|| {
            format!(
                "failed to download {} from {}",
                model.filename, model.repo_id
            )
        })?;

        // hf-hub caches under HF_HOME; copy into the app's directory so the
        // model survives a cache clear and is visible next to the others.
        std::fs::copy(&downloaded, &target).with_context(|| {
            format!(
                "failed to copy {} to {}",
                downloaded.display(),
                target.display()
            )
        })?;

        let size = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
        if size != model.size_bytes {
            // A wrong size means a truncated or substituted file; refuse it
            // rather than loading something unexpected.
            let _ = std::fs::remove_file(&target);
            return Err(anyhow!(
                "downloaded {} has {} bytes, expected {}",
                model.filename,
                size,
                model.size_bytes
            ));
        }
        info!("enhancement model {} ready", model.id);
        Ok(target)
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
        client.load_model(&self.model_path(model), model.reasoning, gpu_layers)
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
pub fn resolve_model(id: Option<&str>, role: ModelRole) -> Option<&'static CatalogModel> {
    if let Some(id) = id {
        if let Some(m) = catalog::find(id) {
            return Some(m);
        }
    }
    match role {
        ModelRole::Editor => catalog::default_editor(),
        ModelRole::Verifier => catalog::default_verifier(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_falls_back_to_the_role_default() {
        assert_eq!(
            resolve_model(None, ModelRole::Editor).map(|m| m.id.as_str()),
            Some("qwen/qwen3-0.6b")
        );
        assert_eq!(
            resolve_model(None, ModelRole::Verifier).map(|m| m.id.as_str()),
            Some("qwen/qwen3-1.7b")
        );
    }

    #[test]
    fn resolve_ignores_an_unknown_id() {
        // A settings file naming a model we no longer ship must not disable the
        // feature; falling back keeps it working across catalog changes.
        assert_eq!(
            resolve_model(Some("deleted/model"), ModelRole::Editor).map(|m| m.id.as_str()),
            Some("qwen/qwen3-0.6b")
        );
    }

    #[test]
    fn resolve_honours_an_explicit_choice() {
        assert_eq!(
            resolve_model(Some("liquidai/lfm2.5-350m"), ModelRole::Editor).map(|m| m.id.as_str()),
            Some("liquidai/lfm2.5-350m")
        );
    }
}
