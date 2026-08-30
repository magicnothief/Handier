//! Tauri commands for the local enhancement layer.

use std::sync::Arc;
use tauri::{AppHandle, Manager};

use crate::enhance::catalog::{self, ModelRole};
use crate::managers::enhance::{resolve_model, EnhanceManager, EnhanceModelInfo, EnhanceStatus};
use crate::settings::{get_settings, write_settings};

fn manager(app: &AppHandle) -> Result<Arc<EnhanceManager>, String> {
    app.try_state::<Arc<EnhanceManager>>()
        .map(|s| s.inner().clone())
        .ok_or_else(|| "Enhancement layer is not initialised".to_string())
}

/// List enhancement models, optionally narrowed to one role.
#[tauri::command]
#[specta::specta]
pub fn enhance_list_models(
    app: AppHandle,
    role: Option<ModelRole>,
) -> Result<Vec<EnhanceModelInfo>, String> {
    Ok(manager(&app)?.list_models(role))
}

/// Current state of the enhancement layer.
#[tauri::command]
#[specta::specta]
pub fn enhance_status(app: AppHandle) -> Result<EnhanceStatus, String> {
    Ok(manager(&app)?.status())
}

/// Download a catalog model.
#[tauri::command]
#[specta::specta]
pub async fn enhance_download_model(app: AppHandle, model_id: String) -> Result<(), String> {
    let model = catalog::find(&model_id).ok_or_else(|| format!("Unknown model: {model_id}"))?;
    let mgr = manager(&app)?;
    mgr.download(model).await.map_err(|e| format!("{e:#}"))?;
    Ok(())
}

/// Delete a downloaded model.
#[tauri::command]
#[specta::specta]
pub fn enhance_delete_model(app: AppHandle, model_id: String) -> Result<(), String> {
    let model = catalog::find(&model_id).ok_or_else(|| format!("Unknown model: {model_id}"))?;
    manager(&app)?.delete(model).map_err(|e| format!("{e:#}"))
}

/// Load the configured editing model into the sidecar.
#[tauri::command]
#[specta::specta]
pub fn enhance_load_model(app: AppHandle) -> Result<(), String> {
    let settings = get_settings(&app);
    let model = resolve_model(settings.enhance_model_id.as_deref(), ModelRole::Editor)
        .ok_or_else(|| "No enhancement model available".to_string())?;
    manager(&app)?
        .load(model, settings.enhance_use_gpu)
        .map_err(|e| format!("{e:#}"))
}

/// Release the model's memory without stopping the sidecar.
#[tauri::command]
#[specta::specta]
pub fn enhance_unload_model(app: AppHandle) -> Result<(), String> {
    manager(&app)?.unload();
    Ok(())
}

/// Run the enhancement layer over a caller-supplied transcript.
///
/// Exists so the settings UI can show the user what the layer does to their own
/// wording before they turn it on for real dictation.
#[tauri::command]
#[specta::specta]
pub fn enhance_preview(app: AppHandle, transcript: String) -> Result<EnhancePreview, String> {
    let settings = get_settings(&app);
    let mgr = manager(&app)?;

    // Loading on demand keeps the preview usable before the layer is enabled.
    let model = resolve_model(settings.enhance_model_id.as_deref(), ModelRole::Editor)
        .ok_or_else(|| "No enhancement model available".to_string())?;
    mgr.load(model, settings.enhance_use_gpu)
        .map_err(|e| format!("{e:#}"))?;

    let out = mgr.enhance(
        &transcript,
        &settings.enhance_options,
        &crate::enhance::AppContext::default(),
    );
    let changed = out.changed();
    Ok(EnhancePreview {
        text: out.text,
        original: out.original,
        changed,
        rejected: out.rejected.map(|r| format!("{r:?}")),
        verdict: out.verdict.map(|v| format!("{v:?}")),
        error: out.error,
        elapsed_ms: u64::try_from(out.elapsed.as_millis()).unwrap_or(u64::MAX),
    })
}

/// Result of a preview run, including why it fell back when it did.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct EnhancePreview {
    /// Text that would be pasted.
    pub text: String,
    /// The untouched transcript.
    pub original: String,
    /// Whether the layer changed anything.
    pub changed: bool,
    /// Set when a mechanical guard refused the rewrite.
    pub rejected: Option<String>,
    /// Set when the meaning check ran.
    pub verdict: Option<String>,
    /// Set when the model could not run.
    pub error: Option<String>,
    /// Wall-clock milliseconds spent.
    pub elapsed_ms: u64,
}

/// Turn the enhancement layer on or off, loading or releasing the model to match.
#[tauri::command]
#[specta::specta]
pub fn enhance_set_enabled(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.enhance_enabled = enabled;
    let keep_loaded = settings.enhance_keep_loaded;
    let use_gpu = settings.enhance_use_gpu;
    let model_id = settings.enhance_model_id.clone();
    write_settings(&app, settings);

    let mgr = manager(&app)?;
    if enabled {
        if keep_loaded {
            if let Some(model) = resolve_model(model_id.as_deref(), ModelRole::Editor) {
                // A failure here is not fatal: the model loads lazily on the
                // first dictation instead.
                if let Err(e) = mgr.load(model, use_gpu) {
                    log::warn!("could not preload enhancement model: {e:#}");
                }
            }
        }
    } else {
        mgr.unload();
    }
    Ok(())
}
