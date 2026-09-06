//! Tauri commands for the local enhancement layer.

use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager};

use crate::enhance::catalog::{self, ModelRole};
use crate::managers::enhance::{
    resolve_model_with, EnhanceManager, EnhanceModelInfo, EnhanceStatus,
};
use crate::settings::{get_settings, write_settings};

/// Persist settings and tell the frontend they changed.
///
/// `write_settings` only writes. The store the UI reads from is refreshed by a
/// `settings-changed` event, which every other settings-mutating area of the
/// app emits by hand -- and which nothing here did. The visible symptom was the
/// enable toggle: the backend flipped and persisted, the frontend never learned
/// its copy was stale, and the switch sprang back under the user's finger.
///
/// Call this *before* any model load. Several commands below load a model after
/// writing, which takes seconds; emitting first means the UI settles
/// immediately instead of after the load.
fn write_settings_and_notify(app: &AppHandle, settings: crate::settings::AppSettings) {
    write_settings(app, settings);
    if let Err(e) = app.emit("settings-changed", serde_json::json!({})) {
        // Losing the event only costs a stale page until the next refresh, so
        // it is not worth failing a command that already persisted.
        log::warn!("could not announce an enhancement settings change: {e}");
    }
}

fn manager(app: &AppHandle) -> Result<Arc<EnhanceManager>, String> {
    app.try_state::<Arc<EnhanceManager>>()
        .map(|s| s.inner().clone())
        .ok_or_else(|| "Enhancement layer is not initialised".to_string())
}

/// Run a blocking manager call off the UI thread.
///
/// A synchronous `#[tauri::command]` runs on the main thread, so anything that
/// loads a model froze the entire window for as long as the load took — around
/// 33 seconds for a 1.6 GB model, which reads as a hung app. The default editor
/// is far smaller than that now, but the catalog still offers models that big.
/// Every command below that can reach the sidecar goes through here.
async fn off_thread<T, F>(work: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String> + Send + 'static,
    T: Send + 'static,
{
    tauri::async_runtime::spawn_blocking(work)
        .await
        .map_err(|e| format!("enhancement task did not finish: {e}"))?
}

/// List enhancement models, optionally narrowed to one role.
///
/// A model the user picked off disk is appended to the catalog listing. It has
/// to appear in the same list as everything else or the selector would show no
/// card as active while the layer was quietly running the user's own weights.
#[tauri::command]
#[specta::specta]
pub fn enhance_list_models(
    app: AppHandle,
    role: Option<ModelRole>,
) -> Result<Vec<EnhanceModelInfo>, String> {
    let mgr = manager(&app)?;
    let mut models = mgr.list_models(role);
    if role.is_none_or(|r| r == ModelRole::Editor) {
        let settings = get_settings(&app);
        if let Some(local) = settings
            .enhance_model_id
            .as_deref()
            .and_then(|id| id.strip_prefix(catalog::LOCAL_PREFIX))
            .and_then(|path| catalog::CatalogModel::from_local(std::path::Path::new(path)))
        {
            let style = settings.enhance_prompt_style.unwrap_or(local.prompt_style);
            models.push(mgr.describe(&catalog::CatalogModel {
                prompt_style: style,
                ..local
            }));
        }
    }
    Ok(models)
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
    let mgr = manager(&app)?;
    // Deleting the target of a running transfer would leave the download
    // writing to a path nobody expects to exist.
    if mgr.is_downloading(&model_id) {
        return Err(format!("{} is still downloading", model.name));
    }
    mgr.delete(model).map_err(|e| format!("{e:#}"))
}

/// Load the configured editing model into the sidecar.
#[tauri::command]
#[specta::specta]
pub async fn enhance_load_model(app: AppHandle) -> Result<(), String> {
    let settings = get_settings(&app);
    let model = resolve_model_with(
        settings.enhance_model_id.as_deref(),
        ModelRole::Editor,
        settings.enhance_prompt_style,
    )
    .ok_or_else(|| "No enhancement model available".to_string())?;
    let mgr = manager(&app)?;
    let use_gpu = settings.enhance_use_gpu;
    off_thread(move || mgr.load(&model, use_gpu).map_err(|e| format!("{e:#}"))).await
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
pub async fn enhance_preview(app: AppHandle, transcript: String) -> Result<EnhancePreview, String> {
    let settings = get_settings(&app);
    let mgr = manager(&app)?;

    // Loading on demand keeps the preview usable before the layer is enabled.
    let model = resolve_model_with(
        settings.enhance_model_id.as_deref(),
        ModelRole::Editor,
        settings.enhance_prompt_style,
    )
    .ok_or_else(|| "No enhancement model available".to_string())?;
    if !mgr.is_downloaded(&model) {
        // Better than letting `load` fail deep inside the sidecar: the caller
        // can point at the download button instead of showing a raw error.
        return Err(format!("{} has not been downloaded yet", model.name));
    }

    let use_gpu = settings.enhance_use_gpu;
    let opts = settings.enhance_options.clone();
    off_thread(move || {
        mgr.load(&model, use_gpu).map_err(|e| format!("{e:#}"))?;
        let out = mgr.enhance(&transcript, &opts, &crate::enhance::AppContext::default());
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
    })
    .await
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
pub async fn enhance_set_enabled(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.enhance_enabled = enabled;
    let keep_loaded = settings.enhance_keep_loaded;
    let use_gpu = settings.enhance_use_gpu;
    let model_id = settings.enhance_model_id.clone();
    let prompt_style = settings.enhance_prompt_style;
    write_settings_and_notify(&app, settings);

    let mgr = manager(&app)?;
    if !enabled {
        mgr.unload();
        return Ok(());
    }
    if !keep_loaded {
        return Ok(());
    }
    let Some(model) = resolve_model_with(model_id.as_deref(), ModelRole::Editor, prompt_style)
    else {
        return Ok(());
    };
    if !mgr.is_downloaded(&model) {
        // Nothing to preload yet; the toggle still takes effect.
        return Ok(());
    }
    // A failure here is not fatal: the model loads lazily on the first
    // dictation instead, so the toggle must still report success.
    off_thread(move || {
        if let Err(e) = mgr.load(&model, use_gpu) {
            log::warn!("could not preload enhancement model: {e:#}");
        }
        Ok(())
    })
    .await
}

/// Choose the editing model.
///
/// Each enhancement setting gets its own command because the frontend store
/// dispatches per key; a setting with no command silently updates the UI and
/// never reaches disk.
#[tauri::command]
#[specta::specta]
pub async fn enhance_set_model(app: AppHandle, model_id: Option<String>) -> Result<(), String> {
    // A missing file would otherwise fall back to the catalog default while the
    // setting still named the file, so the user would see the wrong card marked
    // active and no explanation.
    if let Some(path) = model_id
        .as_deref()
        .and_then(|id| id.strip_prefix(catalog::LOCAL_PREFIX))
    {
        if catalog::CatalogModel::from_local(std::path::Path::new(path)).is_none() {
            return Err(format!("Cannot read {path}"));
        }
    }
    let mut settings = get_settings(&app);
    settings.enhance_model_id = model_id.clone();
    let prompt_style = settings.enhance_prompt_style;
    let use_gpu = settings.enhance_use_gpu;
    let keep_loaded = settings.enhance_keep_loaded;
    let enabled = settings.enhance_enabled;
    write_settings_and_notify(&app, settings);

    // Swap the resident model so the next dictation uses the new choice
    // without waiting for a restart.
    let mgr = manager(&app)?;
    let model = resolve_model_with(model_id.as_deref(), ModelRole::Editor, prompt_style);
    let should_load = enabled && keep_loaded;
    match model.filter(|m| should_load && mgr.is_downloaded(m)) {
        Some(model) => {
            off_thread(move || {
                if let Err(e) = mgr.load(&model, use_gpu) {
                    // Not fatal: it loads lazily on the next dictation instead.
                    log::warn!("could not switch enhancement model: {e:#}");
                }
                Ok(())
            })
            .await
        }
        None => {
            // Whatever is resident is no longer the chosen model, so drop it
            // rather than enhancing with something the user did not pick.
            mgr.unload();
            Ok(())
        }
    }
}

/// Override how a model loaded off disk is prompted.
///
/// `None` trusts the model's own declared style, which for a local file means
/// "assume it was fine-tuned for this" — the only reason to point the layer at
/// your own GGUF. Set it explicitly when that guess is wrong.
#[tauri::command]
#[specta::specta]
pub async fn enhance_set_prompt_style(
    app: AppHandle,
    prompt_style: Option<crate::enhance::PromptStyle>,
) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.enhance_prompt_style = prompt_style;
    let model_id = settings.enhance_model_id.clone();
    let use_gpu = settings.enhance_use_gpu;
    let keep_loaded = settings.enhance_keep_loaded;
    let enabled = settings.enhance_enabled;
    write_settings_and_notify(&app, settings);

    // The style is baked in at load time, so a resident model keeps using the
    // old one until it is reloaded.
    let mgr = manager(&app)?;
    mgr.unload();
    let should_load = enabled && keep_loaded;
    let Some(model) = resolve_model_with(model_id.as_deref(), ModelRole::Editor, prompt_style)
        .filter(|m| should_load && mgr.is_downloaded(m))
    else {
        return Ok(());
    };
    off_thread(move || {
        if let Err(e) = mgr.load(&model, use_gpu) {
            log::warn!("could not reload enhancement model: {e:#}");
        }
        Ok(())
    })
    .await
}

/// Choose the verifying model.
#[tauri::command]
#[specta::specta]
pub fn enhance_set_verifier_model(app: AppHandle, model_id: Option<String>) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.enhance_verifier_model_id = model_id;
    write_settings_and_notify(&app, settings);
    Ok(())
}

/// Replace the set of enabled enhancement behaviours.
#[tauri::command]
#[specta::specta]
pub fn enhance_set_options(
    app: AppHandle,
    options: crate::enhance::EnhanceOptions,
) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.enhance_options = options;
    write_settings_and_notify(&app, settings);
    Ok(())
}

/// Turn GPU offload on or off, reloading so it takes effect immediately.
#[tauri::command]
#[specta::specta]
pub async fn enhance_set_use_gpu(app: AppHandle, use_gpu: bool) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.enhance_use_gpu = use_gpu;
    let model_id = settings.enhance_model_id.clone();
    let prompt_style = settings.enhance_prompt_style;
    let enabled = settings.enhance_enabled;
    let keep_loaded = settings.enhance_keep_loaded;
    write_settings_and_notify(&app, settings);

    let mgr = manager(&app)?;
    // The backend is chosen at load time, so an already-resident model keeps
    // running on the old one until it is reloaded.
    mgr.unload();
    let should_load = enabled && keep_loaded;
    let Some(model) = resolve_model_with(model_id.as_deref(), ModelRole::Editor, prompt_style)
        .filter(|m| should_load && mgr.is_downloaded(m))
    else {
        return Ok(());
    };
    off_thread(move || {
        if let Err(e) = mgr.load(&model, use_gpu) {
            log::warn!("could not reload enhancement model: {e:#}");
        }
        Ok(())
    })
    .await
}

/// Choose whether the model stays resident between dictations.
#[tauri::command]
#[specta::specta]
pub fn enhance_set_keep_loaded(app: AppHandle, keep_loaded: bool) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.enhance_keep_loaded = keep_loaded;
    write_settings_and_notify(&app, settings);
    if !keep_loaded {
        manager(&app)?.unload();
    }
    Ok(())
}
