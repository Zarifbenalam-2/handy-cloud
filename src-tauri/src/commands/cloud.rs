use crate::cloud_engine::{CloudProvider, CloudTranscriptionSettings};
use crate::settings::{get_settings, write_settings};
use tauri::AppHandle;

/// Switch the cloud transcription engine from the settings screen.
/// `provider` `None`/`"off"` disables cloud mode and returns to local
/// models; the key arguments only matter for `gemini`/`soniox`.
#[tauri::command]
#[specta::specta]
pub fn set_cloud_transcription_setting(
    app: AppHandle,
    provider: Option<String>,
    gemini_api_key: Option<String>,
    soniox_api_key: Option<String>,
) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.cloud_transcription = match provider.as_deref() {
        None | Some("") | Some("off") => None,
        Some("perplexity") => Some(CloudTranscriptionSettings {
            provider: CloudProvider::Perplexity,
            gemini_api_key: gemini_api_key.unwrap_or_default(),
            gemini_model: crate::cloud_engine::default_gemini_model(),
            soniox_api_key: soniox_api_key.unwrap_or_default(),
        }),
        Some("gemini") => Some(CloudTranscriptionSettings {
            provider: CloudProvider::Gemini,
            gemini_api_key: gemini_api_key.unwrap_or_default(),
            gemini_model: crate::cloud_engine::default_gemini_model(),
            soniox_api_key: soniox_api_key.unwrap_or_default(),
        }),
        Some("soniox") => Some(CloudTranscriptionSettings {
            provider: CloudProvider::Soniox,
            gemini_api_key: gemini_api_key.unwrap_or_default(),
            gemini_model: crate::cloud_engine::default_gemini_model(),
            soniox_api_key: soniox_api_key.unwrap_or_default(),
        }),
        Some(other) => return Err(format!("unknown provider: {other}")),
    };
    write_settings(&app, settings);
    Ok(())
}
