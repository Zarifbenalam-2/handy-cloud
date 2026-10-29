//! Cloud transcription engines.
//!
//! Selected via `settings.cloud_transcription`. When set, dictation audio is
//! sent to the configured provider instead of the bundled local models; the
//! rest of the pipeline (overlay, paste, history) is unchanged.
//!
//! Providers:
//! - `Gemini` — official Generative Language API: WAV blob inline in a
//!   `generateContent` request, model echoes back the transcript. Needs a
//!   free AI Studio key.
//! - `Soniox` — real-time websocket (`wss://stt-rt.soniox.com/transcribe-websocket`,
//!   model `stt-rt-v5`) with your own Soniox API key (paid service).
//! - `Perplexity` — zero-setup mode: mint a short-lived Soniox key from
//!   Perplexity's web dictation endpoint exactly like their own website does,
//!   then stream over the Soniox websocket. No account, no key, no cost.

use anyhow::{anyhow, Context, Result};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use specta::Type;

use crate::audio_toolkit::WHISPER_SAMPLE_RATE;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum CloudProvider {
    /// Official Google Generative Language API (needs a free AI Studio key).
    Gemini,
    /// Soniox websocket with a key you paste in (paid service).
    Soniox,
    /// Zero-setup mode: mint a short-lived Soniox key from Perplexity's web
    /// endpoint, then stream over the Soniox websocket.
    Perplexity,
}

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct CloudTranscriptionSettings {
    pub provider: CloudProvider,
    #[serde(default)]
    pub gemini_api_key: String,
    #[serde(default = "default_gemini_model")]
    pub gemini_model: String,
    #[serde(default)]
    pub soniox_api_key: String,
}

fn default_gemini_model() -> String {
    "gemini-2.5-flash".to_string()
}

const GEMINI_PROMPT: &str = "Transcribe this audio recording verbatim in the language it is spoken. \
Output ONLY the transcript text with natural punctuation and capitalization. \
No commentary, no quotes, no markdown, no speaker labels.";

pub async fn transcribe(samples: Vec<f32>, cloud: &CloudTranscriptionSettings) -> Result<String> {
    match cloud.provider {
        CloudProvider::Gemini => transcribe_gemini(samples, cloud).await,
        CloudProvider::Soniox => {
            let key = cloud.soniox_api_key.trim().to_string();
            soniox_stream(key, &samples).await
        }
        CloudProvider::Perplexity => {
            let key = ensure_perplexity_key().await?;
            soniox_stream(key, &samples).await
        }
    }
}

/// Encode mono f32 samples at the recording pipeline's rate as a 16-bit PCM WAV.
fn samples_to_wav_bytes(samples: &[f32]) -> Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: WHISPER_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(&mut cursor, spec)
        .context("failed to create WAV writer for cloud transcription")?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer
            .write_sample(v)
            .context("failed to write WAV sample")?;
    }
    writer
        .finalize()
        .context("failed to finalize cloud WAV buffer")?;
    Ok(cursor.into_inner())
}

// ---------------------------------------------------------------------------
// Gemini (official Generative Language API)
// ---------------------------------------------------------------------------

async fn transcribe_gemini(
    samples: Vec<f32>,
    cloud: &CloudTranscriptionSettings,
) -> Result<String> {
    let key = cloud.gemini_api_key.trim();
    if key.is_empty() {
        return Err(anyhow!(
            "Gemini cloud engine selected but no API key is configured \
             (settings key: cloud_transcription.gemini_api_key)"
        ));
    }
    let model = {
        let m = cloud.gemini_model.trim();
        if m.is_empty() {
            default_gemini_model()
        } else {
            m.to_string()
        }
    };

    let wav = samples_to_wav_bytes(&samples)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);

    let body = serde_json::json!({
        "contents": [{
            "parts": [
                { "text": GEMINI_PROMPT },
                { "inline_data": { "mime_type": "audio/wav", "data": b64 } }
            ]
        }],
        "generationConfig": { "temperature": 0.0 }
    });

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    );
    let client = reqwest::Client::new();
    let resp = client
        .post(url)
        .query(&[("key", key)])
        .json(&body)
        .send()
        .await
        .context("Gemini request failed")?;

    let status = resp.status();
    let json: serde_json::Value = resp
        .json()
        .await
        .context("Gemini returned a non-JSON response")?;
    if !status.is_success() {
        let detail = json.to_string();
        let detail: String = detail.chars().take(400).collect();
        return Err(anyhow!("Gemini API error {status}: {detail}"));
    }

    let text = json
        .pointer("/candidates/0/content/parts")
        .and_then(|p| p.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();

    let text = text.trim().to_string();
    if text.is_empty() {
        let reason = json
            .pointer("/candidates/0/finishReason")
            .and_then(|r| r.as_str())
            .unwrap_or("no content");
        return Err(anyhow!("Gemini returned no transcript (finishReason: {reason})"));
    }
    Ok(text)
}

// ---------------------------------------------------------------------------
// Perplexity key minting
// ---------------------------------------------------------------------------

/// A key minted from Perplexity's web dictation endpoint, plus when it dies.
#[derive(Clone)]
struct MintedKey {
    key: String,
    expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Cache the minted key process-wide. The key lives about an hour, so the
/// cache keeps our request volume at "one page visit" levels instead of one
/// mint per dictation.
static MINTED_KEY: once_cell::sync::Lazy<tokio::sync::Mutex<Option<MintedKey>>> =
    once_cell::sync::Lazy::new(|| tokio::sync::Mutex::new(None));

/// Keys are retired this long before their server-stated expiry.
const MINT_SAFETY_MARGIN: chrono::Duration = chrono::Duration::minutes(5);

const PERPLEXITY_MINT_URL: &str =
    "https://www.perplexity.ai/rest/realtime/v1/transcription/soniox-api-key";

async fn ensure_perplexity_key() -> Result<String> {
    {
        let cache = MINTED_KEY.lock().await;
        if let Some(minted) = cache.as_ref() {
            let still_valid = minted
                .expires_at
                .map(|exp| chrono::Utc::now() < exp - MINT_SAFETY_MARGIN)
                // No expiry info: treat as valid; the stream itself will
                // surface an auth error if the key was bad.
                .unwrap_or(true);
            if still_valid {
                return Ok(minted.key.clone());
            }
        }
    }

    let minted = mint_perplexity_key().await?;
    *MINTED_KEY.lock().await = Some(minted.clone());
    Ok(minted.key)
}

async fn mint_perplexity_key() -> Result<MintedKey> {
    let body = serde_json::json!({
        "source": "web",
        // The site sends an IANA zone; chrono cannot get one portably, and
        // the server only records it. A UTC offset string is accepted.
        "timezone": chrono::Local::now().offset().to_string(),
        "turn_detection_threshold": 0.7,
        "response_language": "en",
    });

    // Browser-parity headers: the endpoint sits behind Cloudflare, and bare
    // programmatic clients (curl with no headers) get challenged. A normal
    // client with browser-parity headers is served (verified HTTP 200).
    let client = reqwest::Client::new();
    let mut last_err: Option<String> = None;

    for attempt in 0..2 {
        let resp = client
            .post(PERPLEXITY_MINT_URL)
            .header(
                "user-agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            )
            .header("accept", "application/json")
            .header("accept-language", "en-US,en;q=0.9")
            .header("content-type", "application/json")
            .header("origin", "https://www.perplexity.ai")
            .header("referer", "https://www.perplexity.ai/")
            .json(&body)
            .send()
            .await;

        match resp {
            Ok(resp) => {
                let status = resp.status();
                match resp.json::<serde_json::Value>().await {
                    Ok(json) if status.is_success() => {
                        let key = json
                            .get("api_key")
                            .and_then(|k| k.as_str())
                            .map(|s| s.to_string());
                        let Some(key) = key else {
                            return Err(anyhow!(
                                "Perplexity key mint returned no api_key: {status}"
                            ));
                        };
                        let expires_at = json
                            .get("expires_at")
                            .and_then(|e| e.as_str())
                            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                            .map(|d| d.with_timezone(&chrono::Utc));
                        return Ok(MintedKey { key, expires_at });
                    }
                    Ok(_) => {
                        last_err = Some(format!("Perplexity key mint HTTP {status}"));
                    }
                    Err(e) => {
                        last_err =
                            Some(format!("Perplexity key mint bad response: {e}"));
                    }
                }
            }
            Err(e) => {
                last_err = Some(format!("Perplexity key mint request failed: {e}"));
            }
        }
        if attempt == 0 {
            tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        }
    }

    Err(anyhow!(
        "{last_err:?}. If Perplexity starts blocking programmatic clients, set \
         cloud_transcription.provider to \"soniox\" with your own key, or \
         \"gemini\" with a free AI Studio key."
    ))
}

// ---------------------------------------------------------------------------
// Soniox websocket (shared by the Soniox and Perplexity providers)
// ---------------------------------------------------------------------------

async fn soniox_stream(api_key: String, samples: &[f32]) -> Result<String> {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let key = api_key.trim();
    if key.is_empty() {
        return Err(anyhow!(
            "Soniox cloud engine selected but no API key is configured \
             (settings key: cloud_transcription.soniox_api_key)"
        ));
    }

    let (ws, _resp) = tokio_tungstenite::connect_async(
        "wss://stt-rt.soniox.com/transcribe-websocket",
    )
    .await
    .context("failed to connect to Soniox websocket")?;
    let (mut tx, mut rx) = ws.split();

    // Language hints are omitted so stt-rt-v5 auto-detects the spoken language.
    let config = serde_json::json!({
        "api_key": key,
        "model": "stt-rt-v5",
        "audio_format": "pcm_s16le",
        "sample_rate": WHISPER_SAMPLE_RATE,
        "num_channels": 1,
        "enable_endpoint_detection": true,
        "max_endpoint_delay_ms": 700,
    });
    tx.send(Message::Text(config.to_string().into()))
        .await
        .context("failed to send Soniox config")?;

    let pcm: Vec<u8> = samples
        .iter()
        .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())
        .collect();
    // ~200 ms of audio per frame at 16 kHz mono s16le.
    const CHUNK_BYTES: usize = 3200 * 2;
    for chunk in pcm.chunks(CHUNK_BYTES) {
        tx.send(Message::Binary(chunk.to_vec().into()))
            .await
            .context("failed to stream audio to Soniox")?;
    }
    // End-of-audio signal: an empty binary frame, mirroring the reference
    // client's send-an-empty-ArrayBuffer behavior before closing.
    tx.send(Message::Binary(Vec::new().into()))
        .await
        .context("failed to signal end of audio to Soniox")?;

    let mut final_text = String::new();
    let mut partial = String::new();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);

    loop {
        let msg = match tokio::time::timeout_at(deadline, rx.next()).await {
            Ok(Some(Ok(m))) => m,
            Ok(Some(Err(e))) => {
                if final_text.is_empty() {
                    return Err(anyhow!("Soniox websocket error: {e}"));
                }
                break;
            }
            Ok(None) => break, // server closed the stream
            Err(_) => break,   // collect what we have on timeout
        };
        match msg {
            Message::Text(t) => {
                let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) else {
                    continue;
                };
                if let Some(code) = v.get("error_code").and_then(|c| c.as_i64()) {
                    let msg = v
                        .get("error_message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown error");
                    if final_text.is_empty() {
                        return Err(anyhow!("Soniox error {code}: {msg}"));
                    }
                    break;
                }
                if let Some(tokens) = v.get("tokens").and_then(|t| t.as_array()) {
                    partial.clear();
                    for tok in tokens {
                        let text = tok.get("text").and_then(|t| t.as_str()).unwrap_or("");
                        if text == "<end>" || text == "<fin>" {
                            continue;
                        }
                        if tok.get("is_final").and_then(|f| f.as_bool()).unwrap_or(false) {
                            final_text.push_str(text);
                        } else {
                            partial.push_str(text);
                        }
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    let _ = tx.close().await;

    let out = format!("{final_text}{partial}");
    let out = out.trim_start().to_string();
    if out.is_empty() {
        return Err(anyhow!("Soniox returned no transcript"));
    }
    Ok(out)
}
