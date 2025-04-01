// src/llm/gemini.rs

use crate::config::Config;
use crate::error::Result;
use anyhow::{anyhow, Context};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, instrument, warn};

// --- Request Structs ---

#[derive(Serialize, Debug)]
struct GeminiRequest {
    contents: Vec<Content>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Serialize, Debug)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Part {
    text: String,
}

#[derive(Serialize, Debug, Default)]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

// --- Response Structs ---

#[derive(Deserialize, Debug)]
struct GeminiResponse {
    candidates: Option<Vec<Candidate>>,
    #[serde(rename = "promptFeedback")]
    prompt_feedback: Option<PromptFeedback>,
    error: Option<ApiError>,
}

#[derive(Deserialize, Debug)]
struct Candidate {
    content: Option<ContentResponse>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
    #[serde(rename = "safetyRatings")]
    safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Deserialize, Debug)]
struct ContentResponse {
    parts: Option<Vec<Part>>,
    role: Option<String>,
}

#[derive(Deserialize, Debug)]
struct PromptFeedback {
    #[serde(rename = "blockReason")]
    block_reason: Option<String>,
    #[serde(rename = "safetyRatings")]
    safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Deserialize, Debug)]
struct SafetyRating {
    category: String,
    probability: String,
}

#[derive(Deserialize, Debug)]
struct ApiError {
    code: u16,
    message: String,
    status: String,
}

#[derive(Deserialize, Debug)]
struct GeminiListModelsResponse {
    models: Option<Vec<GeminiModelInfo>>,
    error: Option<ApiError>,
}

#[derive(Deserialize, Debug)]
struct GeminiModelInfo {
    name: String,
    #[serde(rename = "displayName")]
    display_name: Option<String>,
    version: Option<String>,
    #[serde(rename = "supportedGenerationMethods")]
    supported_generation_methods: Option<Vec<String>>,
}

// --- Helper function to handle API responses ---
async fn handle_api_response<T: serde::de::DeserializeOwned + std::fmt::Debug>(
    response: reqwest::Response,
    _url: &str,
    operation_name: &str,
) -> Result<T> {
    let status = response.status();
    let response_bytes = response.bytes().await.context(format!("Failed to read {} response body", operation_name))?;

    match serde_json::from_slice::<T>(&response_bytes) {
        Ok(parsed_response) => {
            debug!(?parsed_response, "Successfully parsed {} response", operation_name);
            Ok(parsed_response)
        }
        Err(parse_error) => {
            let body_string = String::from_utf8_lossy(&response_bytes);
            error!(
                status = ?status,
                error = ?parse_error,
                response_body = ?body_string,
                "Failed to parse {} response", operation_name
            );
            let base_msg = if status.is_success() { format!("Failed to parse successful {} response", operation_name) } else { format!("API {} request failed", operation_name) };
            Err(anyhow!("{} (Status: {}): {}. Body: {}", base_msg, status, parse_error, body_string))
        }
    }
}

// --- generate function ---
#[instrument(skip(client, config, prompt))]
pub async fn generate(
    client: &Client,
    config: &Config,
    prompt: &str,
) -> Result<String> {
    let api_key = config.gemini_api_key.as_deref().ok_or_else(|| anyhow!("GEMINI_API_KEY is not set."))?;
    let model_name = &config.default_gemini_model;
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model_name, api_key
    );

    let mut gen_config = GenerationConfig::default();
    let mut config_set = false;
    if config.gemini_temperature.is_some() { gen_config.temperature = config.gemini_temperature; config_set = true; }
    if config.gemini_top_p.is_some() { gen_config.top_p = config.gemini_top_p; config_set = true; }
    if config.gemini_max_tokens.is_some() { gen_config.max_output_tokens = config.gemini_max_tokens; config_set = true; }

    let request_payload = GeminiRequest {
        contents: vec![Content { parts: vec![Part { text: prompt.to_string(), }], }],
        generation_config: if config_set { Some(gen_config) } else { None },
    };

    debug!(?url, ?request_payload, "Sending generate request to Gemini API");
    let response = client.post(&url).json(&request_payload).send().await.context("Failed to send generate request to Gemini API")?;

    // Use the helper function to handle the response
    let gemini_response: GeminiResponse = handle_api_response(response, &url, "Gemini generate").await?;

    // Handle top-level API errors first
    if let Some(api_error) = gemini_response.error {
        error!(?api_error, "Gemini API returned an error in the response body");
        return Err(anyhow!("Gemini API Error ({} {}): {}", api_error.code, api_error.status, api_error.message));
    }

    // Check prompt feedback for blocking
    if let Some(feedback) = &gemini_response.prompt_feedback {
        if let Some(reason) = &feedback.block_reason {
            error!("Gemini prompt blocked. Reason: {}", reason);
            let safety_details = feedback.safety_ratings.as_ref().map_or("".to_string(), |ratings| {
                format!(" Safety Ratings: {:?}", ratings)
            });
            return Err(anyhow!("Prompt blocked by Gemini due to '{}'.{}", reason, safety_details));
        }
    }

    // --- More Robust Candidate and Content Extraction ---
    let first_candidate = gemini_response.candidates.as_ref()
        .and_then(|c| c.first())
        .ok_or_else(|| {
            error!(?gemini_response, "No candidates found in Gemini response structure.");
            anyhow!("No candidates found in Gemini response")
        })?;

    let finish_reason = first_candidate.finish_reason.as_deref().unwrap_or("UNKNOWN");

    if finish_reason != "STOP" && finish_reason != "UNKNOWN" {
        let safety_info = if finish_reason == "SAFETY" {
            format!(" Safety Ratings: {:?}", first_candidate.safety_ratings)
        } else { "".to_string() };
        warn!("Gemini generation finished due to reason: {}{}", finish_reason, safety_info);
        return Err(anyhow!("Gemini generation finished early: Reason '{}'{}", finish_reason, safety_info));
    }
    if finish_reason == "UNKNOWN" {
        warn!("Gemini response candidate is missing a 'finishReason'. Proceeding cautiously.");
    }

    let content = first_candidate.content.as_ref()
        .ok_or_else(|| {
            error!(?first_candidate, "Candidate content is missing.");
            anyhow!("Candidate content is missing in Gemini response")
        })?;

    let parts = content.parts.as_ref()
        .ok_or_else(|| {
            error!(?content, "Content parts are missing.");
            anyhow!("Content 'parts' are missing in Gemini response")
        })?;

    let text = parts.first()
        .ok_or_else(|| {
            error!(?parts, "Content parts array is empty.");
            anyhow!("Content parts array is empty in Gemini response")
        })?
        .text.clone();

    Ok(text)
}

// --- list_models function ---
#[instrument(skip(client, config))]
pub async fn list_models(client: &Client, config: &Config) -> Result<Vec<String>> {
    let api_key = config.gemini_api_key.as_deref().ok_or_else(|| anyhow!("GEMINI_API_KEY is not set. Cannot list models."))?;
    let url = format!("https://generativelanguage.googleapis.com/v1beta/models?key={}", api_key);
    debug!("Sending list models request to Gemini API: {}", url);

    let response = client.get(&url).send().await.context("Failed to send list models request to Gemini API")?;

    // Use the helper function to handle the response
    let list_response: GeminiListModelsResponse = handle_api_response(response, &url, "Gemini list models").await?;

    // Handle top-level API errors first
    if let Some(api_error) = list_response.error {
        error!(?api_error, "Gemini API returned an error listing models");
        return Err(anyhow!("Gemini API Error listing models ({} {}): {}", api_error.code, api_error.status, api_error.message));
    }

    let models = list_response.models.unwrap_or_default();
    let chat_model_ids: Vec<String> = models.into_iter()
        .filter(|model| model.supported_generation_methods.as_deref().unwrap_or(&[]).contains(&"generateContent".to_string()))
        .filter_map(|model| model.name.strip_prefix("models/").map(String::from))
        .collect();
    if chat_model_ids.is_empty() {
        warn!("Gemini list models parsed successfully, but no models supporting 'generateContent' were found.");
    }
    Ok(chat_model_ids)
}

// --- check_connection function ---
#[instrument(skip(client, config))]
pub async fn check_connection(client: &Client, config: &Config) -> Result<()> {
    debug!("Checking Gemini connection status...");
    list_models(client, config).await?; // Relies on list_models working
    debug!("Gemini connection check successful.");
    Ok(())
}
