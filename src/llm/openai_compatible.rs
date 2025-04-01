// src/llm/openai_compatible.rs

use crate::error::Result;
use anyhow::{anyhow, Context};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, instrument};

// --- Common Request Structures ---

#[derive(Serialize, Debug)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    // Add other common OpenAI params if needed (temperature, max_tokens, stream, etc.)
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub temperature: Option<f32>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub stream: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)] // Clone needed for potential retries or logging
pub struct ChatMessage {
    pub role: String, // "system", "user", "assistant"
    pub content: String,
}

// --- Common Response Structures ---

#[derive(Deserialize, Debug)]
pub struct ChatCompletionResponse {
    pub id: Option<String>,
    pub object: Option<String>,
    pub created: Option<u64>,
    pub model: Option<String>,
    pub choices: Vec<ChatChoice>,
    // pub usage: Option<CompletionUsage>, // Add if needed
    // Error structure can vary, sometimes it's top-level
    pub error: Option<ApiError>,
}

#[derive(Deserialize, Debug)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ResponseMessage,
    #[serde(rename = "finish_reason")]
    pub finish_reason: Option<String>, // e.g., "stop", "length"
}

#[derive(Deserialize, Debug)]
pub struct ResponseMessage {
    pub role: String, // "assistant"
    pub content: Option<String>, // Content can sometimes be null
}

// Reusable error structure (matching previous definition)
#[derive(Deserialize, Debug)]
pub struct ApiError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: Option<String>, // e.g., "invalid_request_error"
    pub param: Option<String>,
    pub code: Option<String>, // Often a string code like "invalid_api_key"
}

// --- Model Listing Structures ---

#[derive(Deserialize, Debug)]
pub struct ListModelsResponse {
    pub object: String, // Typically "list"
    pub data: Vec<ModelInfo>,
    // Error might appear here too
    pub error: Option<ApiError>,
}

#[derive(Deserialize, Debug)]
pub struct ModelInfo {
    pub id: String, // The model ID/name
    pub object: String, // Typically "model"
    pub created: Option<u64>,
    #[serde(rename = "owned_by")]
    pub owned_by: Option<String>,
}


// --- Shared HTTP Client Logic ---

fn build_headers(api_key: &str) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let mut auth_value =
        HeaderValue::from_str(&format!("Bearer {}", api_key)).context("Invalid API key format")?;
    auth_value.set_sensitive(true);
    headers.insert(AUTHORIZATION, auth_value);
    Ok(headers)
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

#[instrument(skip(client, api_key, base_url, prompt))]
pub async fn generate(
    client: &Client,
    api_key: &str,
    base_url: &str,
    model: &str,
    prompt: &str,
    // TODO: Pass temperature, max_tokens etc. if needed by provider
) -> Result<String> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let headers = build_headers(api_key)?;

    let request_payload = ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        // stream: Some(false), // Explicitly non-streaming if needed
    };

    debug!(?url, model, "Sending chat completion request"); // Don't log full payload by default

    let response = client
        .post(&url)
        .headers(headers)
        .json(&request_payload)
        .send()
        .await
        .context(format!("Failed to send request to {}", url))?;

    // Use the helper function to handle the response
    let parsed_response: ChatCompletionResponse = handle_api_response(response, &url, "chat completion").await?;

    // Check for API errors within the JSON body
    if let Some(api_error) = parsed_response.error {
        error!(?api_error, "API returned an error in the response body");
        return Err(anyhow!("API Error: {} (Type: {:?}, Code: {:?})", api_error.message, api_error.error_type, api_error.code));
    }

    // Extract text content
    let text_content = parsed_response.choices
        .first()
        .and_then(|choice| choice.message.content.as_deref())
        .ok_or_else(|| anyhow!("Failed to extract text content from response choices"))?;

    // Consider checking finish_reason if needed

    Ok(text_content.to_string())
}

#[instrument(skip(client, api_key, base_url))]
pub async fn list_models(
    client: &Client,
    api_key: &str,
    base_url: &str,
) -> Result<Vec<String>> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let headers = build_headers(api_key)?;

    debug!("Sending list models request to {}", url);

    let response = client
        .get(&url)
        .headers(headers)
        .send()
        .await
        .context(format!("Failed to send list models request to {}", url))?;

    // Use the helper function to handle the response
    let list_response: ListModelsResponse = handle_api_response(response, &url, "list models").await?;

    // Check for API errors within the JSON body
    if let Some(api_error) = list_response.error {
        error!(?api_error, "API returned an error listing models");
        return Err(anyhow!("API Error listing models: {}", api_error.message));
    }

    let model_ids = list_response.data.into_iter().map(|m| m.id).collect();
    Ok(model_ids)
}


// check_connection simply tries to list models
#[instrument(skip(client, api_key, base_url))]
pub async fn check_connection(
    client: &Client,
    api_key: &str,
    base_url: &str,
) -> Result<()> {
    debug!("Checking OpenAI-compatible connection status via list models...");
    // Use a shorter timeout specifically for the connection check if desired
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let headers = build_headers(api_key)?;
    let response = client
        .get(&url)
        .headers(headers)
        .timeout(Duration::from_secs(10)) // Add timeout for status check
        .send()
        .await
        .context(format!("Failed connection check request to {}", url))?;

    if response.status().is_success() {
        debug!("OpenAI-compatible connection check successful (Status: {}).", response.status());
        Ok(())
    } else {
        let status = response.status();
        let error_body = response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
        error!("OpenAI-compatible connection check failed. Status: {}, Body: {:.100}", status, error_body);
        Err(anyhow!("Connection check failed at {}: Status {} - {}", url, status, error_body))
    }

    // Alternative: Call list_models and ignore the result, but this doesn't allow a separate timeout easily
    // list_models(client, api_key, base_url).await?;
    // debug!("OpenAI-compatible connection check successful.");
    // Ok(())
}
