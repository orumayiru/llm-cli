// src/llm/ollama.rs

use crate::config::Config;
use crate::error::Result;
use anyhow::{anyhow, Context};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, instrument};

// --- Request Structs ---

#[derive(Serialize, Debug)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool, // We want the full response at once for this simple REPL
}

// --- Response Structs ---

#[derive(Deserialize, Debug)]
struct OllamaResponse {
    model: String,
    created_at: String,
    response: String,
    done: bool,
    // Includes timings and context if needed
    // total_duration: Option<u64>,
    // prompt_eval_count: Option<u32>,
    // eval_count: Option<u32>,
}

// --- Model Listing Structs ---

#[derive(Deserialize, Debug)]
struct OllamaTag {
    name: String,
    modified_at: String,
    size: u64,
    // Add digest, details if needed
}

#[derive(Deserialize, Debug)]
struct OllamaTagsResponse {
    models: Vec<OllamaTag>,
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

// --- Generate Function ---
#[instrument(skip(client, config, prompt))]
pub async fn generate(
    client: &Client,
    config: &Config,
    model: Option<&str>, // Allow overriding default model
    prompt: &str,
) -> Result<String> {
    let target_model = model.unwrap_or(&config.default_ollama_model);
    let url = format!("{}/api/generate", config.ollama_base_url);

    let request_payload = OllamaRequest {
        model: target_model.to_string(),
        prompt: prompt.to_string(),
        stream: false,
    };

    debug!(?request_payload, "Sending generate request to Ollama");

    let response = client
        .post(&url)
        .json(&request_payload)
        .send()
        .await
        .context(format!("Failed to send generate request to Ollama at {}", url))?;

    // Use the helper function to handle the response
    let ollama_response: OllamaResponse = handle_api_response(response, &url, "Ollama generate").await?;

    Ok(ollama_response.response)
}

// --- List Models Function ---
#[instrument(skip(client, config))]
pub async fn list_models(client: &Client, config: &Config) -> Result<Vec<String>> {
    let url = format!("{}/api/tags", config.ollama_base_url);
    debug!("Fetching models list from Ollama: {}", url);

    let response = client
        .get(&url)
        .send()
        .await
        .context(format!("Failed to send list models request to Ollama at {}", url))?;

    // Use the helper function to handle the response
    let tags_response: OllamaTagsResponse = handle_api_response(response, &url, "Ollama list models").await?;

    let model_names = tags_response
        .models
        .into_iter()
        .map(|tag| tag.name)
        .collect();

    debug!("Found Ollama models: {:?}", model_names);
    Ok(model_names)
}

// --- Check Connection Function ---
#[instrument(skip(client, config))]
pub async fn check_connection(client: &Client, config: &Config) -> Result<()> {
    let url = &config.ollama_base_url;
    debug!("Checking Ollama connection status at {} ...", url);
    let response = client
        .get(url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .context(format!("Failed to connect to Ollama at {}", url))?;

    if response.status().is_success() {
        debug!("Ollama connection check successful (Status: {}).", response.status());
        Ok(())
    } else {
        let status = response.status();
        let error_body = response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
        error!("Ollama connection check failed. Status: {}, Body: {:.100}", status, error_body);
        Err(anyhow!("Ollama connection check failed at {}: Status {} - {}", url, status, error_body))
    }
}
