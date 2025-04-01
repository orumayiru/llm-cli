// src/llm/huggingface.rs

use crate::config::Config;
use crate::error::Result;
use anyhow::{anyhow, Context};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, instrument};

// --- Request Structs ---

#[derive(Serialize, Debug)]
struct HuggingFaceRequest {
    inputs: String,
    // Add other parameters if needed (temperature, max_length, etc.)
}

// --- Response Structs ---

#[derive(Deserialize, Debug)]
struct HuggingFaceResponse {
    // Adapt this to the actual response structure
    generated_text: Option<String>,
    error: Option<String>,
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
    prompt: &str,
) -> Result<String> {
    let api_key = config.huggingface_api_key.as_deref().ok_or_else(|| anyhow!("HUGGINGFACE_API_KEY is not set."))?;
    let model_name = &config.default_huggingface_model;
    let url = format!("https://api-inference.huggingface.co/models/{}", model_name);

    let request_payload = HuggingFaceRequest {
        inputs: prompt.to_string(),
    };

    debug!(?url, ?request_payload, "Sending generate request to Hugging Face API");

    let response = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&request_payload)
        .send()
        .await
        .context("Failed to send generate request to Hugging Face API")?;

    // Use the helper function to handle the response
    let huggingface_response: HuggingFaceResponse = handle_api_response(response, &url, "Hugging Face generate").await?;

    if let Some(err) = huggingface_response.error {
        return Err(anyhow!("Hugging Face API Error: {}", err));
    }

    huggingface_response.generated_text.ok_or_else(|| anyhow!("No generated text in Hugging Face response"))
}

// --- List Models Function ---
#[instrument(skip(_client, _config))]
pub async fn list_models(_client: &Client, _config: &Config) -> Result<Vec<String>> {
    // TODO: Implement Hugging Face model listing if possible
    // This might require a different API endpoint or a different approach
    // For now, we'll return an empty list
    Ok(vec![])
}

// --- Check Connection Function ---
#[instrument(skip(client, config))]
pub async fn check_connection(client: &Client, config: &Config) -> Result<()> {
    // TODO: Implement a proper connection check for Hugging Face
    // For now, we'll just try to generate something
    debug!("Checking Hugging Face connection status...");
    generate(client, config, "test").await?;
    debug!("Hugging Face connection check successful.");
    Ok(())
}
