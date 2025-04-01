// src/llm/groq.rs

use crate::config::Config;
use crate::error::Result;
use crate::llm::openai_compatible as common_client; // Use the shared client
use anyhow::{anyhow, Context};
use reqwest::Client;
use tracing::instrument;

// --- Generate Function (using common client) ---
#[instrument(skip(client, config, prompt))]
pub async fn generate(
    client: &Client,
    config: &Config,
    prompt: &str,
) -> Result<String> {
    let api_key = config.groq_api_key.as_deref()
        .ok_or_else(|| anyhow!("GROQ_API_KEY is not set. Use '/config' or set environment variable."))?;

    common_client::generate(
        client,
        api_key,
        &config.groq_api_base_url,
        &config.default_groq_model,
        prompt,
        // Pass other Groq-specific params here if needed in common_client::generate
    )
    .await.context("Groq API generate call failed")
}

// --- List Models Function (using common client) ---
#[instrument(skip(client, config))]
pub async fn list_models(client: &Client, config: &Config) -> Result<Vec<String>> {
    let api_key = config.groq_api_key.as_deref()
        .ok_or_else(|| anyhow!("GROQ_API_KEY is not set. Cannot list models."))?;

    common_client::list_models(client, api_key, &config.groq_api_base_url)
        .await.context("Groq API list models call failed")
}

// --- Check Connection Function (using common client) ---
#[instrument(skip(client, config))]
pub async fn check_connection(client: &Client, config: &Config) -> Result<()> {
    let api_key = config.groq_api_key.as_deref()
        .ok_or_else(|| anyhow!("GROQ_API_KEY is not set. Cannot check connection."))?;

    common_client::check_connection(client, api_key, &config.groq_api_base_url)
        .await.context("Groq API connection check failed")
}