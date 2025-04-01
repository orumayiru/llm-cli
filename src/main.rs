// src/main.rs

mod cli;
mod config;
mod error;
mod llm;

use anyhow::Context;
use reqwest::Client;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use tracing::{info, error};

#[tokio::main]
async fn main() -> error::Result<()> {
    // --- Load .env file ---
    // Place this early, before loading config which reads env vars
    dotenvy::dotenv().ok(); // Ignore error if .env is not found
    // ---

    // Initialize tracing
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    info!("Starting LLM Chat CLI...");

    // Load configuration (now reads GEMINI_API_KEY from env)
    let mut config = config::Config::load().context("Failed to load configuration")?;

    // Create reqwest client
    let client = Client::new();

    // Start the interactive REPL mode
    if let Err(e) = cli::repl::run_interactive(&mut config, &client).await {
        error!("Application error: {:?}", e);
        std::process::exit(1);
    }

    Ok(())
}
