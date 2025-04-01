// src/cli/repl.rs

// --- Imports ---
use crate::cli::helper::ReplHelper;
use crate::config::{Config, LlmProvider};
use crate::error::Result;
use crate::llm::{gemini, groq, ollama,huggingface};
use anyhow::{anyhow, Context};
use reqwest::Client;
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use rustyline::Editor;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tracing::{debug, error, info, warn};

// --- Constants ---
const HISTORY_FILE: &str = "history.txt";
const PROMPT_FORMAT: &str = "{}:{}{}";
const UNKNOWN_COMMAND_MSG: &str = "Unknown command: '/{}'. Type '/help' for available commands.";
const SHELL_COMMAND_USAGE: &str = "Usage: !<shell_command>";
const APP_COMMAND_USAGE: &str = "Usage: /<command> [args...]";

// --- History File Helper ---
fn get_history_path() -> PathBuf {
    let mut path = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
    path.push("llm-chat");
    std::fs::create_dir_all(&path).ok();
    path.push(HISTORY_FILE);
    path
}

// --- Main REPL Function ---
pub async fn run_interactive(config: &mut Config, client: &Client) -> Result<()> {
    info!("Starting interactive LLM chat session.");

    // --- Setup Rustyline Editor ---
    let helper = ReplHelper::new();
    let mut rl = Editor::<ReplHelper, DefaultHistory>::new()?;
    rl.set_helper(Some(helper));
    let history_path = get_history_path();
    if let Err(e) = rl.load_history(&history_path) {
        warn!("Failed to load command history from {:?}: {}", history_path, e);
    }
    // --- End Setup ---

    // --- Print initial connection status ---
    print_initial_status(config);

    // --- Main Loop ---
    loop {
        // Display prompt with current provider/model
        let prompt_string = format_prompt(config);

        // Read user input
        let readline_result = rl.readline(&prompt_string);
        match readline_result {
            Ok(line) => {
                let input = line.trim();

                // Add non-empty lines to history
                if !input.is_empty() {
                    if let Err(e) = rl.add_history_entry(line.as_str()) {
                        warn!("Failed to add line to history: {}", e);
                    }
                }

                // Skip empty lines
                if input.is_empty() {
                    continue;
                }

                // Check for exit commands
                if is_exit_command(input) {
                    break;
                }

                // --- Dispatch Input ---
                if input.starts_with('!') {
                    // Handle Shell Command
                    if let Err(e) = handle_shell_command(input) {
                        error!("Shell command failed: {:?}", e);
                        eprintln!("Error executing shell command: {}", e);
                        eprintln!("---");
                    }
                } else if input.starts_with('/') {
                    // Handle App Command
                    if let Err(e) = handle_app_command(input, config, client).await {
                        error!("App command failed: {:?}", e);
                        eprintln!("Error executing app command: {}", e);
                        eprintln!("---");
                    }
                } else {
                    // Handle LLM Prompt
                    if let Err(e) = handle_llm_prompt(input, config, client).await {
                        error!("LLM prompt failed: {:?}", e);
                        eprintln!("Error generating LLM response: {}", e);
                        eprintln!("---");
                    }
                }
            }
            // Handle REPL specific errors
            Err(ReadlineError::Interrupted) => {
                println!("^C"); /* Continue loop */
            }
            Err(ReadlineError::Eof) => {
                println!("^D");
                break; /* Exit loop */
            }
            Err(err) => {
                error!("Readline error: {:?}", err);
                eprintln!("Input Error: {}", err);
                break; // Exit on other errors
            }
        }
    } // --- End Main Loop ---

    // --- Save history ---
    if let Err(e) = rl.save_history(&history_path) {
        error!("Failed to save command history to {:?}: {}", history_path, e);
    }

    println!("Exiting interactive session.");
    info!("Exiting interactive LLM chat session.");
    Ok(())
}

// --- Helper Functions ---

fn print_initial_status(config: &Config) {
    println!("LLM Chat CLI");
    println!("Default Provider: {}", config.active_provider);
    match config.active_provider {
        LlmProvider::Ollama => {
            println!("Ollama Endpoint: {}", config.ollama_base_url);
            println!("Ollama Model: {}", config.default_ollama_model);
        }
        LlmProvider::Gemini => {
            println!("Gemini Model: {}", config.default_gemini_model);
            if config.gemini_api_key.is_none() {
                println!("Warning: GEMINI_API_KEY not set.");
            }
        }
        LlmProvider::Groq => {
            println!("Groq Model: {}", config.default_groq_model);
            if config.groq_api_key.is_none() {
                println!("Warning: GROQ_API_KEY not set.");
            }
        }
        LlmProvider::HuggingFace => {
            println!("Hugging Face Model: {}", config.default_huggingface_model);
            if config.huggingface_api_key.is_none() { println!("Warning: HUGGINGFACE_API_KEY not set."); }
            }
    }
    println!("Type '/help' for commands, '!' followed by a shell command, or your prompt.");
    println!("---");
}

fn format_prompt(config: &Config) -> String {
    let prompt_model = config.get_active_model_name();
    let prompt_provider = &config.active_provider;
    format!("{}{}", format!("{}:{}", prompt_provider, prompt_model), "*")
}

fn is_exit_command(input: &str) -> bool {
    matches!(input, "quit" | "exit" | "/quit" | "/exit")
}

// --- Shell Command Handler ---
fn handle_shell_command(input: &str) -> Result<()> {
    let command_str = input[1..].trim();
    if command_str.is_empty() {
        eprintln!("{}", SHELL_COMMAND_USAGE);
        return Ok(());
    }
    println!("Executing: {}", command_str);
    println!("---");
    let output_result = if cfg!(target_os = "windows") {
        Command::new("cmd").args(["/C", command_str]).output()
    } else {
        Command::new("sh").arg("-c").arg(command_str).output()
    };
    match output_result {
        Ok(output) => {
            if !output.stdout.is_empty() {
                print!("{}", String::from_utf8_lossy(&output.stdout));
            }
            io::stdout().flush().context("Failed to flush stdout after command output")?;
            if !output.stderr.is_empty() {
                eprint!("{}", String::from_utf8_lossy(&output.stderr));
            }
            io::stderr().flush().context("Failed to flush stderr after command output")?;
            if !output.status.success() {
                eprintln!("\nCommand exited with status: {}", output.status);
            }
        }
        Err(e) => {
            error!("Failed to execute command '{}': {}", command_str, e);
            eprintln!("Error executing command: {}", e);
        }
    }
    println!("---");
    Ok(())
}

// --- LLM Prompt Handler ---
async fn handle_llm_prompt(input: &str, config: &Config, client: &Client) -> Result<()> {
    println!("... generating via {} ...", config.active_provider);
    let generation_result = match config.active_provider {
        LlmProvider::Ollama => ollama::generate(client, config, None, input).await,
        LlmProvider::Gemini => {
            if config.gemini_api_key.is_none() {
                Err(anyhow!("GEMINI_API_KEY not set."))
            } else {
                gemini::generate(client, config, input).await
            }
        }
        LlmProvider::Groq => {
            if config.groq_api_key.is_none() {
                Err(anyhow!("GROQ_API_KEY not set."))
            } else {
                groq::generate(client, config, input).await
            }
        }
        LlmProvider::HuggingFace => {
                    if config.huggingface_api_key.is_none() {
                        Err(anyhow!("HUGGINGFACE_API_KEY not set."))
                    } else {
                        huggingface::generate(client, config, input).await
                    }
                }
    };

    // Display LLM result or error
    match generation_result {
        Ok(response) => {
            if let Ok(mut glow_process) = Command::new("glow")
                .stdin(Stdio::piped())
                .stdout(Stdio::inherit()) // Inherit glow's stdout to print to the terminal
                .stderr(Stdio::inherit()) // Inherit glow's stderr for any errors
                .spawn()
            {
                if let Some(mut stdin) = glow_process.stdin.take() {
                    if let Err(e) = stdin.write_all(response.as_bytes()) {
                        eprintln!("Error writing to glow's stdin: {}", e);
                    }
                }

                if let Err(e) = glow_process.wait() {
                    eprintln!("Error waiting for glow to finish: {}", e);
                }
            } else {
                // If glow is not found or fails to start, fall back to plain text
                println!("\n{}", response.trim());
            }
            println!("---");
        }
        Err(e) => {
            error!("Generation error [{}]: {:?}", config.active_provider, e);
            eprintln!("\nError [{}]: {}", config.active_provider, e);
            println!("---");
        }
    }
    Ok(())
}

// --- Application Command Handler ---
async fn handle_app_command(
    input: &str,
    config: &mut Config,
    client: &Client,
) -> Result<()> {
    let parts: Vec<&str> = input[1..].splitn(2, ' ').collect();
    let command = parts[0].trim();
    let args_str = parts.get(1).map(|s| s.trim()).unwrap_or("");
    let args: Vec<&str> = args_str.split_whitespace().collect();

    debug!("Handling app command: '{}', args: {:?}", command, args);

    match command {
        "help" => print_help(),
        "status" => handle_status_command(config, client).await?,
        "use" => handle_use_command(config, &args)?,
        "model" => handle_model_command(config, client, args_str).await?,
        "model_list" => handle_model_list_command(config, client).await?,
        "select_model" => handle_select_model_command(config, client, args_str).await?,
        "gemini_config" => handle_gemini_config_command(config, &args)?,
        "groq_config" => handle_groq_config_command(config, &args)?,
        "huggingface_config" => handle_huggingface_config_command(config, &args)?,
        "config" => handle_config_command(config),
        "quit" | "exit" => {} // Handled in main loop
        _ => {
            println!("{} {}", UNKNOWN_COMMAND_MSG, command);
            println!("---");
        }
    }

    Ok(())
}

// --- Command-Specific Handlers ---

async fn handle_status_command(config: &Config, client: &Client) -> Result<()> {
    println!("Checking connection status...");
    print!(" - Ollama ({}): ", config.ollama_base_url);    
    io::stdout().flush()?;
    match ollama::check_connection(client, config).await {
        Ok(()) => println!("Connected"),
        Err(e) => println!("Error ({})", e),
    }
    print!(" - Gemini API: ");    
    if config.gemini_api_key.is_none() {
        println!("Not configured");
    } else {
        match gemini::check_connection(client, config).await {
            Ok(()) => println!("Connected"),
            Err(e) => println!("Error ({})", e),
        }
    }
    print!(" - Groq API: ");
    io::stdout().flush()?;
    if config.groq_api_key.is_none() {
        println!("Not configured (GROQ_API_KEY not set)");
    } else {
        match groq::check_connection(client, config).await {
            Ok(()) => println!("Connected"),
            Err(e) => println!("Error ({})", e),
        }
    }
    print!(" - Hugging Face API: ");
    io::stdout().flush()?;
    if config.huggingface_api_key.is_none() {
        println!("Not configured (HUGGINGFACE_API_KEY not set)");
    } else {
        match huggingface::check_connection(client, config).await {
            Ok(()) => println!("Connected"),
            Err(e) => println!("Error ({})", e),
        }
     }
    println!("---");
    Ok(())
}

fn handle_use_command(config: &mut Config, args: &[&str]) -> Result<()> {
    if args.len() != 1 {
        println!("Usage: /use <provider> (ollama, gemini, groq)");
    } else {
        match args[0].to_lowercase().as_str() {
            "ollama" => {
                config.active_provider = LlmProvider::Ollama;
                println!("Switched to Ollama (Model: {}).", config.default_ollama_model);
            }
            "gemini" => {
                if config.gemini_api_key.is_none() {
                    println!("Error: GEMINI_API_KEY not set.");
                } else {
                    config.active_provider = LlmProvider::Gemini;
                    println!("Switched to Gemini (Model: {}).", config.default_gemini_model);
                }
            }
            "groq" => {
                if config.groq_api_key.is_none() {
                    println!("Error: GROQ_API_KEY not set.");
                } else {
                    config.active_provider = LlmProvider::Groq;
                    println!("Switched to Groq (Model: {}).", config.default_groq_model);
                }
            }
            "huggingface" => {
                if config.huggingface_api_key.is_none() {
                    println!("Error: HUGGINGFACE_API_KEY not set.");
                } else {
                    config.active_provider = LlmProvider::HuggingFace;
                    println!("Switched to Hugging Face (Model: {}).", config.default_huggingface_model);
                }
            }
            _ => {
                println!(
                    "Unknown provider: '{}'. Available: ollama, gemini, groq",
                    args[0]
                );
            }
        }
    }
    println!("---");
    Ok(())
}

async fn handle_model_command(config: &mut Config, client: &Client, args_str: &str) -> Result<()> {
    if args_str.is_empty() {
        println!("Current model: {}", config.get_active_model_name());
        println!("Usage: /model <name>");
        println!("Use /select_model for interactive selection.");
    } else {
        let model_name = args_str;
        let known_models = match config.active_provider {
            LlmProvider::Ollama => ollama::list_models(client, config).await.ok(),
            LlmProvider::Gemini => gemini::list_models(client, config).await.ok(),
            LlmProvider::Groq => groq::list_models(client, config).await.ok(),
            LlmProvider::HuggingFace => huggingface::list_models(client, config).await.ok(),

        };
        if let Some(models) = known_models {
            if !models.iter().any(|m| m == model_name) {
                warn!(
                    "{} model '{}' not found via /model_list.",
                    config.active_provider.get_provider_name(),
                    model_name
                );
                println!("Warning: Model '{}' not verified.", model_name);
            }
        } else {
            warn!("Could not verify model existence.");
        }
        let active_provider = config.active_provider.clone();
        config.set_provider_model(&active_provider, model_name.to_string());
        println!("Set default model to: {}", config.get_active_model_name());
    }
    println!("---");
    Ok(())
}

async fn handle_model_list_command(config: &Config, client: &Client) -> Result<()> {
    match config.active_provider {
        LlmProvider::Ollama => {
            println!("Fetching available Ollama models...");
            match ollama::list_models(client, config).await {
                Ok(models) => {
                    if models.is_empty() {
                        println!("No Ollama models found.");
                    } else {
                        println!("Available Ollama models:");
                        models.iter().for_each(|m| println!(" - {}", m));
                    }
                }
                Err(e) => {
                    error!("Failed to list Ollama models: {:?}", e);
                    eprintln!("Error: {}", e);
                }
            }
        }
        LlmProvider::Gemini => {
            handle_gemini_list_command(config, client).await?;
        }
        LlmProvider::Groq => {
            handle_groq_list_command(config, client).await?;
        }
        LlmProvider::HuggingFace => {
            handle_huggingface_list_command(config, client).await?;
        }        
    }
    println!("---");
    Ok(())
}

async fn handle_select_model_command(config: &mut Config, client: &Client, args_str: &str) -> Result<()> {
    if !args_str.is_empty() {
        println!("Usage: /select_model");
        println!("---");
        return Ok(());
    }
    let models = match config.active_provider {
        LlmProvider::Ollama => {
            println!("Fetching available Ollama models...");
            ollama::list_models(client, config).await
        }
        LlmProvider::Gemini => {
            println!("Fetching available Gemini models for selection...");
            gemini::list_models(client, config).await
        }
        LlmProvider::Groq => {
            println!("Fetching available Groq models for selection...");
            groq::list_models(client, config).await
        }
        LlmProvider::HuggingFace => {
            println!("Fetching available Hugging Face models for selection...");
            huggingface::list_models(client, config).await
            }
    };
    match models {
        Ok(models) => {
            if models.is_empty() {
                println!("No models found.");
                println!("---");
                return Ok(());
            }
            let prompt = format!("Available {} models:", config.active_provider.get_provider_name());
            if let Some(selected_model) = select_model(&models, &prompt).await? {
                let active_provider = config.active_provider.clone();
                config.set_provider_model(&active_provider, selected_model.clone());

                println!("Selected {} model: {}", config.active_provider.get_provider_name(), config.get_active_model_name());
            }
        }
        Err(e) => {
            error!("Failed to list models: {:?}", e);
            eprintln!("Error: {}", e);
        }
    }
    println!("---");
    Ok(())
}

async fn handle_gemini_list_command(config: &Config, client: &Client) -> Result<()> {
    if config.gemini_api_key.is_none() {
        println!("Error: GEMINI_API_KEY not set.");
        println!("---");
        return Ok(());
    }
    println!("Fetching available Gemini models (requires valid API key)...");
    match gemini::list_models(client, config).await {
        Ok(models) => {
            if models.is_empty() {
                println!("No Gemini models supporting chat found (or API key invalid/restricted?).");
            } else {
                println!("Available Gemini chat models:");
                models.iter().for_each(|m| println!(" - {}", m));
            }
        }
        Err(e) => {
            error!("Failed to list Gemini models: {:?}", e);
            eprintln!("Error fetching Gemini models: {}", e);
        }
    }
    println!("---");
    Ok(())
}

fn handle_gemini_config_command(config: &mut Config, args: &[&str]) -> Result<()> {
    if args.is_empty() {
        println!("Current Gemini Configuration:");
        println!("  Model: {}", config.default_gemini_model);
        println!(
            "  Temperature: {}",
            config
                .gemini_temperature
                .map_or_else(|| "Default".to_string(), |v| v.to_string())
        );
        println!(
            "  Top P: {}",
            config
                .gemini_top_p
                .map_or_else(|| "Default".to_string(), |v| v.to_string())
        );
        println!(
            "  Max Tokens: {}",
            config
                .gemini_max_tokens
                .map_or_else(|| "Default".to_string(), |v| v.to_string())
        );
        println!("Usage: /gemini_config [temp <v|reset>] [top_p <v|reset>] [max_tokens <v|reset>] [reset]");
    } else {
        let mut i = 0;
        while i < args.len() {
            let param = args[i].to_lowercase();
            let value = args.get(i + 1);
            match (param.as_str(), value) {
                ("temp", Some(&vs)) => {
                    if vs.eq_ignore_ascii_case("reset") {
                        config.gemini_temperature = Config::default().gemini_temperature;
                        println!("Reset Temp to default ({:?})", config.gemini_temperature);
                    } else if let Ok(v) = vs.parse::<f32>() {
                        if (0.0..=1.0).contains(&v) {
                            config.gemini_temperature = Some(v);
                            println!("Set Temp to {}", v);
                        } else {
                            println!("Invalid temp '{}'. Must be 0.0-1.0.", vs);
                        }
                    } else {
                        println!("Invalid temp value '{}'", vs);
                    }
                    i += 2;
                }
                ("top_p", Some(&vs)) => {
                    if vs.eq_ignore_ascii_case("reset") {
                        config.gemini_top_p = Config::default().gemini_top_p;
                        println!("Reset Top P to default ({:?})", config.gemini_top_p);
                    } else if let Ok(v) = vs.parse::<f32>() {
                        if (0.0..=1.0).contains(&v) {
                            config.gemini_top_p = Some(v);
                            println!("Set Top P to {}", v);
                        } else {
                            println!("Invalid top_p '{}'. Must be 0.0-1.0.", vs);
                        }
                    } else {
                        println!("Invalid top_p value '{}'", vs);
                    }
                    i += 2;
                }
                ("max_tokens", Some(&vs)) => {
                    if vs.eq_ignore_ascii_case("reset") {
                        config.gemini_max_tokens = Config::default().gemini_max_tokens;
                        println!("Reset Max Tokens to default ({:?})", config.gemini_max_tokens);
                    } else if let Ok(v) = vs.parse::<u32>() {
                        if v > 0 {
                            config.gemini_max_tokens = Some(v);
                            println!("Set Max Tokens to {}", v);
                        } else {
                            println!("Invalid max_tokens '{}'. Must be >0.", vs);
                        }
                    } else {
                        println!("Invalid max_tokens value '{}'", vs);
                    }
                    i += 2;
                }
                ("reset", None) => {
                    config.gemini_temperature = Config::default().gemini_temperature;
                    config.gemini_top_p = Config::default().gemini_top_p;
                    config.gemini_max_tokens = Config::default().gemini_max_tokens;
                    println!("Reset all Gemini parameters to defaults.");
                    i += 1;
                }
                _ => {
                    println!(
                        "Unknown parameter or missing value for '{}'. Use /gemini_config for help.",
                        args[i]
                    );
                    i += 1;
                }
            }
        }
    }
    println!("---");
    Ok(())
}

async fn handle_groq_list_command(config: &Config, client: &Client) -> Result<()> {
    if config.groq_api_key.is_none() {
        println!("Error: GROQ_API_KEY not set.");
        println!("---");
        return Ok(());
    }
    println!("Fetching available Groq models...");
    match groq::list_models(client, config).await {
        Ok(models) => {
            if models.is_empty() {
                println!("No Groq models found.");
            } else {
                println!("Available Groq models:");
                models.iter().for_each(|m| println!(" - {}", m));
            }
        }
        Err(e) => {
            error!("Failed to list Groq models: {:?}", e);
            eprintln!("Error fetching Groq models: {}", e);
        }
    }
    println!("---");
    Ok(())
}

async fn handle_huggingface_list_command(config: &Config, _client: &Client) -> Result<()> {
    if config.huggingface_api_key.is_none() {
        println!("Error: HUGGINGFACE_API_KEY not set.");
        println!("---");
         return Ok(());
     }
    println!("Hugging Face model listing is not implemented yet.");
    println!("---");
    Ok(())
    }
    
    fn handle_huggingface_config_command(_config: &mut Config, _args: &[&str]) -> Result<()> {
    println!("Hugging Face configuration is not implemented yet.");
    println!("---");
    Ok(())
    }
    

fn handle_groq_config_command(_config: &mut Config, args: &[&str]) -> Result<()> {
    if args.is_empty() {
        println!("Current Groq Configuration:");
    } else {
    }
    println!("---");
    Ok(())
}

fn handle_config_command(config: &Config) {
    println!("Current Configuration:");
    println!("  Active Provider: {}", config.active_provider);
    println!("--- Ollama ---");
    println!("  Base URL: {}", config.ollama_base_url);
    println!("  Model:    {}", config.default_ollama_model);
    println!("--- Gemini ---");
    println!("  API Key Set: {}", config.gemini_api_key.is_some());
    println!("  Model:       {}", config.default_gemini_model);
    println!(
        "  Temperature: {}",
        config
            .gemini_temperature
            .map_or_else(|| "Default".to_string(), |v| v.to_string())
    );
    println!(
        "  Top P: {}",
        config
            .gemini_top_p
            .map_or_else(|| "Default".to_string(), |v| v.to_string())
    );
    println!(
        "  Max Tokens: {}",
        config
            .gemini_max_tokens
            .map_or_else(|| "Default".to_string(), |v| v.to_string())
    );
    println!("--- Groq ---");
    println!("  API Key Set: {}", config.groq_api_key.is_some());
    println!("  Base URL:    {}", config.groq_api_base_url);
    println!("  Model:       {}", config.default_groq_model);

    println!("--- Hugging Face ---");
    println!("  API Key Set: {}", config.huggingface_api_key.is_some());
    println!("  Model:       {}", config.default_huggingface_model);
    println!("---");
}

// --- Helper function for selecting a model from a list ---
async fn select_model(models: &[String], prompt: &str) -> Result<Option<String>> {
    if models.is_empty() {
        println!("No models found.");
        return Ok(None);
    }
    println!("{}", prompt);
    models
        .iter()
        .enumerate()
        .for_each(|(i, m)| println!("  {}. {}", i + 1, m));
    loop {
        print!("Enter number (or 0 to cancel): ");
        io::stdout().flush().context("Flush failed")?;
        let mut buf = String::new();
        io::stdin().read_line(&mut buf).context("Read failed")?;
        match buf.trim().parse::<usize>() {
            Ok(0) => {
                println!("Cancelled.");
                return Ok(None);
            }
            Ok(n) if n > 0 && n <= models.len() => {
                return Ok(Some(models[n - 1].clone()));
            }
            _ => println!("Invalid input. Try again."),
        }
    }
}

// --- Help Command ---
fn print_help() {
    println!("Available Commands:");
    println!(" General:");
    println!("  /help                    - Show this help message.");
    println!("  /status                  - Check connection status for configured providers.");
    println!("  /use <provider>          - Switch active LLM provider (ollama, gemini, groq).");
    println!("  /model <name>            - Set default model for the active provider.");
    println!("  /model_list              - List available models for the active provider.");
    println!("  /select_model            - Interactively select a model for the active provider.");
    println!("  /config                  - Show current configuration settings.");
    println!("  /quit | /exit            - Exit the application.");
    println!("  !<command> [args...]     - Execute a shell command.");
    println!(" Gemini Specific:");
    println!("  /gemini_config [...]     - View/Set Gemini generation parameters.");
    println!(" Groq Specific:");
    println!("  /groq_config [...]       - View/Set Groq generation parameters.");
    println!(" Hugging Face Specific:");
    println!("  /huggingface_config [...]       - View/Set Hugging Face generation parameters.");
    println!("Controls:");
    println!("  Up/Down Arrows           - Navigate command history.");
    println!("  Tab                      - Complete commands/paths.");
    println!("  Ctrl+C                   - Interrupt.");
    println!("  Ctrl+D                   - Exit.");
    println!("---");
    println!("Note: Set API keys via GEMINI_API_KEY / GROQ_API_KEY environment variables (or .env file).");
    println!("---");
}
