// src/config.rs
use crate::error::Result;
use std::{env, fmt};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmProvider {
    Ollama,
    Gemini,
    Groq,
    HuggingFace,
}

impl LlmProvider {
    pub fn get_provider_name(&self) -> &str {
        match self {
            LlmProvider::Ollama => "Ollama",
            LlmProvider::Gemini => "Gemini",
            LlmProvider::Groq => "Groq",
            LlmProvider::HuggingFace => "HuggingFace",

        }
    }

    pub fn get_provider_config_name(&self) -> &str {
        match self {
            LlmProvider::Ollama => "ollama",
            LlmProvider::Gemini => "gemini",
            LlmProvider::Groq => "groq",
            LlmProvider::HuggingFace => "huggingface",

        }
    }

    pub fn get_provider_model_name(&self) -> &str {
        match self {
            LlmProvider::Ollama => "default_ollama_model",
            LlmProvider::Gemini => "default_gemini_model",
            LlmProvider::Groq => "default_groq_model",
            LlmProvider::HuggingFace => "default_huggingface_model",

        }
    }

    pub fn get_provider_api_key_name(&self) -> &str {
        match self {
            LlmProvider::Ollama => "",
            LlmProvider::Gemini => "GEMINI_API_KEY",
            LlmProvider::Groq => "GROQ_API_KEY",
            LlmProvider::HuggingFace => "HUGGINGFACE_API_KEY",

        }
    }

    pub fn get_provider_base_url_name(&self) -> &str {
        match self {
            LlmProvider::Ollama => "ollama_base_url",
            LlmProvider::Gemini => "",
            LlmProvider::Groq => "groq_api_base_url",
            LlmProvider::HuggingFace => ""
        }
    }
}

impl fmt::Display for LlmProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get_provider_name())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // General
    pub active_provider: LlmProvider,

    // Ollama specific
    pub ollama_base_url: String,
    pub default_ollama_model: String,

    // Gemini specific
    pub gemini_api_key: Option<String>,
    pub default_gemini_model: String,
    pub gemini_temperature: Option<f32>,
    pub gemini_top_p: Option<f32>,
    pub gemini_max_tokens: Option<u32>,

    // Groq Specific
    pub groq_api_key: Option<String>,
    pub default_groq_model: String, // This is the field we need
    pub groq_api_base_url: String,

    // Hugging Face Specific
    pub huggingface_api_key: Option<String>,
    pub default_huggingface_model: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            active_provider: LlmProvider::Ollama,
            // Ollama
            ollama_base_url: "http://localhost:11434".to_string(),
            default_ollama_model: "llama3".to_string(),
            // Gemini
            gemini_api_key: None,
            default_gemini_model: "gemini-1.5-pro-latest".to_string(),
            gemini_temperature: Some(0.7),
            gemini_top_p: None,
            gemini_max_tokens: Some(2048),
            // Groq
            groq_api_key: None,
            default_groq_model: "llama3-8b-8192".to_string(), // Default Groq model name
            groq_api_base_url: "https://api.groq.com/openai/v1".to_string(),
            // Hugging Face
            huggingface_api_key: None,
            default_huggingface_model: "meta-llama/Llama-2-7b-chat-hf".to_string(),
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        dotenvy::dotenv().ok();

        let mut config = Config::default();

        config.gemini_api_key = env::var("GEMINI_API_KEY").ok();
        if config.active_provider == LlmProvider::Gemini && config.gemini_api_key.is_none() {
            eprintln!("Warning: GEMINI_API_KEY environment variable not set.");
        }

        config.groq_api_key = env::var("GROQ_API_KEY").ok();
        if config.active_provider == LlmProvider::Groq && config.groq_api_key.is_none() {
            eprintln!("Warning: GROQ_API_KEY environment variable not set.");
        }

        config.huggingface_api_key = env::var("HUGGINGFACE_API_KEY").ok();
        if config.active_provider == LlmProvider::HuggingFace && config.huggingface_api_key.is_none() {
            eprintln!("Warning: HUGGINGFACE_API_KEY environment variable not set.");
         }

        Ok(config)
    }

    pub fn reset_to_default(&mut self) {
        *self = Config::default();
    }

    // --- FIX IS HERE ---
    // Helper to get the currently active model name
    pub fn get_active_model_name(&self) -> &str {
        match self.active_provider {
            LlmProvider::Ollama => &self.default_ollama_model,
            LlmProvider::Gemini => &self.default_gemini_model,
            LlmProvider::Groq => &self.default_groq_model,
            LlmProvider::HuggingFace => &self.default_huggingface_model,

        }
    }
    // --- END FIX ---

    // Helper to get the API key for the active provider (if applicable)
    // This function was actually correct before.
    pub fn get_active_api_key(&self) -> Option<&str> {
        match self.active_provider {
            LlmProvider::Ollama => None,
            LlmProvider::Gemini => self.gemini_api_key.as_deref(),
            LlmProvider::Groq => self.groq_api_key.as_deref(),
            LlmProvider::HuggingFace => self.huggingface_api_key.as_deref(),

        }
    }

    pub fn get_provider_config(&self, provider: &LlmProvider) -> Vec<(&str, String, Option<f32>, Option<u32>, Option<f32>)> {
        match provider {
            LlmProvider::Ollama => vec![
                ("ollama_base_url", self.ollama_base_url.clone(), None, None, None),
                ("default_ollama_model", self.default_ollama_model.clone(), None, None, None),
            ],
            LlmProvider::Gemini => vec![
                ("default_gemini_model", self.default_gemini_model.clone(), None, None, None),
                ("gemini_temperature", self.gemini_temperature.map_or_else(|| "Default".to_string(), |v| v.to_string()), self.gemini_temperature, None, Some(0.0)),
                ("gemini_top_p", self.gemini_top_p.map_or_else(|| "Default".to_string(), |v| v.to_string()), self.gemini_top_p, None, Some(0.0)),
                ("gemini_max_tokens", self.gemini_max_tokens.map_or_else(|| "Default".to_string(), |v| v.to_string()), None, self.gemini_max_tokens, None),
            ],
            LlmProvider::Groq => vec![
                ("groq_api_base_url", self.groq_api_base_url.clone(), None, None, None),
                ("default_groq_model", self.default_groq_model.clone(), None, None, None),
            ],
            LlmProvider::HuggingFace => vec![
                ("default_huggingface_model", self.default_huggingface_model.clone(), None, None, None),
                // Add other Hugging Face parameters here if needed
             ],
        }
    }

    pub fn set_provider_model(&mut self, provider: &LlmProvider, model: String) {
        match provider {
            LlmProvider::Ollama => self.default_ollama_model = model,
            LlmProvider::Gemini => self.default_gemini_model = model,
            LlmProvider::Groq => self.default_groq_model = model,
            LlmProvider::HuggingFace => self.default_huggingface_model = model,

        }
    }
}
