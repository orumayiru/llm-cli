// src/cli/helper.rs
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Context, Helper, Result as RustylineResult};

// Define the app commands that we want to complete
const APP_COMMANDS: [&str; 12] = [
    // General
    "/help", "/status", "/use", "/config", "/quit", "/exit","/model","/model_list","/select_model",
    // Gemini
    "/gemini_config",
    // Groq
    "/groq_config",
    //huggingface
    "/huggingface_config",
];

#[derive(Helper)]
pub struct ReplHelper {}

impl ReplHelper {
    pub fn new() -> Self {
        Self {}
    }
}

// --- Manual Implementation for Completer ---
impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> RustylineResult<(usize, Vec<Self::Candidate>)> {
        if line.starts_with('/') && pos > 0 {
            let command_part = if let Some(space_idx) = line.find(' ') {
                if pos <= space_idx {
                    &line[1..pos]
                } else {
                    return Ok((pos, Vec::new()));
                }
            } else {
                &line[1..pos]
            };

            let mut completions = Vec::new();
            for cmd in APP_COMMANDS.iter() {
                if cmd.starts_with(&format!("/{}", command_part)) {
                    completions.push(Pair {
                        display: cmd.to_string(),
                        replacement: cmd.to_string(),
                    });
                }
            }
            let start_pos = 0;
            Ok((start_pos, completions))
        } else if line.starts_with('!') {
            let filename_completer = rustyline::completion::FilenameCompleter::new();
            let mut start_pos = 1;
            while start_pos < line.len() && line.chars().nth(start_pos).map_or(false, |c| c.is_whitespace()) {
                start_pos += 1;
            }
            if pos >= start_pos {
                filename_completer.complete(&line[start_pos..], pos - start_pos, _ctx)
                    .map(|(replace_offset, candidates)| (start_pos + replace_offset, candidates))
            } else {
                Ok((pos, Vec::new()))
            }
        } else {
            Ok((pos, Vec::new()))
        }
    }
}

// --- Implement other traits (optional, rustyline provides defaults via derive) ---
impl Hinter for ReplHelper {
    type Hint = String;
    fn hint(&self, _line: &str, _pos: usize, _ctx: &Context<'_>) -> Option<String> {
        None
    }
}

impl Validator for ReplHelper {}

impl Highlighter for ReplHelper {}
