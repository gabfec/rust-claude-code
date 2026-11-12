use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::{env, error::Error, fs, process::{self, Command}};

mod lsp;
mod lsp_client;
mod skills;

use lsp::Lsp;
use lsp_client::LspClient;
use skills::Skill;

const MODEL: &str = "anthropic/claude-haiku-4.5";

const LSP_SYSTEM_PROMPT: &str = "A language server is connected. Use the LSP tool \
to find where a symbol is defined (goToDefinition) or everything that references it \
(findReferences); its line and character inputs are 1-based. Take positions from \
tool results rather than counting lines yourself: edits move symbols, so after \
editing, rerun goToDefinition from a call site to get a definition's current \
position, then query findReferences at exactly that position. Make every file edit \
with the Write tool so the language server sees the updated code. Tool results may \
end with <new-diagnostics> reported by the language server for any file; treat them \
as feedback on your edits and fix the issues they describe.";

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

struct Agent {
    client: Client<OpenAIConfig>,
    skills: Vec<Skill>,
    tools: Value,
    lsp: Option<Lsp>,
}

impl Agent {
    /// Runs the agent loop over `messages` with the given tools until
    /// the model answers without calling a tool, and returns that answer.
    async fn run(&self, messages: &mut Vec<Value>, tools: &Value) -> Result<String, Box<dyn Error>> {
        loop {
            self.attach_diagnostics(messages);

            let response: Value = self
                .client
                .chat()
                .create_byot(json!({
                    "messages": messages,
                    "model": MODEL,
                    "tools": tools
                }))
                .await?;

            let message = &response["choices"][0]["message"];

            // Add the assistant's message to history
            messages.push(message.clone());

            let Some(tool_calls) = message["tool_calls"].as_array() else {
                return Ok(message["content"].as_str().unwrap_or("").to_string());
            };

            for tool_call in tool_calls {
                let tool_id = tool_call["id"].as_str().unwrap_or("");
                let function_name = tool_call["function"]["name"].as_str().unwrap_or("");
                let arguments_str = tool_call["function"]["arguments"].as_str().unwrap_or("{}");
                let content = match serde_json::from_str::<Value>(arguments_str) {
                    Ok(arguments) => self.execute_tool(function_name, &arguments).await?,
                    Err(e) => format!("Error: invalid tool arguments: {}", e),
                };

                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": content
                }));
            }
        }
    }

    /// Appends language-server diagnostics the model hasn't seen yet to
    /// the latest tool result, so it can repair its edits.
    fn attach_diagnostics(&self, messages: &mut [Value]) {
        let Some(lsp) = &self.lsp else {
            return;
        };
        if let Err(e) = lsp.poll_diagnostics() {
            eprintln!("Failed to read LSP diagnostics: {}", e);
        }

        let Some(last) = messages.last_mut().filter(|m| m["role"] == "tool") else {
            return;
        };
        if let Some(report) = lsp.take_new_diagnostics() {
            let content = last["content"].as_str().unwrap_or("");
            last["content"] = json!(format!("{}\n\n{}", content, report));
        }
    }

    async fn execute_tool(&self, name: &str, arguments: &Value) -> Result<String, Box<dyn Error>> {
        let content = match name {
            "Read" => {
                let file_path = arguments["file_path"].as_str().unwrap_or("");
                fs::read_to_string(file_path).unwrap_or_else(|e| format!("Error: {}", e))
            }
            "Write" => {
                let file_path = arguments["file_path"].as_str().unwrap_or("");
                let content = arguments["content"].as_str().unwrap_or("");
                match fs::write(file_path, content) {
                    Ok(()) => {
                        if let Some(lsp) = &self.lsp {
                            if let Err(e) = lsp.sync_write(file_path, content) {
                                eprintln!("LSP sync failed for {}: {}", file_path, e);
                            }
                        }
                        content.to_string()
                    }
                    Err(e) => format!("Error: {}", e),
                }
            }
            "Bash" => {
                let command = arguments["command"].as_str().unwrap_or("");
                let output = Command::new("sh").arg("-c").arg(command).output()?;
                let mut text = String::from_utf8_lossy(&output.stdout).to_string();
                text.push_str(&String::from_utf8_lossy(&output.stderr));
                text
            }
            "LSP" if self.lsp.is_some() => {
                let lsp = self.lsp.as_ref().unwrap();
                lsp.execute(arguments)
            }
            "Skill" => {
                let name = arguments["name"].as_str().unwrap_or("");
                let skill_args = arguments["args"].as_str().unwrap_or("");
                match self.skills.iter().find(|s| s.name == name) {
                    Some(skill) if skill.fork => self.run_subagent(skill, skill_args).await?,
                    Some(skill) => skill.instructions(skill_args),
                    None => format!("Error: unknown skill `{}`", name),
                }
            }
            _ => {
                eprintln!("Unknown function: {}", name);
                format!("Error: unknown tool `{}`", name)
            }
        };
        Ok(content)
    }

    /// Runs a `context: fork` skill in a fresh conversation holding only
    /// its body, and reports the subagent's answer.
    ///
    /// The subagent gets every tool except Skill: it is there to carry
    /// out one skill, and letting it invoke skills could fork forever.
    async fn run_subagent(&self, skill: &Skill, arguments: &str) -> Result<String, Box<dyn Error>> {
        let mut messages = vec![json!({
            "role": "user",
            "content": skills::substitute_arguments(&skill.body, arguments)
        })];
        let tools: Vec<Value> = self
            .tools
            .as_array()
            .into_iter()
            .flatten()
            .filter(|t| t["function"]["name"] != "Skill")
            .cloned()
            .collect();
        let answer = Box::pin(self.run(&mut messages, &json!(tools))).await?;
        Ok(format!(
            "Skill {} ran in a separate context and returned: {}",
            skill.name, answer
        ))
    }
}

fn lsp_tool_definition() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "LSP",
            "description": "Ask the language server about the symbol at a position in a file. goToDefinition returns the file, line, and character where the symbol is defined. findReferences returns every location that references the symbol, including its declaration. Line and character are 1-based, as shown in an editor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["goToDefinition", "findReferences"]
                    },
                    "filePath": {
                        "type": "string",
                        "description": "Absolute path, or a path relative to the agent's current working directory."
                    },
                    "line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-based line number of the symbol."
                    },
                    "character": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-based character offset in UTF-16 code units (same as character count for ASCII)."
                    }
                },
                "required": ["operation", "filePath", "line", "character"]
            }
        }
    })
}

fn tool_definitions() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file, replacing it. Use this for every file edit so the language server sees the change.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path of the file to write to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Skill",
                "description": "Load a skill's instructions into the conversation",
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the skill to use"
                        },
                        "args": {
                            "type": "string",
                            "description": "Optional arguments for the skill"
                        }
                    }
                }
            }
        }
    ])
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let lsp = LspClient::from_environment()?;
    let mut tools = tool_definitions();
    if lsp.is_some() {
        eprintln!("LSP client connected");
        tools.as_array_mut().unwrap().push(lsp_tool_definition());
    }

    let agent = Agent {
        client: Client::with_config(config),
        skills: skills::discover(),
        tools,
        lsp: lsp.map(Lsp::new),
    };

    let mut system_prompt = Vec::new();
    if !agent.skills.is_empty() {
        system_prompt.push(skills::system_prompt(&agent.skills));
    }
    if agent.lsp.is_some() {
        system_prompt.push(LSP_SYSTEM_PROMPT.to_string());
    }

    let mut messages = Vec::new();
    if !system_prompt.is_empty() {
        messages.push(json!({
            "role": "system",
            "content": system_prompt.join("\n")
        }));
    }
    let mut user_contents = skills::expand_prompt(&agent.skills, &args.prompt);
    if user_contents.is_empty() {
        user_contents.push(args.prompt);
    }
    for content in user_contents {
        messages.push(json!({
            "role": "user",
            "content": content
        }));
    }

    let answer = agent.run(&mut messages, &agent.tools).await?;
    print!("{}", answer);

    Ok(())
}
