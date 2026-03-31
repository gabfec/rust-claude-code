use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::{env, fs, process::{self, Command}};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let client = Client::with_config(config);

    let mut messages = vec![json!({
        "role": "user",
        "content": args.prompt
    })];

    let tools = json!([
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
                "description": "Write content to a file",
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
        }
    ]);

    loop {
        let response: Value = client
            .chat()
            .create_byot(json!({
                "messages": messages,
                "model": "anthropic/claude-haiku-4.5",
                "tools": tools
            }))
            .await?;

        let choice = &response["choices"][0];
        let message = &choice["message"];

        // Add the assistant's message to history
        messages.push(message.clone());

        // If there are tool calls
        if let Some(tool_calls) = message["tool_calls"].as_array() {
            for tool_call in tool_calls {
                let tool_id = tool_call["id"].as_str().unwrap();
                let function_name = tool_call["function"]["name"].as_str().unwrap();
                let arguments_str = tool_call["function"]["arguments"].as_str().unwrap();
                let arguments: Value = serde_json::from_str(arguments_str)?;

                match function_name {
                    "Read" => {
                        let file_path = arguments["file_path"].as_str().unwrap();
                        let contents = fs::read_to_string(file_path)
                            .unwrap_or_else(|e| format!("Error: {}", e));

                        messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": contents
                        }));
                    }
                    "Write" => {
                        let file_path = arguments["file_path"].as_str().unwrap();
                        let content = arguments["content"].as_str().unwrap();
                        let _ = fs::write(file_path, content);
                        messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": content
                        }));
                    }
                    "Bash" => {
                        let command = arguments["command"].as_str().unwrap();
                        let output = Command::new("sh").arg("-c").arg(command).output()?;
                        let content = String::from_utf8_lossy(&output.stdout).to_string();
                        messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": content
                        }));
                    }
                    _ => {
                        eprintln!("Unknown function: {}", function_name);
                    }
                }
            }

            continue;
        }

        // No more tool calls
        if let Some(content) = message["content"].as_str() {
            print!("{}", content);
        }

        break;
    }

    Ok(())
}
