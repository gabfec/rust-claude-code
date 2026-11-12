//! End-to-end tests: run the agent binary against a mock OpenAI-compatible
//! server that answers with scripted replies and records every request.
//! Each test runs in its own temporary project directory.

use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Arc, Mutex};
use std::{env, fs, process, thread};

/// A project directory for one test, removed when dropped.
struct Project(PathBuf);

impl Project {
    fn new(name: &str) -> Self {
        let dir = env::temp_dir().join(format!("rust-claude-code-cli-{}-{}", process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Project(dir)
    }

    fn write(&self, relative: &str, contents: &str) {
        let path = self.0.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    fn read(&self, relative: &str) -> String {
        fs::read_to_string(self.0.join(relative)).unwrap()
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A chat-completions server that replies with the scripted assistant
/// messages in order and records the request bodies.
struct MockModel {
    base_url: String,
    requests: Arc<Mutex<Vec<Value>>>,
}

impl MockModel {
    fn start(replies: Vec<Value>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));

        let recorded = Arc::clone(&requests);
        thread::spawn(move || {
            let mut replies = replies.into_iter();
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let Some(body) = read_request_body(&mut stream) else {
                    continue;
                };
                recorded.lock().unwrap().push(serde_json::from_slice(&body).unwrap());

                let message = replies
                    .next()
                    .unwrap_or_else(|| answer("mock model: no reply left"));
                let payload = json!({
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test",
                    "choices": [{ "index": 0, "message": message, "finish_reason": "stop" }]
                })
                .to_string();
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                );
            }
        });

        MockModel { base_url, requests }
    }

    fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }
}

fn read_request_body(stream: &mut TcpStream) -> Option<Vec<u8>> {
    let mut data = Vec::new();
    let mut chunk = [0u8; 8192];
    let header_end = loop {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        data.extend_from_slice(&chunk[..n]);
        if let Some(i) = data.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
    };

    let headers = String::from_utf8_lossy(&data[..header_end]).to_lowercase();
    let length: usize = headers
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    while data.len() < header_end + length {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        data.extend_from_slice(&chunk[..n]);
    }
    Some(data[header_end..header_end + length].to_vec())
}

/// An assistant reply with no tool calls.
fn answer(text: &str) -> Value {
    json!({ "role": "assistant", "content": text })
}

/// An assistant reply calling tools: `(id, tool, arguments)`.
fn call_tools(calls: &[(&str, &str, Value)]) -> Value {
    let calls: Vec<Value> = calls
        .iter()
        .map(|(id, name, arguments)| {
            json!({
                "id": id,
                "type": "function",
                "function": { "name": name, "arguments": arguments.to_string() }
            })
        })
        .collect();
    json!({ "role": "assistant", "content": null, "tool_calls": calls })
}

fn run(project: &Project, model: &MockModel, prompt: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_codecrafters-claude-code"))
        .args(["-p", prompt])
        .current_dir(project.path())
        .env("OPENROUTER_API_KEY", "test-key")
        .env("OPENROUTER_BASE_URL", &model.base_url)
        .env_remove("CODECRAFTERS_LSP_URL")
        .output()
        .unwrap()
}

fn stdout(output: &Output) -> String {
    assert!(
        output.status.success(),
        "agent failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone()).unwrap()
}

fn tool_names(request: &Value) -> Vec<String> {
    request["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap().to_string())
        .collect()
}

/// The tool results at the end of a request, as `(tool_call_id, content)`.
fn tool_results(request: &Value) -> Vec<(String, String)> {
    request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "tool")
        .map(|m| {
            (
                m["tool_call_id"].as_str().unwrap().to_string(),
                m["content"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

#[test]
fn missing_api_key_is_reported() {
    let project = Project::new("no-key");
    let output = Command::new(env!("CARGO_BIN_EXE_codecrafters-claude-code"))
        .args(["-p", "hi"])
        .current_dir(project.path())
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&output.stderr), "OPENROUTER_API_KEY is not set\n");
    assert!(output.stdout.is_empty());
}

#[test]
fn prints_the_answer_to_a_plain_prompt() {
    let project = Project::new("plain");
    let model = MockModel::start(vec![answer("5")]);

    assert_eq!(stdout(&run(&project, &model, "What is 2+3?")), "5");

    let requests = model.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["model"], "anthropic/claude-haiku-4.5");
    assert_eq!(requests[0]["messages"], json!([{ "role": "user", "content": "What is 2+3?" }]));
    assert_eq!(tool_names(&requests[0]), ["Read", "Write", "Bash", "Skill"]);
}

#[test]
fn tool_results_go_back_to_the_model_until_it_answers() {
    let project = Project::new("read");
    project.write("pineapple.py", "print('Hello, program!')\n");
    let model = MockModel::start(vec![
        call_tools(&[("call_1", "Read", json!({ "file_path": "pineapple.py" }))]),
        answer("print('Hello, program!')"),
    ]);

    let output = run(&project, &model, "Show me pineapple.py");
    assert_eq!(stdout(&output), "print('Hello, program!')");

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    let history = requests[1]["messages"].as_array().unwrap();
    assert_eq!(history[1]["tool_calls"][0]["id"], "call_1");
    assert_eq!(
        tool_results(&requests[1]),
        [("call_1".to_string(), "print('Hello, program!')\n".to_string())]
    );
}

#[test]
fn parallel_tool_calls_each_get_a_result() {
    let project = Project::new("write-bash");
    let model = MockModel::start(vec![
        call_tools(&[
            ("w", "Write", json!({ "file_path": "app/start.py", "content": "print(1)\n" })),
            ("b", "Bash", json!({ "command": "cat app/start.py; echo warning >&2" })),
        ]),
        answer("Done"),
    ]);

    // The app folder doesn't exist, so the Write fails and Bash sees no file.
    assert_eq!(stdout(&run(&project, &model, "Create app/start.py")), "Done");
    let results = tool_results(&model.requests()[1]);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "w");
    assert!(results[0].1.starts_with("Error: "), "{}", results[0].1);
    assert_eq!(results[1].0, "b");
    assert!(results[1].1.ends_with("warning\n"), "{}", results[1].1);

    // With the folder in place, the same calls succeed.
    project.write("app/.keep", "");
    let model = MockModel::start(vec![
        call_tools(&[
            ("w", "Write", json!({ "file_path": "app/start.py", "content": "print(1)\n" })),
            ("b", "Bash", json!({ "command": "cat app/start.py; echo warning >&2" })),
        ]),
        answer("Done"),
    ]);
    stdout(&run(&project, &model, "Create app/start.py"));
    assert_eq!(project.read("app/start.py"), "print(1)\n");
    assert_eq!(
        tool_results(&model.requests()[1]),
        [
            ("w".to_string(), "print(1)\n".to_string()),
            ("b".to_string(), "print(1)\nwarning\n".to_string()),
        ]
    );
}

#[test]
fn unknown_tools_get_an_error_result() {
    let project = Project::new("unknown-tool");
    let model = MockModel::start(vec![
        call_tools(&[("t", "Teleport", json!({}))]),
        answer("Sorry"),
    ]);

    assert_eq!(stdout(&run(&project, &model, "Go")), "Sorry");
    assert_eq!(
        tool_results(&model.requests()[1]),
        [("t".to_string(), "Error: unknown tool `Teleport`".to_string())]
    );
}

fn write_skills(project: &Project) {
    project.write(
        ".claude/skills/apple/SKILL.md",
        "---\nname: apple\ndescription: Deploys the apple service.\n---\n\n\
         Deploy to $0 in region $1. Full request was: $ARGUMENTS\n",
    );
    project.write(
        ".claude/skills/grape/SKILL.md",
        "---\nname: grape\ndescription: Use this skill when asked about the rotation.\n\
         context: fork\n---\n\nRespond with exactly one word: jujube\n",
    );
    project.write(
        ".claude/skills/kiwi/SKILL.md",
        "---\nname: kiwi\ndescription: Reports the data checksum.\n---\n\n\
         Run `scripts/checksum.sh` using the Bash tool.\n",
    );
    project.write(".claude/skills/kiwi/scripts/checksum.sh", "echo 9f2b4c1a\n");
}

#[test]
fn skills_are_advertised_without_their_bodies() {
    let project = Project::new("advertise");
    write_skills(&project);
    let model = MockModel::start(vec![answer("apple")]);

    stdout(&run(&project, &model, "Which skill deploys apple?"));

    let messages = model.requests()[0]["messages"].clone();
    let system = messages[0]["content"].as_str().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert!(system.contains("- apple: Deploys the apple service.\n"), "{system}");
    assert!(system.contains("- grape: Use this skill when asked about the rotation.\n"));
    assert!(system.contains("- kiwi: Reports the data checksum.\n"));
    assert!(!system.contains("Deploy to"), "{system}");
    assert_eq!(messages[1], json!({ "role": "user", "content": "Which skill deploys apple?" }));
}

#[test]
fn slash_commands_expand_and_stack_skills() {
    let project = Project::new("slash");
    write_skills(&project);

    let model = MockModel::start(vec![answer("ok")]);
    stdout(&run(&project, &model, "/apple staging eu-west"));
    assert_eq!(
        model.requests()[0]["messages"][1]["content"],
        "Deploy to staging in region eu-west. Full request was: staging eu-west"
    );

    let model = MockModel::start(vec![answer("ok")]);
    stdout(&run(&project, &model, "/apple /kiwi 4127"));
    let messages = model.requests()[0]["messages"].clone();
    assert_eq!(messages.as_array().unwrap().len(), 3);
    assert_eq!(messages[1]["content"], "Deploy to 4127 in region . Full request was: 4127");
    assert_eq!(
        messages[2]["content"],
        "Skill: kiwi (located at .claude/skills/kiwi)\n\
         Paths in the instructions below are relative to that folder.\n\n\
         Run `scripts/checksum.sh` using the Bash tool."
    );
}

#[test]
fn the_model_can_load_a_skill_with_the_skill_tool() {
    let project = Project::new("skill-tool");
    write_skills(&project);
    let model = MockModel::start(vec![
        call_tools(&[("s", "Skill", json!({ "name": "apple", "args": "prod us" }))]),
        answer("Deployed"),
    ]);

    assert_eq!(stdout(&run(&project, &model, "Deploy apple")), "Deployed");
    assert_eq!(
        tool_results(&model.requests()[1]),
        [(
            "s".to_string(),
            "Deploy to prod in region us. Full request was: prod us".to_string()
        )]
    );
}

#[test]
fn a_forked_skill_runs_in_a_subagent_with_only_its_body() {
    let project = Project::new("fork");
    write_skills(&project);
    let model = MockModel::start(vec![
        call_tools(&[("s", "Skill", json!({ "name": "grape" }))]),
        answer("jujube"),
        answer("The on-call engineer is jujube."),
    ]);

    let output = run(&project, &model, "Who is on the rotation?");
    assert_eq!(stdout(&output), "The on-call engineer is jujube.");

    let requests = model.requests();
    assert_eq!(requests.len(), 3);

    // The subagent sees the skill body and nothing else, and cannot
    // invoke skills itself.
    assert_eq!(
        requests[1]["messages"],
        json!([{ "role": "user", "content": "Respond with exactly one word: jujube" }])
    );
    assert_eq!(tool_names(&requests[1]), ["Read", "Write", "Bash"]);

    // Only its answer comes back to the main conversation.
    assert_eq!(requests[2]["messages"][0]["role"], "system");
    assert_eq!(
        tool_results(&requests[2]),
        [(
            "s".to_string(),
            "Skill grape ran in a separate context and returned: jujube".to_string()
        )]
    );
}
