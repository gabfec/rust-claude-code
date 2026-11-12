use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use crate::lsp_client::LspClient;

/// How long one settle read waits for new notifications, and how many
/// reads a settle makes at most before a Write notification.
const SETTLE_WAIT_MS: u64 = 300;
const SETTLE_ROUNDS: usize = 5;

/// How `Lsp` reaches the language server. `LspClient` is the real
/// implementation; tests substitute a fake that records the calls.
pub trait Transport {
    /// Sends an action (`notify`, `request`, `notifications`) with its
    /// payload and returns the response.
    fn call(&self, action: &str, payload: &Value) -> Result<Value, Box<dyn Error>>;
    /// File extension (with the dot) -> LSP language ID.
    fn extension_to_language(&self) -> &HashMap<String, String>;
}

impl Transport for LspClient {
    fn call(&self, action: &str, payload: &Value) -> Result<Value, Box<dyn Error>> {
        LspClient::call(self, action, payload)
    }

    fn extension_to_language(&self) -> &HashMap<String, String> {
        &self.extension_to_language
    }
}

/// The language server connection plus the state of the documents it
/// has been sent on this connection.
pub struct Lsp {
    transport: Box<dyn Transport>,
    root: PathBuf,
    /// URI -> version last sent for every open document.
    versions: Mutex<HashMap<String, i64>>,
    /// Cursor for the next read of the notification queue.
    cursor: Mutex<Value>,
    diagnostics: Mutex<Diagnostics>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Diagnostic {
    line: u64,
    character: u64,
    message: String,
}

/// Tracks diagnostic reports and which of their issues the model still
/// needs to see. Holds no connection, so it can be tested directly.
#[derive(Default)]
struct Diagnostics {
    /// URI -> issues in the latest current report for that file, all
    /// of which have been shown or queued for the model.
    shown: HashMap<String, HashSet<Diagnostic>>,
    /// Issues from current reports that the model hasn't seen yet.
    pending: Vec<(String, Diagnostic)>,
}

impl Diagnostics {
    /// Applies a batch of notifications. A `publishDiagnostics` report
    /// whose version differs from the last one sent for that file
    /// (`versions`) is stale and ignored; a report without a version is
    /// accepted. A current report replaces the file's previous one, and
    /// its issues that weren't in the previous report are queued.
    ///
    /// Queued issues stay queued even if a later report clears them: the
    /// model must see every current diagnostic, and a report can arrive
    /// and be superseded between two model requests.
    fn ingest(&mut self, notifications: &[Value], versions: &HashMap<String, i64>) {
        for notification in notifications {
            if notification["method"] != "textDocument/publishDiagnostics" {
                continue;
            }
            let params = &notification["params"];
            let Some(uri) = params["uri"].as_str() else {
                continue;
            };
            if let (Some(reported), Some(sent)) = (params["version"].as_i64(), versions.get(uri)) {
                if reported != *sent {
                    continue;
                }
            }

            let issues: Vec<Diagnostic> = params["diagnostics"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|d| {
                    Some(Diagnostic {
                        line: d["range"]["start"]["line"].as_u64()?,
                        character: d["range"]["start"]["character"].as_u64()?,
                        message: d["message"].as_str()?.to_string(),
                    })
                })
                .collect();

            let previous = self.shown.remove(uri).unwrap_or_default();
            for issue in &issues {
                let queued = self.pending.iter().any(|(u, d)| u == uri && d == issue);
                if !previous.contains(issue) && !queued {
                    self.pending.push((uri.to_string(), issue.clone()));
                }
            }
            self.shown.insert(uri.to_string(), issues.into_iter().collect());
        }
    }

    /// Takes the queued issues, grouped by file in arrival order.
    fn take(&mut self) -> Vec<(String, Vec<Diagnostic>)> {
        let mut files: Vec<(String, Vec<Diagnostic>)> = Vec::new();
        for (uri, issue) in std::mem::take(&mut self.pending) {
            match files.iter_mut().find(|(u, _)| *u == uri) {
                Some((_, group)) => group.push(issue),
                None => files.push((uri, vec![issue])),
            }
        }
        files
    }
}

/// A zero-based position in a document, as LSP counts it.
struct Location {
    uri: String,
    line: u64,
    character: u64,
}

impl Lsp {
    /// Wraps a transport; relative paths resolve against `root`.
    pub fn new(transport: impl Transport + 'static, root: PathBuf) -> Self {
        Lsp {
            transport: Box::new(transport),
            root,
            versions: Mutex::new(HashMap::new()),
            cursor: Mutex::new(json!(0)),
            diagnostics: Mutex::new(Diagnostics::default()),
        }
    }

    /// Reads the queued notifications and records their diagnostic
    /// reports; see `Diagnostics::ingest`.
    pub fn poll_diagnostics(&self) -> Result<(), Box<dyn Error>> {
        self.read_notifications(0).map(|_| ())
    }

    /// Reads the notification queue, waiting up to `timeout_ms` for new
    /// ones, records their diagnostic reports, and returns how many
    /// notifications arrived.
    fn read_notifications(&self, timeout_ms: u64) -> Result<usize, Box<dyn Error>> {
        let mut cursor = self.cursor.lock().unwrap();
        let response = self.transport.call(
            "notifications",
            &json!({ "after": *cursor, "timeoutMs": timeout_ms }),
        )?;
        if !response["cursor"].is_null() {
            *cursor = response["cursor"].clone();
        }

        let notifications = response["notifications"].as_array().map(Vec::as_slice).unwrap_or(&[]);
        let versions = self.versions.lock().unwrap();
        self.diagnostics.lock().unwrap().ingest(notifications, &versions);
        Ok(notifications.len())
    }

    /// Lets the server finish publishing reports for earlier changes
    /// before a document's version moves on: reads the queue with a short
    /// wait until a read comes back empty. Reports read now are judged
    /// against the version they were published for; read after the next
    /// notification, they would look stale.
    fn settle(&self) {
        for _ in 0..SETTLE_ROUNDS {
            match self.read_notifications(SETTLE_WAIT_MS) {
                Ok(0) => break,
                Ok(_) => continue,
                Err(e) => {
                    eprintln!("Failed to read LSP diagnostics: {}", e);
                    break;
                }
            }
        }
    }

    /// Formats the queued issues, grouped by file, and clears the queue.
    /// Returns `None` when there is nothing new.
    pub fn take_new_diagnostics(&self) -> Option<String> {
        let files = self.diagnostics.lock().unwrap().take();
        if files.is_empty() {
            return None;
        }
        Some(format_diagnostics(&self.root, &files))
    }

    /// Executes an `LSP` tool call and returns its result text.
    pub fn execute(&self, arguments: &Value) -> String {
        match self.try_execute(arguments) {
            Ok(text) => text,
            Err(e) => format!("Error: {}", e),
        }
    }

    fn try_execute(&self, arguments: &Value) -> Result<String, Box<dyn Error>> {
        let operation = arguments["operation"].as_str().unwrap_or("");
        let file_path = arguments["filePath"].as_str().ok_or("filePath is required")?;
        let line = one_based(&arguments["line"]).ok_or("line must be a positive integer")?;
        let character =
            one_based(&arguments["character"]).ok_or("character must be a positive integer")?;

        let path = self.absolute(file_path);
        let uri = self.ensure_open(&path)?;
        let position = json!({ "line": line - 1, "character": character - 1 });

        match operation {
            "goToDefinition" => {
                let result = self.request(
                    "textDocument/definition",
                    json!({ "textDocument": { "uri": uri }, "position": position }),
                )?;
                let locations = parse_locations(&result);
                if locations.is_empty() {
                    return Ok("No definition found.".to_string());
                }
                Ok(locations
                    .iter()
                    .map(|l| format!("Defined in {}", display_location(&self.root, l)))
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
            "findReferences" => {
                let result = self.request(
                    "textDocument/references",
                    json!({
                        "textDocument": { "uri": uri },
                        "position": position,
                        "context": { "includeDeclaration": true }
                    }),
                )?;
                Ok(format_references(&self.root, &parse_locations(&result)))
            }
            other => Err(format!("unsupported operation `{}`", other).into()),
        }
    }

    /// Brings the server's copy of a just-written file up to date:
    /// `didChange` with the next version if the document is open,
    /// otherwise `didOpen` at version 1. Files outside the client's
    /// language mapping are ignored.
    pub fn sync_write(&self, file_path: &str, text: &str) -> Result<(), Box<dyn Error>> {
        let path = self.absolute(file_path);
        let Some(language_id) = self.mapped_language(&path) else {
            return Ok(());
        };
        let uri = path_to_uri(&path);

        self.settle();

        let current = self.versions.lock().unwrap().get(&uri).copied();

        let version = match current {
            Some(version) => {
                self.notify(
                    "textDocument/didChange",
                    json!({
                        "textDocument": { "uri": uri, "version": version + 1 },
                        "contentChanges": [{ "text": text }]
                    }),
                )?;
                version + 1
            }
            None => {
                self.notify(
                    "textDocument/didOpen",
                    json!({
                        "textDocument": {
                            "uri": uri,
                            "languageId": language_id,
                            "version": 1,
                            "text": text
                        }
                    }),
                )?;
                1
            }
        };
        self.versions.lock().unwrap().insert(uri, version);
        Ok(())
    }

    /// Sends `didOpen` for a document the server hasn't seen yet and
    /// returns its URI.
    fn ensure_open(&self, path: &Path) -> Result<String, Box<dyn Error>> {
        let uri = path_to_uri(path);
        if self.versions.lock().unwrap().contains_key(&uri) {
            return Ok(uri);
        }

        let text = fs::read_to_string(path)?;
        self.notify(
            "textDocument/didOpen",
            json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": self.language_id(path),
                    "version": 1,
                    "text": text
                }
            }),
        )?;
        self.versions.lock().unwrap().insert(uri.clone(), 1);
        Ok(uri)
    }

    fn mapped_language(&self, path: &Path) -> Option<String> {
        path.extension()
            .and_then(|e| e.to_str())
            .and_then(|e| self.transport.extension_to_language().get(&format!(".{e}")))
            .cloned()
    }

    fn language_id(&self, path: &Path) -> String {
        self.mapped_language(path)
            .unwrap_or_else(|| "plaintext".to_string())
    }

    fn notify(&self, method: &str, params: Value) -> Result<(), Box<dyn Error>> {
        self.transport
            .call("notify", &json!({ "method": method, "params": params }))?;
        Ok(())
    }

    fn request(&self, method: &str, params: Value) -> Result<Value, Box<dyn Error>> {
        let response = self
            .transport
            .call("request", &json!({ "method": method, "params": params }))?;
        // Accept either the bare LSP result or a JSON-RPC style envelope.
        if let Some(obj) = response.as_object() {
            if let Some(error) = obj.get("error").filter(|e| !e.is_null()) {
                return Err(format!("language server error: {}", error).into());
            }
            if let Some(result) = obj.get("result") {
                if !obj.contains_key("uri") && !obj.contains_key("targetUri") {
                    return Ok(result.clone());
                }
            }
        }
        Ok(response)
    }

    /// Resolves a path against the working directory and normalizes it,
    /// so `checkout.py` and `./checkout.py` name the same document.
    fn absolute(&self, file_path: &str) -> PathBuf {
        normalize(&self.root.join(file_path))
    }
}

/// The path of a file URI relative to `root`, or absolute if outside it.
fn relative(root: &Path, uri: &str) -> String {
    let path = uri_to_path(uri);
    path.strip_prefix(root)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string_lossy().into_owned())
}

/// `file:line:character` with one-based line and character.
fn display_location(root: &Path, location: &Location) -> String {
    format!(
        "{}:{}:{}",
        relative(root, &location.uri),
        location.line + 1,
        location.character + 1
    )
}

/// Lists every location, grouped by file in the order the server
/// returned them.
fn format_references(root: &Path, locations: &[Location]) -> String {
    if locations.is_empty() {
        return "No references found.".to_string();
    }

    let mut files: Vec<(String, Vec<&Location>)> = Vec::new();
    for location in locations {
        let file = relative(root, &location.uri);
        match files.iter_mut().find(|(f, _)| *f == file) {
            Some((_, group)) => group.push(location),
            None => files.push((file, vec![location])),
        }
    }

    let mut out = format!(
        "Found {} reference{} across {} file{}:\n",
        locations.len(),
        if locations.len() == 1 { "" } else { "s" },
        files.len(),
        if files.len() == 1 { "" } else { "s" },
    );
    for (file, group) in files {
        out.push_str(&format!("\n{}:\n", file));
        for location in group {
            out.push_str(&format!(
                "  Line {}:{}\n",
                location.line + 1,
                location.character + 1
            ));
        }
    }
    out.trim_end().to_string()
}

/// The `<new-diagnostics>` block appended to a tool result, with
/// relative paths and one-based positions.
fn format_diagnostics(root: &Path, files: &[(String, Vec<Diagnostic>)]) -> String {
    let sections: Vec<String> = files
        .iter()
        .map(|(uri, issues)| {
            let lines: Vec<String> = issues
                .iter()
                .map(|d| format!("  [Line {}:{}] {}", d.line + 1, d.character + 1, d.message))
                .collect();
            format!("{}:\n{}", relative(root, uri), lines.join("\n"))
        })
        .collect();
    format!(
        "<new-diagnostics>The following new diagnostic issues were detected:\n\n{}</new-diagnostics>",
        sections.join("\n\n")
    )
}

fn one_based(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_f64().map(|f| f as u64))
        .or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()))
        .filter(|&n| n >= 1)
}

/// Normalizes a `Location`, `Location[]`, `LocationLink[]` or null result.
fn parse_locations(result: &Value) -> Vec<Location> {
    let items: Vec<&Value> = match result {
        Value::Array(items) => items.iter().collect(),
        Value::Object(_) => vec![result],
        _ => Vec::new(),
    };

    items
        .into_iter()
        .filter_map(|item| {
            let uri = item["uri"].as_str().or_else(|| item["targetUri"].as_str())?;
            let range = if item["range"].is_object() {
                &item["range"]
            } else {
                &item["targetSelectionRange"]
            };
            Some(Location {
                uri: uri.to_string(),
                line: range["start"]["line"].as_u64()?,
                character: range["start"]["character"].as_u64()?,
            })
        })
        .collect()
}

/// Removes `.` and `..` components without touching the filesystem.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

pub fn path_to_uri(path: &Path) -> String {
    let mut uri = String::from("file://");
    for byte in path.to_string_lossy().bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                uri.push(byte as char)
            }
            _ => uri.push_str(&format!("%{:02X}", byte)),
        }
    }
    uri
}

pub fn uri_to_path(uri: &str) -> PathBuf {
    let encoded = uri.strip_prefix("file://").unwrap_or(uri);
    let bytes = encoded.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&encoded[i + 1..i + 3], 16) {
                decoded.push(byte);
                i += 3;
                continue;
            }
        }
        decoded.push(bytes[i]);
        i += 1;
    }
    PathBuf::from(String::from_utf8_lossy(&decoded).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TempDir;
    use std::collections::VecDeque;
    use std::sync::Arc;

    /// Records every call, answers requests from a queue of canned
    /// results, and serves a notification queue with a cursor.
    #[derive(Clone, Default)]
    struct FakeTransport {
        calls: Arc<Mutex<Vec<(String, Value)>>>,
        results: Arc<Mutex<VecDeque<Value>>>,
        notifications: Arc<Mutex<Vec<Value>>>,
        languages: HashMap<String, String>,
    }

    impl Transport for FakeTransport {
        fn call(&self, action: &str, payload: &Value) -> Result<Value, Box<dyn Error>> {
            self.calls.lock().unwrap().push((action.to_string(), payload.clone()));
            match action {
                "request" => Ok(self.results.lock().unwrap().pop_front().unwrap_or(Value::Null)),
                "notifications" => {
                    let queue = self.notifications.lock().unwrap();
                    let after = (payload["after"].as_u64().unwrap() as usize).min(queue.len());
                    Ok(json!({ "cursor": queue.len(), "notifications": queue[after..] }))
                }
                _ => Ok(json!({})),
            }
        }

        fn extension_to_language(&self) -> &HashMap<String, String> {
            &self.languages
        }
    }

    impl FakeTransport {
        /// `(method, params)` of every notify and request, in order.
        fn sent(&self) -> Vec<(String, Value)> {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .filter(|(action, _)| action == "notify" || action == "request")
                .map(|(_, p)| (p["method"].as_str().unwrap().to_string(), p["params"].clone()))
                .collect()
        }

        fn methods(&self) -> Vec<String> {
            self.sent().into_iter().map(|(m, _)| m).collect()
        }

        fn publish(&self, uri: &str, version: Option<i64>, issues: &[(u64, u64, &str)]) {
            let diagnostics: Vec<Value> = issues
                .iter()
                .map(|(line, character, message)| {
                    json!({
                        "range": {
                            "start": { "line": line, "character": character },
                            "end": { "line": line, "character": character + 1 }
                        },
                        "severity": 1,
                        "message": message
                    })
                })
                .collect();
            let mut params = json!({ "uri": uri, "diagnostics": diagnostics });
            if let Some(version) = version {
                params["version"] = json!(version);
            }
            self.notifications.lock().unwrap().push(json!({
                "method": "textDocument/publishDiagnostics",
                "params": params
            }));
        }
    }

    const CHECKOUT: &str = "from pricing import total as invoice_total\n\n\ndef checkout_total() -> int:\n    return invoice_total(250, 2)\n";
    const PRICING: &str = "def total(price: int, quantity: int) -> int:\n    return price * quantity\n";

    /// An `Lsp` over a fake transport, rooted in a fresh project folder
    /// holding checkout.py, pricing.py and notes.txt.
    fn setup() -> (Lsp, FakeTransport, TempDir) {
        let root = TempDir::new("lsp");
        root.write("checkout.py", CHECKOUT);
        root.write("pricing.py", PRICING);
        root.write("notes.txt", "notes");

        let fake = FakeTransport {
            languages: HashMap::from([(".py".to_string(), "python".to_string())]),
            ..Default::default()
        };
        (Lsp::new(fake.clone(), root.to_path_buf()), fake, root)
    }

    fn uri(root: &Path, file: &str) -> String {
        path_to_uri(&root.join(file))
    }

    fn location(root: &Path, file: &str, line: u64, character: u64) -> Value {
        json!({
            "uri": uri(root, file),
            "range": {
                "start": { "line": line, "character": character },
                "end": { "line": line, "character": character + 5 }
            }
        })
    }

    fn lookup(operation: &str, file: &str, line: u64, character: u64) -> Value {
        json!({ "operation": operation, "filePath": file, "line": line, "character": character })
    }

    #[test]
    fn definition_opens_the_document_once_and_converts_positions() {
        let (lsp, fake, root) = setup();
        fake.results.lock().unwrap().extend([
            location(&root, "pricing.py", 0, 4),
            location(&root, "pricing.py", 0, 4),
        ]);

        let first = lsp.execute(&lookup("goToDefinition", "checkout.py", 5, 12));
        let second = lsp.execute(&lookup("goToDefinition", "./checkout.py", 5, 12));
        assert_eq!(first, "Defined in pricing.py:1:5");
        assert_eq!(second, first);

        let sent = fake.sent();
        assert_eq!(
            fake.methods(),
            ["textDocument/didOpen", "textDocument/definition", "textDocument/definition"]
        );
        assert_eq!(
            sent[0].1["textDocument"],
            json!({
                "uri": uri(&root, "checkout.py"),
                "languageId": "python",
                "version": 1,
                "text": CHECKOUT
            })
        );
        for (_, params) in &sent[1..] {
            assert_eq!(params["textDocument"]["uri"], uri(&root, "checkout.py"));
            assert_eq!(params["position"], json!({ "line": 4, "character": 11 }));
        }
    }

    #[test]
    fn references_are_listed_by_file_with_one_based_positions() {
        let (lsp, fake, root) = setup();
        fake.results.lock().unwrap().extend([
            json!([
                location(&root, "pricing.py", 0, 4),
                location(&root, "reports.py", 4, 11),
                location(&root, "reports.py", 4, 27),
            ]),
            json!([]),
        ]);

        assert_eq!(
            lsp.execute(&lookup("findReferences", "pricing.py", 1, 5)),
            "Found 3 references across 2 files:\n\n\
             pricing.py:\n  Line 1:5\n\n\
             reports.py:\n  Line 5:12\n  Line 5:28"
        );
        assert_eq!(
            lsp.execute(&lookup("findReferences", "pricing.py", 1, 5)),
            "No references found."
        );

        let (method, params) = &fake.sent()[1];
        assert_eq!(method, "textDocument/references");
        assert_eq!(params["position"], json!({ "line": 0, "character": 4 }));
        assert_eq!(params["context"], json!({ "includeDeclaration": true }));
    }

    #[test]
    fn writes_open_or_change_the_document_with_increasing_versions() {
        let (lsp, fake, root) = setup();

        lsp.sync_write("checkout.py", "v1").unwrap();
        lsp.sync_write("./checkout.py", "v2").unwrap();
        lsp.sync_write("checkout.py", "v3\n").unwrap();
        lsp.sync_write("notes.txt", "not a mapped language").unwrap();

        let sent = fake.sent();
        assert_eq!(
            fake.methods(),
            ["textDocument/didOpen", "textDocument/didChange", "textDocument/didChange"]
        );
        assert_eq!(sent[0].1["textDocument"]["version"], 1);
        assert_eq!(sent[0].1["textDocument"]["text"], "v1");
        assert_eq!(
            sent[2].1,
            json!({
                "textDocument": { "uri": uri(&root, "checkout.py"), "version": 3 },
                "contentChanges": [{ "text": "v3\n" }]
            })
        );
    }

    #[test]
    fn writes_continue_the_version_of_a_document_opened_by_navigation() {
        let (lsp, fake, _root) = setup();
        lsp.execute(&lookup("goToDefinition", "pricing.py", 1, 5));
        lsp.sync_write("pricing.py", PRICING).unwrap();

        let sent = fake.sent();
        assert_eq!(sent[2].0, "textDocument/didChange");
        assert_eq!(sent[2].1["textDocument"]["version"], 2);
    }

    #[test]
    fn diagnostics_skip_stale_reports_and_repeat_nothing() {
        let a = "file:///p/a.py".to_string();
        let b = "file:///p/b.py".to_string();
        let versions = HashMap::from([(a.clone(), 2)]);
        let report = |uri: &str, version: Option<i64>, issues: &[&str]| {
            let fake = FakeTransport::default();
            let issues: Vec<(u64, u64, &str)> = issues.iter().map(|m| (0, 0, *m)).collect();
            fake.publish(uri, version, &issues);
            fake.notifications.lock().unwrap().pop().unwrap()
        };
        let mut diagnostics = Diagnostics::default();

        // A report for an older version is ignored; current and
        // unversioned ones are accepted.
        diagnostics.ingest(
            &[
                report(&a, Some(1), &["stale"]),
                report(&a, Some(2), &["current"]),
                report(&b, None, &["unversioned"]),
            ],
            &versions,
        );
        let taken = diagnostics.take();
        let messages: Vec<&str> = taken
            .iter()
            .flat_map(|(_, issues)| issues.iter().map(|d| d.message.as_str()))
            .collect();
        assert_eq!(messages, ["current", "unversioned"]);

        // The same issues again are not new.
        diagnostics.ingest(&[report(&a, Some(2), &["current"])], &versions);
        assert!(diagnostics.take().is_empty());

        // Clearing a file and reporting the issue again shows it again.
        diagnostics.ingest(&[report(&a, Some(2), &[])], &versions);
        diagnostics.ingest(&[report(&a, Some(2), &["current"])], &versions);
        assert_eq!(diagnostics.take().len(), 1);

        // An issue cleared before the model saw it is still delivered.
        diagnostics.ingest(
            &[report(&a, Some(2), &["brief"]), report(&a, Some(2), &[])],
            &versions,
        );
        assert_eq!(diagnostics.take()[0].1[0].message, "brief");
    }

    #[test]
    fn reports_queued_before_a_write_are_judged_by_their_own_version() {
        let (lsp, fake, root) = setup();
        lsp.execute(&lookup("goToDefinition", "checkout.py", 5, 12));

        // Published for version 1 while the model was busy, then the
        // model rewrites checkout.py, moving it to version 2.
        fake.publish(
            &uri(&root, "checkout.py"),
            Some(1),
            &[(4, 11, "Type Price is not assignable to return type int")],
        );
        lsp.sync_write("checkout.py", CHECKOUT).unwrap();
        lsp.poll_diagnostics().unwrap();

        assert_eq!(
            lsp.take_new_diagnostics().unwrap(),
            "<new-diagnostics>The following new diagnostic issues were detected:\n\n\
             checkout.py:\n  [Line 5:12] Type Price is not assignable to return type int\
             </new-diagnostics>"
        );
        assert_eq!(lsp.take_new_diagnostics(), None);

        // The Write settled the queue first, reading with a wait until a
        // read came back empty; each read continues from the previous
        // cursor, and the plain poll afterwards doesn't wait.
        let reads: Vec<(Value, Value)> = fake
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter(|(action, _)| action == "notifications")
            .map(|(_, p)| (p["after"].clone(), p["timeoutMs"].clone()))
            .collect();
        assert_eq!(
            reads,
            [
                (json!(0), json!(SETTLE_WAIT_MS)),
                (json!(1), json!(SETTLE_WAIT_MS)),
                (json!(1), json!(0)),
            ]
        );
    }

    #[test]
    fn uris_round_trip_with_special_characters() {
        let path = Path::new("/project/cart notes.py");
        let uri = path_to_uri(path);
        assert_eq!(uri, "file:///project/cart%20notes.py");
        assert_eq!(uri_to_path(&uri), path);
    }

    #[test]
    fn normalizes_dot_components() {
        assert_eq!(
            normalize(&Path::new("/project").join("./app/../checkout.py")),
            Path::new("/project/checkout.py")
        );
        assert_eq!(normalize(&Path::new("/project").join("/abs/x.py")), Path::new("/abs/x.py"));
    }

    #[test]
    fn parses_location_shapes() {
        let single = json!({
            "uri": "file:///p/pricing.py",
            "range": { "start": { "line": 0, "character": 4 }, "end": { "line": 0, "character": 9 } }
        });
        let link = json!([{
            "targetUri": "file:///p/pricing.py",
            "targetRange": { "start": { "line": 0, "character": 0 } },
            "targetSelectionRange": { "start": { "line": 0, "character": 4 } }
        }]);
        for value in [single, link] {
            let locations = parse_locations(&value);
            assert_eq!(locations.len(), 1);
            assert_eq!((locations[0].line, locations[0].character), (0, 4));
        }
        assert!(parse_locations(&Value::Null).is_empty());
    }
}
