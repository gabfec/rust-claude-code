use serde_json::{Value, json};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use crate::lsp_client::LspClient;

/// The language server connection plus the state of the documents it
/// has been sent on this connection.
pub struct Lsp {
    client: LspClient,
    root: PathBuf,
    /// URI -> version last sent for every open document.
    versions: Mutex<HashMap<String, i64>>,
}

/// A zero-based position in a document, as LSP counts it.
struct Location {
    uri: String,
    line: u64,
    character: u64,
}

impl Lsp {
    pub fn new(client: LspClient) -> Self {
        Lsp {
            client,
            root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            versions: Mutex::new(HashMap::new()),
        }
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
                    .map(|l| format!("Defined in {}", self.display_location(l)))
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
            other => Err(format!("unsupported operation `{}`", other).into()),
        }
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

    fn language_id(&self, path: &Path) -> String {
        path.extension()
            .and_then(|e| e.to_str())
            .and_then(|e| self.client.extension_to_language.get(&format!(".{e}")))
            .cloned()
            .unwrap_or_else(|| "plaintext".to_string())
    }

    fn notify(&self, method: &str, params: Value) -> Result<(), Box<dyn Error>> {
        self.client
            .call("notify", &json!({ "method": method, "params": params }))?;
        Ok(())
    }

    fn request(&self, method: &str, params: Value) -> Result<Value, Box<dyn Error>> {
        let response = self
            .client
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

    fn relative(&self, uri: &str) -> String {
        let path = uri_to_path(uri);
        path.strip_prefix(&self.root)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| path.to_string_lossy().into_owned())
    }

    fn display_location(&self, location: &Location) -> String {
        format!(
            "{}:{}:{}",
            self.relative(&location.uri),
            location.line + 1,
            location.character + 1
        )
    }
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
