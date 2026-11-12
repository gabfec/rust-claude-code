use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Where skills live, relative to the project directory.
pub const SKILLS_DIR: &str = ".claude/skills";

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
    /// Folder holding SKILL.md and bundled files, as given to `discover`
    /// (project-relative in normal use).
    pub dir: String,
    /// `context: fork` — run the skill in a subagent.
    pub fork: bool,
}

impl Skill {
    fn load(dir: &Path) -> Option<Self> {
        let folder_name = dir.file_name()?.to_str()?.to_string();
        let text = fs::read_to_string(dir.join("SKILL.md")).ok()?;
        let (frontmatter, body) = split_frontmatter(&text);
        let fields = parse_frontmatter(frontmatter);

        Some(Skill {
            // Skills are invoked by folder name, which the standard
            // requires to match the frontmatter name.
            dir: dir.to_string_lossy().into_owned(),
            name: folder_name,
            description: fields.get("description").cloned().unwrap_or_default(),
            body: body.trim().to_string(),
            fork: fields.get("context").is_some_and(|c| c == "fork"),
        })
    }

    /// The body with arguments substituted. When the skill bundles
    /// files next to SKILL.md, the body is prefixed with the skill's
    /// location so the model can resolve paths relative to it.
    ///
    /// Plain skills get the bare body: the header adds nothing for them,
    /// and above a terse instruction the model can mistake it for an
    /// injection attempt and refuse.
    pub fn instructions(&self, arguments: &str) -> String {
        let body = substitute_arguments(&self.body, arguments);
        if !self.has_bundled_files() {
            return body;
        }
        format!(
            "Skill: {} (located at {})\n\
             Paths in the instructions below are relative to that folder.\n\n{}",
            self.name, self.dir, body
        )
    }

    fn has_bundled_files(&self) -> bool {
        fs::read_dir(&self.dir).is_ok_and(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| e.file_name() != "SKILL.md")
        })
    }
}

/// Finds every `<skills_dir>/<name>/SKILL.md`, sorted by name.
pub fn discover(skills_dir: &Path) -> Vec<Skill> {
    let Ok(entries) = fs::read_dir(skills_dir) else {
        return Vec::new();
    };

    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();

    dirs.iter().filter_map(|d| Skill::load(d)).collect()
}

/// Expands the leading `/name` tokens of a prompt into skill bodies.
///
/// Every leading token that names a known skill is expanded; the first
/// token that doesn't ends the run, and it and everything after it
/// become the argument text for each expanded body. Returns an empty
/// list when the prompt doesn't start with a known skill name.
pub fn expand_prompt(skills: &[Skill], prompt: &str) -> Vec<String> {
    let mut invoked = Vec::new();
    let mut rest = prompt.trim_start();

    loop {
        let (token, tail) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        let Some(skill) = token
            .strip_prefix('/')
            .and_then(|name| skills.iter().find(|s| s.name == name))
        else {
            break;
        };
        // A skill that asks for a subagent is never stacked with others.
        if skill.fork && !invoked.is_empty() {
            break;
        }
        invoked.push(skill);
        rest = tail.trim_start();
        if skill.fork {
            break;
        }
    }

    let arguments = rest.trim_end();
    invoked
        .iter()
        .map(|skill| skill.instructions(arguments))
        .collect()
}

/// Replaces `$ARGUMENTS`, `$ARGUMENTS[N]` and `$N` in a skill body.
/// Missing positional arguments become empty strings.
pub fn substitute_arguments(body: &str, arguments: &str) -> String {
    let args: Vec<&str> = arguments.split_whitespace().collect();
    let positional = |index: &str| -> &str {
        index
            .parse::<usize>()
            .ok()
            .and_then(|i| args.get(i).copied())
            .unwrap_or("")
    };

    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    while let Some(pos) = rest.find('$') {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + 1..];

        if let Some(tail) = after.strip_prefix("ARGUMENTS") {
            if let Some(indexed) = tail.strip_prefix('[') {
                if let Some(close) = indexed.find(']') {
                    let index = &indexed[..close];
                    if !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()) {
                        out.push_str(positional(index));
                        rest = &indexed[close + 1..];
                        continue;
                    }
                }
            }
            out.push_str(arguments);
            rest = tail;
            continue;
        }

        let digits = after.bytes().take_while(|b| b.is_ascii_digit()).count();
        if digits > 0 {
            out.push_str(positional(&after[..digits]));
            rest = &after[digits..];
            continue;
        }

        out.push('$');
        rest = after;
    }
    out.push_str(rest);
    out
}

pub fn system_prompt(skills: &[Skill]) -> String {
    let mut prompt = String::from("You have access to the following skills:\n\n");
    for skill in skills {
        prompt.push_str(&format!("- {}: {}\n", skill.name, skill.description));
    }
    prompt.push_str(
        "\nIf a skill matches the user's request, call the Skill tool with its name\n\
         and follow the instructions it returns.\n",
    );
    prompt
}

/// Splits a SKILL.md into its frontmatter (between the `---` markers)
/// and its body. A file without frontmatter is all body.
fn split_frontmatter(text: &str) -> (&str, &str) {
    let mut lines = text.split_inclusive('\n');
    match lines.next() {
        Some(first) if first.trim_end() == "---" => {}
        _ => return ("", text),
    }

    let start = text.find('\n').map(|i| i + 1).unwrap_or(text.len());
    let mut offset = start;
    for line in lines {
        if line.trim_end() == "---" {
            return (&text[start..offset], &text[offset + line.len()..]);
        }
        offset += line.len();
    }
    ("", text)
}

/// Parses the subset of YAML used by skill frontmatter: top-level
/// `key: value` pairs with optional quoting, and `|` / `>` block scalars.
fn parse_frontmatter(text: &str) -> HashMap<String, String> {
    let mut fields = HashMap::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        i += 1;

        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        if line.starts_with([' ', '\t']) {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_string();
        let value = value.trim();

        if value == "|" || value == ">" || value == "|-" || value == ">-" {
            let mut block = Vec::new();
            while i < lines.len()
                && (lines[i].trim().is_empty() || lines[i].starts_with([' ', '\t']))
            {
                block.push(lines[i].trim());
                i += 1;
            }
            let sep = if value.starts_with('|') { "\n" } else { " " };
            fields.insert(key, block.join(sep).trim().to_string());
        } else {
            fields.insert(key, unquote(value));
        }
    }

    fields
}

fn unquote(value: &str) -> String {
    let value = strip_comment(value);
    if value.len() >= 2 {
        if let Some(inner) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
            return inner.replace("\\\"", "\"").replace("\\\\", "\\");
        }
        if let Some(inner) = value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')) {
            return inner.replace("''", "'");
        }
    }
    value.to_string()
}

fn strip_comment(value: &str) -> &str {
    if value.starts_with(['"', '\'']) {
        return value;
    }
    match value.find(" #") {
        Some(i) => value[..i].trim_end(),
        None => value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitutes_all_placeholder_kinds() {
        let body = "Deploy to $0 in region $1. Full request was: $ARGUMENTS ($ARGUMENTS[1])";
        assert_eq!(
            substitute_arguments(body, "staging eu-west"),
            "Deploy to staging in region eu-west. Full request was: staging eu-west (eu-west)"
        );
    }

    #[test]
    fn missing_arguments_become_empty() {
        assert_eq!(substitute_arguments("[$ARGUMENTS][$0][$ARGUMENTS[3]]", ""), "[][][]");
        assert_eq!(substitute_arguments("costs $ 5", "x"), "costs $ 5");
    }

    fn skill(name: &str, body: &str) -> Skill {
        Skill {
            name: name.into(),
            description: String::new(),
            body: body.into(),
            dir: format!("{SKILLS_DIR}/{name}"),
            fork: false,
        }
    }

    #[test]
    fn stacks_leading_skills() {
        let skills = [skill("apple", "a-$ARGUMENTS"), skill("grape", "g-$ARGUMENTS")];
        let bodies = |prompt| -> Vec<String> {
            expand_prompt(&skills, prompt)
                .iter()
                .map(|m| m.rsplit("\n\n").next().unwrap().to_string())
                .collect()
        };
        assert_eq!(bodies("/apple /grape 4127"), ["a-4127", "g-4127"]);
        assert_eq!(bodies("/apple 4127 /grape"), ["a-4127 /grape"]);
        assert_eq!(bodies("/apple /pear 4127"), ["a-/pear 4127"]);
        assert_eq!(bodies("/grape"), ["g-"]);
        assert!(expand_prompt(&skills, "hello /apple").is_empty());
    }

    #[test]
    fn instructions_name_the_folder_only_with_bundled_files() {
        let dir = std::env::temp_dir().join(format!("skill-test-{}", std::process::id()));
        fs::create_dir_all(dir.join("scripts")).unwrap();
        fs::write(dir.join("SKILL.md"), "---\nname: apple\n---\n").unwrap();

        let mut apple = skill("apple", "Run `scripts/x.sh`.");
        apple.dir = dir.to_string_lossy().into_owned();
        assert_eq!(
            apple.instructions(""),
            format!(
                "Skill: apple (located at {})\n\
                 Paths in the instructions below are relative to that folder.\n\n\
                 Run `scripts/x.sh`.",
                apple.dir
            )
        );

        fs::remove_dir_all(dir.join("scripts")).unwrap();
        assert_eq!(apple.instructions(""), "Run `scripts/x.sh`.");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn splits_frontmatter_from_body() {
        let text = "---\nname: apple\ndescription: \"Deploys: apple\"\ncontext: fork\n---\n\nBody here\n";
        let (fm, body) = split_frontmatter(text);
        let fields = parse_frontmatter(fm);
        assert_eq!(fields["description"], "Deploys: apple");
        assert_eq!(fields["context"], "fork");
        assert_eq!(body.trim(), "Body here");
    }
}
