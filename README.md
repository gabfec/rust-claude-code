# Rust Claude Code

[![Tests](https://github.com/gabfec/rust-claude-code/actions/workflows/rust.yml/badge.svg)](https://github.com/gabfec/rust-claude-code/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Made with Rust](https://img.shields.io/badge/Made%20with-Rust-red.svg)]()
[![Rust](https://img.shields.io/badge/rust-1.93+-orange.svg)](https://www.rust-lang.org)

An AI coding agent in the style of Claude Code, written in Rust. It sends
your prompt to an LLM over an OpenAI-compatible API, runs the tools the
model asks for, feeds the results back, and loops until the model answers.

It implements:

- The agent loop, with tool calling over the chat completions API
- File and shell tools: `Read`, `Write`, `Bash`
- Skills with progressive disclosure: discovered from `.claude/skills/`,
  advertised by name and description, loaded only when used
- Slash commands with arguments (`/deploy staging eu-west`) and stacking
- Skills that bundle scripts, skills the model picks on its own, and
  skills that run in a subagent with a conversation of their own
- An `LSP` tool backed by a language server: go to definition, find
  references, edits kept in sync, and diagnostics fed back to the model

---

## Features

### Tools

| Tool | Description |
|------|-------------|
`Read` | return the contents of a file
`Write` | write a file, replacing it (and sync it to the language server)
`Bash` | run a shell command and return its stdout and stderr
`Skill` | load a skill's instructions, or run it in a subagent
`LSP` | `goToDefinition` / `findReferences` at a 1-based line and character (only when a language server is connected)

---

### Skills

A skill is a folder under `.claude/skills/` holding a `SKILL.md`:

```markdown
---
name: deploy
description: Use this skill when the user asks to deploy a service.
---

Deploy to $0 in region $1. Full request was: $ARGUMENTS
```

| Feature | How it works |
|---------|--------------|
Advertising | names and descriptions go into the system prompt; bodies stay on disk
Slash commands | `/deploy staging eu-west` sends the substituted body instead of the prompt
Arguments | `$ARGUMENTS` (all), `$ARGUMENTS[N]` and `$N` (one, zero-based)
Stacking | `/lint /test src` expands every leading skill name, sharing the arguments
Bundled files | a skill folder with scripts is named so relative paths resolve
Model-invoked | the model calls the `Skill` tool when a description matches the request
Subagents | `context: fork` runs the body in a fresh conversation; only the answer comes back

---

### Language server

When `CODECRAFTERS_LSP_URL` points at a language server bridge, the agent:

- advertises the `LSP` tool and opens each document with `didOpen` on first use
- answers `goToDefinition` and `findReferences` with relative paths and
  1-based positions
- sends every `Write` to the server (`didChange` with the next version,
  or `didOpen` for a file it hasn't seen)
- reads `publishDiagnostics` reports, ignores those for an older version,
  and appends new issues to the latest tool result so the model can fix them

---

### Configuration

| Variable | Description |
|----------|-------------|
`OPENROUTER_API_KEY` | API key (required)
`OPENROUTER_BASE_URL` | OpenAI-compatible endpoint (default `https://openrouter.ai/api/v1`)
`CODECRAFTERS_LSP_URL` | language server bridge (optional)

The model is `anthropic/claude-haiku-4.5`.

---

## Example usage

```bash
export OPENROUTER_API_KEY=...
./your_program.sh -p "Read README.md and create the file it describes."
```

Invoke a skill directly:

```bash
./your_program.sh -p "/deploy staging eu-west"
```

The answer goes to stdout; logs such as `LSP client connected` go to stderr.

---

## Project Structure

```
src/
├── main.rs          agent loop, tools and subagents
├── skills.rs        skill discovery, frontmatter, slash commands, arguments
├── lsp.rs           LSP tool: navigation, document sync, diagnostics
├── lsp_client.rs    course-supplied transport to the language server bridge
└── test_util.rs     temporary directories for tests
tests/
└── cli.rs           end-to-end tests against a mock model server
```

---

## Tests

Run the tests:

```bash
cargo test
```

Unit tests sit next to the code they cover; the LSP tests use a fake
transport that records every call. `tests/cli.rs` runs the built agent
against a mock OpenAI-compatible server that answers with scripted replies,
so no API key or network is needed.

## Educational Purpose

This project was built as part of the CodeCrafters
["Build Your Own Claude Code"](https://codecrafters.io/challenges/claude-code)
challenge, which focuses on implementing real-world systems from scratch.

The goal was to understand how an AI coding agent works internally,
including tool calling and the agent loop, progressive disclosure of
skills, subagents, and language server integration.

https://codecrafters.io

---

## License

MIT License — see the `LICENSE` file.
