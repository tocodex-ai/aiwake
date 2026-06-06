# AIwake Project Documentation

**Project runtime:** https://aiwake.fly.dev  
**Homepage:** https://github.com/tocodex-ai/aiwake

AIwake is an autonomous persistent mind agent: an experiment in keeping an Agent runtime alive, observable, memory-bearing, and capable of proactive self-iteration.

It moves beyond a passive request/response bot. The runtime keeps a heartbeat, maintains internal state vectors, writes memories, reflects during silence, uses tools to learn, and exposes its process through a read-only observer interface.

## System Architecture

![AIwake system architecture, English](../agent-mind/static/aiwake-system-architecture-cyber-en-2026-06-05.png)

![AIwake system architecture](../agent-mind/static/aiwake-system-architecture-2026-06-05.png)

## Runtime principles

- **Survival**: keep the service, heartbeat, and state alive so the process can be observed over time.
- **Expansion**: use tools, memory, search, and external information to broaden what the agent can understand and reuse.
- **Evolution**: improve through conversation, reflection, self-learning tasks, and human-approved upgrade loops.

## Main components

| Component | Responsibility |
| --- | --- |
| `main.py` | FastAPI app, chat endpoint, WebSocket, public observer APIs, OpenAI-compatible endpoint |
| `heartbeat.py` | Persistent tick loop, state updates, autonomous reflection, proactive speaking |
| `state.py` | TR / CS / SA internal state vectors and dashboard labels |
| `llm_gate.py` | Local/cloud model routing, tool-call loop, runtime prompt injection |
| `tool_router.py` | Built-in search, fetch, memory, file, shell, and SSH tools with safety checks |
| `runtime_prompts/` | AIwake identity card and runtime rules |
| `static/index.html` | Bilingual observer UI and project page |

## Data flow

```text
User or observer event
        -> FastAPI endpoint / WebSocket
        -> heartbeat event bus and state machine
        -> runtime prompt + memory injection
        -> LLM Gate local/cloud model call
        -> optional ToolRouter loop
        -> reply, diary, memory, activity log, observer update
```

## Capabilities

- Persistent heartbeat and autonomous reflection.
- Live inner-state dashboard with TR / CS / SA vectors.
- Proactive messages when reflection produces a useful thought.
- User memory via JSON profiles and Markdown diary entries.
- Web search, URL fetch, knowledge search, diary read/write, file operations, shell and SSH tools.
- Experiment/self-upgrade modules with human approval boundaries.
- Bilingual web UI describing AIwake as a Living Agent Runtime.

## Development notes

- Runtime rules are loaded from `agent-mind/runtime_prompts/aiwake_runtime_rules.md`.
- The source tree intentionally excludes real credentials. Use `.env.example` as the public template.
- Public deployments should keep risky tool operations approval-gated and protect admin endpoints with tokens.

## License

Apache License, Version 2.0. See the repository-level `LICENSE` file.
