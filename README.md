# pesnik.dev

Personal portfolio website — a single-page landing for **pesnik**, a Data Engineer & entrepreneur from Bangladesh.

Features a terminal-themed interactive chat TUI that doubles as a portfolio browser. Live at [pesnik.dev](https://pesnik.dev).

## About

- **Role**: Data Engineer
- **Location**: Bangladesh 🇧🇩
- **Spirit animal**: 🦫 The Beaver
- **Stack**: Rust, Zig, Go, Python
- **Building**: Calf LLM (AI for institutional memory)

## Projects

| Project | Language | Description |
|---------|----------|-------------|
| Zygote | Zig + Java | JDBC to ODBC bridge |
| Spotlight Search | Rust | Windows file/application launcher |
| Superset Playground | Python/Docker/K8s | Apache Superset deep-dive |
| AoC in Zig | Zig | Advent of Code solutions |
| MorseChat | Android/Kotlin | Morse-code real-time chat |
| Gin Repertoire | Go | Gin framework collection |

## Ventures

- **Calf** — LLM for institutional memory
- **গুদাম (Gudam)** — Lightweight data warehouse in Rust
- **PlugNest** — Go plugin ecosystem
- **একের ভিতর সব** — STEM EdTech platform

## Terminal TUI Chat

The hero section features an interactive terminal. Type commands or ask anything:

```
~ whoami           — identity
~ ls               — root directory
~ ls projects      — open-source projects
~ ls ventures      — ventures
~ cat whoami.json  — identity as JSON
~ help             — all commands
```

The chat supports markdown rendering (`**bold**`, `` `code` ``, fenced code blocks, links) and uses OpenRouter AI through a Cloudflare Worker proxy.

## Architecture

```
index.html  ──→  worker.js (proxy)  ──→  OpenWebUI / OpenRouter API
                      │
            Cloudflare Workers + Assets
```

API key stays server-side in `worker.js` via `OPENWEBUI_API_KEY` secret.

## Stack

- Static single `index.html` (no build step)
- Cloudflare Workers for the chat proxy
- Cloudflare Pages Assets for static serving

## Deploy

```bash
npx wrangler deploy
```
