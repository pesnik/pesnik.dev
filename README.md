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

## Custom AI Agent

Fine-tune a TinyLlama 1.1B model on your portfolio so the terminal chat runs a custom model instead of OpenRouter.

### Pipeline

```
scripts/prepare_dataset.py   →   data/train.jsonl
scripts/train_tinyllama.py   →   pesnik-tinyllama-lora/ (LoRA adapters)
scripts/train_tinyllama.ipynb     ← Colab notebook (one-click)
```

### Usage

```bash
# 1. Build training data from portfolio content
python scripts/prepare_dataset.py --output data/train.jsonl

# 2. Upload to Colab and run the notebook, OR run locally:
python scripts/train_tinyllama.py --dataset data/train.jsonl

# 3. Push to HuggingFace Hub
python scripts/train_tinyllama.py --hf-token YOUR_TOKEN --push-to-hub
```

Training runs on Colab free T4 (~20 min, $0). Uses QLoRA (4-bit) with LoRA rank 32. Produces LoRA adapters (~6MB) that can be merged or loaded at inference.

### Architecture change for custom model

Swap the endpoint in `worker.js` and `index.html` from OpenRouter to your HuggingFace Inference Endpoint running the fine-tuned model.

## Deploy

```bash
npx wrangler deploy
```
