# LLM CLI Chatbot with RAG

A local LLM chatbot that runs in your terminal with Retrieval-Augmented Generation (RAG) support. Powered by Ollama for local models and optionally by cloud APIs (xAI Grok, Groq).

## Features

- **Local models** via Ollama (Qwen, LLaMA, DeepSeek, and more)
- **Cloud API** support via xAI (Grok) and Groq
- **RAG pipeline** — index TXT/Markdown docs and get context-aware answers
- **Streaming responses** with a clean Rich terminal UI
- **Token count** displayed after each response (prompt / completion / total)
- **Slash commands** for in-chat control

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) (for local models)

## Installation

```bash
# Install Ollama (macOS)
brew install ollama
brew services start ollama

# Install Python dependencies
pip install -r requirements.txt

# Copy env template (optional, for cloud APIs)
cp .env.example .env
```

## Usage

### Pull a model

```bash
python main.py models pull llama3.2:3b
```

### Start chat

```bash
# Interactive model selection
python main.py chat

# Specific model
python main.py chat --model llama3.2:3b

# With RAG enabled
python main.py chat --rag

# Cloud API (requires API key in .env)
python main.py chat --backend api --api-provider xai --model grok-2-1212
```

### RAG — index documents

```bash
# Index a single file
python main.py rag add ./docs/guide.md

# Index a directory
python main.py rag add ./docs/

# Check status
python main.py rag status

# Clear index
python main.py rag clear
```

### List installed models

```bash
python main.py models list
```

## In-chat slash commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/quit` | Exit |
| `/clear` | Clear conversation history |
| `/system <prompt>` | Set a new system prompt mid-chat |
| `/system` | Show the current system prompt |
| `/rag add <path>` | Index a document while chatting |
| `/rag status` | Show RAG store stats |
| `/model` | Show current model info |
| `/model <name>` | Switch to a different model mid-chat |
| `/history save <name>` | Save conversation to `histories/<name>.json` |
| `/history load <name>` | Load a saved conversation |
| `/history list` | List all saved conversations |
| `/export [name]` | Export conversation as Markdown to `exports/` |
| `/multiline` | Toggle multiline input mode (blank line to submit) |
| `/retry` | Regenerate the last assistant response |
| `/undo` | Remove the last user + assistant message pair |

## Supported Models

| Model | Backend | Description |
|-------|---------|-------------|
| `llama3.2:3b` | Ollama | LLaMA 3.2 3B |
| `llama3.1:8b` | Ollama | LLaMA 3.1 8B |
| `qwen2.5:7b` | Ollama | Qwen 2.5 7B |
| `qwen2.5:14b` | Ollama | Qwen 2.5 14B |
| `deepseek-r1:7b` | Ollama | DeepSeek R1 7B |
| `grok-2-1212` | xAI API | Grok 2 (cloud) |
| `llama-3.3-70b-versatile` | Groq API | LLaMA 70B via Groq |

## Response Footer

After each response, a dim status line shows:
- **Token counts + speed**: `tokens: 37 in / 404 out / 441 total @ 26.2 tok/s`
- **RAG chunks** (when enabled): `RAG: 2 chunk(s) | tokens: 512 in / 210 out / 722 total @ 24.5 tok/s`

Speed is measured using Ollama's internal `eval_duration` for accuracy.

## RAG Pipeline

Documents are split into 100-word chunks with 20-word overlap, embedded with `all-MiniLM-L6-v2` (sentence-transformers), and stored in ChromaDB at `.rag_store/`. At query time, chunks with cosine similarity ≥ 0.2 are injected as context into the LLM prompt.

## Project Structure

```
CLI_chatbot/
├── main.py           # CLI entry point (click + Rich)
├── model_manager.py  # Ollama and API backends
├── rag_engine.py     # RAG pipeline (ChromaDB + sentence-transformers)
├── requirements.txt  # Python dependencies
├── .env.example      # API key template
└── setup.sh          # Install script
```
