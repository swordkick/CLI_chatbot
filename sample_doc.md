# CLI Chatbot User Guide

## Overview
This is a local LLM chatbot with RAG (Retrieval-Augmented Generation) support.
It runs models locally via Ollama and supports TXT and Markdown documents.

## Supported Models
- **llama3.2:3b** — LLaMA 3.2 3B, fast and lightweight
- **qwen2.5:7b** — Qwen 2.5 7B, balanced performance
- **qwen2.5:14b** — Qwen 2.5 14B, high quality
- **deepseek-r1:7b** — DeepSeek R1 7B, reasoning focused

## Commands

### CLI Commands
- `python main.py chat` — Start an interactive chat session
- `python main.py models list` — List installed models
- `python main.py models pull <name>` — Download a model
- `python main.py rag add <path>` — Index a document or directory
- `python main.py rag status` — Show indexed document stats
- `python main.py rag clear` — Clear all indexed documents

### Slash Commands (inside chat)
- `/help` — Show help
- `/quit` — Exit the chat
- `/clear` — Clear conversation history
- `/rag add <path>` — Add a document while chatting
- `/rag status` — Check RAG store status
- `/model` — Show current model info

## RAG Pipeline
Documents are chunked into 100-word segments with 20-word overlap.
Embeddings are generated using the `all-MiniLM-L6-v2` model from sentence-transformers.
Vectors are stored in ChromaDB under the `.rag_store/` directory.
Only chunks with cosine similarity above 0.2 are injected as context.

## API Backend
Set environment variables in `.env` to use cloud models:
- `XAI_API_KEY` — for Grok via xAI
- `GROQ_API_KEY` — for LLaMA 70B via Groq
