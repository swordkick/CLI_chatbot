#!/usr/bin/env bash
# setup.sh — Install dependencies for the LLM CLI Chatbot

set -e

echo "=== LLM CLI Chatbot Setup ==="
echo

# ---- 1. Check Ollama -------------------------------------------------------
if command -v ollama &>/dev/null; then
    echo "[ok] Ollama is already installed: $(ollama --version 2>&1 | head -1)"
else
    echo "[!] Ollama is not installed."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "    Install with: brew install ollama"
        echo "    Or download from: https://ollama.com/download"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "    Install with: curl -fsSL https://ollama.com/install.sh | sh"
    else
        echo "    Visit https://ollama.com/download for instructions."
    fi
    read -r -p "Continue without Ollama? (API backend will still work) [y/N]: " cont
    [[ "$cont" =~ ^[Yy]$ ]] || exit 1
fi

echo

# ---- 2. Python dependencies ------------------------------------------------
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "[ok] Dependencies installed."
echo

# ---- 3. Copy .env.example --------------------------------------------------
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    echo "[ok] Created .env from .env.example. Edit it to add API keys."
else
    echo "[ok] .env already exists."
fi
echo

# ---- 4. Optional: pull a starter model ------------------------------------
read -r -p "Pull a starter model now? [y/N]: " pull_model
if [[ "$pull_model" =~ ^[Yy]$ ]]; then
    echo "Suggested models:"
    echo "  1) llama3.2:3b   (small, fast)"
    echo "  2) qwen2.5:7b    (balanced)"
    echo "  3) llama3.1:8b   (capable)"
    read -r -p "Enter model name (or leave blank to skip): " model_name
    if [[ -n "$model_name" ]]; then
        python main.py models pull "$model_name"
    fi
fi

echo
echo "=== Setup complete ==="
echo "Run: python main.py chat"
