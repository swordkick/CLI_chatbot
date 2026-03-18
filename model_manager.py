"""Model backend manager supporting Ollama (local) and OpenAI-compatible APIs."""

from __future__ import annotations

import os
from typing import Generator, Iterator

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

MODEL_CATALOG: list[dict] = [
    {"name": "qwen2.5:7b",       "backend": "ollama", "description": "Qwen 2.5 7B"},
    {"name": "qwen2.5:14b",      "backend": "ollama", "description": "Qwen 2.5 14B"},
    {"name": "llama3.2:3b",      "backend": "ollama", "description": "LLaMA 3.2 3B"},
    {"name": "llama3.1:8b",      "backend": "ollama", "description": "LLaMA 3.1 8B"},
    {"name": "deepseek-r1:7b",   "backend": "ollama", "description": "DeepSeek R1 7B"},
    {"name": "grok-2-1212",      "backend": "api/xai",  "description": "Grok 2 (cloud)"},
    {"name": "llama-3.3-70b-versatile", "backend": "api/groq", "description": "LLaMA 70B via Groq"},
    {"name": "phi4:14b",                "backend": "ollama", "description": "Microsoft Phi-4 14B"},
]

# Known context window sizes (tokens) — keyed by model name prefix
MODEL_CONTEXT_SIZES: dict[str, int] = {
    "qwen2.5:7b":               32768,
    "qwen2.5:14b":              32768,
    "llama3.2:3b":              131072,
    "llama3.2:latest":          131072,
    "llama3.1:8b":              131072,
    "llama3.1:latest":          131072,
    "llama3.3:latest":          131072,
    "llama2:latest":            4096,
    "deepseek-r1:7b":           65536,
    "grok-2-1212":              131072,
    "llama-3.3-70b-versatile":  131072,
    "mistral:latest":           32768,
    "phi4:14b":                 16384,
}

# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaBackend:
    def __init__(self) -> None:
        import ollama as _ollama
        self._ollama = _ollama

    def list_models(self) -> list[str]:
        """Return names of locally installed Ollama models."""
        try:
            response = self._ollama.list()
            models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
            return [m["name"] if isinstance(m, dict) else m.model for m in models]
        except Exception as exc:
            raise RuntimeError(f"Cannot connect to Ollama: {exc}") from exc

    def pull_model(self, name: str) -> Generator[dict, None, None]:
        """Stream pull progress dicts: {'status': str, 'completed': int, 'total': int}."""
        for chunk in self._ollama.pull(name, stream=True):
            if isinstance(chunk, dict):
                yield chunk
            else:
                yield {"status": getattr(chunk, "status", ""),
                       "completed": getattr(chunk, "completed", 0),
                       "total": getattr(chunk, "total", 0)}

    def generate(self, messages: list[dict], model: str, stream: bool = True) -> Iterator[str | dict]:
        """Yield streamed text tokens, then a final stats dict with token counts."""
        if stream:
            for chunk in self._ollama.chat(model=model, messages=messages, stream=True):
                if isinstance(chunk, dict):
                    content = chunk.get("message", {}).get("content", "")
                    done = chunk.get("done", False)
                else:
                    content = getattr(getattr(chunk, "message", None), "content", "") or ""
                    done = getattr(chunk, "done", False)
                if content:
                    yield content
                if done:
                    if isinstance(chunk, dict):
                        prompt_tokens = chunk.get("prompt_eval_count", 0) or 0
                        completion_tokens = chunk.get("eval_count", 0) or 0
                        eval_duration_ns = chunk.get("eval_duration", 0) or 0
                    else:
                        prompt_tokens = getattr(chunk, "prompt_eval_count", 0) or 0
                        completion_tokens = getattr(chunk, "eval_count", 0) or 0
                        eval_duration_ns = getattr(chunk, "eval_duration", 0) or 0
                    tokens_per_sec = (
                        completion_tokens / (eval_duration_ns / 1e9)
                        if eval_duration_ns > 0 else 0.0
                    )
                    yield {"__tokens__": True, "prompt": prompt_tokens,
                           "completion": completion_tokens, "tokens_per_sec": tokens_per_sec}
        else:
            response = self._ollama.chat(model=model, messages=messages, stream=False)
            if isinstance(response, dict):
                yield response.get("message", {}).get("content", "")
                eval_duration_ns = response.get("eval_duration", 0) or 0
                completion_tokens = response.get("eval_count", 0) or 0
                yield {"__tokens__": True,
                       "prompt": response.get("prompt_eval_count", 0) or 0,
                       "completion": completion_tokens,
                       "tokens_per_sec": completion_tokens / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0}
            else:
                yield getattr(getattr(response, "message", None), "content", "") or ""
                eval_duration_ns = getattr(response, "eval_duration", 0) or 0
                completion_tokens = getattr(response, "eval_count", 0) or 0
                yield {"__tokens__": True,
                       "prompt": getattr(response, "prompt_eval_count", 0) or 0,
                       "completion": completion_tokens,
                       "tokens_per_sec": completion_tokens / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0}


# ---------------------------------------------------------------------------
# API backend (OpenAI-compatible)
# ---------------------------------------------------------------------------

API_CONFIGS: dict[str, dict] = {
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
}


class APIBackend:
    def __init__(self, provider: str = "xai") -> None:
        from openai import OpenAI

        if provider not in API_CONFIGS:
            raise ValueError(f"Unknown API provider: {provider!r}. Choose from {list(API_CONFIGS)}")

        cfg = API_CONFIGS[provider]
        api_key = os.environ.get(cfg["api_key_env"], "")
        if not api_key:
            raise RuntimeError(
                f"Environment variable {cfg['api_key_env']!r} is not set. "
                f"Add it to your .env file."
            )

        self._client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        self.provider = provider

    def generate(self, messages: list[dict], model: str, stream: bool = True) -> Iterator[str]:
        """Yield streamed text tokens from an OpenAI-compatible API."""
        if stream:
            with self._client.chat.completions.create(
                model=model, messages=messages, stream=True
            ) as response:
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield delta
        else:
            response = self._client.chat.completions.create(
                model=model, messages=messages, stream=False
            )
            yield response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Unified model manager
# ---------------------------------------------------------------------------

class ModelManager:
    def __init__(self, backend: str = "ollama", api_provider: str = "xai") -> None:
        self.backend_name = backend
        self.api_provider = api_provider
        self._backend: OllamaBackend | APIBackend | None = None

    def _get_backend(self) -> OllamaBackend | APIBackend:
        if self._backend is None:
            if self.backend_name == "ollama":
                self._backend = OllamaBackend()
            else:
                self._backend = APIBackend(provider=self.api_provider)
        return self._backend

    def list_models(self) -> list[str]:
        backend = self._get_backend()
        if isinstance(backend, OllamaBackend):
            return backend.list_models()
        # API backends don't enumerate models; return catalog entries for that provider
        prefix = f"api/{self.api_provider}"
        return [m["name"] for m in MODEL_CATALOG if m["backend"] == prefix]

    def pull_model(self, name: str) -> Generator[dict, None, None]:
        backend = self._get_backend()
        if not isinstance(backend, OllamaBackend):
            raise RuntimeError("Model pull is only supported for the Ollama backend.")
        return backend.pull_model(name)

    def generate(self, messages: list[dict], model: str, stream: bool = True) -> Iterator[str]:
        return self._get_backend().generate(messages=messages, model=model, stream=stream)
