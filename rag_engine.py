"""RAG pipeline: load TXT/Markdown → chunk → embed → ChromaDB → retrieve."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import NamedTuple

# Suppress noisy logs from transformers / HuggingFace Hub / sentence-transformers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CHROMA_DIR = ".rag_store"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 100       # words per chunk
CHUNK_OVERLAP = 20     # overlap words between consecutive chunks
SIMILARITY_THRESHOLD = 0.2
TOP_K = 5


class Chunk(NamedTuple):
    text: str
    source: str
    index: int


# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------

def _split_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# RAG engine
# ---------------------------------------------------------------------------

class RAGEngine:
    def __init__(self) -> None:
        self._chroma = None
        self._collection = None
        self._embedder = None

    # ---- lazy init --------------------------------------------------------

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=CHROMA_DIR)
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_embedder(self):
        if self._embedder is None:
            import contextlib
            import sys
            from sentence_transformers import SentenceTransformer

            with open(os.devnull, "w") as devnull, \
                 contextlib.redirect_stderr(devnull), \
                 contextlib.redirect_stdout(devnull):
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    # ---- public API -------------------------------------------------------

    def add_document(self, path: str | Path) -> int:
        """Load, chunk, embed and store a TXT or Markdown file. Returns chunk count."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix.lower() not in {".txt", ".md", ".markdown"}:
            raise ValueError(f"Unsupported file type: {p.suffix}. Only .txt and .md are supported.")

        text = p.read_text(encoding="utf-8", errors="replace")
        raw_chunks = _split_words(text, CHUNK_SIZE, CHUNK_OVERLAP)

        collection = self._get_collection()
        embedder = self._get_embedder()

        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict] = []

        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{p.name}::chunk_{i}"
            ids.append(chunk_id)
            texts.append(chunk_text)
            metas.append({"source": str(p), "chunk_index": i})

        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
        return len(raw_chunks)

    def add_directory(self, path: str | Path) -> dict[str, int]:
        """Recursively index all TXT/Markdown files in a directory."""
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            raise NotADirectoryError(f"Not a directory: {p}")

        results: dict[str, int] = {}
        for file in sorted(p.rglob("*")):
            if file.suffix.lower() in {".txt", ".md", ".markdown"} and file.is_file():
                try:
                    count = self.add_document(file)
                    results[str(file)] = count
                except Exception as exc:
                    results[str(file)] = -1  # mark as failed
        return results

    def query(self, text: str, top_k: int = TOP_K) -> list[dict]:
        """Return relevant chunks above SIMILARITY_THRESHOLD."""
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        embedder = self._get_embedder()
        query_embedding = embedder.encode([text], show_progress_bar=False).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[dict] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance: 0 = identical, 1 = orthogonal
            # Convert to similarity: similarity = 1 - distance
            similarity = 1.0 - dist
            if similarity >= SIMILARITY_THRESHOLD:
                chunks.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "similarity": round(similarity, 4),
                })

        return chunks

    def build_context(self, query: str) -> tuple[str, int]:
        """Return (context_string, num_chunks) to inject into the LLM prompt."""
        chunks = self.query(query)
        if not chunks:
            return "", 0

        parts = []
        for i, c in enumerate(chunks, 1):
            source_name = Path(c["source"]).name
            parts.append(f"[{i}] (source: {source_name}, similarity: {c['similarity']})\n{c['text']}")

        context = "Relevant context from documents:\n\n" + "\n\n---\n\n".join(parts)
        return context, len(chunks)

    def status(self) -> dict:
        """Return stats about the current vector store."""
        try:
            collection = self._get_collection()
            count = collection.count()
            # Get unique sources
            if count > 0:
                all_metas = collection.get(include=["metadatas"])["metadatas"]
                sources = sorted({m.get("source", "unknown") for m in all_metas})
            else:
                sources = []
            return {"total_chunks": count, "sources": sources}
        except Exception:
            return {"total_chunks": 0, "sources": []}

    def clear(self) -> None:
        """Delete all stored chunks."""
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None
