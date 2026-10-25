"""Embed chunk JSONL files with a local Ollama embedding model.

By default, reads source/chunks/*.jsonl and writes one JSONL file per input
under source/embeddings/. Each output record retains the original chunk fields
and adds an embedding vector and embedding_model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "source" / "chunks"
DEFAULT_OUTPUT = PROJECT_DIR / "source" / "embeddings"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def ollama_embed(texts: list[str], *, model: str, base_url: str) -> list[list[float]]:
    """Request embeddings from Ollama's /api/embed endpoint."""
    url = f"{base_url.rstrip('/')}/api/embed"
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}. Start Ollama and make sure the "
            f"'{model}' model is available (ollama pull {model}). Details: {exc}"
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Ollama returned invalid JSON from {url}: {exc}") from exc

    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RuntimeError(
            "Unexpected Ollama response: expected one embedding per input text. "
            f"Received {len(embeddings) if isinstance(embeddings, list) else 'no'} "
            f"embeddings for {len(texts)} chunks."
        )
    return embeddings


def read_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path}, line {line_number}.")
            content = record.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Missing non-empty 'content' in {path}, line {line_number}."
                )
            chunks.append(record)
    return chunks


def embed_file(
    input_path: Path,
    output_path: Path,
    *,
    model: str,
    base_url: str,
    batch_size: int,
) -> int:
    chunks = read_chunks(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary sibling so an interrupted run does not leave a
    # partially written output looking complete.
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                vectors = ollama_embed(
                    [chunk["content"] for chunk in batch], model=model, base_url=base_url
                )
                for chunk, vector in zip(batch, vectors, strict=True):
                    record = {**chunk, "embedding": vector, "embedding_model": model}
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return len(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path, default=DEFAULT_INPUT,
        help="Chunk JSONL file or directory (default: source/chunks).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory (default: source/embeddings).",
    )
    parser.add_argument(
        "--model", default=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_MODEL),
        help=f"Ollama embedding model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--ollama-url", default=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_URL),
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Number of chunks per Ollama request (default: 16).",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = sorted(input_path.glob("*.jsonl"))
    else:
        parser.error(f"Input path does not exist: {input_path}")
    if not input_files:
        parser.error(f"No .jsonl chunk files found at: {input_path}")

    try:
        for input_file in input_files:
            output_file = output_dir / input_file.name
            count = embed_file(
                input_file, output_file, model=args.model,
                base_url=args.ollama_url, batch_size=args.batch_size,
            )
            print(f"Embedded {count} chunks: {output_file}")
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
