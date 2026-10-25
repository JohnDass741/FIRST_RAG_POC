"""Upsert locally generated chunk embeddings into a Pinecone index.

Reads source/embeddings/RAG demo.jsonl by default. Pinecone credentials and
index settings are read from the project root .env file (or the process
environment, which takes precedence).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "source" / "embeddings" / "RAG demo.jsonl"
ENV_FILE = PROJECT_DIR / ".env"


def load_env_file(env_path: Path) -> None:
    """Load basic KEY=VALUE entries without overriding existing environment."""
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\"", "'"):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def read_vectors(input_path: Path) -> list[dict[str, Any]]:
    vectors: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_path}, line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {input_path}, line {line_number}.")

            vector_id = record.get("chunk_id")
            values = record.get("embedding")
            if not isinstance(vector_id, str) or not vector_id:
                raise ValueError(f"Missing chunk_id at {input_path}, line {line_number}.")
            if not isinstance(values, list) or not values or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ):
                raise ValueError(f"Invalid embedding vector at {input_path}, line {line_number}.")

            # Pinecone stores vectors separately from metadata. Keep the chunk
            # content and other original fields as metadata, excluding values.
            metadata = {key: value for key, value in record.items() if key != "embedding"}
            vectors.append({"id": vector_id, "values": values, "metadata": metadata})
    return vectors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path, default=DEFAULT_INPUT,
        help="Embedding JSONL file (default: source/embeddings/RAG demo.jsonl).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Number of vectors per Pinecone upsert request (default: 100).",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    input_path = args.input.resolve()
    if not input_path.is_file():
        parser.error(f"Embedding file not found: {input_path}")

    load_env_file(ENV_FILE)
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    namespace = os.getenv("PINECONE_NAMESPACE", "")
    host = os.getenv("PINECONE_HOST")
    if not api_key:
        parser.error("PINECONE_API_KEY is missing from .env and the process environment")
    if not index_name:
        parser.error("PINECONE_INDEX_NAME is missing from .env and the process environment")

    try:
        from pinecone import Pinecone
    except ImportError:
        print("Missing Pinecone SDK. Install it with: python -m pip install pinecone", file=sys.stderr)
        return 1

    try:
        vectors = read_vectors(input_path)
        if not vectors:
            print(f"No vectors found in {input_path}")
            return 0

        client = Pinecone(api_key=api_key)
        description = client.indexes.describe(index_name)
        schema_fields = getattr(getattr(description, "schema", None), "fields", None) or {}
        integrated_fields = []
        for field_name, field_config in schema_fields.items():
            embed_config = getattr(field_config, "embed", None)
            if embed_config:
                integrated_fields.append(
                    (field_name, getattr(embed_config, "model", "unknown"))
                )
        if integrated_fields:
            configured = ", ".join(
                f"{field_name} (Pinecone model: {model})"
                for field_name, model in integrated_fields
            )
            raise ValueError(
                "The configured index uses Pinecone integrated-embedding fields: "
                f"{configured}. The input file contains precomputed embeddings "
                "(embedding_model is stored in each row), and this index requires "
                "text through its Documents API instead. To upload these existing "
                "vectors, use a vector index configured for the same embedding "
                "dimension and metric, then point PINECONE_INDEX_NAME/HOST to it."
            )

        dense_fields = [
            field_name
            for field_name, field_config in schema_fields.items()
            if getattr(field_config, "dimension", None) is not None
            and getattr(field_config, "metric", None) is not None
        ]
        if schema_fields and not dense_fields:
            raise ValueError(
                "The index uses a document schema without a dense-vector field. "
                "Add a dense_vector field or use a vector index that matches the "
                "embedding dimension and metric."
            )
        if len(dense_fields) > 1:
            raise ValueError(
                f"The index has multiple dense-vector fields ({dense_fields}); "
                "this JSONL uploader needs one unambiguous target field."
            )

        document_mode = bool(schema_fields)
        if document_mode:
            index = client.index(name=index_name, host=host or "")
        else:
            index = client.Index(host=host) if host else client.Index(index_name)

        for start in range(0, len(vectors), args.batch_size):
            batch = vectors[start : start + args.batch_size]
            if document_mode:
                dense_field = dense_fields[0]
                documents = [
                    {
                        "_id": vector["id"],
                        dense_field: vector["values"],
                        **vector["metadata"],
                    }
                    for vector in batch
                ]
                index.documents.upsert(namespace=namespace, documents=documents)
            else:
                index.upsert(vectors=batch, namespace=namespace)
            print(f"Upserted {min(start + len(batch), len(vectors))}/{len(vectors)} vectors")
    except Exception as exc:
        print(f"Pinecone ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Finished: {len(vectors)} vectors inserted into index '{index_name}' "
        f"in namespace '{namespace}'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
