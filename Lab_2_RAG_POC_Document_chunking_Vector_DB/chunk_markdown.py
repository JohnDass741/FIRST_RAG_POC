"""Split a Markdown document into heading-based, context-rich JSONL chunks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*#*\s*$")


def chunk_markdown(markdown: str, source_name: str) -> list[dict[str, object]]:
    """Create one chunk per Markdown section, retaining parent heading context."""
    chunks: list[dict[str, object]] = []
    heading_stack: list[tuple[int, str]] = []
    section_lines: list[str] = []
    section_heading: str | None = None
    section_path: list[str] = []
    in_fence = False

    def emit() -> None:
        body = "\n".join(section_lines).strip()
        if not body:
            return
        heading_path = list(section_path)
        context = " > ".join(heading_path) if heading_path else "Document introduction"
        content = f"Context: {context}\n\n{body}"
        chunks.append(
            {
                "chunk_id": f"{Path(source_name).stem}-{len(chunks) + 1:04d}",
                "source": source_name,
                "section": section_heading or "Document introduction",
                "heading_path": heading_path,
                "content": content,
            }
        )

    for line in markdown.splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence

        heading_match = None if in_fence else HEADING_RE.match(line)
        if heading_match:
            emit()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            section_heading = title
            section_path = [heading for _, heading in heading_stack]
            section_lines = []
        else:
            section_lines.append(line)

    emit()
    return chunks


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=project_dir / "source" / "markdown" / "RAG demo.md",
        help="Markdown file to chunk (default: source/markdown/RAG demo.md)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path (default: source/chunks/<input name>.jsonl)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Markdown input not found: {input_path}")

    output_path = args.output or (project_dir / "source" / "chunks" / f"{input_path.stem}.jsonl")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = input_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(markdown, input_path.name)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} section chunks to: {output_path}")


if __name__ == "__main__":
    main()
