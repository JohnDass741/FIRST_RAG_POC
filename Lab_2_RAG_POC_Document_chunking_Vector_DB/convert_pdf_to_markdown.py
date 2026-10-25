"""Convert the RAG demo PDF to Markdown with Docling."""

from pathlib import Path

from docling.document_converter import DocumentConverter


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    source_pdf = project_dir / "source" / "pdf" / "RAG demo.pdf"
    # The current sample sits in the project root; keep the requested source/pdf
    # location as the primary path and support the existing sample as a fallback.
    if not source_pdf.is_file():
        root_sample = project_dir / "RAG demo.pdf"
        if root_sample.is_file():
            source_pdf = root_sample
    markdown_dir = project_dir / "source" / "markdown"
    markdown_path = markdown_dir / f"{source_pdf.stem}.md"

    if not source_pdf.is_file():
        raise FileNotFoundError(f"Input PDF not found: {source_pdf}")

    markdown_dir.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    result = converter.convert(source_pdf)
    markdown_path.write_text(result.document.export_to_markdown(), encoding="utf-8")
    print(f"Markdown written to: {markdown_path}")


if __name__ == "__main__":
    main()
