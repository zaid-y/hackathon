"""Discover and extract every supported file in a local document folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.models import ExtractionBatch, ExtractionFailure
from app.text_extractor import DocumentTextExtractor, ExtractionError


class DocumentFolderError(RuntimeError):
    """Raised when the configured document folder cannot be used."""


class DocumentLoader:
    """Load a folder without allowing one bad document to stop the batch."""

    def __init__(self, extractor: DocumentTextExtractor | None = None) -> None:
        self.extractor = extractor or DocumentTextExtractor()

    def discover(self, folder: str | Path) -> list[Path]:
        document_dir = Path(folder).expanduser().resolve()
        if not document_dir.exists():
            raise DocumentFolderError(f"Document folder does not exist: {document_dir}")
        if not document_dir.is_dir():
            raise DocumentFolderError(f"Document path is not a folder: {document_dir}")

        return sorted(
            (
                path
                for path in document_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in self.extractor.SUPPORTED_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        )

    def load(self, folder: str | Path) -> ExtractionBatch:
        batch = ExtractionBatch()
        for path in self.discover(folder):
            try:
                batch.documents.append(self.extractor.extract(path))
            except ExtractionError as exc:
                batch.errors.append(
                    ExtractionFailure(
                        document=path.name,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
            except Exception as exc:
                batch.errors.append(
                    ExtractionFailure(
                        document=path.name,
                        error_type="UnexpectedExtractionError",
                        message=f"Unexpected failure while reading {path.name}: {exc}",
                    )
                )
        return batch


def main() -> None:
    # Windows may otherwise use a legacy code page that cannot print Thai JSON.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Extract local competition documents and print a JSON report."
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=get_settings().documents_dir,
        help="Folder containing PDF, TXT, and DOCX files.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include full extracted text in the JSON output.",
    )
    args = parser.parse_args()

    try:
        result = DocumentLoader().load(args.documents)
    except DocumentFolderError as exc:
        parser.error(str(exc))

    print(json.dumps(result.to_dict(include_text=args.include_text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
