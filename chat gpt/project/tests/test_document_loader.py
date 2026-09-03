from __future__ import annotations

from pathlib import Path

import pytest

from app.document_loader import DocumentFolderError, DocumentLoader


def test_loader_discovers_supported_files_and_skips_other_types(tmp_path: Path) -> None:
    (tmp_path / "b.TXT").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("not supported", encoding="utf-8")

    result = DocumentLoader().load(tmp_path)

    assert [document.document for document in result.documents] == ["a.txt", "b.TXT"]
    assert result.errors == []


def test_loader_isolates_bad_documents(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("usable content", encoding="utf-8")
    (tmp_path / "bad.pdf").write_bytes(b"not a real pdf")

    result = DocumentLoader().load(tmp_path)

    assert [document.document for document in result.documents] == ["good.txt"]
    assert len(result.errors) == 1
    assert result.errors[0].document == "bad.pdf"
    assert result.errors[0].error_type == "InvalidDocumentError"


def test_missing_folder_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentFolderError, match="does not exist"):
        DocumentLoader().load(tmp_path / "missing")
