from __future__ import annotations

from pathlib import Path

import pytest

from app.chunker import ChunkingConfigurationError, DocumentChunker
from app.models import ExtractedDocument, ExtractedPage


def test_short_page_becomes_one_chunk_with_metadata() -> None:
    page = ExtractedPage(document="rules.pdf", page=12, text="Eligibility is 18 years.")

    chunks = DocumentChunker(chunk_size=100, chunk_overlap=10).chunk_page(page)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "rules_p12_c01"
    assert chunks[0].document == "rules.pdf"
    assert chunks[0].page == 12
    assert chunks[0].text == page.text
    assert (chunks[0].start_char, chunks[0].end_char) == (0, len(page.text))


def test_unknown_page_is_explicit_in_chunk_id() -> None:
    page = ExtractedPage(document="ประกาศ.txt", page=None, text="ข้อความทดสอบ")

    chunk = DocumentChunker(chunk_size=100, chunk_overlap=0).chunk_page(page)[0]

    assert chunk.chunk_id == "ประกาศ_pna_c01"
    assert chunk.page is None


def test_prefers_sentence_boundary_before_size_limit() -> None:
    text = "First sentence. Second sentence. Third sentence."
    page = ExtractedPage(document="rules.pdf", page=1, text=text)

    chunks = DocumentChunker(chunk_size=34, chunk_overlap=0).chunk_page(page)

    assert len(chunks) == 2
    assert chunks[0].text == "First sentence. Second sentence."
    assert chunks[1].text == "Third sentence."


def test_thai_newline_is_a_safe_boundary() -> None:
    text = (
        "บรรทัดแรกมีข้อมูลสำคัญ\n"
        "บรรทัดที่สองมีเงื่อนไขเพิ่มเติม\n"
        "บรรทัดสุดท้ายเป็นกำหนดเวลา"
    )
    page = ExtractedPage(document="กติกา.pdf", page=3, text=text)

    chunks = DocumentChunker(chunk_size=58, chunk_overlap=0).chunk_page(page)

    assert len(chunks) >= 2
    assert chunks[0].text.endswith("บรรทัดที่สองมีเงื่อนไขเพิ่มเติม")
    assert all(len(chunk.text) <= 58 for chunk in chunks)


def test_long_unbroken_thai_text_falls_back_to_hard_limit() -> None:
    text = "ก" * 55
    page = ExtractedPage(document="thai.txt", page=None, text=text)

    chunks = DocumentChunker(chunk_size=20, chunk_overlap=5).chunk_page(page)

    assert len(chunks) == 4
    assert all(len(chunk.text) <= 20 for chunk in chunks)
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text)
    assert all(text[chunk.start_char : chunk.end_char] == chunk.text for chunk in chunks)


def test_overlap_repeats_context_without_exceeding_configured_maximum() -> None:
    text = "Sentence A. Sentence B. Sentence C. Sentence D."
    page = ExtractedPage(document="rules.pdf", page=2, text=text)

    chunks = DocumentChunker(chunk_size=30, chunk_overlap=15).chunk_page(page)

    assert len(chunks) >= 2
    for previous, current in zip(chunks, chunks[1:]):
        actual_overlap = previous.end_char - current.start_char
        assert 0 < actual_overlap <= 15


def test_chunk_ids_are_deterministic_and_page_local() -> None:
    document = ExtractedDocument(
        source_path=Path("rules.pdf"),
        document="rules.pdf",
        pages=(
            ExtractedPage(document="rules.pdf", page=1, text="A" * 30),
            ExtractedPage(document="rules.pdf", page=2, text="B" * 30),
        ),
    )
    chunker = DocumentChunker(chunk_size=20, chunk_overlap=0)

    first = chunker.chunk_document(document)
    second = chunker.chunk_document(document)

    assert [chunk.chunk_id for chunk in first] == [
        "rules_p1_c01",
        "rules_p1_c02",
        "rules_p2_c01",
        "rules_p2_c02",
    ]
    assert first == second


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "greater than zero"),
        (100, -1, "cannot be negative"),
        (100, 100, "smaller than chunk_size"),
        (100, 101, "smaller than chunk_size"),
    ],
)
def test_rejects_unsafe_configuration(
    chunk_size: int, chunk_overlap: int, message: str
) -> None:
    with pytest.raises(ChunkingConfigurationError, match=message):
        DocumentChunker(chunk_size, chunk_overlap)
