from __future__ import annotations

from pathlib import Path

import pytest

from app.models import TextChunk
from app.retriever import BM25Retriever, IndexFormatError, expand_query, tokenize


def _chunk(chunk_id: str, text: str, document: str, page: int) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        text=text,
        document=document,
        page=page,
        chunk_index=1,
        start_char=0,
        end_char=len(text),
    )


@pytest.fixture
def retriever() -> BM25Retriever:
    index = BM25Retriever()
    index.build(
        [
            _chunk(
                "rules_p1_c01",
                "ผู้สมัครต้องมีอายุอย่างน้อย 18 ปี และสมัครภายในวันที่ 15 มกราคม 2569",
                "rules.pdf",
                1,
            ),
            _chunk(
                "rules_p2_c01",
                "ทีมต้องมีสมาชิก 3 ถึง 5 คน หัวหน้าทีมชื่อสมชาย",
                "rules.pdf",
                2,
            ),
            _chunk(
                "prizes_p1_c01",
                "รางวัลชนะเลิศมีมูลค่า 50,000 บาท ประกาศผลวันที่ 28 กุมภาพันธ์ 2569",
                "prizes.pdf",
                1,
            ),
        ]
    )
    return index


def test_thai_tokenizer_handles_text_without_spaces() -> None:
    tokens = tokenize("กำหนดส่งผลงาน")

    assert "กำ" in tokens
    assert "กำห" in tokens
    assert "กำหนดส่งผลงาน" in tokens


def test_clear_question_ranks_answered_document_first(retriever: BM25Retriever) -> None:
    results = retriever.search("ผู้สมัครต้องมีอายุอย่างน้อยเท่าไร")

    assert results[0].chunk.document == "rules.pdf"
    assert results[0].chunk.page == 1
    assert results[0].confidence > 0.4


def test_exact_number_and_date_lookup(retriever: BM25Retriever) -> None:
    results = retriever.search("50,000 บาท ประกาศผลวันที่ 28 กุมภาพันธ์ 2569")

    assert results[0].chunk.document == "prizes.pdf"
    assert "50000" not in results[0].matched_terms
    assert {"50", "000", "28", "2569"}.intersection(results[0].matched_terms)


def test_unrelated_question_has_no_confident_result(retriever: BM25Retriever) -> None:
    results = retriever.search("สนามบินนานาชาติอยู่จังหวัดอะไร")

    assert not results or results[0].confidence < 0.35


def test_top_k_is_respected(retriever: BM25Retriever) -> None:
    assert len(retriever.search("วันที่", top_k=2)) <= 2


def test_curriculum_credit_query_expands_to_document_wording() -> None:
    expanded = expand_query("หลักสูตรนี้มีจำนวนหน่วยกิตทั้งหมดกี่หน่วยกิต")

    assert "หน่วยกิตรวมตลอดหลักสูตร" in expanded
    assert "โครงสร้างหลักสูตร" in expanded


def test_credit_expansion_ranks_decisive_curriculum_chunk_first() -> None:
    index = BM25Retriever()
    index.build(
        [
            _chunk(
                "cover_p1_c01",
                "หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาเทคโนโลยีสารสนเทศ",
                "curriculum.pdf",
                1,
            ),
            _chunk(
                "structure_p15_c01",
                "โครงสร้างหลักสูตร จำนวนหน่วยกิตรวมตลอดหลักสูตร 120 หน่วยกิต",
                "curriculum.pdf",
                15,
            ),
        ]
    )

    results = index.search("หลักสูตรนี้มีจำนวนหน่วยกิตทั้งหมดกี่หน่วยกิต")

    assert results[0].chunk.chunk_id == "structure_p15_c01"


def test_saved_index_round_trip_preserves_ranking(
    retriever: BM25Retriever, tmp_path: Path
) -> None:
    path = tmp_path / "index.json"
    retriever.save(path)

    loaded = BM25Retriever.load(path)

    assert loaded.corpus_fingerprint() == retriever.corpus_fingerprint()
    assert loaded.search("หัวหน้าทีมชื่ออะไร")[0].chunk.page == 2


def test_bad_index_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 999}', encoding="utf-8")

    with pytest.raises(IndexFormatError, match="Unsupported index schema"):
        BM25Retriever.load(path)
