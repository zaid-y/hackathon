import pytest
from app.multilingual import normalize_query
from app.grounding import question_facets
from app.retriever import preferred_documents

@pytest.mark.parametrize("question,facets", [
    ("KMITL信息技术学院的人工智能技术专业(AIT)总共需要修满多少学分?学制几年?", {"credits","duration"}),
    ("How many credits and how many years for AIT?", {"credits","duration"}),
    ("IT2565专业的学期是怎么划分的?", {"semesters"}),
    ("AIT专业毕业生可以从事哪些职业?该专业的实习(合作教育)学分是多少?", {"careers","co_op"}),
])
def test_local_multilingual_facets(question,facets):
    normalized=normalize_query(question)
    assert normalized.startswith(question)
    assert facets <= set(question_facets(normalized))

def test_original_codes_dates_and_nouns_are_not_rewritten():
    q="AIT IT2565 06046443 2026-09-04 credits careers"
    assert normalize_query(q).startswith(q)
    assert preferred_documents(normalize_query("AIT多少学分")) == ("AIT.pdf",)

def test_thai_semester_synonym():
    assert "ภาคการศึกษา" in normalize_query("IT2565 แบ่งเทอมอย่างไร")
