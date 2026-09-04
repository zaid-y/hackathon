from evaluation.run_closed_document_benchmark import score_case


def test_non_refusal_with_phrase_no_more_information_is_not_misclassified() -> None:
    case = {
        "required": [["ออนไลน์"], ["ผสมผสาน"]],
        "required_sources": ["IT_inter2565.pdf"],
    }
    response = {
        "answer": "มีแบบออนไลน์และแบบผสมผสาน โดยไม่มีข้อมูลเพิ่มเติม",
        "grounded": True,
        "sources": [{"document": "IT_inter2565.pdf"}],
    }

    result = score_case(case, response)

    assert result["refused"] is False
    assert result["passed"] is True


def test_explicit_guardrail_refusal_is_recognized() -> None:
    case = {"expect_refusal": True, "required_sources": []}
    response = {
        "answer": "ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในเอกสารที่กำหนด",
        "grounded": True,
        "sources": [{"document": "IT2565.pdf"}],
    }

    result = score_case(case, response)

    assert result["refused"] is True
    assert result["passed"] is True
