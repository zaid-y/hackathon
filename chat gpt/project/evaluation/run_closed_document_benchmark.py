"""Run the reusable closed-document benchmark against the local RAG API."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "closed_document_cases.json"
RESULTS_PATH = ROOT / "latest_thailmm_results.json"
REFUSAL_PATTERNS = (
    "ไม่พบข้อมูล",
    "ข้อมูลไม่เพียงพอ",
    "ไม่เพียงพอ",
    "ไม่มีข้อมูล",
    "ไม่สามารถตอบ",
)


def normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    for attempt in range(3):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def source_names(response: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for source in response.get("sources", []):
        name = source.get("document") or source.get("filename") or ""
        if name and name not in names:
            names.append(name)
    return names


def score_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer = str(response.get("answer", ""))
    normalized = normalize(answer)
    refused = response.get("grounded") is False or any(
        normalized.startswith(normalize(pattern)) for pattern in REFUSAL_PATTERNS
    )
    expected_refusal = bool(case.get("expect_refusal"))

    missing_requirements: list[list[str]] = []
    for alternatives in case.get("required", []):
        if not any(normalize(term) in normalized for term in alternatives):
            missing_requirements.append(alternatives)

    actual_sources = source_names(response)
    missing_sources = [
        source for source in case.get("required_sources", []) if source not in actual_sources
    ]

    if expected_refusal:
        answer_correct = refused
        citation_correct = True
    else:
        answer_correct = not refused and not missing_requirements
        citation_correct = not missing_sources

    return {
        "answer_correct": answer_correct,
        "citation_correct": citation_correct,
        "passed": answer_correct and citation_correct,
        "refused": refused,
        "missing_requirements": missing_requirements,
        "missing_sources": missing_sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []

    for case in cases:
        payload = {
            "question": case["question"],
            "history": case.get("history", []),
        }
        try:
            response = post_json(f"{args.base_url}/api/ask", payload, args.timeout)
            score = score_case(case, response)
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": case["question"],
                    "reference_answer": case["reference_answer"],
                    "answer": response.get("answer", ""),
                    "grounded": response.get("grounded"),
                    "confidence": response.get("retrieval_confidence"),
                    "sources": source_names(response),
                    **score,
                }
            )
            label = "PASS" if score["passed"] else "FAIL"
            print(f"{case['id']}: {label}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": case["question"],
                    "reference_answer": case["reference_answer"],
                    "error": str(error),
                    "answer_correct": False,
                    "citation_correct": False,
                    "passed": False,
                }
            )
            print(f"{case['id']}: ERROR - {error}")
        time.sleep(0.5)

    passed = sum(bool(result["passed"]) for result in results)
    answer_correct = sum(bool(result["answer_correct"]) for result in results)
    citation_correct = sum(bool(result["citation_correct"]) for result in results)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": "ThaiLLM Document Assistant",
        "rule": "Use only the supplied PDFs; no internet or outside knowledge.",
        "summary": {
            "total": len(results),
            "passed": passed,
            "answer_correct": answer_correct,
            "citation_correct": citation_correct,
        },
        "results": results,
    }
    RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nOverall: {passed}/{len(results)} passed")
    print(f"Saved: {RESULTS_PATH}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
