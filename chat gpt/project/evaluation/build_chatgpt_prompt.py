"""Build the fair closed-context ChatGPT comparison prompt from indexed chunks."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "bm25_index.json"
CASES_PATH = EVALUATION_DIR / "closed_document_cases.json"
OUTPUT_PATH = EVALUATION_DIR / "chatgpt_closed_context_prompt.txt"

EVIDENCE_CHUNK_IDS = (
    "IT2565_p15_c01",
    "IT2565_p2_c02",
    "AIT_p12_c02",
    "DSBA_p14_c01",
    "DSBA_p12_c01",
    "IT_inter2565_p14_c01",
)


def main() -> None:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    chunks = {chunk["chunk_id"]: chunk for chunk in index["chunks"]}
    missing = [chunk_id for chunk_id in EVIDENCE_CHUNK_IDS if chunk_id not in chunks]
    if missing:
        raise RuntimeError(f"Missing evidence chunks: {missing}")

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    context_blocks = []
    for chunk_id in EVIDENCE_CHUNK_IDS:
        chunk = chunks[chunk_id]
        context_blocks.append(
            "\n".join(
                [
                    f"[DOCUMENT: {chunk['document']}, PAGE: {chunk.get('page')}]",
                    chunk["text"],
                ]
            )
        )

    question_blocks = []
    for case in cases:
        if case.get("history"):
            history = " | ".join(
                f"{turn['role']}: {turn['content']}" for turn in case["history"]
            )
            question_blocks.append(
                f"{case['id']}. HISTORY: {history}\nQUESTION: {case['question']}"
            )
        else:
            question_blocks.append(f"{case['id']}. {case['question']}")

    prompt = "\n\n".join(
        [
            "คุณกำลังทำแบบทดสอบ closed-document QA ภาษาไทย",
            (
                "กฎ: ตอบโดยใช้เฉพาะ CONTEXT ด้านล่างเท่านั้น ห้ามค้นอินเทอร์เน็ต "
                "ห้ามใช้ความรู้ภายนอก ห้ามเดา หากข้อมูลไม่พอให้ตอบว่า "
                '"ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในเอกสารที่กำหนด" '
                "ตอบทุกข้อเป็น JSON array เท่านั้น แต่ละรายการต้องมี id และ answer "
                "ห้ามใส่ markdown และห้ามกล่าวชื่อไฟล์หรือเลขหน้าใน answer"
            ),
            "CONTEXT:\n" + "\n\n".join(context_blocks),
            "QUESTIONS:\n" + "\n\n".join(question_blocks),
        ]
    )
    OUTPUT_PATH.write_text(prompt, encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")
    print(f"Characters: {len(prompt)}")


if __name__ == "__main__":
    main()
