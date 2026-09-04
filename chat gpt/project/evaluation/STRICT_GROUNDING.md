# Strict extractive grounding

The existing ingestion, index, API and UI are retained. The answer pipeline now
uses local query expansion, per-facet evidence extraction, a model evidence-ID
selection contract, deterministic validation, and server-side rendering.

## Guarantee boundary

No free-form factual model output is accepted. Every displayed factual passage
is a contiguous excerpt verified against a retrieved chunk. Citations are built
from that chunk's metadata, never model fields. This proves textual provenance,
not the truth of documents or flawless PDF extraction. It is deliberately more
conservative than a general semantic entailment checker.

The model receives the original question and a JSON catalog of evidence. It must
return only `{"evidence_ids":[...]}`. Unknown IDs, extra keys, invalid JSON, or
omitted evidence (including competing sources) produce the exact not-found text,
`grounded: false`, and no citations. Model prose cannot leak through the renderer.

Locally recognized fields include curriculum credits, specialized credits,
duration, explicitly headed career/specialization lists, semesters, co-op rows,
and the existing sample age/prize/date/name fields. Unrecognized questions are
refused. Course-code credit questions currently fail closed pending a verified
table-row extractor; they never fall back to a program's total credits.

## Partial answers and conflicts

Supported requested fields appear under `ข้อมูลที่พบ:`. Missing fields appear
under `ข้อมูลที่ไม่พบ:`. If no field is supported, only the exact Thai refusal is
returned. The full refusal is never appended to a successful partial answer.

Matching source accounts are retained separately. Different numeric values for
the same extracted field receive an explicit conflict warning and each source is
shown. Other multi-source accounts are shown separately without deciding which
is correct. This is conservative disclosure, not a universal semantic conflict
classifier. It cannot report contradictions absent from the retrieved index.

## Retrieval and UI tradeoffs

Chinese/English expansion is a deterministic local terminology dictionary. It
appends search terms without deleting the original names, codes, nouns, numbers
or dates. No LLM is used for translation or query processing.

Strict mode scans matching passages instead of cutting evidence/conflicts at
top-k. Explicit section/field matches provide the relevance gate; the optional
strict score filter also requires the BM25 threshold. The context limit is 80
evidence records / 24,000 excerpt characters; exceeding it causes refusal rather
than silent truncation. These limits may reduce recall for broad questions.

Generated translations and style settings are disabled in this mode; factual
excerpts retain their source language and extraction glyphs. Stored settings are
preserved. This is an intentional accuracy tradeoff, visibly explained in Settings.

## Debugging

The existing `/admin/debug` remains available only with server `DEBUG=true`.
Its snapshot includes `query`, `normalized_query`, `retrieval_query`, scored
`retrieved_chunks`, `requested_facets`, `context_sent`, `thailmm_response`,
`validation_errors`, `final_validated_answer`, and `cited_sources`.
Configured API-key values are redacted. Do not enable this document-content
debug endpoint on a publicly accessible deployment without authentication.

## Verification, 4 September 2026

Added negative tests for fabricated careers, numbers, dates, names, statistics,
page numbers, unknown IDs, extra model fields, mixed answer/refusal, missing
career evidence, partial credit/duration answers, numeric conflicts, separate
non-numeric accounts, unknown programs, course-code confusion, fee/semester
confusion, source instructions, debug redaction and stale debug state.

Four previously approved live cases were rerun against the configured ThaiLLM:
Chinese IT semester structure, Thai IT specializations/specialized credits,
Thai AIT careers/co-op, and Chinese AIT careers/co-op. Final runs accepted all four
evidence catalogs. IT credits now cite 93 specialized credits; AIT careers come
from the explicit list on PDF page 4. Initial career runs fabricated an evidence
ID and were correctly refused; simplifying IDs to E1, E2, ... fixed compatibility
without relaxing validation. This is not a new official 20-case accuracy score.
