# Settings and multilingual verification — 2026-09-04

## Implemented

- Settings categories: General, Chat & answers, Advanced, Data controls, About.
- Persistent appearance, composer, source display, answer style/language, retrieval depth, evidence filtering, and conversation context controls.
- Local JSON chat export; export only includes explicit chat/source fields, not server configuration.
- Non-Thai questions are translated into Thai search queries by the configured ThaiLLM provider. Translation is used only for retrieval, never as evidence in the answer context. The original question is preserved for generation.
- Answers default to the question language. Thai, English, or Chinese can also be chosen explicitly in Settings.
- Duration evidence is retained alongside credit evidence for program-specific multi-part questions.
- Document-only grounding instructions and insufficient-evidence refusal remain enabled. There is no web search or alternate provider.

## Live ThaiLLM checks

User approved sending the test questions and retrieved PDF excerpts to the configured ThaiLLM API. These tests called the actual RAG runtime and provider, not a mock or the browser UI.

Chinese question:

> KMITL信息技术学院的人工智能技术专业(AIT)总共需要修满多少学分?学制几年?

Final response:

> AIT专业总学分要求为120学分，学制为4年。

English question:

> How many credits are required for the Artificial Intelligence Technology (AIT) program at KMITL, and how many years does it take?

Final response:

> The Artificial Intelligence Technology (AIT) program at KMITL requires a total of 120 credits and takes 4 years to complete.

Both responses used only AIT.pdf excerpts. PDF page 2 contains the 120-credit total and the selected 4-year program option; page 12 corroborates the credit total. Initial live runs returned Thai despite correct facts. Explicit output-language instructions corrected this; both final live checks passed.

## Automated checks

- Python: 99 passed; two existing dependency deprecation warnings.
- JavaScript: 14 passed.
- Coverage includes settings validation, request isolation, context limits, export field selection, persistence, translation failures, original-question preservation, document-only refusal, and real-corpus AIT evidence selection with a mocked translator.

## Limits

- Chinese and English were live-tested. Other languages depend on the configured model's translation/generation abilities and have not been live-verified.
- Non-Thai requests add one model call, increasing latency and API usage.
- No option guarantees factual accuracy; strict filtering can refuse questions that balanced mode answers.
- Insufficient-evidence refusals retain the prescribed Thai text.
- The new settings were tested with mocked DOM interactions, not visually inspected in a live browser during this update.
