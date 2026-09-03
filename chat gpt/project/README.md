# ThaiLLM Document Assistant

A competition-oriented Retrieval-Augmented Generation application that will
answer questions only from supplied documents and will use ThaiLLM for document
question answering. The project is being built in eight small, testable phases.

## Current status: Phase 8 complete

Phase 1 provides:

- a clean Python/FastAPI project structure;
- local discovery of PDF, TXT, and DOCX files;
- deterministic text extraction without any AI service;
- real, 1-based page metadata for PDFs;
- `page: null` for TXT and DOCX, which do not expose reliable physical pages;
- Thai-friendly TXT decoding using UTF-8 and CP874 fallbacks;
- paragraph and table extraction from DOCX files;
- isolated error reporting for empty, damaged, or unreadable documents;
- unit tests and a small Thai sample document.

Phase 2 adds:

- configurable character-based chunk size and overlap;
- sentence and paragraph boundary preference for Thai and English text;
- whitespace fallback, then a safe hard split for long unbroken Thai text;
- deterministic IDs such as `rules_p12_c03`;
- exact filename and real PDF page inheritance;
- source character offsets for checking every chunk against extracted text;
- a chunking command, admin API route, and focused unit tests.

Phases 3-8 add:

- persisted, dependency-free BM25 retrieval with Thai bigrams/trigrams;
- exact number/keyword matching, top-K ranking, and confidence scores;
- a fail-closed ThaiLLM HTTP boundary with explicit official-API hooks;
- grounded prompts with document-instruction isolation;
- retrieval confidence gating before ThaiLLM is called;
- citations built only from stored filename/page/chunk metadata;
- key-free debug inspection of query, scores, chunks, and final prompt;
- a responsive, persistent multi-turn chat interface with loading state,
  per-answer sources, New chat, and reindex controls;
- acceptance tests for the six required accuracy scenarios.

## Architecture

```text
Competition documents
        |
        v
DocumentLoader ---- discovers supported local files
        |
        v
DocumentTextExtractor
   | PDF -> one source unit per non-empty page (real page number)
   | TXT -> one source unit (page unknown)
   ` DOCX -> one source unit (page unknown)
        |
        v
ExtractedDocument + ExtractedPage metadata
        |
        v
Chunking -> persisted BM25 -> confidence gate
        |
        v
ThaiLLM -> grounded answer -> source citations
        |
        v
Web interface + acceptance tests
```

Text extraction is intentionally non-AI and local. ThaiLLM will be isolated in
`app/thailmm.py` when the official API format is available. No alternate LLM or
embedding model is part of the document-answering path.

## Project structure

```text
project/
|-- app/
|   |-- main.py              # FastAPI entry point
|   |-- config.py            # Environment-based settings
|   |-- models.py            # Extraction data models
|   |-- document_loader.py   # Folder discovery and batch error isolation
|   |-- text_extractor.py    # PDF/TXT/DOCX extraction
|   |-- chunker.py           # Sentence-aware metadata-preserving chunking
|   |-- retriever.py         # Thai-aware BM25 and persisted index
|   |-- thailmm.py           # Fail-closed official ThaiLLM boundary
|   |-- prompts.py           # Grounded prompt construction
|   `-- answer.py            # Confidence gate, orchestration, citations
|-- data/
|   |-- documents/           # Put competition documents here
|   `-- index/               # Persisted retrieval index (later phase)
|-- frontend/                # Persistent multi-turn document chat interface
|-- tests/
|-- .env.example
|-- requirements.txt
`-- README.md
```

## Install

Python 3.11 or newer is recommended.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## Add documents

Copy the competition's `.pdf`, `.txt`, and `.docx` files into
`data/documents/`. Other file types are ignored by folder discovery. Files are
never sent to an external service during Phase 1.

If needed, set an absolute or project-relative folder in `.env`:

```dotenv
DOCUMENTS_DIR=data/documents
```

### Active competition dataset

The supplied archive `drive-download-20260903T014640Z-1-001.zip` was safely
ingested on 2026-09-03. The live corpus contains only these four files:

- `AIT.pdf`
- `DSBA.pdf`
- `IT2565.pdf`
- `IT_inter2565.pdf`

The previous sample was moved to `tests/fixtures/sample_rules.txt` and cannot
contaminate production retrieval. The active index contains 1,463 unique chunks
covering 1,034 text-bearing PDF pages, with no extraction errors. File hashes,
per-document counts, and the corpus fingerprint are recorded in
`data/ingestion_report.json`.

## Test extraction from the command line

From the project directory:

```powershell
python -m app.document_loader
```

This prints compact JSON metadata. To inspect the extracted text while
debugging:

```powershell
python -m app.document_loader --include-text
```

To scan another folder:

```powershell
python -m app.document_loader --documents C:\path\to\documents
```

Each good document is returned even if another file is damaged. Failures appear
under `errors` with the filename, error type, and safe message.

## Inspect Phase 2 chunks

The defaults come from `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env`:

```powershell
python -m app.chunker --include-text
```

Competition-time values can be tried without editing configuration:

```powershell
python -m app.chunker --chunk-size 800 --chunk-overlap 120 --include-text
```

Chunk size is measured in Unicode characters, keeping the implementation local
and independent of any model tokenizer. Overlap is a maximum: it may be shorter
when moving forward to a clean sentence or whitespace boundary prevents a
mid-sentence start.

## Build and test the Phase 3 index

Documents are indexed once and queries load the persisted local index:

```powershell
python -m app.retriever index
python -m app.retriever search "กำหนดส่งผลงานวันไหน" --include-text
```

The default index is `data/index/bm25_index.json`. Run the index command again
after changing the competition documents or chunk settings.

## Configure the official ThaiLLM API

Copy `.env.example` to `.env` and paste your ThaiLLM Playground API key:

```dotenv
THAILLM_API_KEY=your-private-key
THAILLM_BASE_URL=https://thaillm.or.th/api/typhoon/v1
THAILLM_MODEL=/model
```

The client uses ThaiLLM's OpenAI-compatible `/chat/completions` contract, the
gateway's `apikey` header, and its accepted `litellm` user agent. The configured
base URL selects Typhoon; `/model` is the model
identifier used by the official ThaiLLM workshop configuration. The application
never falls back to OpenAI, Gemini, Claude, or another LLM.

## Run the application

```powershell
python -m uvicorn app.main:app --reload
```

Then open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/admin/documents`
- `http://127.0.0.1:8000/admin/documents?include_text=true`
- `http://127.0.0.1:8000/admin/chunks`
- `http://127.0.0.1:8000/admin/chunks?include_text=true`
- `http://127.0.0.1:8000/` for the web interface
- `POST http://127.0.0.1:8000/api/ask` for question answering
- `POST http://127.0.0.1:8000/admin/reindex` after document changes
- `http://127.0.0.1:8000/admin/debug` when `DEBUG=true`
- `http://127.0.0.1:8000/docs` for interactive API documentation

The admin route never returns secrets. Full text is omitted unless explicitly
requested with `include_text=true`.

## Run tests

```powershell
python -m pytest -q
```

The suite checks all eight phases. `data/test_questions.json` contains the six
competition acceptance cases: clearly answered, multi-chunk, absent information,
similar-but-incorrect information, Thai question, and exact number/date/name.
Tests use a deterministic test double at the ThaiLLM boundary and never send
competition text to another model.

## Metadata decisions

PDF page numbers are taken directly from `pypdf` enumeration and are 1-based.
Blank PDF pages are omitted, but later chunks retain the page number from the
original PDF. DOCX and TXT physical page numbers are unknowable from their file
structure, so this system stores `null`; it never fabricates a citation page.

Phase 2 transforms each `ExtractedPage` into chunks shaped like:

```json
{
  "chunk_id": "rules_p12_c03",
  "text": "...",
  "document": "rules.pdf",
  "page": 12
}
```

## Configuration

All competition tuning values are centralized in `app/config.py`
and documented in `.env.example`: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`,
`RELEVANCE_THRESHOLD`, and the three ThaiLLM connection variables. Folder and
chunk settings are active through Phase 2.

## Competition readiness checklist

1. Replace the sample file with the official competition documents.
2. Run `python -m app.retriever index` and inspect several real queries.
3. Tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, and `RELEVANCE_THRESHOLD` against
   labeled questions; do not tune from intuition alone.
4. Add your ThaiLLM Playground API key to `.env`.
5. Set `DEBUG=false` before the final presentation.
6. Run `python -m pytest -q`, start the server, and test at least one real ThaiLLM answer.

## Known limitations

- Scanned/image PDFs need a competition-approved OCR path; `pypdf` cannot read
  text that exists only as pixels.
- Thai character n-grams are reliable and dependency-free but less precise than
  a permitted Thai word segmenter. Evaluate one only if competition rules allow.
- The index changes only when explicitly rebuilt; the UI includes a reindex
  button to make that action visible and predictable.
- A live ThaiLLM call cannot be verified until a private ThaiLLM API key is
  credentials are provided.
