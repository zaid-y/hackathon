"""
Example: Loading Documents into ThaiLLM RAG Pipeline
Demonstrates the new document extraction and loading capabilities
"""
import os
import tempfile
from thaillm_rag import create_rag_pipeline, PipelineMode
from thaillm_rag.document_extractor import (
    ThaiTextChunker,
    ChunkingConfig,
    extract_documents,
    load_documents_for_rag,
    PDFExtractor,
    DocxExtractor,
    TextExtractor,
    HTMLExtractor,
    WebExtractor,
    create_extractor,
)


def create_sample_files():
    """Create sample files for testing"""
    files = {}

    # Text file
    txt_content = """# ThaiLLM Hackathon 2024 - Official Rules

## Eligibility
- Open to all currently enrolled students (high school, university, vocational)
- Teams must have 2-4 members
- Individual participation not allowed
- Registration closes at 09:00 AM on the event day

## Submission Requirements
- All work must be original and not previously published
- Code must be written in Python 3.10 or higher
- Must include a README.md with installation and run instructions
- Total submission size must not exceed 100 MB
- Repository must be public on GitHub

## Judging Criteria
1. Creativity & Innovation (30%) - Novel approach, unique solution
2. Feasibility & Practical Impact (25%) - Real-world applicability
3. Code Quality & Architecture (20%) - Clean, maintainable, well-documented
4. Presentation & Demo (15%) - Clear explanation, working demo
5. Teamwork & Collaboration (10%) - Effective team dynamics

## Prizes
- 1st Place: 50,000 THB + Trophy
- 2nd Place: 30,000 THB + Certificate
- 3rd Place: 20,000 THB + Certificate
- Special Award: 10,000 THB for Most Creative Solution

## Schedule
- 08:00-09:00: Check-in & Registration
- 09:00-09:30: Opening Ceremony
- 09:30-18:00: Development Time (8.5 hours)
- 18:00-19:00: Presentation Preparation
- 19:00-21:00: Presentations & Judging
- 21:00-21:30: Awards & Closing
"""

    # Markdown file
    md_content = """# Technical Requirements

## Stack
- **Language**: Python 3.10+
- **Framework**: FastAPI, Streamlit, or any Python web framework
- **ML Libraries**: PyTorch, TensorFlow, scikit-learn, sentence-transformers
- **Database**: SQLite, PostgreSQL, or vector databases (Chroma, FAISS, etc.)

## API Integration
- ThaiLLM API endpoint: `https://api.thaillm.or.th/v1/chat/completions`
- Authentication: Bearer token
- Models available: OpenThaiGPT, Typhoon, Pathumma, THaLLE, Qwen variants

## Evaluation
- Code quality: Linting (ruff, black), type hints, tests
- Documentation: README, API docs, architecture diagram
- Performance: Response time, throughput, memory usage
"""

    # Create temp files
    for name, content in [("rules.txt", txt_content), ("tech_req.md", md_content)]:
        with tempfile.NamedTemporaryFile(mode='w', suffix=os.path.splitext(name)[1], delete=False, encoding='utf-8') as f:
            f.write(content)
            files[name] = f.name

    return files


def example_basic_extraction():
    """Example 1: Basic document extraction"""
    print("=" * 60)
    print("Example 1: Basic Document Extraction")
    print("=" * 60)

    files = create_sample_files()

    try:
        # Extract from single file
        chunks = extract_documents(files["rules.txt"])
        print(f"Extracted {len(chunks)} chunks from rules.txt")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  Chunk {i+1} ({chunk.metadata['chunk_tokens']} tokens): {chunk.content[:100]}...")

        # Extract from multiple files
        all_chunks = extract_multiple_sources(list(files.values()))
        print(f"\nTotal chunks from all files: {len(all_chunks)}")

    finally:
        for path in files.values():
            os.unlink(path)


def example_load_for_rag():
    """Example 2: Load documents directly for RAG pipeline"""
    print("\n" + "=" * 60)
    print("Example 2: Load Documents for RAG Pipeline")
    print("=" * 60)

    files = create_sample_files()

    try:
        # Load and format for pipeline
        docs = load_documents_for_rag(
            list(files.values()),
            chunk_size=200,
            chunk_overlap=30
        )
        print(f"Prepared {len(docs)} documents for RAG")
        for i, doc in enumerate(docs[:3]):
            print(f"  Doc {i+1}: {doc['content'][:100]}...")
            print(f"    Metadata: {list(doc['metadata'].keys())}")

    finally:
        for path in files.values():
            os.unlink(path)


def example_pipeline_integration():
    """Example 3: Integration with RAG Pipeline"""
    print("\n" + "=" * 60)
    print("Example 3: Pipeline Integration (add_documents_from_files)")
    print("=" * 60)

    files = create_sample_files()

    try:
        # Create pipeline
        pipeline = create_rag_pipeline(mode=PipelineMode.ENHANCED)

        # Load documents directly into pipeline
        num_added = pipeline.add_documents_from_files(
            list(files.values()),
            chunk_size=200,
            chunk_overlap=30
        )
        print(f"Added {num_added} chunks to pipeline")

        # Test queries (will use fallback since no API key)
        test_questions = [
            "เงื่อนไขการสมัครคืออะไร?",
            "รางวัลอันดับ 1 ได้เท่าไหร่?",
            "ต้องใช้ Python กี่ขึ้นไป?",
            "ตารางเวลาการแข่งขันเป็นอย่างไร?",
        ]

        for q in test_questions:
            print(f"\n❓ {q}")
            try:
                response = pipeline.query(q)
                print(f"💡 {response.answer[:200]}...")
            except Exception as e:
                print(f"   (Expected error without API key: {type(e).__name__})")

        pipeline.close()

    finally:
        for path in files.values():
            os.unlink(path)


def example_directory_loading():
    """Example 4: Load all documents from a directory"""
    print("\n" + "=" * 60)
    print("Example 4: Directory Loading (add_documents_from_directory)")
    print("=" * 60)

    import tempfile
    import shutil

    # Create temp directory with multiple files
    temp_dir = tempfile.mkdtemp()

    try:
        # Create multiple files
        sample_docs = {
            "doc1.txt": "Document 1: ThaiLLM RAG overview and features.",
            "doc2.txt": "Document 2: Installation guide and requirements.",
            "doc3.md": "# Document 3\n\nAPI usage examples and best practices.",
        }

        for name, content in sample_docs.items():
            with open(os.path.join(temp_dir, name), 'w', encoding='utf-8') as f:
                f.write(content)

        pipeline = create_rag_pipeline(mode=PipelineMode.ENHANCED)
        num_added = pipeline.add_documents_from_directory(
            temp_dir,
            extensions=[".txt", ".md"],
            chunk_size=100,
            chunk_overlap=20
        )
        print(f"Added {num_added} chunks from directory: {temp_dir}")

        pipeline.close()

    finally:
        shutil.rmtree(temp_dir)


def example_extractor_usage():
    """Example 5: Using extractors directly"""
    print("\n" + "=" * 60)
    print("Example 5: Using Extractors Directly")
    print("=" * 60)

    files = create_sample_files()

    try:
        # Use specific extractor
        extractor = TextExtractor()
        docs = extractor.extract(files["rules.txt"])
        print(f"TextExtractor: {len(docs)} document(s)")
        print(f"  Metadata: {docs[0].metadata}")

        # Auto-detect extractor
        auto_extractor = create_extractor(files["tech_req.md"])
        print(f"Auto-detected: {auto_extractor.__class__.__name__}")

        # Custom chunking
        chunker = ThaiTextChunker(ChunkingConfig(
            chunk_size=150,
            chunk_overlap=25,
            preserve_sentences=True
        ))
        chunks = chunker.chunk(docs[0].content, docs[0].metadata)
        print(f"Custom chunking: {len(chunks)} chunks")

    finally:
        for path in files.values():
            os.unlink(path)


def extract_multiple_sources(sources, chunker=None):
    """Helper to extract from multiple sources"""
    from thaillm_rag.document_extractor import extract_multiple_sources as ems
    return ems(sources, chunker)


if __name__ == "__main__":
    print("📚 ThaiLLM RAG - Document Loading Examples")
    print()

    example_basic_extraction()
    example_load_for_rag()
    example_pipeline_integration()
    example_directory_loading()
    example_extractor_usage()

    print("\n" + "=" * 60)
    print("✅ Examples completed!")
    print("=" * 60)
    print("\nTo use with real API:")
    print("  1. Set environment variables:")
    print("     $env:THAILLM_API_KEY='your-key'")
    print("     $env:THAILLM_API_URL='https://api.thaillm.or.th/v1/chat/completions'")
    print("  2. Install optional dependencies:")
    print("     pip install pdfplumber python-docx beautifulsoup4")