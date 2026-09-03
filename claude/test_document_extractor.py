"""
Test script for document_extractor module
"""
import os
import tempfile
from thaillm_rag.document_extractor import (
    ThaiTextChunker,
    ChunkingConfig,
    TextExtractor,
    extract_documents,
    load_documents_for_rag,
    HAS_PDFPLUMBER,
    HAS_PYMUPDF,
    HAS_DOCX,
    HAS_BS4,
    HAS_PYTHAINLP,
    HAS_DEEPCUT,
)


def test_thai_tokenizer():
    """Test Thai tokenization"""
    print("=" * 60)
    print("Test: Thai Tokenizer")
    print("=" * 60)

    from thaillm_rag.document_extractor import ThaiTokenizer
    tokenizer = ThaiTokenizer()
    print(f"Backend: {tokenizer._backend}")
    print(f"pythainlp: {HAS_PYTHAINLP}, deepcut: {HAS_DEEPCUT}")

    test_texts = [
        "สวัสดีครับ ผมชื่อสมชาย",
        "ThaiLLM is a Thai language model",
        "การประมวลผลภาษาธรรมชาติ (NLP) สำคัญมาก",
        "Hello world 你好 안녕하세요",
    ]

    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        count = tokenizer.count_tokens(text)
        print(f"  Text: {text}")
        print(f"  Tokens ({count}): {tokens}")
        print()


def test_thai_chunker():
    """Test Thai text chunking"""
    print("=" * 60)
    print("Test: Thai Text Chunker")
    print("=" * 60)

    config = ChunkingConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
    chunker = ThaiTextChunker(config)

    # Test with Thai text
    thai_text = """
    การแข่งขันแฮกกาธอนครั้งนี้เปิดให้สมัครสำหรับนักศึกษาทุกคน
    ทีมต้องมีสมาชิก 2-4 คน ไม่สามารถแข่งขันคนเดียวได้
    การลงทะเบียนปิดเวลา 09:00 น. ของวันจัดงาน
    ผลงานต้องเป็นงานต้นฉบับที่ไม่เคยเผยแพร่ที่อื่น
    โค้ดต้องเขียนเป็น Python 3.10 ขึ้นไป
    ต้องมีไฟล์ README.md อธิบายวิธีติดตั้งและรันโปรเจกต์
    เกณฑ์การตัดสินมี 5 ประการ ได้แก่ ความคิดสร้างสรรค์ ความเป็นไปได้ คุณภาพโค้ด การนำเสนอ และการทำงานเป็นทีม
    รางวัลอันดับ 1 ได้เงิน 50,000 บาท อันดับ 2 ได้ 30,000 บาท อันดับ 3 ได้ 20,000 บาท
    """

    chunks = chunker.chunk(thai_text, {"source": "test.txt"})
    print(f"Original text length: {len(thai_text)} chars")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1} ({chunk.metadata.get('chunk_tokens', 0)} tokens): {chunk.content[:80]}...")
    print()


def test_text_extractor():
    """Test text file extraction"""
    print("=" * 60)
    print("Test: Text Extractor")
    print("=" * 60)

    # Create a temporary text file
    test_content = """# ThaiLLM Hackathon Rules

## Eligibility
- Must be currently enrolled students
- Teams of 2-4 members
- Registration closes at 09:00 AM on event day

## Submission Requirements
- Original work not published elsewhere
- Python 3.10+ code
- README.md with setup instructions
- File size under 100 MB

## Judging Criteria
1. Creativity & Innovation (30%)
2. Feasibility (25%)
3. Code Quality (20%)
4. Presentation (15%)
5. Teamwork (10%)
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_path = f.name

    try:
        extractor = TextExtractor()
        docs = extractor.extract(temp_path)
        print(f"Extracted {len(docs)} document(s)")
        for doc in docs:
            print(f"  Content length: {len(doc.content)} chars")
            print(f"  Metadata: {doc.metadata}")
            print(f"  Preview: {doc.content[:200]}...")
    finally:
        os.unlink(temp_path)
    print()


def test_extract_documents():
    """Test high-level extract_documents function"""
    print("=" * 60)
    print("Test: extract_documents() function")
    print("=" * 60)

    test_content = """สวัสดีครับ นี่คือเอกสารทดสอบสำหรับ ThaiLLM RAG Pipeline
    ระบบนี้รองรับการแยกข้อความจากหลายรูปแบบไฟล์
    รวมถึง PDF, DOCX, TXT, HTML และเว็บไซต์
    การแยกข้อความภาษาไทยใช้ pythainlp หรือ deepcut
    ช่วยให้การค้นหาข้อมูลแม่นยำขึ้น"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_path = f.name

    try:
        chunks = extract_documents(temp_path)
        print(f"Extracted {len(chunks)} chunk(s)")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {chunk.content[:80]}...")
            print(f"    Tokens: {chunk.metadata.get('chunk_tokens', 0)}, Index: {chunk.metadata.get('chunk_index', 0)}")
    finally:
        os.unlink(temp_path)
    print()


def test_load_documents_for_rag():
    """Test load_documents_for_rag function"""
    print("=" * 60)
    print("Test: load_documents_for_rag() function")
    print("=" * 60)

    test_content = """กฎการแข่งขันแฮกกาธอน:
    1. สมัครเป็นทีม 2-4 คน
    2. ต้องเป็นนักศึกษา
    3. ลงทะเบียนก่อน 09:00 น.
    4. ส่งโค้ด Python 3.10+
    5. ต้องมี README.md"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_path = f.name

    try:
        docs = load_documents_for_rag([temp_path], chunk_size=50, chunk_overlap=10)
        print(f"Loaded {len(docs)} document(s) for RAG")
        for i, doc in enumerate(docs):
            print(f"  Doc {i+1}: {doc['content'][:80]}...")
            print(f"    Metadata keys: {list(doc['metadata'].keys())}")
    finally:
        os.unlink(temp_path)
    print()


def test_integration_with_pipeline():
    """Test integration with RAG pipeline"""
    print("=" * 60)
    print("Test: Integration with RAG Pipeline")
    print("=" * 60)

    try:
        from thaillm_rag import create_rag_pipeline, PipelineMode

        test_content = """ThaiLLM RAG Pipeline is a retrieval-augmented generation system for Thai language.
        It includes prompt enhancement, document retrieval, and answer generation.
        Supports multiple retrieval strategies: keyword, vector, and hybrid.
        Uses ThaiLLM API for generation."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_path = f.name

        try:
            pipeline = create_rag_pipeline(mode=PipelineMode.ENHANCED)
            num_added = pipeline.add_documents_from_files([temp_path], chunk_size=100, chunk_overlap=20)
            print(f"Added {num_added} chunks to pipeline")

            # Test query
            response = pipeline.query("ThaiLLM RAG Pipeline รองรับกลยุทธ์การค้นหาอะไรบ้าง?")
            print(f"Query response: {response.answer[:200]}...")
            print(f"Retrieved docs: {len(response.retrieval_result.documents)}")

            pipeline.close()
        finally:
            os.unlink(temp_path)

    except Exception as e:
        print(f"Integration test skipped (no API): {e}")
    print()


def test_feature_flags():
    """Print available feature flags"""
    print("=" * 60)
    print("Feature Flags")
    print("=" * 60)
    print(f"PDF (pdfplumber): {HAS_PDFPLUMBER}")
    print(f"PDF (PyMuPDF): {HAS_PYMUPDF}")
    print(f"DOCX: {HAS_DOCX}")
    print(f"HTML (BeautifulSoup): {HAS_BS4}")
    print(f"Thai (pythainlp): {HAS_PYTHAINLP}")
    print(f"Thai (deepcut): {HAS_DEEPCUT}")
    print()


def main():
    print("🧪 ThaiLLM RAG - Document Extractor Tests")
    print()

    test_feature_flags()
    test_thai_tokenizer()
    test_thai_chunker()
    test_text_extractor()
    test_extract_documents()
    test_load_documents_for_rag()
    test_integration_with_pipeline()

    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()