"""
Example Usage of ThaiLLM RAG Pipeline
Demonstrates various ways to use the pipeline
"""
import os
from thaillm_rag.config import ThaiLLMConfig, RAGConfig, load_config_from_env
from thaillm_rag.rag_pipeline import ThaiLLMRAGPipeline, PipelineMode
from thaillm_rag.retriever import Document, RetrievalStrategy, create_retriever


def setup_environment():
    """Setup environment variables (replace with your actual values)"""
    # ThaiLLM API Configuration
    os.environ["THAILLM_API_URL"] = "https://playground.thaillm.or.th/chat"
    os.environ["THAILLM_API_KEY"] = "LKo7nialAX9hfwzxHr65RMzl5v96zN7N"
    os.environ["THAILLM_MODEL"] = "thaillm-7b-instruct"

    # RAG Configuration
    os.environ["RAG_TOP_K"] = "5"
    os.environ["RAG_ENHANCE_PROMPTS"] = "true"
    os.environ["RAG_MAX_CONTEXT_LENGTH"] = "4000"


def create_sample_documents() -> list[Document]:
    """Create sample Thai documents for testing"""
    return [
        Document(
            content="""การสมัครแข่งขาวแฮกกאתอน: ผู้สมัครต้องเป็นนักศึกษาที่กำลังศึกษาอยู่ในระดับมัธยมศึกษาหรือเทียบเท่า
ทีมต้องมีสมาชิก 2-4 คน ไม่สามารถแข่งขันคนเดียวได้
การลงทะเบียนปิดเวลา 09:00 น. ของวันจัดงาน""",
            metadata={"source": "hackathon_rules.pdf", "section": "eligibility"}
        ),
        Document(
            content="""กฎระเบียบการส่งผลงาน: ผลงานต้องเป็นงานต้นฉบับที่ไม่เคยเผยแพร่ที่อื่น
โค้ดต้องเขียนเป็น Python 3.10 ขึ้นไป
ต้องมีไฟล์ README.md อธิบายวิธีติดตั้งและรันโปรเจกต์
ขนาดไฟล์ต้องไม่เกิน 100 MB""",
            metadata={"source": "hackathon_rules.pdf", "section": "submission"}
        ),
        Document(
            content="""เกณฑ์การตัดสิน:
1. ความคิดสร้างสรรค์และนวัตกรรม (30%)
2. ความเป็นไปได้และความสามารถในการนำไปใช้จริง (25%)
3. คุณภาพโค้ดและสถาปัตยกรรม (20%)
4. การนำเสนอและการสาธิต (15%)
5. การทำงานเป็นทีม (10%)""",
            metadata={"source": "hackathon_rules.pdf", "section": "judging"}
        ),
        Document(
            content="""รางวัล:
- อันดับ 1: เงินรางวัล 50,000 บาท + ถ้วยรางวัล
- อันดับ 2: เงินรางวัล 30,000 บาท + ใบประกาศนียบัตร
- อันดับ 3: เงินรางวัล 20,000 บาท + ใบประกาศนียบัตร
- รางวัลพิเศษ: รางวัลความคิดสร้างสรรค์ 10,000 บาท""",
            metadata={"source": "hackathon_rules.pdf", "section": "prizes"}
        ),
        Document(
            content="""ตารางเวลากิจกรรม:
- 08:00-09:00 น. : เช็คอินและลงทะเบียน
- 09:00-09:30 น. : พิธีเปิดและแนะนำกิจกรรม
- 09:30-18:00 น. : เวลาพัฒนาโปรเจกต์ (8.5 ชั่วโมง)
- 18:00-19:00 น. : เตรียมการนำเสนอ
- 19:00-21:00 น. : การนำเสนอและตัดสิน
- 21:00-21:30 น. : ประกาศรางวัลและปิดงาน""",
            metadata={"source": "hackathon_schedule.pdf", "section": "timeline"}
        ),
    ]


def example_basic_usage():
    """Basic RAG pipeline usage"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Load config from environment
    thaillm_config, rag_config = load_config_from_env()

    # Create pipeline
    pipeline = ThaiLLMRAGPipeline(
        thaillm_config=thaillm_config,
        rag_config=rag_config,
        mode=PipelineMode.ENHANCED
    )

    # Add documents
    documents = create_sample_documents()
    pipeline.add_documents(documents)

    # Query
    questions = [
        "เงื่อนไขการสมัครคืออะไร?",
        "รางวัลอันดับ 1 ได้เท่าไหร่?",
        "ต้องใช้ภาษาโปรแกรมอะไร?",
        "วันจัดงานมีตารางเวลาอย่างไร?",
    ]

    for q in questions:
        print(f"\n❓ คำถาม: {q}")
        try:
            response = pipeline.query(q)
            print(f"💡 คำตอบ: {response.answer}")
            print(f"   ⏱️  เวลา: {response.total_time_ms:.0f}ms | 📄 เอกสารที่ดึงมา: {len(response.retrieval_result.documents)}")
        except Exception as e:
            print(f"❌ Error: {e}")

    pipeline.close()


def example_with_streaming():
    """Streaming response example"""
    print("\n" + "=" * 60)
    print("Example 2: Streaming Response")
    print("=" * 60)

    thaillm_config, rag_config = load_config_from_env()
    pipeline = ThaiLLMRAGPipeline(
        thaillm_config=thaillm_config,
        rag_config=rag_config,
        mode=PipelineMode.ENHANCED
    )
    pipeline.add_documents(create_sample_documents())

    question = "อธิบายเกณฑ์การตัดสินให้ฟังครับ"
    print(f"❓ คำถาม: {question}")
    print("💡 คำตอบ (streaming): ", end="", flush=True)

    try:
        for chunk in pipeline.query_stream(question):
            print(chunk, end="", flush=True)
        print()  # Newline after streaming
    except Exception as e:
        print(f"\n❌ Error: {e}")

    pipeline.close()


def example_multi_query_mode():
    """Multi-query retrieval mode"""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Query Mode (Better Recall)")
    print("=" * 60)

    thaillm_config, rag_config = load_config_from_env()
    pipeline = ThaiLLMRAGPipeline(
        thaillm_config=thaillm_config,
        rag_config=rag_config,
        mode=PipelineMode.MULTI_QUERY
    )
    pipeline.add_documents(create_sample_documents())

    # Complex question that benefits from decomposition
    question = "ผมอยากรู้ว่าสมัครอย่างไร ต้องมีคุณสมบัติอะไรบ้าง และรางวัลมีอะไรบ้าง"
    print(f"❓ คำถามซับซ้อน: {question}")

    try:
        response = pipeline.query(question)
        print(f"💡 คำตอบ: {response.answer}")

        if response.enhanced_query:
            print(f"\n🔍 คำถามที่ปรับปรุงแล้ว:")
            print(f"   - Expanded: {response.enhanced_query.expanded}")
            print(f"   - Keywords: {response.enhanced_query.keyword_focused}")
            print(f"   - Decomposed: {response.enhanced_query.decomposed}")
            print(f"   - ทุก variants ({len(response.enhanced_query.all_variants)}): {response.enhanced_query.all_variants}")

    except Exception as e:
        print(f"❌ Error: {e}")

    pipeline.close()


def example_with_vector_retriever():
    """Example using vector retriever (requires embedder)"""
    print("\n" + "=" * 60)
    print("Example 4: Vector Retrieval (requires embedder)")
    print("=" * 60)

    # This requires sentence-transformers or similar
    # pip install sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer

        # Use a multilingual model that supports Thai
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        def embedder(texts: list[str]) -> list[list[float]]:
            return model.encode(texts, normalize_embeddings=True).tolist()

        # Create hybrid retriever
        retriever = create_retriever(
            strategy=RetrievalStrategy.HYBRID,
            embedder=embedder
        )

        thaillm_config, rag_config = load_config_from_env()
        pipeline = ThaiLLMRAGPipeline(
            thaillm_config=thaillm_config,
            rag_config=rag_config,
            retriever=retriever,
            mode=PipelineMode.ENHANCED
        )
        pipeline.add_documents(create_sample_documents())

        question = "การตัดสินให้คะแนนอย่างไร?"
        print(f"❓ คำถาม: {question}")

        response = pipeline.query(question)
        print(f"💡 คำตอบ: {response.answer}")

        pipeline.close()

    except ImportError:
        print("⚠️  sentence-transformers not installed. Skipping vector example.")
        print("   Install with: pip install sentence-transformers")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_custom_prompts():
    """Example with custom prompts"""
    print("\n" + "=" * 60)
    print("Example 5: Custom System Prompt & Templates")
    print("=" * 60)

    thaillm_config, rag_config = load_config_from_env()

    # Customize for specific domain
    rag_config.system_prompt = """คุณคือผู้ช่วยด้านกฎหมายแฮกกאתอน
ตอบคำถามโดยอ้างอิงจากบริบทที่ให้มาเท่านั้น
ใช้ภาษาทางการ สุภาพ ตรงไปตรงมา
หากไม่มีข้อมูล ให้บอกว่า "ไม่พบข้อมูลในระเบียบการแข่งขัน" """

    rag_config.rag_prompt_template = """[กฎระเบียบการแข่งขัน]
{context}

[คำถามจากผู้แข่งขัน]
{question}

[คำตอบอย่างเป็นทางการ]"""

    pipeline = ThaiLLMRAGPipeline(
        thaillm_config=thaillm_config,
        rag_config=rag_config,
        mode=PipelineMode.ENHANCED
    )
    pipeline.add_documents(create_sample_documents())

    question = "ถ้าผมส่งโค้ดที่เคยเผยแพร่แล้ว จะเกิดอะไรขึ้น?"
    print(f"❓ คำถาม: {question}")

    try:
        response = pipeline.query(question)
        print(f"💡 คำตอบ: {response.answer}")
    except Exception as e:
        print(f"❌ Error: {e}")

    pipeline.close()


def example_statistics():
    """Show pipeline statistics"""
    print("\n" + "=" * 60)
    print("Example 6: Pipeline Statistics")
    print("=" * 60)

    thaillm_config, rag_config = load_config_from_env()
    pipeline = ThaiLLMRAGPipeline(
        thaillm_config=thaillm_config,
        rag_config=rag_config
    )
    pipeline.add_documents(create_sample_documents())

    # Run several queries
    questions = [
        "สมัครอย่างไร?",
        "รางวัลอะไร?",
        "ตารางเวลา?",
        "เกณฑ์ตัดสิน?",
        "ส่งโค้ดยังไง?",
    ]

    for q in questions:
        try:
            pipeline.query(q)
        except:
            pass

    stats = pipeline.get_stats()
    print(f"📊 สถิติการใช้งาน:")
    print(f"   - คำถามทั้งหมด: {stats.total_queries}")
    print(f"   - สำเร็จ: {stats.successful_queries}")
    print(f"   - ล้มเหลว: {stats.failed_queries}")
    print(f"   - เวลาดึงข้อมูลเฉลี่ย: {stats.avg_retrieval_time_ms:.1f}ms")
    print(f"   - เวลาสร้างคำตอบเฉลี่ย: {stats.avg_generation_time_ms:.1f}ms")
    print(f"   - เวลาทั้งหมดเฉลี่ย: {stats.avg_total_time_ms:.1f}ms")

    pipeline.close()


def example_interactive_mode():
    """Interactive chat loop"""
    print("\n" + "=" * 60)
    print("Example 7: Interactive Mode (Ctrl+C to exit)")
    print("=" * 60)

    thaillm_config, rag_config = load_config_from_env()
    pipeline = ThaiLLMRAGPipeline(
        thaillm_config=thaillm_config,
        rag_config=rag_config,
        mode=PipelineMode.ENHANCED
    )
    pipeline.add_documents(create_sample_documents())

    print("🤖 ThaiLLM RAG พร้อมใช้งาน! พิมพ์ 'exit' เพื่อออก")
    print("-" * 40)

    try:
        while True:
            user_input = input("\n👤 คุณ: ").strip()
            if user_input.lower() in ('exit', 'quit', 'ออก', 'bye'):
                break

            if not user_input:
                continue

            try:
                print("🤖 ThaiLLM: ", end="", flush=True)
                for chunk in pipeline.query_stream(user_input):
                    print(chunk, end="", flush=True)
                print()
            except Exception as e:
                print(f"\n❌ Error: {e}")

    except KeyboardInterrupt:
        print("\n\n👋 ลาก่อนครับ!")
    finally:
        pipeline.close()


if __name__ == "__main__":
    # Setup environment (in real use, set these in your shell)
    setup_environment()

    # Run examples (comment out ones you don't want to run)
    print("🚀 ThaiLLM RAG Pipeline Examples")
    print("Note: Set THAILLM_API_KEY and THAILLM_API_URL in environment first!")

    # example_basic_usage()
    # example_with_streaming()
    # example_multi_query_mode()
    # example_with_vector_retriever()
    # example_custom_prompts()
    # example_statistics()
    # example_interactive_mode()

    print("\n✅ Examples ready. Uncomment the function calls to run them.")
    print("\nQuick start:")
    print("  1. Set environment variables:")
    print("     $env:THAILLM_API_KEY='your-key'")
    print("     $env:THAILLM_API_URL='https://api.thaillm.example/v1/chat'")
    print("  2. Run: python example_usage.py")