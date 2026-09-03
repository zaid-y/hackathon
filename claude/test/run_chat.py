from thaillm_rag import create_rag_pipeline, PipelineMode

pipeline = create_rag_pipeline(mode=PipelineMode.ENHANCED)
pipeline.add_texts([
    'กฎการสมัคร: ต้องเป็นนักศึกษา ทีม 2-4 คน ลงทะเบียนก่อน 09:00 น.',
    'รางวัล: อันดับ 1 ได้ 50,000 บาท อันดับ 2 ได้ 30,000 บาท',
    'การตัดสิน: สร้างสรรค์ 30% ความเป็นไปได้ 25% โค้ด 20% นำเสนอ 15% ทีม 10%',
])

print('🤖 ThaiLLM RAG พร้อมใช้งาน! พิมพ์ exit เพื่อออก')
print('-' * 40)

while True:
    user_input = input('\n👤 คุณ: ').strip()
    if user_input.lower() in ('exit', 'quit', 'ออก', 'bye'):
        break
    if not user_input:
        continue
    try:
        print('🤖 ThaiLLM: ', end='', flush=True)
        for chunk in pipeline.query_stream(user_input):
            print(chunk, end='', flush=True)
        print()
    except Exception as e:
        print(f'\n❌ Error: {e}')

pipeline.close()
print('\n👋 ลาก่อนครับ!')