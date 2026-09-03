from app.config import get_settings
from app.answer import RAGRuntime
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retriever import BM25Retriever
import re

settings = get_settings()
runtime = RAGRuntime(settings)
service = runtime.ensure_ready()

retrieved = service.retriever.search('หลักสูตร', top_k=settings.top_k)
relevant = [r for r in retrieved if r.confidence >= settings.relevance_threshold]

user_prompt = build_user_prompt('หลักสูตร', relevant, history=())

# Read test file
with open('test_thailmm.py', 'r', encoding='utf-8') as f:
    content = f.read()

test_user = re.search(r'user = """(.*?)"""', content, re.DOTALL)

print('=== SERVICE USER PROMPT (last 500 chars) ===')
print(user_prompt[-500:])
print()
print('=== TEST USER PROMPT (last 500 chars) ===')
print(test_user.group(1)[-500:])
print()
print('=== LENGTHS ===')
print(f'Service: {len(user_prompt)}')
print(f'Test: {len(test_user.group(1))}')
print()
# Find first difference
for i, (a, b) in enumerate(zip(user_prompt, test_user.group(1))):
    if a != b:
        print(f'First diff at {i}: {repr(a)} vs {repr(b)}')
        print(f'Service context: {repr(user_prompt[max(0,i-50):i+50])}')
        print(f'Test context: {repr(test_user.group(1)[max(0,i-50):i+50])}')
        break