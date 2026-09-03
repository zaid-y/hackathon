from app.config import get_settings
from app.answer import RAGRuntime
import re

settings = get_settings()
runtime = RAGRuntime(settings)
service = runtime.ensure_ready()

# Monkey-patch to capture raw response
original_answer = service.provider.answer
captured = {}
def capture_answer(system_prompt, user_prompt):
    response = original_answer(system_prompt, user_prompt)
    captured['system'] = system_prompt
    captured['user'] = user_prompt
    captured['response'] = response
    return response

service.provider.answer = capture_answer

result = service.answer('หลักสูตร')

# Compare with the test file that worked
with open('test_thailmm.py', 'r', encoding='utf-8') as f:
    content = f.read()

test_system = re.search(r'system = """(.*?)"""', content, re.DOTALL)
test_user = re.search(r'user = """(.*?)"""', content, re.DOTALL)

print('=== SYSTEM PROMPT COMPARISON ===')
print('Equal:', captured['system'] == test_system.group(1))
if captured['system'] != test_system.group(1):
    for i, (a, b) in enumerate(zip(captured['system'], test_system.group(1))):
        if a != b:
            print('Diff at', i, ':', repr(a), 'vs', repr(b))
            print('  Service:', repr(captured['system'][max(0,i-20):i+20]))
            print('  Test:', repr(test_system.group(1)[max(0,i-20):i+20]))
            break

print()
print('=== USER PROMPT COMPARISON ===')
print('Equal:', captured['user'] == test_user.group(1))
if captured['user'] != test_user.group(1):
    for i, (a, b) in enumerate(zip(captured['user'], test_user.group(1))):
        if a != b:
            print('Diff at', i, ':', repr(a), 'vs', repr(b))
            print('  Service:', repr(captured['user'][max(0,i-20):i+20]))
            print('  Test:', repr(test_user.group(1)[max(0,i-20):i+20]))
            break