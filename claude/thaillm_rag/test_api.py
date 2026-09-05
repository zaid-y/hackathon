import os
os.environ["THAILLM_API_KEY"] = "LKo7nialAX9hfwzxHr65RMzl5v96zN7N"
os.environ["THAILLM_API_URL"] = "https://playground.thaillm.or.th/v1/chat/completions"

import requests

# Test health
r = requests.get('http://localhost:8001/api/health', timeout=5)
print('Health:', r.status_code, r.json())

# Test query
r = requests.post('http://localhost:8001/api/query', json={'query': 'เงื่อนไขการสมัครคืออะไร?', 'mode': 'enhanced'}, timeout=60)
print('Query:', r.status_code)
import json
print(json.dumps(r.json(), ensure_ascii=False, indent=2))