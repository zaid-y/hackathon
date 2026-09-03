import requests
import json

url = "https://api.thaillm.or.th/v1/chat"
api_key = "LKo7nialAX9hfwzxHr65RMzl5v96zN7N"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "model": "OpenThaiGPT-ThaiLLM-8B-Instruct-v7.2",
    "messages": [{"role": "user", "content": "สวัสดี"}]
}

print("Testing with Bearer token...")
try:
    response = requests.post(url, headers=headers, json=data, timeout=30)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Try X-API-Key
headers2 = {
    "Content-Type": "application/json",
    "X-API-Key": api_key
}

print("\nTesting with X-API-Key...")
try:
    response = requests.post(url, headers=headers2, json=data, timeout=30)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Try api_key in body
data3 = {
    "model": "OpenThaiGPT-ThaiLLM-8B-Instruct-v7.2",
    "messages": [{"role": "user", "content": "สวัสดี"}],
    "api_key": api_key
}

print("\nTesting with api_key in body...")
try:
    response = requests.post(url, headers={"Content-Type": "application/json"}, json=data3, timeout=30)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")