import os
os.environ["THAILLM_API_KEY"] = "LKo7nialAX9hfwzxHr65RMzl5v96zN7N"
os.environ["THAILLM_API_URL"] = "https://playground.thaillm.or.th/v1/chat/completions"

from thaillm_rag.api_server import run_server

if __name__ == "__main__":
    run_server(port=8001)