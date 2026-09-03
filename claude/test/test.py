"""
#Step 1 example: Calling an LLM API safely from Python.

#This demonstrates the core skills you need:
#- Reading an API key from an environment variable (never hardcode it!)
#- Sending a POST request with JSON
#- Handling network / API errors gracefully
#- Structuring a question + context prompt (the pattern you'll reuse for ThaiLLM)

#Before running:
#pip install requests
#export LLM_API_KEY="your-key-here"       (Mac/Linux)
#set LLM_API_KEY=your-key-here            (Windows cmd)

#Swap API_URL, headers, and the request body shape to match ThaiLLM's actual
#API once you have their docs -- the STRUCTURE of this script stays the same.
"""

import os
import requests


API_URL = "https://api.example-thaillm.com/v1/chat"  # placeholder — replace with real endpoint
API_KEY = os.environ.get("LLM_API_KEY")


def call_llm(question: str, context: str = "") -> str:
    """
    Send a question (optionally with retrieved document context) to the LLM
    and return its text answer. Raises a clear error if something goes wrong,
    rather than crashing with an ugly traceback.
    """
    if not API_KEY:
        raise RuntimeError(
            "No API key found. Set the LLM_API_KEY environment variable."
        )

    # This is the prompt template you'll reuse in step 5 (prompt engineering).
    # For now, keep it simple -- just prove the call works end to end.
    system_prompt = (
        "You answer questions using only the provided context. "
        "If the answer isn't in the context, say you don't know."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

    payload = {
        "model": "thaillm-default",  # replace with the real model name
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()  # raises an error for 4xx/5xx responses
    except requests.exceptions.Timeout:
        raise RuntimeError("Request timed out. Check your connection or the API status.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"API returned an error: {e}\nResponse body: {response.text}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error calling the API: {e}")

    data = response.json()

    # Adjust this line once you see ThaiLLM's real response shape --
    # this assumes an OpenAI-style {"choices": [{"message": {"content": ...}}]} format.
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected response format: {data}")


if __name__ == "__main__":
    # Quick manual test — this is what you run first to prove the pipeline works.
    fake_context = (
        "Applicants must be currently enrolled high school or vocational "
        "students. Teams may have 2-4 members. Registration closes at 9:00 AM "
        "on the day of the event."
    )
    question = "What are the requirements for applying?"

    try:
        answer = call_llm(question, fake_context)
        print("ANSWER:\n", answer)
    except RuntimeError as e:
        print("Something went wrong:", e)