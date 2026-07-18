import os

for key in list(os.environ.keys()):
    if 'proxy' in key.lower():
        os.environ.pop(key)

API_KEY = "su8-8e384b5f169adcf2def9f570dd40aa9b"
BASE_URL = "https://www.su8.codes/v1"
MODEL = "gpt-5.3-codex-spark"

from openai import OpenAI


def make_client():
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        default_headers={"User-Agent": "python-httpx/0.28.1"},
    )


def test_chat():
    print(f"\n{'='*50}")
    print("Test 1: /chat/completions")
    print(f"BASE_URL: {BASE_URL}")
    print(f"MODEL: {MODEL}")
    print(f"{'='*50}")

    client = make_client()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "用中文说'Hello, test OK!'"}],
            max_tokens=50,
            timeout=120,
        )
        content = resp.choices[0].message.content
        print(f"[OK] {content}")
        return True
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return False


def test_responses():
    print(f"\n{'='*50}")
    print("Test 2: /responses")
    print(f"{'='*50}")

    client = make_client()
    try:
        resp = client.responses.create(
            model=MODEL,
            input="用中文说'Responses OK!'",
            max_output_tokens=50,
            timeout=120,
        )
        print(f"[OK] {resp.output_text}")
        return True
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return False


def test_chat_stream():
    print(f"\n{'='*50}")
    print("Test 3: /chat/completions (stream)")
    print(f"{'='*50}")

    client = make_client()
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Count from 1 to 5 in English."}],
            max_tokens=100,
            timeout=120,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        content = "".join(chunks)
        print(f"[OK] {content}")
        return True
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    results = {}
    results["chat"] = test_chat()
    results["responses"] = test_responses()
    results["stream"] = test_chat_stream()

    print(f"\n{'='*50}")
    print("SUMMARY:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAIL'}")
    if all(results.values()):
        print("All tests passed - API fully available!")
    elif any(results.values()):
        print("Partial success - some endpoints work, check above for details.")
    else:
        print("All tests failed - API not available.")
    print(f"{'='*50}")
