from config import API_KEY, BASE_URL, MODEL
from openai import OpenAI
from stage_1_function import call_llm_with_retry
import boolean

if __name__ == "__main__":
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    boolean.BooleanAlgebra.tokenize
    print(f"Base URL: {BASE_URL}")
    print(f"Model: {MODEL}")
    print("Testing connection...")

    result = call_llm_with_retry(
        client, MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, please reply with 'OK' if you receive this message."},
        ]
    )
    print(f"Response: {result}")
    print("Connection test passed.")