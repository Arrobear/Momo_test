import os

# Force clear ALL proxy env vars
for key in list(os.environ.keys()):
    if 'proxy' in key.lower():
        print(f"Clearing: {key}={os.environ.pop(key)}")

# Now import and create client AFTER clearing
from openai import OpenAI

client = OpenAI(api_key="su8-8e384b5f169adcf2def9f570dd40aa9b", base_url="https://www.su8.codes/v1")
try:
    resp = client.chat.completions.create(model="gpt-5.5", messages=[{"role": "user", "content": "Say hi"}], max_tokens=10, timeout=30)
    print("OK:", resp.choices[0].message.content)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
