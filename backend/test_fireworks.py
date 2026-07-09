import os
from openai import OpenAI

api_key = os.getenv("FIREWORKS_API_KEY", "")
base_url = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
model = os.getenv("REMOTE_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct")

print("API key set:", bool(api_key))
print("Base URL:", base_url)
print("Model:", model)

client = OpenAI(api_key=api_key, base_url=base_url)

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say 'hello' in one short sentence."}],
    max_tokens=32,
    temperature=0.0,
)
print(resp.choices[0].message.content)