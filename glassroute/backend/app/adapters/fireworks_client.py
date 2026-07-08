import httpx
from app.core.config import settings

class FireworksClient:
    def __init__(self):
        self.api_key = settings.fireworks_api_key
        self.base_url = settings.fireworks_base_url.rstrip('/')

    def generate(self, prompt: str, model: str, max_tokens: int = 256, temperature: float = 0.2) -> dict:
        if not self.api_key:
            raise RuntimeError('FIREWORKS_API_KEY is not set')
        url = f"{self.base_url}/chat/completions"
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'Be concise. Return only the best final answer.'},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        answer = data['choices'][0]['message']['content'].strip()
        usage = data.get('usage', {})
        return {
            'answer': answer,
            'prompt_tokens': usage.get('prompt_tokens'),
            'completion_tokens': usage.get('completion_tokens'),
            'total_tokens': usage.get('total_tokens'),
            'raw': data,
        }
