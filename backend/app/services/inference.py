from app.adapters.fireworks_client import FireworksClient
from app.core.config import settings

fw = FireworksClient()

CLASSIFIER_PROMPT = 'Classify this task into one label only: factual, math, sentiment, summarization, reasoning, extraction. Task: '


def infer_with_small_model(prompt: str) -> dict:
    return fw.generate(prompt=prompt, model=settings.small_model_name, max_tokens=settings.max_output_tokens, temperature=0.1)


def infer_with_large_model(prompt: str) -> dict:
    return fw.generate(prompt=prompt, model=settings.large_model_name, max_tokens=settings.max_output_tokens, temperature=0.2)
