import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    app_env = os.getenv('APP_ENV', 'dev')
    fireworks_api_key = os.getenv('FIREWORKS_API_KEY', '')
    fireworks_base_url = os.getenv('FIREWORKS_BASE_URL', 'https://api.fireworks.ai/inference/v1')
    small_model_name = os.getenv('SMALL_MODEL_NAME', 'accounts/fireworks/models/llama-v3p1-8b-instruct')
    large_model_name = os.getenv('LARGE_MODEL_NAME', 'accounts/fireworks/models/llama-v3p1-70b-instruct')
    confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.80'))
    validation_threshold = float(os.getenv('VALIDATION_THRESHOLD', '0.65'))
    max_output_tokens = int(os.getenv('MAX_OUTPUT_TOKENS', '256'))
    max_retries = int(os.getenv('MAX_RETRIES', '1'))

settings = Settings()
