class VLLMClient:
    def generate(self, prompt: str) -> str:
        return f"vLLM response for: {prompt[:120]}"
