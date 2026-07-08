def estimate_text_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def estimate_tokens(text_in: str, text_out: str) -> int:
    return estimate_text_tokens(text_in) + estimate_text_tokens(text_out)
