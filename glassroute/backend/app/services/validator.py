def validate_answer(answer: str) -> bool:
    text = (answer or '').strip().lower()
    if len(text) < 2:
        return False
    bad_markers = ['i cannot help', 'as an ai language model', 'stub answer']
    return not any(m in text for m in bad_markers)
