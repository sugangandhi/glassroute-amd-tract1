def classify_task(prompt: str) -> str:
    text = prompt.lower()
    if any(k in text for k in ['sentiment', 'positive', 'negative', 'tone']):
        return 'sentiment'
    if any(k in text for k in ['summarize', 'summary', 'tl;dr']):
        return 'summarization'
    if any(k in text for k in ['extract', 'json', 'entities', 'fields']):
        return 'extraction'
    if any(k in text for k in ['calculate', 'equation', 'solve', 'math', '+', '-', '*', '/']):
        return 'math'
    if any(k in text for k in ['why', 'compare', 'analyze', 'reason']):
        return 'reasoning'
    return 'factual'
