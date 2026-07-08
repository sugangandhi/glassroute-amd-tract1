def choose_route(task_type: str, prompt: str):
    n = len(prompt)
    if task_type in {'sentiment', 'factual'} and n < 300:
        return 'small_model', 0.92
    if task_type in {'summarization', 'extraction'} and n < 1200:
        return 'small_model', 0.84
    if task_type == 'math':
        return 'large_model', 0.68
    if task_type == 'reasoning':
        return 'large_model', 0.70
    return 'large_model', 0.66
