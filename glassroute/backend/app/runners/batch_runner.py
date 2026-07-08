import json
from app.services.classifier import classify_task
from app.services.router import choose_route
from app.services.validator import validate_answer
from app.services.token_meter import estimate_tokens, estimate_text_tokens
from app.services.inference import infer_with_small_model, infer_with_large_model


def solve_one(task_id: str, prompt: str):
    task_type = classify_task(prompt)
    route, confidence = choose_route(task_type, prompt)
    result = infer_with_small_model(prompt) if route == 'small_model' else infer_with_large_model(prompt)
    answer = result['answer']
    valid = validate_answer(answer)

    if route == 'small_model' and (not valid or confidence < 0.75):
        route = 'large_model_fallback'
        confidence = 0.78
        result = infer_with_large_model(prompt)
        answer = result['answer']
        valid = validate_answer(answer)

    tokens_input = result.get('prompt_tokens') or estimate_text_tokens(prompt)
    tokens_output = result.get('completion_tokens') or estimate_text_tokens(answer)
    tokens_total = result.get('total_tokens') or estimate_tokens(prompt, answer)
    baseline = max(tokens_total + 120, int(tokens_total * 1.8))

    return {
        'task_id': task_id,
        'route': route,
        'task_type': task_type,
        'confidence': confidence,
        'tokens_input': tokens_input,
        'tokens_output': tokens_output,
        'tokens_total': tokens_total,
        'baseline_tokens_estimate': baseline,
        'tokens_saved_estimate': max(baseline - tokens_total, 0),
        'valid': valid,
        'answer': answer,
    }


def run_batch_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    results = [solve_one(item['task_id'], item['prompt']) for item in tasks]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    return {'count': len(results), 'output_path': output_path}
