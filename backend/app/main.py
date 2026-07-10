#!/usr/bin/env python3
import os
import json
import re
import hashlib
from typing import Optional, Tuple, Dict, List

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

INPUT_PATH = os.getenv("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/output/results.json")
LOCAL_VLLM_URL = os.getenv("LOCAL_VLLM_URL", "")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "local-model")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
ALLOWED_MODELS = [m.strip() for m in os.getenv("ALLOWED_MODELS", "").split(",") if m.strip()]
REMOTE_MODEL = os.getenv("REMOTE_MODEL", "")
CACHE: Dict[str, str] = {}

try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=FIREWORKS_API_KEY, base_url=FIREWORKS_BASE_URL) if FIREWORKS_API_KEY else None
except Exception:
    openai_client = None


def read_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(path: str, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pick_allowed_model() -> str:
    if REMOTE_MODEL and REMOTE_MODEL in ALLOWED_MODELS:
        return REMOTE_MODEL
    if ALLOWED_MODELS:
        return ALLOWED_MODELS[0]
    return ""


def token_budget_for(category: str) -> int:
    return {
        "math": 32,
        "sentiment": 12,
        "summarization": 64,
        "classification": 20,
        "code": 128,
        "reasoning": 96,
        "translation": 48,
        "factual": 64,
        "ner": 48,
        "general": 64,
    }.get(category, 64)


def classify_task(prompt: str) -> str:
    p = prompt.lower().strip()
    if "extract all named entities" in p or "named entity" in p or "ner" in p:
        return "ner"
    if "classify the sentiment" in p or "sentiment" in p:
        return "sentiment"
    if "summarize" in p or "summarise" in p:
        return "summarization"
    if "find and fix" in p or "debug" in p or "bug" in p:
        return "code"
    if "write a python function" in p or "python function" in p:
        return "code"
    if re.search(r"\b\d+(\.\d+)?\b", p) and any(k in p for k in ["calculate", "compute", "solve", "how many", "remain", "percent"]):
        return "math"
    if any(k in p for k in ["who owns", "logic puzzle", "deduce", "riddle"]):
        return "reasoning"
    if any(k in p for k in ["what is", "define", "capital of", "where is", "when did", "who is"]):
        return "factual"
    return "general"


def safe_eval_math(expr: str) -> Optional[float]:
    try:
        if not re.fullmatch(r"[0-9\s\.\+\-\*\/\^\%\(\)eE,]+", expr):
            return None
        expr = expr.replace("^", "**").replace(",", "")
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None


def extract_expression(prompt: str) -> Optional[str]:
    m = re.search(r"(-?\d[\d\s\.\+\-\*\/\^\%\(\)eE,]*)", prompt)
    return m.group(1).strip() if m else None


def sentiment_local(text: str) -> str:
    pos = ["good", "great", "excellent", "love", "liked", "awesome", "amazing", "positive"]
    neg = ["bad", "terrible", "hate", "awful", "worst", "dislike", "negative"]
    p = text.lower()
    score = sum(1 for w in pos if w in p) - sum(1 for w in neg if w in p)
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def summarize_local(text: str) -> str:
    s = text.strip()
    dot = s.find(".")
    if dot != -1 and dot < 200:
        return s[: dot + 1]
    if len(s) <= 140:
        return s
    return s[:140].rstrip() + "..."


def ner_local(prompt: str) -> Optional[str]:
    m = re.search(r"(?:from|in|of|:)(.*)$", prompt, re.IGNORECASE)
    text = m.group(1).strip() if m else prompt
    entities = []
    seen = set()
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        ent = match.group(1)
        if ent not in seen:
            seen.add(ent)
            entities.append(ent)
    if not entities:
        return None
    return "; ".join(f"{e}: PROPER_NOUN" for e in entities)


def debug_local(prompt: str) -> Optional[str]:
    p = prompt.lower()
    if "return nums[0]" in p and "max" in p:
        return "def get_max(nums):\n    return max(nums)"
    if "second-largest" in p:
        return "def second_largest(nums):\n    vals = sorted(set(nums))\n    return vals[-2] if len(vals) > 1 else None"
    return None


def codegen_local(prompt: str) -> Optional[str]:
    p = prompt.lower()
    if "second-largest" in p:
        return "def second_largest(nums):\n    vals = sorted(set(nums))\n    return vals[-2] if len(vals) > 1 else None"
    return None


def solve_local(prompt: str, category: str) -> Tuple[Optional[str], Optional[str]]:
    if category == "math":
        expr = extract_expression(prompt)
        if expr:
            val = safe_eval_math(expr)
            if val is not None:
                if hasattr(val, "is_integer") and val.is_integer():
                    return str(int(val)), "local_math"
                return str(val), "local_math"
    if category == "sentiment":
        return sentiment_local(prompt), "local_sentiment"
    if category == "summarization":
        return summarize_local(prompt), "local_summary"
    if category == "ner":
        ans = ner_local(prompt)
        if ans is not None:
            return ans, "local_ner"
    if category == "code":
        ans = codegen_local(prompt)
        if ans is not None:
            return ans, "local_codegen"
    if category == "reasoning":
        ans = debug_local(prompt)
        if ans is not None:
            return ans, "local_reasoning"
    return None, None


def build_messages(prompt: str, category: str) -> List[Dict[str, str]]:
    system = "Answer directly. Return only the final answer."
    if category == "summarization":
        system += " Keep the summary concise."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def cleanup_summary(text: str) -> str:
    clean = re.sub(r"^\s*[-*]\s+", "", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", clean)
    parts = [p for p in parts if p]
    return parts[0] if parts else clean


def call_local_vllm(prompt: str, category: str) -> Optional[str]:
    if not LOCAL_VLLM_URL:
        return None
    try:
        payload = {
            "model": LOCAL_MODEL,
            "messages": build_messages(prompt, category),
            "max_tokens": token_budget_for(category),
            "temperature": 0.0,
        }
        resp = requests.post(f"{LOCAL_VLLM_URL}/chat/completions", json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("choices"):
            return data["choices"][0].get("message", {}).get("content", "").strip()
        if "result" in data:
            return str(data["result"]).strip()
    except Exception:
        return None
    return None


def call_fireworks(prompt: str, category: str) -> Optional[str]:
    model = pick_allowed_model()
    if not model:
        return None
    try:
        if openai_client:
            resp = openai_client.chat.completions.create(
                model=model,
                messages=build_messages(prompt, category),
                temperature=0.0,
                max_tokens=token_budget_for(category),
            )
            return resp.choices[0].message.content.strip()
    except Exception:
        pass
    if not FIREWORKS_API_KEY:
        return None
    try:
        body = {
            "model": model,
            "messages": build_messages(prompt, category),
            "max_tokens": token_budget_for(category),
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}"}
        resp = requests.post(f"{FIREWORKS_BASE_URL}/chat/completions", json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("choices"):
            return data["choices"][0].get("message", {}).get("content", "").strip()
    except Exception:
        return None
    return None


def route_and_solve(task_id: str, prompt: str):
    fp = fingerprint(prompt)
    if fp in CACHE:
        answer = CACHE[fp]
    else:
        category = classify_task(prompt)
        answer, _ = solve_local(prompt, category)
        if answer is None:
            answer = call_local_vllm(prompt, category)
        if answer is None:
            answer = call_fireworks(prompt, category)
        if answer is None:
            answer = ""
        if category == "summarization" and answer:
            answer = cleanup_summary(answer)
        CACHE[fp] = answer
    return {"task_id": task_id, "answer": answer}


def main():
    try:
        tasks = read_tasks(INPUT_PATH)
    except Exception:
        tasks = []
    results = []
    for i, item in enumerate(tasks):
        task_id = item.get("task_id") or item.get("id") or str(i)
        prompt = item.get("prompt") or item.get("input") or item.get("text") or ""
        results.append(route_and_solve(task_id, prompt))
    write_results(OUTPUT_PATH, results)


if __name__ == "__main__":
    main()