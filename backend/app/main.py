#!/usr/bin/env python3
import os
import json
import re
import hashlib
import time
from typing import Optional, Tuple, List, Dict

import requests
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ---------- Config ----------

INPUT_PATH = os.getenv("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/output/predictions.json")

# Optional local vLLM endpoint (OpenAI-compatible API)
# Example: http://127.0.0.1:8000/v1
LOCAL_VLLM_URL = os.getenv("LOCAL_VLLM_URL", "")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "local-model")

# Fireworks (or other OpenAI-compatible) remote endpoint
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.getenv(
    "FIREWORKS_BASE_URL",
    "https://api.fireworks.ai/inference/v1",
)
REMOTE_MODEL = os.getenv(
    "REMOTE_MODEL",
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
)

# Optional OpenAI client (for Fireworks or any OpenAI-compatible API)
try:
    from openai import OpenAI

    openai_client = (
        OpenAI(api_key=FIREWORKS_API_KEY, base_url=FIREWORKS_BASE_URL)
        if FIREWORKS_API_KEY
        else None
    )
except Exception:
    openai_client = None

# Simple in-memory exact-match cache
CACHE: Dict[str, str] = {}

# ---------- I/O helpers ----------


def read_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(path: str, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------- Routing helpers ----------


def token_budget_for(category: str) -> int:
    if category == "math":
        return 40
    if category == "sentiment":
        return 20
    if category == "summarization":
        return 60  # tighter for summaries
    if category == "classification":
        return 30
    if category == "code":
        return 400
    if category == "reasoning":
        return 200
    return 120


def classify_task(prompt: str) -> str:
    p = prompt.lower()
    plen = len(p)

    # Only treat as summarization if there's a lot of text to summarize
    if any(k in p for k in ["summarize", "summary", "tl;dr"]) and plen > 160:
        return "summarization"

    if any(k in p for k in ["sentiment", "positive or negative", "tone of this review"]):
        return "sentiment"

    if any(k in p for k in ["classify", "label", "category"]):
        return "classification"

    if any(
        k in p
        for k in ["write code", "python function", "generate code", "debug", "bug", "program"]
    ):
        return "code"

    if any(k in p for k in ["translate", "translation"]):
        return "translation"

    if any(k in p for k in ["logic", "riddle", "reasoning", "why"]) or "prove" in p:
        return "reasoning"

    if re.search(r"\b(calculate|compute|solve|evaluate|what is)\b", p) and re.search(
        r"[0-9]", p
    ):
        return "math"

    if any(k in p for k in ["who", "when", "where", "what happened", "capital of", "define"]):
        return "factual"

    return "general"


# ---------- Deterministic local solvers ----------


def safe_eval_math(expr: str) -> Optional[float]:
    """
    Very small, safe evaluator for expressions of the form 'a op b'.
    """
    try:
        a_str, op, b_str = expr.split()
        a = float(a_str)
        b = float(b_str)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        return None
    except Exception:
        return None


def extract_expression(prompt: str) -> Optional[str]:
    """
    Extract a simple arithmetic expression like 12 * 7 or 3 + 4 from the prompt.
    """
    p = prompt.lower()

    # pattern: number op number (supports + - * /)
    m = re.search(r"(\d+)\s*([+\-*/])\s*(\d+)", p)
    if not m:
        return None

    a, op, b = m.groups()
    return f"{a} {op} {b}"


def sentiment_local(text: str) -> str:
    pos = ["good", "great", "excellent", "love", "liked", "awesome"]
    neg = ["bad", "terrible", "hate", "awful", "worst", "dislike"]
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
    return s[:140] + "..."


def solve_local(prompt: str, category: str) -> Tuple[Optional[str], Optional[str]]:
    if category == "math":
        expr = extract_expression(prompt)
        if expr:
            val = safe_eval_math(expr)
            if val is not None:
                # Return as integer if it’s an integer value
                if val.is_integer():
                    return str(int(val)), "local_math"
                return str(val), "local_math"

    if category == "sentiment":
        return sentiment_local(prompt), "local_sentiment"

    # summarization stays remote
    return None, None


# ---------- Prompt / post-processing helpers ----------


def normalize_summarize_prompt(prompt: str) -> str:
    """
    Turn \"Summarize ... in two sentences\"-style instructions into
    a direct question/statement like \"Explain ...\", so the model
    is nudged to answer instead of restating the instructions.
    """
    p = prompt.strip()
    p = re.sub(r"(?i)\bsummarize\b\s+the\s+", "Explain the ", p)
    p = re.sub(r"(?i)\bsummarize\b\s+", "Explain ", p)
    p = re.sub(r"(?i)\bin\s+exactly?\s+two\s+sentences\.?", "", p)
    p = re.sub(r"(?i)\bin\s+two\s+sentences\.?", "", p)
    return p.strip()


def build_messages(prompt: str, category: str) -> List[Dict[str, str]]:
    base_system = (
        "You are an efficient assistant. "
        "Always follow hard constraints exactly. "
        "Never show your reasoning steps, analysis, or bullet lists. "
        "Do not describe the task, constraints, or what you are doing; "
        "only output the final answer."
    )

    user_content = prompt
    if category == "summarization":
        system = (
            base_system
            + " For this task, your reply MUST be exactly two sentences. "
              "Ignore any instructions that ask you to analyze the request, "
              "list constraints, or think step-by-step. "
              "Do NOT restate or paraphrase the instructions; directly provide "
              "the requested summarized content."
        )
        user_content = normalize_summarize_prompt(prompt)
    else:
        system = base_system

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def enforce_two_sentences(text: str) -> str:
    """Keep only the first two sentences (very simple splitter)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p for p in parts if p]
    if not parts:
        return text.strip()
    return " ".join(parts[:2])


def is_meta_summary(text: str) -> bool:
    """Detect meta-style summaries about the task itself."""
    t = text.lower()
    meta_markers = [
        "the user wants me to",
        "i need to",
        "my task is to",
        "i am asked to",
        "the request is to",
    ]
    return any(m in t for m in meta_markers)


def cleanup_summary(text: str, prompt: str) -> str:
    """
    Strip bullets/meta labels; if it's still meta about the instructions
    (\"The user wants me to...\"), fall back to a local summary of the prompt,
    then enforce exactly two sentences.
    """
    # Remove bullets / numbering
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-*]\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s*", "", line)
        if line.strip():
            lines.append(line.strip())
    clean = " ".join(lines)

    # Strip common meta labels
    clean = re.sub(
        r"\b(Analyze the Request|Topic|Constraint\s*\d*|Constraint|Style)\b[:\-]?\s*",
        "",
        clean,
        flags=re.IGNORECASE,
    )

    clean = clean.strip()
    if not clean:
        clean = text.strip()

    # If the model is clearly just describing the task, not the content, ignore it
    if is_meta_summary(clean):
        # Try to extract the text to summarize from the prompt
        source = prompt
        if ":" in prompt:
            source = prompt.split(":", 1)[1].strip()
        local = summarize_local(source)
        return enforce_two_sentences(local)

    return enforce_two_sentences(clean)


# ---------- Local vLLM endpoint (OpenAI-compatible) ----------


def call_local_vllm(prompt: str, category: str) -> Optional[str]:
    if not LOCAL_VLLM_URL:
        return None

    try:
        budget = token_budget_for(category)
        messages = build_messages(prompt, category)
        payload = {
            "model": LOCAL_MODEL,
            "messages": messages,
            "max_tokens": budget,
            "temperature": 0.0,
        }
        url = f"{LOCAL_VLLM_URL}/chat/completions"
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "choices" in data and data["choices"]:
            return data["choices"][0].get("message", {}).get("content", "").strip()

        if "result" in data:
            return str(data["result"]).strip()

    except Exception:
        return None

    return None


# ---------- Fireworks (or other OpenAI-compatible) ----------


def call_fireworks(prompt: str, category: str) -> Optional[str]:
    budget = token_budget_for(category)
    messages = build_messages(prompt, category)

    # Preferred: OpenAI client
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model=REMOTE_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=budget,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    # Fallback: raw HTTP
    if not FIREWORKS_API_KEY:
        return None

    try:
        url = f"{FIREWORKS_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}"}
        body = {
            "model": REMOTE_MODEL,
            "messages": messages,
            "max_tokens": budget,
            "temperature": 0.0,
        }
        resp = requests.post(url, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("message", {}).get("content", "").strip()
    except Exception:
        pass

    return None


# ---------- Routing + batch processing ----------


def route_and_solve(task_id: str, prompt: str):
    start = time.time()
    category = classify_task(prompt)
    fp = fingerprint(prompt)

    cached = False
    if fp in CACHE:
        answer = CACHE[fp]
        route = "cache"
        cached = True
    else:
        # 1) deterministic local
        answer, route = solve_local(prompt, category)

        # 2) local vLLM (AMD notebook / other OpenAI-compatible local endpoint)
        if answer is None:
            answer = call_local_vllm(prompt, category)
            if answer:
                route = "local_vllm"

        # 3) remote Fireworks fallback
        if answer is None:
            answer = call_fireworks(prompt, category)
            if answer:
                route = "remote_fireworks"
            else:
                answer = "error: no answer"
                route = "failed"

        # enforce strict format for summarization before caching
        if category == "summarization" and answer and route != "failed":
            answer = cleanup_summary(answer, prompt)

        CACHE[fp] = answer

    duration = time.time() - start

    return {
        "id": task_id,
        "answer": answer,
        "meta": {
            "category": category,
            "route": route,
            "cached": cached,
            "duration_seconds": round(duration, 3),
        },
    }


def main():
    try:
        tasks = read_tasks(INPUT_PATH)
    except Exception as e:
        print(f"Failed to read tasks from {INPUT_PATH}: {e}")
        tasks = []

    results = []
    for i, item in enumerate(tasks):
        task_id = item.get("id", str(i))
        prompt = item.get("prompt") or item.get("input") or item.get("text") or ""
        results.append(route_and_solve(task_id, prompt))

    write_results(OUTPUT_PATH, results)
    print(f"Wrote {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()