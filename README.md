# GlassRoute Hybrid Token-Efficient Routing Agent

A lightweight, Dockerized routing agent for the **AMD Developer Hackathon: ACT II – Track 1**. The agent processes a batch of natural-language tasks, routes low-risk tasks to deterministic local solvers, and sends harder tasks to Fireworks using an OpenAI-compatible API to minimize paid token usage while preserving answer quality.[1][2]

## Overview

The project is designed around the Track 1 evaluation pattern: solve tasks accurately, then reduce token spend wherever possible through selective routing.[1][2] It reads a task batch from `/input/tasks.json`, generates predictions, and writes results to `/output/predictions.json` in a container-friendly format suitable for automated evaluation.[3][4]

## Routing strategy

The agent uses a simple hybrid policy:

- **Local math solver** for straightforward arithmetic expressions.
- **Local sentiment solver** for obvious positive/negative sentiment classification.
- **Remote Fireworks inference** for factual, open-ended, and harder natural-language tasks.
- **Exact-match cache** to avoid repeated work inside a batch.

This routing style aligns with the token-efficiency goals emphasized across Track 1 submissions: easy tasks should be solved locally, while only harder tasks should consume remote inference tokens.[2][5]

## Features

- Batch task processing from JSON input.[3]
- Deterministic local solvers for zero-token math and sentiment handling.[6]
- Fireworks integration through the OpenAI-compatible chat completions API.[7]
- Docker-first execution for submission readiness.[4][8]
- Simple metadata logging for category, route, cache usage, and latency.

## Project structure

```text
backend/
├── Dockerfile
├── requirements.txt
└── app/
    └── main.py
```

## Input format

The container expects a JSON array mounted at `/input/tasks.json`.

Example:

```json
[
  {
    "id": "0",
    "prompt": "What is 12 * 7?"
  },
  {
    "id": "1",
    "prompt": "What is the capital of France?"
  }
]
```

## Output format

The container writes predictions to `/output/predictions.json`.

Example:

```json
[
  {
    "id": "0",
    "answer": "84",
    "meta": {
      "category": "math",
      "route": "local_math",
      "cached": false,
      "duration_seconds": 0.001
    }
  }
]
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `FIREWORKS_API_KEY` | Yes for remote tasks | None | API key for Fireworks inference.[9][10] |
| `FIREWORKS_BASE_URL` | No | `https://api.fireworks.ai/inference/v1` | OpenAI-compatible Fireworks base URL.[7] |
| `REMOTE_MODEL` | No | `accounts/fireworks/models/llama-v3p1-8b-instruct` | Remote model identifier used for fallback inference.[7] |
| `INPUT_PATH` | No | `/input/tasks.json` | Input task file path. |
| `OUTPUT_PATH` | No | `/output/predictions.json` | Output predictions file path. |
| `LOCAL_VLLM_URL` | No | empty | Optional OpenAI-compatible local endpoint. |
| `LOCAL_MODEL` | No | `local-model` | Model name for the optional local vLLM endpoint. |

## Local run

Install dependencies and run the script directly:

```bash
pip install -r requirements.txt
python app/main.py
```

For local testing, set environment variables first:

```bash
export FIREWORKS_API_KEY="YOUR_FIREWORKS_API_KEY"
export REMOTE_MODEL="accounts/fireworks/models/llama-v3p1-8b-instruct"
export INPUT_PATH=./input/tasks.json
export OUTPUT_PATH=./output/predictions.json
python app/main.py
```

On Windows PowerShell:

```powershell
$env:FIREWORKS_API_KEY="YOUR_FIREWORKS_API_KEY"
$env:REMOTE_MODEL="accounts/fireworks/models/llama-v3p1-8b-instruct"
$env:INPUT_PATH="./input/tasks.json"
$env:OUTPUT_PATH="./output/predictions.json"
python app/main.py
```

## Docker build

Build the image from the `backend/` directory:

```bash
docker build -t glassroute-router .
```

## Docker run

Run the container with mounted input/output directories:

### Linux/macOS

```bash
docker run --rm \
  -e FIREWORKS_API_KEY="YOUR_FIREWORKS_API_KEY" \
  -e REMOTE_MODEL="accounts/fireworks/models/llama-v3p1-8b-instruct" \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  glassroute-router
```

### Windows PowerShell

```powershell
docker run --rm `
  -e FIREWORKS_API_KEY="YOUR_FIREWORKS_API_KEY" `
  -e REMOTE_MODEL="accounts/fireworks/models/llama-v3p1-8b-instruct" `
  -v "${PWD}\input:/input" `
  -v "${PWD}\output:/output" `
  glassroute-router
```

## Submission notes

The evaluation harness for Track 1 supplies its own hidden task batch to `/input/tasks.json`, then reads the container output after execution.[3][4] For that reason, local sample `tasks.json` files are only for testing; the real evaluation data is injected by the organizers at runtime.[3][11]

## Design decisions

- **Accuracy first, token efficiency second:** low-risk tasks are solved locally whenever possible.[12][5]
- **Minimal dependencies:** the implementation stays lightweight and easy to containerize.[13]
- **OpenAI-compatible remote API:** simplifies Fireworks integration and model swapping.[7]
- **Simple heuristics over expensive routing:** the classifier avoids spending tokens just to decide where to send a task.[14][15]

## License

This project is intended for hackathon submission and experimentation. Add the license that matches your repository policy before publishing.
