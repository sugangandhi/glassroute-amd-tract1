#!/usr/bin/env bash
vllm serve $SMALL_MODEL_NAME --host 0.0.0.0 --port 8001
