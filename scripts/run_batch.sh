#!/usr/bin/env bash
cd backend && python -c "from app.runners.batch_runner import run_batch_file; print(run_batch_file('../input/tasks.json','../output/predictions.json'))"
