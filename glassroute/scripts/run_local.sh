#!/usr/bin/env bash
cd backend && uvicorn app.main:app --reload --port 8000
