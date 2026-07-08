# GlassRoute

Working Fireworks-based starter for a token-efficient hybrid routing agent.

## Setup
1. Copy `.env.example` to `.env`
2. Put your Fireworks API key in `FIREWORKS_API_KEY`
3. Start backend locally or with Docker

## Local backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Run batch
```bash
cd scripts
./run_batch.sh
```

## Endpoints
- `GET /health`
- `POST /solve`
- `POST /batch`

## Notes
- Small model is used first for cheaper tasks.
- Large model is used for harder tasks or fallback.
- Usage values are pulled from Fireworks when available.
