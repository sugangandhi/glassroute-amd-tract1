from fastapi import APIRouter, HTTPException
from app.models.schemas import SolveRequest, SolveResponse
from app.runners.batch_runner import run_batch_file, solve_one

router = APIRouter()

@router.get('/health')
def health():
    return {'status': 'ok'}

@router.post('/solve', response_model=SolveResponse)
def solve(payload: SolveRequest):
    try:
        return solve_one(payload.task_id, payload.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/batch')
def batch():
    try:
        return run_batch_file('/app/input/tasks.json', '/app/output/predictions.json')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
