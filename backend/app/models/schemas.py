from pydantic import BaseModel

class SolveRequest(BaseModel):
    task_id: str
    prompt: str
    category_hint: str | None = None

class SolveResponse(BaseModel):
    task_id: str
    route: str
    task_type: str
    confidence: float
    tokens_input: int
    tokens_output: int
    tokens_total: int
    baseline_tokens_estimate: int
    tokens_saved_estimate: int
    valid: bool
    answer: str

class BatchItem(BaseModel):
    task_id: str
    prompt: str
    category_hint: str | None = None
