"""Evaluation Service API."""

from fastapi import APIRouter, HTTPException

from shared.schemas import (
    EvaluateComparisonRequest,
    EvaluateComparisonSummary,
    EvaluateRequest,
    EvaluateSummary,
    HealthResponse,
)
from services.evaluation.service import EvaluationUnavailableError, run_comparison, run_eval

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(service="evaluation")


@router.post("/run", response_model=EvaluateSummary)
def evaluate(request: EvaluateRequest):
    try:
        result = run_eval(
            request.dataset,
            request.method,
            k=request.k,
            limit=request.limit,
            mode=request.mode,
        )
        return EvaluateSummary(**result)
    except EvaluationUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/compare", response_model=EvaluateComparisonSummary)
def compare(request: EvaluateComparisonRequest):
    try:
        result = run_comparison(
            request.dataset,
            request.methods,
            k=request.k,
            limit=request.limit,
            include_embeddings=request.include_embeddings,
        )
        return EvaluateComparisonSummary(**result)
    except EvaluationUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
