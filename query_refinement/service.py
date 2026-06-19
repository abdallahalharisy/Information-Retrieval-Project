"""Query Refinement Service API."""

from fastapi import APIRouter, HTTPException

from shared.schemas import (
    HealthResponse,
    RefineQueryRequest,
    RefineQueryResponse,
    SuggestRequest,
    SuggestResponse,
)
from services.query_refinement.service import refine_query, suggest

router = APIRouter(prefix="/query-refinement", tags=["Query Refinement"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(service="query_refinement")


@router.post("/refine", response_model=RefineQueryResponse)
def refine(request: RefineQueryRequest):
    result = refine_query(
        request.query,
        expand_synonyms=request.expand_synonyms,
        do_spell_check=request.do_spell_check,
        query_history=request.query_history,
    )
    return RefineQueryResponse(**result)


@router.post("/suggest", response_model=SuggestResponse)
def suggest_terms(request: SuggestRequest):
    try:
        suggestions = suggest(request.prefix, request.dataset, request.limit)
        return SuggestResponse(prefix=request.prefix, suggestions=suggestions)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
