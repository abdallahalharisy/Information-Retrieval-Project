"""Retrieval Service API."""

from fastapi import APIRouter, HTTPException

from shared.schemas import HealthResponse, SearchRequest, SearchResponse, SearchResultItem
from services.query_refinement.service import refine_query
from services.retrieval.service import RETRIEVAL_METHODS, retrieve

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(service="retrieval")


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if request.method not in RETRIEVAL_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Use ranking service for hybrid methods. Allowed: {sorted(RETRIEVAL_METHODS)}",
        )
    try:
        refined = refine_query(
            request.query,
            query_history=request.query_history,
        )
        results = retrieve(
            request.query,
            request.dataset,
            request.method,
            request.top_k,
            request.query_history,
            request.bm25_k1,
            request.bm25_b,
        )
        return SearchResponse(
            query=request.query,
            refined_query=refined["refined"],
            method=request.method,
            dataset=request.dataset,
            results=[SearchResultItem(**r) for r in results],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
