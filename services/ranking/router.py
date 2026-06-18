"""Ranking Service API."""

from fastapi import APIRouter, HTTPException

from shared.schemas import HealthResponse, SearchRequest, SearchResponse, SearchResultItem
from services.query_refinement.service import refine_query
from services.ranking.service import RANKING_METHODS, rank

router = APIRouter(prefix="/ranking", tags=["Ranking"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(service="ranking")


@router.post("/hybrid/serial", response_model=SearchResponse)
def hybrid_serial(request: SearchRequest):
    return _hybrid_search(request, "serial")


@router.post("/hybrid/parallel", response_model=SearchResponse)
def hybrid_parallel(request: SearchRequest):
    return _hybrid_search(request, "parallel")


@router.post("/hybrid/rrf", response_model=SearchResponse)
def hybrid_rrf(request: SearchRequest):
    return _hybrid_search(request, "rrf")


@router.post("/search", response_model=SearchResponse)
def hybrid_search(request: SearchRequest):
    if request.method not in RANKING_METHODS:
        raise HTTPException(status_code=400, detail=f"Hybrid method required: {sorted(RANKING_METHODS)}")
    return _hybrid_search(request, request.method)


def _hybrid_search(request: SearchRequest, method: str) -> SearchResponse:
    try:
        refined = refine_query(request.query, query_history=request.query_history)
        results = rank(
            request.query,
            request.dataset,
            method,
            request.top_k,
            request.query_history,
            request.bm25_k1,
            request.bm25_b,
        )
        return SearchResponse(
            query=request.query,
            refined_query=refined["refined"],
            method=method,
            dataset=request.dataset,
            results=[SearchResultItem(**r) for r in results],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
