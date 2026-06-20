"""RAG Service API."""

from fastapi import APIRouter, HTTPException

from shared.schemas import HealthResponse, RagRequest, RagResponse
from services.rag.service import answer_with_rag

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(service="rag")


@router.post("/answer", response_model=RagResponse)
def answer(request: RagRequest):
    try:
        return answer_with_rag(
            query=request.query,
            dataset=request.dataset,
            search_method=request.search_method,
            ranking_method=request.ranking_method,
            execution_mode=request.execution_mode,
            method=request.method,
            top_k=request.top_k,
            context_k=request.context_k,
            query_history=request.query_history,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b,
            model=request.model,
            rag_mode=request.rag_mode,
            temperature=request.temperature,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
