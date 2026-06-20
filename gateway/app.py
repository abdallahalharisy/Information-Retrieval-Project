"""API Gateway — single entry point for all SOA services."""

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.datasets import DATASETS
from shared.schemas import SearchRequest, SearchResponse, SearchResultItem
from services.preprocessing.router import router as preprocessing_router
from services.indexing.router import router as indexing_router
from services.retrieval.router import router as retrieval_router
from services.ranking.router import router as ranking_router
from services.query_refinement.router import router as query_refinement_router
from services.evaluation.router import router as evaluation_router
from services.rag.router import router as rag_router
from services.retrieval.service import RETRIEVAL_METHODS, retrieve
from services.ranking.service import RANKING_METHODS, rank
from services.query_refinement.service import refine_query, suggest
from shared.engine_registry import warmup_engines

app = FastAPI(
    title="IR Search Engine API Gateway",
    description="Service-Oriented Architecture gateway for the IR system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(preprocessing_router, prefix="/api/v1")
app.include_router(indexing_router, prefix="/api/v1")
app.include_router(retrieval_router, prefix="/api/v1")
app.include_router(ranking_router, prefix="/api/v1")
app.include_router(query_refinement_router, prefix="/api/v1")
app.include_router(evaluation_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")


@app.on_event("startup")
def start_engine_warmup():
    threading.Thread(target=warmup_engines, daemon=True).start()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gateway": "running",
        "services": [
            "preprocessing", "indexing", "retrieval",
            "ranking", "query_refinement", "evaluation", "rag",
        ],
    }


@app.post("/api/v1/warmup")
def warmup():
    warmup_engines()
    return {"status": "ready"}


@app.get("/api/v1/datasets")
def list_datasets():
    return {
        key: {
            "name": cfg["name"],
            "description": cfg["description"],
            "data_file": cfg["file"],
        }
        for key, cfg in DATASETS.items()
    }


@app.post("/api/v1/search", response_model=SearchResponse)
def unified_search(request: SearchRequest):
    """
    Orchestrated search: Query Refinement → Retrieval or Ranking.
    """
    try:
        refined = refine_query(
            request.query,
            expand_synonyms=False,
            do_spell_check=False,
            query_history=request.query_history,
        )
        suggestions = []
        if len(request.query.strip()) >= 2:
            try:
                suggestions = suggest(request.query.strip(), request.dataset)
            except Exception:
                pass

        method = request.resolved_method()
        if method in RETRIEVAL_METHODS:
            results = retrieve(
                request.query, request.dataset, method,
                request.top_k, request.query_history,
                request.bm25_k1, request.bm25_b,
            )
        elif method in RANKING_METHODS:
            results = rank(
                request.query, request.dataset, method,
                request.top_k, request.query_history,
                request.bm25_k1, request.bm25_b,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown method '{method}'. "
                       f"Retrieval: {sorted(RETRIEVAL_METHODS)} | "
                       f"Ranking: {sorted(RANKING_METHODS)}",
            )

        return SearchResponse(
            query=request.query,
            refined_query=refined["refined"],
            method=method,
            search_method=request.search_method,
            ranking_method=request.ranking_method,
            execution_mode=request.execution_mode,
            dataset=request.dataset,
            results=[SearchResultItem(**r) for r in results],
            suggestions=suggestions,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
