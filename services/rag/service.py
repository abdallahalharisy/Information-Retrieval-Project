"""RAG service wrapper around the production hybrid RAG pipeline."""

from collections import OrderedDict
from typing import List, Optional

from shared.schemas import RagResponse, resolve_engine_method
from services.query_refinement.service import refine_query
from services.rag.pipeline import run_hybrid_rag

MAX_RAG_CACHE_SIZE = 64

_rag_cache = OrderedDict()


def answer_with_rag(
    query: str,
    dataset: str = "msmarco",
    search_method: str = "bm25",
    ranking_method: Optional[str] = "none",
    execution_mode: str = "parallel",
    method: Optional[str] = None,
    top_k: int = 10,
    context_k: int = 4,
    query_history: Optional[List[str]] = None,
    bm25_k1: Optional[float] = None,
    bm25_b: Optional[float] = None,
    model: Optional[str] = None,
    rag_mode: str = "hybrid",
    temperature: float = 0.2,
) -> RagResponse:
    requested_method = resolve_engine_method(method, search_method, ranking_method)
    cache_key = (
        query,
        dataset,
        requested_method,
        top_k,
        context_k,
        tuple((query_history or [])[-3:]),
        bm25_k1,
        bm25_b,
        model,
        rag_mode,
        temperature,
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    refined = refine_query(query, query_history=query_history or [])
    smalltalk_answer = _smalltalk_answer(query)
    if smalltalk_answer:
        response = RagResponse(
            query=query,
            refined_query=refined["refined"],
            method="hybrid_rag",
            search_method=search_method,
            ranking_method=ranking_method,
            execution_mode=execution_mode,
            dataset=dataset,
            model="smalltalk",
            answer=smalltalk_answer,
            sources=[],
            generated_answer=smalltalk_answer,
            source_documents=[],
        )
        _cache_set(cache_key, response)
        return response

    pipeline_result = run_hybrid_rag(
        query,
        dataset=dataset,
        top_n=top_k,
        context_k=context_k,
        model=model,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        use_semantic=(rag_mode == "hybrid"),
        temperature=temperature,
    )
    warning = pipeline_result.warning
    if len(pipeline_result.source_documents) > len(pipeline_result.context_documents):
        display_note = (
            f"Showing {len(pipeline_result.source_documents)} fused source documents; "
            f"Gemini used the top {len(pipeline_result.context_documents)} as context."
        )
        warning = f"{warning} {display_note}" if warning else display_note

    response = RagResponse(
        query=query,
        refined_query=pipeline_result.refined_query,
        method="hybrid_rag",
        search_method=search_method,
        ranking_method=ranking_method,
        execution_mode=execution_mode,
        dataset=dataset,
        model=pipeline_result.model,
        answer=pipeline_result.generated_answer,
        sources=pipeline_result.source_documents,
        generated_answer=pipeline_result.generated_answer,
        source_documents=pipeline_result.source_documents,
        warning=warning,
    )
    _cache_set(cache_key, response)
    return response


def _smalltalk_answer(query: str) -> Optional[str]:
    normalized = " ".join(query.lower().strip().split())
    if normalized in {"hi", "hello", "hey", "مرحبا", "اهلا", "أهلا", "هاي"}:
        return "Hi. Ask me a question about the indexed MSMARCO documents and I will retrieve the most relevant sources."
    return None


def _cache_get(key):
    if key not in _rag_cache:
        return None
    value = _rag_cache.pop(key)
    _rag_cache[key] = value
    return value


def _cache_set(key, value):
    _rag_cache[key] = value
    if len(_rag_cache) > MAX_RAG_CACHE_SIZE:
        _rag_cache.popitem(last=False)
