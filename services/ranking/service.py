"""Ranking service — hybrid fusion strategies."""

from typing import List

from shared.engine_registry import get_engine

RANKING_METHODS = {"serial", "parallel", "rrf"}


def rank(query: str, dataset: str, method: str, top_k: int = 10,
         query_history: List[str] = None, bm25_k1: float = None, bm25_b: float = None) -> List[dict]:
    if method not in RANKING_METHODS:
        raise ValueError(f"Unknown ranking method: {method}. Use one of {sorted(RANKING_METHODS)}")

    engine = get_engine(dataset)
    if bm25_k1 is not None or bm25_b is not None:
        engine.bm25.update_parameters(k1=bm25_k1, b=bm25_b)

    return engine.search(query, method=method, top_k=top_k, query_history=query_history or [])
