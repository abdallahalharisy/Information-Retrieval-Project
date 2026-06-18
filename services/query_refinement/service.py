"""Query refinement service."""

from typing import List

from config import QUERY_REFINEMENT
from preprocess import refine_query_text
from shared.engine_registry import get_engine


def refine_query(query: str, expand_synonyms: bool = True, do_spell_check: bool = True,
                 query_history: List[str] = None) -> dict:
    history = None
    if query_history and QUERY_REFINEMENT.get("use_search_history", True):
        history = [
            refine_query_text(q, expand_synonyms=False, do_spell_check=False, max_synonyms_per_term=0)
            for q in query_history
        ]
    refined = refine_query_text(
        query,
        expand_synonyms=expand_synonyms,
        do_spell_check=do_spell_check,
        max_synonyms_per_term=QUERY_REFINEMENT["max_synonyms_per_term"],
        history_queries=history,
    )
    return {
        "original": query,
        "refined": refined if refined else query,
        "tokens": (refined if refined else query).split(),
    }


def suggest(prefix: str, dataset: str, limit: int = 8) -> List[str]:
    engine = get_engine(dataset)
    return engine.suggest_queries(prefix, limit=limit)
