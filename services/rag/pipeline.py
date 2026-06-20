"""Production RAG pipeline: hybrid retrieval, fusion, context assembly, Gemini generation."""

import importlib.util
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from hybrid_fusion import RRFFusion, RankerResult
from preprocess import refine_query_text
from shared.datasets import DATASETS, resolve_dataset_key
from shared.document_store import get_documents
from shared.engine_registry import get_engine
from shared.schemas import RagSourceItem

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TOP_N = int(os.environ.get("RAG_TOP_N", "50"))
DEFAULT_CONTEXT_K = int(os.environ.get("RAG_CONTEXT_K", "5"))
MAX_CONTEXT_CHARS_PER_DOC = int(os.environ.get("RAG_CONTEXT_CHARS_PER_DOC", "1800"))
MAX_OUTPUT_TOKENS = int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "512"))


@dataclass
class RagPipelineResult:
    refined_query: str
    generated_answer: str
    source_documents: List[RagSourceItem]
    context_documents: List[RagSourceItem]
    model: str
    warning: Optional[str] = None


def run_hybrid_rag(
    query: str,
    dataset: str = "msmarco",
    top_n: int = DEFAULT_TOP_N,
    context_k: int = DEFAULT_CONTEXT_K,
    model: Optional[str] = None,
    bm25_k1: Optional[float] = None,
    bm25_b: Optional[float] = None,
    use_semantic: bool = True,
    temperature: float = 0.2,
) -> RagPipelineResult:
    """Run lexical + semantic retrieval, fuse results, and generate a cited answer."""
    dataset_key = resolve_dataset_key(dataset)
    cfg = DATASETS[dataset_key]
    engine = get_engine(dataset_key, build_if_missing=False)
    if bm25_k1 is not None or bm25_b is not None:
        engine.bm25.update_parameters(k1=bm25_k1, b=bm25_b)

    top_n = max(1, min(top_n, 100))
    context_k = max(1, min(context_k, 10, top_n))

    refined_query = refine_query_text(
        query,
        expand_synonyms=False,
        do_spell_check=False,
        max_synonyms_per_term=0,
    ) or query

    lexical_results = _run_lexical_track(engine, refined_query, top_n)
    semantic_results = _run_semantic_track(engine, query, top_n) if use_semantic else []
    fused = _fuse_results(lexical_results, semantic_results, top_n)
    sources = _build_source_documents(fused[:top_n], cfg.get("document_db"))
    context_sources = sources[:context_k]

    selected_model = model or DEFAULT_GEMINI_MODEL
    if not context_sources:
        return RagPipelineResult(
            refined_query=refined_query,
            generated_answer="I cannot find the answer in the provided documents.",
            source_documents=[],
            context_documents=[],
            model=selected_model,
            warning="No documents were retrieved for the RAG context.",
        )

    try:
        answer = _generate_with_gemini(query, context_sources, selected_model, temperature)
        warning = None
    except RuntimeError as exc:
        answer = _fallback_answer(context_sources)
        warning = f"Gemini generation failed; returned retrieval-only answer. {exc}"
        selected_model = "retrieval-fallback"

    return RagPipelineResult(
        refined_query=refined_query,
        generated_answer=answer,
        source_documents=sources,
        context_documents=context_sources,
        model=selected_model,
        warning=warning,
    )


def _run_lexical_track(engine, refined_query: str, top_n: int) -> List[Tuple[str, float]]:
    bm25 = engine._rank_bm25_with_tfidf_candidates(refined_query, top_n)
    tfidf = engine.tfidf.get_top_k(refined_query, k=top_n)
    return _merge_lexical_scores(bm25, tfidf, top_n)


def _merge_lexical_scores(
    bm25: List[Tuple[str, float]],
    tfidf: List[Tuple[str, float]],
    top_n: int,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for weight, ranked in ((0.65, bm25), (0.35, tfidf)):
        if not ranked:
            continue
        max_score = max(abs(score) for _, score in ranked) or 1.0
        for doc_id, score in ranked:
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (float(score) / max_score)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]


def _run_semantic_track(engine, raw_query: str, top_n: int) -> List[Tuple[str, float]]:
    return engine.rank_bert(raw_query, top_n)


def _fuse_results(
    lexical_results: List[Tuple[str, float]],
    semantic_results: List[Tuple[str, float]],
    top_n: int,
) -> List[Tuple[str, float, dict]]:
    fusion_inputs = []
    if lexical_results:
        fusion_inputs.append(
            RankerResult(
                "BM25+TF-IDF",
                [doc_id for doc_id, _ in lexical_results],
                [score for _, score in lexical_results],
            )
        )
    if semantic_results:
        fusion_inputs.append(
            RankerResult(
                "BERT",
                [doc_id for doc_id, _ in semantic_results],
                [score for _, score in semantic_results],
            )
        )
    if not fusion_inputs:
        return []
    return RRFFusion().fuse(fusion_inputs, top_k=top_n)


def _build_source_documents(
    fused: List[Tuple[str, float, dict]],
    document_db: Optional[str],
) -> List[RagSourceItem]:
    doc_ids = [doc_id for doc_id, _, _ in fused]
    raw_docs = get_documents(doc_ids, document_db) if document_db else {}
    sources = []
    for idx, (doc_id, score, explanation) in enumerate(fused, 1):
        raw_text = _clean_text(raw_docs.get(doc_id, ""))
        if not raw_text:
            continue
        sources.append(
            RagSourceItem(
                rank=idx,
                label=f"D{idx}",
                doc_id=doc_id,
                score=float(score),
                method=explanation.get("fusion_type", "rrf"),
                snippet=_trim_text(raw_text, 320),
                document_text=raw_text,
            )
        )
    return sources


def _generate_with_gemini(
    query: str,
    sources: List[RagSourceItem],
    model_name: str,
    temperature: float,
) -> str:
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    if importlib.util.find_spec("google.generativeai") is None:
        raise RuntimeError("google-generativeai is not installed.")

    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        _build_prompt(query, sources),
        generation_config=genai.types.GenerationConfig(
            temperature=max(0.0, min(float(temperature), 2.0)),
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
        request_options={"timeout": 45},
    )
    answer = getattr(response, "text", "") or ""
    if not answer.strip():
        raise RuntimeError("Gemini returned an empty answer.")
    return answer.strip()


def _build_prompt(query: str, sources: List[RagSourceItem]) -> str:
    context_blocks = []
    for source in sources:
        context = _trim_text(source.document_text, MAX_CONTEXT_CHARS_PER_DOC)
        context_blocks.append(f"Document ID: {source.doc_id}\n{context}")
    return f"""You are a precise assistant. Answer the User Question using ONLY the provided Document Context.
If the context doesn't contain the answer, say "I cannot find the answer in the provided documents."
For every claim you make, explicitly append the Document ID as a citation (e.g., [Document ID]).

---
DOCUMENT CONTEXT:
{chr(10).join(context_blocks)}
---
USER QUESTION:
{query}
"""


def _fallback_answer(sources: List[RagSourceItem]) -> str:
    lines = ["I cannot generate a synthesized answer right now, but these are the top retrieved sources:"]
    for source in sources:
        lines.append(f"- [{source.doc_id}] {_trim_text(source.snippet, 300)}")
    return "\n".join(lines)


def _clean_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split())


def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit(" ", 1)[0]
    return f"{trimmed}..."
