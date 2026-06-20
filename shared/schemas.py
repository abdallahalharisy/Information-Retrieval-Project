"""Pydantic schemas for inter-service communication."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

SearchMethod = Literal["tfidf", "bm25", "word2vec", "bert"]
RankingMethod = Literal["none", "rrf"]
ExecutionMode = Literal["serial", "parallel"]


def resolve_engine_method(
    method: Optional[str],
    search_method: str,
    ranking_method: Optional[str],
) -> str:
    """Map the public IR controls to the legacy engine method name."""
    if method:
        return method
    if ranking_method == "rrf":
        return "rrf"
    return search_method


class HealthResponse(BaseModel):
    service: str
    status: str = "ok"


class PreprocessDocumentRequest(BaseModel):
    text: str


class PreprocessDocumentResponse(BaseModel):
    tokens: List[str]


class PreprocessBatchRequest(BaseModel):
    documents: Dict[str, str]


class PreprocessBatchResponse(BaseModel):
    processed: Dict[str, List[str]]
    count: int


class IndexBuildRequest(BaseModel):
    dataset: str = "msmarco"
    fit_word2vec: bool = False
    fit_bert: bool = False


class IndexStatusResponse(BaseModel):
    dataset: str
    is_ready: bool
    num_documents: int
    cache_prefix: str
    data_file: str


class SearchRequest(BaseModel):
    query: str
    dataset: str = "msmarco"
    search_method: SearchMethod = "bm25"
    ranking_method: Optional[RankingMethod] = "none"
    execution_mode: ExecutionMode = "parallel"
    method: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)
    query_history: List[str] = Field(default_factory=list)
    bm25_k1: Optional[float] = None
    bm25_b: Optional[float] = None

    def resolved_method(self) -> str:
        return resolve_engine_method(self.method, self.search_method, self.ranking_method)


class SearchResultItem(BaseModel):
    doc_id: str
    score: float
    method: str
    document_text: str
    explanation: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    refined_query: str
    method: str
    search_method: Optional[str] = None
    ranking_method: Optional[str] = None
    execution_mode: Optional[str] = None
    dataset: str
    results: List[SearchResultItem]
    suggestions: List[str] = Field(default_factory=list)


class RagRequest(BaseModel):
    query: str
    dataset: str = "msmarco"
    search_method: SearchMethod = "bm25"
    ranking_method: Optional[RankingMethod] = "none"
    execution_mode: ExecutionMode = "parallel"
    method: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)
    context_k: int = Field(default=4, ge=1, le=10)
    query_history: List[str] = Field(default_factory=list)
    bm25_k1: Optional[float] = None
    bm25_b: Optional[float] = None
    model: Optional[str] = None
    rag_mode: Literal["bm25", "hybrid"] = "hybrid"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    def resolved_method(self) -> str:
        return resolve_engine_method(self.method, self.search_method, self.ranking_method)


class RagSourceItem(BaseModel):
    rank: int
    label: str
    doc_id: str
    score: float
    method: str
    snippet: str
    document_text: str


class RagResponse(BaseModel):
    query: str
    refined_query: str
    method: str
    search_method: Optional[str] = None
    ranking_method: Optional[str] = None
    execution_mode: Optional[str] = None
    dataset: str
    model: str
    answer: str
    sources: List[RagSourceItem]
    generated_answer: Optional[str] = None
    source_documents: List[RagSourceItem] = Field(default_factory=list)
    warning: Optional[str] = None


class RefineQueryRequest(BaseModel):
    query: str
    expand_synonyms: bool = True
    do_spell_check: bool = True
    query_history: List[str] = Field(default_factory=list)


class RefineQueryResponse(BaseModel):
    original: str
    refined: str
    tokens: List[str]


class SuggestRequest(BaseModel):
    prefix: str
    dataset: str = "msmarco"
    limit: int = 8


class SuggestResponse(BaseModel):
    prefix: str
    suggestions: List[str]


class EvaluateRequest(BaseModel):
    dataset: str = "msmarco"
    method: str = "bm25"
    k: int = 10
    limit: Optional[int] = None
    mode: str = "enhanced"


class EvaluateComparisonRequest(BaseModel):
    dataset: str = "msmarco"
    methods: List[str] = Field(default_factory=lambda: ["tfidf", "bm25", "index", "serial", "parallel", "rrf"])
    k: int = 10
    limit: Optional[int] = None
    include_embeddings: bool = False


class EvaluateSummary(BaseModel):
    dataset: str
    method: str
    k: int
    num_queries: int
    qrels_query_count: int = 0
    attempted_query_count: int = 0
    avg_precision: float
    avg_recall: float
    map_score: float
    avg_ndcg: float
    report_path: Optional[str] = None
    note: Optional[str] = None


class EvaluateComparisonSummary(BaseModel):
    dataset: str
    k: int
    limit: Optional[int] = None
    qrels_query_count: int = 0
    attempted_query_count: int = 0
    methods: List[str]
    json_report_path: str
    markdown_report_path: str
    comparisons: List[Dict[str, Any]]
