"""Pydantic schemas for inter-service communication."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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
    method: str = "parallel"
    top_k: int = Field(default=10, ge=1, le=100)
    query_history: List[str] = Field(default_factory=list)
    bm25_k1: Optional[float] = None
    bm25_b: Optional[float] = None


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
    dataset: str
    results: List[SearchResultItem]
    suggestions: List[str] = Field(default_factory=list)


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
    limit: int = 50
    mode: str = "enhanced"


class EvaluateComparisonRequest(BaseModel):
    dataset: str = "msmarco"
    methods: List[str] = Field(default_factory=lambda: ["tfidf", "bm25", "index", "serial", "parallel", "rrf"])
    k: int = 10
    limit: int = 50
    include_embeddings: bool = False


class EvaluateSummary(BaseModel):
    dataset: str
    method: str
    k: int
    num_queries: int
    avg_precision: float
    avg_recall: float
    map_score: float
    avg_ndcg: float
    report_path: Optional[str] = None
    note: Optional[str] = None


class EvaluateComparisonSummary(BaseModel):
    dataset: str
    k: int
    limit: int
    methods: List[str]
    json_report_path: str
    markdown_report_path: str
    comparisons: List[Dict[str, Any]]
