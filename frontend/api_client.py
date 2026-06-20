"""HTTP client for the API Gateway."""

import os
from typing import List, Optional

import httpx

GATEWAY_URL = os.environ.get("IR_GATEWAY_URL", "http://127.0.0.1:8000")


class GatewayClient:
    def __init__(self, base_url: str = GATEWAY_URL, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()

    def list_datasets(self) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(f"{self.base_url}/api/v1/datasets")
            r.raise_for_status()
            return r.json()

    def search(
        self,
        query: str,
        dataset: str = "msmarco",
        search_method: str = "bm25",
        ranking_method: str = "none",
        execution_mode: str = "parallel",
        top_k: int = 10,
        query_history: Optional[List[str]] = None,
        bm25_k1: Optional[float] = None,
        bm25_b: Optional[float] = None,
    ) -> dict:
        payload = {
            "query": query,
            "dataset": dataset,
            "search_method": search_method,
            "ranking_method": ranking_method,
            "execution_mode": execution_mode,
            "top_k": top_k,
            "query_history": query_history or [],
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/api/v1/search", json=payload)
            r.raise_for_status()
            return r.json()

    def rag_answer(
        self,
        query: str,
        dataset: str = "msmarco",
        search_method: str = "bm25",
        ranking_method: str = "none",
        execution_mode: str = "parallel",
        top_k: int = 10,
        context_k: int = 4,
        query_history: Optional[List[str]] = None,
        bm25_k1: Optional[float] = None,
        bm25_b: Optional[float] = None,
        model: Optional[str] = None,
        rag_mode: str = "hybrid",
        temperature: float = 0.2,
    ) -> dict:
        payload = {
            "query": query,
            "dataset": dataset,
            "search_method": search_method,
            "ranking_method": ranking_method,
            "execution_mode": execution_mode,
            "top_k": top_k,
            "context_k": context_k,
            "query_history": query_history or [],
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "model": model,
            "rag_mode": rag_mode,
            "temperature": temperature,
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/api/v1/rag/answer", json=payload)
            if r.is_error:
                try:
                    detail = r.json().get("detail", r.text)
                except ValueError:
                    detail = r.text
                raise RuntimeError(detail)
            return r.json()

    def run_evaluation(
        self,
        dataset: str = "msmarco",
        method: str = "bm25",
        k: int = 10,
        limit: Optional[int] = None,
        mode: str = "enhanced",
    ) -> dict:
        payload = {
            "dataset": dataset,
            "method": method,
            "k": k,
            "limit": limit,
            "mode": mode,
        }
        with httpx.Client(timeout=max(self.timeout, 900.0)) as client:
            r = client.post(f"{self.base_url}/api/v1/evaluation/run", json=payload)
            if r.is_error:
                try:
                    detail = r.json().get("detail", r.text)
                except ValueError:
                    detail = r.text
                raise RuntimeError(detail)
            return r.json()

    def compare_evaluation(
        self,
        dataset: str = "msmarco",
        methods: Optional[List[str]] = None,
        k: int = 10,
        limit: Optional[int] = None,
        include_embeddings: bool = False,
    ) -> dict:
        payload = {
            "dataset": dataset,
            "methods": methods or ["tfidf", "bm25", "index", "serial", "parallel", "rrf"],
            "k": k,
            "limit": limit,
            "include_embeddings": include_embeddings,
        }
        with httpx.Client(timeout=max(self.timeout, 900.0)) as client:
            r = client.post(f"{self.base_url}/api/v1/evaluation/compare", json=payload)
            if r.is_error:
                try:
                    detail = r.json().get("detail", r.text)
                except ValueError:
                    detail = r.text
                raise RuntimeError(detail)
            return r.json()

    def suggest(self, prefix: str, dataset: str = "msmarco", limit: int = 8) -> List[str]:
        payload = {"prefix": prefix, "dataset": dataset, "limit": limit}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/api/v1/query-refinement/suggest", json=payload)
            r.raise_for_status()
            return r.json().get("suggestions", [])

    def index_status(self, dataset: str) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(f"{self.base_url}/api/v1/indexing/status/{dataset}")
            r.raise_for_status()
            return r.json()
