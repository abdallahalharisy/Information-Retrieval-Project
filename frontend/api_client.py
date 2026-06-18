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
        method: str = "parallel",
        top_k: int = 10,
        query_history: Optional[List[str]] = None,
        bm25_k1: Optional[float] = None,
        bm25_b: Optional[float] = None,
    ) -> dict:
        payload = {
            "query": query,
            "dataset": dataset,
            "method": method,
            "top_k": top_k,
            "query_history": query_history or [],
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/api/v1/search", json=payload)
            r.raise_for_status()
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
