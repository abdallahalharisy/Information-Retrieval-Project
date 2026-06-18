"""Prepare local runtime artifacts once for fast daily startup.

Run once after downloading/preprocessing MSMARCO:
    python prepare_runtime.py

It creates/validates:
  - SQLite original document store: data/documents_msmarco.db
  - Ranking cache: cache/engine_cache_msmarco.joblib

After this, normal startup is:
  - python run_gateway.py
  - streamlit run frontend/search_interface.py
"""

import json
import os

from build_document_store import build_msmarco_document_store
from ranking_engine import RankingEngine
from shared.datasets import DATASETS
from shared.document_store import count_documents
from shared.engine_registry import read_cache_meta


def main():
    cfg = DATASETS["msmarco"]
    data_file = cfg["file"]
    db_path = cfg["document_db"]

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"{data_file} not found. Run first: python main.py msmarco 210000"
        )

    cache_meta = read_cache_meta("msmarco")
    expected_docs = cache_meta.get("num_documents", 0)
    processed_docs = None
    if not expected_docs:
        with open(data_file, encoding="utf-8") as f:
            processed_docs = json.load(f)
        expected_docs = len(processed_docs)
    print(f"Expected processed documents: {expected_docs:,}")

    original_count = count_documents(db_path)
    if original_count < expected_docs:
        print(
            f"Original document DB has {original_count:,}/{expected_docs:,}. "
            "Building document store..."
        )
        build_msmarco_document_store(limit=expected_docs)
    else:
        print(f"Original document DB ready: {original_count:,} docs")

    cache_ready = (
        cache_meta.get("cache_exists")
        and cache_meta.get("meta_exists")
        and cache_meta.get("num_documents") == expected_docs
    )

    if cache_ready:
        print(f"Ranking cache ready: {cache_meta['cache_file']}")
    else:
        print("Ranking cache missing or stale. Building once...")
        if processed_docs is None:
            with open(data_file, encoding="utf-8") as f:
                processed_docs = json.load(f)
        engine = RankingEngine()
        engine.fit(processed_docs, fit_word2vec=False, fit_bert=False)
        print("Saving ranking cache to disk...")
        engine.save_cache(data_file, cache_prefix="msmarco")
        refreshed_meta = read_cache_meta("msmarco")
        if refreshed_meta.get("cache_exists") and refreshed_meta.get("meta_exists"):
            print("Ranking cache built.")
        else:
            raise RuntimeError("Ranking cache was not written. Check disk space and permissions.")

    print("\nRuntime is ready. Daily startup:")
    print("  1) python run_gateway.py")
    print("  2) streamlit run frontend/search_interface.py")


if __name__ == "__main__":
    main()
