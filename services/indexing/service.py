"""Indexing service business logic."""

import logging

from shared.datasets import DATASETS, resolve_dataset_key
from shared.engine_registry import clear_engine, get_engine, list_engine_status

logger = logging.getLogger(__name__)


def build_index(dataset_key: str, fit_word2vec: bool = False, fit_bert: bool = False) -> dict:
    key = resolve_dataset_key(dataset_key)
    clear_engine(key)
    engine = get_engine(key, build_if_missing=True)
    if fit_word2vec:
        logger.info("Word2Vec uses query-time candidate re-ranking; skipping full-corpus training.")
    if fit_bert:
        logger.info("BERT uses query-time candidate re-ranking; loading encoder only.")
        engine.bert.load_model()
    cfg = DATASETS[key]
    return {
        "dataset": key,
        "is_ready": engine.is_fitted,
        "num_documents": len(engine.doc_ids),
        "cache_prefix": cfg["cache_prefix"],
        "data_file": cfg["file"],
    }


def get_index_status(dataset_key: str) -> dict:
    key = resolve_dataset_key(dataset_key)
    status = list_engine_status()[key]
    cfg = DATASETS[key]
    return {
        "dataset": key,
        "is_ready": status["cache_exists"] and status["meta_exists"],
        "num_documents": status["num_documents"],
        "cache_prefix": cfg["cache_prefix"],
        "data_file": cfg["file"],
    }


def get_all_status() -> dict:
    return list_engine_status()
