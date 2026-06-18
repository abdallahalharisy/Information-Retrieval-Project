"""Central registry for RankingEngine instances (one per dataset)."""

import json
import logging
import os
from typing import Dict

from ranking_engine import RankingEngine
from shared.datasets import DATASETS, resolve_dataset_key

logger = logging.getLogger(__name__)

_engines: Dict[str, RankingEngine] = {}


def get_engine(dataset_key: str, build_if_missing: bool = True) -> RankingEngine:
    """Return a fitted engine for the dataset, building cache if needed."""
    key = resolve_dataset_key(dataset_key)
    if key in _engines:
        return _engines[key]

    cfg = DATASETS[key]
    data_file = cfg["file"]
    cache_prefix = cfg["cache_prefix"]

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"{data_file} not found. Preprocess the dataset first: python main.py {key} 210000"
        )

    engine = RankingEngine()
    if not engine.load_cache(data_file, cache_prefix=cache_prefix):
        if not build_if_missing:
            raise RuntimeError(f"Index not built for dataset '{key}'. Call POST /indexing/build")
        logger.info("Building index for %s (%s)...", key, data_file)
        with open(data_file, encoding="utf-8") as f:
            documents = json.load(f)
        engine.fit(documents, fit_word2vec=False, fit_bert=False)
        engine.save_cache(data_file, cache_prefix=cache_prefix)

    _engines[key] = engine
    return engine


def clear_engine(dataset_key: str) -> None:
    key = resolve_dataset_key(dataset_key)
    _engines.pop(key, None)


def list_engine_status() -> Dict[str, dict]:
    status = {}
    for key, cfg in DATASETS.items():
        data_file = cfg["file"]
        cache_file = os.path.join("cache", f"engine_cache_{cfg['cache_prefix']}.joblib")
        status[key] = {
            "name": cfg["name"],
            "data_file": data_file,
            "data_exists": os.path.exists(data_file),
            "cache_exists": os.path.exists(cache_file),
            "loaded": key in _engines,
            "num_documents": len(_engines[key].doc_ids) if key in _engines else 0,
        }
    return status
