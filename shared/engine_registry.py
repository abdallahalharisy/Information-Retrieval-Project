"""Central registry for RankingEngine instances (one per dataset)."""

import json
import logging
import os
import threading
from typing import Dict, Iterator, List, Optional, Tuple

from ranking_engine import RankingEngine
from shared.datasets import DATASETS, resolve_dataset_key
from shared.document_store import count_documents

logger = logging.getLogger(__name__)

_engines: Dict[str, RankingEngine] = {}
_engine_lock = threading.Lock()


def _stream_processed_items(path: str, chunk_size: int = 1024 * 1024) -> Iterator[Tuple[str, str]]:
    """Stream the huge processed JSON object without loading it all at once."""
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0

    with open(path, encoding="utf-8") as f:
        eof = False

        def fill() -> None:
            nonlocal buffer, eof
            chunk = f.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        fill()
        while True:
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer):
                    break
                if eof:
                    return
                buffer = ""
                pos = 0
                fill()

            if buffer[pos] == "{":
                pos += 1
                break
            raise ValueError("Expected JSON object")

        while True:
            while True:
                while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","):
                    pos += 1
                if pos < len(buffer):
                    break
                if eof:
                    return
                buffer = ""
                pos = 0
                fill()

            if buffer[pos] == "}":
                return

            while True:
                try:
                    doc_id, pos = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    buffer = buffer[pos:]
                    pos = 0
                    fill()

            while True:
                while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ":"):
                    pos += 1
                if pos < len(buffer):
                    break
                if eof:
                    raise ValueError("Unexpected end of JSON after key")
                buffer = buffer[pos:]
                pos = 0
                fill()

            while True:
                try:
                    tokens, pos = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    buffer = buffer[pos:]
                    pos = 0
                    fill()

            text = " ".join(tokens) if isinstance(tokens, list) else str(tokens)
            yield doc_id, text

            if pos > chunk_size:
                buffer = buffer[pos:]
                pos = 0


def _load_processed_texts(path: str) -> Tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    processed_docs: List[str] = []
    for doc_id, text in _stream_processed_items(path):
        doc_ids.append(doc_id)
        processed_docs.append(text)
    return doc_ids, processed_docs


def _cache_paths(cache_prefix: str) -> tuple:
    return (
        os.path.join("cache", f"engine_cache_{cache_prefix}.joblib"),
        os.path.join("cache", f"engine_cache_{cache_prefix}_meta.json"),
    )


def read_cache_meta(dataset_key: str) -> dict:
    key = resolve_dataset_key(dataset_key)
    cfg = DATASETS[key]
    cache_file, meta_file = _cache_paths(cfg["cache_prefix"])
    meta = {}
    if os.path.exists(meta_file):
        try:
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    return {
        "cache_file": cache_file,
        "meta_file": meta_file,
        "cache_exists": os.path.exists(cache_file),
        "meta_exists": os.path.exists(meta_file),
        **meta,
    }


def get_loaded_engine(dataset_key: str) -> Optional[RankingEngine]:
    key = resolve_dataset_key(dataset_key)
    return _engines.get(key)


def get_engine(dataset_key: str, build_if_missing: bool = False) -> RankingEngine:
    """Return a fitted engine. By default, never rebuild at runtime."""
    key = resolve_dataset_key(dataset_key)
    with _engine_lock:
        if key in _engines:
            return _engines[key]

        cfg = DATASETS[key]
        data_file = cfg["file"]
        cache_prefix = cfg["cache_prefix"]

        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"{data_file} not found. Preprocess the dataset first: python main.py {key}"
            )

        engine = RankingEngine()
        if not engine.load_cache(data_file, cache_prefix=cache_prefix):
            if not build_if_missing:
                raise RuntimeError(
                    f"Index cache is not ready for dataset '{key}'. "
                    "Run once: python prepare_runtime.py"
                )
            logger.info("Building index for %s (%s)...", key, data_file)
            doc_ids, processed_docs = _load_processed_texts(data_file)
            engine.fit_processed_texts(processed_docs, doc_ids, fit_word2vec=False, fit_bert=False)
            engine.save_cache(data_file, cache_prefix=cache_prefix)

        _engines[key] = engine
        return engine


def warmup_engines() -> None:
    """Load available engine caches so the first interactive search is fast."""
    for key in DATASETS:
        try:
            get_engine(key, build_if_missing=False)
            logger.info("Warmup complete for dataset '%s'.", key)
        except Exception as exc:
            logger.warning("Warmup skipped for dataset '%s': %s", key, exc)


def clear_engine(dataset_key: str) -> None:
    key = resolve_dataset_key(dataset_key)
    _engines.pop(key, None)


def list_engine_status() -> Dict[str, dict]:
    status = {}
    for key, cfg in DATASETS.items():
        data_file = cfg["file"]
        cache_meta = read_cache_meta(key)
        db_path = cfg.get("document_db")
        status[key] = {
            "name": cfg["name"],
            "data_file": data_file,
            "data_exists": os.path.exists(data_file),
            "cache_exists": cache_meta["cache_exists"],
            "meta_exists": cache_meta["meta_exists"],
            "loaded": key in _engines,
            "num_documents": (
                len(_engines[key].doc_ids)
                if key in _engines
                else cache_meta.get("num_documents", 0)
            ),
            "document_db": db_path,
            "document_db_exists": os.path.exists(db_path) if db_path else False,
            "original_documents": count_documents(db_path) if db_path else 0,
        }
    return status
