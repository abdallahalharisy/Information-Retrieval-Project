# build_engine.py
"""
Build and cache the ranking engine for a preprocessed dataset (no internet required).
Usage:
    python build_engine.py msmarco
"""

import logging
import sys

from ranking_engine import RankingEngine
from shared.engine_registry import _load_processed_texts

DATASETS = {
    'msmarco': ('processed_data_msmarco.json', 'msmarco'),
    '1': ('processed_data_msmarco.json', 'msmarco'),
    '2': ('processed_data_msmarco.json', 'msmarco'),
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_engine.py <msmarco>")
        sys.exit(1)

    choice = sys.argv[1].lower()
    if choice not in DATASETS:
        print(f"Unknown dataset: {choice}")
        sys.exit(1)

    data_file, cache_prefix = DATASETS[choice]
    engine = RankingEngine()
    if engine.load_cache(data_file, cache_prefix=cache_prefix):
        logger.info(f"Cache already up to date for '{cache_prefix}' ({len(engine.doc_ids):,} docs)")
        return

    logger.info(f"Streaming {data_file}...")
    doc_ids, processed_docs = _load_processed_texts(data_file)
    logger.info(f"Loaded {len(doc_ids):,} documents")

    logger.info("Building TF-IDF and BM25 cache (this may take a while for full MSMARCO)...")
    engine.fit_processed_texts(processed_docs, doc_ids, fit_word2vec=False, fit_bert=False)
    engine.save_cache(data_file, cache_prefix=cache_prefix)
    logger.info(f"Done. Cache saved to cache/engine_cache_{cache_prefix}.joblib")


if __name__ == '__main__':
    main()
