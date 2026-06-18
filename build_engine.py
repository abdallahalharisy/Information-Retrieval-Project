# build_engine.py
"""
Build and cache the ranking engine for a preprocessed dataset (no internet required).
Usage:
    python build_engine.py msmarco
    python build_engine.py fever
"""

import json
import logging
import sys

from ranking_engine import RankingEngine

DATASETS = {
    'msmarco': ('processed_data_msmarco.json', 'msmarco'),
    'fever': ('processed_data_fever.json', 'fever'),
    '1': ('processed_data_fever.json', 'fever'),
    '2': ('processed_data_msmarco.json', 'msmarco'),
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_engine.py <msmarco|fever>")
        sys.exit(1)

    choice = sys.argv[1].lower()
    if choice not in DATASETS:
        print(f"Unknown dataset: {choice}")
        sys.exit(1)

    data_file, cache_prefix = DATASETS[choice]
    logger.info(f"Loading {data_file}...")
    with open(data_file, encoding='utf-8') as f:
        documents = json.load(f)
    logger.info(f"Loaded {len(documents):,} documents")

    engine = RankingEngine()
    if engine.load_cache(data_file, cache_prefix=cache_prefix):
        logger.info(f"Cache already up to date for '{cache_prefix}' ({len(engine.doc_ids):,} docs)")
        return

    logger.info("Building TF-IDF, BM25, and inverted index (this may take several minutes)...")
    engine.fit(documents, fit_word2vec=False, fit_bert=False)
    engine.save_cache(data_file, cache_prefix=cache_prefix)
    logger.info(f"Done. Cache saved to cache/engine_cache_{cache_prefix}.joblib")


if __name__ == '__main__':
    main()
