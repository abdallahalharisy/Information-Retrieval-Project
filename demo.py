# demo.py
"""Demo script: load MSMARCO, fit engine, run sample queries."""

import json
import logging
from ranking_engine import RankingEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = 'processed_data_msmarco.json'
CACHE_PREFIX = 'msmarco'


def main():
    logger.info("=== IR Search Engine Demo ===")

    with open(DATA_FILE, encoding='utf-8') as f:
        processed_documents = json.load(f)
    logger.info(f"Loaded {len(processed_documents):,} documents")

    engine = RankingEngine()
    if not engine.load_cache(DATA_FILE, cache_prefix=CACHE_PREFIX):
        engine.fit(processed_documents, fit_word2vec=False, fit_bert=False)
        engine.save_cache(DATA_FILE, cache_prefix=CACHE_PREFIX)

    report = engine.get_parameter_report()
    logger.info(f"Engine ready: {report['num_documents']:,} docs")

    sample_queries = [
        "information retrieval search engine",
        "machine learning algorithms",
        "natural language processing",
    ]

    for query in sample_queries:
        logger.info(f"\n{'=' * 60}\nQuery: {query}\n{'=' * 60}")
        for method in ['tfidf', 'bm25', 'serial', 'parallel', 'rrf']:
            results = engine.search(query, method=method, top_k=3)
            logger.info(f"  {method.upper()}:")
            for i, r in enumerate(results, 1):
                logger.info(f"    {i}. {r['doc_id']}: {r['score']:.4f}")

    logger.info("\n=== Demo Complete ===")


if __name__ == '__main__':
    main()
