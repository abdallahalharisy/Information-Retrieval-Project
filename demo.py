# demo.py
"""
Demo script: Load processed data, fit ranking engine, and run sample queries
"""

import json
import logging
from ranking_engine import RankingEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=== IR Search Engine Demo ===\n")
    
    # 1. Load preprocessed documents
    logger.info("Step 1: Loading preprocessed documents...")
    try:
        with open('processed_data.json', 'r', encoding='utf-8') as f:
            processed_documents = json.load(f)
        logger.info(f"Loaded {len(processed_documents)} preprocessed documents\n")
    except FileNotFoundError:
        logger.error("processed_data.json not found. Run main.py first.")
        return
    
    # 2. Initialize and fit ranking engine
    logger.info("Step 2: Initializing and fitting ranking engine...")
    engine = RankingEngine()
    engine.fit(processed_documents)
    logger.info()
    
    # 3. Print parameter report
    logger.info("Step 3: Current parameters:")
    report = engine.get_parameter_report()
    for key, value in report.items():
        logger.info(f"  {key}: {value}")
    logger.info()
    
    # 4. Run sample queries
    sample_queries = [
        "information retrieval search engine",
        "machine learning algorithms",
        "natural language processing"
    ]
    
    logger.info("Step 4: Running sample queries...\n")
    
    for query in sample_queries:
        logger.info(f"{'='*60}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*60}")
        
        # Try different ranking methods
        methods = ['tfidf', 'bm25', 'parallel', 'serial', 'rrf']
        
        for method in methods:
            try:
                logger.info(f"\n  Method: {method.upper()}")
                results = engine.search(query, method=method, top_k=5)
                
                for i, result in enumerate(results, 1):
                    doc_id = result['doc_id']
                    score = result['score']
                    logger.info(f"    {i}. {doc_id}: {score:.4f}")
                    
                    if 'explanation' in result:
                        exp = result['explanation']
                        logger.info(f"       Fusion: {exp.get('fusion_type', 'N/A')}")
                
            except Exception as e:
                logger.warning(f"  {method.upper()} failed: {e}")
        
        logger.info()
    
    logger.info("=== Demo Complete ===")


if __name__ == '__main__':
    main()
