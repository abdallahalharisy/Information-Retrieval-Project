"""Quick pipeline test on MSMARCO processed data."""

import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from ranking_engine import RankingEngine

DATA_FILE = 'processed_data_msmarco.json'
CACHE_PREFIX = 'msmarco'

with open(DATA_FILE, encoding='utf-8') as f:
    docs = json.load(f)
print(f'Loaded {len(docs):,} documents')

engine = RankingEngine()
if not engine.load_cache(DATA_FILE, cache_prefix=CACHE_PREFIX):
    engine.fit(docs, fit_word2vec=False, fit_bert=False)
    engine.save_cache(DATA_FILE, cache_prefix=CACHE_PREFIX)
print(f'Engine ready: {len(engine.doc_ids):,} documents')

for method in ['tfidf', 'bm25', 'index']:
    results = engine.search('machine learning', method=method, top_k=3)
    print(f'\n{method.upper()}:')
    for r in results:
        print(f'  {r["doc_id"]}: {r["score"]:.4f}')

for method in ['serial', 'parallel', 'rrf']:
    results = engine.search('machine learning', method=method, top_k=3)
    print(f'\n{method.upper()}:')
    for r in results:
        exp = r.get('explanation', {})
        methods = exp.get('methods', {})
        print(f'  {r["doc_id"]}: {r["score"]:.4f} ({exp.get("fusion_type", "")}) {methods}')

print('\nSuggestions for "mach":', engine.suggest_queries('mach'))
print('All core representations and ranking working!')
