# ranking_engine.py
"""
Main ranking engine orchestrating all representation methods and fusion strategies
"""

import json
import logging
import os
import joblib
from typing import List, Dict, Tuple, Optional
from representations import (
    TFIDFRepresentation,
    BM25Representation,
    InvertedIndexRepresentation,
    Word2VecEmbedding,
    BERTEmbedding
)
from hybrid_fusion import (
    RankerResult,
    SerialFusion,
    ParallelFusion,
    RRFFusion
)
from config import FUSION_WEIGHTS, TOP_K, QUERY_REFINEMENT, SERIAL_FUSION_ORDER, RRF_FUSION_ORDER
from preprocess import refine_query_text
from shared.datasets import DATASETS
from shared.document_store import get_document

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RankingEngine:
    """Main IR engine supporting multiple ranking methods and fusion strategies"""

    def __init__(self):
        self.tfidf = TFIDFRepresentation()
        self.bm25 = BM25Representation()
        self.index = InvertedIndexRepresentation()
        self.word2vec = Word2VecEmbedding()
        self.bert = BERTEmbedding()

        self.is_fitted = False
        self.documents = {}
        self.processed_docs = []
        self.doc_ids = []

        self.serial_fusion = SerialFusion()
        self.parallel_fusion = ParallelFusion(weights=FUSION_WEIGHTS)
        self.rrf_fusion = RRFFusion()

        self.cache_dir = 'cache'
        self.cache_prefix = 'default'
        self.cache_version = 2
        self._set_cache_paths('default')

    def _set_cache_paths(self, cache_prefix: str):
        self.cache_prefix = cache_prefix
        self.cache_file = os.path.join(self.cache_dir, f'engine_cache_{cache_prefix}.joblib')
        self.cache_meta_file = os.path.join(self.cache_dir, f'engine_cache_{cache_prefix}_meta.json')

    def load_cache(self, processed_data_file: str, cache_prefix: str = 'default') -> bool:
        """Load cached TF-IDF/BM25/Index state from disk if available."""
        self._set_cache_paths(cache_prefix)
        if not os.path.exists(self.cache_file) or not os.path.exists(self.cache_meta_file):
            return False

        try:
            with open(self.cache_meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            if meta.get('cache_version') != self.cache_version:
                return False
            if not os.path.exists(processed_data_file):
                return False

            current_mtime = os.path.getmtime(processed_data_file)
            if meta.get('processed_data_mtime') != current_mtime:
                return False

            cached = joblib.load(self.cache_file)
            self._restore_from_cache(cached)
            logger.info(f"Loaded cached engine for '{cache_prefix}' ({len(self.doc_ids)} docs).")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cached engine: {e}")
            return False

    def _restore_from_cache(self, cached: dict):
        self.doc_ids = cached['doc_ids']
        self.processed_docs = cached['processed_docs']
        self.documents = dict(zip(self.doc_ids, self.processed_docs))

        self.tfidf.vectorizer = cached['tfidf_vectorizer']
        self.tfidf.tfidf_matrix = cached['tfidf_matrix']
        self.tfidf.doc_ids = self.doc_ids
        self.tfidf.documents = self.processed_docs

        self.bm25.k1 = cached['bm25_k1']
        self.bm25.b = cached['bm25_b']
        self.bm25.bm25 = cached['bm25_model']
        self.bm25.documents = self.processed_docs
        self.bm25.doc_ids = self.doc_ids
        self.bm25.tokenized_docs = cached['bm25_tokenized_docs']

        self.index.postings = cached['index_postings']
        self.index.doc_freq = cached['index_doc_freq']
        self.index.doc_lengths = cached['index_doc_lengths']
        self.index.num_docs = len(self.doc_ids)
        self.index.documents = self.processed_docs
        self.index.doc_ids = self.doc_ids

        self.is_fitted = True

    def save_cache(self, processed_data_file: str, cache_prefix: str = 'default'):
        """Save TF-IDF/BM25/Index state to disk for future app startups."""
        self._set_cache_paths(cache_prefix)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cached = {
                'doc_ids': self.doc_ids,
                'processed_docs': self.processed_docs,
                'tfidf_vectorizer': self.tfidf.vectorizer,
                'tfidf_matrix': self.tfidf.tfidf_matrix,
                'bm25_k1': self.bm25.k1,
                'bm25_b': self.bm25.b,
                'bm25_model': self.bm25.bm25,
                'bm25_tokenized_docs': self.bm25.tokenized_docs,
                'index_postings': dict(self.index.postings),
                'index_doc_freq': self.index.doc_freq,
                'index_doc_lengths': self.index.doc_lengths,
            }
            joblib.dump(cached, self.cache_file)

            meta = {
                'cache_version': self.cache_version,
                'cache_prefix': cache_prefix,
                'processed_data_mtime': os.path.getmtime(processed_data_file) if os.path.exists(processed_data_file) else None,
                'num_documents': len(self.doc_ids)
            }
            with open(self.cache_meta_file, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved engine cache for '{cache_prefix}'.")
        except Exception as e:
            logger.warning(f"Failed to save engine cache: {e}")

    def fit(self, documents: Dict[str, List[str]], fit_word2vec: bool = True, fit_bert: bool = True):
        """Fit ranking methods on documents."""
        self.documents = {}
        processed_docs = []
        doc_ids = []

        for doc_id, tokens in documents.items():
            if isinstance(tokens, list):
                text = ' '.join(tokens)
            else:
                text = str(tokens)

            self.documents[doc_id] = text
            processed_docs.append(text)
            doc_ids.append(doc_id)

        self.doc_ids = doc_ids
        self.processed_docs = processed_docs

        logger.info(f"Fitting ranking engine on {len(doc_ids)} documents")

        for name, fn in [
            ('TF-IDF', lambda: self.tfidf.fit(processed_docs, doc_ids)),
            ('BM25', lambda: self.bm25.fit(processed_docs, doc_ids)),
            ('Inverted Index', lambda: self.index.fit(processed_docs, doc_ids)),
        ]:
            try:
                fn()
                logger.info(f"✓ {name} fitted")
            except Exception as e:
                logger.warning(f"✗ {name} fitting failed: {e}")

        if fit_word2vec:
            try:
                self.word2vec.fit(processed_docs, doc_ids)
                logger.info("✓ Word2Vec fitted")
            except Exception as e:
                logger.warning(f"✗ Word2Vec fitting failed: {e}")
        else:
            logger.info("Skipping Word2Vec until first use")

        if fit_bert:
            try:
                logger.info("Fitting BERT (this may take a while)...")
                self.bert.fit(processed_docs, doc_ids)
                logger.info("✓ BERT fitted")
            except Exception as e:
                logger.warning(f"✗ BERT fitting failed: {e}")
        else:
            logger.info("Skipping BERT until first use")

        self.is_fitted = True
        logger.info("Ranking engine ready!")

    def _normalize_query(self, query: str, query_history: Optional[List[str]] = None) -> str:
        """Apply preprocessing and query refinement aligned with document indexing."""
        if not query:
            return ''
        history = None
        if query_history and QUERY_REFINEMENT.get('use_search_history', True):
            history = [
                refine_query_text(
                    q,
                    expand_synonyms=False,
                    do_spell_check=False,
                    max_synonyms_per_term=0,
                )
                for q in query_history
            ]
        refined = refine_query_text(
            query,
            expand_synonyms=QUERY_REFINEMENT['expand_synonyms'],
            do_spell_check=QUERY_REFINEMENT['do_spell_check'],
            max_synonyms_per_term=QUERY_REFINEMENT['max_synonyms_per_term'],
            history_queries=history,
        )
        return refined if refined else query

    def suggest_queries(self, prefix: str, limit: int = 8) -> List[str]:
        """Suggest query terms from the inverted index vocabulary."""
        if not prefix or not self.is_fitted:
            return []

        prefix = prefix.strip().lower()
        if len(prefix) < 2:
            return []

        matches = [
            (term, len(postings))
            for term, postings in self.index.postings.items()
            if term.startswith(prefix)
        ]
        matches.sort(key=lambda x: x[1], reverse=True)
        return [term for term, _ in matches[:limit]]

    def _full_rank(self, method: str, query: str, query_history: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        normalized = self._normalize_query(query, query_history)
        n = len(self.doc_ids)
        if method == 'TF-IDF':
            return self.tfidf.get_top_k(normalized, k=n)
        if method == 'BM25':
            return self.bm25.get_top_k(normalized, k=n)
        if method == 'Word2Vec':
            if self.word2vec.model is None:
                self.word2vec.fit(self.processed_docs, self.doc_ids)
            return self.word2vec.score_query(normalized, top_k=n)
        if method == 'BERT':
            if self.bert.model is None:
                self.bert.fit(self.processed_docs, self.doc_ids)
            return self.bert.score_query(normalized, top_k=n)
        raise ValueError(f"Unknown ranker method: {method}")

    def _collect_ranker_results(self, methods: List[str], query: str,
                                query_history: Optional[List[str]] = None) -> List[RankerResult]:
        results = []
        for method in methods:
            try:
                ranked = self._full_rank(method, query, query_history)
                results.append(RankerResult(
                    method,
                    [d for d, _ in ranked],
                    [s for _, s in ranked],
                ))
            except Exception as e:
                logger.warning(f"Skipping {method} in fusion: {e}")
        return results

    def rank_tfidf(self, query: str, top_k: int = TOP_K, query_history=None) -> List[Tuple[str, float]]:
        return self.tfidf.get_top_k(self._normalize_query(query, query_history), k=top_k)

    def rank_bm25(self, query: str, top_k: int = TOP_K, query_history=None) -> List[Tuple[str, float]]:
        return self.bm25.get_top_k(self._normalize_query(query, query_history), k=top_k)

    def rank_index(self, query: str, top_k: int = TOP_K, query_history=None) -> List[Tuple[str, float]]:
        return self.index.get_top_k(self._normalize_query(query, query_history), k=top_k)

    def rank_word2vec(self, query: str, top_k: int = TOP_K, query_history=None) -> List[Tuple[str, float]]:
        if self.word2vec.model is None:
            logger.info("Lazy fitting Word2Vec embeddings on demand...")
            self.word2vec.fit(self.processed_docs, self.doc_ids)
        return self.word2vec.score_query(
            self._normalize_query(query, query_history), top_k=top_k
        )

    def rank_bert(self, query: str, top_k: int = TOP_K, query_history=None) -> List[Tuple[str, float]]:
        if self.bert.model is None:
            logger.info("Lazy fitting/loading BERT embeddings on demand...")
            self.bert.fit(self.processed_docs, self.doc_ids)
        return self.bert.score_query(
            self._normalize_query(query, query_history), top_k=top_k
        )

    def rank_serial_hybrid(self, query: str, top_k: int = TOP_K,
                           query_history=None) -> List[Tuple[str, float, Dict]]:
        """Serial: TF-IDF → BM25 → Word2Vec → BERT"""
        results = self._collect_ranker_results(SERIAL_FUSION_ORDER, query, query_history)
        if not results:
            return []
        return self.serial_fusion.fuse(results, top_k=top_k)

    def rank_parallel_hybrid(self, query: str, top_k: int = TOP_K,
                             query_history=None) -> List[Tuple[str, float, Dict]]:
        """Parallel: TF-IDF + BM25 + Word2Vec + BERT with weighted fusion."""
        results = self._collect_ranker_results(list(FUSION_WEIGHTS.keys()), query, query_history)
        if not results:
            return []
        return self.parallel_fusion.fuse(results, top_k=top_k)

    def rank_rrf_hybrid(self, query: str, top_k: int = TOP_K,
                        query_history=None) -> List[Tuple[str, float, Dict]]:
        results = self._collect_ranker_results(RRF_FUSION_ORDER, query, query_history)
        if not results:
            return []
        return self.rrf_fusion.fuse(results, top_k=top_k)

    def _build_search_result(self, doc_id: str, score: float, method: str, explanation=None) -> Dict:
        dataset_cfg = next(
            (cfg for cfg in DATASETS.values() if cfg["cache_prefix"] == self.cache_prefix),
            None,
        )
        original_text = ""
        if dataset_cfg and "document_db" in dataset_cfg:
            original_text = get_document(doc_id, dataset_cfg["document_db"])

        result = {
            'doc_id': doc_id,
            'score': float(score),
            'method': method,
            'document_text': original_text or self.documents.get(doc_id, '')
        }
        if explanation is not None:
            result['explanation'] = explanation
        return result

    def search(self, query: str, method: str = 'parallel', top_k: int = TOP_K,
               query_history: Optional[List[str]] = None) -> List[Dict]:
        if not self.is_fitted:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        dispatch = {
            'tfidf': lambda: self.rank_tfidf(query, top_k, query_history),
            'bm25': lambda: self.rank_bm25(query, top_k, query_history),
            'word2vec': lambda: self.rank_word2vec(query, top_k, query_history),
            'bert': lambda: self.rank_bert(query, top_k, query_history),
            'index': lambda: self.rank_index(query, top_k, query_history),
            'serial': lambda: self.rank_serial_hybrid(query, top_k, query_history),
            'parallel': lambda: self.rank_parallel_hybrid(query, top_k, query_history),
            'rrf': lambda: self.rank_rrf_hybrid(query, top_k, query_history),
        }

        labels = {
            'tfidf': 'TF-IDF', 'bm25': 'BM25', 'word2vec': 'Word2Vec',
            'bert': 'BERT', 'index': 'InvertedIndex',
            'serial': 'SerialFusion', 'parallel': 'ParallelFusion', 'rrf': 'RRF',
        }

        if method not in dispatch:
            raise ValueError(f"Unknown method: {method}")

        ranked = dispatch[method]()
        if method in ('serial', 'parallel', 'rrf'):
            return [self._build_search_result(d, s, labels[method], exp) for d, s, exp in ranked]
        return [self._build_search_result(d, s, labels[method]) for d, s in ranked]

    def get_parameter_report(self) -> Dict:
        return {
            'bm25': {'k1': self.bm25.k1, 'b': self.bm25.b},
            'fusion_weights': FUSION_WEIGHTS,
            'serial_pipeline': SERIAL_FUSION_ORDER,
            'rrf_pipeline': RRF_FUSION_ORDER,
            'num_documents': len(self.doc_ids),
            'cache_prefix': self.cache_prefix,
            'is_fitted': self.is_fitted,
        }
