# ranking_engine.py
"""
Main ranking engine orchestrating all representation methods and fusion strategies
"""

import json
import logging
import os
import joblib
from collections import OrderedDict
from typing import Any, List, Dict, Tuple, Optional
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
from shared.document_store import get_document, get_documents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BM25_CANDIDATE_MULTIPLIER = 10
FUSION_CANDIDATE_MULTIPLIER = 6
SEMANTIC_CANDIDATE_MULTIPLIER = 4
MIN_LEXICAL_CANDIDATES = 40
MIN_FUSION_CANDIDATES = 60
MIN_SEMANTIC_CANDIDATES = 30
MAX_SEARCH_CACHE_SIZE = 256


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
        self.raw_docs = []
        self.doc_ids = []
        self._normalized_query_cache = {}
        self._search_cache = OrderedDict()

        self.serial_fusion = SerialFusion()
        self.parallel_fusion = ParallelFusion(weights=FUSION_WEIGHTS)
        self.rrf_fusion = RRFFusion()

        self.cache_dir = 'cache'
        self.cache_prefix = 'default'
        self.cache_version = 3
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
        self.raw_docs = cached.get('raw_docs') or list(self.processed_docs)
        self.documents = dict(zip(self.doc_ids, self.processed_docs))

        self.tfidf.vectorizer = cached['tfidf_vectorizer']
        self.tfidf.tfidf_matrix = cached['tfidf_matrix']
        self.tfidf.doc_ids = self.doc_ids
        self.tfidf.documents = self.processed_docs

        self.bm25.k1 = cached['bm25_k1']
        self.bm25.b = cached['bm25_b']
        self.bm25.documents = self.processed_docs
        self.bm25.doc_ids = self.doc_ids
        self.bm25.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        self.bm25.tokenized_docs = cached.get('bm25_tokenized_docs', [])
        self.bm25.idf = cached.get('bm25_idf', {})
        self.bm25.avgdl = cached['bm25_avgdl']
        self.bm25._build_term_index()

        if cached.get('index_cached', False):
            self.index.postings = cached['index_postings']
            self.index.doc_freq = cached['index_doc_freq']
            self.index.doc_lengths = cached['index_doc_lengths']
            self.index.num_docs = len(self.doc_ids)
            self.index.documents = self.processed_docs
            self.index.doc_ids = self.doc_ids
        else:
            # The inverted index can be very large for 210K docs, so it is
            # built lazily only when the user selects the "index" method.
            self.index.postings = {}
            self.index.doc_freq = {}
            self.index.doc_lengths = {}
            self.index.num_docs = 0
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
                'raw_docs': self.raw_docs if self.raw_docs != self.processed_docs else None,
                'tfidf_vectorizer': self.tfidf.vectorizer,
                'tfidf_matrix': self.tfidf.tfidf_matrix,
                'bm25_k1': self.bm25.k1,
                'bm25_b': self.bm25.b,
                'bm25_idf': self.bm25.idf,
                'bm25_avgdl': self.bm25.avgdl,
                'index_cached': False,
            }
            tmp_cache_file = f"{self.cache_file}.tmp"
            tmp_meta_file = f"{self.cache_meta_file}.tmp"
            for tmp_path in (tmp_cache_file, tmp_meta_file):
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            joblib.dump(cached, tmp_cache_file)

            meta = {
                'cache_version': self.cache_version,
                'cache_prefix': cache_prefix,
                'processed_data_mtime': os.path.getmtime(processed_data_file) if os.path.exists(processed_data_file) else None,
                'num_documents': len(self.doc_ids)
            }
            with open(tmp_meta_file, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            os.replace(tmp_cache_file, self.cache_file)
            os.replace(tmp_meta_file, self.cache_meta_file)

            logger.info(f"Saved engine cache for '{cache_prefix}'.")
        except Exception as e:
            logger.warning(f"Failed to save engine cache: {e}")

    def fit_processed_texts(self, processed_docs: List[str], doc_ids: List[str],
                            raw_docs: Optional[List[str]] = None,
                            fit_word2vec: bool = False, fit_bert: bool = False):
        """Fit ranking methods from already-normalized document strings."""
        self.doc_ids = doc_ids
        self.processed_docs = processed_docs
        self.raw_docs = raw_docs if raw_docs is not None else list(processed_docs)
        self.documents = dict(zip(doc_ids, processed_docs))

        logger.info(f"Fitting ranking engine on {len(doc_ids)} documents")

        for name, fn in [
            ('TF-IDF', lambda: self.tfidf.fit(processed_docs, doc_ids)),
            ('BM25', lambda: self.bm25.fit(processed_docs, doc_ids)),
        ]:
            try:
                fn()
                logger.info(f"✓ {name} fitted")
            except Exception as e:
                logger.warning(f"✗ {name} fitting failed: {e}")

        self.index.documents = processed_docs
        self.index.doc_ids = doc_ids
        logger.info("Skipping inverted index until first index search")

        if fit_word2vec:
            try:
                self.word2vec.fit(self.raw_docs, doc_ids)
                logger.info("✓ Word2Vec fitted")
            except Exception as e:
                logger.warning(f"✗ Word2Vec fitting failed: {e}")
        else:
            logger.info("Skipping Word2Vec until first use")

        if fit_bert:
            try:
                logger.info("Fitting BERT (this may take a while)...")
                self.bert.fit(self.raw_docs, doc_ids)
                logger.info("✓ BERT fitted")
            except Exception as e:
                logger.warning(f"✗ BERT fitting failed: {e}")
        else:
            logger.info("Skipping BERT until first use")

        self.is_fitted = True
        logger.info("Ranking engine ready!")

    def fit(self, documents: Dict[str, Any], fit_word2vec: bool = False, fit_bert: bool = False):
        """Fit ranking methods on documents."""
        processed_docs = []
        raw_docs = []
        doc_ids = []

        for doc_id, value in documents.items():
            if isinstance(value, dict):
                raw_text = str(
                    value.get('raw_text')
                    or value.get('text')
                    or value.get('document_text')
                    or ""
                )
                tokens = value.get('tokens') or value.get('processed') or raw_text
                text = ' '.join(tokens) if isinstance(tokens, list) else str(tokens)
                raw_text = raw_text or text
            elif isinstance(value, list):
                text = ' '.join(value)
                raw_text = text
            else:
                text = str(value)
                raw_text = text

            processed_docs.append(text)
            raw_docs.append(raw_text)
            doc_ids.append(doc_id)

        self.fit_processed_texts(
            processed_docs,
            doc_ids,
            raw_docs=raw_docs,
            fit_word2vec=fit_word2vec,
            fit_bert=fit_bert,
        )

    def _normalize_query(self, query: str, query_history: Optional[List[str]] = None) -> str:
        """Apply preprocessing and query refinement aligned with document indexing."""
        if not query:
            return ''
        cache_key = (query, tuple(query_history or []))
        if cache_key in self._normalized_query_cache:
            return self._normalized_query_cache[cache_key]

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
        normalized = refined if refined else query
        self._normalized_query_cache[cache_key] = normalized
        return normalized

    def _cache_get(self, key):
        if key not in self._search_cache:
            return None
        value = self._search_cache.pop(key)
        self._search_cache[key] = value
        return value

    def _cache_set(self, key, value):
        self._search_cache[key] = value
        if len(self._search_cache) > MAX_SEARCH_CACHE_SIZE:
            self._search_cache.popitem(last=False)

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

    def _fusion_candidate_k(self, top_k: int) -> int:
        return min(len(self.doc_ids), max(top_k * FUSION_CANDIDATE_MULTIPLIER, MIN_FUSION_CANDIDATES))

    def _semantic_candidate_k(self, top_k: int) -> int:
        return min(len(self.doc_ids), max(top_k * SEMANTIC_CANDIDATE_MULTIPLIER, MIN_SEMANTIC_CANDIDATES))

    def _rank_bm25_with_tfidf_candidates(self, normalized_query: str, top_k: int,
                                         candidate_k: Optional[int] = None) -> List[Tuple[str, float]]:
        candidate_k = candidate_k or min(
            len(self.doc_ids),
            max(top_k * BM25_CANDIDATE_MULTIPLIER, MIN_LEXICAL_CANDIDATES),
        )
        cache_key = ("bm25_tfidf_candidates", normalized_query, top_k, candidate_k)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        candidates = [doc_id for doc_id, _ in self.tfidf.get_top_k(normalized_query, k=candidate_k)]
        ranked = self.bm25.rerank_candidates(normalized_query, candidates, k=top_k)
        self._cache_set(cache_key, ranked)
        return ranked

    def _raw_text_for_doc_id(self, doc_id: str) -> str:
        doc_idx = self.bm25.doc_id_to_idx.get(doc_id)
        if doc_idx is not None and doc_idx < len(self.raw_docs):
            raw_text = self.raw_docs[doc_idx]
            if raw_text and raw_text != self.processed_docs[doc_idx]:
                return raw_text

        dataset_cfg = next(
            (cfg for cfg in DATASETS.values() if cfg["cache_prefix"] == self.cache_prefix),
            None,
        )
        if dataset_cfg and "document_db" in dataset_cfg:
            raw_text = get_document(doc_id, dataset_cfg["document_db"])
            if raw_text:
                return raw_text

        if doc_idx is not None and doc_idx < len(self.processed_docs):
            return self.processed_docs[doc_idx]
        return ""

    def _lexical_candidate_raw_texts(
        self,
        normalized_query: str,
        top_k: int,
        candidate_k: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        candidate_k = candidate_k or self._semantic_candidate_k(top_k)
        candidates = [
            doc_id
            for doc_id, _ in self._rank_bm25_with_tfidf_candidates(
                normalized_query,
                top_k=candidate_k,
                candidate_k=max(candidate_k, MIN_LEXICAL_CANDIDATES),
            )
        ]
        dataset_cfg = next(
            (cfg for cfg in DATASETS.values() if cfg["cache_prefix"] == self.cache_prefix),
            None,
        )
        raw_by_id = {}
        if dataset_cfg and "document_db" in dataset_cfg:
            raw_by_id = get_documents(candidates, dataset_cfg["document_db"])

        candidate_pairs = [
            (doc_id, raw_by_id.get(doc_id) or self._raw_text_for_doc_id(doc_id))
            for doc_id in candidates
        ]
        candidate_pairs = [
            (doc_id, text)
            for doc_id, text in candidate_pairs
            if text
        ]
        candidate_doc_ids = [doc_id for doc_id, _ in candidate_pairs]
        candidate_texts = [text for _, text in candidate_pairs]
        return candidate_doc_ids, candidate_texts

    def _word2vec_rerank_with_lexical_candidates(
        self,
        raw_query: str,
        normalized_query: str,
        top_k: int,
        candidate_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        candidate_doc_ids, candidate_texts = self._lexical_candidate_raw_texts(
            normalized_query,
            top_k,
            candidate_k,
        )
        return self.word2vec.rerank_candidates(
            raw_query,
            candidate_doc_ids,
            candidate_texts,
            top_k=top_k,
        )

    def _bert_rerank_with_lexical_candidates(
        self,
        raw_query: str,
        normalized_query: str,
        top_k: int,
        candidate_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        candidate_doc_ids, candidate_texts = self._lexical_candidate_raw_texts(
            normalized_query,
            top_k,
            candidate_k,
        )
        return self.bert.rerank_candidates(
            raw_query,
            candidate_doc_ids,
            candidate_texts,
            top_k=top_k,
        )

    def _full_rank(self, method: str, query: str, query_history: Optional[List[str]] = None,
                   candidate_k: Optional[int] = None) -> List[Tuple[str, float]]:
        normalized = self._normalize_query(query, query_history)
        n = min(candidate_k or len(self.doc_ids), len(self.doc_ids))
        cache_key = ("full_rank", method, query, normalized, tuple(query_history or []), n, candidate_k)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        if method == 'TF-IDF':
            ranked = self.tfidf.get_top_k(normalized, k=n)
        elif method == 'BM25':
            ranked = self._rank_bm25_with_tfidf_candidates(normalized, n, candidate_k=n)
        elif method == 'Word2Vec':
            ranked = self._word2vec_rerank_with_lexical_candidates(query, normalized, n, candidate_k)
        elif method == 'BERT':
            ranked = self._bert_rerank_with_lexical_candidates(query, normalized, n, candidate_k)
        else:
            raise ValueError(f"Unknown ranker method: {method}")
        self._cache_set(cache_key, ranked)
        return ranked

    def _collect_ranker_results(self, methods: List[str], query: str,
                                query_history: Optional[List[str]] = None,
                                candidate_k: Optional[int] = None) -> List[RankerResult]:
        results = []
        for method in methods:
            try:
                ranked = self._full_rank(method, query, query_history, candidate_k)
                results.append(RankerResult(
                    method,
                    [d for d, _ in ranked],
                    [s for _, s in ranked],
                ))
            except Exception as e:
                logger.warning(f"Skipping {method} in fusion: {e}")
        return results

    def rank_tfidf(self, query: str, top_k: int = TOP_K, query_history=None) -> List[Tuple[str, float]]:
        normalized = self._normalize_query(query, query_history)
        cache_key = ("rank", "tfidf", normalized, top_k)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        ranked = self.tfidf.get_top_k(normalized, k=top_k)
        self._cache_set(cache_key, ranked)
        return ranked

    def rank_bm25(self, query: str, top_k: int = TOP_K, query_history=None) -> List[Tuple[str, float]]:
        normalized = self._normalize_query(query, query_history)
        cache_key = ("rank", "bm25", normalized, top_k)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        ranked = self._rank_bm25_with_tfidf_candidates(normalized, top_k)
        self._cache_set(cache_key, ranked)
        return ranked

    def rank_index(self, query: str, top_k: int = TOP_K, query_history=None) -> List[Tuple[str, float]]:
        # Use the sparse TF-IDF matrix for interactive index-style retrieval.
        # Building/scanning a Python inverted index over 210K docs is too slow per request.
        return self.rank_tfidf(query, top_k, query_history)

    def rank_word2vec(self, query: str, top_k: int = TOP_K, query_history=None) -> List[Tuple[str, float]]:
        logger.info("Using Word2Vec to re-rank lexical candidates.")
        return self._word2vec_rerank_with_lexical_candidates(
            query,
            self._normalize_query(query, query_history),
            top_k,
        )

    def rank_bert(self, query: str, top_k: int = TOP_K, query_history=None) -> List[Tuple[str, float]]:
        logger.info("Using BERT to re-rank lexical candidates.")
        return self._bert_rerank_with_lexical_candidates(
            query,
            self._normalize_query(query, query_history),
            top_k,
        )

    def rank_serial_hybrid(self, query: str, top_k: int = TOP_K,
                           query_history=None) -> List[Tuple[str, float, Dict]]:
        """Serial: TF-IDF → BM25 → Word2Vec → BERT"""
        results = self._collect_ranker_results(
            SERIAL_FUSION_ORDER, query, query_history, self._fusion_candidate_k(top_k)
        )
        if not results:
            return []
        return self.serial_fusion.fuse(results, top_k=top_k)

    def rank_parallel_hybrid(self, query: str, top_k: int = TOP_K,
                             query_history=None) -> List[Tuple[str, float, Dict]]:
        """Parallel: TF-IDF + BM25 + Word2Vec + BERT with weighted fusion."""
        results = self._collect_ranker_results(
            list(FUSION_WEIGHTS.keys()), query, query_history, self._fusion_candidate_k(top_k)
        )
        if not results:
            return []
        return self.parallel_fusion.fuse(results, top_k=top_k)

    def rank_rrf_hybrid(self, query: str, top_k: int = TOP_K,
                        query_history=None) -> List[Tuple[str, float, Dict]]:
        results = self._collect_ranker_results(
            RRF_FUSION_ORDER, query, query_history, self._fusion_candidate_k(top_k)
        )
        if not results:
            return []
        return self.rrf_fusion.fuse(results, top_k=top_k)

    def _build_search_result(
        self,
        doc_id: str,
        score: float,
        method: str,
        explanation=None,
        include_document_text: bool = True,
    ) -> Dict:
        dataset_cfg = next(
            (cfg for cfg in DATASETS.values() if cfg["cache_prefix"] == self.cache_prefix),
            None,
        )
        original_text = ""
        if include_document_text and dataset_cfg and "document_db" in dataset_cfg:
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
               query_history: Optional[List[str]] = None,
               include_document_text: bool = True) -> List[Dict]:
        if not self.is_fitted:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        cache_key = (
            "search",
            method,
            query,
            tuple(query_history or []),
            top_k,
            include_document_text,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

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
            results = [
                self._build_search_result(d, s, labels[method], exp, include_document_text)
                for d, s, exp in ranked
            ]
        else:
            results = [
                self._build_search_result(d, s, labels[method], include_document_text=include_document_text)
                for d, s in ranked
            ]
        self._cache_set(cache_key, results)
        return results

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
