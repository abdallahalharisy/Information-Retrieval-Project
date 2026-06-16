# ranking_engine.py
"""
Main ranking engine orchestrating all representation methods and fusion strategies
"""

import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from representations import (
    TFIDFRepresentation,
    BM25Representation,
    Word2VecEmbedding,
    BERTEmbedding
)
from hybrid_fusion import (
    RankerResult,
    SerialFusion,
    ParallelFusion,
    RRFFusion
)
from config import FUSION_WEIGHTS, TOP_K

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RankingEngine:
    """Main IR engine supporting multiple ranking methods and fusion strategies"""
    
    def __init__(self):
        self.tfidf = TFIDFRepresentation()
        self.bm25 = BM25Representation()
        self.word2vec = Word2VecEmbedding()
        self.bert = BERTEmbedding()
        
        self.is_fitted = False
        self.documents = {}  # doc_id -> raw text
        self.doc_ids = []
        
        # Fusion strategies
        self.serial_fusion = SerialFusion()
        self.parallel_fusion = ParallelFusion(weights=FUSION_WEIGHTS)
        self.rrf_fusion = RRFFusion()
    
    def fit(self, documents: Dict[str, List[str]]):
        """
        Fit all ranking methods on documents
        Args:
            documents: dict mapping doc_id -> list of preprocessed tokens (or string)
        """
        # Convert to text format if needed
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
        
        logger.info(f"Fitting ranking engine on {len(doc_ids)} documents")
        
        # Fit TF-IDF
        try:
            self.tfidf.fit(processed_docs, doc_ids)
            logger.info("✓ TF-IDF fitted")
        except Exception as e:
            logger.warning(f"✗ TF-IDF fitting failed: {e}")
        
        # Fit BM25
        try:
            self.bm25.fit(processed_docs, doc_ids)
            logger.info("✓ BM25 fitted")
        except Exception as e:
            logger.warning(f"✗ BM25 fitting failed: {e}")
        
        # Fit Word2Vec
        try:
            self.word2vec.fit(processed_docs, doc_ids)
            logger.info("✓ Word2Vec fitted")
        except Exception as e:
            logger.warning(f"✗ Word2Vec fitting failed: {e}")
        
        # Fit BERT (can be slow)
        try:
            logger.info("Fitting BERT (this may take a while)...")
            self.bert.fit(processed_docs, doc_ids)
            logger.info("✓ BERT fitted")
        except Exception as e:
            logger.warning(f"✗ BERT fitting failed: {e}")
        
        self.is_fitted = True
        logger.info("Ranking engine ready!")
    
    def rank_tfidf(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        """Rank documents using TF-IDF"""
        return self.tfidf.get_top_k(query, k=top_k)
    
    def rank_bm25(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        """Rank documents using BM25"""
        return self.bm25.get_top_k(query, k=top_k)
    
    def rank_word2vec(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        """Rank documents using Word2Vec embeddings"""
        return self.word2vec.score_query(query, top_k=top_k)
    
    def rank_bert(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        """Rank documents using BERT embeddings"""
        return self.bert.score_query(query, top_k=top_k)
    
    def rank_serial_hybrid(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float, Dict]]:
        """Rank documents using serial hybrid fusion"""
        # Get results from each method
        tfidf_results = self.rank_tfidf(query, top_k=len(self.doc_ids))
        bm25_results = self.rank_bm25(query, top_k=len(self.doc_ids))
        
        # Convert to RankerResult objects
        results = [
            RankerResult('TF-IDF', [d for d, s in tfidf_results], [s for d, s in tfidf_results]),
            RankerResult('BM25', [d for d, s in bm25_results], [s for d, s in bm25_results])
        ]
        
        return self.serial_fusion.fuse(results, top_k=top_k)
    
    def rank_parallel_hybrid(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float, Dict]]:
        """Rank documents using parallel hybrid fusion"""
        # Get results from each method
        tfidf_results = self.rank_tfidf(query, top_k=len(self.doc_ids))
        bm25_results = self.rank_bm25(query, top_k=len(self.doc_ids))
        
        # Try to get embedding results if available
        try:
            w2v_results = self.rank_word2vec(query, top_k=len(self.doc_ids))
            results = [
                RankerResult('TF-IDF', [d for d, s in tfidf_results], [s for d, s in tfidf_results]),
                RankerResult('BM25', [d for d, s in bm25_results], [s for d, s in bm25_results]),
                RankerResult('Word2Vec', [d for d, s in w2v_results], [s for d, s in w2v_results])
            ]
        except:
            results = [
                RankerResult('TF-IDF', [d for d, s in tfidf_results], [s for d, s in tfidf_results]),
                RankerResult('BM25', [d for d, s in bm25_results], [s for d, s in bm25_results])
            ]
        
        return self.parallel_fusion.fuse(results, top_k=top_k)
    
    def rank_rrf_hybrid(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float, Dict]]:
        """Rank documents using RRF (Reciprocal Rank Fusion)"""
        # Get results from each method
        tfidf_results = self.rank_tfidf(query, top_k=len(self.doc_ids))
        bm25_results = self.rank_bm25(query, top_k=len(self.doc_ids))
        
        results = [
            RankerResult('TF-IDF', [d for d, s in tfidf_results], [s for d, s in tfidf_results]),
            RankerResult('BM25', [d for d, s in bm25_results], [s for d, s in bm25_results])
        ]
        
        return self.rrf_fusion.fuse(results, top_k=top_k)
    
    def search(self, query: str, method: str = 'parallel', top_k: int = TOP_K) -> List[Dict]:
        """
        Search for documents using specified method
        
        Args:
            query: search query
            method: one of 'tfidf', 'bm25', 'word2vec', 'bert', 'serial', 'parallel', 'rrf'
            top_k: number of results to return
        
        Returns:
            List of result dictionaries with doc_id, score, and explanation
        """
        if not self.is_fitted:
            raise RuntimeError("Engine not fitted. Call fit() first.")
        
        results = []
        
        try:
            if method == 'tfidf':
                ranked = self.rank_tfidf(query, top_k=top_k)
                results = [{'doc_id': d, 'score': float(s), 'method': 'TF-IDF'} 
                          for d, s in ranked]
            
            elif method == 'bm25':
                ranked = self.rank_bm25(query, top_k=top_k)
                results = [{'doc_id': d, 'score': float(s), 'method': 'BM25'} 
                          for d, s in ranked]
            
            elif method == 'word2vec':
                ranked = self.rank_word2vec(query, top_k=top_k)
                results = [{'doc_id': d, 'score': float(s), 'method': 'Word2Vec'} 
                          for d, s in ranked]
            
            elif method == 'bert':
                ranked = self.rank_bert(query, top_k=top_k)
                results = [{'doc_id': d, 'score': float(s), 'method': 'BERT'} 
                          for d, s in ranked]
            
            elif method == 'serial':
                ranked = self.rank_serial_hybrid(query, top_k=top_k)
                results = [{'doc_id': d, 'score': float(s), 'explanation': exp} 
                          for d, s, exp in ranked]
            
            elif method == 'parallel':
                ranked = self.rank_parallel_hybrid(query, top_k=top_k)
                results = [{'doc_id': d, 'score': float(s), 'explanation': exp} 
                          for d, s, exp in ranked]
            
            elif method == 'rrf':
                ranked = self.rank_rrf_hybrid(query, top_k=top_k)
                results = [{'doc_id': d, 'score': float(s), 'explanation': exp} 
                          for d, s, exp in ranked]
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
        
        return results
    
    def get_parameter_report(self) -> Dict:
        """Generate report of current parameters"""
        return {
            'bm25': {
                'k1': self.bm25.k1,
                'b': self.bm25.b
            },
            'fusion_weights': FUSION_WEIGHTS,
            'num_documents': len(self.doc_ids),
            'is_fitted': self.is_fitted
        }
