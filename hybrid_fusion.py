# hybrid_fusion.py
"""
Hybrid fusion strategies for combining multiple ranking methods
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RankerResult:
    """Represents ranking results from a single method"""
    method: str
    doc_ids: List[str]
    scores: List[float]


class FusionStrategy:
    """Base class for fusion strategies"""
    
    def fuse(self, results: List[RankerResult], top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Fuse multiple ranking results"""
        raise NotImplementedError


class SerialFusion(FusionStrategy):
    """Sequential fusion: cascade through multiple methods"""
    
    def fuse(self, results: List[RankerResult], top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Apply methods sequentially: TF-IDF → BM25 → Embedding
        Each method refines results from the previous step
        """
        if not results:
            return []
        
        # Start with all documents from first method
        candidates = set(results[0].doc_ids)
        candidate_scores = {}
        
        logger.info(f"Serial fusion: Starting with {len(candidates)} candidates from {results[0].method}")
        
        # Apply each subsequent method to refine candidates
        for result in results[1:]:
            new_candidates = set()
            
            # Score candidates using this method
            for doc_id, score in zip(result.doc_ids, result.scores):
                if doc_id in candidates:
                    new_candidates.add(doc_id)
                    # Accumulate scores
                    if doc_id not in candidate_scores:
                        candidate_scores[doc_id] = 0
                    candidate_scores[doc_id] += score
            
            candidates = new_candidates if new_candidates else candidates
            logger.info(f"After {result.method}: {len(candidates)} candidates")
        
        # Sort by accumulated scores
        sorted_results = sorted(
            [(doc_id, candidate_scores.get(doc_id, 0.0)) for doc_id in candidates],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-k with explanation
        final_results = []
        for doc_id, score in sorted_results[:top_k]:
            method_scores = {}
            for result in results:
                method_scores[result.method] = dict(zip(result.doc_ids, result.scores)).get(doc_id, 0.0)
            explanation = {
                'fusion_type': 'serial',
                'methods': method_scores,
                'pipeline': [r.method for r in results],
                'combined_score': score
            }
            final_results.append((doc_id, score, explanation))
        
        return final_results


class ParallelFusion(FusionStrategy):
    """Parallel fusion: combine scores from multiple methods"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize with weights for each method
        Args:
            weights: dict mapping method name to weight (should sum to ~1.0)
        """
        self.weights = weights or {}
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores or len(scores) == 0:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def fuse(self, results: List[RankerResult], top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Fuse scores from multiple methods in parallel
        Weighted average of normalized scores
        """
        if not results:
            return []
        
        # Collect all documents
        all_docs = set()
        for result in results:
            all_docs.update(result.doc_ids)
        
        logger.info(f"Parallel fusion: {len(all_docs)} unique documents from {len(results)} methods")
        
        # Create score mapping for each method
        method_scores = {}
        for result in results:
            method_scores[result.method] = dict(zip(result.doc_ids, result.scores))
        
        # Compute fused scores
        fused_scores = {}
        for doc_id in all_docs:
            total_weight = 0
            weighted_score = 0
            
            for result in results:
                method = result.method
                weight = self.weights.get(method, 1.0 / len(results))
                
                # Get score from this method (0 if not in top results)
                score = method_scores[method].get(doc_id, 0.0)
                
                weighted_score += weight * score
                total_weight += weight
            
            # Normalize by total weight
            fused_scores[doc_id] = weighted_score / total_weight if total_weight > 0 else 0
        
        # Sort by fused score
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k with explanation
        final_results = []
        for doc_id, fused_score in sorted_results[:top_k]:
            explanation = {
                'fusion_type': 'parallel',
                'methods': {r.method: method_scores[r.method].get(doc_id, 0.0) for r in results},
                'weights': {r.method: self.weights.get(r.method, 1.0/len(results)) for r in results},
                'fused_score': fused_score
            }
            final_results.append((doc_id, fused_score, explanation))
        
        return final_results


class RRFFusion(FusionStrategy):
    """Reciprocal Rank Fusion (RRF)"""
    
    def fuse(self, results: List[RankerResult], top_k: int = 10, k: float = 60) -> List[Tuple[str, float, Dict]]:
        """
        RRF formula: score = sum(1 / (k + rank))
        where k is a constant (default 60) and rank is 1-based position
        """
        if not results:
            return []
        
        rrf_scores = {}
        
        for result in results:
            # Assign ranks (1-indexed)
            for rank, doc_id in enumerate(result.doc_ids, 1):
                rrf_score = 1.0 / (k + rank)
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += rrf_score
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k with explanation
        final_results = []
        for doc_id, rrf_score in sorted_results[:top_k]:
            method_scores = {}
            for result in results:
                rank_map = {d: i + 1 for i, d in enumerate(result.doc_ids)}
                if doc_id in rank_map:
                    method_scores[result.method] = 1.0 / (k + rank_map[doc_id])
                else:
                    method_scores[result.method] = 0.0
            explanation = {
                'fusion_type': 'rrf',
                'methods': method_scores,
                'rrf_score': rrf_score,
                'rrf_k': k,
            }
            final_results.append((doc_id, rrf_score, explanation))
        
        return final_results
