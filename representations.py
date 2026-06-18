# representations.py
"""
Document representation methods: TF-IDF, BM25, and Embeddings
"""

import math
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import logging
from config import BM25_K1, BM25_B, EMBEDDING_MODELS
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class TFIDFRepresentation:
    """Vector Space Model using TF-IDF weights"""
    
    def __init__(self, max_features=5000, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.documents = []
        self.doc_ids = []
        
    def fit(self, documents: List[str], doc_ids: List[str]):
        """Fit TF-IDF model on documents"""
        self.documents = documents
        self.doc_ids = doc_ids
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        logger.info(f"TF-IDF fitted on {len(documents)} documents")
        
    def score_query(self, query: str) -> Tuple[List[str], List[float]]:
        """Score query against all documents"""
        query_vector = self.vectorizer.transform([query])
        scores_matrix = query_vector.dot(self.tfidf_matrix.T)
        scores = scores_matrix.toarray().ravel()
        
        # Sort by score descending
        ranked_indices = np.argsort(scores)[::-1]
        ranked_docs = [self.doc_ids[i] for i in ranked_indices]
        ranked_scores = [float(scores[i]) for i in ranked_indices]
        
        return ranked_docs, ranked_scores
    
    def get_top_k(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k documents for query"""
        doc_ids, scores = self.score_query(query)
        return list(zip(doc_ids[:k], scores[:k]))


class BM25Representation:
    """Probabilistic ranking function (BM25)"""
    
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.tokenized_docs = []
        
    def fit(self, documents: List[str], doc_ids: List[str]):
        """Fit BM25 model on tokenized documents"""
        self.documents = documents
        self.doc_ids = doc_ids
        # Tokenize: simple split on whitespace (assumes preprocessing already done)
        self.tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
        logger.info(f"BM25 fitted on {len(documents)} documents (k1={self.k1}, b={self.b})")
    
    def score_query(self, query: str) -> Tuple[List[str], List[float]]:
        """Score query using BM25"""
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Sort by score descending
        ranked_indices = np.argsort(scores)[::-1]
        ranked_docs = [self.doc_ids[i] for i in ranked_indices]
        ranked_scores = [float(scores[i]) for i in ranked_indices]
        
        return ranked_docs, ranked_scores
    
    def get_top_k(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k documents for query"""
        doc_ids, scores = self.score_query(query)
        return list(zip(doc_ids[:k], scores[:k]))
    
    def update_parameters(self, k1: float = None, b: float = None):
        """Update BM25 parameters and refit"""
        if k1 is not None:
            self.k1 = k1
        if b is not None:
            self.b = b
        if self.documents:
            self.fit(self.documents, self.doc_ids)


class InvertedIndexRepresentation:
    """Simple inverted index for fast boolean and weighted retrieval."""

    def __init__(self):
        self.postings = defaultdict(dict)
        self.doc_freq = {}
        self.doc_lengths = {}
        self.num_docs = 0
        self.documents = []
        self.doc_ids = []

    def fit(self, documents: List[str], doc_ids: List[str]):
        """Build the inverted index from preprocessed documents."""
        self.documents = documents
        self.doc_ids = doc_ids
        self.num_docs = len(doc_ids)
        self.postings = defaultdict(dict)
        self.doc_lengths = {}

        tokenized_docs = [doc.split() for doc in documents]
        for doc_id, tokens in zip(doc_ids, tokenized_docs):
            self.doc_lengths[doc_id] = len(tokens)
            for token in tokens:
                self.postings[token][doc_id] = self.postings[token].get(doc_id, 0) + 1

        self.doc_freq = {term: len(docs) for term, docs in self.postings.items()}
        logger.info(f"Inverted index built on {self.num_docs} documents")

    def score_query(self, query: str) -> Tuple[List[str], List[float]]:
        """Score documents using term frequency and IDF from the inverted index."""
        query_terms = query.split()
        if not query_terms:
            return self.doc_ids, [0.0] * len(self.doc_ids)

        scores = defaultdict(float)
        for term in query_terms:
            postings = self.postings.get(term, {})
            if not postings:
                continue
            idf = math.log((self.num_docs + 1) / (len(postings) + 1)) + 1.0
            for doc_id, tf in postings.items():
                scores[doc_id] += (1.0 + math.log(tf)) * idf

        if not scores:
            return self.doc_ids, [0.0] * len(self.doc_ids)

        # Normalize by document length to reduce bias toward long documents
        scored_items = [
            (doc_id, score / (self.doc_lengths.get(doc_id, 1) + 1.0))
            for doc_id, score in scores.items()
        ]
        scored_items.sort(key=lambda x: x[1], reverse=True)

        ranked_docs = [doc_id for doc_id, _ in scored_items]
        ranked_scores = [float(score) for _, score in scored_items]
        return ranked_docs, ranked_scores

    def get_top_k(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Return the top-k documents for the query."""
        doc_ids, scores = self.score_query(query)
        return list(zip(doc_ids[:k], scores[:k]))


class EmbeddingRepresentation:
    """Base class for embedding-based representations"""
    
    def __init__(self, embedding_type: str = 'word2vec'):
        self.embedding_type = embedding_type
        self.model = None
        self.doc_vectors = {}
        self.doc_ids = []
        
    def fit(self, documents: List[str], doc_ids: List[str]):
        """Fit embedding model (to be implemented in subclasses)"""
        raise NotImplementedError
    
    def get_vector(self, text: str) -> np.ndarray:
        """Get embedding vector for text (to be implemented in subclasses)"""
        raise NotImplementedError
    
    def score_query(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Score query using cosine similarity"""
        query_vector = self.get_vector(query)
        
        if query_vector is None or np.allclose(query_vector, 0):
            logger.warning("Query vector is zero or None")
            return [(doc_id, 0.0) for doc_id in self.doc_ids[:top_k]]
        
        scores = {}
        for doc_id in self.doc_ids:
            if doc_id in self.doc_vectors:
                doc_vector = self.doc_vectors[doc_id]
                # Cosine similarity
                similarity = np.dot(query_vector, doc_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(doc_vector) + 1e-10
                )
                scores[doc_id] = similarity
        
        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


class Word2VecEmbedding(EmbeddingRepresentation):
    """Word2Vec-based embedding representation"""
    
    def __init__(self):
        super().__init__('word2vec')
        try:
            from gensim.models import Word2Vec
            self.Word2Vec = Word2Vec
        except ImportError:
            logger.warning("Gensim not installed. Word2Vec will be disabled.")
            self.Word2Vec = None
    
    def fit(self, documents: List[str], doc_ids: List[str]):
        """Train Word2Vec on documents"""
        if not self.Word2Vec:
            logger.warning("Skipping Word2Vec: gensim not installed")
            return
        
        self.doc_ids = doc_ids
        # Tokenize documents
        tokenized = [doc.split() for doc in documents]
        
        # Train Word2Vec
        config = EMBEDDING_MODELS['word2vec']
        self.model = self.Word2Vec(
            sentences=tokenized,
            vector_size=config['vector_size'],
            window=config['window'],
            min_count=config['min_count'],
            epochs=config['epochs'],
            workers=4
        )
        
        # Compute document vectors as average of word vectors
        for doc_id, tokens in zip(doc_ids, tokenized):
            vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
            if vectors:
                self.doc_vectors[doc_id] = np.mean(vectors, axis=0)
            else:
                self.doc_vectors[doc_id] = np.zeros(config['vector_size'])
        
        logger.info(f"Word2Vec trained on {len(documents)} documents")
    
    def get_vector(self, text: str) -> np.ndarray:
        """Get Word2Vec embedding for text"""
        if not self.model:
            return None
        
        tokens = text.split()
        vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(EMBEDDING_MODELS['word2vec']['vector_size'])


class BERTEmbedding(EmbeddingRepresentation):
    """BERT-based embedding representation"""
    
    def __init__(self):
        super().__init__('bert')
        try:
            from sentence_transformers import SentenceTransformer
            self.SentenceTransformer = SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not installed. BERT will be disabled.")
            self.SentenceTransformer = None
    
    def fit(self, documents: List[str], doc_ids: List[str]):
        """Compute BERT embeddings for documents"""
        if not self.SentenceTransformer:
            logger.warning("Skipping BERT: sentence-transformers not installed")
            return
        
        self.doc_ids = doc_ids
        config = EMBEDDING_MODELS['bert']
        
        # Load pretrained BERT model
        self.model = self.SentenceTransformer(config['model_name'])
        
        # Compute embeddings
        logger.info(f"Computing BERT embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(documents, batch_size=32, show_progress_bar=True)
        
        for doc_id, embedding in zip(doc_ids, embeddings):
            self.doc_vectors[doc_id] = embedding
        
        logger.info(f"BERT embeddings computed for {len(documents)} documents")
    
    def get_vector(self, text: str) -> np.ndarray:
        """Get BERT embedding for text"""
        if not self.model:
            return None
        
        return self.model.encode(text)
