# representations.py
"""
Document representation methods: TF-IDF, BM25, and Embeddings
"""

import math
import heapq
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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
        scores = query_vector.dot(self.tfidf_matrix.T).tocsr()
        if scores.nnz == 0:
            return [], []

        row = scores.getrow(0)
        ranked_positions = np.argsort(row.data)[::-1]
        ranked_indices = row.indices[ranked_positions]
        ranked_docs = [self.doc_ids[i] for i in ranked_indices]
        ranked_scores = [float(row.data[i]) for i in ranked_positions]
        
        return ranked_docs, ranked_scores
    
    def get_top_k(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k documents for query"""
        query_vector = self.vectorizer.transform([query])
        scores = query_vector.dot(self.tfidf_matrix.T).tocsr()
        if scores.nnz == 0:
            return []

        row = scores.getrow(0)
        k = min(k, row.nnz)
        if k <= 0:
            return []
        if row.nnz > k:
            top_positions = np.argpartition(row.data, -k)[-k:]
            top_positions = top_positions[np.argsort(row.data[top_positions])[::-1]]
        else:
            top_positions = np.argsort(row.data)[::-1]
        return [(self.doc_ids[row.indices[i]], float(row.data[i])) for i in top_positions]


class BM25Representation:
    """Probabilistic ranking function (BM25)"""
    
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.tokenized_docs = []
        self.term_postings = defaultdict(dict)
        self.doc_lengths = []
        self.avgdl = 0.0
        self.idf = {}
        self.doc_id_to_idx = {}

    def _build_term_index(self):
        """Reset lazy postings used to score only documents containing query terms."""
        self.term_postings = defaultdict(dict)
        if self.tokenized_docs:
            self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        else:
            self.doc_lengths = [len(doc.split()) for doc in self.documents]
        self.avgdl = float(np.mean(self.doc_lengths)) if self.doc_lengths else 0.0

    def _get_term_postings(self, token: str) -> Dict[int, int]:
        if token in self.term_postings:
            return self.term_postings[token]

        postings = {}
        tokenized_iter = self.tokenized_docs or (doc.split() for doc in self.documents)
        for idx, tokens in enumerate(tokenized_iter):
            freq = tokens.count(token)
            if freq:
                postings[idx] = freq
        self.term_postings[token] = postings
        return postings

    def _get_idf(self, token: str, doc_freq: int) -> float:
        if token not in self.idf:
            total_docs = len(self.doc_ids)
            self.idf[token] = math.log(total_docs - doc_freq + 0.5) - math.log(doc_freq + 0.5)
        return self.idf[token]
        
    def fit(self, documents: List[str], doc_ids: List[str]):
        """Fit BM25 model on tokenized documents"""
        self.documents = documents
        self.doc_ids = doc_ids
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        # Keep processed strings only; per-term postings are built lazily per query.
        self.tokenized_docs = []
        self._build_term_index()
        logger.info(f"BM25 fitted on {len(documents)} documents (k1={self.k1}, b={self.b})")

    def _candidate_scores(self, query_tokens: List[str]) -> Dict[int, float]:
        if not query_tokens:
            return {}

        avgdl = self.avgdl or 1.0
        scores = defaultdict(float)
        for token in query_tokens:
            postings = self._get_term_postings(token)
            if not postings:
                continue
            idf = self._get_idf(token, len(postings))
            for doc_idx, freq in postings.items():
                doc_len = self.doc_lengths[doc_idx] if doc_idx < len(self.doc_lengths) else 0
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                if denom:
                    scores[doc_idx] += idf * (freq * (self.k1 + 1) / denom)
        return scores
    
    def score_query(self, query: str) -> Tuple[List[str], List[float]]:
        """Score query using BM25"""
        query_tokens = query.split()
        scores = self._candidate_scores(query_tokens)
        if not scores:
            return [], []

        ranked_indices = sorted(scores, key=scores.get, reverse=True)
        ranked_docs = [self.doc_ids[i] for i in ranked_indices]
        ranked_scores = [float(scores[i]) for i in ranked_indices]
        
        return ranked_docs, ranked_scores
    
    def get_top_k(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k documents for query"""
        query_tokens = query.split()
        scores = self._candidate_scores(query_tokens)
        if not scores:
            return []

        top_items = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
        return [(self.doc_ids[i], float(score)) for i, score in top_items]

    def rerank_candidates(self, query: str, candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
        """Score a bounded candidate set with BM25."""
        query_tokens = query.split()
        if not query_tokens or not candidate_doc_ids:
            return []

        avgdl = self.avgdl or 1.0
        scored = []
        for doc_id in candidate_doc_ids:
            doc_idx = self.doc_id_to_idx.get(doc_id)
            if doc_idx is None:
                continue
            tokens = self.documents[doc_idx].split()
            doc_len = self.doc_lengths[doc_idx] if doc_idx < len(self.doc_lengths) else len(tokens)
            score = 0.0
            for token in query_tokens:
                freq = tokens.count(token)
                if not freq:
                    continue
                idf = self.idf.get(token, 1.0)
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                if denom:
                    score += idf * (freq * (self.k1 + 1) / denom)
            if score:
                scored.append((doc_id, float(score)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]
    
    def update_parameters(self, k1: float = None, b: float = None):
        """Update BM25 parameters and refit only when they actually change."""
        next_k1 = self.k1 if k1 is None else k1
        next_b = self.b if b is None else b
        if abs(float(next_k1) - float(self.k1)) < 1e-9 and abs(float(next_b) - float(self.b)) < 1e-9:
            return
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

    def _ensure_lengths(self):
        if not self.doc_lengths:
            self.doc_lengths = {
                doc_id: len(doc.split())
                for doc_id, doc in zip(self.doc_ids, self.documents)
            }

    def _get_postings(self, term: str) -> Dict[str, int]:
        if term in self.postings:
            return self.postings[term]

        postings = {}
        for doc_id, doc in zip(self.doc_ids, self.documents):
            freq = doc.split().count(term)
            if freq:
                postings[doc_id] = freq
        self.postings[term] = postings
        self.doc_freq[term] = len(postings)
        return postings

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
        self.num_docs = self.num_docs or len(self.doc_ids)
        self._ensure_lengths()
        query_terms = query.split()
        if not query_terms:
            return self.doc_ids, [0.0] * len(self.doc_ids)

        scores = defaultdict(float)
        for term in query_terms:
            postings = self._get_postings(term)
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

    def rerank_candidates(
        self,
        query: str,
        candidate_doc_ids: List[str],
        candidate_texts: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Train a small query-time Word2Vec model and re-rank lexical candidates."""
        if not self.Word2Vec:
            logger.warning("Skipping Word2Vec: gensim not installed")
            return []
        if not candidate_doc_ids:
            return []

        query_tokens = query.split()
        tokenized_docs = [text.split() for text in candidate_texts]
        training_sentences = [tokens for tokens in [query_tokens, *tokenized_docs] if tokens]
        if not query_tokens or not training_sentences:
            return []

        config = EMBEDDING_MODELS['word2vec']
        model = self.Word2Vec(
            sentences=training_sentences,
            vector_size=config['vector_size'],
            window=config['window'],
            min_count=1,
            epochs=config['epochs'],
            workers=1,
        )

        def average_vector(tokens):
            vectors = [model.wv[token] for token in tokens if token in model.wv]
            if not vectors:
                return np.zeros(config['vector_size'])
            return np.mean(vectors, axis=0)

        query_vector = average_vector(query_tokens)
        query_norm = np.linalg.norm(query_vector) + 1e-10
        scored = []
        for doc_id, tokens in zip(candidate_doc_ids, tokenized_docs):
            doc_vector = average_vector(tokens)
            score = np.dot(query_vector, doc_vector) / (
                query_norm * (np.linalg.norm(doc_vector) + 1e-10)
            )
            scored.append((doc_id, float(score)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


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

    def load_model(self):
        """Load the encoder without embedding the whole corpus."""
        if self.model is not None:
            return self.model
        if not self.SentenceTransformer:
            logger.warning("Skipping BERT: sentence-transformers not installed")
            return None
        config = EMBEDDING_MODELS['bert']
        self.model = self.SentenceTransformer(config['model_name'])
        return self.model

    def rerank_candidates(
        self,
        query: str,
        candidate_doc_ids: List[str],
        candidate_texts: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Use BERT as a query-time re-ranker over a small candidate set."""
        model = self.load_model()
        if model is None or not candidate_doc_ids:
            return []

        query_embedding = model.encode(query, convert_to_numpy=True)
        doc_embeddings = model.encode(
            candidate_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        query_norm = np.linalg.norm(query_embedding) + 1e-10
        doc_norms = np.linalg.norm(doc_embeddings, axis=1) + 1e-10
        scores = doc_embeddings.dot(query_embedding) / (doc_norms * query_norm)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (candidate_doc_ids[i], float(scores[i]))
            for i in ranked_indices
        ]
    
    def get_vector(self, text: str) -> np.ndarray:
        """Get BERT embedding for text"""
        if not self.load_model():
            return None
        
        return self.model.encode(text)
