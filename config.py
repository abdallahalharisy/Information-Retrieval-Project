# config.py
"""Configuration and parameters for IR system"""

# ============ BM25 Parameters ============
BM25_K1 = 1.5  # Controls term frequency saturation point
BM25_B = 0.75  # Controls how much effect document length has on relevance

# ============ Hybrid Fusion Weights ============
# For parallel fusion: how much each method contributes to final score
FUSION_WEIGHTS = {
    'tfidf': 0.33,      # TF-IDF weight
    'bm25': 0.33,       # BM25 weight
    'embedding': 0.34   # Embedding-based similarity weight
}

# ============ Embedding Models ============
EMBEDDING_MODELS = {
    'word2vec': {
        'enabled': True,
        'vector_size': 100,
        'window': 5,
        'min_count': 2,
        'epochs': 5
    },
    'bert': {
        'enabled': True,
        'model_name': 'bert-base-uncased',
        'cache_dir': './bert_cache'
    }
}

# ============ Ranking Settings ============
TOP_K = 10  # Return top-10 results by default
BATCH_SIZE = 32  # Process documents in batches for efficiency

# ============ Serialization ============
CACHE_DIR = './ir_cache'
MODELS_DIR = './models'
REPORTS_DIR = './reports'

# ============ Logging ============
LOG_LEVEL = 'INFO'
VERBOSE = True
