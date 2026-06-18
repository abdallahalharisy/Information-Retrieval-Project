# config.py
"""Configuration and parameters for IR system"""

# ============ BM25 Parameters ============
BM25_K1 = 1.5  # Controls term frequency saturation point
BM25_B = 0.75  # Controls how much effect document length has on relevance

# ============ Hybrid Fusion Weights ============
# For parallel fusion: how much each method contributes to final score
FUSION_WEIGHTS = {
    'TF-IDF': 0.5,
    'BM25': 0.5,
}

# Serial hybrid applies methods in this order
SERIAL_FUSION_ORDER = ['TF-IDF', 'BM25']
RRF_FUSION_ORDER = ['TF-IDF', 'BM25']

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

# ============ Query Refinement ============
QUERY_REFINEMENT = {
    'expand_synonyms': False,
    'do_spell_check': False,
    'max_synonyms_per_term': 1,
    'use_search_history': True,
}

# ============ Serialization ============
CACHE_DIR = './ir_cache'
MODELS_DIR = './models'
REPORTS_DIR = './reports'

# ============ Logging ============
LOG_LEVEL = 'INFO'
VERBOSE = True
