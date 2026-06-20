"""Dataset registry shared across services."""

DATASETS = {
    "msmarco": {
        "name": "MSMARCO-Document",
        "file": "processed_data_msmarco.json",
        "document_db": "data/documents_msmarco.db",
        "cache_prefix": "msmarco",
        "description": "MSMARCO web documents",
        "ir_name": "msmarco-document/dev",
        "dataset_url": "https://ir-datasets.com/msmarco-document.html#msmarco-document/dev",
        "qrels_url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz",
        "queries_url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz",
    },
}

DATASET_ALIASES = {
    "MSMARCO-Document": "msmarco",
    "1": "msmarco",
    "2": "msmarco",
}


def resolve_dataset_key(key: str) -> str:
    key = key.strip()
    if key in DATASETS:
        return key
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    raise ValueError(f"Unknown dataset: {key}")
