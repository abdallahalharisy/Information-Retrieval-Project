"""Dataset registry shared across services."""

DATASETS = {
    "msmarco": {
        "name": "MSMARCO-Document",
        "file": "processed_data_msmarco.json",
        "cache_prefix": "msmarco",
        "description": "210K+ web documents",
        "ir_name": "msmarco-document/dev",
    },
    "fever": {
        "name": "BEIR/Fever",
        "file": "processed_data_fever.json",
        "cache_prefix": "fever",
        "description": "Fact verification dataset",
        "ir_name": "beir/fever",
    },
}

DATASET_ALIASES = {
    "MSMARCO-Document": "msmarco",
    "BEIR/Fever": "fever",
    "1": "fever",
    "2": "msmarco",
}


def resolve_dataset_key(key: str) -> str:
    key = key.strip()
    if key in DATASETS:
        return key
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    raise ValueError(f"Unknown dataset: {key}")
