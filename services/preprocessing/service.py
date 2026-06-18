"""Preprocessing service business logic."""

from typing import Dict, List

from preprocess import clean_and_preprocess, preprocess_text


def preprocess_document(text: str) -> List[str]:
    return clean_and_preprocess(text)


def preprocess_batch(documents: Dict[str, str]) -> Dict[str, List[str]]:
    return {doc_id: clean_and_preprocess(text) for doc_id, text in documents.items() if text}


def preprocess_raw(text: str, **kwargs) -> List[str]:
    return preprocess_text(text, **kwargs)
