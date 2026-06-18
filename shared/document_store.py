"""SQLite document store for original, unprocessed document text."""

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterable, Tuple

DEFAULT_DB_PATH = os.path.join("data", "documents_msmarco.db")


@contextmanager
def connect(db_path: str = DEFAULT_DB_PATH):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_document_store(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create the documents table if it does not exist."""
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                raw_text TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id)")


def upsert_documents(documents: Iterable[Tuple[str, str]], db_path: str = DEFAULT_DB_PATH) -> int:
    """Insert or replace original documents in batches."""
    init_document_store(db_path)
    count = 0
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO documents (doc_id, raw_text) VALUES (?, ?)",
            documents,
        )
        count = conn.execute("SELECT changes()").fetchone()[0]
    return count


def get_document(doc_id: str, db_path: str = DEFAULT_DB_PATH) -> str:
    """Return original text for a document id, or an empty string if missing."""
    if not os.path.exists(db_path):
        return ""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT raw_text FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
    return row[0] if row else ""


def count_documents(db_path: str = DEFAULT_DB_PATH) -> int:
    if not os.path.exists(db_path):
        return 0
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
