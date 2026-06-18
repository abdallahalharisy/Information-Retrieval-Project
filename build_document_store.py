"""Build SQLite store for original MSMARCO documents.

This is used at the final display step: ranking returns doc_ids, then the
original unprocessed text is fetched from SQLite by doc_id.

Usage:
    python build_document_store.py 210000
"""

import sys

import ir_datasets

from main import _extract_document_text
from shared.datasets import DATASETS
from shared.document_store import count_documents, init_document_store, upsert_documents


def build_msmarco_document_store(limit: int = 210000, batch_size: int = 1000) -> int:
    cfg = DATASETS["msmarco"]
    db_path = cfg["document_db"]
    dataset = ir_datasets.load(cfg["ir_name"])

    init_document_store(db_path)
    batch = []
    total = 0
    for i, doc in enumerate(dataset.docs_iter()):
        if i >= limit:
            break

        raw_text = _extract_document_text(doc)
        if not raw_text:
            continue

        batch.append((doc.doc_id, raw_text))
        if len(batch) >= batch_size:
            upsert_documents(batch, db_path)
            total += len(batch)
            batch = []
            print(f"Stored {total} original documents...")

    if batch:
        upsert_documents(batch, db_path)
        total += len(batch)

    stored_count = count_documents(db_path)
    print("==============================================")
    print("Document store build complete")
    print(f"SQLite DB: {db_path}")
    print(f"Stored documents in this run: {total}")
    print(f"Total documents in DB: {stored_count}")
    print("==============================================")
    return stored_count


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 210000
    build_msmarco_document_store(limit=limit)


if __name__ == "__main__":
    main()
