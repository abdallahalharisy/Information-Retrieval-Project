# main.py
import ir_datasets
import json
import os
import sys
from preprocess import clean_and_preprocess
from shared.datasets import DATASETS
from shared.document_store import init_document_store, upsert_documents


def _extract_document_text(doc):
    """Extract original text from common ir_datasets document fields."""
    raw_text = None

    if hasattr(doc, "default_text"):
        raw_text = getattr(doc, "default_text", None)
        if callable(raw_text):
            raw_text = raw_text()

    if not raw_text:
        raw_text = getattr(doc, "body", "")

    if not raw_text:
        raw_text = getattr(doc, "text", "")

    return raw_text or ""


def _parse_limit(value):
    if value is None:
        return None
    parsed = int(value)
    return None if parsed <= 0 else parsed


def preprocess_dataset_to_json(dataset_name, output_filename, limit=None, document_db=None):
    """
    Load and preprocess a dataset from ir_datasets, streaming output to JSON.
    
    Args:
        dataset_name: name of the dataset (e.g., 'msmarco-document/dev')
        output_filename: JSON output path
        limit: maximum number of documents to process; None means all documents
    """
    print(f"Connecting to dataset {dataset_name}...")
    dataset = ir_datasets.load(dataset_name)
    
    raw_batch = []
    batch_size = 1000
    if document_db:
        init_document_store(document_db)
        print(f"Original documents will be stored in SQLite: {document_db}")

    scope = "all documents" if limit is None else f"the first {limit} documents"
    print(f"Starting preprocessing for {scope}...")
    
    processed_count = 0
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("{\n")
        first_item = True
        for i, doc in enumerate(dataset.docs_iter()):
            if limit is not None and i >= limit:
                break

            doc_id = doc.doc_id
            raw_text = _extract_document_text(doc)

            if raw_text:
                tokens = clean_and_preprocess(raw_text)
                if not first_item:
                    f.write(",\n")
                json.dump(doc_id, f, ensure_ascii=False)
                f.write(": ")
                json.dump(tokens, f, ensure_ascii=False)
                first_item = False
                processed_count += 1

                if document_db:
                    raw_batch.append((doc_id, raw_text))
                    if len(raw_batch) >= batch_size:
                        upsert_documents(raw_batch, document_db)
                        raw_batch = []

                if processed_count % 500 == 0:
                    print(f"Processed {processed_count} documents...")
        f.write("\n}\n")

    if document_db and raw_batch:
        upsert_documents(raw_batch, document_db)
    
    return processed_count

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <dataset> [limit]")
        print("\nAvailable datasets:")
        print("  1 or msmarco   -> MSMARCO-Document (3.2M docs)")
        print("\nExamples:")
        print("  python main.py msmarco       # all documents")
        print("  python main.py msmarco 0     # all documents")
        print("  python main.py msmarco 210000")
        sys.exit(1)
    
    dataset_choice = sys.argv[1].lower()
    limit = _parse_limit(sys.argv[2]) if len(sys.argv) > 2 else None
    
    datasets = {
        "1": ("msmarco-document/dev", "msmarco"),
        "2": ("msmarco-document/dev", "msmarco"),
        "msmarco": ("msmarco-document/dev", "msmarco"),
    }
    
    if dataset_choice not in datasets:
        print(f"Unknown dataset: {dataset_choice}")
        print("Use: msmarco, 1, or 2")
        sys.exit(1)
    
    dataset_name, output_prefix = datasets[dataset_choice]
    document_db = DATASETS[output_prefix]["document_db"]
    
    # Save processed data to a local file
    output_filename = f"processed_data_{output_prefix}.json"
    processed_count = preprocess_dataset_to_json(
        dataset_name,
        output_filename,
        limit=limit,
        document_db=document_db,
    )
    
    print("\n==============================================")
    print(f"Processing complete!")
    print(f"Dataset: {dataset_name}")
    print(f"Documents processed: {processed_count}")
    print(f"Output file: {output_filename}")
    print(f"Original documents DB: {document_db}")
    print("==============================================")

if __name__ == "__main__":
    main()