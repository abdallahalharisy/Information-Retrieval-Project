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


def load_dataset(dataset_name, limit=210000, document_db=None):
    """
    Load and preprocess a dataset from ir_datasets
    
    Args:
        dataset_name: name of the dataset (e.g., 'msmarco-document/dev')
        limit: maximum number of documents to process
    """
    print(f"Connecting to dataset {dataset_name}...")
    dataset = ir_datasets.load(dataset_name)
    
    processed_corpus = {}
    raw_batch = []
    batch_size = 1000
    if document_db:
        init_document_store(document_db)
        print(f"Original documents will be stored in SQLite: {document_db}")

    print(f"Starting preprocessing for the first {limit} documents...")
    
    # Main loop to read and clean data
    for i, doc in enumerate(dataset.docs_iter()):
        if i >= limit:
            break
            
        doc_id = doc.doc_id
        
        raw_text = _extract_document_text(doc)
        
        # Apply the preprocessing function
        if raw_text:
            processed_corpus[doc_id] = clean_and_preprocess(raw_text)
            if document_db:
                raw_batch.append((doc_id, raw_text))
                if len(raw_batch) >= batch_size:
                    upsert_documents(raw_batch, document_db)
                    raw_batch = []
            
            # Print progress every 500 documents
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1} documents...")

    if document_db and raw_batch:
        upsert_documents(raw_batch, document_db)
    
    return processed_corpus

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <dataset> [limit]")
        print("\nAvailable datasets:")
        print("  1 or msmarco   -> MSMARCO-Document (3.2M docs)")
        print("\nExamples:")
        print("  python main.py msmarco 210000")
        sys.exit(1)
    
    dataset_choice = sys.argv[1].lower()
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    
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
    
    # Load and process the dataset
    processed_corpus = load_dataset(dataset_name, limit, document_db=document_db)
    
    # Save processed data to a local file
    output_filename = f"processed_data_{output_prefix}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(processed_corpus, f, ensure_ascii=False, indent=4)
    
    print("\n==============================================")
    print(f"Processing complete!")
    print(f"Dataset: {dataset_name}")
    print(f"Documents processed: {len(processed_corpus)}")
    print(f"Output file: {output_filename}")
    print(f"Original documents DB: {document_db}")
    print("==============================================")

if __name__ == "__main__":
    main()