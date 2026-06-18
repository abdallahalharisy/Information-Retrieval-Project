# main.py
import ir_datasets
import json
import os
import sys
from preprocess import clean_and_preprocess

def load_dataset(dataset_name, limit=210000):
    """
    Load and preprocess a dataset from ir_datasets
    
    Args:
        dataset_name: name of the dataset (e.g., 'beir/fever', 'msmarco-document/dev')
        limit: maximum number of documents to process
    """
    print(f"Connecting to dataset {dataset_name}...")
    dataset = ir_datasets.load(dataset_name)
    
    processed_corpus = {}
    print(f"Starting preprocessing for the first {limit} documents...")
    
    # Main loop to read and clean data
    for i, doc in enumerate(dataset.docs_iter()):
        if i >= limit:
            break
            
        doc_id = doc.doc_id
        
        # Try different ways to get document text
        raw_text = None
        
        # Try default_text() method first
        if hasattr(doc, "default_text"):
            raw_text = getattr(doc, "default_text", None)
            if callable(raw_text):
                raw_text = raw_text()
        
        # Fallback to body attribute
        if not raw_text:
            raw_text = getattr(doc, "body", "")
        
        # Fallback to text attribute
        if not raw_text:
            raw_text = getattr(doc, "text", "")
        
        # Apply the preprocessing function
        if raw_text:
            processed_corpus[doc_id] = clean_and_preprocess(raw_text)
            
            # Print progress every 500 documents
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1} documents...")
    
    return processed_corpus

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <dataset> [limit]")
        print("\nAvailable datasets:")
        print("  1 or fever     -> BEIR/Fever (5.4M docs)")
        print("  2 or msmarco   -> MSMARCO-Document (3.2M docs)")
        print("\nExamples:")
        print("  python main.py fever 200000")
        print("  python main.py msmarco 300000")
        sys.exit(1)
    
    dataset_choice = sys.argv[1].lower()
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    
    # Map choices to actual dataset names
    datasets = {
        "1": ("beir/fever", "fever"),
        "2": ("msmarco-document/dev", "msmarco"),
        "fever": ("beir/fever", "fever"),
        "msmarco": ("msmarco-document/dev", "msmarco"),
    }
    
    if dataset_choice not in datasets:
        print(f"Unknown dataset: {dataset_choice}")
        print("Use: fever, msmarco, 1, or 2")
        sys.exit(1)
    
    dataset_name, output_prefix = datasets[dataset_choice]
    
    # Load and process the dataset
    processed_corpus = load_dataset(dataset_name, limit)
    
    # Save processed data to a local file
    output_filename = f"processed_data_{output_prefix}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(processed_corpus, f, ensure_ascii=False, indent=4)
    
    print("\n==============================================")
    print(f"Processing complete!")
    print(f"Dataset: {dataset_name}")
    print(f"Documents processed: {len(processed_corpus)}")
    print(f"Output file: {output_filename}")
    print("==============================================")

if __name__ == "__main__":
    main()