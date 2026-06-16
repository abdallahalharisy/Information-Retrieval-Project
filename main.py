# main.py
import ir_datasets
import json
import os
from preprocess import clean_and_preprocess

# 1. Download the real dataset locally from the source
print("Connecting to dataset msmarco-document/dev...")
dataset = ir_datasets.load("msmarco-document/dev")

processed_corpus = {}
LIMIT = 2000  # We'll start with 2000 documents to ensure everything works, then increase as needed

print(f"Starting preprocessing for the first {LIMIT} documents...")

# 2. Main loop to read and clean data
for i, doc in enumerate(dataset.docs_iter()):
    if i >= LIMIT:
        break
        
    doc_id = doc.doc_id

    # MSMARCO documents expose the text through `default_text()` (or `body`)
    raw_text = getattr(doc, "default_text", None)
    if callable(raw_text):
        raw_text = raw_text()
    if not raw_text:
        raw_text = getattr(doc, "body", "")

    # Apply the preprocessing function implemented in preprocess.py
    processed_corpus[doc_id] = clean_and_preprocess(raw_text)
    
    # Print a simple progress message every 500 documents
    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1} documents...")

# 3. Save processed data to a local file (caching)
output_filename = "processed_data.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(processed_corpus, f, ensure_ascii=False, indent=4)

print("\n==============================================")
print(f"Processing complete! Processed data saved to: {output_filename}")
print("==============================================")