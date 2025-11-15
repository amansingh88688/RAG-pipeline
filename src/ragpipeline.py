"""
ragpipeline.py
High-level orchestration:
- ingestion (PDF/TXT)
- chunking
- embedding + vectorstore
- updating manifest

This version imports the sanitizer from embedding.py and writes the sanitized
collection name into the manifest so manifest stays consistent.
"""

import os
from src.ingestion import extract_text_from_pdf, extract_text_from_txt, init_ocr_model
from src.chunking import chunk_documents
from src.embedding import get_embeddings, _sanitize_collection_name
from src.embedding import index_chunks_into_chroma
from src.utils import timestamp


def process_file(filename, sha, settings, manifest):
    """Process a single file → update manifest entry."""
    raw_path = os.path.join(settings["raw_folder"], filename)
    chunks_folder = settings["chunks_folder"]

    # OCR model init
    ocr_processor, ocr_model = init_ocr_model(settings["ocr_model_name"])

    # Extract text
    if filename.lower().endswith(".pdf"):
        docs, low_conf = extract_text_from_pdf(raw_path, ocr_processor, ocr_model)
    else:
        docs, low_conf = extract_text_from_txt(raw_path)

    # Chunking
    chunks_file = os.path.join(
        chunks_folder,
        f"{os.path.splitext(filename)[0]}.jsonl"
    )
    chunks = chunk_documents(docs, settings, chunks_file)

    # Embedding
    embed_model = get_embeddings(settings["embedding_model_name"])
    raw_collection_name = f"col_{os.path.splitext(filename)[0]}"
    collection_name = _sanitize_collection_name(raw_collection_name)

    index_chunks_into_chroma(
        chunks=chunks,
        chroma_path=settings["vector_db_path"],
        collection_name=collection_name,
        embed_model=embed_model,
    )

    # Update manifest with sanitized collection name
    entry = {
        "filename": filename,
        "raw_path": raw_path,
        "chunks_file": chunks_file,
        "chroma_collection": collection_name,
        "sha1": sha,
        "page_count": len(docs),
        "ocr_low_confidence_pages": low_conf,
        "upload_timestamp": manifest.get(filename, {}).get("upload_timestamp", timestamp()),
        "last_processed": timestamp(),
    }

    manifest[filename] = entry
    return manifest
