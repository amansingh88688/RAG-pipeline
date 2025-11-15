"""
chunking.py
Uses LangChain RecursiveCharacterTextSplitter approximating token size via characters.
Creates chunk JSONL file.
"""

import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs, settings, chunks_file_path):
    """
    Convert page-level docs into list of chunks using LangChain text splitter.
    Save JSONL: one chunk per line {text, metadata}.
    """
    chunk_size = settings["chunk_size_tokens"] * 4       # approx chars per token
    chunk_overlap = settings["chunk_overlap_tokens"] * 4 # approx chars per token

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    for d in docs:
        chunks = splitter.split_text(d["text"])
        for c in chunks:
            meta = dict(d["metadata"])
            all_chunks.append({"text": c, "metadata": meta})

    # write JSONL
    os.makedirs(os.path.dirname(chunks_file_path), exist_ok=True)
    with open(chunks_file_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    return all_chunks
