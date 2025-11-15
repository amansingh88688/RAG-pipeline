"""
embedding.py
Uses LangChain HuggingFace embeddings wrapper with BGE model.
Stores vectors in local Chroma using LangChain's Chroma integration.

This version sanitizes Chroma collection names so they meet Chroma's requirements:
 - allowed chars: a-zA-Z0-9._-
 - length roughly 3-512 chars
 - must start and end with an alphanumeric character
"""

import os
import re
import uuid
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def get_embeddings(model_name):
    """Create embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


def _sanitize_collection_name(name: str) -> str:
    """
    Sanitize the collection name to match Chroma constraints:
      - only [a-zA-Z0-9._-]
      - replace other chars with underscore
      - collapse multiple underscores
      - ensure starts/ends with alnum (prefix/suffix 'c' if needed)
      - enforce min length 3
    """
    # Replace invalid chars with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    # Collapse multiple underscores/dots/dashes
    sanitized = re.sub(r"[_]{2,}", "_", sanitized)
    sanitized = re.sub(r"[.]{2,}", ".", sanitized)
    sanitized = re.sub(r"[-]{2,}", "-", sanitized)
    # Trim to reasonable length (max 200)
    sanitized = sanitized[:200]
    # Ensure starts and ends with alnum
    if not re.match(r"^[a-zA-Z0-9]", sanitized):
        sanitized = "c" + sanitized
    if not re.match(r".*[a-zA-Z0-9]$", sanitized):
        sanitized = sanitized + "c"
    # Ensure minimum length
    if len(sanitized) < 3:
        sanitized = (sanitized + "ccc")[:3]
    return sanitized


def index_chunks_into_chroma(chunks, chroma_path, collection_name, embed_model):
    """
    Index chunks → Chroma collection.
    Returns vectorstore object.
    """
    os.makedirs(chroma_path, exist_ok=True)

    # sanitize collection name to avoid chroma errors
    safe_name = _sanitize_collection_name(collection_name)

    texts = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    print(f"  • Indexing {len(texts)} chunks into collection '{safe_name}'")

    try:
        vectordb = Chroma(
            persist_directory=os.path.join(chroma_path, safe_name),
            embedding_function=embed_model,
            collection_name=safe_name,
        )
        # add_texts will persist under modern Chroma versions automatically
        ids = [str(uuid.uuid4()) for _ in texts]
        vectordb.add_texts(texts=texts, metadatas=metas, ids=ids)
        try:
            # older LangChain wrappers used explicit persist; safe to call if present
            vectordb.persist()
        except Exception:
            # ignore if persist not supported (Chroma auto-persist)
            pass

    except Exception as e:
        # Provide clearer error message
        raise RuntimeError(f"Failed to create or write to Chroma collection '{safe_name}': {e}") from e

    return vectordb
