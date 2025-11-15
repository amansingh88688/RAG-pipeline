"""
retrieval.py
Creates retriever from multiple Chroma collections and calls Groq LLM.

This version:
- sanitizes collection names when loading (keeps compatibility)
- performs retrieval per-collection
- guarantees at least one hit per collection (if available) to increase source diversity
- then fills remaining slots with the best remaining hits
"""

import os
import requests
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.embedding import _sanitize_collection_name


def load_all_vectorstores(manifest, settings, embed_model):
    """
    Load each file's collection and combine into a list of vectorstores.
    Sanitizes the collection name if necessary.
    """
    chroma_base = settings["vector_db_path"]
    stores = []

    for f, entry in manifest.items():
        raw_collection = entry.get("chroma_collection") or f"col_{os.path.splitext(f)[0]}"
        safe_collection = _sanitize_collection_name(raw_collection)

        # Update in-memory manifest entry so downstream sees sanitized name
        entry["chroma_collection"] = safe_collection

        persist_dir = os.path.join(chroma_base, safe_collection)
        try:
            store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embed_model,
                collection_name=safe_collection,
            )
            stores.append(store)
        except Exception as e:
            print(f"Warning: could not load collection '{safe_collection}' for file '{f}': {e}")
            continue

    return stores


def _try_similarity_search_with_score(store, query, k):
    """
    Try to call similarity_search_with_score if available; else fallback to similarity_search.
    Returns list of tuples: (Document, score or None)
    """
    try:
        results = store.similarity_search_with_score(query, k=k)
        # results are list of (Document, score)
        return results
    except Exception:
        # fallback to similarity_search which returns Document list
        try:
            docs = store.similarity_search(query, k=k)
            return [(d, None) for d in docs]
        except Exception:
            return []


def combine_retrieval(stores, query, k):
    """
    Retrieve results from each store and combine them, with diversity:
    - Take top 1 from each store (if any)
    - Then take remaining best candidates across all stores until we have k documents
    Returns a list of langchain.schema.Document objects (max length k)
    """
    per_store_results = []
    all_candidates = []

    # Step 1: collect per-store results (with optional score)
    for store in stores:
        hits = _try_similarity_search_with_score(store, query, k)
        if not hits:
            continue
        # hits: list of (doc, score_or_none)
        per_store_results.append(hits)
        # also add to global candidate pool
        all_candidates.extend(hits)

    if not all_candidates:
        return []

    # Step 2: guarantee at least one per store (take the top one from each store)
    selected_docs = []
    seen = set()  # track (source_file, page, text snippet) to prevent duplicates

    for store_hits in per_store_results:
        top = store_hits[0]  # (doc, score)
        doc, score = top
        key = (doc.metadata.get("source_file"), doc.metadata.get("page"), (doc.page_content or "")[:60])
        if key not in seen:
            selected_docs.append((doc, score))
            seen.add(key)

    # If we already have enough, trim and return
    if len(selected_docs) >= k:
        return [d for (d, s) in selected_docs][:k]

    # Step 3: fill remaining slots from all_candidates, excluding already selected
    # We don't assume score polarity; just preserve the original order per-store as returned,
    # which is typically descending relevance. We'll prioritize items not yet selected.
    filled = list(selected_docs)
    for doc, score in all_candidates:
        if len(filled) >= k:
            break
        key = (doc.metadata.get("source_file"), doc.metadata.get("page"), (doc.page_content or "")[:60])
        if key in seen:
            continue
        filled.append((doc, score))
        seen.add(key)

    # Return Documents only
    return [d for (d, s) in filled][:k]


def call_groq_llm(api_key, model_name, prompt):
    """
    Minimal Groq API wrapper.
    Logs error body if any.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    r = requests.post(url, json=payload, headers=headers)
    if r.status_code != 200:
        print("Groq API Error:", r.status_code)
        print("Response body:", r.text)
        raise RuntimeError("Groq API error")

    data = r.json()
    return data["choices"][0]["message"]["content"]
