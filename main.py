"""
main.py
Simple, clean print statements (no logging, no timestamps).
"""

import os
from dotenv import load_dotenv

from src.utils import load_settings, load_manifest, save_manifest
from src.management import sync_files, delete_file_metadata
from src.ragpipeline import process_file
from src.embedding import get_embeddings
from src.retrieval import load_all_vectorstores, combine_retrieval, call_groq_llm

# -----------------------------
# USER QUERY (edit this)
query = "your question here"
# -----------------------------


def format_final_answer(llm_answer, docs):
    """Add citations with 50-char preview."""
    lines = []
    for d in docs:
        meta = d.metadata
        filename = meta.get("source_file", "unknown")
        page = meta.get("page", "?")
        preview = (d.page_content or "")[:50].replace("\n", " ").strip()
        lines.append(f"Source: {filename}, page {page} → \"{preview}...\"")
    return llm_answer + "\n\n---\nSources:\n" + "\n".join(lines)


def main():
    load_dotenv()

    print("Loading settings...")
    settings = load_settings("config/settings.yaml")

    print("Loading manifest...")
    manifest = load_manifest(settings["manifest_path"])

    print("Syncing files...")
    changes = sync_files(settings, manifest)
    print(f"  New: {len(changes['new'])}, Replaced: {len(changes['replaced'])}, Removed: {len(changes['removed'])}")

    # Removed
    for f in changes["removed"]:
        print(f"Removing metadata for {f}...")
        manifest = delete_file_metadata(f, manifest, settings)

    # Replaced
    for (f, sha) in changes["replaced"]:
        print(f"Reprocessing replaced file: {f}...")
        manifest = delete_file_metadata(f, manifest, settings)
        manifest = process_file(f, sha, settings, manifest)
        print(f"  Done: {f}")

    # New
    for (f, sha) in changes["new"]:
        print(f"Processing new file: {f}...")
        manifest = process_file(f, sha, settings, manifest)
        print(f"  Done: {f}")

    print("Saving manifest...")
    save_manifest(settings["manifest_path"], manifest)

    print("Preparing embeddings...")
    embed_model = get_embeddings(settings["embedding_model_name"])

    print("Loading vectorstores...")
    stores = load_all_vectorstores(manifest, settings, embed_model)

    if not stores:
        print("No documents to search. Exiting.")
        return

    print("Retrieving relevant chunks...")
    retrieved = combine_retrieval(stores, query, settings["k_retrieval"])
    # print(f"  Retrieved: {len(retrieved)} chunks")

    print("Generating Answer...")
    context = "\n\n".join(
        [f"[page {d.metadata.get('page')} from {d.metadata.get('source_file')}]\n{d.page_content}"
         for d in retrieved]
    )
    prompt = f"Use the context to answer. Add citations.\n\nContext:\n{context}\n\nQuestion: {query}"

    llm_answer = call_groq_llm(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=settings["llm_model_name"],
        prompt=prompt
    )

    print("\n=== FINAL ANSWER ===\n")
    print(format_final_answer(llm_answer, retrieved))


if __name__ == "__main__":
    main()
