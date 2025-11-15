"""
management.py
Handles sync logic:
- Detect added / removed / replaced files.
- Enforce max document count.
"""

import os
import shutil
from src.utils import sha1_of_file, timestamp


def sync_files(settings, manifest):
    """
    Compare data/raw vs manifest to determine:
    - new files
    - replaced files (same name, different hash)
    - removed files
    Returns dict: { "new": [], "replaced": [], "removed": [] }
    """
    raw_folder = settings["raw_folder"]
    max_docs = settings["max_documents"]

    existing_files = set(manifest.keys())
    raw_files = {
        f for f in os.listdir(raw_folder)
        if f.lower().endswith((".pdf", ".txt"))
    }

    # Detect removed
    removed = list(existing_files - raw_files)

    new = []
    replaced = []

    for f in raw_files:
        raw_path = os.path.join(raw_folder, f)
        sha = sha1_of_file(raw_path)

        if f not in manifest:
            new.append((f, sha))
        else:
            if manifest[f]["sha1"] != sha:
                replaced.append((f, sha))

    # Enforce max document count → delete oldest uploads
    current_total = len(raw_files)
    if current_total > max_docs:
        overflow = current_total - max_docs
        # Sort manifest by upload time
        sorted_by_time = sorted(
            manifest.items(),
            key=lambda x: x[1].get("upload_timestamp", "")
        )
        to_delete = [name for name, _ in sorted_by_time[:overflow]]

        for f in to_delete:
            removed.append(f)
            # also delete file physically
            try:
                os.remove(os.path.join(raw_folder, f))
            except:
                pass

    return {
        "new": new,
        "replaced": replaced,
        "removed": removed
    }


def delete_file_metadata(f, manifest, settings):
    """
    Remove manifest entry + chunk file + chroma collection directory.
    """
    if f not in manifest:
        return manifest

    entry = manifest[f]

    # delete chunks JSONL
    if "chunks_file" in entry and os.path.exists(entry["chunks_file"]):
        try:
            os.remove(entry["chunks_file"])
        except:
            pass

    # delete chroma collection folder
    if "chroma_collection" in entry:
        chroma_base = settings["vector_db_path"]
        col_path = os.path.join(chroma_base, entry["chroma_collection"])
        if os.path.exists(col_path):
            shutil.rmtree(col_path, ignore_errors=True)

    # remove from manifest
    manifest.pop(f, None)
    return manifest
