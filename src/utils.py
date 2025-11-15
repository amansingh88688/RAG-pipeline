"""
utils.py
General helper functions: hashing, file listing, YAML loading, manifest I/O.
"""

import os
import json
import hashlib
import yaml
from datetime import datetime


def load_settings(path: str):
    """Load YAML settings."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha1_of_file(path: str) -> str:
    """Return SHA1 hash of a file."""
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def load_manifest(path: str):
    """Load JSON manifest; create empty if missing."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: str, data: dict):
    """Write manifest JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def timestamp():
    return datetime.utcnow().isoformat()
