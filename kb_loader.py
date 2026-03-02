"""
kb_loader.py
------------
Loads and chunks all knowledge-base documents from the kb/ folder.

Chunking strategy:
  - Split on whitespace tokens (words).
  - Each chunk: 400 tokens with 50-token overlap for context continuity.
  - Each chunk carries source file metadata for traceability.
"""

from __future__ import annotations

import os
from typing import List, Dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 400    # tokens (words) per chunk
DEFAULT_OVERLAP = 50        # overlapping tokens between consecutive chunks


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_documents(kb_dir: str) -> List[Dict]:
    """
    Read every .txt file in the knowledge base directory.

    Args:
        kb_dir: Path to the kb/ folder.

    Returns:
        List of dicts: {'source': filename, 'content': full text}.
    """
    if not os.path.isdir(kb_dir):
        raise FileNotFoundError(f"KB directory not found: {kb_dir}")

    documents: List[Dict] = []
    txt_files = [f for f in os.listdir(kb_dir) if f.endswith(".txt")]

    if not txt_files:
        raise ValueError(f"No .txt files found in {kb_dir}")

    for filename in sorted(txt_files):
        filepath = os.path.join(kb_dir, filename)
        with open(filepath, "r", encoding="utf-8") as fh:
            content = fh.read().strip()

        if content:
            documents.append({"source": filename, "content": content})
            print(f"[KBLoader] Loaded '{filename}' ({len(content)} chars)")

    print(f"[KBLoader] Total documents loaded: {len(documents)}")
    return documents


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict]:
    """
    Split documents into overlapping token-level chunks.

    Args:
        documents:  Output of load_documents().
        chunk_size: Number of words per chunk.
        overlap:    Number of words to overlap between adjacent chunks.

    Returns:
        List of chunk dicts:
            {
              'chunk_id':   int,
              'source':     str (filename),
              'chunk_index': int (0-based within document),
              'text':       str (chunk content),
            }
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: List[Dict] = []
    chunk_id = 0
    step = chunk_size - overlap

    for doc in documents:
        words = doc["content"].split()
        total_words = len(words)

        if total_words == 0:
            continue

        start = 0
        chunk_index = 0

        while start < total_words:
            end = min(start + chunk_size, total_words)
            chunk_text = " ".join(words[start:end])

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": doc["source"],
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                }
            )

            chunk_id += 1
            chunk_index += 1
            start += step

    print(f"[KBLoader] Total chunks created: {len(chunks)}")
    return chunks
