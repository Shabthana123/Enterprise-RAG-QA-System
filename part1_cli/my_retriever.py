# my_retrriever.py

import os
import json
import time
import faiss
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

# Global Configurations
MODEL_NAME = "all-MiniLM-L6-v2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "data/index")

# Load SentenceTransformer model once
print("Loading SentenceTransformer model into memory...")
model = SentenceTransformer(MODEL_NAME)
model.to("cpu")

# Load FAISS index + metadata ONCE at module level
_index = None
_metadatas = None
_chunks = None
_index_load_time = 0.0

def _load_index_once():
    global _index, _metadatas, _chunks, _index_load_time
    if _index is not None:
        return  # already loaded

    index_path = os.path.join(INDEX_DIR, "index.faiss")
    meta_path = os.path.join(INDEX_DIR, "meta.json")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"FAISS index or metadata not found in {INDEX_DIR}. "
            "Please run 'python ingest.py --url https://www.transfi.com' first."
        )

    load_start = time.perf_counter()
    _index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    load_end = time.perf_counter()

    _metadatas = meta["metadatas"]
    _chunks = meta["chunks"]
    _index_load_time = round(load_end - load_start, 3)

async def async_encode_texts(texts, batch_size=32):
    """
    Encode texts asynchronously in parallel batches for better performance.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        # Split texts into smaller batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        # Encode each batch concurrently
        tasks = [
            loop.run_in_executor(
                executor,
                lambda b=b: model.encode(b, convert_to_numpy=True).astype("float32")
            )
            for b in batches
        ]

        # Wait for all batches to finish
        results = await asyncio.gather(*tasks)

    # Combine all encoded batches into one numpy array
    return np.vstack(results)

# Async Retrieve Documents (uses cached index)
async def retrieve_documents(query, top_k=5):
    total_start = time.perf_counter()

    # Ensure index is loaded (safe for concurrent calls)
    _load_index_once()

    # Encode query
    embedding_start = time.perf_counter()
    q_emb = await async_encode_texts([query])
    embedding_end = time.perf_counter()

    faiss.normalize_L2(q_emb)

    # FAISS search (blocking, run in thread)
    loop = asyncio.get_event_loop()
    search_start = time.perf_counter()
    scores, ids = await loop.run_in_executor(None, lambda: _index.search(q_emb, top_k))
    search_end = time.perf_counter()

    # Build results
    results = []
    for idx, score in zip(ids[0], scores[0]):
        if 0 <= idx < len(_chunks):
            snippet = _chunks[idx][:400].replace("\n", " ").strip()
            if len(_chunks[idx]) > 400:
                snippet += "..."
            results.append({
                "title": _metadatas[idx].get("title", "Untitled"),
                "source": _metadatas[idx].get("url", "N/A"),
                "snippet": snippet,
                "score": float(score)
            })

    total_end = time.perf_counter()

    metrics = {
        "Index Load Time (s)": _index_load_time,
        "Embedding Time (s)": round(embedding_end - embedding_start, 3),
        "Search Time (s)": round(search_end - search_start, 3),
        "Total Retrieval Time (s)": round(total_end - total_start, 3),
        "Documents Retrieved": len(results)
    }

    return results, metrics
