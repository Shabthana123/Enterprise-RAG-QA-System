# ingest.py
import os
import time
import json
import argparse
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE_DIR, "data")

RAW_DIR = os.path.join(OUTDIR, "raw_html")
TEXT_DIR = os.path.join(OUTDIR, "clean_text")
INDEX_DIR = os.path.join(OUTDIR, "index")

errors = []

# chunking params
CHUNK_WORDS = 200
CHUNK_OVERLAP = 50
EMBED_BATCH = 64  # embed in batches

# model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# helper
def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

async def fetch(session, url):
    try:
        async with session.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=20) as resp:
            if resp.status != 200:
                msg = f"Failed to fetch {url} (status {resp.status})"
                print(msg)
                errors.append(msg)
                return None
            return await resp.text()
    except Exception as e:
        msg = f"Error fetching {url}: {type(e).__name__} - {e}"
        print(msg)
        errors.append(msg)
        return None

async def save_file(path, content):
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(content)

def extract_page_info(html, url):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else "No Title"
    
    # Remove unwanted tags
    for tag in soup(["script","style","noscript","header","footer","nav","svg", "aside"]):
        tag.decompose()
        
    # Try to find the main content container
    main_content = soup.find("main") or soup.find("article") or soup.find("section")
    if not main_content:
        main_content = soup  # fallback to entire body
        
    # meta description
    meta_desc = soup.find("meta", attrs={"name":"description"})
    if meta_desc and meta_desc.get("content"):
        short = meta_desc["content"].strip()
    else:
        candidates = soup.select("main p, section p")
        if not candidates:
            candidates = soup.find_all("p")
        short = candidates[0].get_text(strip=True)[:300] if candidates else "N/A"
        
    # Extract long description from prioritized main content
    long = " ".join(main_content.get_text(separator=" ").split())
    # long = " ".join(soup.get_text(separator=" ").split())
    return {
            "title": title, 
            "url": url, 
            "short_description": short, 
            "long_description": long
            }

def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_words]
        chunks.append(" ".join(chunk))
        i += chunk_words - overlap
    return chunks

async def scrape_and_save(session, idx, label, url):
    html = await fetch(session, url)
    if not html:
        # errors.append(url)
        return None
    
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "root"
    fname = f"page_{idx}_{path}"

    # save raw html
    html_path = os.path.join(RAW_DIR, f"{fname}.html")
    await save_file(html_path, html)

    info = extract_page_info(html, url)
    content = (
        f"Title: {info['title']}\nURL: {info['url']}\n\n"
        f"Short Description:\n{info['short_description']}\n\n"
        f"Long Description:\n{info['long_description']}\n"
    )
    text_path = os.path.join(TEXT_DIR, f"{fname}.txt")
    await save_file(text_path, content)
    return info

# CORE LOGIC — returns structured metrics
async def run_ingest(base_url: str):
    global errors
    errors = []
    start_time = time.time()
    ensure_dirs()

    async with aiohttp.ClientSession() as session:
        home_html = await fetch(session, base_url)
        if not home_html:
            return {"error": "Failed to fetch base URL"}

        soup = BeautifulSoup(home_html, "html.parser")
        anchors = soup.find_all("a", href=True)
        keywords = ("product", "products", "solution", "solutions", "learn")
        matches = []
        for a in anchors:
            text = a.get_text(strip=True)
            href = a["href"]
            href_clean = href.split("#")[0]
            full = urljoin(base_url, href_clean)
            if any(k in (text or "").lower() for k in keywords) or any(k in (href_clean or "").lower() for k in keywords):
                matches.append((text, full))

        seen = set()
        unique = [(t, l) for t, l in matches if not (l in seen or seen.add(l))]

        scrape_start = time.time()
        tasks = [scrape_and_save(session, i, label, url) for i, (label, url) in enumerate(unique, start=1)]
        results = await asyncio.gather(*tasks)
        scrape_time = time.time() - scrape_start

    pages = [r for r in results if r]
    all_chunks = []
    metadata = []
    total_tokens = 0
    for p in pages:
        chunks = chunk_text(p["long_description"])
        for c in chunks:
            all_chunks.append(c)
            metadata.append({"title": p["title"], "url": p["url"]})
            total_tokens += len(c.split())

    embed_start = time.time()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = []
    for i in range(0, len(all_chunks), EMBED_BATCH):
        batch = all_chunks[i:i + EMBED_BATCH]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)
    embed_time = time.time() - embed_start

    index_start = time.time()
    if embeddings:
        X = np.vstack(embeddings).astype("float32")
        faiss.normalize_L2(X)
        dim = X.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(X)
        faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
        with open(os.path.join(INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"metadatas": metadata, "chunks": all_chunks}, f, ensure_ascii=False, indent=2)
    index_time = time.time() - index_start

    elapsed = time.time() - start_time
    total_chunks = len(all_chunks)
    total_pages = len(pages)
    pages_failed = len(errors)
    avg_scrape_time = scrape_time / max(total_pages, 1)

    return {
        "total_time_s": round(elapsed, 2),
        "pages_scraped": total_pages,
        "pages_failed": pages_failed,
        "total_chunks_created": total_chunks,
        "total_tokens_processed": total_tokens,
        "embedding_generation_time_s": round(embed_time, 2),
        "indexing_time_s": round(index_time, 2),
        "average_scraping_time_per_page_s": round(avg_scrape_time, 2),
        "errors": errors.copy()
    }

# CLI entry point
async def main(base_url):
    metrics = await run_ingest(base_url)
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return
    print("\n=== Ingestion Metrics ===")
    print(f"Total Time: {metrics['total_time_s']:.2f}s")
    print(f"Pages Scraped: {metrics['pages_scraped']}")
    print(f"Pages Failed: {metrics['pages_failed']}")
    print(f"Total Chunks Created: {metrics['total_chunks_created']}")
    print(f"Total Tokens Processed: {metrics['total_tokens_processed']:,}")
    print(f"Embedding Generation Time: {metrics['embedding_generation_time_s']:.2f}s")
    print(f"Indexing Time: {metrics['indexing_time_s']:.2f}s")
    print(f"Average Scraping Time per Page: {metrics['average_scraping_time_per_page_s']:.2f}s")
    print(f"Errors: {metrics['errors'] if metrics['errors'] else 'None'}")
    print("==========================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async TransFi website scraper and RAG index builder")
    parser.add_argument("--url", required=True, help="Main website URL (e.g. https://www.transfi.com)")
    args = parser.parse_args()
    asyncio.run(main(args.url.rstrip("/")))
