# ingest.py
import argparse
import os
import shutil
import time
import hashlib
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
# Disable chroma telemetry (avoid noisy errors)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Windows-safe PDF loader import
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma

import requests
from bs4 import BeautifulSoup

from get_embedding_function import get_embedding_function

BASE_DIR = os.path.dirname(__file__)
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")
DATA_PATH = os.path.join(BASE_DIR, "data")
WEBSITES_FILE = os.path.join(BASE_DIR, "websites.txt")

# --- Scraping helpers ---
HEADERS = {
    "User-Agent": "RAGChatBot/1.0 (+mailto:you@example.com)",
}

def polite_get(url, timeout=15):
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp

def scrape_website(url, delay=1.0):
    """Return (clean_text, title). Basic requests + BeautifulSoup scraper."""
    try:
        resp = polite_get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts/styles and common noisy tags
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
            tag.decompose()
        # Try to get a human title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else urlparse(url).netloc
        # Extract visible text
        text = soup.get_text(separator="\n", strip=True)
        time.sleep(delay)  # be polite
        return text, title
    except Exception as e:
        print(f"[scrape_website] failed for {url}: {e}")
        return "", url

def read_websites(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return lines

# --- Text chunking helpers ---
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def sha256_text(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def calculate_chunk_ids(chunks):
    """
    Create stable IDs and chunk metadata.
    For PDFs, expects chunk.metadata['source'] includes path and chunk.metadata['page'] exists.
    For web docs, source=URL, page=0.
    """

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", 0)
        # Normalize Windows path separators into forward slashes for IDs
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id

        # Add dedup hash
        chunk_hash = sha256_text(chunk.page_content)
        chunk.metadata["sha256"] = chunk_hash

        last_page_id = current_page_id

    return chunks

# --- Main add/update logic ---
def add_to_chroma(chunks):
    embedding_fn = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_fn)

    # Get existing metadata to identify already-ingested items
    existing_items = {}
    try:
        existing_items = db.get(include=["metadatas", "ids"])
    except Exception:
        try:
            existing_items = db.get(include=[])
        except Exception:
            existing_items = {}

    existing_hashes = set()
    if existing_items:
        metadatas = existing_items.get("metadatas", []) or []
        for m in metadatas:
            if not m:
                continue
            h = m.get("sha256")
            if h:
                existing_hashes.add(h)

    print(f"Number of existing documents in DB (by hash): {len(existing_hashes)}")

    # Only add non-duplicate chunks
    new_chunks = [c for c in chunks if c.metadata.get("sha256") not in existing_hashes]

    if not new_chunks:
        print("✅ No new documents to add")
        return

    print(f"👉 Adding new documents: {len(new_chunks)}")
    new_ids = [c.metadata["id"] for c in new_chunks]
    db.add_documents(new_chunks, ids=new_ids)
    # No db.persist() - langchain_chroma auto-persists when using persist_directory

def load_pdf_documents():
    if not os.path.exists(DATA_PATH):
        return []
    loader = PyPDFDirectoryLoader(DATA_PATH)
    try:
        docs = loader.load()
        # Ensure metadata has 'source' and 'page' for each chunk produced by the loader
        # (PyPDFDirectoryLoader usually adds this)
        return docs
    except Exception as e:
        print(f"[load_pdf_documents] loader failed: {e}")
        return []

def build_web_documents(websites):
    docs = []
    for url in websites:
        text, title = scrape_website(url)
        if not text:
            continue
        # Create single Document for the page; splitter will chunk it
        doc = Document(page_content=text, metadata={"source": url, "source_type": "web", "title": title, "page": 0})
        docs.append(doc)
    return docs

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Chroma directory removed.")

def ingest(reset: bool = False, websites_file: str = WEBSITES_FILE):
    if reset:
        print("✨ Clearing Database")
        clear_database()

    # Load PDFs
    pdf_docs = load_pdf_documents()
    # Load websites if provided
    websites = read_websites(websites_file)
    web_docs = build_web_documents(websites) if websites else []

    all_docs = pdf_docs + web_docs
    if not all_docs:
        print("No documents found (PDFs or websites). Nothing to do.")
        return

    chunks = split_documents(all_docs)
    chunks = calculate_chunk_ids(chunks)
    add_to_chroma(chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--websites-file", type=str, default=WEBSITES_FILE, help="Path to websites.txt")
    args = parser.parse_args()
    ingest(reset=args.reset, websites_file=args.websites_file)  
