import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")




def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()




def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)




def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0


    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks




def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"]) if existing_items and "ids" in existing_items else set()
    print(f"Number of existing documents in DB: {len(existing_ids)}")


    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]
    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [c.metadata["id"] for c in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")




def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)




def ingest(reset: bool = False):
    if reset:
        print("✨ Clearing Database")
        clear_database()
    docs = load_documents()
    chunks = split_documents(docs)
    add_to_chroma(chunks)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    ingest(reset=args.reset)