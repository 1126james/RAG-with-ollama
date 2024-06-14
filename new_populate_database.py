import os
import argparse
import shutil
import spacy
import jieba
import pdfplumber
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma

# Load spaCy model for English
nlp = spacy.load("en_core_web_sm")

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, filename)
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        metadata = {"source": filename, "page": i + 1}
                        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def split_documents(documents: list[Document], max_chunk_size: int = 800) -> list[Document]:
    chunks = []
    for document in documents:
        text = document.page_content
        metadata = document.metadata
        metadata['source'] = os.path.basename(metadata['source'])  # Ensure filename is stored
        chunks.extend(split_text_by_semantic_units(text, metadata, max_chunk_size))
    return chunks

def split_text_by_semantic_units(text: str, metadata: dict, max_chunk_size: int = 800) -> list[Document]:
    # Tokenize text using spaCy for English and jieba for Chinese
    doc = nlp(text)
    english_sentences = [sent.text.strip() for sent in doc.sents]
    chinese_sentences = list(jieba.cut(text, cut_all=False))

    chunks = []
    current_chunk = ""
    counter = 0  # Reset counter for each new document

    for sentence in english_sentences + chinese_sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            chunk_id = f"{metadata['source']}:{metadata['page']}:{counter}"
            chunks.append(Document(page_content=current_chunk, metadata={"id": chunk_id}))
            current_chunk = sentence
            counter += 1
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk:
        chunk_id = f"{metadata['source']}:{metadata['page']}:{counter}"
        chunks.append(Document(page_content=current_chunk, metadata={"id": chunk_id}))
    
    return chunks

def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    existing_ids = set(db.get(include=[])["ids"])
    new_ids = {chunk.metadata["id"] for chunk in chunks}

    if new_ids.issubset(existing_ids):
        print("✅ No new documents to add")
    else:
        ids_to_add = new_ids - existing_ids
        print(f"👉 Number of new documents to be added: {len(ids_to_add)}")
        for chunk in chunks:
            new_chunk_id = chunk.metadata["id"]
            if new_chunk_id not in existing_ids:
                print(f"Adding ID {new_chunk_id} to the database.")
                db.add_documents([chunk], ids=[new_chunk_id])
                existing_ids.add(new_chunk_id)
            else:
                print(f"Duplicate ID {new_chunk_id} detected, not adding to the database.")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()