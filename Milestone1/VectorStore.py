import os
import shutil
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from DirectoryLoader import load_documents  # your previous file


# -----------------------------
# STEP 15: CHUNKING STRATEGY
# -----------------------------
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


# -----------------------------
# STEP 16: METADATA CLEANING
# -----------------------------
def clean_metadata(documents):
    cleaned_docs = []

    for doc in documents:
        # Keep only simple metadata types (str, int, float)
        clean_meta = {
            k: v
            for k, v in doc.metadata.items()
            if isinstance(v, (str, int, float))
        }

        doc.metadata = clean_meta
        cleaned_docs.append(doc)

    print("Metadata cleaned successfully")

    return cleaned_docs


# -----------------------------
# STEP 19 & 20: VECTOR STORE
# -----------------------------
def build_vector_store():
    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Cleaning metadata...")
    cleaned_chunks = clean_metadata(chunks)

    # -----------------------------
    # STEP 17: MODEL SELECTION
    # -----------------------------
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # -----------------------------
    # STEP 18: EMBEDDING INIT
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )

    # -----------------------------
    # STARTUP CLEANUP (Step 11)
    # -----------------------------
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        print("Old chroma_db deleted")

    # -----------------------------
    # STEP 20: UNIQUE COLLECTION NAME
    # -----------------------------
    collection_name = f"collection_{uuid.uuid4().hex}"

    print("Creating vector store...")

    vector_store = Chroma.from_documents(
        documents=cleaned_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name=collection_name
    )

    vector_store.persist()

    print("Vector store built successfully!")
    print(f"Collection Name: {collection_name}")


if __name__ == "__main__":
    build_vector_store()
