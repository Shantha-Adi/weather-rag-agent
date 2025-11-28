import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from config import *


def ingest_documents_from_directory(directory_path: str):
    print(f"--- Starting Ingestion for directory: {directory_path} ---")

    # 1. Initialize Client & Create Collection (Done once)
    client = QdrantClient(path=QDRANT_PATH)
    
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    # 2. Find all PDF files
    # This looks for any file ending in .pdf inside the target directory
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    # 3. Setup Embeddings & Vector Store
    embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID, region_name=REGION_NAME)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # 4. Loop through each file
    total_chunks = 0
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\nProcessing: {filename}...")
        
        try:
            # Load
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"  - Loaded {len(documents)} pages.")

            # Split
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            print(f"  - Split into {len(chunks)} chunks.")

            # Ingest
            if chunks:
                vector_store.add_documents(documents=chunks)
                total_chunks += len(chunks)
                print(f"Successfully added to Qdrant.")
            else:
                print(f"No text found to ingest.")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print(f"\n--- Ingestion Complete! Total chunks stored: {total_chunks} ---")

if __name__ == "__main__":
    ingest_documents_from_directory(DEFAULT_RAG_DATASET_PATH)
    