import os
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

def build_documents_from_csv(csv_path):
    pf = pd.read_csv(csv_path)
    docs = []
    for _, row in pf.iterrows():
        content = f"{row['header']} {row['text']}"
        metadata = {
            "name": row['name'],
            "header": row['header'],
            "chunk_id": row['chunk_id'],
        }
        doc = Document(page_content=content, metadata=metadata)
        docs.append(doc)
    return docs

def build_qdrant_index(docs, embedding_model, url, api_key, collection_name):
    print(f"Uploading {len(docs)} documents to Qdrant collection '{collection_name}'...")
    qdrant = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedding_model,
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        prefer_grpc=True  # often faster for cloud Qdrant
    )
    print(f"Qdrant upload complete.")
    return qdrant

def main():
    CSV_PATH = "all_chunks.csv"
    QDRANT_URL = "https://20840cd3-a3bf-4a62-af36-72b49fe3bed0.us-east-1-0.aws.cloud.qdrant.io"
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Ensure your API key is set in env
    EMBED_MODEL_DEPLOYMENT = "ace-text-embedding-3-large"
    QDRANT_COLLECTION = "law"
    
    docs = build_documents_from_csv(CSV_PATH)
    
    embeddings = AzureOpenAIEmbeddings(model=EMBED_MODEL_DEPLOYMENT)
    
    build_qdrant_index(
        docs,
        embedding_model=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION
    )

if __name__ == "__main__":
    main()
