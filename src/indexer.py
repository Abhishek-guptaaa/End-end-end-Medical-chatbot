
import os
import pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm
from src.config import INDEX_NAME

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def init_pinecone():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # Adjust according to your embeddings dimension
            metric="cosine",
            spec=pinecone.ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    return pc.Index(INDEX_NAME)

def generate_embeddings(text_chunks, embeddings):
    embedded_texts = []
    for i, chunk in enumerate(tqdm(text_chunks, desc="Generating embeddings")):
        vector = embeddings.embed_query(chunk.page_content)
        embedded_texts.append({
            "id": f"chunk_{i}",
            "values": vector,
            "metadata": {"text": chunk.page_content}
        })
    return embedded_texts

def upsert_embeddings(embedded_texts):
    index = init_pinecone()  # Initialize Pinecone index

    for i in tqdm(range(0, len(embedded_texts), 100), desc="Upserting embeddings"):
        batch = embedded_texts[i:i+100]
        index.upsert(vectors=batch)

    print("Upserted text chunks into Pinecone index successfully.")

def query_index(question, embeddings):
    index = init_pinecone()  # Initialize Pinecone index
    vector = embeddings.embed_query(question)
    
    query_results = index.query(
        vector=vector,
        top_k=3,
        include_values=True
    )

    return query_results
