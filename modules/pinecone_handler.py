import pinecone
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def initialize_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    pc = pinecone.Pinecone(api_key=api_key)
    index_name = "medical"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=pinecone.ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            )
        )

    return pc.Index(index_name)

def upsert_embeddings(embedded_texts, index):
    batch_size = 128

    for i in tqdm(range(0, len(embedded_texts), batch_size), desc="Upserting embeddings"):
        batch = embedded_texts[i:i + batch_size]
        to_upsert = list(zip([item["id"] for item in batch], 
                             [item["values"] for item in batch], 
                             [item["metadata"] for item in batch]))
        index.upsert(vectors=to_upsert)
