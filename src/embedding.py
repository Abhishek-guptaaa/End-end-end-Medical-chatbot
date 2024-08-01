# embedding.py

from langchain.embeddings import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME

def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return embeddings
