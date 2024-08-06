from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

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
