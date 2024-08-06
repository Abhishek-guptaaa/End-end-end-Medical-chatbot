from modules.data_loader import load_pdf, text_split
from modules.embedding_generator import download_hugging_face_embeddings, generate_embeddings
from modules.pinecone_handler import initialize_pinecone, upsert_embeddings
import os

# Load PDF and extract data
extracted_data = load_pdf("D:/Medical_chatbot/data/")
print("Documents extracted:", len(extracted_data))

# Split text into chunks
text_chunks = text_split(extracted_data)
print("Length of my chunks:", len(text_chunks))

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
index = initialize_pinecone()

# Generate embeddings for text chunks
embedded_texts = generate_embeddings(text_chunks, embeddings)

# Upsert embeddings to Pinecone index
upsert_embeddings(embedded_texts, index)
print("Upserted text chunks into Pinecone index successfully.")
