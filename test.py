from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from tqdm import tqdm
from dotenv import load_dotenv
import os
from groq import Groq


load_dotenv()

pinecone_api_key=os.getenv("PINECONE_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

# Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

extracted_data = load_pdf("D:/Medical_chatbot/data/")
print("Documents extracted:", len(extracted_data))


# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = text_split(extracted_data)
print("Length of my chunks:", len(text_chunks))

# Download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()


# Initialize Pinecone
pc = pinecone.Pinecone(api_key=pinecone_api_key)

index_name = "medical"

# Create the index if it doesn't exist
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


    # Connect to the index
index = pc.Index(index_name)

# Generate embeddings for text chunks and prepare for upsert
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

embedded_texts = generate_embeddings(text_chunks, embeddings)


# Upsert embeddings to Pinecone index
batch_size = 128  # Define your batch size

for i in tqdm(range(0, len(embedded_texts), batch_size), desc="Upserting embeddings"):
    batch = embedded_texts[i:i + batch_size]
    to_upsert = list(zip([item["id"] for item in batch], 
                         [item["values"] for item in batch], 
                         [item["metadata"] for item in batch]))
    index.upsert(vectors=to_upsert)

print("Upserted text chunks into Pinecone index successfully.")

from groq import Groq
import getpass
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize Groq client


import getpass
groq_api_key = getpass.getpass("Enter your Groq API key: ")
groq_client = Groq(api_key=groq_api_key)


def query_pinecone(query: str, top_k: int, index, embeddings):
    # Generate the embedding for the question
    query_vector = embeddings.embed_query(query)
    
    # Perform the similarity search
    query_results = index.query(vector=query_vector, top_k=top_k, include_values=True, include_metadata=True)
    
    # Extract and print the text content of the top matches
    docs = []
    for match in query_results['matches']:
        print(f"Score: {match['score']}")
        text_content = match['metadata'].get('text', 'Text not found')
        print(f"Text Content: {text_content}\n")
        docs.append(text_content)
    return docs



def generate(query: str, docs: list[str]):
    # Construct the system message
    system_message = (
        "You are a helpful assistant that answers questions about AI using the context provided below.\n\n"
        "CONTEXT:\n" + "\n\n---\n".join(docs) + "\n\n"
    )
    
    # Prepare the messages for the chat
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    
    # Generate and return the response
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    return chat_response.choices[0].message.content


# Example usage
query = "What is COMMON LLM ELIGIBILITY CRITERIA"
docs = query_pinecone(query, top_k=3, index=index, embeddings=embeddings)

out = generate(query=query, docs=docs)
print(out)


