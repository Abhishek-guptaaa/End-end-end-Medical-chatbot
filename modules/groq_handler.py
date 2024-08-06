from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

def query_pinecone(query: str, top_k: int, index, embeddings):
    query_vector = embeddings.embed_query(query)
    query_results = index.query(vector=query_vector, top_k=top_k, include_values=True, include_metadata=True)
    
    docs = []
    for match in query_results['matches']:
        text_content = match['metadata'].get('text', 'Text not found')
        docs.append(text_content)
    return docs

def generate(query: str, docs: list[str]):
    system_message = (
        "You are a helpful assistant that answers questions about AI using the context provided below.\n\n"
        "CONTEXT:\n" + "\n\n---\n".join(docs) + "\n\n"
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    return chat_response.choices[0].message.content
