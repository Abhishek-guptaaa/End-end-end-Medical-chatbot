from src.data_loader import load_pdf, text_split
from src.embedding import load_embeddings
from src.indexer import generate_embeddings, upsert_embeddings, query_index
from src.response_generator import generate_response

def main():
    # Load and process data
    extracted_data = load_pdf("D:/Medical_chatbot/data/")
    text_chunks = text_split(extracted_data)
    
    # Initialize embeddings
    embeddings = load_embeddings()
    
    # Generate and upsert embeddings
    embedded_texts = generate_embeddings(text_chunks, embeddings)
    upsert_embeddings(embedded_texts)
    
    # Perform a query
    question = "What causes asthma?"
    query_results = query_index(question, embeddings)
    
    # Map chunk IDs to text content
    id_to_text = {doc['id']: doc['metadata']['text'] for doc in embedded_texts}
    
    # Extract and print results
    for match in query_results['matches']:
        chunk_id = match['id']
        score = match['score']
        text_content = id_to_text.get(chunk_id, "Text not found")
        print(f"Chunk ID: {chunk_id}")
        print(f"Score: {score}")
        print(f"Text Content: {text_content}\n")
    
    # Generate final response
    context = "\n".join([id_to_text[match['id']] for match in query_results['matches']])
    response = generate_response(question, context)
    print("Final Response:", response)

if __name__ == "__main__":
    main()
