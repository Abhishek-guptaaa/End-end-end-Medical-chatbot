from flask import Flask, request, jsonify, render_template
from modules.embedding_generator import download_hugging_face_embeddings
from modules.pinecone_handler import initialize_pinecone
from modules.groq_handler import query_pinecone, generate

app = Flask(__name__)

# Initialize embeddings and Pinecone
embeddings = download_hugging_face_embeddings()
index = initialize_pinecone()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data['query']
    
    docs = query_pinecone(user_query, top_k=2, index=index, embeddings=embeddings)
    response = generate(query=user_query, docs=docs)
    
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)


