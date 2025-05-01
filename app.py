# from flask import Flask, request, jsonify, render_template
# from modules.embedding_generator import download_hugging_face_embeddings
# from modules.pinecone_handler import initialize_pinecone
# from modules.groq_handler import query_pinecone, generate

# app = Flask(__name__)

# # Initialize embeddings and Pinecone
# embeddings = download_hugging_face_embeddings()
# index = initialize_pinecone()

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/query', methods=['POST'])
# def query():
#     data = request.json
#     user_query = data['query']
    
#     docs = query_pinecone(user_query, top_k=2, index=index, embeddings=embeddings)
#     response = generate(query=user_query, docs=docs)
    
#     return jsonify({'response': response})

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=5000)


import streamlit as st
import os
from modules.embedding_generator import download_hugging_face_embeddings
from modules.pinecone_handler import initialize_pinecone
from modules.groq_handler import query_pinecone, generate

# Sidebar: API Key Input
st.sidebar.title("🔐 API Configuration")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
pinecone_api_key = st.sidebar.text_input("Enter Pinecone API Key", type="password")

# Stop the app if keys aren't provided
if not groq_api_key or not pinecone_api_key:
    st.warning("Please enter both Groq and Pinecone API keys in the sidebar.")
    st.stop()

# Optional: Set environment variables if used elsewhere
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

# Initialize resources with caching
@st.cache_resource
def init_resources(pinecone_api_key):
    embeddings = download_hugging_face_embeddings()
    index = initialize_pinecone(api_key=pinecone_api_key)
    return embeddings, index

# Pass key to the cached function
embeddings, index = init_resources(pinecone_api_key)

# Streamlit UI
st.title("🤖 AI Query Assistant")

user_query = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Processing your query..."):
            docs = query_pinecone(user_query, top_k=2, index=index, embeddings=embeddings)
            response = generate(query=user_query, docs=docs)
        st.success("Response:")
        st.write(response)

