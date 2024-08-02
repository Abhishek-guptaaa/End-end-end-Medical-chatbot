from flask import render_template, jsonify, request
from app import app
from app.helper import download_hugging_face_embeddings
from app.pinecone_manager import init_pinecone, generate_embeddings, upsert_embeddings, query_index
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from app.prompt import prompt_template
import os

load_dotenv()

embeddings = download_hugging_face_embeddings()

# Initialize Pinecone and load the index
index = init_pinecone()
docsearch = Pinecone.from_existing_index(index, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    results = query_index(question, embeddings)
    answer = qa({"question": question, "context": results})
    return jsonify(answer)
