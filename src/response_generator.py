# response_generator.py
import os
from langchain_groq import ChatGroq
from src.config import GROQ_API_URL
from dotenv import load_dotenv
load_dotenv()


GROQ_API_KEY=os.getenv('GROQ_API_KEY')

llm=ChatGroq()

def generate_response(question, context):
    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = prompt_template.format(context=context, question=question)
    return generate_response_groq(prompt)
