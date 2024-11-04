import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from werkzeug.utils import secure_filename
import pandas as pd
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from eureka.client import EurekaClient

app = Flask(__name__)
CORS(app)
load_dotenv()

# Eureka client setup for service discovery
eureka_client = EurekaClient(
    app_name="chatbot-service",
    port=5000,
    eureka_url="http://localhost:8761/eureka"
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(question, context):
    prompt = f"""
    You are an AI banking assistant designed to help customers with banking services and inquiries. 
    Use the following guidelines:

    1. Focus on providing accurate information about banking services based on the bank's documentation
    2. Handle inquiries about:
       - Account services
       - Wire transfers
       - Beneficiary management
       - Transaction types (Normal vs Instant transfers)
       - Banking procedures and requirements
    3. Always maintain confidentiality and privacy standards
    4. If information isn't available in the context, clearly state that you need to refer to a banking representative
    5. For specific account inquiries or transactions, direct users to appropriate banking channels
    6. Explain banking terms and procedures in simple, clear language
    7. Never make assumptions about banking policies or procedures not explicitly stated in the documentation

    Customer Question: {question}
    Bank Documentation Context: {context}

    Your response:
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

@app.route('/api/chatbot/query', methods=['POST'])
def send_message():
    try:
        user_question = request.json.get('message')
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)
        context = "\n".join([doc.page_content for doc in docs])
        response = get_gemini_response(user_question, context)
        return jsonify({
            "status": "success",
            "response": response,
            "service": "chatbot-service"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/chatbot/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "UP",
        "service": "chatbot-service"
    })

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

if __name__ == "__main__":
    print("Starting Banking Chatbot Service...")
    eureka_client.start()
    app.run(host='0.0.0.0', port=5000)