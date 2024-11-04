import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """Extract text from banking service documentation PDFs."""
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text() or ""
            logger.info(f"Successfully processed {pdf}")
        except Exception as e:
            logger.error(f"Error processing {pdf}: {str(e)}")
    return text

def get_text_chunks(text):
    """Split banking documentation into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save vector embeddings for banking documentation."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def train_banking_chatbot():
    """Train the chatbot using banking service documentation."""
    logger.info("Starting banking chatbot training...")
    
    # Directory containing banking documentation PDFs
    pdf_directory = "./data/"
    pdf_docs = [os.path.join(pdf_directory, file) 
                for file in os.listdir(pdf_directory) 
                if file.endswith(".pdf")]
    
    if not pdf_docs:
        logger.error("No banking documentation PDFs found in directory")
        return
    
    logger.info(f"Found {len(pdf_docs)} banking documentation PDF(s)")
    
    text = get_pdf_text(pdf_docs)
    if not text.strip():
        logger.error("No text could be extracted from PDFs")
        return
        
    text_chunks = get_text_chunks(text)
    logger.info(f"Created {len(text_chunks)} text chunks")
    
    vector_store = get_vector_store(text_chunks)
    logger.info("Vector store created and saved successfully")
    
    # Validate the vector store
    try:
        test_query = "What are the types of wire transfers available?"
        results = vector_store.similarity_search(test_query)
        logger.info("Vector store validation successful")
    except Exception as e:
        logger.error(f"Vector store validation failed: {str(e)}")

if __name__ == "__main__":
    train_banking_chatbot()