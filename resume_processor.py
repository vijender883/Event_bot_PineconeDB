import os
import sys
import tempfile
import hashlib
from dotenv import load_dotenv
from datetime import datetime
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from google import genai

# Load environment variables from .env file
load_dotenv()

# Constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
EMBEDDING_MODEL_NAME = "models/embedding-001"
TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 200

def process_resume(pdf_path, user_name=None, user_email=None):
    """Process a resume PDF and upload it to the vector database."""
    try:
        # Validate that the file exists
        if not os.path.exists(pdf_path):
            return {"status": "error", "message": f"File not found: {pdf_path}"}
            
        # Initialize Google AI client
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Load the PDF using PyPDFLoader
        print(f"Loading PDF from {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        if not documents:
            return {"status": "error", "message": "No content found in the PDF."}
                
        # Split documents into chunks
        print("Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_CHUNK_SIZE,
            chunk_overlap=TEXT_CHUNK_OVERLAP
        )
        docs = text_splitter.split_documents(documents)
        
        if not docs:
            return {"status": "error", "message": "Failed to split document into chunks."}
        
        print(f"Created {len(docs)} document chunks.")
        
        # Add metadata to each document chunk
        timestamp = datetime.now().isoformat()
        user_id = hashlib.md5((user_email or "anonymous").encode()).hexdigest()
        
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "source": "resume",
                "chunk_id": f"{user_id}_{i}",
                "user_name": user_name or "Anonymous",
                "user_email": user_email or "Not provided",
                "upload_timestamp": timestamp,
                "document_type": "resume"
            })
        
        # Initialize the embeddings
        print("Initializing embeddings...")
        embedding_function = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            google_api_key=GEMINI_API_KEY
        )
        
        # Get the Pinecone index
        print(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Create vector store and add documents
        print("Creating vector store...")
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_function,
            text_key="text"
        )
        
        # Add documents to the vector store
        print("Adding documents to vector store...")
        vectorstore.add_documents(docs)
        
        return {
            "status": "success", 
            "message": f"Resume processed and added to the database. {len(docs)} chunks created."
        }
        
    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resume_processor.py <path_to_resume.pdf> [user_name] [user_email]")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    user_name = sys.argv[2] if len(sys.argv) > 2 else None
    user_email = sys.argv[3] if len(sys.argv) > 3 else None
    
    print(f"Processing resume for {user_name or 'Anonymous'} ({user_email or 'No email'})...")
    result = process_resume(pdf_path, user_name, user_email)
    
    if result["status"] == "success":
        print(f"SUCCESS: {result['message']}")
    else:
        print(f"ERROR: {result['message']}")