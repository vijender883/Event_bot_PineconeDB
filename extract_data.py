import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from google import genai

# Load environment variables
load_dotenv()

# Set Google API key directly
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

client = genai.Client(api_key=gemini_api_key)

# Initialize Pinecone with the new SDK
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key  # Pass API key explicitly
)

cloud = os.getenv("PINECONE_CLOUD") or "aws"
region = os.getenv("PINECONE_REGION") or "us-east-1"
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = os.getenv("PINECONE_INDEX")

# Check if index exists and create if needed
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Dimension for Gemini embedding model
        metric="cosine",
        spec=spec
    )
    time.sleep(1)  # Wait for index to be ready

index = pc.Index(index_name)

# --- NEW: Delete all existing vectors in the index ---
print(f"Deleting all existing vectors from index '{index_name}'...")
index.delete(delete_all=True)
print("All vectors deleted.\n")

# See index stats before upsert
print("Index before upsert:")
print(index.describe_index_stats())
print("\n")

# Load PDF documents from the 'documents' folder
loader = DirectoryLoader("./documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=400)
docs = text_splitter.split_documents(documents)

# Initialize Pinecone with your documents using the newer SDK approach
vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=index_name,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

print(f"Uploaded {len(docs)} document chunks to Pinecone index '{index_name}'")

# Optionally check index stats after upsert
print("Index after upsert:")
print(index.describe_index_stats())
