import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import time
# Fix 1: Update imports to use langchain_community
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from google import genai

# Load environment variables
load_dotenv()

# Fix 2: Set Google API key directly
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Fix 3: Create a client instance instead of using configure
client = genai.Client(api_key=gemini_api_key)

# Initialize Pinecone with the new SDK
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create embedding function with explicit API key
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key  # Pass API key explicitly
)

# Set up Pinecone configuration
cloud = os.getenv("PINECONE_CLOUD") or "aws"
region = os.getenv("PINECONE_REGION") or "us-east-1"
spec = ServerlessSpec(cloud=cloud, region=region)

# Get index name from environment variables
index_name = os.getenv("PINECONE_INDEX")

# Check if index exists and create if needed
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Dimension for Gemini embedding model
        metric="cosine",
        spec=spec
    )
    # Wait for index to be ready
    time.sleep(1)

# See index stats before upsert
print("Index before upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")

# Load PDF documents from the 'documents' folder
loader = DirectoryLoader("./documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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
print(pc.Index(index_name).describe_index_stats())

