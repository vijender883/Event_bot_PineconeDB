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
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
     raise ValueError("PINECONE_API_KEY not found in environment variables")

pc = Pinecone(api_key=pinecone_api_key)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key  # Pass API key explicitly
)

cloud = os.getenv("PINECONE_CLOUD") or "aws"
region = os.getenv("PINECONE_REGION") or "us-east-1"
spec = ServerlessSpec(cloud=cloud, region=region)

# Check if index name is provided
index_name = os.getenv("PINECONE_INDEX")
if not index_name:
    raise ValueError("PINECONE_INDEX environment variable not found or is empty")

# Check if index exists and create if needed
print(f"Checking for index '{index_name}'...")
if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' not found. Creating index...")
    pc.create_index(
        name=index_name,
        dimension=768,  # Dimension for Gemini embedding model
        metric="cosine",
        spec=spec
    )
    print(f"Index '{index_name}' created.")
    # Give index a moment to become ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    print(f"Index '{index_name}' is ready.\n")
else:
    print(f"Index '{index_name}' found.\n")


index = pc.Index(index_name)

# Get index stats to check if it's empty before deleting
try:
    index_stats_before_delete = index.describe_index_stats()
    print("Index stats before potential deletion:")
    print(index_stats_before_delete)
    print("\n")

    # --- NEW: Delete all existing vectors in the index if not empty ---
    if index_stats_before_delete.total_vector_count > 0:
        print(f"Index '{index_name}' contains {index_stats_before_delete.total_vector_count} vectors. Deleting all...")
        index.delete(delete_all=True)
        print("All vectors deleted.\n")
    else:
        print(f"Index '{index_name}' is already empty. Skipping deletion.\n")

    # Get index stats again to confirm (should show 0 vectors before upsert)
    index_stats_before_upsert = index.describe_index_stats()
    print("Index stats before upsert:")
    print(index_stats_before_upsert)
    print("\n")

except Exception as e:
    print(f"Error accessing index stats or deleting vectors: {e}")
    # Decide how to proceed if there's an error here.
    # For now, we'll print the error and potentially stop or continue depending on its severity.
    # If the index object itself is invalid, subsequent operations will fail.
    # Add a check or re-initialize if necessary based on error type.
    # For this optimization task, just printing the error is sufficient based on user request.
    pass # Or re-raise or handle the error appropriately

# Load PDF documents from the 'documents' folder
# Add a check if the 'documents' directory exists
docs_directory = "./documents"
if not os.path.exists(docs_directory):
     raise FileNotFoundError(f"Documents directory '{docs_directory}' not found.")

print(f"Loading documents from '{docs_directory}'...")
try:
    loader = DirectoryLoader(docs_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        print("No PDF documents found in the directory.")
        # Decide whether to exit or continue if no documents are found.
        # If no documents, there's nothing to embed or upsert.
        # We can exit here.
        exit() # Exit the script if no documents are found
except Exception as e:
    print(f"Error loading documents: {e}")
    exit() # Exit on document loading error


# Split documents
print(f"Loaded {len(documents)} documents. Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=800)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")

if not docs:
     print("No document chunks created after splitting. Exiting.")
     exit() # Exit if no chunks are created

# Initialize Pinecone with your documents using the newer SDK approach
print(f"Upserting {len(docs)} document chunks to Pinecone index '{index_name}'...")
try:
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=pinecone_api_key # Pass the variable directly
    )
    print(f"Successfully uploaded {len(docs)} document chunks to Pinecone index '{index_name}'")
except Exception as e:
    print(f"Error during upserting documents to Pinecone: {e}")
    # Decide how to handle upsert errors. Print and exit, or try again?
    # For now, print error and continue to stat check.

# Optionally check index stats after upsert
print("Index after upsert:")
try:
    print(index.describe_index_stats())
except Exception as e:
     print(f"Error getting index stats after upsert: {e}")