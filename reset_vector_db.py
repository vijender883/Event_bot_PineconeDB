import os
import time
import hashlib # Import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from google import genai

# Load environment variables
load_dotenv()

# --- Define Directories and Files ---
DOCS_DIR = "./documents"
RESUME_DIR = "./resume_submitted"
PROCESSED_HASHES_FILE = "processed_resume_hashes.txt" # File to store hashes of CURRENTLY processed resumes
# --- End Define Directories and Files ---

# Ensure resume directory exists
os.makedirs(RESUME_DIR, exist_ok=True)

# --- Helper functions for hashing ---
def get_pdf_text_hash(pdf_file_path):
    """Loads PDF, extracts text, and returns SHA256 hash of the concatenated text."""
    try:
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])
        if not full_text.strip(): # Handle empty text content
             print(f"Warning: No text extracted from PDF: {os.path.basename(pdf_file_path)}")
             return None
        return hashlib.sha256(full_text.encode('utf-8')).hexdigest()
    except Exception as e:
        print(f"Error extracting text or hashing PDF '{os.path.basename(pdf_file_path)}': {e}")
        return None

# --- Removed load_processed_hashes() and save_processed_hash() ---
# We will collect hashes during the run and write them all at once at the end.

# --- End Helper functions ---


# Set Google API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
     raise ValueError("PINECONE_API_KEY not found in environment variables")

pc = Pinecone(api_key=pinecone_api_key)

# Initialize Embeddings
print("Initializing embeddings model...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)
print("Embeddings model initialized.")

cloud = os.getenv("PINECONE_CLOUD") or "aws"
region = os.getenv("PINECONE_REGION") or "us-east-1"
spec = ServerlessSpec(cloud=cloud, region=region)

# Check if index name is provided
index_name = os.getenv("PINECONE_INDEX")
if not index_name:
    raise ValueError("PINECONE_INDEX environment variable not found or is empty")

# Check if index exists and create if needed
print(f"\nChecking for index '{index_name}'...")
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
    print(f"Index '{index_name}' is ready.")
else:
    print(f"Index '{index_name}' found.")

index = pc.Index(index_name)

# --- Delete all existing vectors in the index ---
# Keeping this from your provided script, implying a full rebuild on each run.
try:
    index_stats_before_delete = index.describe_index_stats()
    print("\nIndex stats before potential deletion:")
    print(index_stats_before_delete)

    if index_stats_before_delete.total_vector_count > 0:
        print(f"\nIndex '{index_name}' contains {index_stats_before_delete.total_vector_count} vectors. Deleting all...")
        index.delete(delete_all=True)
        print("All vectors deleted.")
    else:
        print(f"\nIndex '{index_name}' is already empty. Skipping deletion.")

    # Give index a moment to register deletion
    time.sleep(5) # Adjust sleep if needed based on index size/type

    index_stats_before_upsert = index.describe_index_stats()
    print("Index stats before upsert:")
    print(index_stats_before_upsert)


except Exception as e:
    print(f"\nError accessing index stats or deleting vectors: {e}")
    print("Proceeding with upsert, but index state might be inconsistent.")
    # Depending on error, might want to exit here. For now, continue.


# --- Process Event Documents ---
print(f"\n--- Processing Event Documents from '{DOCS_DIR}' ---")
if not os.path.exists(DOCS_DIR):
     print(f"Error: Documents directory '{DOCS_DIR}' not found. Skipping event documents.")
     event_documents = [] # Empty list if directory doesn't exist
else:
    print(f"Loading documents from '{DOCS_DIR}'...")
    try:
        # Load ALL pdfs from the documents directory into a single list of documents
        loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
        event_documents = loader.load()
        print(f"Loaded {len(event_documents)} event documents.")
        if not event_documents:
             print("No PDF documents found in the event documents directory.")
    except Exception as e:
        print(f"Error loading event documents: {e}")
        event_documents = [] # Continue even if event docs fail


event_chunks = []
if event_documents:
    # Split event documents into chunks
    print(f"Splitting {len(event_documents)} event documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=800)
    event_chunks = text_splitter.split_documents(event_documents)
    print(f"Created {len(event_chunks)} event chunks.")

    if not event_chunks:
         print("No event chunks created after splitting.")

# Upsert event document chunks
if event_chunks:
    print(f"Upserting {len(event_chunks)} event document chunks to Pinecone index '{index_name}'...")
    try:
        # Use add_documents which appends to the index
        # Initialize PineconeVectorStore for upserting
        vectorstore = PineconeVectorStore(
            index_name=index_name, # Specify index name here
            embedding=embeddings,
            text_key="text"
        )
        vectorstore.add_documents(event_chunks)
        print(f"Successfully uploaded {len(event_chunks)} event document chunks.")
    except Exception as e:
        print(f"Error during upserting event documents to Pinecone: {e}")


# --- Process Resume Documents ---
print(f"\n--- Processing Resume Documents from '{RESUME_DIR}' ---")
# --- MODIFICATION START: Use a set to track hashes processed THIS RUN ---
successfully_processed_resume_hashes = set()
# --- MODIFICATION END ---

resume_files = [f for f in os.listdir(RESUME_DIR) if f.endswith('.pdf')]
if not resume_files:
    print(f"No PDF files found in the '{RESUME_DIR}' directory. Skipping resume processing.")
else:
    print(f"Found {len(resume_files)} potential resume files.")
    processed_count = 0
    skipped_count = 0
    error_count = 0 # Track errors separately

    for resume_file in resume_files:
        resume_file_path = os.path.join(RESUME_DIR, resume_file)
        print(f"\nProcessing resume: {resume_file}")

        # 1. Calculate hash of the resume's content
        content_hash = get_pdf_text_hash(resume_file_path)

        if content_hash:
            # --- MODIFICATION START: Check against hashes processed THIS RUN ---
            if content_hash in successfully_processed_resume_hashes:
                print(f"  Content of '{resume_file}' has already been processed in this run. Skipping.")
                skipped_count += 1
            # --- MODIFICATION END ---
            else:
                print(f"  Attempting to load and embed '{resume_file}'...")
                try:
                    # Load the SINGLE resume document
                    loader = PyPDFLoader(resume_file_path)
                    resume_documents = loader.load()
                    if not resume_documents:
                         print(f"  Warning: No content extracted from '{resume_file}' for splitting.")
                         skipped_count += 1 # Treat as skipped, not error, as hash was generated
                         continue # Skip this file

                    # Split the single resume document into chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=800)
                    resume_chunks = text_splitter.split_documents(resume_documents)
                    print(f"  Split into {len(resume_chunks)} chunks.")

                    if not resume_chunks:
                         print(f"  Warning: No chunks created after splitting '{resume_file}'. Skipping.")
                         skipped_count += 1 # Treat as skipped
                         continue # Skip this file

                    # Upsert the resume chunks to Pinecone
                    print(f"  Upserting {len(resume_chunks)} chunks for '{resume_file}'...")
                    # Initialize PineconeVectorStore for upserting
                    vectorstore = PineconeVectorStore(
                         index_name=index_name, # Specify index name here
                         embedding=embeddings,
                         text_key="text"
                    )
                    vectorstore.add_documents(resume_chunks)

                    # --- MODIFICATION START: Add hash to the set for THIS RUN ---
                    successfully_processed_resume_hashes.add(content_hash)
                    # --- MODIFICATION END ---

                    print(f"  Successfully uploaded chunks for '{resume_file}'.")
                    processed_count += 1

                except Exception as e:
                    print(f"  Error processing and embedding '{resume_file}': {e}")
                    error_count += 1
                    # Do not add the hash to the set if an error occurred during processing/embedding

        else:
             print(f"  Could not generate content hash for '{resume_file}'. Skipping.")
             skipped_count += 1

    # --- MODIFICATION START: Write successfully processed hashes to file AFTER loop ---
    print(f"\nWriting {len(successfully_processed_resume_hashes)} successfully processed resume hashes to '{PROCESSED_HASHES_FILE}'...")
    
    try:
        with open(PROCESSED_HASHES_FILE, "w") as f: # Use "w" to overwrite
            for h in successfully_processed_resume_hashes:
                f.write(f"{h}\n")
        print(f"Successfully updated '{PROCESSED_HASHES_FILE}'.")
    except Exception as e:
        print(f"Error writing processed hashes to file '{PROCESSED_HASHES_FILE}': {e}")
    # --- MODIFICATION END ---


    print(f"\nResume processing summary:")
    print(f"  Successfully processed and added {processed_count} resumes to Pinecone.")
    print(f"  Skipped {skipped_count} resumes (duplicate content in this run, empty PDF, or hash error).")
    print(f"  Encountered errors during processing/embedding for {error_count} resumes.")


# --- Final Check ---
print("\nFinal Index stats after all upserts:")
try:
    print(index.describe_index_stats())
except Exception as e:
     print(f"Error getting final index stats: {e}")

print("\nIngestion process finished.")