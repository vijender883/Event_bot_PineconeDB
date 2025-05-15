import streamlit as st
import os
import html
# Ensure correct imports for Google GenAI and LangChain
from google.generativeai import types
import google.generativeai as genai # Use this alias for clarity
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
# --- New Imports for PDF processing ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- New Imports for Check-In Feature ---
import requests
import json
# --- End New Imports ---

# Load environment variables
load_dotenv()

# --- Directory for submitted resumes ---
RESUME_DIR = "resume_submitted"
os.makedirs(RESUME_DIR, exist_ok=True)
# --- End Directory and Hash File Creation ---

# --- New function for Check-In Feature ---
def check_registered_user(name):
    """
    Check if a user is registered for the event by calling the API.
    
    Args:
        name (str): The name to check against the registration list.
        
    Returns:
        tuple: (success (bool), message (str), data (list))
    """
    api_url = "https://api.practicalsystemdesign.com/api/eventbot/getRegisteredUserList"
    
    try:
        # Prepare request payload
        payload = {
            "name": name
        }
        
        # Make POST request to the API
        response = requests.post(api_url, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200 or response.status_code == 201:
            response_data = response.json()
            
            # Check if the API returned success
            if response_data.get("success", False):
                # Get the count and data from the response
                count = response_data.get("count", 0)
                data = response_data.get("data", [])
                
                if count == 0 or not data:
                    return False, "No registered user found with that name.", []
                
                return True, f"Found {count} registered user(s).", data
            else:
                return False, "API returned an unsuccessful response.", []
                
        else:
            return False, f"API request failed with status code {response.status_code}.", []
            
    except requests.exceptions.RequestException as e:
        return False, f"Failed to connect to the registration API: {str(e)}", []
    except json.JSONDecodeError as e:
        return False, f"Failed to parse API response: {str(e)}", []
    except Exception as e:
        return False, f"An unexpected error occurred: {str(e)}", []

def perform_checkin(user_id):
    """
    Perform the check-in operation for a user by calling the check-in API.
    
    Args:
        user_id (str): The ID of the user to check in.
        
    Returns:
        tuple: (success (bool), message (str))
    """
    api_url = "https://api.practicalsystemdesign.com/api/eventbot/checkinUser"
    
    try:
        # Make the API request with the user ID
        response = requests.post(
            api_url,
            json={"userId": user_id},
            headers={"Content-Type": "application/json"}
        )
        
        # Parse the response
        response_data = response.json()
        
        # Check if the request was successful
        if response.status_code == 200 and response_data.get("success", False):
            return True, response_data.get("message", "Successfully checked in")
        else:
            error_message = response_data.get("message", f"API returned status code {response.status_code}")
            return False, error_message
            
    except requests.exceptions.RequestException as e:
        return False, f"Failed to connect to the check-in API: {str(e)}"
    except json.JSONDecodeError as e:
        return False, f"Failed to parse API response: {str(e)}"
    except Exception as e:
        return False, f"An unexpected error occurred during check-in: {str(e)}"
# --- End New functions for Check-In Feature ---

class EventAssistantRAGBot:
    def __init__(self, api_key, pinecone_api_key, pinecone_cloud, pinecone_region, index_name, bucket_name, aws_access_key_id, aws_secret_access_key, region_name):
        self.api_key = api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_cloud = pinecone_cloud
        self.pinecone_region = pinecone_region
        self.index_name = index_name
        self.bucket_name = bucket_name # FIX: Assign bucket_name
        self.aws_access_key_id=aws_access_key_id
        self.aws_secret_access_key=aws_secret_access_key
        self.region_name= region_name

        # Initialize Pinecone with new SDK
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            # Optional: Check if index exists early, though VectorStore init will also fail
            # if self.index_name not in self.pc.list_indexes():
            #    st.error(f"Pinecone index '{self.index_name}' not found.")
            #    st.stop() # Or raise an exception
        except Exception as e:
            st.error(f"Failed to initialize Pinecone: {e}")
            st.stop() # Stop the app if Pinecone can't initialize

        # Initialize Gemini client
        try:
            self.client = genai.GenerativeModel("gemini-2.0-flash") # Use GenerativeModel directly for consistency
            # You can optionally check the model exists:
            # list(genai.list_models()) # Check if API key is valid and models are accessible
        except Exception as e:
             st.error(f"Failed to initialize Gemini client: {e}")
             st.stop()

        # --- Initialize Embeddings once ---
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
        except Exception as e:
             st.error(f"Failed to initialize Google Generative AI Embeddings: {e}")
             st.stop()
        # --- End Embeddings Initialization ---

        # --- Initialize S3 client once ---
        try:
             self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key, # FIX: Use self attribute
                region_name=self.region_name
            )
             # Optional: Check if bucket exists and is accessible
             # self.s3_client.head_bucket(Bucket=self.bucket_name)
        except (NoCredentialsError, PartialCredentialsError) as e:
            st.error(f"AWS credentials error: {e}")
            st.stop()
        except ClientError as e:
             st.error(f"AWS S3 client error: {e}")
             st.stop()
        except Exception as e:
             st.error(f"Failed to initialize AWS S3 client: {e}")
             st.stop()
        # --- End S3 Client Initialization ---


        self.prompt_template = """
        You are a friendly Event Information Assistant. Your primary purpose is to answer questions about the event described in the provided context. You can also answer questions based on user-submitted resumes if they have been provided. Follow these guidelines:

        1. You can respond to basic greetings like "hi", "hello", or "how are you" in a warm, welcoming manner
        2. For event information or resume content, only provide details that are present in the context
        3. If information is not in the context, politely say "I'm sorry, I don't have that specific information" (for event) or "I'm sorry, I don't have that information from the resume" (for resume).
        4. Keep responses concise but conversational
        5. Do not make assumptions beyond what's explicitly stated in the context
        6. Always prioritize factual accuracy while maintaining a helpful tone
        7. Do not introduce information that isn't in the context
        8. If unsure about any information, acknowledge uncertainty rather than guess
        9. You may suggest a few general questions users might want to ask about the event
        10. Remember to maintain a warm, friendly tone in all interactions
        11. You should refer to yourself as "Event Bot"
        12. You should not greet if the user has not greeted to you

        Remember: While you can be conversational, your primary role is providing accurate information based on the context provided (event details and/or resume content).

        Context information (event details and/or resume content):
        {context}
        --------

        Now, please answer this question: {question}
        """

    def upload_pdf_to_s3_resumes(self, local_file_path):
        """
        Uploads a local PDF file to the 'resumes' folder in an S3 bucket.

        Args:
            bucket_name (str): The name of the S3 bucket.
            local_file_path (str): The full path to the local PDF file to upload.

        Returns:
            str or None: The S3 key (path within the bucket, e.g., 'resumes/your_file.pdf')
                        if the upload is successful, None otherwise.
        """
        # Ensure the local file exists
        if not os.path.exists(local_file_path):
            print(f"Error: Local file not found at '{local_file_path}'")
            return None

        # Get the base name of the local file to use as the S3 object name
        # This extracts 'resume.pdf' from '/path/to/your/resume.pdf'
        file_name = os.path.basename(local_file_path)

        # Construct the S3 key (path in the bucket)
        # S3 uses '/' to simulate folders. The key will be 'folder_name/file_name'
        s3_folder = 'resumes'
        s3_key = f"{s3_folder}/{file_name}"

        # Initialize the S3 client
        s3 = boto3.client('s3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )

        print(f"Attempting to upload '{local_file_path}' to '{bucket_name}/{s3_key}'...")

        try:
            # Upload the file
            # upload_file is generally preferred for local files as it handles multipart uploads for large files
            s3.upload_file(local_file_path, bucket_name, s3_key)
            print(f"Successfully uploaded '{local_file_path}' to S3 as '{s3_key}'")
            return s3_key

        except FileNotFoundError:
            # This check is redundant due to the initial os.path.exists check,
            # but included for robustness in case file disappears between checks.
            print(f"Error: Local file was not found.")
            return None
        except NoCredentialsError:
            print("Error: AWS credentials not found.")
            print("Please configure your credentials (e.g., ~/.aws/credentials or environment variables).")
            return None
        except PartialCredentialsError:
            print("Error: AWS partial credentials found. Need access key ID and secret access key.")
            return None
        except ClientError as e:
            # Boto3 specific errors (e.g., bucket doesn't exist, permissions denied)
            print(f"Error uploading file to S3: {e}")
            return None
        except Exception as e:
            # Catch any other unexpected errors during the upload process
            print(f"An unexpected error occurred during S3 upload: {e}")
            return None
    

    def process_and_embed_resume(self, pdf_file_path):
        """
        Loads a PDF resume from the given path, splits it, adds metadata,
        embeds chunks, and upserts them to Pinecone.
        Assumes the check for existing content hash has already passed.
        """
        try:
            file_name = os.path.basename(pdf_file_path)
            st.sidebar.write(f"Processing '{file_name}' for embedding...") # Feedback

            # 1. Load PDF
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()
            if not documents:
                st.sidebar.warning(f"No content extracted from '{file_name}' for splitting.")
                return False

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=800)
            docs = text_splitter.split_documents(documents)
            if not docs:
                st.sidebar.warning(f"No text chunks created after splitting '{file_name}'.")
                return False

            # 3. Add Metadata (Simplified version - add more if user info is collected)

            for i, doc in enumerate(docs):
                doc.metadata.update({
                    "source": "resume_upload", # Distinguish source
                    "filename": file_name,
                    "document_type": "resume" # Consistent type
                    # Add user_name, user_email here if collected during upload
                })

            # 4. Upsert to Pinecone using the existing index and embeddings
            with st.spinner(f"Embedding {len(docs)} chunks from '{file_name}' and uploading..."):
                 index = self.pc.Index(self.index_name) # Get the index object
                 vectorstore = PineconeVectorStore(
                    index=index, # Pass the index object
                    embedding=self.embeddings,
                    text_key="text" # Ensure text key is defined
                 )
                 vectorstore.add_documents(docs) # Use add_documents to append

            st.sidebar.success(f"Successfully processed and added '{file_name}' ({len(docs)} chunks) to the knowledge base.")
            return True

        except Exception as e:
            st.sidebar.error(f"Error processing/embedding resume '{os.path.basename(pdf_file_path)}': {e}")
            # Log the full error for debugging if needed
            print(f"Detailed Embedding Error: {e}")
            import traceback
            traceback.print_exc()
            return False


    def post_process_response(self, response, query):
        """Format responses for better readability based on query type."""
        if "lunch" in query.lower() or "food" in query.lower() or "eat" in query.lower():
            formatted = "Regarding lunch:\n\n"
            points = []
            response_lower = response.lower()
            if any(phrase in response_lower for phrase in ["provided to all", "participants", "check-in"]):
                 points.append("• Lunch will be provided to all participants who have checked in at the venue.")
            if any(phrase in response_lower for phrase in ["cafeteria", "floor", "level 5", "5th"]):
                time_info = ""
                time_patterns = [r"1(?:[:.]00)?\s?(?:pm)?(?:\s?ist)?", r"2(?:[:.]00)?\s?(?:pm)?(?:\s?ist)?", r"13(?::?00)?", r"14(?::?00)?"]
                times_found = [re.search(pattern, response_lower) for pattern in time_patterns]
                if any(times_found):
                    time_info = "between 1:00 PM and 2:00 PM IST"
                points.append(f"• It will be served in the Cafeteria on the 5th floor {time_info}.".strip())
            if any(phrase in response_lower for phrase in ["check-in", "registration desk"]):
                points.append("• Please ensure you've completed the check-in process at the registration desk to be eligible.")
            if any(phrase in response_lower for phrase in ["volunteer", "directions", "help", "ask"]):
                points.append("• Feel free to ask a volunteer if you need directions to the cafeteria.")

            if not points: return response
            # Use dict.fromkeys to preserve order and remove duplicates effectively
            unique_points = list(dict.fromkeys(points))
            return formatted + "\n".join(unique_points)

        # Default: return original response
        return response


    def answer_question(self, query):
        """Use RAG with Google Gemini to answer a question based on retrieved context,
        and measure component response times."""
        vector_db_time = 0
        llm_time = 0
        raw_response = ""
        processed_response_text = ""

        try:
            embedding_function = self.embeddings
            index = self.pc.Index(self.index_name)
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=embedding_function,
                text_key="text"
            )

            with st.spinner("Retrieving relevant information..."):
                start_time = time.time()
                results = vectorstore.similarity_search_with_score(query, k=5) # k=5, can be tuned
                end_time = time.time()
                vector_db_time = end_time - start_time

                context_text = "\n\n --- \n\n".join([doc.page_content for doc, _score in results])
                if not context_text and results:
                    context_text = "No specific details found in the documents for your query."
                elif not results:
                    context_text = "No information found in the knowledge base for your query."

                prompt_template_obj = ChatPromptTemplate.from_template(self.prompt_template)
                prompt = prompt_template_obj.format(context=context_text, question=query)

                # FIX: Use the correct approach for the GenerativeModel API
                # Instead of using types.Content objects, just pass the text directly

            with st.spinner("Generating response..."):
                start_time = time.time()
                # Check if the model client attribute exists before using it
                if not hasattr(self, 'client') or self.client is None:
                    raise ValueError("Gemini client not initialized.")

                # FIX: Use the proper API for the GenerativeModel class
                response_genai = self.client.generate_content(prompt)
                
                end_time = time.time()
                llm_time = end_time - start_time

                try:
                    raw_response = response_genai.text
                    processed_response_text = self.post_process_response(raw_response, query)
                except ValueError as ve: # Catch potential errors from accessing parts/text
                    raw_response = f"Error processing GenAI response: {ve}\nDetails: {response_genai}"
                    processed_response_text = "I encountered an issue interpreting the response."
                    print(f"GenAI response error: {ve}")
                    print(f"GenAI response object: {response_genai}")
                except Exception as genai_err:
                    raw_response = f"Error generating response: {genai_err}\nDetails: {response_genai}"
                    processed_response_text = "I encountered an issue generating the response."
                    print(f"GenAI response error: {genai_err}")
                    print(f"GenAI response object: {response_genai}")

            return {
                "text": processed_response_text,
                "vector_db_time": vector_db_time,
                "llm_time": llm_time
            }

        except Exception as e:
            error_message = f"An error occurred during question answering: {str(e)}"
            if "permission" in str(e).lower() and "model" in str(e).lower():
                error_message += "\nPlease ensure the API key has permissions for the 'gemini-2.0-flash' model."
            # Print detailed error for server logs
            print(f"Detailed Answering Error: {e}")
            import traceback
            traceback.print_exc()

            return {
                "text": error_message,
                "vector_db_time": vector_db_time,
                "llm_time": llm_time
            }




# Set page configuration
st.set_page_config(
    page_title="Build with AI - RAG Event Bot",
    page_icon="🚀",
    layout="centered"
)

# Load the CSS
st.markdown("""
<style>
/* Bot message formatting */
.bot-message {
    white-space: pre-line !important;
    line-height: 1.5 !important;
    margin-bottom: 0 !important;
    color: black !important;
    background-color: #fcf8ed !important; /* Off-white background */
    padding: 10px 15px !important; /* Add padding */
    border-radius: 18px !important; /* Add border-radius */
    max-width: 80% !important; /* Set max width */
    margin-left: 10px !important; /* Add margin */
    word-wrap: break-word !important; /* Handle long words */
    display: flex !important; /* Use flexbox */
    flex-direction: column !important; /* Stack content and timings */
}
.bot-message ol { margin-top: 8px !important; margin-bottom: 8px !important; padding-left: 25px !important; }
.bot-message li { margin-bottom: 6px !important; padding-bottom: 0 !important; line-height: 1.4 !important; }
.bot-message p { margin-bottom: 10px !important; }
.bot-message-content { flex-grow: 1 !important; margin-bottom: 5px !important; line-height: 1.5 !important; white-space: pre-line !important; }
.custom-chat-container { display: flex !important; flex-direction: column !important; gap: 10px !important; margin-bottom: 20px !important; max-width: 800px !important; background-color: white !important; }
.message-container { display: flex !important; align-items: flex-start !important; margin-bottom: 10px !important; background-color: white !important; color: black !important; }
.message-container.user { flex-direction: row-reverse !important; }
.avatar-icon { width: 36px !important; height: 36px !important; border-radius: 50% !important; background-color: #E8F0FE !important; display: flex !important; justify-content: center !important; align-items: center !important; font-size: 20px !important; margin: 0 10px !important; flex-shrink: 0 !important; }
.user-avatar-icon { background-color: #F0F2F5 !important; }
.user-message { background-color: #F0F2F5 !important; padding: 10px 15px !important; border-radius: 18px !important; max-width: 80% !important; margin-right: 10px !important; word-wrap: break-word !important; }
.bot-message-timings { font-size: 0.75em !important; color: #555 !important; margin-top: 5px !important; display: block !important; align-self: flex-end !important; }
div.custom-chat-container { border-radius: 15px; border: 1px solid #ccc; padding: 10px; }
.success-message { 
    background-color: #d4edda !important;
    color: #155724 !important;
    padding: 10px !important;
    border-radius: 5px !important;
    margin: 10px 0 !important;
    border: 1px solid #c3e6cb !important;
}
.error-message { 
    background-color: #f8d7da !important;
    color: #721c24 !important;
    padding: 10px !important;
    border-radius: 5px !important;
    margin: 10px 0 !important;
    border: 1px solid #f5c6cb !important;
}
</style>
""", unsafe_allow_html=True)

# Main app title
st.title("Build with AI - RAG Event Bot")

# Initialize session state for chat history and processed file ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_processed_upload_key" not in st.session_state:
    st.session_state.last_processed_upload_key = None
# Initialize check-in related session state variables
if "checked_in" not in st.session_state:
    st.session_state.checked_in = False
if "checkin_message" not in st.session_state:
    st.session_state.checkin_message = ""
if "checkin_status" not in st.session_state:
    st.session_state.checkin_status = ""
if "selected_user_id" not in st.session_state:
    st.session_state.selected_user_id = None


# Get API keys and configurations from environment variables
api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws") # Default to aws if not set
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1") # Default region if not set
pinecone_index = os.getenv("PINECONE_INDEX")
# FIX: Add BUCKET_NAME check
bucket_name = os.getenv("BUCKET_NAME")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key =os.getenv("AWS_SECRET_ACCESS_KEY")
region_name = "ap-south-1"
# Ensure necessary directories exist
os.makedirs(RESUME_DIR, exist_ok=True)

if not api_key:
    st.error("GEMINI_API_KEY not found. Please set it in your environment variables or .env file.")
    st.stop()
if not pinecone_api_key or not pinecone_index:
    st.error("PINECONE_API_KEY or PINECONE_INDEX not found. Please set them in your environment or .env file.")
    st.stop()
if not bucket_name:
     st.error("BUCKET_NAME not found. Please set it in your environment variables or .env file.")
     st.stop()
if not aws_access_key_id:
    st.error("AWS_ACCESS_KEY_ID not found. Please set it in your environment variables or .env file.")
    st.stop()
if not aws_secret_access_key:
    st.error("AWS_SECRET_ACCESS_KEY not found. Please set it in your environment variables or .env file.")
    st.stop()
if not region_name:
    st.error("AWS_DEFAULT_REGION not found. Please set it in your environment variables or .env file.")
    st.stop()

# Initialize the bot
if "bot" not in st.session_state:
    with st.spinner("Initializing assistant..."):
        try:
            st.session_state.bot = EventAssistantRAGBot(
                api_key=api_key,
                pinecone_api_key=pinecone_api_key,
                pinecone_cloud=pinecone_cloud,
                pinecone_region=pinecone_region,
                index_name=pinecone_index,
                bucket_name=bucket_name, # Pass the loaded bucket name
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
            # Add welcome message only if bot initializes successfully and no messages exist
            if not st.session_state.messages:
                welcome_message_text = """Hello! I'm Event Bot.
I can help you with the following:
1. Agenda of the "Build with AI" workshop
2. Important Dates of this workshop
3. Details of the AI Hackathon
4. Presentation of Interesting projects in AI, ML
5. Locating the washrooms
6. Details of lunch at the venue
7. Information from your uploaded resume (if you provide one via the sidebar).

How can I help you with information about this event?"""
                st.session_state.messages.append(
                    {"role": "assistant", "content": {"text": welcome_message_text, "vector_db_time": None, "llm_time": None}}
                )
        except Exception as init_err:
            st.error(f"Failed to initialize assistant: {init_err}")
            # Log the error for debugging
            print(f"Initialization Error: {init_err}")
            import traceback
            traceback.print_exc()
            st.stop() # Stop the app if initialization fails


# --- Check-in Feature in Sidebar ---
with st.sidebar:
    st.header("Event Check-in")
    
    # Check if user is already checked in
    if st.session_state.checked_in:
        st.success(f"✅ Successfully checked in: {st.session_state.checkin_message}")
    else:
        # Input field for name
        name_input = st.text_input("Enter your name to check in:", key="checkin_name")
        
        # Process the name for check-in when user submits
        if st.button("Check-in", key="checkin_button"):
            if not name_input:
                st.error("Please enter your name to check in.")
            else:
                # Call the function to check if user is registered
                with st.spinner("Checking registration..."):
                    success, message, user_data = check_registered_user(name_input)
                
                if success:
                    # User(s) found in registration list
                    if len(user_data) == 1:
                        # Only one user found, proceed with check-in
                        user = user_data[0]
                        user_id = user.get("_id")
                        name = user.get("name")
                        additional_details = user.get("additional_details", "")
                        
                        # Perform check-in
                        if perform_checkin(user_id):
                            st.session_state.checked_in = True
                            st.session_state.checkin_message = f"{name}" + (f" ({additional_details})" if additional_details else "")
                            st.session_state.checkin_status = "success"
                            st.session_state.selected_user_id = user_id
                            st.success(f"✅ Successfully checked in: {st.session_state.checkin_message}")
                        else:
                            st.error("Failed to update check-in status. Please try again.")
                    else:
                        # Multiple users found, show dropdown to select
                        st.write(f"Found {len(user_data)} users with similar names. Please select your name:")
                        
                        # Store the user data in session state so it persists
                        st.session_state.multiple_users_found = True
                        st.session_state.multiple_user_data = user_data
                
                else:
                    # User not found in registration list
                    st.error(f"❌ {message}")
                    st.session_state.checkin_status = "error"
        
        # Display the dropdown and confirm button if multiple users were found
        if st.session_state.get('multiple_users_found', False):
            user_data = st.session_state.multiple_user_data
            
            # Create options for the dropdown
            options = [f"{user.get('name')}" + (f" ({user.get('additional_details')})" if user.get('additional_details') else "") for user in user_data]
            
            # Show the dropdown and get the selected index
            selected_option = st.selectbox("Select your name:", options, key="user_select")
            
            if st.button("Confirm Check-in", key="confirm_checkin"):
                # Find the selected user data
                selected_index = options.index(selected_option)
                selected_user = user_data[selected_index]
                user_id = selected_user.get("_id")
                
                # Perform check-in for the selected user
                if perform_checkin(user_id):
                    st.session_state.checked_in = True
                    st.session_state.checkin_message = selected_option
                    st.session_state.checkin_status = "success"
                    st.session_state.selected_user_id = user_id
                    # Clear the multiple users found flag
                    st.session_state.multiple_users_found = False
                    st.success(f"✅ Successfully checked in: {selected_option}")
                else:
                    st.error("Failed to update check-in status. Please try again.")

    # Add a separator between check-in and resume upload
    st.markdown("---")
    
# --- End Check-in Feature ---
# --- Resume Upload Section in Sidebar ---
with st.sidebar:
    st.header("Submit Your Resume")
    st.markdown("Upload your resume (PDF). New content will be added to the knowledge base.")
    # Use a unique key for the file uploader widget itself
    uploaded_resume = st.file_uploader("Upload PDF", type="pdf", key="pdf_resume_uploader_widget")

    # Logic to process the uploaded resume
    if uploaded_resume is not None:
        # Create a unique key for the current upload *instance* using name and size
        current_upload_key = f"{uploaded_resume.name}_{uploaded_resume.size}"

        # Check if this *specific upload instance* has already been processed in this session
        if st.session_state.get("last_processed_upload_key") != current_upload_key:

            file_path = os.path.join(RESUME_DIR, uploaded_resume.name)
            try:
                with open(file_path, 'wb') as f:
                    f.write(uploaded_resume.read())
                print(f"PDF saved successfully to: {file_path}")
            except Exception as e:
                print(f"Error saving PDF: {e}")
            bot_instance = st.session_state.get("bot")

            if not bot_instance:
                st.sidebar.error("Assistant not ready. Please wait a moment and try uploading again.")
                # Don't set the processed key if the bot isn't ready
            else:
                # Mark this specific upload attempt as being processed for this session run *before* starting
                st.session_state.last_processed_upload_key = current_upload_key
                try:
                    # 1. Save the uploaded file temporarily for hashing
                    with st.spinner(f"Checking '{uploaded_resume.name}'..."):
                        success = bot_instance.upload_pdf_to_s3_resumes(file_path)
                        success2 = bot_instance.process_and_embed_resume(file_path)
                except Exception as e:
                    st.sidebar.error(f"An error occurred during upload processing: {e}")
                    # Log error for debugging
                    print(f"Upload Processing Error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Reset upload key so user can retry if the error was temporary
                    # Only reset if it wasn't successfully marked processed
                    if st.session_state.get("last_processed_upload_key") == current_upload_key:
                         st.session_state.last_processed_upload_key = None

# --- End Resume Upload Section ---

# Custom Chat UI Implementation
# (No changes needed in this section, but included for completeness)
# Custom Chat UI Implementation
chat_html = '<div class="custom-chat-container">'
for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = '<div class="avatar-icon user-avatar-icon">👤</div>'
        chat_html += f'<div class="message-container user">{avatar}<div class="user-message">{html.escape(message["content"])}</div></div>'
    else:  # assistant
        avatar = '<div class="avatar-icon">🤖</div>'
        content_dict = message["content"]
        content_text = content_dict["text"]
        vector_db_time = content_dict.get("vector_db_time")
        llm_time = content_dict.get("llm_time")

        chat_html += f'<div class="message-container">{avatar}<div class="bot-message">'
        
        # Format message content (welcome message special handling)
        if "I can help you with the following:" in content_text and "1." in content_text and "Hello! I'm Event bot." in content_text :
            welcome_html = content_text.replace(
                "Hello! I'm Event bot.\nI can help you with the following:",
                "Hello! I'm Event bot.<br><br>I can help you with the following:"
            )
            # Dynamically create list items for robustness
            import re
            # Match numbered list items like "1. Text" or "\n1. Text"
            welcome_html = re.sub(r"(\n)?([1-9]\d*)\. (.*?)(?=(\n[1-9]\d*\. )|\n\n|$)", 
                                  lambda m: f"</li><li style='margin-bottom:4px;'>{m.group(3).strip()}" if m.group(1) or m.start() > 0 else f"<ol style='margin-top:8px;margin-bottom:8px;padding-left:25px;'><li style='margin-bottom:4px;'>{m.group(3).strip()}", 
                                  welcome_html, flags=re.DOTALL)
            # Close the ol tag if it was opened
            if "<ol" in welcome_html and "</ol>" not in welcome_html:
                 welcome_html += "</li></ol>"
            welcome_html = welcome_html.replace("How can I help you", "<br>How can I help you")

            chat_html += f'<div class="bot-message-content">{welcome_html}</div>'
        else:
            escaped_content = html.escape(content_text)
            formatted_content = escaped_content.replace('\n', '<br>')
            chat_html += f'<div class="bot-message-content">{formatted_content}</div>'

        if vector_db_time is not None and llm_time is not None:
             timings_html = f'<span class="bot-message-timings">Vector DB: {vector_db_time:.2f}s | LLM: {llm_time:.2f}s</span>'
             chat_html += timings_html
        chat_html += '</div></div>' # Closes bot-message and message-container

chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask a question about the event or your resume...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response_dict = st.session_state.bot.answer_question(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response_dict})
    st.rerun()