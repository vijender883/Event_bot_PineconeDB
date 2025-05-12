# --- START OF FILE app.py ---

import streamlit as st
import os
import html
import hashlib # Import hashlib
import re # Import re for post-processing regex
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
import shutil # Needed for copying files
from datetime import datetime # Needed for timestamp metadata
from dotenv import load_dotenv

# --- New Imports for PDF processing ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- End New Imports ---

# Load environment variables
load_dotenv()

# --- Directory for submitted resumes ---
RESUME_DIR = "resume_submitted"
os.makedirs(RESUME_DIR, exist_ok=True)

# --- Directory for temporary uploads ---
TEMP_RESUME_DIR = "temp_resume_uploads"
os.makedirs(TEMP_RESUME_DIR, exist_ok=True) # Ensure temp dir exists

# --- File to store hashes of processed resume content ---
# This file tracks hashes of resumes processed *by the app* or the ingestion script
PROCESSED_HASHES_FILE = "processed_resume_hashes.txt"
# --- End Directory and Hash File Creation ---

class EventAssistantRAGBot:
    def __init__(self, api_key, pinecone_api_key, pinecone_cloud, pinecone_region, index_name):
        self.api_key = api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_cloud = pinecone_cloud
        self.pinecone_region = pinecone_region
        self.index_name = index_name

        # Initialize Pinecone with new SDK
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

        # --- Initialize Embeddings once ---
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", # Consistent with resume_processor.py
            google_api_key=self.api_key
        )
        # --- End Embeddings Initialization ---

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

        Remember: While you can be conversational, your primary role is providing accurate information based on the context provided (event documents and submitted resumes).

        Context information (event details and/or resume content):
        {context}
        --------

        Now, please answer this question: {question}
        """

    # --- Helper function to get text hash of a PDF ---
    def get_pdf_text_hash(self, pdf_file_path):
        """Loads PDF, extracts text, and returns SHA256 hash of the concatenated text."""
        try:
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()
            full_text = "\n".join([doc.page_content for doc in documents])
            if not full_text.strip(): # Handle cases where no text is extracted
                 # Use warning instead of error to avoid stopping the process for empty files
                 st.warning(f"No text extracted from PDF: {os.path.basename(pdf_file_path)}")
                 return None
            # Use sha256 for hashing
            return hashlib.sha256(full_text.encode('utf-8')).hexdigest()
        except Exception as e:
            st.error(f"Error extracting text or hashing PDF '{os.path.basename(pdf_file_path)}': {e}")
            return None

    # --- Helper function to load processed hashes ---
    def load_processed_hashes(self):
        """Loads hashes from the persistent file into a set."""
        hashes = set()
        if os.path.exists(PROCESSED_HASHES_FILE):
            try:
                with open(PROCESSED_HASHES_FILE, "r") as f:
                    for line in f:
                        hashes.add(line.strip())
            except Exception as e:
                st.error(f"Error loading processed hashes file: {e}")
        return hashes

    # --- Helper function to save a new hash ---
    def save_processed_hash(self, new_hash):
        """Appends a new hash to the persistent file."""
        try:
            with open(PROCESSED_HASHES_FILE, "a") as f:
                f.write(f"{new_hash}\n")
        except Exception as e:
            st.error(f"Error saving new hash to file: {e}")

    # --- Method to process and embed resume (Enhanced) ---
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

            # 2. Split documents (using consistent parameters from resume_processor.py)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, # From resume_processor.py
                chunk_overlap=200 # From resume_processor.py
            )
            docs = text_splitter.split_documents(documents)
            if not docs:
                st.sidebar.warning(f"No text chunks created after splitting '{file_name}'.")
                return False

            # 3. Add Metadata (Simplified version - add more if user info is collected)
            timestamp = datetime.now().isoformat()
            # Simple unique ID based on filename and timestamp hash part
            file_hash_part = hashlib.sha1(f"{file_name}-{timestamp}".encode()).hexdigest()[:8]

            for i, doc in enumerate(docs):
                doc.metadata.update({
                    "source": "resume_upload", # Distinguish source
                    "filename": file_name,
                    "chunk_id": f"resume_{file_hash_part}_{i}", # Unique chunk ID
                    "upload_timestamp": timestamp,
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
    # --- End enhanced method ---


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

                contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

            with st.spinner("Generating response..."):
                start_time = time.time()
                # Check if the model client attribute exists before using it
                if not hasattr(self, 'client') or self.client is None:
                     raise ValueError("Gemini client not initialized.")
                if not hasattr(self.client, 'models') or not hasattr(self.client.models, 'generate_content'):
                     raise AttributeError("Gemini client does not have 'models.generate_content' method.")

                response_genai = self.client.models.generate_content(
                    model="gemini-2.0-flash",  # Using Gemini 2.0 Flash model
                    contents=contents,
                )
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
                error_message += "\nPlease ensure the API key has permissions for the 'gemini-1.5-flash-latest' model."
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
</style>
""", unsafe_allow_html=True)

# Main app title
st.title("Build with AI - RAG Event Bot")

# Initialize session state for chat history and processed file ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_processed_upload_key" not in st.session_state:
    st.session_state.last_processed_upload_key = None


# Get API keys from environment variables
api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws") # Default values added
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1") # Default values added
pinecone_index = os.getenv("PINECONE_INDEX")

# Ensure necessary directories exist
os.makedirs(RESUME_DIR, exist_ok=True)
os.makedirs(TEMP_RESUME_DIR, exist_ok=True)

if not api_key:
    st.error("GEMINI_API_KEY not found. Please set it in your environment variables or .env file.")
    st.stop()
if not pinecone_api_key or not pinecone_index:
    st.error("PINECONE_API_KEY or PINECONE_INDEX not found. Please set them in your environment or .env file.")
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
                index_name=pinecone_index
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

            temp_file_path = os.path.join(TEMP_RESUME_DIR, uploaded_resume.name)
            final_file_path = os.path.join(RESUME_DIR, uploaded_resume.name)
            file_saved_temporarily = False
            # Get bot instance safely
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
                        os.makedirs(TEMP_RESUME_DIR, exist_ok=True) # Ensure dir exists
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_resume.getbuffer())
                        file_saved_temporarily = True

                        # 2. Calculate hash of the temporary file's content
                        content_hash = bot_instance.get_pdf_text_hash(temp_file_path)

                    if not content_hash:
                         st.sidebar.error(f"Could not process '{uploaded_resume.name}'. File might be empty or corrupted.")
                         # No further action needed, finally block will clean up temp file
                    else:
                        # 3. Load previously processed hashes
                        processed_hashes = bot_instance.load_processed_hashes()

                        # 4. Check if this content (hash) is new
                        if content_hash in processed_hashes:
                            st.sidebar.warning(f"Content of '{uploaded_resume.name}' has already been processed.")
                            # No need to upload or process further, hash already exists.
                        else:
                            # 5. Content is new - Save to final directory and process
                            st.sidebar.info(f"New content detected in '{uploaded_resume.name}'. Processing...")
                            try:
                                # Ensure final directory exists
                                os.makedirs(RESUME_DIR, exist_ok=True)
                                # Copy the temporary file to the final destination
                                shutil.copy2(temp_file_path, final_file_path) # copy2 preserves metadata

                                # --->>> PROCESS AND EMBED THE NEW RESUME <<<---
                                success = bot_instance.process_and_embed_resume(final_file_path)
                                # --->>> END PROCESSING <<<---

                                if success:
                                    # 6. Save the new hash ONLY if processing was successful
                                    bot_instance.save_processed_hash(content_hash)
                                    # Confirmation moved to process_and_embed_resume method
                                else:
                                   st.sidebar.error(f"Failed to process '{uploaded_resume.name}' after saving. It won't be added to the knowledge base this time.")
                                   # Optionally remove the saved file if processing failed critically
                                   # if os.path.exists(final_file_path):
                                   #     os.remove(final_file_path)

                            except Exception as copy_err:
                                st.sidebar.error(f"Error saving file '{uploaded_resume.name}' to destination: {copy_err}")
                                # Don't save hash or process if copy failed

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

                finally:
                    # 7. Clean up the temporary file if it was created
                    if file_saved_temporarily and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except OSError as e_clean:
                            st.sidebar.warning(f"Could not clean up temporary file {temp_file_path}: {e_clean}")

        # else:
            # Optional: Inform user if the same upload instance (name/size) was already handled in this session
            # st.sidebar.info(f"'{uploaded_resume.name}' was already handled in this session.")
            # pass # No action needed if already handled this session

# --- End Resume Upload Section ---

# Custom Chat UI Implementation
# (No changes needed in this section, but included for completeness)
chat_html = '<div class="custom-chat-container">'
for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = '<div class="avatar-icon user-avatar-icon">👤</div>'
        # Ensure user content is escaped
        escaped_user_content = html.escape(str(message.get("content", "")))
        chat_html += f'<div class="message-container user">{avatar}<div class="user-message">{escaped_user_content}</div></div>'
    elif message["role"] == "assistant": # Use elif for clarity
        avatar = '<div class="avatar-icon">🤖</div>'
        # Safely access content dictionary and its keys
        content_dict = message.get("content", {})
        content_text = str(content_dict.get("text", "...") ) # Default to "..." if text missing
        vector_db_time = content_dict.get("vector_db_time")
        llm_time = content_dict.get("llm_time")

        chat_html += f'<div class="message-container">{avatar}<div class="bot-message">'

        # Format message content (escape HTML, replace newlines)
        escaped_content = html.escape(content_text)
        formatted_content = escaped_content.replace('\n', '<br>')

        # Special handling for the structured welcome message (minor adjustments for robustness)
        if "Hello! I'm Event Bot." in content_text and "I can help you with the following:" in content_text:
             parts = content_text.split("I can help you with the following:", 1) # Split only once
             header = parts[0].strip().replace('\n', '<br>')
             list_section = parts[1].strip() if len(parts) > 1 else ""

             # Escape header HTML just in case
             formatted_welcome = f'<p>{html.escape(header)}</p><p>I can help you with the following:</p>'

             list_items = []
             how_can_i_help_text = ""
             # Extract list items and trailing text more carefully
             lines = list_section.split('\n')
             list_started = False
             for line in lines:
                 line_strip = line.strip()
                 match = re.match(r"(\d+)\.?\s+(.*)", line_strip)
                 if match:
                     list_started = True
                     list_items.append(match.group(2).strip())
                 elif list_started and line_strip: # Append to previous item if indented or part of multi-line
                     if list_items: list_items[-1] += f" {line_strip}"
                 elif not list_started and "How can I help you" in line_strip: # Capture trailing line
                     how_can_i_help_text = line_strip
                 elif list_started and "How can I help you" in line_strip: # Capture trailing after list
                     how_can_i_help_text = line_strip

             if list_items:
                 formatted_welcome += '<ol style="margin-top:8px;margin-bottom:8px;padding-left:25px;">'
                 for item in list_items:
                     if item:
                         # Escape list item content
                         formatted_welcome += f'<li style="margin-bottom:6px; line-height:1.4;">{html.escape(item)}</li>'
                 formatted_welcome += '</ol>'

             if how_can_i_help_text:
                 # Escape trailing text
                 formatted_welcome += f'<p>{html.escape(how_can_i_help_text)}</p>'

             chat_html += f'<div class="bot-message-content">{formatted_welcome}</div>'
        else:
            # Standard formatting for other bot messages
             chat_html += f'<div class="bot-message-content">{formatted_content}</div>'

        # Add timings if available
        if vector_db_time is not None and llm_time is not None:
             timings_html = f'<span class="bot-message-timings">Vector DB: {vector_db_time:.2f}s | LLM: {llm_time:.2f}s</span>'
             chat_html += timings_html
        chat_html += '</div></div>' # Closes bot-message and message-container

chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)


# Chat input
user_input = st.chat_input("Ask a question about the event or your resume...")

if user_input:
    # Prevent processing empty input
    user_input_stripped = user_input.strip()
    if not user_input_stripped:
        st.warning("Please enter a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input_stripped})
        # Use the initialized bot instance from session state
        bot_instance = st.session_state.get("bot")
        if bot_instance:
            response_dict = bot_instance.answer_question(user_input_stripped)
            st.session_state.messages.append({"role": "assistant", "content": response_dict})
        else:
            st.session_state.messages.append({"role": "assistant", "content": {"text": "Error: Assistant not initialized. Cannot answer.", "vector_db_time": None, "llm_time": None}})

        # Rerun to update the chat display
        st.rerun()

# --- END OF FILE app.py ---