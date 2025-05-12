import streamlit as st
import os
import html
import google.generativeai as genai
# --- MongoDB Imports (Replace Pinecone imports) ---
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
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
# --- End Directory Creation ---

class EventAssistantRAGBot:
    def __init__(self, api_key, mongodb_uri, mongodb_db_name, mongodb_collection):
        self.api_key = api_key
        self.mongodb_uri = mongodb_uri
        self.mongodb_db_name = mongodb_db_name
        self.mongodb_collection = mongodb_collection
        self.index_name = "vector_db_index"  # Changed from "default" to "vector_db_index"

        # Initialize MongoDB client
        self.mongo_client = MongoClient(self.mongodb_uri)
        self.db = self.mongo_client[self.mongodb_db_name]
        self.collection = self.db[self.mongodb_collection]

        # Initialize Gemini client
        genai.configure(api_key=self.api_key)

        # --- Initialize Embeddings once ---
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
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

        Remember: While you can be conversational, your primary role is providing accurate information based on the context provided (event details and/or resume content).

        Context information (event details and/or resume content):
        {context}
        --------

        Now, please answer this question: {question}
        """


    def post_process_response(self, response, query):
        """Format responses for better readability based on query type."""
        if "lunch" in query.lower() or "food" in query.lower() or "eat" in query.lower():
            formatted = "Regarding lunch:\n\n"
            points = []
            if "provided to all" in response:
                points.append("• Lunch will be provided to all participants who have checked in at the venue.")
            if "cafeteria" in response.lower() and "floor" in response.lower():
                time_info = ""
                if "1:00" in response and "2:00" in response:
                    time_info = "between 1:00 PM and 2:00 PM IST"
                points.append(f"• It will be served in the Cafeteria on the 5th floor {time_info}.")
            if "check-in" in response.lower() or "registration" in response.lower():
                points.append("• Please ensure you've completed the check-in process at the registration desk to be eligible.")
            if "volunteer" in response.lower() or "direction" in response.lower():
                points.append("• Feel free to ask a volunteer if you need directions to the cafeteria.")
            if not points: return response
            return formatted + "\n".join(points)
        return response

    # --- New method to process and embed resume ---
    def add_resume_to_vectorstore(self, pdf_file_path):
        """
        Loads a PDF resume, splits it into chunks, embeds them, and upserts to MongoDB.
        """
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()
            if not documents:
                st.warning(f"No content extracted from the PDF: {os.path.basename(pdf_file_path)}")
                return False

            # Split documents (using same parameters as extract_data.py)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=800)
            docs = text_splitter.split_documents(documents)
            if not docs:
                st.warning(f"No text chunks created after splitting the resume: {os.path.basename(pdf_file_path)}")
                return False
            
            # Create MongoDB vector store and add documents
            vector_store = MongoDBAtlasVectorSearch.from_documents(
                documents=docs,
                embedding=self.embeddings,
                collection=self.collection,
                index_name=self.index_name  # Changed from "default" to self.index_name
            )
            
            return True
        except Exception as e:
            st.error(f"Error processing and embedding resume '{os.path.basename(pdf_file_path)}': {e}")
            return False
    # --- End new method ---

    def answer_question(self, query):
        """Use RAG with Google Gemini to answer a question based on retrieved context,
        and measure component response times."""
        vector_db_time = 0
        llm_time = 0
        raw_response = ""
        processed_response_text = "" 

        try:
            # Create vector store from the existing MongoDB collection
            vectorstore = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embeddings,
                index_name=self.index_name,  # Changed from "default" to self.index_name
                text_key="text"
            )

            with st.spinner("Retrieving relevant information..."):
                start_time = time.time()
                results = vectorstore.similarity_search_with_score(query, k=5) # k=5, can be tuned
                end_time = time.time()
                vector_db_time = end_time - start_time

                context_text = "\n\n --- \n\n".join([doc.page_content for doc, _score in results])
                if not context_text and results: # If results found but context_text is empty (e.g. empty page_content)
                    context_text = "No specific details found in the documents for your query."
                elif not results:
                     context_text = "No information found in the knowledge base for your query."

                # Format prompt using the template
                prompt_template_obj = ChatPromptTemplate.from_template(self.prompt_template)
                prompt = prompt_template_obj.format(context=context_text, question=query)

            with st.spinner("Generating response..."):
                start_time = time.time()
                
                # Use the latest Gemini API method
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                
                end_time = time.time()
                llm_time = end_time - start_time
                
                # Extract text from response
                if hasattr(response, 'text'):
                    raw_response = response.text
                elif hasattr(response, 'parts'):
                    raw_response = response.parts[0].text
                else:
                    # Fallback for any other response structure
                    raw_response = str(response)
                
                processed_response_text = self.post_process_response(raw_response, query)

            return {
                "text": processed_response_text,
                "vector_db_time": vector_db_time,
                "llm_time": llm_time
            }

        except Exception as e:
            # Check for specific API errors if needed, e.g. permission denied for model
            error_message = f"An error occurred: {str(e)}"
            if "permission" in str(e).lower() and "model" in str(e).lower():
                error_message += "\nPlease ensure the API key has permissions for the Gemini models."
            
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

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file_id" not in st.session_state: # For tracking processed resume
    st.session_state.processed_file_id = None

# Get API keys from environment variables
api_key = os.getenv("GEMINI_API_KEY")
mongodb_uri = os.getenv("MONGODB_URI")
mongodb_db_name = os.getenv("MONGODB_DB_NAME")  # Using MONGODB_DB_NAME from .env
mongodb_collection = os.getenv("MONGODB_COLLECTION")

if not api_key:
    st.error("GEMINI_API_KEY not found. Please set it in your environment variables or .env file.")
    st.stop()
if not mongodb_uri or not mongodb_db_name or not mongodb_collection:
    st.error("MongoDB connection parameters not found. Please set MONGODB_URI, MONGODB_DB_NAME, and MONGODB_COLLECTION in your environment or .env file.")
    st.stop()

# Initialize the bot
if "bot" not in st.session_state:
    with st.spinner("Initializing assistant..."):
        st.session_state.bot = EventAssistantRAGBot(
            api_key=api_key,
            mongodb_uri=mongodb_uri,
            mongodb_db_name=mongodb_db_name,
            mongodb_collection=mongodb_collection
        )
    # Add welcome message if no messages yet
    if not st.session_state.messages:
        welcome_message_text = """Hello! I'm Event bot.
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

# --- Resume Upload Section in Sidebar ---
with st.sidebar:
    st.header("Submit Your Resume")
    st.markdown("Upload your resume in PDF format. I can then answer questions based on its content.")
    uploaded_resume = st.file_uploader("Upload PDF", type="pdf", key="pdf_resume_uploader")

    if uploaded_resume is not None:
        current_file_id = f"{uploaded_resume.name}_{uploaded_resume.size}"

        if st.session_state.processed_file_id != current_file_id:
            st.sidebar.info(f"New resume detected: '{uploaded_resume.name}'.")
            file_path = os.path.join(RESUME_DIR, uploaded_resume.name)
            
            # Save the uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_resume.getbuffer())
            st.sidebar.write(f"Resume '{uploaded_resume.name}' saved to server.")

            # Process and embed the resume
            with st.spinner(f"Processing '{uploaded_resume.name}'..."):
                bot_instance = st.session_state.get("bot")
                if bot_instance:
                    success = bot_instance.add_resume_to_vectorstore(file_path)
                    if success:
                        st.sidebar.success(f"Resume '{uploaded_resume.name}' processed and added to knowledge base!")
                        st.session_state.processed_file_id = current_file_id # Mark as processed
                    else:
                        st.sidebar.error(f"Failed to process '{uploaded_resume.name}'.")
                        # Optionally allow re-processing by clearing the ID
                        # st.session_state.processed_file_id = None 
                else:
                    st.sidebar.error("Bot not initialized. Cannot process resume.")
# --- End Resume Upload Section ---


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