import streamlit as st
import os
import html
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        self.prompt_template = """
        You are a friendly Event Information Assistant. Your primary purpose is to answer questions about the event described in the provided context. Follow these guidelines:

        1. You can respond to basic greetings like "hi", "hello", or "how are you" in a warm, welcoming manner
        2. For event information, only provide details that are present in the context
        3. If information is not in the context, politely say "I'm sorry, I don't have that specific information about the event"
        4. Keep responses concise but conversational
        5. Do not make assumptions beyond what's explicitly stated in the context
        6. Always prioritize factual accuracy while maintaining a helpful tone
        7. Do not introduce information that isn't in the context
        8. If unsure about any information, acknowledge uncertainty rather than guess
        9. You may suggest a few general questions users might want to ask about the event
        10. Remember to maintain a warm, friendly tone in all interactions
        11. You should refer to yourself as "Event Bot"
        12. You should not greet if the user has not greeted to you

        Remember: While you can be conversational, your primary role is providing accurate information about this specific event based on the context provided.

        Context information about the event:
        {context}
        --------

        Now, please answer this question about the event: {question}
        """


    def post_process_response(self, response, query):
        """Format responses for better readability based on query type."""
        # If it's a lunch-related query, format the response
        if "lunch" in query.lower() or "food" in query.lower() or "eat" in query.lower():
            # Just start with "Regarding lunch:" without the greeting
            formatted = "Regarding lunch:\n\n"

            # Split into readable bullet points
            points = []

            # Extract key information using common phrases and format as separate points
            if "provided to all" in response:
                points.append("• Lunch will be provided to all participants who have checked in at the venue.")
            if "cafeteria" in response.lower() and "floor" in response.lower():
                # Extract time info if available
                time_info = ""
                if "1:00" in response and "2:00" in response:
                    time_info = "between 1:00 PM and 2:00 PM IST"
                points.append(f"• It will be served in the Cafeteria on the 5th floor {time_info}.")
            if "check-in" in response.lower() or "registration" in response.lower():
                points.append("• Please ensure you've completed the check-in process at the registration desk to be eligible.")
            if "volunteer" in response.lower() or "direction" in response.lower():
                points.append("• Feel free to ask a volunteer if you need directions to the cafeteria.")

            # If we couldn't extract structured points, just use the original
            if not points:
                return response

            # Combine all points with line breaks
            return formatted + "\n".join(points)

        # For other responses, just return the original
        return response

    def answer_question(self, query):
        """Use RAG with Google Gemini to answer a question based on retrieved context,
        and measure component response times."""
        vector_db_time = 0
        llm_time = 0
        raw_response = ""
        processed_response_text = "" # Initialize processed response text

        try:
            # Using Google embeddings
            embedding_function = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )

            # Retrieve relevant documents from Pinecone using the new SDK
            index = self.pc.Index(self.index_name)

            # Create vector store from the existing index
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=embedding_function,
                text_key="text"  # Adjust this field if your schema uses a different key
            )

            with st.spinner("Retrieving relevant information..."):
                # Perform similarity search and measure time
                start_time = time.time()
                results = vectorstore.similarity_search_with_score(query, k=5)
                end_time = time.time()
                vector_db_time = end_time - start_time

                context_text = "\n\n --- \n\n".join([doc.page_content for doc, _score in results])

                # Format the prompt
                prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
                prompt = prompt_template.format(context=context_text, question=query)

                # Create the content for Gemini
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                        ],
                    ),
                ]

            with st.spinner("Generating response..."):
                # Generate response using Gemini 2.0 Flash model and measure time
                start_time = time.time()
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",  # Using Gemini 2.0 Flash model
                    contents=contents,
                )
                end_time = time.time()
                llm_time = end_time - start_time

                raw_response = response.text
                processed_response_text = self.post_process_response(raw_response, query)

            # Return a dictionary including the text and timing information
            return {
                "text": processed_response_text,
                "vector_db_time": vector_db_time,
                "llm_time": llm_time
            }

        except Exception as e:
            # Return error message and zero times in case of error
            return {
                "text": f"An error occurred: {str(e)}",
                "vector_db_time": vector_db_time,
                "llm_time": llm_time
            }


# Set page configuration
st.set_page_config(
    page_title="Build with AI - RAG Event Bot",
    page_icon="🚀",
    layout="centered"
)

# Load the CSS, matching the styles from the Chroma version
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

.bot-message ol {
    margin-top: 8px !important;
    margin-bottom: 8px !important;
    padding-left: 25px !important;
}

.bot-message li {
    margin-bottom: 6px !important;
    padding-bottom: 0 !important;
    line-height: 1.4 !important;
}

.bot-message p {
    margin-bottom: 10px !important;
}

.bot-message-content {
    /* Styles for the main text content */
    flex-grow: 1 !important; /* Allows content to take available space */
    margin-bottom: 5px !important; /* Space between content and timings */
    line-height: 1.5 !important;
    white-space: pre-line !important;
}


/* Chat container and message styles */
.custom-chat-container {
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
    margin-bottom: 20px !important;
    max-width: 800px !important;
    background-color: white !important;
}

.message-container {
    display: flex !important;
    align-items: flex-start !important;
    margin-bottom: 10px !important;
    background-color: white !important; /* White background, same as container */
    color: black !important;
}

.message-container.user {
    flex-direction: row-reverse !important;
}

.avatar-icon {
    width: 36px !important;
    height: 36px !important;
    border-radius: 50% !important;
    background-color: #E8F0FE !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    font-size: 20px !important;
    margin: 0 10px !important;
    flex-shrink: 0 !important;
}

.user-avatar-icon {
    background-color: #F0F2F5 !important;
}

.user-message {
    background-color: #F0F2F5 !important;
    padding: 10px 15px !important;
    border-radius: 18px !important;
    max-width: 80% !important;
    margin-right: 10px !important;
    word-wrap: break-word !important;
}


.bot-message-timings {
    font-size: 0.75em !important; /* Smaller font */
    color: #555 !important; /* Darker gray */
    margin-top: 5px !important; /* Add spacing */
    display: block !important; /* Ensure it's on its own line */
    align-self: flex-end !important; /* Align timings to the right */
}

div.custom-chat-container {
    border-radius: 15px;
    border: 1px solid #ccc; /* Optional border */
    padding: 10px;          /* Optional padding */
}
</style>
""", unsafe_allow_html=True)

# Main app title
st.title("Build with AI - RAG Event Bot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get API keys from environment variables
api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
pinecone_index = os.getenv("PINECONE_INDEX")

if not api_key:
    st.error("API key not found in .env file. Please add GEMINI_API_KEY to your .env file.")
    st.stop()

if not pinecone_api_key or not pinecone_index:
    st.error("Pinecone credentials not found in .env file. Please add PINECONE_API_KEY and PINECONE_INDEX to your .env file.")
    st.stop()

# Initialize the bot
if "bot" not in st.session_state:
    with st.spinner("Initializing assistant..."):
        st.session_state.bot = EventAssistantRAGBot(
            api_key=api_key,
            pinecone_api_key=pinecone_api_key,
            pinecone_cloud=pinecone_cloud,
            pinecone_region=pinecone_region,
            index_name=pinecone_index
        )

    # Add welcome message with options - Store as a dictionary like in Chroma version
    if not st.session_state.messages:
        welcome_message_text = """Hello! I'm Event bot.
I can help you with the following:
1. Agenda of the "Build with AI" workshop
2. Important Dates of this workshop
3. Details of the AI Hackathon
4. Presentation of Interesting projects in AI, ML
5. Locating the washrooms
6. Details of lunch at the venue

How can I help you with information about this event?"""

        # Store welcome message as a dictionary with None for timings
        st.session_state.messages.append(
            {"role": "assistant", "content": {"text": welcome_message_text, "vector_db_time": None, "llm_time": None}}
        )

# Custom Chat UI Implementation - Matching Chroma version
chat_html = '<div class="custom-chat-container">'

# Add all messages to the custom chat HTML
for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = '<div class="avatar-icon user-avatar-icon">👤</div>'
        chat_html += f'<div class="message-container user">'
        chat_html += avatar
        # User messages are simple text
        chat_html += f'<div class="user-message">{html.escape(message["content"])}</div>'
        chat_html += '</div>'
    else:  # assistant
        avatar = '<div class="avatar-icon">🤖</div>'
        # We apply the bot-message class to the INNER div, which contains content and timings
        chat_html += f'<div class="message-container">'
        chat_html += avatar

        # Access the content dictionary
        content_dict = message["content"]
        content_text = content_dict["text"]
        vector_db_time = content_dict.get("vector_db_time")
        llm_time = content_dict.get("llm_time")

        # Start the inner bot message div that holds both content and timings
        chat_html += '<div class="bot-message">'

        # Format message content - Special handling for the welcome message
        # Check if it's the welcome message by checking for the list items
        if "I can help you with the following:" in content_text and "1." in content_text:
             # For the welcome message - Use the existing welcome_html logic
            # Note: Escaping is not done here for the welcome message's structured HTML parts
            welcome_html = content_text.replace(
                "Hello! I'm Event bot.\nI can help you with the following:",
                "Hello! I'm Event bot.<br><br>I can help you with the following:"
            )
            welcome_html = welcome_html.replace('\n1. ', "<ol style='margin-top:8px;margin-bottom:8px;padding-left:25px;'><li style='margin-bottom:4px;'>")
            welcome_html = welcome_html.replace('\n2. ', "</li><li style='margin-bottom:4px;'>")
            welcome_html = welcome_html.replace('\n3. ', "</li><li style='margin-bottom:4px;'>")
            welcome_html = welcome_html.replace('\n4. ', "</li><li style='margin-bottom:4px;'>")
            welcome_html = welcome_html.replace('\n5. ', "</li><li style='margin-bottom:4px;'>")
            welcome_html = welcome_html.replace('\n6. ', "</li><li style='margin-bottom:4px;'>")
            welcome_html = welcome_html.replace('\n\nHow can I help you', "</li></ol><br>How can I help you")

            chat_html += '<div class="bot-message-content">' + welcome_html + '</div>'

        else:
            # For regular messages - escape and format text content
            escaped_content = html.escape(content_text)
            formatted_content = escaped_content.replace('\n', '<br>')

            chat_html += f'<div class="bot-message-content">{formatted_content}</div>'

        # Add timings if available (not for the welcome message which has None)
        if vector_db_time is not None and llm_time is not None:
             timings_html = f'<span class="bot-message-timings">Vector DB: {vector_db_time:.2f}s | LLM: {llm_time:.2f}s</span>'
             chat_html += timings_html

        # Close the inner bot message div
        chat_html += '</div>' # Closes bot-message

        chat_html += '</div>' # Closes message-container

# Close the container div
chat_html += '</div>'

# Render the custom chat container
st.markdown(chat_html, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask a question about the event...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response (this now returns a dict)
    response_dict = st.session_state.bot.answer_question(user_input)

    # Add assistant response (the dict) to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_dict})

    # Rerun to update the UI
    st.rerun()