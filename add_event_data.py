import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Get environment variables
api_key = os.getenv("GEMINI_API_KEY")
mongodb_uri = os.getenv("MONGODB_URI")
mongodb_db_name = os.getenv("MONGODB_DB_NAME")
mongodb_collection = os.getenv("MONGODB_COLLECTION")
index_name = "vector_db_index"  # Changed from "default" to "vector_db_index"

# Check if environment variables are loaded
if not all([api_key, mongodb_uri, mongodb_db_name, mongodb_collection]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

print(f"Using database: {mongodb_db_name}")
print(f"Using collection: {mongodb_collection}")
print(f"Using index name: {index_name}")

# Event details - key facts about the event
event_key_facts = """
Build with AI Workshop Details:
Date: May 18, 2025
Location: ScaleOrange technologies, Masthan Nagar, Kavuri Hills, Madhapur, Hyderabad, Telangana, 500081, India
Workshop timing: 10:00 AM to 4:00 PM

Agenda:
10:00 - 11:00 AM: Hands On Workshop: Automation using Claude and MCP Server - Vijender P, Alumnx
11:00 - 12:00 PM: Hands On Workshop: Agentic AI - Jitendra Gupta (Google Developer Expert)
12:00 - 1:00 PM: Industry Connect Session - Ravi Babu, Apex Cura Healthcare
1:00 - 2:00 PM: Lunch
2:00 - 3:00 PM: Hands On Workshop: Build an Event Bot using RAG - Vishvas Dubey, TCS
3:00 - 3:30 PM: Industry Application of AI: Building Multi AI Agents - Surendranath Reddy, QAPilot
3:30 - 4:00 PM: Workshop: Building Multi AI Agents - Mahidhar, NexusHub

Lunch Details:
Lunch will be provided to all participants who have checked in at the venue.
It will be served in the Cafeteria on the 5th floor between 1:00 PM and 2:00 PM IST.
Please ensure you've completed the check-in process at the registration desk to be eligible.

Washroom Location:
As you enter the room, stay on the same side as the entrance and walk straight along the corridor. 
The washrooms are at the corner end of the room, right at the end of the passage.
"""

# Additional event information
event_full_description = """Build with AI – A Workshop in Collaboration with 
Google for Developers 
Hyderabad's developer community is invited to an experiential 1 day workshop on Artificial 
Intelligence (AI). This hands-on event, hosted in collaboration with Google for Developers, brings 
together industry experts, innovators, and aspiring AI practitioners. 
Who Should Attend - Developers & Engineers - Tech Professionals - Students & Recent Graduates - Entrepreneurs & Product Managers 

Announcing the AI Hackathon: Build the Event Bot 
Do you love solving real-world problems with AI? Join us for an exciting AI Hackathon where 
you'll build a smart assistant called Event Bot to enhance the participant experience during our 
upcoming workshop. 
What's the Challenge? 
Your mission is to create an AI powered Event Bot that helps workshop participants with 
useful, real-time information. The bot will be powered in part by resumes of registered attendees 
and should be capable of answering the following: 
1. What's the meeting agenda? 
2. Who are the other participants who've worked on similar technical areas? (based 
on the resumes uploaded by the other participants) 
3. Which workshop sessions are relevant to the user? (resume-based 
recommendation) 
4. Where's the washroom? (simple location guidance) 
5. How much time is left until lunch? (real-time updates) 
6. Collect feedback after each session. (no forms—make it conversational!) 
This is your chance to design a context-aware, resume-smart, and user-friendly assistant 
that brings AI into the heart of event experience. 
Prizes 
Top 3 winners of the Hackathon will receive goodies. 
The First Prize winner's solution will be implemented live at the Workshop on 18th May 
2025, and the winner will also get an opportunity to present their solution to a live audience! 
Important Dates 
● Submission Deadline: 11:59 PM, 15th May 2025, 
● Evaluation: 16th, 17th May 2025 
● Winner Announcement:18th May 2025 
Submission Requirements 
● A fully working prototype addressing all or most of the 6 points above 
● Participants must submit the working solutions ONLY using Google Gemini. 
● You can submit as an Individual or in a Team of up to 5 members 
● Submit your entries at 
https://docs.google.com/forms/d/e/1FAIpQLSfboOPGcNwvXKxonjvPchucRN_hSA51Eei
oPWZBSRtmkj-8Pw/viewform?usp=sharing 
If you have any questions, please reach out to the contact details mentioned below. 
● Submission should include: 
○ GitHub link with README.md explaining the solution and architecture 
○ Link to the deployed working solution 
○ Short demo video (optional but encouraged) 
Need Help with the AI Hackathon? 
We're here to support you! If you have questions or need guidance, feel free to reach out via 
support@alumnx.com or call +91 9346139833. 
Interesting Projects in AI, ML using Gemini 
Are you exploring the frontiers of AI and ML? Have you built something cool using Google 
Gemini? Here's your chance to showcase your innovation and get recognized! 

What's the Opportunity? 
We're inviting developers, students, and AI enthusiasts to submit their most interesting 
projects built using Google Gemini. Whether it's a smart chatbot, a recommendation system, 
or a unique data-driven application—if it's built with Gemini, we want to see it! 

Top 5 submissions will get an exclusive opportunity to present their projects live during 
the upcoming workshop! 
Only 2 members per team will be allowed to make the presentation at the event. 

Project Criteria 
Your project must utilize Google Gemini in a meaningful way. The more impactful, innovative, 
and well-executed your idea is, the better your chances! 

Why Participate? 
● Get a platform to present your work in front of an engaged tech audience 
● Get feedback from mentors and industry professionals 
● Inspire others with your creativity and skills 
● Boost your portfolio and visibility 
Important Dates 

Submission Deadline: 11:59 PM IST, 15th May 2025 

Shortlisting & Communication: 16th–17th May 2025 

Presentation at Workshop: 18th May 2025 

How to Submit? 
Fill out the submission form here: 
https://docs.google.com/forms/d/e/1FAIpQLSeTahbGLCCCF8FJHYpl2rCLVmlP7sXXvoDV4-17d
q39XEDwKw/viewform?usp=sharing 

Team Size: Individual or team (max 5 members). Only 2 members per team can present at 
the workshop. 
Need Help with submitting your projects for "Interesting Projects in AI, ML using 
Gemini" ? 
Drop us a message at support@alumnx.com or call us at +91 9346139833. 

Time to Shine with Gemini! 
This isn't just a submission—it's your chance to put your Gemini-powered project on stage 
and inspire others in the AI community. 
Whether you're working on large language models, creative AI tools, or intelligent assistants, 
this is the platform to showcase your skills. 
Build something amazing. Submit it. Get featured. 
Let's see what you can do with Gemini!"""

def verify_mongodb_connection(client):
    """Verify that we can connect to MongoDB"""
    try:
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

def verify_vector_search_index(collection, index_name):
    """Verify that the vector search index exists"""
    try:
        indexes = collection.list_indexes()
        index_exists = False
        for index in indexes:
            if index.get('name') == index_name:
                index_exists = True
                break
        
        if index_exists:
            print(f"✅ Vector search index '{index_name}' exists!")
        else:
            print(f"❌ Vector search index '{index_name}' NOT found. You may need to create it in MongoDB Atlas.")
            print("Go to your MongoDB Atlas cluster > Collections > Search tab > Create Index")
            print("Use the following configuration for your index:")
            print("""
            {
              "mappings": {
                "dynamic": true,
                "fields": {
                  "embedding": {
                    "dimensions": 768,
                    "similarity": "cosine",
                    "type": "knnVector"
                  }
                }
              }
            }
            """)
        
        return index_exists
    except Exception as e:
        print(f"❌ Error checking vector search index: {e}")
        return False

def main():
    print("Connecting to MongoDB...")
    # Initialize MongoDB client
    mongo_client = MongoClient(mongodb_uri)
    db = mongo_client[mongodb_db_name]
    collection = db[mongodb_collection]
    
    # Verify MongoDB connection
    if not verify_mongodb_connection(mongo_client):
        print("Exiting due to MongoDB connection issues.")
        return
    
    # Verify vector search index exists
    verify_vector_search_index(collection, index_name)
    
    # Clear existing data (optional, comment out if you want to keep existing data)
    print("Clearing existing data from collection...")
    collection.delete_many({})
    
    # Initialize the embedding model
    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Split the text into chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better retrieval
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )