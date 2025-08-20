# Build with AI - RAG Event Bot

Welcome to the RAG Event Bot! This is an interactive, AI-powered chatbot designed to assist attendees at events like workshops, hackathons, and conferences. It provides real-time information about the event, can be personalized with your resume, and even helps with event check-in.

This project is built to be easy to set up and run, even for those new to GitHub or deploying applications.

![Event Bot Demo](https://placehold.co/800x400/d3d3d3/000000/png?text=Event+Bot+Screenshot+Here)

## ✨ Features

*   **AI-Powered Q&A:** Ask questions in natural language about the event agenda, speakers, locations, and more.
*   **RAG Architecture:** The bot uses a Retrieval-Augmented Generation (RAG) pipeline with Google's Gemini model to provide answers based on a specific knowledge base.
*   **Resume Analysis:** Upload your PDF resume, and the bot can answer questions about your skills and experience.
*   **Event Check-In:** A built-in feature to check-in attendees by verifying their registration via an API.
*   **Real-time System Messages:** A websocket connection allows event organizers to send real-time announcements to all users.
*   **Chat History:** For checked-in users, the bot remembers your conversation history.

## 🛠️ Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **Language:** Python
*   **AI/ML:** [LangChain](https://www.langchain.com/), [Google Gemini](https://deepmind.google/technologies/gemini/), [Pinecone](https://www.pinecone.io/) (Vector Store)
*   **Data Storage:** [AWS S3](https://aws.amazon.com/s3/) (for resumes)
*   **Real-time Communication:** WebSockets

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

Make sure you have the following installed:
*   [Python 3.8+](https://www.python.org/downloads/)
*   [Git](https://git-scm.com/downloads/)
*   A code editor (e.g., [VS Code](https://code.visualstudio.com/))

### 2. Installation & Setup

**A. Fork and Clone the Repository**

First, fork this repository to your own GitHub account, and then clone it to your local machine.

```bash
git clone https://github.com/YOUR_USERNAME/RAG-Event-Bot.git
cd RAG-Event-Bot
```

*(For a detailed, beginner-friendly guide on forking, see our [Forking and PR Guide](./docs/fork-and-pr-guide.md)).*

**B. Create a Virtual Environment**

It's a best practice to create a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**C. Install Dependencies**

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

**D. Set Up Environment Variables**

This project requires several API keys and configuration settings. Create a file named `.env` in the root of the project directory and add the following variables:

```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
PINECONE_INDEX="your-pinecone-index-name"
BUCKET_NAME="your-s3-bucket-name"
AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
AWS_DEFAULT_REGION="your-aws-region" # e.g., us-east-1
```

*   **`GEMINI_API_KEY`**: Your API key for Google Gemini.
*   **`PINECONE_API_KEY` / `PINECONE_INDEX`**: Your Pinecone API key and the name of the index you want to use.
*   **`BUCKET_NAME` / `AWS_*`**: Your AWS S3 bucket name and credentials for storing uploaded resumes.

### 3. Running the Application

This application has two parts that need to be running at the same time: the Streamlit web app and the WebSocket server.

**A. Run the WebSocket Server**

Open a terminal and run the following command to start the WebSocket server. This server is used to send real-time system messages to the app.

```bash
python websocket_server.py
```

You should see a message indicating the server has started.

**B. Run the Streamlit App**

Open a *second* terminal, make sure your virtual environment is activated, and run the following command to start the main application.

```bash
streamlit run app.py
```

Your web browser should open with the application running.

## 🤝 How to Contribute

We welcome contributions! If you'd like to help improve the Event Bot, please see our **[Contributing Guide](./docs/fork-and-pr-guide.md)** for detailed instructions on how to submit your changes.

## ☁️ Deployment

Thinking about deploying this application to the cloud? We've put together some general guidelines and tips to help you get started.

See our **[Deployment Guide](./docs/deployment-guide.md)** for more information.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
