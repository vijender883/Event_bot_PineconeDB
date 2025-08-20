# Deployment Guide

This guide provides general advice and considerations for deploying the RAG Event Bot to a cloud platform.

## 1. Deployment Overview

Deploying this application involves two main components that must be running:

1.  **The Streamlit Web Application (`app.py`)**: This is the user-facing chatbot interface.
2.  **The WebSocket Server (`websocket_server.py`)**: This server is responsible for sending real-time system messages to the application.

You will need a hosting solution that can run Python applications and, ideally, support long-running processes or background tasks for the WebSocket server.

## 2. Handling Environment Variables

Your `.env` file containing your API keys and secrets should **not** be committed to your Git repository. When deploying your application, you will need to set these environment variables in your hosting provider's dashboard.

Most cloud platforms provide a way to set "secrets" or "config vars" that are securely injected into your application's environment at runtime. Refer to your provider's documentation for how to do this.

The required environment variables are:
*   `GEMINI_API_KEY`
*   `PINECONE_API_KEY`
*   `PINECONE_INDEX`
*   `BUCKET_NAME`
*   `AWS_ACCESS_KEY_ID`
*   `AWS_SECRET_ACCESS_KEY`
*   `AWS_DEFAULT_REGION`

## 3. Choosing a Deployment Platform

Here are a few popular options for deploying Streamlit applications.

### A. Streamlit Community Cloud

[Streamlit Community Cloud](https://streamlit.io/cloud) is one of the easiest ways to deploy public Streamlit apps directly from your GitHub repository.

*   **Pros**: Free for public apps, easy to set up, and well-integrated with Streamlit.
*   **Cons**: May not be suitable for running the WebSocket server as a separate process. You might need a different solution for that part of the application.

### B. Heroku

[Heroku](https://www.heroku.com/) is a Platform-as-a-Service (PaaS) that makes it easy to deploy, manage, and scale applications.

*   **Pros**: Supports Python, has a free tier, and can run multiple processes using a `Procfile`.
*   **Cons**: Can become expensive as you scale.

To deploy on Heroku, you would typically create a `Procfile` in your project root with content like this:

```
web: streamlit run app.py
worker: python websocket_server.py
```
This tells Heroku to run both the web app and the WebSocket server.

### C. AWS (Amazon Web Services)

Deploying on AWS offers the most flexibility and scalability but also requires more configuration.

*   **Pros**: Highly scalable, powerful, and flexible.
*   **Cons**: Can be complex to set up.

You could use services like:
*   **AWS Elastic Beanstalk**: A PaaS similar to Heroku.
*   **Amazon EC2**: A virtual server where you have full control over the environment.
*   **AWS App Runner**: A fully managed service for containerized web applications.

## 4. WebSocket Server Considerations

The WebSocket server needs to be accessible to the Streamlit application. When deploying, you'll need to update the `ws_url` in `app.py` from `ws://localhost:8765` to the public URL of your deployed WebSocket server.

For example, if you deploy the WebSocket server to `wss://your-websocket-server.com`, you would need to change this line in `app.py`:

```python
# In app.py, inside the start_websocket_connection function
ws_url = "ws://your-websocket-server.com:8765" # Or wss:// for secure connections
```

This might require some code changes to dynamically set the URL based on the environment (e.g., development vs. production).

---

Deploying an application can be a complex process, and the best solution depends on your specific needs and budget. We recommend starting with a platform like Heroku or Streamlit Community Cloud to get a feel for the process.
