# Intelligent FAQ Assistant using Flask and RAG

### Overview:
This project is a Flask web application that integrates a Retrieval-Augmented Generation (RAG)-based approach, leveraging LangChain, Google Gemini, and FAISS for creating an intelligent FAQ assistant. The application is designed to dynamically answer user queries based on a pre-loaded knowledge base and includes features for updating the knowledge base, logging user interactions, and providing fallback responses for unclear queries.

### Key Features:
1. Knowledge Base Integration
    Pre-load the assistant with structured/unstructured text files (e.g., JSON, PDF, TXT, etc.).
    Use FAISS for vector-based storage and similarity search.
    Periodic updates to the knowledge base via an admin interface or API.
   
3. Dynamic Query Handling
    Leverage LangChain to create a pipeline that combines the retrieval step (using FAISS) with generative capabilities of Google Gemini.
    Generate contextually relevant responses dynamically by searching the knowledge base for similar queries/documents and feeding relevant information to the LLM.

4. Flask Application Features
    Frontend:
        Query input box for user interaction.
        Response display area to show answers.
    API Endpoint:
        Expose a /ask endpoint to accept user queries programmatically.

5. Admin Features
    Admin interface to:
        Update the knowledge base.
        View interaction logs and analyze user queries and responses.

6. Interaction Logging
    Log all user interactions (queries and responses) into a MongoDB database for analysis and improvement.

7. Fallback Mechanism
    Handle cases where no clear answer is found by:
        Returning a polite fallback response (e.g., "I'm sorry, I couldn't find an answer. Please check our documentation.")

### Technologies Used:
  Backend: Flask
  Frontend: HTML, CSS, JavaScript
  LLM Framework: LangChain, Google Gemini
  Vector Store: FAISS (Facebook AI Similarity Search)
  Database: MongoDB (for logging)
  Other Libraries: Python libraries such as Flask-Cors, Flask-PyMongo, and PyPDF2
   
