from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv
from pymongo import MongoClient
from utils import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
db = mongo_client["faq_app"]
logs_collection = db["query_logs"]

# Initialize Flask App
app = Flask(__name__)

def rag(user_query):
    try:
        text = get_pdf_text(['Tharun_Data_Scientist_Resume.pdf'])
        chunks = get_text_chunks(text)
        retrived_vectors = get_vector_store(chunks)
        response = get_conversational_chain(user_query, retrived_vectors)
        return response
    except Exception as e:
        return {"error": str(e)}  # Return the error as a dictionary

# Home Route
@app.route("/")
def home():
    return render_template("index.html")  # Ensure you have an index.html file for the frontend

# Query Endpoint for user questions
@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query")
    
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    # Get the response from the LLM model
    response = rag(user_query)
    
    # If the response contains an error, return it
    if isinstance(response, dict) and "error" in response:
        return jsonify(response), 500  # Return error as JSON with status 500
    
    # Log the query and response to MongoDB
    logs_collection.insert_one({"query": user_query, "response": response})
    
    # Return the response in JSON format
    return jsonify({"response": response})

# Admin: View all logs (queries and their responses)
@app.route("/logs", methods=["GET"])
def view_logs():
    logs = list(logs_collection.find({}, {"_id": 0}))  # Exclude MongoDB _id from logs
    return jsonify(logs)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
