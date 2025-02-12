import os
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow API requests from Softr

# OpenAI & Pinecone API Keys
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def generate_query_embedding(query):
    """Generate an embedding for a search query using OpenAI."""
    client = openai.OpenAI()
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

@app.route("/search", methods=["POST"])
def search_documents():
    """API Endpoint: Search for relevant documents."""
    data = request.get_json()
    query = data.get("query")
    permission_level = data.get("permission_level", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    query_embedding = generate_query_embedding(query)

    # Query Pinecone for results
    search_results = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True
    )

    # Filter based on score threshold
    min_score_threshold = 0.75
    filtered_results = [
        {
            "score": match["score"],
            "document": match["metadata"].get("document", "No URL available")
        }
        for match in search_results["matches"]
        if match["score"] >= min_score_threshold
    ]

    return jsonify({"results": filtered_results})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)





