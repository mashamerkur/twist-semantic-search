import os
import openai
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pinecone API Credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def generate_query_embedding(query):
    """Generate an embedding for a search query using OpenAI."""
    client = openai.OpenAI()
    response = client.embeddings.create(
        input=[query],  # Query must be a list
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def search_documents(query, top_k=10, min_score_threshold=0.75):
    """Search Pinecone for relevant documents with a minimum score threshold."""
    query_embedding = generate_query_embedding(query)

    # Query Pinecone index
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,  # Get up to 10 results
        include_metadata=True
    )

    # Filter results based on similarity score
    if "matches" in search_results and len(search_results["matches"]) > 0:
        filtered_results = [m for m in search_results["matches"] if m["score"] >= min_score_threshold]

        if len(filtered_results) == 0:
            print("⚠️ No relevant documents found. Try rephrasing your search.")
            return

        print("\n🔍 **Search Results:**")
        for match in filtered_results:
            score = match["score"]
            document_url = match["metadata"].get("document", "No URL available")
            print(f"- Score: {score:.4f}, Document: {document_url}")
    else:
        print("⚠️ No relevant documents found.")

if __name__ == "__main__":
    query = input("\nEnter your search query: ")
    search_documents(query)
