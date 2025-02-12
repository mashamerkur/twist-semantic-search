import os
import openai
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from fetch_airtable import fetch_airtable_data

# Load environment variables from .env
load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pinecone API Credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists
if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    print(f"Index '{PINECONE_INDEX_NAME}' not found. Please create it first.")
    exit()

index = pc.Index(PINECONE_INDEX_NAME)

def generate_embedding(text):
    """Generate an embedding using OpenAI's new API format."""
    client = openai.OpenAI()
    response = client.embeddings.create(
        input=[text],  # OpenAI v1.0+ requires a list
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def store_embeddings():
    """Fetches Airtable data, generates embeddings, and stores in Pinecone."""
    df = fetch_airtable_data()
    
    if df is None or df.empty:
        print("No data found in Airtable.")
        return
    
    # Prepare data for Pinecone
    vectors = []
    for _, row in df.iterrows():
        text = row["extracted_text"]
        if not text.strip():
            continue  # Skip empty texts
        
        embedding = generate_embedding(text)

        # Extract the first URL from the document field
        document_data = row["original_document"]
        if isinstance(document_data, list) and len(document_data) > 0:
            document_url = document_data[0].get("url", "")  # Get only the URL
        else:
            document_url = str(document_data)  # Ensure it's a string

        vectors.append((row["id"], embedding, {"document": document_url}))

    # Store in Pinecone
    if vectors:
        index.upsert(vectors=vectors)
        print("✅ Embeddings stored successfully in Pinecone.")
    else:
        print("⚠️ No valid text data to store.")

if __name__ == "__main__":
    store_embeddings()
