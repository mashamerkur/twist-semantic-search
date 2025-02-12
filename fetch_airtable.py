import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Airtable API credentials
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")
AIRTABLE_TEXT_COLUMN = os.getenv("AIRTABLE_TEXT_COLUMN")
AIRTABLE_DOC_COLUMN = os.getenv("AIRTABLE_DOC_COLUMN")

# Airtable API URL
AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME.replace(' ', '%20')}"

# Headers for API request
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}"
}

def fetch_airtable_data():
    """Fetches data from Airtable and returns a DataFrame."""
    records = []
    offset = None

    while True:
        params = {"offset": offset} if offset else {}
        response = requests.get(AIRTABLE_URL, headers=HEADERS, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.text}")
            return None

        data = response.json()
        for record in data.get("records", []):
            fields = record.get("fields", {})
            records.append({
                "id": record["id"],
                "extracted_text": fields.get(AIRTABLE_TEXT_COLUMN, ""),
                "original_document": fields.get(AIRTABLE_DOC_COLUMN, "")
            })

        offset = data.get("offset")  # Check if there are more records
        if not offset:
            break

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = fetch_airtable_data()
    if df is not None:
        print(df.head())  # Display first few rows
