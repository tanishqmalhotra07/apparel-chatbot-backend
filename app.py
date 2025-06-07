import os
import json
import time # You might not need time if removing chat functions
from datetime import datetime # You might not need datetime if removing chat functions
import traceback
from flask import Flask, request, jsonify, session, Response # Remove session, Response if not needed
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from flask_cors import CORS

# --- Configuration ---
load_dotenv() # Load environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID") # Can be removed if not using Assistant directly here
# if not ASSISTANT_ID:
#     raise ValueError("ASSISTANT_ID not found in .env file.")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_very_secret_key_for_dev_only")
CORS(app) # Enable CORS for all routes

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ChromaDB setup (ensure this path matches your populate_chroma.py)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "apparel_products"

# Define the ChromaDB collection (ensure it exists and is populated)
try:
    product_collection = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection(name=collection_name)
    # Changed to get_or_create_collection for robustness on initial deploy
    print(f"Connected to ChromaDB collection '{collection_name}'. Contains {product_collection.count()} items.")
except Exception as e:
    print(f"Error connecting to ChromaDB collection '{collection_name}': {e}")
    print("Please ensure populate_chroma.py has been run successfully OR your data is mounted.")
    product_collection = None # Set to None if collection not found


# ... (Your VALID_GENDERS, VALID_MASTER_CATEGORIES, etc. remain here) ...

# --- Helper Function for Apparel Search (NO CHANGES NEEDED HERE) ---
def find_apparel(user_query: str, gender: str = None, master_category: str = None,
                 subcategory: str = None, color: str = None, season: str = None,
                 sleeve_length: str = None, item_length: str = None,
                 category: str = None):
    # ... (Your existing find_apparel logic remains exactly as is) ...
    # ... (It correctly handles individual parameters) ...
    return {"products": found_products} # Ensure it always returns a dict with 'products' list

# --- NEW: API Endpoint for Linromi to call directly ---
@app.route('/api/find_apparel', methods=['POST'])
def find_apparel_api():
    try:
        request_data = request.json
        print(f"Received API call with data: {request_data}")

        if not request_data:
            return jsonify({"error": "Request must be JSON"}), 400

        user_query = request_data.get('user_query')
        if not user_query:
            return jsonify({"error": "Missing 'user_query' in request"}), 400

        # Extract filters, defaulting to an empty dict if not present
        filters = request_data.get('filters', {})

        # Call your find_apparel function with the unpacked arguments
        # It's crucial that find_apparel expects separate arguments, not a single 'filters' dict
        search_results = find_apparel(
            user_query=user_query,
            gender=filters.get('gender'),
            master_category=filters.get('master_category'),
            subcategory=filters.get('subcategory'),
            color=filters.get('color'),
            season=filters.get('season'),
            sleeve_length=filters.get('sleeve_length'),
            item_length=filters.get('item_length'),
            category=filters.get('category')
        )

        return jsonify(search_results), 200

    except Exception as e:
        print(f"Error in /api/find_apparel: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)