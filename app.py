import os
import json
import time # Keep time, it might be used internally for delays, although less crucial without streaming
# from datetime import datetime # No longer needed if chat streaming is removed
import traceback
from flask import Flask, request, jsonify # session, Response removed as chat streaming is gone
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from flask_cors import CORS

# --- Configuration ---
load_dotenv() # Load environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# ASSISTANT_ID is no longer strictly needed in app.py if it's only an API endpoint
# for Linromi to call after Linromi handles OpenAI Assistant interaction.
# Keeping it commented out for clarity.
# ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
# if not ASSISTANT_ID:
#     raise ValueError("ASSISTANT_ID not found in .env file.")

app = Flask(__name__)
# app.secret_key is no longer strictly needed if Flask sessions are not used.
# Keeping it as it's harmless.
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_very_secret_key_for_dev_only")
CORS(app) # Enable CORS for all routes

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ChromaDB setup (ensure this path matches your populate_chroma.py)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "apparel_products"

# Define the ChromaDB collection (ensure it exists and is populated)
try:
    # Use get_or_create_collection for robustness on initial deploy
    product_collection = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection(name=collection_name)
    print(f"Connected to ChromaDB collection '{collection_name}'. Contains {product_collection.count()} items.")
except Exception as e:
    print(f"Error connecting to ChromaDB collection '{collection_name}': {e}")
    print("Please ensure populate_chroma.py has been run successfully OR your data is mounted.")
    product_collection = None # Set to None if collection not found

# --- Valid Enum Values (Must match your products.json and tool definition) ---
# These are used for validation and constructing filters
VALID_GENDERS = {"male", "female", "unisex"}
VALID_MASTER_CATEGORIES = {"top", "bottom", "accessory", "footwear", "top & foot combined"}
VALID_SUBCATEGORIES = {
    "shirt", "t-shirt", "polo shirt", "dress", "gown", "shorts", "jeans",
    "sweater", "top", "flats", "heels", "sneakers", "boots", "sandals",
    "jewelry", "bag", "hat", "scarf", "belt", "sunglasses", "N/A" # Include N/A if used in data
}
VALID_SEASONS = {"summer", "winter", "spring", "fall", "all-season"}
VALID_SLEEVE_LENGTHS = {"full sleeve", "half sleeve", "quarter sleeve", "sleeveless", "strapless", "N/A"}
VALID_ITEM_LENGTHS = {"mini", "short", "knee-length", "midi", "maxi", "full length", "N/A"}
VALID_CATEGORIES = {"dresses", "shirts", "jeans", "tops", "footwear", "accessories", "sweaters", "shorts", "pants"}

# --- Tool Definitions (This part should match your *Python function signature*) ---
# This TOOLS definition is what your `app.py` would use if it were internally calling OpenAI Assistants.
# It should match the arguments that your `find_apparel` Python function expects directly (flattened).
# The Linromi side's JSON schema (for data collection) is what dictates what Linromi *sends*.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_apparel",
            "description": "Finds clothing and apparel products based on user query and filters. Provides detailed filtering capabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "A concise, descriptive natural language query summarizing the user's apparel request, including desired style, occasion, or specific features (e.g., 'stylish dress for a summer wedding', 'comfortable running shoes', 'vibrant shirt'). This is crucial for semantic search."
                    },
                    "gender": { # Directly here, NOT nested under 'filters' for Python signature
                        "type": "string",
                        "enum": list(VALID_GENDERS),
                        "description": "Filter by gender (male, female, unisex). This is a hard filter."
                    },
                    "master_category": { # Directly here
                        "type": "string",
                        "enum": list(VALID_MASTER_CATEGORIES),
                        "description": "Filter by broad clothing category (top, bottom, accessory, footwear, top & foot combined)."
                    },
                    "subcategory": { # Directly here
                        "type": "string",
                        "enum": list(VALID_SUBCATEGORIES),
                        "description": "Filter by specific clothing subcategory (e.g., shirt, t-shirt, dress, jeans, heels, etc.)."
                    },
                    "color": { # Directly here
                        "type": "string",
                        "description": "Filter by primary color (e.g., red, blue, black, white, multi-color). This is a soft filter, semantic search will help find similar colors."
                    },
                    "season": { # Directly here
                        "type": "string",
                        "enum": list(VALID_SEASONS),
                        "description": "Filter by season (summer, winter, spring, fall, all-season). This is a hard filter."
                    },
                    "sleeve_length": { # Directly here
                        "type": "string",
                        "enum": list(VALID_SLEEVE_LENGTHS),
                        "description": "Filter by sleeve length (full sleeve, half sleeve, quarter sleeve, sleeveless, strapless). Applies to tops/dresses."
                    },
                    "item_length": { # Directly here
                        "type": "string",
                        "enum": list(VALID_ITEM_LENGTHS),
                        "description": "Filter by item length (mini, short, knee-length, midi, maxi, full length). Applies to dresses/bottoms."
                    },
                    "category": { # Directly here
                        "type": "string",
                        "enum": list(VALID_CATEGORIES),
                        "description": "Filter by general product category (e.g., 'dresses', 'shirts', 'pants'). Broader than subcategory."
                    }
                },
                "required": ["user_query"]
            }
        }
    }
]


# --- Helper Function for Apparel Search ---
def find_apparel(user_query: str, gender: str = None, master_category: str = None,
                 subcategory: str = None, color: str = None, season: str = None,
                 sleeve_length: str = None, item_length: str = None,
                 category: str = None):
    """
    Finds apparel products using ChromaDB with a multi-stage filtering strategy.
    Gender and Season are treated as hard filters across all stages.
    Other filters (master_category, subcategory, color, length) are soft and relaxed
    in later stages if strict search yields no results.
    """
    if not product_collection:
        print("ChromaDB collection not initialized. Cannot perform search.")
        return {"products": [], "message": "ChromaDB not initialized. Please run populate_chroma.py."}

    print(f"Tool Call: find_apparel(user_query='{user_query}', gender='{gender}', master_category='{master_category}', subcategory='{subcategory}', color='{color}', season='{season}', sleeve_length='{sleeve_length}', item_length='{item_length}', category='{category}')")

    # Initialize found_products here (CRUCIAL FIX FOR NameError)
    found_products = [] 

    # Get query embedding
    try:
        query_embedding = client.embeddings.create(input=[user_query], model="text-embedding-3-small").data[0].embedding
    except Exception as e:
        print(f"Error creating embedding for query: {e}")
        traceback.print_exc()
        return {"products": [], "message": f"Error processing query: {e}"}

    # Helper to construct and clean filters
    # This matches the Python find_apparel function signature
    def get_chromadb_filters(g, mc, sc, c, s, sl, il, cat):
        individual_filters = []

        # --- GENDER LOGIC ---
        if g:
            g_lower = g.lower()
            if g_lower in ['male', 'female']:
                gender_or_filters = [
                    {"gender": g_lower},
                    {"gender": "unisex"}
                ]
                individual_filters.append({"$or": gender_or_filters})
            elif g_lower == 'unisex':
                individual_filters.append({"gender": "unisex"})

        if mc and mc.lower() in VALID_MASTER_CATEGORIES:
            individual_filters.append({"master_category": mc.lower()})

        # --- CATEGORY LOGIC ---
        if cat and cat.lower() in VALID_CATEGORIES:
            individual_filters.append({"category": cat.lower()})

        # --- SUBCATEGORY LOGIC (Strict for Stage 1, relaxed later by passing None) ---
        if sc and sc.lower() in VALID_SUBCATEGORIES and sc.lower() != "n/a":
            individual_filters.append({"subcategory": sc.lower()})

        if c:
            individual_filters.append({"color": {"$eq": c.lower()}})

        # --- SEASON LOGIC ---
        if s:
            s_lower = s.lower()
            if s_lower == "all-season":
                all_possible_seasons = [valid_s for valid_s in VALID_SEASONS if valid_s != "n/a"]
                season_or_filters = [{"season": single_s} for single_s in all_possible_seasons]
                individual_filters.append({"$or": season_or_filters})
            elif s_lower in VALID_SEASONS:
                individual_filters.append({"season": s_lower})

        if sl and sl.lower() in VALID_SLEEVE_LENGTHS and sl.lower() != "n/a":
            individual_filters.append({"sleeve_length": sl.lower()})
        if il and il.lower() in VALID_ITEM_LENGTHS and il.lower() != "n/a":
            individual_filters.append({"item_length": il.lower()})

        if not individual_filters:
            return {}
        elif len(individual_filters) == 1:
            return individual_filters[0]
        else:
            return {"$and": individual_filters}

    # --- Stage 1: Strict Search (All provided filters) ---
    current_filters = get_chromadb_filters(gender, master_category, subcategory, color, season, sleeve_length, item_length, category)
    print(f"Stage 1: Performing search with strict filters: {current_filters}")
    try:
        results = product_collection.query(
            query_embeddings=[query_embedding],
            n_results=10, # Increased n_results for better chances of finding more
            where=current_filters
        )

        print(f"\n--- DEBUG: Raw ChromaDB Query Results for current_filters: {current_filters} ---")
        print(f"Results metadata: {results.get('metadatas')}")
        print(f"Results documents: {results.get('documents')}")
        print(f"Results distances: {results.get('distances')}")
        print(f"-----------------------------------------\n")

        if results.get('metadatas') and results['metadatas'] and results['metadatas'][0]:
            found_products = results['metadatas'][0]
        else:
            found_products = [] # Explicitly set to empty list if no results

        if found_products:
            print(f"Stage 1: Found {len(found_products)} products with strict filters.")
            for i, product_meta in enumerate(found_products):
                print(f"  Found Product {i+1}: Name: {product_meta.get('name')}, ID: {product_meta.get('id')}, Master Category: {product_meta.get('master_category')}, Color: {product_meta.get('color')}, Gender: {product_meta.get('gender')}")
            return {"products": found_products}
        else:
            print("Stage 1: No products found with strict filters. Proceeding to relax.")
    except Exception as e:
        print(f"Error in Stage 1 search: {e}")
        traceback.print_exc()
        return {"products": [], "message": f"Error in Stage 1 search: {e}"}

    # --- Stage 2: Relax Subcategory, Color, and Item Length (Keep Gender, Master Category, Season, Category) ---
    current_filters = get_chromadb_filters(gender, master_category, None, None, season, sleeve_length, None, category)
    # Only proceed to Stage 2 if there are meaningful filters remaining or if we started with only query
    if current_filters or (not gender and not season and not master_category and not category):
        print(f"Stage 2: Performing search with relaxed subcategory, color, lengths. Filters: {current_filters}")
        try:
            results = product_collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                where=current_filters
            )
            print(f"\n--- DEBUG: Raw ChromaDB Query Results for relaxed_filters: {current_filters} ---")
            print(f"Results metadata: {results.get('metadatas')}")
            print(f"Results documents: {results.get('documents')}")
            print(f"Results distances: {results.get('distances')}")
            print(f"-----------------------------------------\n")

            if results.get('metadatas') and results['metadatas'] and results['metadatas'][0]:
                found_products = results['metadatas'][0]
            else:
                found_products = []

            if found_products:
                print(f"Stage 2: Found {len(found_products)} products with relaxed subcategory, color, lengths.")
                for i, product_meta in enumerate(found_products):
                    print(f"  Found Product {i+1}: Name: {product_meta.get('name')}, ID: {product_meta.get('id')}, Master Category: {product_meta.get('master_category')}, Color: {product_meta.get('color')}, Gender: {product_meta.get('gender')}")
                return {"products": found_products}
            else:
                print("Stage 2: No products found with relaxed subcategory, color, lengths. Proceeding to relax further.")
        except Exception as e:
            print(f"Error in Stage 2 search: {e}")
            traceback.print_exc()
            return {"products": [], "message": f"Error in Stage 2 search: {e}"}
    else:
        print("Stage 2: Skipping because no relevant gender, master_category, season, or category filters were provided for this stage.")


    # --- Stage 3: Relax Master Category, Category (Keep Gender and Season as hard filters) ---
    current_filters = get_chromadb_filters(gender, None, None, None, season, None, None, None)
    if current_filters: # Only proceed to Stage 3 if gender/season filters are present
        print(f"Stage 3: Performing search with only gender and season filters (if provided). Filters: {current_filters}")
        try:
            results = product_collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                where=current_filters
            )
            print(f"\n--- DEBUG: Raw ChromaDB Query Results for broad_filters: {current_filters} ---")
            print(f"Results metadata: {results.get('metadatas')}")
            print(f"Results documents: {results.get('documents')}")
            print(f"Results distances: {results.get('distances')}")
            print(f"-----------------------------------------\n")

            if results.get('metadatas') and results['metadatas'] and results['metadatas'][0]:
                found_products = results['metadatas'][0]
            else:
                found_products = []

            if found_products:
                print(f"Stage 3: Found {len(found_products)} products with only gender and season filters.")
                for i, product_meta in enumerate(found_products):
                    print(f"  Found Product {i+1}: Name: {product_meta.get('name')}, ID: {product_meta.get('id')}, Master Category: {product_meta.get('master_category')}, Color: {product_meta.get('color')}, Gender: {product_meta.get('gender')}")
                return {"products": found_products}
            else:
                print("Stage 3: No products found even with just gender/season filters. Returning empty.")
        except Exception as e:
            print(f"Error in Stage 3 search: {e}")
            traceback.print_exc()
            return {"products": [], "message": f"Error in Stage 3 search: {e}"}
    else:
        print("Stage 3: Skipping because neither gender nor season filters were provided (and no other filters apply).")


    # If no products are found after all stages, return an empty list
    print("No products found after all search stages.")
    return {"products": []}

# --- NEW: API Endpoint for Linromi to call directly ---
@app.route('/api/find_apparel', methods=['POST'])
def find_apparel_api():
    try:
        raw_request_data = request.json
        print(f"Received API call with raw data: {raw_request_data}")

        if not raw_request_data:
            return jsonify({"error": "Request must be JSON"}), 400

        # Linromi seems to wrap the actual tool call arguments in 'apparel_search_data' and it's a STRING
        if 'apparel_search_data' not in raw_request_data:
            return jsonify({"error": "Missing 'apparel_search_data' key in request, check Linromi configuration."}), 400

        # Attempt to parse the nested 'apparel_search_data' string
        try:
            tool_call_arguments_json_str = raw_request_data['apparel_search_data']
            # If apparel_search_data is itself a stringified JSON, parse it
            if isinstance(tool_call_arguments_json_str, str):
                 arguments = json.loads(tool_call_arguments_json_str)
            else: # If Linromi somehow passes it as a dict directly
                 arguments = tool_call_arguments_json_str

            print(f"Parsed tool call arguments: {arguments}")

        except json.JSONDecodeError as e:
            return jsonify({"error": f"Failed to parse 'apparel_search_data' as JSON: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"Error extracting arguments from 'apparel_search_data': {str(e)}"}), 400


        # Now, arguments should be the dictionary that OpenAI Assistant would have passed directly
        # Example: arguments = {"user_query": "...", "filters": {...}}
        user_query = arguments.get('user_query')
        if not user_query:
            return jsonify({"error": "Missing 'user_query' in parsed arguments"}), 400

        # Extract filters from the nested 'filters' object, defaulting to an empty dict
        filters = arguments.get('filters', {})

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