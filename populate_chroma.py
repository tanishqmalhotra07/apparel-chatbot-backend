import json
import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Initialize OpenAI client for embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")
client = OpenAI(api_key=OPENAI_API_KEY) # <--- CORRECTED: Changed api_api_key to api_key

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "apparel_products"

# Delete collection if it already exists (useful for fresh start during development)
try:
    chroma_client.delete_collection(name=collection_name)
    print(f"Existing collection '{collection_name}' deleted.")
except Exception as e:
    # Catch specific exception if collection doesn't exist to avoid verbose error on first run
    if "does not exist" in str(e):
        print(f"Collection '{collection_name}' did not exist. Creating new one.")
    else:
        print(f"Error deleting collection '{collection_name}': {e}. Proceeding to create/get.")

# Get or create the collection
collection = chroma_client.get_or_create_collection(name=collection_name)

def get_embedding(text):
    """Generates an embedding for the given text using OpenAI's API."""
    try:
        response = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text: '{text[:50]}...' - {e}")
        traceback.print_exc()
        return None

def populate_chroma_db(json_file_path):
    """Reads products from a JSON file and populates ChromaDB."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        products = json.load(f)

    ids = []
    documents = []
    embeddings_list = []
    metadatas = []

    print(f"Processing {len(products)} products from {json_file_path}...")

    for i, product in enumerate(products):
        product_id = product.get("id")
        name = product.get("name", "")
        short_description = product.get("short_description", "")

        embedding_text = f"{name}. {short_description}"
        embedding = get_embedding(embedding_text)

        if embedding is None:
            print(f"Skipping product {product_id} due to embedding error.")
            continue

        # --- Extract ALL metadata fields and convert lists to strings ---
        occasion_tags = product.get("occasion_tags")
        if isinstance(occasion_tags, list):
            occasion_tags = ", ".join(occasion_tags) # Convert list to comma-separated string
        
        style_tags = product.get("style_tags")
        if isinstance(style_tags, list):
            style_tags = ", ".join(style_tags) # Convert list to comma-separated string

        metadata = {
            "id": product_id,
            "name": name,
            "price": product.get("price"),
            "image_url": product.get("image_url"),
            "product_url": product.get("product_url"),
            "category": product.get("category"),
            "gender": product.get("gender"),
            "occasion_tags": occasion_tags, # Now a string
            "style_tags": style_tags,     # Now a string
            "color": product.get("color"),
            "short_description": short_description,
            # --- NEW FIELDS ---
            "master_category": product.get("master_category"),
            "subcategory": product.get("subcategory"),
            "season": product.get("season"),
            "sleeve_length": product.get("sleeve_length"),
            "item_length": product.get("item_length")
        }
        
        # Filter out None values. "N/A" strings will be kept.
        metadata = {k: v for k, v in metadata.items() if v is not None}


        ids.append(product_id)
        documents.append(embedding_text)
        embeddings_list.append(embedding)
        metadatas.append(metadata)

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1} products...")

    # Add to collection
    try:
        if ids:
            collection.add(
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added {len(ids)} documents to ChromaDB.")
            print(f"Total documents in collection: {collection.count()}")
        else:
            print("No valid products to add to ChromaDB.")
    except Exception as e:
        print(f"Error adding documents to ChromaDB: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    products_json_path = 'products.json'
    if os.path.exists(products_json_path):
        populate_chroma_db(products_json_path)
    else:
        print(f"Error: {products_json_path} not found. Please ensure the file is in the same directory.")