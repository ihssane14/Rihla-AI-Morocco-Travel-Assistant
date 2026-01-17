
import json
import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rihla-morocco")

print(f" Checking API keys...")
if not PINECONE_API_KEY:
    print(" Error: PINECONE_API_KEY not found!")
    sys.exit(1)

print(" API key found")

# Initialize
try:
    print("\n Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(" Embedding model loaded")
    
    print(" Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print(" Connected to Pinecone")
    
except Exception as e:
    print(f" Error: {e}")
    sys.exit(1)

# Delete old data and recreate
index_name = PINECONE_INDEX_NAME
print(f"\n Resetting index '{index_name}'...")

try:
    existing_indexes = pc.list_indexes().names()
    
    if index_name in existing_indexes:
        print(f"  Deleting old index...")
        pc.delete_index(index_name)
        time.sleep(5)
    
    print(f" Creating fresh index...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f" Index created! Waiting 10 seconds...")
    time.sleep(10)
    
    index = pc.Index(index_name)
    print(f" Connected to index")
    
except Exception as e:
    print(f" Error: {e}")
    sys.exit(1)

# Load data
data_path = "data/morocco_destinations.json"
if not os.path.exists(data_path):
    print(f" File not found: {data_path}")
    sys.exit(1)

print(f"\n Loading destinations...")
with open(data_path, "r", encoding="utf-8") as f:
    destinations = json.load(f)

print(f" Loaded {len(destinations)} destinations")

if len(destinations) < 30:
    print(f"  WARNING: Only {len(destinations)} destinations found!")
    print("Expected 40. Make sure you updated the JSON file!")

# Create embeddings
print(f"\n Creating embeddings for {len(destinations)} destinations...\n")
vectors = []
success_count = 0

for i, dest in enumerate(destinations, 1):
    try:
        text = f"""
        City: {dest['city']}
        Place: {dest['place']}
        Type: {dest['type']}
        Description: {dest['description']}
        Tips: {dest['tips']}
        Cost: {dest['cost']}
        Best Time: {dest['best_time']}
        """
        
        embedding = model.encode(text.strip()).tolist()
        
        vectors.append({
            "id": f"dest_{dest['id']}",
            "values": embedding,
            "metadata": {
                "id": dest['id'],
                "city": dest['city'],
                "place": dest['place'],
                "type": dest.get('type', 'General'),
                "description": dest['description'],
                "tips": dest['tips'],
                "cost": dest['cost'],
                "best_time": dest.get('best_time', 'Anytime')
            }
        })
        
        print(f"✓ [{i}/{len(destinations)}] {dest['place']} ({dest['city']})")
        success_count += 1
        
        # Upload in batches of 10
        if len(vectors) >= 10:
            index.upsert(vectors=vectors)
            print(f"   Uploaded batch of {len(vectors)} vectors")
            vectors = []
        
    except Exception as e:
        print(f" Error processing {dest.get('place', 'unknown')}: {e}")

# Upload remaining
if vectors:
    index.upsert(vectors=vectors)
    print(f"\n Uploaded final batch of {len(vectors)} vectors")

# Get stats
print("\n" + "="*60)
print(" RESULTS")
print("="*60)

time.sleep(2)
stats = index.describe_index_stats()
print(f"Total vectors in index: {stats.total_vector_count}")
print(f"Dimension: {stats.dimension}")
print(f"Successfully processed: {success_count}/{len(destinations)}")

print("\n" + "="*60)
print(" EMBEDDING UPDATE COMPLETE!")
print("="*60)
print("\n Now restart your backend: python main.py")