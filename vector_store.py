# vector_store.py
import os
from qdrant_client import QdrantClient
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = None
_embeddings_client = None
COLLECTION_NAME = "member_messages"

def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60
        )
    return _client

def get_embeddings_client():
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = OpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1"
        )
    return _embeddings_client

def get_embedding(text: str) -> List[float]:
    """Get embedding from NVIDIA API (no local model)"""
    client = get_embeddings_client()
    
    response = client.embeddings.create(
        input=text,
        model="nvidia/nv-embedqa-e5-v5",  # NVIDIA's embedding model
        encoding_format="float"
    )
    
    return response.data[0].embedding

def get_collection_stats() -> Dict:
    try:
        client = get_client()
        count = client.count(COLLECTION_NAME).count
        return {"total_documents": count, "initialized": True, "type": "qdrant_cloud"}
    except:
        return {"total_documents": 0, "initialized": False, "type": "qdrant_cloud"}

def search_relevant_messages(question: str, top_k: int = 15) -> List[Dict]:
    client = get_client()
    
    # Get embedding from NVIDIA API (lightweight, no model loading)
    question_embedding = get_embedding(question)
    
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=question_embedding,
            limit=top_k
        )
    except Exception as e:
        print(f"Search error: {e}")
        return []
    
    relevant_messages = [
        {
            'user_name': r.payload['user_name'],
            'user_id': r.payload['user_id'],
            'timestamp': r.payload['timestamp'],
            'message': r.payload['message']
        }
        for r in results
    ]
    
    print(f"🔍 Found {len(relevant_messages)} relevant messages")
    return relevant_messages

def initialize_vector_store(messages: List[Dict], force_recreate: bool = False):
    client = get_client()
    
    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if exists and not force_recreate:
        count = client.count(COLLECTION_NAME).count
        if count == len(messages):
            print(f"✓ Using existing {count} embeddings")
            return
        client.delete_collection(COLLECTION_NAME)
    elif exists and force_recreate:
        client.delete_collection(COLLECTION_NAME)
    
    # Create collection (1024 dims for NVIDIA embeddings)
    from qdrant_client.models import Distance, VectorParams, PointStruct
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)  # NVIDIA embedding size
    )
    
    print(f"📝 Embedding {len(messages)} messages using NVIDIA API...")
    
    # Prepare data
    points = []
    for idx, msg in enumerate(messages):
        text = f"User: {msg['user_name']}\nDate: {msg['timestamp']}\nMessage: {msg['message']}"
        embedding = get_embedding(text)  # API call, not local model
        
        points.append(PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "user_name": msg['user_name'],
                "user_id": msg['user_id'],
                "timestamp": msg['timestamp'],
                "message": msg['message']
            }
        ))
        
        if (idx + 1) % 50 == 0:
            print(f"  Embedded {idx + 1}/{len(messages)}")
    
    # Upload in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  Uploaded {min(i+batch_size, len(points))}/{len(points)}")
    
    print(f"✅ Stored {len(messages)} embeddings in Qdrant")
