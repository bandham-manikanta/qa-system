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

def get_embedding(text: str, input_type: str = "passage") -> List[float]:
    """
    Get embedding from NVIDIA API
    
    Args:
        text: Text to embed
        input_type: "query" for questions, "passage" for documents
    """
    client = get_embeddings_client()
    
    response = client.embeddings.create(
        input=text,
        model="nvidia/nv-embedqa-e5-v5",
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "END"}  # ADD THIS
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
    
    # Use "query" type for questions
    question_embedding = get_embedding(question, input_type="query")
    
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
    
    from qdrant_client.models import Distance, VectorParams, PointStruct
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    
    print(f"📝 Embedding {len(messages)} messages using NVIDIA API...")
    
    points = []
    for idx, msg in enumerate(messages):
        text = f"User: {msg['user_name']}\nDate: {msg['timestamp']}\nMessage: {msg['message']}"
        # Use "passage" type for documents being indexed
        embedding = get_embedding(text, input_type="passage")
        
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
    
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  Uploaded {min(i+batch_size, len(points))}/{len(points)}")
    
    print(f"✅ Stored {len(messages)} embeddings in Qdrant")
