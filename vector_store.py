# vector_store.py - add rate limit handling
import os
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional, Callable
from dotenv import load_dotenv
import httpx
import time

load_dotenv()

_client = None
COLLECTION_NAME = "member_messages"
EXPECTED_DIM = 1024


def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60
        )
    return _client


async def get_embedding_async(text: str, input_type: str = "passage", max_retries: int = 5) -> List[float]:
    """Get embedding with retry logic for rate limits"""
    url = "https://integrate.api.nvidia.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": "nvidia/nv-embedqa-e5-v5",
        "encoding_format": "float",
        "input_type": input_type,
        "truncate": "END"
    }
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                    print(f"⚠️ Rate limit hit, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⚠️ Rate limit (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                raise
    
    raise Exception("Max retries exceeded")


def get_embedding_sync(text: str, input_type: str = "passage") -> List[float]:
    """Synchronous version for /ask queries"""
    url = "https://integrate.api.nvidia.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": "nvidia/nv-embedqa-e5-v5",
        "encoding_format": "float",
        "input_type": input_type,
        "truncate": "END"
    }
    
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


def get_collection_stats() -> Dict:
    try:
        client = get_client()
        count = client.count(COLLECTION_NAME).count
        collection_info = get_client().get_collection(COLLECTION_NAME)
        vector_size = collection_info.config.params.vectors.size
        
        return {
            "total_documents": count, 
            "initialized": True, 
            "type": "qdrant_cloud",
            "dimension": vector_size,
            "dimension_match": vector_size == EXPECTED_DIM
        }
    except:
        return {
            "total_documents": 0, 
            "initialized": False, 
            "type": "qdrant_cloud"
        }


def search_relevant_messages(question: str, top_k: int = 15) -> List[Dict]:
    """Search for relevant messages"""
    client = get_client()
    
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        count = client.count(COLLECTION_NAME).count
        
        if count == 0:
            print("❌ Vector store is empty. Run /refresh to initialize.")
            return []
        
        vector_size = collection_info.config.params.vectors.size
        if vector_size != EXPECTED_DIM:
            print(f"❌ Dimension mismatch: {vector_size} vs {EXPECTED_DIM}. Run /refresh.")
            return []
            
    except Exception as e:
        print(f"❌ Collection not found: {e}. Run /refresh.")
        return []
    
    question_embedding = get_embedding_sync(question, input_type="query")
    
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=question_embedding,
            limit=top_k
        )
    except Exception as e:
        print(f"❌ Search error: {e}")
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


async def embed_batch_async(messages: List[Dict], start_idx: int) -> List[PointStruct]:
    """Embed a batch with delay between requests"""
    points = []
    
    for idx, msg in enumerate(messages):
        text = f"User: {msg['user_name']}\nDate: {msg['timestamp']}\nMessage: {msg['message']}"
        
        # Add small delay to avoid rate limits
        if idx > 0:
            await asyncio.sleep(0.1)  # 100ms delay between requests
        
        embedding = await get_embedding_async(text, input_type="passage")
        
        points.append(PointStruct(
            id=start_idx + idx,
            vector=embedding,
            payload={
                "user_name": msg['user_name'],
                "user_id": msg['user_id'],
                "timestamp": msg['timestamp'],
                "message": msg['message']
            }
        ))
    
    return points


async def initialize_vector_store_async(
    messages: List[Dict], 
    force_recreate: bool = False,
    progress_callback: Optional[Callable] = None,
    concurrent_batches: int = 3  # Reduced from 10 to avoid rate limits
):
    """Initialize vector store with rate limit handling"""
    client = get_client()
    
    # Check/create collection
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if exists:
        if force_recreate:
            print("🔄 Deleting existing collection...")
            client.delete_collection(COLLECTION_NAME)
        else:
            try:
                collection_info = client.get_collection(COLLECTION_NAME)
                vector_size = collection_info.config.params.vectors.size
                count = client.count(COLLECTION_NAME).count
                
                if vector_size != EXPECTED_DIM:
                    print(f"⚠️ Wrong dimensions, recreating...")
                    client.delete_collection(COLLECTION_NAME)
                elif count == len(messages):
                    print(f"✓ Already has {count} embeddings")
                    return
                else:
                    print(f"⚠️ Count mismatch, recreating...")
                    client.delete_collection(COLLECTION_NAME)
            except:
                pass
    
    # Create collection
    print(f"📝 Creating collection ({EXPECTED_DIM}-dim)")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EXPECTED_DIM, distance=Distance.COSINE)
    )
    
    # Process in smaller batches with rate limiting
    batch_size = 20  # Smaller batches
    all_points = []
    
    print(f"🔧 Embedding {len(messages)} messages ({concurrent_batches} concurrent batches)...")
    print(f"⏱️ Estimated time: ~{len(messages) * 0.15 / 60:.1f} minutes")
    
    for i in range(0, len(messages), batch_size * concurrent_batches):
        batch_groups = []
        for j in range(concurrent_batches):
            start = i + (j * batch_size)
            end = min(start + batch_size, len(messages))
            if start < len(messages):
                batch_groups.append((messages[start:end], start))
        
        # Process batches
        tasks = [embed_batch_async(batch, start_idx) for batch, start_idx in batch_groups]
        batch_results = await asyncio.gather(*tasks)
        
        for batch_points in batch_results:
            all_points.extend(batch_points)
        
        progress = len(all_points)
        percentage = progress / len(messages) * 100
        print(f"  📊 {progress}/{len(messages)} ({percentage:.1f}%)")
        
        if progress_callback:
            progress_callback(progress, len(messages), "embedding")
        
        # Delay between batch groups to avoid rate limits
        await asyncio.sleep(0.5)
    
    # Upload to Qdrant
    print("💾 Uploading to Qdrant...")
    upload_batch_size = 100
    
    for i in range(0, len(all_points), upload_batch_size):
        batch = all_points[i:i+upload_batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  📤 Uploaded {min(i + upload_batch_size, len(all_points))}/{len(all_points)}")
    
    print(f"✅ Successfully stored {len(messages)} embeddings")


def initialize_vector_store(messages: List[Dict], force_recreate: bool = False):
    """Sync wrapper"""
    asyncio.run(initialize_vector_store_async(messages, force_recreate))
