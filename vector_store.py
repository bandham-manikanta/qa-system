# vector_store.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict

_model = None
_collection = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory="./chroma_db"
        ))
        try:
            _collection = client.get_collection("member_messages")
        except:
            pass
    return _collection

def initialize_vector_store(messages: List[Dict], force_recreate: bool = False):
    global _collection
    
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory="./chroma_db"
    ))
    
    if force_recreate:
        try:
            client.delete_collection("member_messages")
        except:
            pass
    
    try:
        _collection = client.get_collection("member_messages")
        if _collection.count() == len(messages):
            print(f"✓ Using existing {_collection.count()} embeddings")
            return _collection
    except:
        pass
    
    _collection = client.create_collection("member_messages")
    
    print(f"📝 Embedding {len(messages)} messages...")
    
    documents = []
    metadatas = []
    ids = []
    
    for msg in messages:
        doc = f"User: {msg['user_name']}\nDate: {msg['timestamp']}\nMessage: {msg['message']}"
        documents.append(doc)
        metadatas.append({
            "user_name": msg['user_name'],
            "user_id": msg['user_id'],
            "timestamp": msg['timestamp'],
            "message": msg['message']
        })
        ids.append(msg['id'])
    
    model = get_model()
    embeddings = model.encode(documents, show_progress_bar=False).tolist()
    
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        _collection.add(
            documents=documents[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end]
        )
    
    print(f"✅ Stored {len(messages)} embeddings")
    return _collection


def search_relevant_messages(question: str, top_k: int = 15) -> List[Dict]:
    collection = get_collection()
    
    if collection is None or collection.count() == 0:
        # Initialize on first use
        from message_fetcher import get_messages
        messages = get_messages()
        initialize_vector_store(messages)
        collection = get_collection()
    
    model = get_model()
    question_embedding = model.encode([question]).tolist()
    
    results = collection.query(query_embeddings=question_embedding, n_results=top_k)
    
    relevant_messages = []
    if results['metadatas'] and len(results['metadatas'][0]) > 0:
        for metadata in results['metadatas'][0]:
            relevant_messages.append({
                'user_name': metadata['user_name'],
                'user_id': metadata['user_id'],
                'timestamp': metadata['timestamp'],
                'message': metadata['message']
            })
    
    print(f"🔍 Found {len(relevant_messages)} relevant messages")
    return relevant_messages


def get_collection_stats() -> Dict:
    collection = get_collection()
    if collection:
        return {"total_documents": collection.count(), "initialized": True}
    return {"total_documents": 0, "initialized": False}
