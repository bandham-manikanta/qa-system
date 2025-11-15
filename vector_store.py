# vector_store.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import os

# Initialize embedding model (runs locally, free)
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight

# Initialize ChromaDB client
client = chromadb.Client(Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory="./chroma_db"  # Local storage
))

COLLECTION_NAME = "member_messages"


def initialize_vector_store(messages: List[Dict], force_recreate: bool = False):
    """
    Initialize ChromaDB with all messages.
    Creates embeddings and stores them.
    """
    print(f"\n🔧 Initializing vector store...")
    
    # Delete existing collection if recreating
    if force_recreate:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("✓ Deleted existing collection")
        except:
            pass
    
    # Get or create collection
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"✓ Found existing collection with {collection.count()} documents")
        
        # If collection exists and has same number of docs, skip initialization
        if collection.count() == len(messages) and not force_recreate:
            print("✓ Vector store already initialized, skipping")
            return collection
    except:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Member messages for QA system"}
        )
        print("✓ Created new collection")
    
    # Prepare documents for embedding
    print(f"📝 Processing {len(messages)} messages...")
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, msg in enumerate(messages):
        # Create rich text representation for embedding
        doc_text = f"User: {msg['user_name']}\nDate: {msg['timestamp']}\nMessage: {msg['message']}"
        documents.append(doc_text)
        
        # Store metadata
        metadatas.append({
            "user_name": msg['user_name'],
            "user_id": msg['user_id'],
            "timestamp": msg['timestamp'],
            "message": msg['message']
        })
        
        ids.append(msg['id'])
    
    # Generate embeddings and add to ChromaDB
    print("🧮 Generating embeddings (this may take 30-60 seconds)...")
    
    # ChromaDB can handle embeddings automatically, but we'll do it explicitly for control
    embeddings = EMBEDDING_MODEL.encode(documents, show_progress_bar=True).tolist()
    
    print("💾 Storing in ChromaDB...")
    
    # Add in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end]
        )
        print(f"  Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
    
    print(f"✅ Vector store initialized with {len(messages)} messages\n")
    return collection


def search_relevant_messages(question: str, top_k: int = 15) -> List[Dict]:
    """
    Search for relevant messages using semantic similarity.
    
    Args:
        question: User's natural language question
        top_k: Number of most relevant messages to return
    
    Returns:
        List of relevant message dictionaries
    """
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except:
        print("⚠️ Vector store not initialized")
        return []
    
    # Generate embedding for question
    question_embedding = EMBEDDING_MODEL.encode([question]).tolist()
    
    # Query ChromaDB for similar messages
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    
    # Convert results to message format
    relevant_messages = []
    
    if results['metadatas'] and len(results['metadatas'][0]) > 0:
        for metadata in results['metadatas'][0]:
            relevant_messages.append({
                'user_name': metadata['user_name'],
                'user_id': metadata['user_id'],
                'timestamp': metadata['timestamp'],
                'message': metadata['message']
            })
    
    print(f"🔍 Found {len(relevant_messages)} relevant messages for query")
    return relevant_messages


def get_collection_stats() -> Dict:
    """Get statistics about the vector store."""
    try:
        collection = client.get_collection(COLLECTION_NAME)
        return {
            "total_documents": collection.count(),
            "initialized": True
        }
    except:
        return {
            "total_documents": 0,
            "initialized": False
        }
