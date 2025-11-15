# main.py
import uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from message_fetcher import get_messages
from answer_generator import generate_answer
from vector_store import initialize_vector_store, get_collection_stats
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Question-Answering System",
    description="Ask natural language questions about member data using RAG (Retrieval-Augmented Generation)",
    version="2.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize vector store on startup."""
    logger.info("🚀 Starting up application...")
    
    try:
        # Fetch messages
        messages = get_messages()
        logger.info(f"✓ Fetched {len(messages)} messages")
        
        # Initialize vector store (will skip if already initialized)
        initialize_vector_store(messages, force_recreate=False)
        logger.info("✓ Vector store ready")
        
    except Exception as e:
        logger.error(f"✗ Startup error: {e}")


@app.get("/")
def root():
    """Health check endpoint."""
    stats = get_collection_stats()
    
    return {
        "status": "healthy",
        "service": "Question-Answering System (RAG)",
        "version": "2.0.0",
        "approach": "Vector Search + LLM",
        "vector_store": stats,
        "timestamp": datetime.utcnow().isoformat(),
        "usage": "GET /ask?question=YOUR_QUESTION",
        "examples": [
            "/ask?question=When is Layla planning her trip to London?",
            "/ask?question=How many cars does Vikram Desai have?",
            "/ask?question=What are Amira's favorite restaurants?"
        ]
    }


@app.get("/health")
def health_check():
    """Health check for deployment platforms."""
    try:
        messages = get_messages()
        stats = get_collection_stats()
        
        return {
            "status": "healthy",
            "messages_cached": len(messages),
            "vector_store": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/ask")
def ask_question(question: str = Query(..., min_length=3, description="Natural language question")):
    """
    Answer a natural language question using RAG approach.
    """
    logger.info(f"Question received: {question}")
    
    try:
        # Generate answer using vector search + LLM
        answer = generate_answer(question)
        
        logger.info(f"Answer generated successfully")
        return {"answer": answer}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/refresh")
def refresh_cache():
    """Force refresh both message cache and vector store."""
    logger.info("Cache refresh requested")
    
    messages = get_messages(force_refresh=True)
    initialize_vector_store(messages, force_recreate=True)
    
    return {
        "message": "Cache and vector store refreshed successfully",
        "total_messages": len(messages),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/stats")
def get_stats():
    """Get statistics about cached messages and vector store."""
    messages = get_messages()
    vector_stats = get_collection_stats()
    
    # Count messages per user
    user_counts = {}
    for msg in messages:
        user = msg['user_name']
        user_counts[user] = user_counts.get(user, 0) + 1
    
    top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "total_messages": len(messages),
        "unique_users": len(user_counts),
        "top_users": [{"name": name, "message_count": count} for name, count in top_users],
        "vector_store": vector_stats,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
