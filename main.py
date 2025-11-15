# main.py
import uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from message_fetcher import get_messages
from answer_generator import generate_answer
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
    description="Ask natural language questions about member data",
    version="1.0.0"
)

# Add CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Question-Answering System",
        "version": "1.0.0",
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
        return {
            "status": "healthy",
            "messages_cached": len(messages),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/ask")
def ask_question(question: str = Query(..., min_length=3, description="Natural language question")):
    """
    Answer a natural language question based on member messages.
    """
    logger.info(f"Question received: {question}")
    
    try:
        # Fetch all messages (cached after first call)
        messages = get_messages()
        
        if not messages:
            logger.error("No messages available")
            raise HTTPException(status_code=503, detail="No messages available from API")
        
        # Generate answer using LLM
        answer = generate_answer(question, messages)
        
        logger.info(f"Answer generated successfully")
        return {"answer": answer}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/refresh")
def refresh_cache():
    """Force refresh the message cache."""
    logger.info("Cache refresh requested")
    messages = get_messages(force_refresh=True)
    return {
        "message": "Cache refreshed successfully",
        "total_messages": len(messages),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/stats")
def get_stats():
    """Get statistics about cached messages."""
    messages = get_messages()
    
    # Count messages per user
    user_counts = {}
    for msg in messages:
        user = msg['user_name']
        user_counts[user] = user_counts.get(user, 0) + 1
    
    # Sort by count
    top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "total_messages": len(messages),
        "unique_users": len(user_counts),
        "top_users": [{"name": name, "message_count": count} for name, count in top_users],
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
