# main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from datetime import datetime
import logging
from answer_generator import generate_answer
from message_fetcher import get_messages
from vector_store import initialize_vector_store, get_collection_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Question-Answering System",
    description="Ask natural language questions about member data using RAG",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/ask")
def ask_question(question: str = Query(..., min_length=3)):
    logger.info(f"Question: {question}")
    try:
        answer = generate_answer(question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/refresh")
def refresh_cache():
    """Force refresh embeddings (takes ~3-5 minutes)"""
    try:
        logger.info("Starting refresh...")
        messages = get_messages(force_refresh=True)
        logger.info(f"Fetched {len(messages)} messages")
        
        initialize_vector_store(messages, force_recreate=True)
        
        return {
            "status": "success",
            "message": f"Successfully refreshed {len(messages)} embeddings",
            "total_messages": len(messages),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Refresh error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    try:
        messages = get_messages()
        user_counts = {}
        for msg in messages:
            user = msg['user_name']
            user_counts[user] = user_counts.get(user, 0) + 1
        
        top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_messages": len(messages),
            "unique_users": len(user_counts),
            "top_users": [{"name": name, "count": count} for name, count in top_users],
            "vector_store": get_collection_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
