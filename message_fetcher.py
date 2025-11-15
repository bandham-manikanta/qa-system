# message_fetcher.py
import httpx
from typing import List, Dict

API_URL = "https://november7-730026606190.europe-west1.run.app/messages/"

def fetch_all_messages() -> List[Dict]:
    """
    Try to fetch all messages in one request with high limit.
    """
    headers = {
        "accept": "application/json",
        "User-Agent": "QA-System/1.0"
    }
    
    print("Fetching messages from API...")
    
    try:
        # Try fetching with very high limit
        response = httpx.get(
            API_URL,
            params={"skip": 0, "limit": 5000},  # High limit to get all
            headers=headers,
            timeout=60.0
        )
        response.raise_for_status()
        data = response.json()
        
        messages = data.get("items", [])
        total = data.get("total", 0)
        
        print(f"✓ Fetched {len(messages)} messages (total available: {total})")
        return messages
        
    except Exception as e:
        print(f"Error fetching messages: {e}")
        raise

_message_cache = None

def get_messages(force_refresh=False) -> List[Dict]:
    """Get messages with simple in-memory caching."""
    global _message_cache
    
    if _message_cache is None or force_refresh:
        _message_cache = fetch_all_messages()
    
    return _message_cache
