# message_fetcher.py
import httpx
import time
from typing import List, Dict

API_URL = "https://november7-730026606190.europe-west1.run.app/messages/"

def fetch_all_messages() -> List[Dict]:
    """Fetch all messages with retry logic and rate limit handling"""
    
    all_messages = []
    skip = 0
    limit = 4000  # Smaller batches
    max_retries = 3
    
    print("Fetching messages from API...")
    
    while True:
        for attempt in range(max_retries):
            try:
                response = httpx.get(
                    API_URL,
                    params={"skip": skip, "limit": limit},
                    headers={"accept": "application/json"},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])
                    
                    if not items:
                        print(f"✓ Fetched {len(all_messages)} messages total")
                        return all_messages
                    
                    all_messages.extend(items)
                    total = data.get("total", 0)
                    print(f"  Fetched {len(all_messages)}/{total}")
                    
                    if len(all_messages) >= total:
                        print(f"✓ Fetched all {len(all_messages)} messages")
                        return all_messages
                    
                    skip += limit
                    time.sleep(0.2)  # Small delay between requests
                    break  # Success, exit retry loop
                    
                elif response.status_code == 401:
                    print(f"⚠️ 401 Unauthorized - retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
                elif response.status_code == 402:
                    print(f"⚠️ Rate limit hit at {len(all_messages)} messages")
                    if all_messages:
                        return all_messages
                    raise Exception("Rate limited from start")
                    
                else:
                    print(f"⚠️ Status {response.status_code}, attempt {attempt + 1}/{max_retries}")
                    time.sleep(1)
                    
            except httpx.TimeoutException:
                print(f"⚠️ Timeout, attempt {attempt + 1}/{max_retries}")
                time.sleep(2)
                
            except Exception as e:
                print(f"⚠️ Error: {e}, attempt {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    if all_messages:
                        print(f"⚠️ Returning {len(all_messages)} messages fetched before error")
                        return all_messages
                    raise
                time.sleep(2)
        
        # If all retries failed for this batch
        if all_messages:
            print(f"⚠️ Stopping at {len(all_messages)} messages due to repeated failures")
            return all_messages
        else:
            raise Exception("Failed to fetch any messages")
    
    return all_messages


_message_cache = None

def get_messages(force_refresh=False) -> List[Dict]:
    global _message_cache
    
    if _message_cache is None or force_refresh:
        _message_cache = fetch_all_messages()
    
    return _message_cache
