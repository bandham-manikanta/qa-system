# answer_generator.py
import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv
from vector_store import search_relevant_messages

load_dotenv()

client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url=os.getenv("NVIDIA_BASE_URL")
)

MODEL_NAME = "qwen/qwen3-next-80b-a3b-instruct"


def prepare_context(messages: List[Dict]) -> str:
    """Convert messages to a readable format for the LLM."""
    context_lines = []
    
    for msg in messages:
        line = (
            f"User: {msg['user_name']}\n"
            f"Date: {msg['timestamp']}\n"
            f"Message: {msg['message']}\n"
        )
        context_lines.append(line)
    
    return "\n---\n".join(context_lines)


def generate_answer(question: str, messages: List[Dict] = None) -> str:
    """
    Use vector search + LLM to answer the question.
    
    Args:
        question: User's question
        messages: Optional - if provided, uses these instead of vector search
                  (used for backward compatibility)
    """
    # Use vector search to find relevant messages
    if messages is None:
        print(f"🔎 Using vector search for: {question}")
        relevant_messages = search_relevant_messages(question, top_k=15)
    else:
        # Fallback to old method if messages provided
        relevant_messages = messages
    
    if not relevant_messages:
        return "I don't have any relevant information to answer this question."

    context = prepare_context(relevant_messages)
    
    system_prompt = """You are a precise assistant that answers questions based on member messages.

CRITICAL RULES:
1. Answer based ONLY on the information in the messages provided
2. Be CONCISE - one or two sentences maximum
3. If information is not available in the messages, say ONLY: "I don't have that information"
4. For dates: provide the exact date/time mentioned
5. For counts: provide just the number
6. For lists: provide comma-separated items
7. Do NOT add explanations or caveats unless necessary
8. Extract ONLY the specific information asked for"""

    user_prompt = f"""Messages:

{context}

Question: {question}

Answer concisely:"""

    print(f"Sending question to LLM: {question}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Context size: {len(relevant_messages)} messages")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"✓ Got answer from LLM")
        return answer
        
    except Exception as e:
        print(f"✗ Error calling LLM: {e}")
        return f"Error generating answer: {str(e)}"
