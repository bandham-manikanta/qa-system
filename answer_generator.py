# answer_generator.py
import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Initializing NVIDIA NIM client
client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url=os.getenv("NVIDIA_BASE_URL")
)

# EXACT model name from NVIDIA documentation (case-sensitive!)
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


def generate_answer(question: str, messages: List[Dict]) -> str:
    """
    Use LLM to answer the question based on the messages.
    """
    context = prepare_context(messages)
    
    max_context_chars = 100000
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n\n[... truncated ...]"
    
    system_prompt = """You are a helpful assistant that answers questions based on member messages.

RULES:
1. Answer based ONLY on the information in the messages
2. Be CONCISE - one or two sentences maximum
3. If information is not available, say ONLY: "I don't have that information"
4. For dates: provide the exact date/time mentioned
5. For counts: provide just the number
6. For lists: provide comma-separated items
7. Do NOT add explanations or caveats unless necessary
8. Extract ONLY the specific information asked for"""

    user_prompt = f"""Based on these member messages:

{context}

Question: {question}

Provide a direct answer based only on the information above."""

    print(f"Sending question to LLM: {question}")
    print(f"Using model: {MODEL_NAME}")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"✓ Got answer from LLM")
        return answer
        
    except Exception as e:
        print(f"✗ Error calling LLM: {e}")
        return f"Error generating answer: {str(e)}"
