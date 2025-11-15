# test_questions.py
import requests
import json

BASE_URL = "http://localhost:8000"

test_cases = [
    "When is Layla planning her trip to London?",
    "How many cars does Vikram Desai have?",
    "What are Amira's favorite restaurants?",
    "What does Sophia Al-Farsi want?",
    "When did Armand Dupont update his phone number?",
    "What is Hans Müller asking about?",
    "Who requested tickets to the opera in Milan?",
    "What type of car does Layla prefer?",
    "How many people is Fatima's dinner reservation for?",
    "What did Sophia book for Friday?"
]

print("🧪 Testing Question-Answering System\n")
print("=" * 80)

for i, question in enumerate(test_cases, 1):
    print(f"\n{i}. Question: {question}")
    print("-" * 80)
    
    try:
        response = requests.get(
            f"{BASE_URL}/ask",
            params={"question": question},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer")
            print(f"✅ Answer: {answer}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("-" * 80)

print("\n✅ Testing complete!")
