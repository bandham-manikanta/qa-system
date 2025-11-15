from message_fetcher import get_messages

messages = get_messages()

layla_messages = [msg for msg in messages if 'layla' in msg['user_name'].lower()]

print(f"Found {len(layla_messages)} messages from Layla:\n")

for msg in layla_messages:
    print(f"Date: {msg['timestamp']}")
    print(f"Message: {msg['message']}")
    print("-" * 80)

london_messages = [msg for msg in messages if 'london' in msg['message'].lower()]

print(f"\nFound {len(london_messages)} messages mentioning 'London':\n")

for msg in london_messages[:10]:
    print(f"User: {msg['user_name']}")
    print(f"Date: {msg['timestamp']}")
    print(f"Message: {msg['message']}")
    print("-" * 80)
