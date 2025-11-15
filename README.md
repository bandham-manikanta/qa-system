# Q&A System for Member Messages

Spent the last couple days building this - it's an API that answers questions about customer messages. You ask something in normal English, it digs through the data and gives you an answer.

**Try it:** https://qa-system-lbbr.onrender.com/

## How it works

Got about 3300 messages from luxury concierge clients in a database. When you ask a question, the system:
1. Converts your question into a vector (embedding)
2. Searches for similar messages using ChromaDB
3. Takes the top 15 matches
4. Feeds them to NVIDIA's Qwen LLM
5. Returns the answer

The whole vector search part is what makes this work. Without it you'd be sending thousands of messages to the LLM every time which gets expensive fast.

**Built with:**
- FastAPI - handles the web requests
- ChromaDB - vector database running locally
- sentence-transformers - creates the embeddings
- NVIDIA NIM - the actual LLM (Qwen 3)
- Render - where it's deployed

## Current implementation:

My first version just grabbed every message and sent them all to the LLM with the question. Coded it in maybe 20 minutes. But then I realized it was hitting token limits and taking forever. Plus burning through API credits. Not gonna work.

Next I tried being smarter about filtering. If someone asks "what does Vikram want", just pull Vikram's messages. Way faster. But then questions like "who's going to Paris?" didn't work because there's no name to filter on. Back to the drawing board.

That's when I switched to vector search. Every message gets embedded once at startup. Questions get embedded on the fly. Find the closest matches, send just those to the LLM. Takes 45 seconds to set up the first time but after that it's quick and way more accurate.

I looked into fine-tuning a model but that felt like that is way too complicated for this use case so I didnt choose that route. Would need training data, GPU time.

## Things I didn't do

Thought about mixing keyword and semantic search together. Like if someone mentions "Paris" specifically, weight those messages higher. Might help in some edge cases but seemed like not worth imlementing in this short time.

Almost added response caching. If someone asks the same question twice, just return the cached answer. But looked at the data and most questions are actually different. Plus Render's free tier sleeps anyway so cache would just get deleted.

## The data:

Working with 3349 messages from about 10 people. They're all asking for high-end stuff - private jets, Michelin star restaurants, villa bookings, opera tickets, that kind of thing.

A lot of messages say things like "next Friday" or "this weekend" or "in two weeks". Problem is next Friday from when? If the timestamp is May 2025 but I'm querying this in December, what does "next Friday" even mean? The LLM sometimes (correctly) says it doesn't know because there's no real date.

There's one message at the very end that just stops: "I finally" - and that's it. Something got cut off during data generation I guess.

Noticed a bunch of personal info in the messages too. Phone numbers, passport numbers (saw "BA8493921"), credit card last four digits, actual addresses. Not a problem for demo data but you'd definitely need to scrub that in production.

The messages don't connect to each other. Someone asks "can you update my phone number" in one message and maybe "did that get done?" in another, but there's nothing linking them. No thread IDs, no parent message references. Makes it hard to understand context sometimes.

Another weird thing - it's all customer messages. Zero responses from the concierge service. No "yes we booked that" or "sorry that's not available". So when I answer "what did someone book" I'm really saying what they asked for, not what actually happened.

## London question - example:

One of the example questions was "When is Layla planning her trip to London?"

I searched through her messages. She mentions London once - needs a chauffeur "for her stay in London next month". But that's it. No date. Just "next month".

So the system says "I don't have that information" which actually makes sense. There really isn't a specific date in the data. The search finds the right message, the LLM reads it, sees "next month" isn't a date, and correctly reports that.

Kind of a perfect example of the system working right even when it looks wrong.

## What I'd change

1. There's no memory between questions. Can't do follow-ups. Like you can't ask "when is Layla's trip" and then "where is she staying" - the second question doesn't know what the first one was about.

Startup is slow. 45 seconds to embed everything. Could save the embeddings to disk and load them but didn't get to it.

Prompts could be better. Sometimes answers are too long, sometimes too cautious. Needs tuning.

Free tier on Render sleeps after 15 minutes which means first request takes 30+ seconds to wake everything up. Annoying but it's free so can't complain.

## Steps to run this locally:

Clone it, make a virtual environment, install dependencies:

```bash
git clone https://github.com/bandham-manikanta/qa-system.git
cd qa-system
pip install uv
uv venv
./.venv/Scripts/activate (Windows)
```

Add a .env file with your NVIDIA key:
```
NVIDIA_API_KEY=your-key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

Run it:
```bash
uv run uvicorn main:app 
```

First time takes a minute to pull all the messages and build embeddings. After that starts up in a couple seconds.

## What you can hit

- `GET /` - shows you it's working, gives examples
- `GET /ask?question=your question` - the main endpoint
- `GET /health` - status check for monitoring
- `GET /stats` - how many messages per user
- `GET /refresh` - rebuilds everything (slow)

## Deployment:

Running on Render free tier. Sleeps after 15 minutes of inactivity. First hit after sleep takes a while.

ChromaDB stores files on disk so at least it doesn't rebuild the index every deploy.

Need to set environment variables in Render dashboard:
- NVIDIA_API_KEY
- NVIDIA_BASE_URL

## Current problems

Startup time could be better. (Cold starts on free tier are slow.)

Count questions ("how many times did X") sometimes miss thigns because we only look at top 15 messages.

---

Took longer than I thought it would but came out pretty good. Vector search really does help compared to just dumping everything into an LLM through prompt.

---