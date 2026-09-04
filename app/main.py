import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Create Groq client
client = Groq(api_key=api_key)

# Send first question to AI
response = client.chat.completions.create(
   model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Hello! Explain what RAG is in simple words."
        }
    ]
)

print(response.choices[0].message.content)