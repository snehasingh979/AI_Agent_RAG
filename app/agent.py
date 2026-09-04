from pathlib import Path
import os
import re

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. Project Path
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# ==========================================
# 2. Connect to Groq
# ==========================================

print("Connecting to Groq...")

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

groq_client = Groq(
    api_key=groq_api_key
)

print("Groq connected!")


# ==========================================
# 3. Load Embedding Model
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ==========================================
# 4. Connect to ChromaDB
# ==========================================

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db_new")
)

collection = client.get_collection(
    name="ai_agent_rag"
)

print("ChromaDB connected!")
print("Total documents:", collection.count())


# ==========================================
# 5. Calculator Tool
# ==========================================

def calculator(expression):

    try:

        # Allow only numbers and basic mathematical operators
        if not re.fullmatch(
            r"[0-9+\-*/().%\s]+",
            expression
        ):
            return "Invalid calculation."

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:

        return "Invalid calculation."


# ==========================================
# 6. RAG Tool
# ==========================================

def rag_search(question):

    print("\n🔎 Agent is using RAG...")

    # --------------------------------------
    # Create Query Embedding
    # --------------------------------------

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # --------------------------------------
    # Search ChromaDB
    # --------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    if not documents:
        return "No relevant information found."

    # --------------------------------------
    # Create Context
    # --------------------------------------

    context = "\n\n".join(documents)

    # --------------------------------------
    # Groq Prompt
    # --------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context, say:

"I don't have enough information in the provided documents."

Do not make up information.

Context:
{context}

User Question:
{question}
"""

    # --------------------------------------
    # Generate Answer
    # --------------------------------------

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ==========================================
# 7. AI Agent
# ==========================================

def run_agent(question):

    question_lower = question.lower().strip()

    # --------------------------------------
    # Calculator Detection
    # --------------------------------------

    calculator_patterns = [
        r"^[0-9+\-*/().%\s]+$",
        r"^what is [0-9+\-*/().%\s]+$",
        r"^calculate [0-9+\-*/().%\s]+$"
    ]

    for pattern in calculator_patterns:

        if re.fullmatch(
            pattern,
            question_lower
        ):

            # Remove words like "what is" or "calculate"
            expression = re.sub(
                r"^(what is|calculate)\s+",
                "",
                question_lower
            )

            print(
                "\n🔧 Agent is using Calculator Tool..."
            )

            result = calculator(expression)

            return f"Calculation result: {result}"

    # --------------------------------------
    # Normal Question → RAG
    # --------------------------------------

    return rag_search(question)


# ==========================================
# 8. Test Agent
# ==========================================

if __name__ == "__main__":

    print("\n🤖 AI Agent Started")

    while True:

        question = input(
            "\nAsk Agent: "
        ).strip()

        # ----------------------------------
        # Exit
        # ----------------------------------

        if question.lower() == "exit":

            print("Goodbye! 👋")
            break

        # ----------------------------------
        # Empty Question
        # ----------------------------------

        if not question:

            print("Please enter a question.")
            continue

        # ----------------------------------
        # Run Agent
        # ----------------------------------

        answer = run_agent(question)

        print("\nAI Agent Answer:")
        print(answer)