from pathlib import Path
import os

from dotenv import load_dotenv
from groq import Groq
import chromadb
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
# 5. Chat History / Memory
# ==========================================

chat_history = []


# ==========================================
# 6. Continuous Chatbot
# ==========================================

while True:

    question = input("\nAsk your question: ").strip()


    # ======================================
    # Exit
    # ======================================

    if question.lower() == "exit":

        print("Goodbye! 👋")

        break


    # ======================================
    # Empty question
    # ======================================

    if not question:

        print("Please enter a question.")

        continue


    # ======================================
    # 7. Create Query Embedding
    # ======================================

    print("\nSearching relevant information...")

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    print("Query embedding created!")

    print(
        "Embedding dimension:",
        len(query_embedding)
    )


    # ======================================
    # 8. Search ChromaDB
    # ======================================

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )


    # ======================================
    # 9. Get Relevant Documents
    # ======================================

    relevant_chunks = results["documents"][0]

    if not relevant_chunks:

        print("No relevant information found.")

        continue

    context = "\n\n".join(relevant_chunks)

    print("\nRelevant information found!")


    # ======================================
    # 10. Prepare Chat History
    # ======================================

    history_text = ""

    for chat in chat_history:

        history_text += f"""
User: {chat['question']}
AI: {chat['answer']}
"""


    # ======================================
    # 11. Generate Answer using Groq
    # ======================================

    print("\nGenerating answer...")


    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the document context below.

You may also use the previous conversation to understand
what the user is referring to.

If the answer is not available in the document context,
say:

"I don't have enough information in the provided documents."

Do not make up information.


Previous Conversation:
{history_text}


Document Context:
{context}


Current User Question:
{question}
"""


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


    # ======================================
    # 12. Get AI Answer
    # ======================================

    answer = response.choices[0].message.content


    # ======================================
    # 13. Save Conversation to Memory
    # ======================================

    chat_history.append({
        "question": question,
        "answer": answer
    })


    # ======================================
    # 14. Display Answer
    # ======================================

    print("\n================================")
    print("AI ANSWER")
    print("================================")

    print(answer)