from pathlib import Path
import chromadb


# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent


# Connect to existing ChromaDB
print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db_new")
)

print("ChromaDB connected!")


# Get existing collection
collection = client.get_collection(
    name="ai_agent_rag"
)

print("Collection loaded!")
print("Total documents:", collection.count())


# Test query
query = "What is this document about?"


# Temporary query embedding
vector = [0.0] * 384

for i, char in enumerate(query):
    vector[i % 384] += ord(char)

total = sum(vector)

if total != 0:
    vector = [x / total for x in vector]


# Search ChromaDB
print("Searching ChromaDB...")

results = collection.query(
    query_embeddings=[vector],
    n_results=2
)


# Display results
print("\nRelevant chunks:\n")

for i, document in enumerate(results["documents"][0]):
    print(f"--- Chunk {i + 1} ---")
    print(document)
    print()