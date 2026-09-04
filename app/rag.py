from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. Project Path
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "documents"

print("Documents folder:", DOCUMENTS_DIR)
print("Folder exists:", DOCUMENTS_DIR.exists())


# ==========================================
# 2. Find All PDFs
# ==========================================

pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

print("\nPDF files found:", len(pdf_files))

for pdf in pdf_files:
    print("-", pdf.name)


if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found in documents folder."
    )


# ==========================================
# 3. Load All PDFs
# ==========================================

all_documents = []

print("\nLoading PDFs...")


for pdf_path in pdf_files:

    print(f"\nLoading: {pdf_path.name}")

    loader = PyPDFLoader(
        str(pdf_path)
    )

    documents = loader.load()

    print(
        "Pages:",
        len(documents)
    )

    # Add source information
    for document in documents:

        document.metadata["source"] = pdf_path.name

    all_documents.extend(documents)


print("\nTotal pages from all PDFs:", len(all_documents))


# ==========================================
# 4. Split Documents into Chunks
# ==========================================

print("\nSplitting documents...")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


chunks = text_splitter.split_documents(
    all_documents
)


print("Total chunks:", len(chunks))


# ==========================================
# 5. Connect to ChromaDB
# ==========================================

print("\nConnecting to ChromaDB...")


client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db_new")
)


print("ChromaDB client created!")


# ==========================================
# 6. Create Collection
# ==========================================

print("Creating collection...")


collection = client.get_or_create_collection(
    name="ai_agent_rag",
    embedding_function=None
)


print("Collection created!")


# ==========================================
# 7. Load Embedding Model
# ==========================================

print("\nLoading embedding model...")


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print("Embedding model loaded!")


# ==========================================
# 8. Prepare Text
# ==========================================

documents_list = [
    chunk.page_content
    for chunk in chunks
]


# ==========================================
# 9. Create IDs
# ==========================================

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]


# ==========================================
# 10. Create Embeddings
# ==========================================

print("\nCreating embeddings...")


embeddings = model.encode(
    documents_list,
    normalize_embeddings=True
).tolist()


print(
    "Embeddings created:",
    len(embeddings)
)


print(
    "Embedding dimension:",
    len(embeddings[0])
)


# ==========================================
# 11. Store in ChromaDB
# ==========================================

print("\nStarting ChromaDB add...")


collection.add(
    documents=documents_list,
    embeddings=embeddings,
    ids=ids
)


print("ChromaDB add completed!")


# ==========================================
# 12. Final Information
# ==========================================

print("\n================================")
print("RAG INGESTION COMPLETED")
print("================================")

print(
    "PDFs processed:",
    len(pdf_files)
)

print(
    "Total pages:",
    len(all_documents)
)

print(
    "Total chunks:",
    len(chunks)
)

print(
    "Total documents in ChromaDB:",
    collection.count()
)

print("\nPDF sources:")

for pdf in pdf_files:
    print("-", pdf.name)

print("\nDone! ✅")