# AI Agent RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that can understand user questions, retrieve relevant information from PDF documents, and generate answers using an LLM.

The project also includes an AI Agent that can decide whether to use the RAG system or a Calculator Tool.

## 🚀 Features

- 📄 Multiple PDF document support
- 🔍 Semantic search using embeddings
- 🧠 Retrieval-Augmented Generation (RAG)
- 🤖 AI Agent with tool selection
- 🧮 Calculator Tool for mathematical expressions
- 🗃️ ChromaDB vector database
- 🔤 Sentence Transformers embeddings
- ⚡ Groq LLM integration
- 🌐 Streamlit Web UI
- 🔐 API key stored securely using `.env`

## 🛠️ Tech Stack

- Python
- Streamlit
- Groq
- ChromaDB
- Sentence Transformers
- LangChain Community
- PyPDF
- HTML/CSS through Streamlit

## 📁 Project Structure

```text
AI_Agent_RAG/
│
├── app/
│   ├── agent.py
│   ├── chatbot.py
│   ├── main.py
│   ├── rag.py
│   ├── retrieve.py
│   ├── streamlit_app.py
│   ├── tools.py
│   └── __init__.py
│
├── documents/
│   ├── AI.pdf
│   ├── ML.pdf
│   ├── DeepLearning.pdf
│   └── sample.pdf
│
├── .gitignore
├── README.md
└── requirements.txt

User
  ↓
Streamlit Web UI
  ↓
AI Agent
  ├── Calculator Tool
  │       ↓
  │    Result
  │
  └── RAG System
          ↓
    Query Embedding
          ↓
       ChromaDB
          ↓
   Relevant PDF Chunks
          ↓
       Groq LLM
          ↓
      AI Answer

      ## 🏗️ System Architecture

```text
                    User
                      ↓
              Streamlit Web UI
                      ↓
                  AI Agent
                 ↙        ↘
        Calculator Tool    RAG System
               ↓                ↓
          Calculation     Query Embedding
                                ↓
                            ChromaDB
                                ↓
                       Relevant PDF Chunks
                                ↓
                           Groq LLM
                                ↓
                           AI Answer

                           RAG Pipeline
PDF Documents
      ↓
PDF Loading
      ↓
Text Chunking
      ↓
Sentence Transformers
      ↓
Embeddings
      ↓
ChromaDB
      ↓
Semantic Retrieval
      ↓
Relevant Context
      ↓
Groq LLM
      ↓
Final Answer
