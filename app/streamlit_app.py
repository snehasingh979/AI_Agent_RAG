from pathlib import Path
import sys

import streamlit as st


# ==========================================
# Project Path
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

APP_DIR = BASE_DIR / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# ==========================================
# Import Agent
# ==========================================

from agent import run_agent


# ==========================================
# Streamlit Page Configuration
# ==========================================

st.set_page_config(
    page_title="AI Agent RAG",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# Session State - Chat History
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.title("📚 RAG Information")

    st.write("Your AI Agent can answer questions from your PDF documents.")

    st.divider()

    st.subheader("Supported PDFs")

    documents_folder = BASE_DIR / "documents"

    if documents_folder.exists():

        pdf_files = sorted(documents_folder.glob("*.pdf"))

        if pdf_files:

            for pdf in pdf_files:
                st.write(f"• {pdf.name}")

        else:
            st.write("No PDFs found.")

    st.divider()

    st.subheader("Agent Tools")

    st.write("🔎 RAG Search")
    st.write("🔧 Calculator")
    st.write("🧠 Groq LLM")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# ==========================================
# Main UI
# ==========================================

st.title("🤖 AI Agent RAG Chatbot")

st.write(
    "Ask questions from your PDF documents or use the calculator."
)


# ==========================================
# Display Previous Messages
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# User Input
# ==========================================

question = st.chat_input("Ask your question...")


# ==========================================
# Process Question
# ==========================================

if question:

    # --------------------------------------
    # Display User Question
    # --------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------
    # AI Agent
    # --------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🤖 Agent is thinking..."):

            try:

                answer = run_agent(question)

            except Exception as e:

                answer = f"Error: {e}"

            st.markdown(answer)


    # --------------------------------------
    # Save AI Response
    # --------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )