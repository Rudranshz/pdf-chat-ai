# --- 1. CLOUD COMPATIBILITY PATCH (Vital for Streamlit Cloud) ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ---------------------------------------------------------------

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- 2. SETUP & KEYS ---
st.set_page_config(page_title="Global PDF AI", page_icon="🌐", layout="wide")
st.title("🌐 Live Cloud Research Assistant")

# Streamlit automatically finds these keys in your ".streamlit/secrets.toml" (Local)
# OR in the Streamlit Cloud Dashboard (Deployment)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
HF_TOKEN = st.secrets.get("HF_TOKEN")

if not GROQ_API_KEY or not HF_TOKEN:
    st.error("⚠️ Secrets are missing! Please set GROQ_API_KEY and HF_TOKEN.")
    st.stop()

# --- 3. CLOUD BRAIN ---
@st.cache_resource
def load_models():
    # Groq LPU (Fast Inference)
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
    # HuggingFace Embeddings (Cloud CPU)
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embed_model

llm, embed_model = load_models()

# --- 4. IN-MEMORY DATABASE (Cloud Optimized) ---
def process_docs(files):
    all_docs = []
    for file in files:
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f: f.write(file.getbuffer())
        loader = PyPDFLoader(temp_path)
        all_docs.extend(loader.load())
        os.remove(temp_path)
    
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
    # No persist_directory = Runs in RAM (Prevents Cloud Errors)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
    return vectorstore.as_retriever()

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []

with st.sidebar:
    st.header("☁️ Cloud Panel")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing in Cloud..."):
        retriever = process_docs(uploaded_files)
        st.success("Docs Indexed!")
    
    # Display Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if user_input := st.chat_input("Ask about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        # RAG Chains
        context_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            ("user", "Rephrase as a standalone search query.")
        ])
        history_retriever = create_history_aware_retriever(llm, retriever, context_prompt)
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer purely from this context:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}")
        ])
        rag_chain = create_retrieval_chain(history_retriever, create_stuff_documents_chain(llm, qa_prompt))
        
        response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
        
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        with st.chat_message("assistant"): st.markdown(response["answer"])  