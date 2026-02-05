import streamlit as st
import os
import shutil
import gc
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- 1. CONFIGURATION ---
DB_DIR = "d:/project/pdf-chat-ai/chroma_db"
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="Perfect Local PDF AI", page_icon="⚡", layout="wide")
st.title("⚡ Perfect Local Research Assistant")

# --- 2. DATABASE MANAGEMENT (Fixes WinError 32) ---
def reset_database():
    """Forces the system to release file locks before deleting."""
    if "retriever" in st.session_state:
        st.session_state.retriever = None
        del st.session_state.retriever
    
    # Force Python to "let go" of the files
    gc.collect()
    time.sleep(1) # Give Windows a second to unlock
    
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
            st.toast("✅ Database wiped successfully!", icon="🗑️")
        except PermissionError:
            st.error("⚠️ Windows is holding the file. Please stop the app (Ctrl+C) and try again.")
    
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.rerun()

# --- 3. CORE LOGIC ---
def get_vectorstore(files=None):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    # Load existing if available
    if not files:
        if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
            return Chroma(persist_directory=DB_DIR, embedding_function=embeddings).as_retriever()
        return None

    # Process new files
    all_docs = []
    for file in files:
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f: f.write(file.getbuffer())
        loader = PyPDFLoader(temp_path)
        all_docs.extend(loader.load())
        os.remove(temp_path)
    
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
    
    # Save to disk
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
    return vectorstore.as_retriever()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Controls")
    model_name = st.selectbox("Select Model", ["gemma3:4b", "gemma3:1b"], index=0)
    uploaded_files = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("🔄 Index New Files"):
        if uploaded_files:
            with st.spinner("Processing..."):
                st.session_state.retriever = get_vectorstore(uploaded_files)
                st.success("Indexed!")
    
    st.divider()
    if st.button("🗑️ Wipe Database"):
        reset_database()

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "retriever" not in st.session_state: st.session_state.retriever = get_vectorstore()

# Display Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# Handle Input
if user_input := st.chat_input("Ask a question..."):
    if st.session_state.retriever:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        try:
            llm = ChatOllama(model=model_name, temperature=0)
            
            # Context Chain
            context_q_prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                ("user", "Rephrase the above as a standalone search query.")
            ])
            history_retriever = create_history_aware_retriever(llm, st.session_state.retriever, context_q_prompt)
            
            # Answer Chain
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer strictly using this context:\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}")
            ])
            rag_chain = create_retrieval_chain(history_retriever, create_stuff_documents_chain(llm, qa_prompt))

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
                    st.markdown(result["answer"])
                    
                    with st.expander("🔍 View Sources"):
                        for doc in result["context"]:
                            st.caption(f"Page {doc.metadata.get('page', 0)+1}: {doc.page_content[:200]}...")

            st.session_state.chat_history.extend([HumanMessage(content=user_input), AIMessage(content=result["answer"])])
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        
        except Exception as e:
            st.error(f"❌ Connection Error: Is Ollama running? Details: {e}")
    else:
        st.warning("⚠️ Please upload and index a PDF first.")