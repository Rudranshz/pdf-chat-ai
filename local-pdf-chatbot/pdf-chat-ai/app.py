import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def run_private_rag():
    # --- 1. CONFIGURATION & PATHS ---
    # Automatically finds the folder where this script is saved
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_PATH = os.path.join(BASE_DIR, "data", "RUDRANSH.pdf")
    
    # Brain (Reasoning) and Memory (Embeddings) models
    LLM_MODEL = "gemma3:4b"
    EMBED_MODEL = "nomic-embed-text"
    
    print(f"[*] Target PDF: {PDF_PATH}")
    
    if not os.path.exists(PDF_PATH):
        print(f"Error: Could not find PDF. Please ensure it is in: {os.path.join(BASE_DIR, 'data')}")
        return

    # --- 2. DOCUMENT PROCESSING ---
    print(f"[*] Reading PDF content...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    
    # Split the document into 1000-character chunks with overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"[+] Document split into {len(splits)} chunks.")

    # --- 3. VECTOR STORAGE (CHROMA DB) ---
    print(f"[*] Creating embeddings with {EMBED_MODEL}...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    # Create the local vector database
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

    # --- 4. THE BRAIN (GEMMA 3) & PROMPT ---
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    
    template = """Answer the question based ONLY on the following context:
    {context}

    Question: {question}
    
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # The LCEL Chain: Retrieve -> Format -> LLM -> Output
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- 5. INTERACTIVE STREAMING CHAT ---
    print(f"\n[+] SUCCESS: Private AI is active.")
    print(f"[*] Using {LLM_MODEL} for reasoning.")
    print("[*] Type 'exit' or 'quit' to stop.")
    
    while True:
        user_input = input("\nYour Question: ")
        if user_input.lower() in ["exit", "quit"]: 
            break
        
        if not user_input.strip():
            continue

        print("\nAI is thinking...", end="\r") # Temporary status
        
        try:
            print("AI Answer: ", end="", flush=True)
            
            # This loop enables 'Streaming' (real-time typing)
            for chunk in rag_chain.stream(user_input):
                print(chunk, end="", flush=True)
            
            print("\n") # Final newline after response ends
            
        except Exception as e:
            print(f"\n[!] An error occurred: {e}")

if __name__ == "__main__":
    run_private_rag()