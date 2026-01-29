import streamlit as st
import os
import glob
from typing import List, Tuple
from app.loaders.pdf_loader import PDFLoader
from app.chunker import TextChunker
from app.embedder import BedrockEmbedder
from app.vector_store import FaissVectorStore
from app.rag_pipeline import RAGPipeline
from app.models import Document

# Page configuration
st.set_page_config(
    page_title="Multilingual RAG Chatbot",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, professional look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    .title-container {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .title-container h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .title-container p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Input box styling */
    .stChatInputContainer {
        border-top: 2px solid #e0e0e0;
        padding-top: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Context card styling */
    .context-card {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    
    .context-meta {
        color: #666;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        font-style: italic;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Info boxes */
    .info-box {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .info-label {
        font-weight: 600;
        color: #667eea;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .info-value {
        font-size: 1rem;
        color: #333;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

INDEX_PATH = "data/index/faiss.index"
DOCS_PATH = "data/index/documents.pkl"

@st.cache_resource
def load_vector_store():
    """Load or build the FAISS vector store"""
    embedder = BedrockEmbedder()
    
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        vector_store = FaissVectorStore(embedding_dim=1024)
        vector_store.load(INDEX_PATH, DOCS_PATH)
        return vector_store
    else:
        # Build index if not exists
        loader = PDFLoader()
        chunker = TextChunker(chunk_size=200, overlap=20)
        
        all_docs = []
        pdf_files = glob.glob("data/raw/*.pdf")
        
        if not pdf_files:
            st.error("No PDF files found in data/raw folder")
            st.stop()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, file_path in enumerate(pdf_files):
            status_text.text(f"Loading: {os.path.basename(file_path)}")
            all_docs.extend(loader.load(file_path))
            progress_bar.progress((idx + 1) / len(pdf_files))
        
        status_text.text("Creating chunks...")
        chunks = chunker.chunk_documents(all_docs)
        
        status_text.text("Generating embeddings...")
        embeddings = embedder.embed_documents(chunks)
        
        vector_store = FaissVectorStore(embedding_dim=len(embeddings[0]))
        vector_store.add_embeddings(embeddings, chunks)
        
        vector_store.save(INDEX_PATH, DOCS_PATH)
        
        progress_bar.empty()
        status_text.empty()
        
        return vector_store

def get_answer_with_context(rag: RAGPipeline, question: str, top_k: int = 10) -> Tuple[str, List[Document]]:
    """Get answer and retrieved context"""
    from app.models import Document
    
    query_doc = Document(content=question, metadata={"type": "query"})
    query_embedding = rag.embedder.embed_documents([query_doc])[0]
    
    retrieved_docs = rag.vector_store.search(query_embedding, top_k=top_k)
    
    context = "\n\n".join([doc.content for doc in retrieved_docs])
    
    prompt = f"""
You are a multilingual knowledge assistant with STRICT grounding requirements.

⚠️ CRITICAL RULES:
1. LANGUAGE: Detect the language of the user's question and respond in THE SAME LANGUAGE. Do not translate or switch languages.
2. GROUNDING: You MUST answer using ONLY the information in the Context section below
3. If the Context does not contain information to answer the question, you MUST respond with:
   "I cannot answer this question as the information is not available in the provided documents."
4. DO NOT use any knowledge outside the provided Context
5. DO NOT make up or infer information that is not explicitly in the Context

Context:
{context}

Question:
{question}

Answer (respond in the same language as the question, or say information is not available):
"""
    
    answer = rag.llm.generate(prompt)
    return answer, retrieved_docs

def main():
    # Header
    st.markdown("""
    <div class="title-container">
        <h1>🌐 Multilingual RAG Chatbot</h1>
        <p>Powered by Amazon Bedrock | Ask questions in any language</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Information")
        
        st.markdown("""
        <div class="info-box">
            <div class="info-label">LLM Model</div>
            <div class="info-value">Claude 3.5 Sonnet</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <div class="info-label">Embeddings</div>
            <div class="info-value">Amazon Titan v2</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <div class="info-label">Vector Database</div>
            <div class="info-value">FAISS</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <div class="info-label">Mode</div>
            <div class="info-value">Multilingual RAG</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Toggle for showing context
        show_context = st.toggle("📄 Show Retrieved Context", value=False)
        
        st.markdown("---")
        
        st.markdown("### 🌍 Supported Languages")
        st.markdown("""
        - 🇮🇳 Telugu
        - 🇮🇳 Hindi
        - 🇬🇧 English
        - 🇫🇷 French
        - And many more...
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag" not in st.session_state:
        with st.spinner("🔄 Loading vector store and initializing RAG pipeline..."):
            vector_store = load_vector_store()
            st.session_state.rag = RAGPipeline(vector_store)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show context if available and toggle is on
            if show_context and message["role"] == "assistant" and "context" in message:
                with st.expander("📚 Retrieved Context Chunks"):
                    for i, doc in enumerate(message["context"], 1):
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "?")
                        ocr = doc.metadata.get("ocr", False)
                        
                        st.markdown(f"""
                        <div class="context-card">
                            <div class="context-meta">
                                📄 Chunk {i} | Source: {os.path.basename(source)} | Page: {page} | OCR: {ocr}
                            </div>
                            <div>{doc.content[:500]}{'...' if len(doc.content) > 500 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask your question in any language (Telugu, Hindi, English, French...)"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    answer, context_docs = get_answer_with_context(
                        st.session_state.rag, 
                        prompt, 
                        top_k=10
                    )
                    
                    st.markdown(answer)
                    
                    # Store message with context
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "context": context_docs
                    })
                    
                    # Show context if toggle is on
                    if show_context:
                        with st.expander("📚 Retrieved Context Chunks"):
                            for i, doc in enumerate(context_docs, 1):
                                source = doc.metadata.get("source", "Unknown")
                                page = doc.metadata.get("page", "?")
                                ocr = doc.metadata.get("ocr", False)
                                
                                st.markdown(f"""
                                <div class="context-card">
                                    <div class="context-meta">
                                        📄 Chunk {i} | Source: {os.path.basename(source)} | Page: {page} | OCR: {ocr}
                                    </div>
                                    <div>{doc.content[:500]}{'...' if len(doc.content) > 500 else ''}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()
