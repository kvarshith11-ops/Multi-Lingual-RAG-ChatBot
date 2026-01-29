from app.embedder import BedrockEmbedder
from app.vector_store import FaissVectorStore
from app.models import Document
from app.llm import ClaudeClient


class RAGPipeline:
    def __init__(self, vector_store: FaissVectorStore):
        self.embedder = BedrockEmbedder()
        self.vector_store = vector_store
        self.llm = ClaudeClient()

    def answer(self, question: str, top_k: int = 10) -> str:
        query_doc = Document(content=question, metadata={"type": "query"})
        query_embedding = self.embedder.embed_documents([query_doc])[0]

        retrieved_docs = self.vector_store.search(query_embedding, top_k=top_k)

        # 🔍 DEBUG: Print retrieved chunks
        print("\n--- Retrieved Chunks from Vector DB ---")
        for i, doc in enumerate(retrieved_docs, 1):
            src = doc.metadata.get("source")
            page = doc.metadata.get("page")
            ocr = doc.metadata.get("ocr")
            print(
                f"\nChunk {i} (source={src}, page={page}, ocr={ocr}):\n"
                f"{doc.content[:800]}"
            )
        print("\n-------------------------------------\n")

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

        return self.llm.generate(prompt)
