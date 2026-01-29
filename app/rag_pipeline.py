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
You are a multilingual knowledge assistant.

Your task is to answer the user's question using ONLY the information provided in the context below.
The context may be in one or more languages.

Instructions:
1. Understand the meaning of the question, even if it is in a different language from the context.
2. Find the most relevant information from the context.
3. If the answer is directly stated, use it.
4. If the answer is implied, infer it logically from the context.
5. Do NOT use any external knowledge.
6. If the answer cannot be found or inferred from the context, say clearly that the information is not available.
7. Always respond in the SAME language as the user's question.

Context:
{context}

Question:
{question}

Provide a clear, concise, and accurate answer:
"""

        return self.llm.generate(prompt)
