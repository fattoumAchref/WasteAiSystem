# agents/retrieval_agent.py
from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from langchain.vectorstores import FAISS

class RetrievalAgent(BaseRetriever):  # Inherit from BaseRetriever
    vectorstore: FAISS  # Proper type annotation
    
    def get_relevant_documents(self, query: str) -> list[Document]:
        """Required by BaseRetriever interface"""
        return self.vectorstore.similarity_search(query)
    
    async def aget_relevant_documents(self, query: str) -> list[Document]:
        """Async version (optional)"""
        return await self.vectorstore.asimilarity_search(query)