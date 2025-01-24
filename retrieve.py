import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

class DocumentRetriever:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.persist_dir = "chroma_db"
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize with empty collection if needed
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="main_collection"  # Fixed collection name
        )
        
        # Workaround for Chroma's empty DB issue
        if not self.vector_store.get()['documents']:
            self.vector_store.add_texts(["Initial empty document"])
            self.vector_store.delete(ids=["0"])  # Remove placeholder
    
    def query_documents(self, query: str, k: int = 5):
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            metadata = doc.metadata
            formatted_results.append({
                "text": doc.page_content,
                "source": metadata["source"],
                "page": metadata["page_number"],
                "chunk_id": metadata["chunk_id"],
                "score": float(score)
            })
        
        return formatted_results
