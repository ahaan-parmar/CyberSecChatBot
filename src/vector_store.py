import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
from utils.config import Config
from src.data_loader import CybersecurityDataLoader

logger = logging.getLogger(__name__)

class CybersecurityVectorStore:
    """Manages vector store for cybersecurity knowledge base"""
    
    def __init__(self, force_rebuild: bool = False):
        self.config = Config()
        self.embedding_model = self._initialize_embeddings()
        self.vector_store = None
        self.data_loader = CybersecurityDataLoader()
        
        # Initialize vector store
        if force_rebuild or not self._vector_store_exists():
            self._build_vector_store()
        else:
            self._load_vector_store()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            if "openai" in Config.EMBEDDING_MODEL.lower():
                if not Config.OPENAI_API_KEY:
                    raise ValueError("OpenAI API key required for OpenAI embeddings")
                return OpenAIEmbeddings(
                    openai_api_key=Config.OPENAI_API_KEY,
                    model=Config.EMBEDDING_MODEL
                )
            else:
                # Use HuggingFace embeddings (default)
                return HuggingFaceEmbeddings(
                    model_name=Config.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            # Fallback to basic HuggingFace embeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def _vector_store_exists(self) -> bool:
        """Check if vector store already exists"""
        store_path = Path(Config.VECTOR_STORE_PATH)
        return store_path.exists() and any(store_path.iterdir())
    
    def _build_vector_store(self):
        """Build vector store from data sources"""
        logger.info("Building vector store from data sources...")
        
        # Load all documents
        documents = self.data_loader.load_all_data()
        
        if not documents:
            logger.error("No documents loaded. Cannot build vector store.")
            return
        
        # Create vector store
        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=Config.VECTOR_STORE_PATH,
                collection_name=Config.COLLECTION_NAME
            )
            
            # Persist the vector store
            self.vector_store.persist()
            
            # Save metadata
            self._save_metadata(documents)
            
            logger.info(f"Vector store built successfully with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            raise
    
    def _load_vector_store(self):
        """Load existing vector store"""
        try:
            self.vector_store = Chroma(
                persist_directory=Config.VECTOR_STORE_PATH,
                embedding_function=self.embedding_model,
                collection_name=Config.COLLECTION_NAME
            )
            logger.info("Vector store loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            logger.info("Rebuilding vector store...")
            self._build_vector_store()
    
    def _save_metadata(self, documents: List[Document]):
        """Save metadata about the vector store"""
        metadata = {
            "total_documents": len(documents),
            "sources": {},
            "embedding_model": Config.EMBEDDING_MODEL,
            "collection_name": Config.COLLECTION_NAME
        }
        
        # Count documents by source
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            metadata["sources"][source] = metadata["sources"].get(source, 0) + 1
        
        # Save metadata to file
        metadata_path = Path(Config.VECTOR_STORE_PATH) / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Metadata saved: {metadata}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get vector store metadata"""
        metadata_path = Path(Config.VECTOR_STORE_PATH) / "metadata.pkl"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        
        return {"status": "metadata not available"}
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None,
        filter_dict: Dict[str, str] = None
    ) -> List[Document]:
        """Perform similarity search"""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        k = k or Config.RETRIEVAL_K
        
        try:
            if filter_dict:
                # Filter by metadata
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k
                )
            
            logger.debug(f"Retrieved {len(results)} documents for query: {query[:100]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = None,
        filter_dict: Dict[str, str] = None
    ) -> List[tuple]:
        """Perform similarity search with relevance scores"""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        k = k or Config.RETRIEVAL_K
        
        try:
            if filter_dict:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            logger.debug(f"Retrieved {len(results)} documents with scores for query: {query[:100]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with score: {e}")
            return []
    
    def get_retriever(self, search_kwargs: Dict[str, Any] = None):
        """Get retriever interface for the vector store"""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return None
        
        default_kwargs = {"k": Config.RETRIEVAL_K}
        if search_kwargs:
            default_kwargs.update(search_kwargs)
        
        return self.vector_store.as_retriever(search_kwargs=default_kwargs)
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to the vector store"""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return
        
        try:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            logger.info(f"Added {len(documents)} new documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
    
    def delete_documents(self, ids: List[str]):
        """Delete documents from vector store by IDs"""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return
        
        try:
            self.vector_store.delete(ids)
            self.vector_store.persist()
            logger.info(f"Deleted {len(ids)} documents from vector store")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
    
    def update_documents(self, documents: List[Document]):
        """Update existing documents in vector store"""
        # For Chroma, we need to delete and re-add
        # In a production system, you might want more sophisticated update logic
        try:
            # Extract IDs if available
            ids = [doc.metadata.get("id") for doc in documents if doc.metadata.get("id")]
            
            if ids:
                self.delete_documents(ids)
            
            self.add_documents(documents)
            logger.info(f"Updated {len(documents)} documents in vector store")
            
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
    
    def search_by_source(self, query: str, source: str, k: int = None) -> List[Document]:
        """Search documents from a specific source"""
        filter_dict = {"source": source}
        return self.similarity_search(query, k, filter_dict)
    
    def search_by_document_type(self, query: str, doc_type: str, k: int = None) -> List[Document]:
        """Search documents by document type"""
        filter_dict = {"doc_type": doc_type}
        return self.similarity_search(query, k, filter_dict)
    
    def get_all_sources(self) -> List[str]:
        """Get list of all available sources in the vector store"""
        metadata = self.get_metadata()
        return list(metadata.get("sources", {}).keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store"""
        stats = {
            "vector_store_exists": self._vector_store_exists(),
            "embedding_model": Config.EMBEDDING_MODEL,
            "collection_name": Config.COLLECTION_NAME,
            "vector_store_path": Config.VECTOR_STORE_PATH
        }
        
        # Add metadata if available
        metadata = self.get_metadata()
        stats.update(metadata)
        
        # Add vector store specific stats if available
        if self.vector_store:
            try:
                # Get collection info
                collection = self.vector_store._collection
                stats["total_vectors"] = collection.count()
                
            except Exception as e:
                logger.debug(f"Could not get vector store stats: {e}")
        
        return stats
    
    def rebuild(self):
        """Force rebuild of the vector store"""
        logger.info("Force rebuilding vector store...")
        
        # Remove existing vector store
        store_path = Path(Config.VECTOR_STORE_PATH)
        if store_path.exists():
            import shutil
            shutil.rmtree(store_path)
        
        # Rebuild
        self._build_vector_store()

def main():
    """Main function to initialize and test vector store"""
    Config.setup_logging()
    
    # Initialize vector store
    print("Initializing vector store...")
    vector_store = CybersecurityVectorStore()
    
    # Get statistics
    stats = vector_store.get_statistics()
    print(f"\nVector Store Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search
    test_query = "SQL injection"
    print(f"\nTesting search with query: '{test_query}'")
    results = vector_store.similarity_search(test_query, k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()