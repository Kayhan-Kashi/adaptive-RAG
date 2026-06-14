# services/rag_service.py
import logging
import os
from injector import inject
from langchain_community.vectorstores import FAISS
from common.kafka.producer import KafkaProducer #type: ignore
from src.core.document_loader import DocumentLoader
from src.core.text_chunker import TextChunker
from src.core.text_preprocessor import TextPreprocessor
from langchain_core.documents import Document
from typing import List, Optional
from src.services.embedding_service import EmbeddingService
from src.core.JinaLangChainWrapper import JinaLangChainWrapper

logger = logging.getLogger(__name__)


class RagService:
    """RAG Service using injected EmbeddingService"""
    
    @inject
    def __init__(self,
                document_loader: DocumentLoader, 
                text_preprocessor: TextPreprocessor, 
                text_chunker: TextChunker,
                embedding_service: EmbeddingService,  # Injected directly
                kafka_producer: Optional[KafkaProducer] = None):
        self.doc_loader = document_loader
        self.text_preprocessor = text_preprocessor
        self.text_chunker = text_chunker
        self.embedding_service = embedding_service
        self.kafka_producer = kafka_producer
        self.vector_store = None
        self.index_path = "./faiss_index"
        self.index_file = f"{self.index_path}/index.faiss"


        
        # Try to load existing vector store
        self._load_vector_store()
        
        logger.info("✅ RagService initialized")
    
    def _load_vector_store(self):
        """Load vector store from disk if it exists"""
        index_file = f"{self.index_path}/index.faiss"
        if os.path.exists(index_file):
            try:
                # CRITICAL FIX: Wrap the model for LangChain compatibility BEFORE loading
                from src.core.JinaLangChainWrapper import JinaLangChainWrapper
                
                # Create proper embedding wrapper
                embedding_wrapper = JinaLangChainWrapper(self.embedding_service.model)
                
                # Load with the wrapper, NOT the raw model
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    embedding_wrapper,  # ← Use wrapper here, not raw model
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✅ Loaded existing vector store from {self.index_path}")
                logger.info(f"   Total vectors: {self.vector_store.index.ntotal}")
                
                # Test retrieval to verify it works
                test_query = "test"
                test_results = self.vector_store.similarity_search(test_query, k=1)
                logger.info(f"   Test retrieval: ✓ successful")
                
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        else:
            logger.info("No existing vector store found. Will create new one when documents are added.")
    
    def save_vector_store(self, path: Optional[str] = None) -> bool:
        """
        Save FAISS vector store to disk
        
        Args:
            path: Path to save the vector store (defaults to self.index_path)
        
        Returns:
            True if successful, False otherwise
        """
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return False
        
        save_path = path or self.index_path
        
        try:
            self.vector_store.save_local(save_path)
            logger.info(f"💾 Saved FAISS vector store to {save_path}")
            logger.info(f"   Total vectors saved: {self.vector_store.index.ntotal}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save vector store: {e}")
            return False
    
    def store_vector(self, chunks: List[Document]) -> None:
        """Store document chunks in FAISS vector store"""
        try:
            if not chunks:
                logger.warning("No chunks to store")
                return
            
            # Create wrapper consistently
            langchain_compatible_model = JinaLangChainWrapper(self.embedding_service.model)
            
            if self.vector_store is None:
                # Create new vector store with wrapper
                self.vector_store = FAISS.from_documents(chunks, langchain_compatible_model)
                logger.info(f"✅ Created new FAISS vector store with {len(chunks)} chunks")
            else:
                # Add to existing vector store
                self.vector_store.add_documents(chunks)
                logger.info(f"✅ Added {len(chunks)} chunks to existing vector store")
            
            logger.info(f"   Total vectors: {self.vector_store.index.ntotal}")
            logger.info(f"   Dimension: {self.vector_store.index.d}")
            
            # Auto-save after each addition
            self.save_vector_store()
            
        except Exception as e:
            logger.error(f"❌ Failed to store vectors: {e}")
            raise
    
    
    def retrieve_similar_chunks(self, query: str, k: int = 10) -> List[Document]:
        """
        Find the top k most similar chunks for a given query.
        This retrieves the 'Values' (Documents) associated with the closest 'Keys' (Vectors).
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty, cannot perform retrieval.")
            return []
        
        try:
            # Note: similarity_search automatically uses the embedding model 
            # (our JinaLangChainWrapper) passed during initialization/creation.
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"🔍 Retrieved {len(results)} chunks for query: '{query[:30]}...'")
            return results
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return []
        
    def prepare_document(self, file_path: str, document_id: str, filename: str, 
                        filetype: str, chunk_size: int = 200, 
                        chunk_overlap: int = 20) -> List[Document]:
        """Load PDF, preprocess, chunk, store, and verify with a retrieval test"""
        logger.info(f"📄 Processing document: {filename}")
        
        try:
            # 1. LOAD
            self.doc_loader.load_pdf(file_path=file_path)
            raw_text = self.doc_loader.text
            logger.info(f"DEBUG [1/3] Raw text length: {len(raw_text) if raw_text else 0}")

            if not raw_text:
                logger.error("❌ No text extracted from document.")
                return []

            # 2. PREPROCESS
            preprocessed_text = self.text_preprocessor.preprocess(raw_text)
            logger.info(f"DEBUG [2/3] Preprocessed text length: {len(preprocessed_text)}")

            if not preprocessed_text:
                logger.error("❌ Preprocessing removed all text.")
                return []

            # 3. CREATE DOCUMENT
            doc = Document(
                page_content=preprocessed_text,
                metadata={
                    "source": file_path,
                    "document_id": document_id,
                    "filename": filename
                }
            )

            # 4. CHUNK
            chunks = self.text_chunker.chunk(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            logger.info(f"DEBUG [3/3] Chunk count: {len(chunks)}")

            if len(chunks) > 0:
                logger.info(f"DEBUG: First chunk preview: {chunks[0].page_content[:120]}")

            # 5. Add metadata
            for chunk in chunks:
                chunk.metadata['document_id'] = document_id
                chunk.metadata['filename'] = filename

            # 6. Store in FAISS
            logger.info(f"📊 Storing {len(chunks)} chunks in FAISS vector store...")
            self.store_vector(chunks)

            # 7. Retrieval sanity check
            if chunks:
                logger.info("🧪 Running retrieval sanity check (Top 10)...")
                sample_query = "what are query key and values"
                top_10 = self.retrieve_similar_chunks(sample_query, k=10)

                logger.info(f"✨ Retrieved {len(top_10)} chunks for verification:")
                for i, doc in enumerate(top_10):
                    logger.info(
                        f"[{i+1}] {doc.page_content[:60]}... "
                        f"(ID: {doc.metadata.get('document_id')})"
                    )

            logger.info(f"✅ Document processing completed: {filename}")
            return chunks

        except Exception as e:
            logger.error(f"❌ Failed to process document: {e}")
            raise


    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents in vector store"""
        if self.vector_store is None:
            logger.warning("No vector store available. Please prepare a document first.")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"🔍 Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Search with similarity scores"""
        if self.vector_store is None:
            logger.warning("No vector store available")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            logger.info(f"🔍 Found {len(results)} results with scores")
            return results
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_stats(self) -> dict:
        """Get vector store statistics"""
        if self.vector_store is None:
            return {
                "status": "empty",
                "total_vectors": 0,
                "dimension": 0
            }
        
        return {
            "status": "active",
            "total_vectors": self.vector_store.index.ntotal,
            "dimension": self.vector_store.index.d,
            "index_path": self.index_path
        }
    
    def clear_vector_store(self) -> None:
        """Clear the vector store"""
        self.vector_store = None
        logger.info("🗑️ Vector store cleared")
        
        # Also delete the saved file if it exists
        if os.path.exists(f"{self.index_path}.faiss"):
            os.remove(f"{self.index_path}.faiss")
            os.remove(f"{self.index_path}.pkl")
            logger.info(f"🗑️ Deleted saved vector store files")