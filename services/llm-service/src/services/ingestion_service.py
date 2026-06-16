# services/ingestion_service.py
import logging
import os
import pickle
import json
from typing import Any, Dict, List, Optional
from injector import inject
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.core.document_loader import DocumentLoader
from src.core.text_chunker import TextChunker
from src.core.text_preprocessor import TextPreprocessor
from src.core.JinaLangChainWrapper import JinaLangChainWrapper
from src.core.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Handles document ingestion: loading, preprocessing, chunking, and indexing.
    Responsible for creating and updating FAISS and BM25 indexes.
    """
    
    @inject
    def __init__(
        self,
        document_loader: DocumentLoader,
        text_preprocessor: TextPreprocessor,
        text_chunker: TextChunker,
        embedding_model: EmbeddingModel
    ):
        self.doc_loader = document_loader
        self.text_preprocessor = text_preprocessor
        self.text_chunker = text_chunker
        self.embedding_model = embedding_model
        
        # Set paths based on environment
        if os.path.exists("/app"):
            base_dir = "/app"
        else:
            base_dir = "."
        
        # Index paths - use environment variables or defaults
        self.faiss_index_path = os.getenv("FAISS_INDEX_PATH", os.path.join(base_dir, "faiss_index"))
        self.bm25_index_path = os.getenv("BM25_INDEX_PATH", os.path.join(base_dir, "bm25_index"))
        
        # Ensure paths are not empty
        if not self.faiss_index_path:
            self.faiss_index_path = os.path.join(base_dir, "faiss_index")
        if not self.bm25_index_path:
            self.bm25_index_path = os.path.join(base_dir, "bm25_index")
        
        self.bm25_chunks_file = os.path.join(self.bm25_index_path, "chunks.pkl")
        self.bm25_metadata_file = os.path.join(self.bm25_index_path, "metadata.json")
        
        # State
        self.vector_store = None
        self.bm25_retriever = None
        self.all_chunks = []
        
        # ============ OPTIMIZED PARAMETERS ============
        
        # MMR configuration (for diversity in FAISS retrieval)
        self.mmr_fetch_k = int(os.getenv("MMR_FETCH_K", "200"))  # Increased from 100
        self.mmr_lambda_mult = float(os.getenv("MMR_LAMBDA_MULT", "0.5"))  # Balanced diversity
        
        # Chunking parameters (optimized for better retrieval)
        self.default_chunk_size = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))  # Increased from 200
        self.default_chunk_overlap = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))  # 10% overlap
        
        # FAISS index parameters
        self.faiss_index_type = os.getenv("FAISS_INDEX_TYPE", "flat")  # flat, ivf, or hnsw
        
        # ============ END OPTIMIZED PARAMETERS ============
        
        # Initialize
        self._ensure_directories()
        self._load_vector_store()
        self._load_bm25_retriever()
        
        logger.info("✅ IngestionService initialized")
        logger.info(f"   FAISS index: {self.faiss_index_path}")
        logger.info(f"   BM25 index: {self.bm25_index_path}")
        logger.info(f"   FAISS vectors: {self.vector_store.index.ntotal if self.vector_store else 0}")
        logger.info(f"   BM25 chunks: {len(self.all_chunks)}")
        logger.info(f"   MMR: fetch_k={self.mmr_fetch_k}, lambda={self.mmr_lambda_mult}")
        logger.info(f"   Chunking: size={self.default_chunk_size}, overlap={self.default_chunk_overlap}")
    
    def _ensure_directories(self):
        """Create necessary directories for indexes"""
        os.makedirs(self.faiss_index_path, exist_ok=True)
        os.makedirs(self.bm25_index_path, exist_ok=True)
        logger.debug(f"📁 Ensured directories: {self.faiss_index_path}, {self.bm25_index_path}")
    
    def _load_vector_store(self):
        """Load FAISS vector store from disk"""
        index_file = os.path.join(self.faiss_index_path, "index.faiss")
        if os.path.exists(index_file):
            try:
                embedding_wrapper = JinaLangChainWrapper(self.embedding_model.model)
                self.vector_store = FAISS.load_local(
                    self.faiss_index_path,
                    embedding_wrapper,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✅ Loaded FAISS vector store: {self.vector_store.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load FAISS vector store: {e}")
    
    def _load_bm25_retriever(self):
        """Load BM25 retriever from disk"""
        if os.path.exists(self.bm25_chunks_file):
            try:
                with open(self.bm25_chunks_file, 'rb') as f:
                    self.all_chunks = pickle.load(f)
                if self.all_chunks:
                    self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
                    logger.info(f"✅ Loaded BM25 retriever: {len(self.all_chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to load BM25 retriever: {e}")
    
    def save_vector_store(self) -> bool:
        """Save FAISS vector store to disk"""
        if self.vector_store is None:
            logger.warning("No FAISS vector store to save")
            return False
        try:
            self.vector_store.save_local(self.faiss_index_path)
            logger.info(f"💾 Saved FAISS vector store: {self.vector_store.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to save FAISS: {e}")
            return False
    
    def save_bm25_retriever(self) -> bool:
        """Save BM25 chunks to disk"""
        if not self.all_chunks:
            logger.warning("No BM25 chunks to save")
            return False
        try:
            with open(self.bm25_chunks_file, 'wb') as f:
                pickle.dump(self.all_chunks, f)
            
            metadata = {
                "total_chunks": len(self.all_chunks),
                "last_updated": str(__import__('datetime').datetime.now()),
                "index_path": self.bm25_index_path,
                "type": "BM25",
                "chunk_size": self.default_chunk_size,
                "chunk_overlap": self.default_chunk_overlap
            }
            with open(self.bm25_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Saved BM25 retriever: {len(self.all_chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to save BM25: {e}")
            return False
    
    def store_chunks(self, chunks: List[Document]) -> None:
        """
        Store chunks in both FAISS and BM25 indexes.
        
        Args:
            chunks: List of Document chunks to index
        """
        if not chunks:
            logger.warning("No chunks to store")
            return
        
        # Update BM25
        self.all_chunks.extend(chunks)
        self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
        logger.info(f"   BM25: {len(self.all_chunks)} total chunks")
        
        # Update FAISS
        langchain_compatible_model = JinaLangChainWrapper(self.embedding_model.model)
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, langchain_compatible_model)
            logger.info(f"✅ Created FAISS store: {len(chunks)} chunks")
        else:
            self.vector_store.add_documents(chunks)
            logger.info(f"✅ Added {len(chunks)} chunks to FAISS (total: {self.vector_store.index.ntotal})")
        
        # Save both
        self.save_vector_store()
        self.save_bm25_retriever()
    
    def ingest_document(
        self,
        file_path: str,
        document_id: str,
        filename: str,
        filetype: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """
        Load, preprocess, chunk, and index a document with optimized parameters.
        
        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            filename: Display name of the document
            filetype: Type of document (pdf, docx, etc.)
            chunk_size: Size of each chunk in characters (defaults to 500)
            chunk_overlap: Overlap between chunks (defaults to 50)
        
        Returns:
            List of created chunks
        """
        # Use optimized defaults if not provided
        if chunk_size is None:
            chunk_size = self.default_chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.default_chunk_overlap
        
        logger.info(f"📄 Processing document: {filename}")
        logger.info(f"   Chunk size: {chunk_size}, overlap: {chunk_overlap}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load
        self.doc_loader.load_pdf(file_path=file_path)
        raw_text = self.doc_loader.text
        if not raw_text:
            raise ValueError("No text extracted from document")
        logger.info(f"   Loaded: {len(raw_text)} characters")
        
        # Preprocess
        preprocessed_text = self.text_preprocessor.preprocess(raw_text)
        if not preprocessed_text:
            raise ValueError("Preprocessing removed all text")
        logger.info(f"   Preprocessed: {len(preprocessed_text)} characters")
        
        # Create document with enhanced metadata
        doc = Document(
            page_content=preprocessed_text,
            metadata={
                "source": file_path,
                "document_id": document_id,
                "filename": filename,
                "filetype": filetype,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "total_chars": len(preprocessed_text)
            }
        )
        
        # Chunk with optimized parameters
        chunks = self.text_chunker.chunk(
            doc, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Add rich metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata['document_id'] = document_id
            chunk.metadata['filename'] = filename
            chunk.metadata['filetype'] = filetype
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['chunk_size'] = chunk_size
            chunk.metadata['chunk_overlap'] = chunk_overlap
        
        logger.info(f"   Chunked: {len(chunks)} chunks")
        
        # Store
        self.store_chunks(chunks)
        logger.info(f"✅ Document processed: {filename}")
        return chunks
    
    def get_chunks_by_file_ids(self, file_ids: List[str]) -> List[Document]:
        """Retrieve chunks belonging to specific file IDs"""
        if not self.all_chunks:
            logger.warning("No chunks available in storage")
            return []
        
        file_id_set = set(str(fid) for fid in file_ids)
        filtered_chunks = [
            chunk for chunk in self.all_chunks
            if str(chunk.metadata.get('document_id')) in file_id_set
        ]
        logger.info(f"📁 Filtered {len(filtered_chunks)} chunks from {len(file_ids)} files")
        return filtered_chunks
    
    def get_all_chunks(self) -> List[Document]:
        """Get all indexed chunks"""
        return self.all_chunks
    
    def get_vector_store(self):
        """Get FAISS vector store"""
        return self.vector_store
    
    def get_bm25_retriever(self):
        """Get BM25 retriever"""
        return self.bm25_retriever
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        return {
            "faiss": {
                "total_vectors": self.vector_store.index.ntotal if self.vector_store else 0,
                "dimension": self.vector_store.index.d if self.vector_store else 0,
                "index_path": self.faiss_index_path,
                "index_type": self.faiss_index_type
            },
            "bm25": {
                "total_chunks": len(self.all_chunks),
                "index_path": self.bm25_index_path
            },
            "chunking": {
                "default_size": self.default_chunk_size,
                "default_overlap": self.default_chunk_overlap
            },
            "mmr": {
                "fetch_k": self.mmr_fetch_k,
                "lambda_mult": self.mmr_lambda_mult
            }
        }
    
    def clear_indexes(self):
        """Clear all indexes"""
        self.vector_store = None
        self.bm25_retriever = None
        self.all_chunks = []
        logger.info("🗑️ Indexes cleared")