# services/retrieval_service.py
import logging
import os
from typing import List, Optional, Dict, Any
from injector import inject
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from src.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

# Try to import reranker
try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("FlagEmbedding not installed. Reranking will be disabled.")


class RetrievalService:
    """
    Handles retrieval operations using indexes from IngestionService.
    
    Responsibilities:
    - FAISS similarity search (with MMR)
    - BM25 keyword search
    - Ensemble combination
    - Cross-encoder reranking
    - File filtering
    """
    
    @inject
    def __init__(self, ingestion_service: IngestionService):
        self.ingestion = ingestion_service
        self.reranker = None
        
        # MMR configuration
        self.mmr_fetch_k = int(os.getenv("MMR_FETCH_K", "100"))
        self.mmr_lambda_mult = float(os.getenv("MMR_LAMBDA_MULT", "0.7"))
        
        # Load reranker
        self._load_reranker()
        
        logger.info("✅ RetrievalService initialized")
        logger.info(f"   Reranker available: {self.reranker is not None}")
        logger.info(f"   MMR: fetch_k={self.mmr_fetch_k}, lambda={self.mmr_lambda_mult}")
    
    def _load_reranker(self):
        """Load reranker model"""
        if not RERANKER_AVAILABLE:
            return
        
        try:
            # Try multiple paths
            paths = [
                '/app/models/BAAI/models--BAAI--bge-reranker-v2-m3',
                '../../../models/BAAI/models--BAAI--bge-reranker-v2-m3',
                './models/BAAI/models--BAAI--bge-reranker-v2-m3'
            ]
            
            for base_path in paths:
                if os.path.exists(base_path):
                    # Check for snapshots (HuggingFace cache format)
                    snapshots_path = os.path.join(base_path, 'snapshots')
                    if os.path.exists(snapshots_path):
                        snapshots = [d for d in os.listdir(snapshots_path) 
                                   if os.path.isdir(os.path.join(snapshots_path, d))]
                        if snapshots:
                            model_path = os.path.join(snapshots_path, snapshots[0])
                            self.reranker = FlagReranker(model_path, use_fp16=True)
                            logger.info("✅ Reranker loaded from snapshot")
                            return
                    
                    # Direct load
                    self.reranker = FlagReranker(base_path, use_fp16=True)
                    logger.info("✅ Reranker loaded")
                    return
                    
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}")
            self.reranker = None
    
    # ============ CORE RETRIEVAL ============
    
    def retrieve(self, query: str, k: int = 5, file_ids: Optional[List[str]] = None) -> List[Document]:
        """
        Main retrieval pipeline:
        1. FAISS (with MMR) + BM25 ensemble
        2. Optional file filtering
        3. Reranking (cross-encoder)
        
        Args:
            query: User query
            k: Number of results to return
            file_ids: Optional list of file IDs to filter by
        
        Returns:
            List of top-k relevant Document chunks
        """
        # Step 1: Get results from FAISS and BM25
        faiss_results = self._faiss_search(query, k * 3 if file_ids else k)
        bm25_results = self._bm25_search(query, k * 3 if file_ids else k)
        
        # Step 2: Filter by file_ids if provided
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            faiss_results = [d for d in faiss_results if str(d.metadata.get('document_id')) in file_id_set]
            bm25_results = [d for d in bm25_results if str(d.metadata.get('document_id')) in file_id_set]
            logger.info(f"📁 Filtered to {len(faiss_results)} FAISS, {len(bm25_results)} BM25 results")
        
        # Step 3: Combine and deduplicate
        combined = self._combine_results(faiss_results, bm25_results)
        logger.info(f"📊 Combined {len(combined)} unique results")
        
        if not combined:
            return []
        
        # Step 4: Rerank (cross-encoder)
        if self.reranker:
            combined = self._rerank(query, combined, top_k=k)
        else:
            combined = combined[:k]
        
        logger.info(f"✅ Returning {len(combined)} results")
        return combined
    
    def _faiss_search(self, query: str, k: int) -> List[Document]:
        """FAISS search with MMR"""
        vector_store = self.ingestion.get_vector_store()
        if vector_store is None:
            return []
        
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": self.mmr_fetch_k,
                "lambda_mult": self.mmr_lambda_mult
            }
        )
        results = retriever.invoke(query)
        logger.debug(f"FAISS: {len(results)} results")
        return results
    
    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """BM25 keyword search"""
        all_chunks = self.ingestion.get_all_chunks()
        if not all_chunks:
            return []
        
        bm25 = BM25Retriever.from_documents(all_chunks)
        bm25.k = k
        results = bm25.invoke(query)
        logger.debug(f"BM25: {len(results)} results")
        return results
    
    def _combine_results(self, faiss_results: List[Document], bm25_results: List[Document]) -> List[Document]:
        """Combine and deduplicate results from both retrievers"""
        seen = set()
        combined = []
        
        # FAISS results first (semantic quality)
        for doc in faiss_results:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        
        # BM25 results (keyword quality)
        for doc in bm25_results:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        
        return combined
    
    def _rerank(self, query: str, chunks: List[Document], top_k: int) -> List[Document]:
        """Rerank chunks using cross-encoder"""
        if not self.reranker or not chunks:
            return chunks[:top_k]
        
        try:
            texts = [chunk.page_content for chunk in chunks]
            scores = self.reranker.compute_score([(query, text) for text in texts])
            
            ranked = list(zip(chunks, scores))
            ranked.sort(key=lambda x: x[1], reverse=True)
            
            # Store scores in metadata
            for chunk, score in ranked:
                chunk.metadata['reranker_score'] = score
            
            # Log top scores
            logger.info("🎯 Reranker scores:")
            for i, (chunk, score) in enumerate(ranked[:3], 1):
                preview = chunk.page_content[:100].replace('\n', ' ')
                logger.info(f"   {i}. Score={score:.4f} - {preview}...")
            
            return [chunk for chunk, _ in ranked[:top_k]]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks[:top_k]
    
    # ============ UTILITY ============
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            "reranker_available": self.reranker is not None,
            "mmr": {
                "fetch_k": self.mmr_fetch_k,
                "lambda_mult": self.mmr_lambda_mult
            }
        }