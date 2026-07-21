import logging
import os
from typing import List, Optional, Dict, Any, Tuple
from injector import inject
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from .ingestion_service import IngestionService
from .reranker import RerankerModel

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Handles retrieval operations using indexes from IngestionService.
    
    Responsibilities:
    - FAISS similarity search (with MMR)
    - BM25 keyword search
    - Ensemble combination
    - Cross-encoder reranking (optimized for speed)
    - File filtering
    - Hybrid retrieval with sparse ratio
    """
    @inject
    def __init__(self, ingestion: IngestionService, reranker_model: RerankerModel):
        self.ingestion = ingestion
        self.reranker_model = reranker_model
        
        self.mmr_fetch_k = int(os.getenv("MMR_FETCH_K", "200"))
        self.mmr_lambda_mult = float(os.getenv("MMR_LAMBDA_MULT", "0.5"))
        
        self.default_faiss_weight = float(os.getenv("FAISS_WEIGHT", "0.6"))
        self.default_bm25_weight = float(os.getenv("BM25_WEIGHT", "0.4"))
        
        self.default_faiss_k = int(os.getenv("FAISS_RETRIEVAL_K", "100"))
        self.default_bm25_k = int(os.getenv("BM25_RETRIEVAL_K", "100"))
        
        self.default_dense_k = int(os.getenv("FAISS_RETRIEVAL_K", "100"))
        self.default_sparse_k = int(os.getenv("BM25_RETRIEVAL_K", "100"))
        
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "10"))
        self.enable_reranker = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
        
        # NEW: Sparse ratio configuration (default: 0.2 = 20% of k)
        self.sparse_ratio = float(os.getenv("SPARSE_RETRIEVAL_RATIO", "0.2"))
        self.min_sparse_results = int(os.getenv("MIN_SPARSE_RESULTS", "1"))
        
        self._score_cache = self.reranker_model._score_cache
        self._cache_max_size = self.reranker_model._cache_max_size
        
        logger.info("✅ RetrievalService initialized")
        logger.info(f"   Reranker available: {self.reranker_model.is_available()}")
        logger.info(f"   Reranker batch size: {self.reranker_model.batch_size}")
        logger.info(f"   Reranker FP16: {self.reranker_model.use_fp16}")
        logger.info(f"   Reranker max length: {self.reranker_model.max_length}")
        logger.info(f"   MMR: fetch_k={self.mmr_fetch_k}, lambda={self.mmr_lambda_mult}")
        logger.info(f"   Ensemble: FAISS={self.default_faiss_weight}, BM25={self.default_bm25_weight}")
        logger.info(f"   Retrieval k: FAISS={self.default_faiss_k}, BM25={self.default_bm25_k}")
        logger.info(f"   Sparse ratio: {self.sparse_ratio} (min: {self.min_sparse_results})")
    
    def retrieve_with_score(
        self, 
        query: str, 
        k: int = 5, 
        file_ids: Optional[List[str]] = None,
        use_reranker: Optional[bool] = None
    ) -> List[Document]:
        """
        Main retrieval pipeline with optimized parameters:
        1. FAISS (with MMR) + BM25 ensemble
        2. Optional file filtering
        3. Reranking (cross-encoder) - optimized for speed
        """
        if use_reranker is None:
            use_reranker = self.enable_reranker and self.reranker_model.is_available()
        
        candidate_k = k * 3 if use_reranker else k
        if file_ids:
            candidate_k = candidate_k * 2
        
        faiss_results = self._faiss_search(query, candidate_k, file_ids)
        bm25_results = self._bm25_search(query, candidate_k, file_ids)
        
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            faiss_results = [d for d in faiss_results if str(d.metadata.get('document_id')) in file_id_set]
            bm25_results = [d for d in bm25_results if str(d.metadata.get('document_id')) in file_id_set]
            logger.info(f"📁 Filtered to {len(faiss_results)} FAISS, {len(bm25_results)} BM25 results")
        
        combined = self._combine_results(faiss_results, bm25_results)
        logger.info(f"📊 Combined {len(combined)} unique results")
        
        if not combined:
            return []
        
        if use_reranker and self.reranker_model.is_available():
            combined = self._rerank_optimized(query, combined, top_k=k)
        else:
            combined = combined[:k]
        
        logger.info(f"✅ Returning {len(combined)} results")
        return combined
    
    def retrieve_ensemble(
        self,
        query: str,
        k: int = 5,
        file_ids: Optional[List[str]] = None 
    ) -> List[Document]:
        if file_ids:
            candidate_k = k * 2
            
        faiss_results = self._faiss_search(query, candidate_k, file_ids)
        bm25_results = self._bm25_search(query, candidate_k, file_ids)
        
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            faiss_results = [d for d in faiss_results if str(d.metadata.get('document_id')) in file_id_set]
            bm25_results = [d for d in bm25_results if str(d.metadata.get('document_id')) in file_id_set]
            logger.info(f"📁 Filtered to {len(faiss_results)} FAISS, {len(bm25_results)} BM25 results")
            
        combined = self._combine_results(faiss_results, bm25_results)
        logger.info(f"📊 Combined {len(combined)} unique results")
        
        if not combined:
            return []
        
        logger.info(f"✅ Returning {len(combined)} results")
        return combined
    
    def _faiss_search_with_score(self, query: str, k: int, file_ids: Optional[List[str]] = None) -> List[Document]:
        """FAISS search with MMR and file filtering."""
        vector_store = self.ingestion.get_vector_store()
        if vector_store is None:
            return []
        
        fetch_k = min(k * 2, self.mmr_fetch_k)
        
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": self.mmr_lambda_mult
            }
        )
        results = retriever.invoke(query)
        
        # Filter by file_ids if provided
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            results = [doc for doc in results if str(doc.metadata.get('document_id')) in file_id_set]
            logger.debug(f"📁 Filtered to {len(results)} results for file_ids: {file_ids}")
        
        logger.debug(f"FAISS: {len(results)} results (k={k}, fetch_k={fetch_k})")
        return results
    
    def _faiss_search(self, query: str, k: int, file_ids: Optional[List[str]] = None) -> List[Document]:
        """FAISS search with MMR and file filtering."""
        vector_store = self.ingestion.get_vector_store()
        if vector_store is None:
            return []
        
        fetch_k = min(k * 2, self.mmr_fetch_k)
        
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": self.mmr_lambda_mult
            }
        )
        results = retriever.invoke(query)
        
        # Filter by file_ids if provided
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            results = [doc for doc in results if str(doc.metadata.get('document_id')) in file_id_set]
            logger.debug(f"📁 Filtered to {len(results)} results for file_ids: {file_ids}")
        
        logger.debug(f"FAISS: {len(results)} results (k={k}, fetch_k={fetch_k})")
        return results
    
    def _bm25_search(self, query: str, k: int, file_ids: Optional[List[str]] = None) -> List[Document]:
        """BM25 keyword search with file filtering."""
        all_chunks = self.ingestion.get_all_chunks()
        if not all_chunks:
            return []
        
        # Filter chunks by file_ids before building BM25
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            filtered_chunks = [
                chunk for chunk in all_chunks 
                if str(chunk.metadata.get('document_id')) in file_id_set
            ]
            logger.debug(f"📁 Filtered to {len(filtered_chunks)} chunks for BM25")
        else:
            filtered_chunks = all_chunks
        
        bm25 = BM25Retriever.from_documents(filtered_chunks)
        bm25.k = k
        results = bm25.invoke(query)
        logger.debug(f"BM25: {len(results)} results (k={k})")
        return results
    
    def _combine_results(self, faiss_results: List[Document], bm25_results: List[Document]) -> List[Document]:
        """Combine and deduplicate results from both retrievers"""
        seen = set()
        combined = []
        
        for doc in faiss_results:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        
        for doc in bm25_results:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        
        return combined
    
    def _rerank_optimized(self, query: str, chunks: List[Document], top_k: int) -> List[Document]:
        """
        Optimized reranking using cross-encoder with:
        - Batch processing
        - Text truncation
        - Score caching
        - Parallel processing (via batch)
        """
        if not self.reranker_model.is_available() or not chunks:
            return chunks[:top_k]
        
        try:
            rerank_limit = min(len(chunks), self.reranker_model.limit)
            chunks_to_rerank = chunks[:rerank_limit]
            
            texts = []
            for chunk in chunks_to_rerank:
                content = chunk.page_content
                if len(content) > self.reranker_model.max_length:
                    content = content[:self.reranker_model.max_length]
                texts.append(content)
            
            cached_scores = []
            chunks_to_compute = []
            chunks_to_compute_indices = []
            
            for i, text in enumerate(texts):
                cache_key = f"{query[:100]}:{text[:100]}"
                if cache_key in self._score_cache:
                    cached_scores.append(self._score_cache[cache_key])
                else:
                    chunks_to_compute.append(text)
                    chunks_to_compute_indices.append(i)
                    cached_scores.append(None)
            
            if chunks_to_compute:
                batch_size = self.reranker_model.batch_size
                computed_scores = []
                
                for i in range(0, len(chunks_to_compute), batch_size):
                    batch_texts = chunks_to_compute[i:i + batch_size]
                    batch_scores = self.reranker_model.compute_scores(query, batch_texts)
                    computed_scores.extend(batch_scores)
                
                for idx, score in zip(chunks_to_compute_indices, computed_scores):
                    cache_key = f"{query[:100]}:{texts[idx][:100]}"
                    self._score_cache[cache_key] = score
                    
                    if len(self._score_cache) > self._cache_max_size:
                        self._score_cache.pop(next(iter(self._score_cache)))
                    
                    cached_scores[idx] = score
            
            ranked = list(zip(chunks_to_rerank, cached_scores))
            ranked.sort(key=lambda x: x[1], reverse=True)
            
            if not self.reranker_model.skip_scores:
                for chunk, score in ranked:
                    chunk.metadata['reranker_score'] = score
            
            logger.info(f"🎯 Reranked {len(ranked)} chunks:")
            for i, (chunk, score) in enumerate(ranked[:3], 1):
                preview = chunk.page_content[:100].replace('\n', ' ')
                filename = chunk.metadata.get('filename', 'unknown')
                logger.info(f"   {i}. Score={score:.4f} - {preview}... (from: {filename})")
            
            return [chunk for chunk, _ in ranked[:top_k]]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks[:top_k]
    
    def rerank_with_scores(self, query: str, chunks: List[Document], top_k: int) -> tuple[List[Document], List[float]]:
        """
        Optimized reranking using cross-encoder with:
        - Batch processing
        - Text truncation
        - Score caching
        
        Returns:
            Tuple of (List[Document], List[float]) - chunks and their scores
        """
        if not self.reranker_model.is_available() or not chunks:
            return chunks[:top_k], [None] * len(chunks[:top_k])  #type: ignore
        
        try:
            rerank_limit = min(len(chunks), self.reranker_model.limit)
            chunks_to_rerank = chunks[:rerank_limit]
            
            # Prepare texts with truncation
            texts = []
            for chunk in chunks_to_rerank:
                content = chunk.page_content
                if len(content) > self.reranker_model.max_length:
                    content = content[:self.reranker_model.max_length]
                texts.append(content)
            
            # Check cache for scores
            cached_scores = []
            chunks_to_compute = []
            chunks_to_compute_indices = []
            
            for i, text in enumerate(texts):
                cache_key = f"{query[:100]}:{text[:100]}"
                if cache_key in self._score_cache:
                    cached_scores.append(self._score_cache[cache_key])
                else:
                    chunks_to_compute.append(text)
                    chunks_to_compute_indices.append(i)
                    cached_scores.append(None)
            
            # Compute scores for uncached chunks
            if chunks_to_compute:
                batch_size = self.reranker_model.batch_size
                computed_scores = []
                
                for i in range(0, len(chunks_to_compute), batch_size):
                    batch_texts = chunks_to_compute[i:i + batch_size]
                    batch_scores = self.reranker_model.compute_scores(query, batch_texts)
                    computed_scores.extend(batch_scores)
                
                # Update cache
                for idx, score in zip(chunks_to_compute_indices, computed_scores):
                    cache_key = f"{query[:100]}:{texts[idx][:100]}"
                    self._score_cache[cache_key] = score
                    
                    # Limit cache size
                    if len(self._score_cache) > self._cache_max_size:
                        self._score_cache.pop(next(iter(self._score_cache)))
                    
                    cached_scores[idx] = score
            
            # Pair chunks with scores and sort
            ranked = list(zip(chunks_to_rerank, cached_scores))
            ranked.sort(key=lambda x: x[1], reverse=True)
            
            # Store scores in metadata
            if not self.reranker_model.skip_scores:
                for chunk, score in ranked:
                    chunk.metadata['reranker_score'] = score
            
            # Log results
            logger.info(f"🎯 Reranked {len(ranked)} chunks:")
            for i, (chunk, score) in enumerate(ranked[:3], 1):
                preview = chunk.page_content[:100].replace('\n', ' ')
                filename = chunk.metadata.get('filename', 'unknown')
                logger.info(f"   {i}. Score={score:.4f} - {preview}... (from: {filename})")
            
            # Separate documents and scores
            docs = [chunk for chunk, _ in ranked[:top_k]]
            scores = [score for _, score in ranked[:top_k]]
            
            return docs, scores
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks[:top_k], [None] * len(chunks[:top_k])  #type: ignore
    
    def search_with_ensemble(
        self,
        query: str,
        k: int = 5,
        faiss_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        faiss_k: Optional[int] = None,
        bm25_k: Optional[int] = None,
        file_ids: Optional[List[str]] = None
    ) -> List[Document]:
        """Search using ensemble retriever with custom weights and file filtering."""
        faiss_weight = faiss_weight if faiss_weight is not None else self.default_faiss_weight
        bm25_weight = bm25_weight if bm25_weight is not None else self.default_bm25_weight
        faiss_k = faiss_k if faiss_k is not None else self.default_faiss_k
        bm25_k = bm25_k if bm25_k is not None else self.default_bm25_k
        
        faiss_retriever = self._get_faiss_retriever(k=faiss_k, file_ids=file_ids)
        bm25_retriever = self._get_bm25_retriever(k=bm25_k, file_ids=file_ids)
        
        if faiss_retriever is None or bm25_retriever is None:
            logger.warning("Ensemble retriever not available")
            return self._faiss_search(query, k, file_ids) or self._bm25_search(query, k, file_ids)
        
        ensemble = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[faiss_weight, bm25_weight]
        )
        
        results = ensemble.invoke(query)
        
        # Final filtering in case some slipped through
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            results = [doc for doc in results if str(doc.metadata.get('document_id')) in file_id_set]
        
        return results[:k]
    
    def _get_faiss_retriever(self, k: int = 100, file_ids: Optional[List[str]] = None):
        """Get FAISS retriever with optimized MMR."""
        vector_store = self.ingestion.get_vector_store()
        if vector_store is None:
            return None
        
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": self.mmr_fetch_k,
                "lambda_mult": self.mmr_lambda_mult
            }
        )
    
    def _get_bm25_retriever(self, k: int = 100, file_ids: Optional[List[str]] = None):
        """Get BM25 retriever with file filtering."""
        all_chunks = self.ingestion.get_all_chunks()
        if not all_chunks:
            return None
        
        # Filter chunks by file_ids if provided
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            filtered_chunks = [
                chunk for chunk in all_chunks 
                if str(chunk.metadata.get('document_id')) in file_id_set
            ]
            logger.debug(f"📁 Filtered to {len(filtered_chunks)} chunks for BM25")
        else:
            filtered_chunks = all_chunks
        
        bm25 = BM25Retriever.from_documents(filtered_chunks)
        bm25.k = k
        return bm25
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            "reranker": self.reranker_model.get_stats(),
            "mmr": {
                "fetch_k": self.mmr_fetch_k,
                "lambda_mult": self.mmr_lambda_mult
            },
            "ensemble": {
                "faiss_weight": self.default_faiss_weight,
                "bm25_weight": self.default_bm25_weight,
                "faiss_k": self.default_faiss_k,
                "bm25_k": self.default_bm25_k
            },
            "rerank": {
                "top_k": self.rerank_top_k,
                "enabled": self.enable_reranker
            },
            "sparse_ratio": {
                "ratio": self.sparse_ratio,
                "min_results": self.min_sparse_results
            }
        }
    
    def clear_cache(self):
        """Clear reranker score cache"""
        self.reranker_model.clear_cache()
    
    def get_dense_retriever(self, file_ids: Optional[List[str]] = None):
        """
        Get the dense retriever (FAISS with MMR) for direct use.
        Returns a configured retriever object.
        """
        vector_store = self.ingestion.get_vector_store()
        if vector_store is None:
            return None
        
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.mmr_fetch_k,
                "fetch_k": self.mmr_fetch_k,
                "lambda_mult": self.mmr_lambda_mult
            }
        )

    def get_sparse_retriever(self, file_ids: Optional[List[str]] = None):
        """
        Get the sparse retriever (BM25) for direct use.
        Returns a configured retriever object.
        """
        all_chunks = self.ingestion.get_all_chunks()
        if not all_chunks:
            return None
        
        # Filter by file_ids if provided
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            filtered_chunks = [
                chunk for chunk in all_chunks 
                if str(chunk.metadata.get('document_id')) in file_id_set
            ]
            logger.debug(f"📁 Filtered to {len(filtered_chunks)} chunks for BM25")
        else:
            filtered_chunks = all_chunks
        
        bm25 = BM25Retriever.from_documents(filtered_chunks)
        bm25.k = self.default_sparse_k
        return bm25

    def is_dense_available(self) -> bool:
        """Check if dense (FAISS) index is available."""
        return self.ingestion.get_vector_store() is not None

    def is_sparse_available(self) -> bool:
        """Check if sparse (BM25) index is available."""
        return self.ingestion.get_bm25_retriever() is not None

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the underlying indexes.
        """
        dense_store = self.ingestion.get_vector_store()
        sparse_store = self.ingestion.get_bm25_retriever()
        all_chunks = self.ingestion.get_all_chunks()
        
        return {
            "dense": {
                "available": dense_store is not None,
                "vectors": dense_store.index.ntotal if dense_store else 0,
                "dimension": dense_store.index.d if dense_store else 0
            },
            "sparse": {
                "available": sparse_store is not None,
                "chunks": len(all_chunks)
            },
            "config": {
                "mmr_fetch_k": self.mmr_fetch_k,
                "mmr_lambda_mult": self.mmr_lambda_mult,
                "reranker_available": self.reranker_model.is_available(),
                "sparse_ratio": self.sparse_ratio
            }
        }
    
    def retrieve_dense(
        self,
        query: str,
        k: int = 10,
        file_ids: Optional[List[str]] = None,
        use_mmr: bool = True,
        mmr_fetch_k: Optional[int] = None,
        mmr_lambda_mult: Optional[float] = None
    ) -> List[Document]:
        """
        Dense vector retrieval with configurable MMR and file filtering.
        """
        logger.info(f"🔍 Dense retrieval: {query[:100]}...")
        logger.info(f"   MMR enabled: {use_mmr}")
        if file_ids:
            logger.info(f"   📁 Filtering to file_ids: {file_ids}")
        
        # Get dense vector results with file filtering
        if use_mmr:
            dense_results = self._dense_search_with_mmr(
                query, k, 
                fetch_k=mmr_fetch_k or self.mmr_fetch_k,
                lambda_mult=mmr_lambda_mult or self.mmr_lambda_mult,
                file_ids=file_ids
            )
        else:
            dense_results = self._dense_search_similarity(
                query, k,
                file_ids=file_ids
            )
        
        # Add rank-based scores (normalized 0-1)
        for i, doc in enumerate(dense_results):
            doc.metadata['dense_score'] = 1.0 - (i / max(len(dense_results), 1))
            doc.metadata['mmr_used'] = use_mmr
        
        logger.info(f"✅ Returning {len(dense_results)} dense results")
        return dense_results

    def retrieve_dense_with_scores(
        self,
        query: str,
        k: int = 10,
        file_ids: Optional[List[str]] = None,
        use_mmr: bool = True,
        mmr_fetch_k: Optional[int] = None,
        mmr_lambda_mult: Optional[float] = None
    ) -> Tuple[List[Document], List[float]]:
        """
        Dense vector retrieval with configurable MMR returning both documents and scores.
        
        Returns:
            Tuple of (List[Document], List[float])
        """
        docs = self.retrieve_dense(
            query, k, file_ids, use_mmr, mmr_fetch_k, mmr_lambda_mult
        )
        scores = [doc.metadata.get('dense_score', 0.0) for doc in docs]
        return docs, scores

    def _dense_search_with_mmr(
        self, 
        query: str, 
        k: int, 
        fetch_k: int, 
        lambda_mult: float,
        file_ids: Optional[List[str]] = None
    ) -> List[Document]:
        """Dense search with MMR for diversity and file filtering."""
        vector_store = self.ingestion.get_vector_store()
        if vector_store is None:
            return []
        
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult
            }
        )
        results = retriever.invoke(query)
        
        # Filter by file_ids if provided
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            results = [
                doc for doc in results 
                if str(doc.metadata.get('document_id')) in file_id_set
            ]
            logger.debug(f"📁 Filtered to {len(results)} dense results for file_ids: {file_ids}")
        
        logger.debug(f"Dense (MMR): {len(results)} results (k={k}, fetch_k={fetch_k}, lambda={lambda_mult})")
        return results

    def _dense_search_similarity(
        self, 
        query: str, 
        k: int,
        file_ids: Optional[List[str]] = None
    ) -> List[Document]:
        """Simple similarity search without MMR with file filtering."""
        vector_store = self.ingestion.get_vector_store()
        if vector_store is None:
            return []
        
        # Use similarity_search_with_score to get actual scores
        results_with_scores = vector_store.similarity_search_with_score(query, k=k)
        results = [doc for doc, _ in results_with_scores]
        
        # Filter by file_ids if provided
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            results = [
                doc for doc in results 
                if str(doc.metadata.get('document_id')) in file_id_set
            ]
            logger.debug(f"📁 Filtered to {len(results)} dense results for file_ids: {file_ids}")
        
        # Store raw scores in metadata
        for doc, score in results_with_scores:
            if doc in results:  # Only store scores for filtered results
                doc.metadata['dense_score_raw'] = score
        
        logger.debug(f"Dense (Similarity): {len(results)} results (k={k})")
        return results

    def _sparse_search(self, query: str, k: int, file_ids: Optional[List[str]] = None) -> List[Document]:
        """Sparse (BM25) search with file filtering."""
        results = self._bm25_search(query, k, file_ids)
        return results

    def evaluate_retrieval_quality(
        self,
        documents: List[Document],
        threshold: float = 0.5,
        min_docs_required: int = 3,
        score_field: str = "dense_score"
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality by checking if documents meet the threshold.
        
        Args:
            documents: List of documents with scores in metadata
            threshold: Minimum score threshold (0-1)
            min_docs_required: Minimum number of documents above threshold
            score_field: Metadata field containing the score (default: "dense_score")
        
        Returns:
            Dict with:
            - passed: bool - Whether quality check passed
            - total_docs: int - Total number of documents
            - docs_above_threshold: int - Number of documents above threshold
            - docs_below_threshold: int - Number of documents below threshold
            - threshold: float - Threshold used
            - min_docs_required: int - Minimum docs required
            - avg_score: float - Average score of all documents
            - max_score: float - Maximum score
            - min_score: float - Minimum score
            - scores_above: List[float] - Scores above threshold
            - scores_below: List[float] - Scores below threshold
            - passed_documents: List[Document] - Documents that passed threshold
        """
        logger.info("=" * 60)
        logger.info("📊 Evaluating Retrieval Quality")
        logger.info("=" * 60)
        
        if not documents:
            logger.warning("   No documents to evaluate")
            return {
                "passed": False,
                "total_docs": 0,
                "docs_above_threshold": 0,
                "docs_below_threshold": 0,
                "threshold": threshold,
                "min_docs_required": min_docs_required,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "scores_above": [],
                "scores_below": [],
                "passed_documents": [],
                "reason": "No documents to evaluate"
            }
        
        # Extract scores from metadata
        scores = []
        for doc in documents:
            score = doc.metadata.get(score_field, 0.0)
            scores.append(score)
        
        # Split scores by threshold
        scores_above = [s for s in scores if s >= threshold]
        scores_below = [s for s in scores if s < threshold]
        
        # Identify documents that passed threshold
        passed_documents = [
            doc for doc in documents 
            if doc.metadata.get(score_field, 0.0) >= threshold
        ]
        
        # Calculate statistics
        total_docs = len(documents)
        docs_above = len(scores_above)
        docs_below = len(scores_below)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        
        # Check if quality passes
        passed = docs_above >= min_docs_required
        
        # Determine reason
        if passed:
            reason = f"✅ {docs_above} documents above threshold {threshold} (required: {min_docs_required})"
        else:
            reason = f"❌ Only {docs_above} documents above threshold {threshold} (required: {min_docs_required})"
        
        # Log results
        logger.info(f"   Total documents: {total_docs}")
        logger.info(f"   Threshold: {threshold}")
        logger.info(f"   Min docs required: {min_docs_required}")
        logger.info(f"   Docs above threshold: {docs_above}")
        logger.info(f"   Docs below threshold: {docs_below}")
        logger.info(f"   Score range: {min_score:.4f} - {max_score:.4f}")
        logger.info(f"   Average score: {avg_score:.4f}")
        logger.info(f"   Result: {reason}")
        
        if passed:
            logger.info("   ✅ QUALITY CHECK PASSED")
        else:
            logger.info("   ❌ QUALITY CHECK FAILED")
        
        logger.info("=" * 60)
        
        return {
            "passed": passed,
            "total_docs": total_docs,
            "docs_above_threshold": docs_above,
            "docs_below_threshold": docs_below,
            "threshold": threshold,
            "min_docs_required": min_docs_required,
            "avg_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "scores_above": scores_above,
            "scores_below": scores_below,
            "passed_documents": passed_documents,
            "reason": reason
        }

    def filter_by_threshold(
        self,
        documents: List[Document],
        threshold: float = 0.5,
        score_field: str = "dense_score"
    ) -> List[Document]:
        """
        Filter documents by threshold and return only those above it.
        
        Args:
            documents: List of documents with scores in metadata
            threshold: Minimum score threshold (0-1)
            score_field: Metadata field containing the score
        
        Returns:
            List of documents with scores >= threshold
        """
        filtered = [
            doc for doc in documents
            if doc.metadata.get(score_field, 0.0) >= threshold
        ]
        
        logger.info(f"🔍 Filtered {len(filtered)} documents above threshold {threshold}")
        return filtered

    def get_quality_stats(
        self,
        documents: List[Document],
        score_field: str = "dense_score"
    ) -> Dict[str, Any]:
        """
        Get quality statistics without threshold evaluation.
        
        Args:
            documents: List of documents with scores
            score_field: Metadata field containing the score
        
        Returns:
            Dict with quality statistics
        """
        if not documents:
            return {
                "total_docs": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "score_distribution": {}
            }
        
        scores = [doc.metadata.get(score_field, 0.0) for doc in documents]
        
        # Create score distribution bins
        bins = {
            "0.9-1.0": 0,
            "0.8-0.9": 0,
            "0.7-0.8": 0,
            "0.6-0.7": 0,
            "0.5-0.6": 0,
            "0.4-0.5": 0,
            "0.3-0.4": 0,
            "0.2-0.3": 0,
            "0.1-0.2": 0,
            "0.0-0.1": 0
        }
        
        for score in scores:
            if score >= 0.9:
                bins["0.9-1.0"] += 1
            elif score >= 0.8:
                bins["0.8-0.9"] += 1
            elif score >= 0.7:
                bins["0.7-0.8"] += 1
            elif score >= 0.6:
                bins["0.6-0.7"] += 1
            elif score >= 0.5:
                bins["0.5-0.6"] += 1
            elif score >= 0.4:
                bins["0.4-0.5"] += 1
            elif score >= 0.3:
                bins["0.3-0.4"] += 1
            elif score >= 0.2:
                bins["0.2-0.3"] += 1
            elif score >= 0.1:
                bins["0.1-0.2"] += 1
            else:
                bins["0.0-0.1"] += 1
        
        return {
            "total_docs": len(documents),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "score_distribution": bins
        }

    # ===================== NEW HYBRID RETRIEVAL WITH SPARSE RATIO =====================

    def retrieve_hybrid_with_sparse_ratio(
        self,
        query: str,
        k: int = 20,
        file_ids: Optional[List[str]] = None,
        use_reranker: Optional[bool] = None,
        sparse_ratio: Optional[float] = None,
        min_sparse: Optional[int] = None,
        use_mmr: bool = True,
        mmr_fetch_k: Optional[int] = None,
        mmr_lambda_mult: Optional[float] = None
    ) -> List[Document]:
        """
        Hybrid retrieval that combines dense and sparse results with a configurable ratio.
        
        This method is designed to be called AFTER quality check passes.
        It retrieves:
        1. Dense (FAISS) results: (k - sparse_count) documents
        2. Sparse (BM25) results: sparse_count documents (calculated from sparse_ratio)
        
        Args:
            query: User query
            k: Total number of results to return (default: 20)
            file_ids: Optional list of file IDs to filter by
            use_reranker: Whether to use reranker (default: from env)
            sparse_ratio: Ratio of sparse results (0.0-1.0, default: from env 0.2)
            min_sparse: Minimum number of sparse results (default: from env 1)
            use_mmr: Whether to use MMR for dense retrieval (default: True)
            mmr_fetch_k: Number of candidates for MMR
            mmr_lambda_mult: MMR lambda parameter
        
        Returns:
            List of Document objects (combined dense + sparse results)
        """
        # Set defaults
        if use_reranker is None:
            use_reranker = self.enable_reranker and self.reranker_model.is_available()
        
        sparse_ratio = sparse_ratio if sparse_ratio is not None else self.sparse_ratio
        min_sparse = min_sparse if min_sparse is not None else self.min_sparse_results
        
        # Calculate number of sparse results
        sparse_count = max(min_sparse, int(k * sparse_ratio))
        dense_count = k - sparse_count
        
        # Ensure we don't exceed k
        if dense_count < 0:
            dense_count = 0
            sparse_count = k
        
        logger.info("=" * 60)
        logger.info(f"🔍 HYBRID RETRIEVAL (After Quality Pass)")
        logger.info("=" * 60)
        logger.info(f"   Total k: {k}")
        logger.info(f"   Dense count: {dense_count} ({dense_count/k*100:.1f}%)")
        logger.info(f"   Sparse count: {sparse_count} ({sparse_count/k*100:.1f}%)")
        logger.info(f"   Sparse ratio: {sparse_ratio}")
        logger.info(f"   Query: {query[:100]}...")
        
        # Perform dense retrieval
        dense_results = []
        if dense_count > 0:
            logger.info(f"   📊 Dense retrieval (k={dense_count})...")
            dense_results = self.retrieve_dense(
                query=query,
                k=dense_count,
                file_ids=file_ids,
                use_mmr=use_mmr,
                mmr_fetch_k=mmr_fetch_k,
                mmr_lambda_mult=mmr_lambda_mult
            )
            logger.info(f"   ✅ Dense retrieved: {len(dense_results)} documents")
        else:
            logger.info("   ⏭️  Dense retrieval skipped (dense_count=0)")
        
        # Perform sparse retrieval
        sparse_results = []
        if sparse_count > 0:
            logger.info(f"   📊 Sparse retrieval (k={sparse_count})...")
            sparse_results = self.retrieve_sparse(
                query=query,
                k=sparse_count,
                file_ids=file_ids
            )
            logger.info(f"   ✅ Sparse retrieved: {len(sparse_results)} documents")
        else:
            logger.info("   ⏭️  Sparse retrieval skipped (sparse_count=0)")
        
        # Combine results
        combined = self._combine_results_with_priority(
            dense_results, 
            sparse_results,
            dense_count,
            sparse_count
        )
        
        logger.info(f"   📊 Combined: {len(combined)} unique documents")
        
        # Apply reranking if enabled
        if use_reranker and self.reranker_model.is_available() and combined:
            logger.info(f"   🔄 Reranking {len(combined)} documents...")
            combined = self._rerank_optimized(query, combined, top_k=k)
            logger.info(f"   ✅ Reranked to {len(combined)} documents")
        else:
            # If no reranker, just return top k
            combined = combined[:k]
        
        # Final score assignment
        combined = self._assign_hybrid_scores(combined)
        
        logger.info(f"✅ Returning {len(combined)} hybrid results")
        logger.info("=" * 60)
        
        return combined

    def retrieve_sparse(
        self,
        query: str,
        k: int = 4,
        file_ids: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Sparse (BM25) retrieval with file filtering.
        """
        logger.info(f"🔍 Sparse retrieval: {query[:100]}...")
        if file_ids:
            logger.info(f"   📁 Filtering to file_ids: {file_ids}")
        
        # Get BM25 results with file filtering
        sparse_results = self._sparse_search(query, k, file_ids)
        
        # Add rank-based scores (normalized 0-1)
        for i, doc in enumerate(sparse_results):
            doc.metadata['sparse_score'] = 1.0 - (i / max(len(sparse_results), 1))
            doc.metadata['retrieval_type'] = 'sparse'
        
        logger.info(f"✅ Returning {len(sparse_results)} sparse results")
        return sparse_results

    def retrieve_sparse_with_scores(
        self,
        query: str,
        k: int = 4,
        file_ids: Optional[List[str]] = None
    ) -> Tuple[List[Document], List[float]]:
        """
        Sparse retrieval returning both documents and scores.
        
        Returns:
            Tuple of (List[Document], List[float])
        """
        docs = self.retrieve_sparse(query, k, file_ids)
        scores = [doc.metadata.get('sparse_score', 0.0) for doc in docs]
        return docs, scores

    def _combine_results_with_priority(
        self,
        dense_results: List[Document],
        sparse_results: List[Document],
        dense_target: int,
        sparse_target: int
    ) -> List[Document]:
        """
        Combine dense and sparse results with priority.
        
        Strategy:
        1. Take all dense results (up to dense_target)
        2. Fill remaining slots with sparse results
        3. Deduplicate using content similarity
        
        Args:
            dense_results: List of dense retrieval results
            sparse_results: List of sparse retrieval results
            dense_target: Target number of dense results
            sparse_target: Target number of sparse results
        
        Returns:
            Combined list of documents
        """
        seen = set()
        combined = []
        
        # First, add dense results (prioritized)
        for doc in dense_results[:dense_target]:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                doc.metadata['retrieval_type'] = 'dense'
                combined.append(doc)
        
        # Then, add sparse results to fill remaining slots
        remaining = dense_target + sparse_target - len(combined)
        for doc in sparse_results[:sparse_target]:
            if len(combined) >= dense_target + sparse_target:
                break
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                doc.metadata['retrieval_type'] = 'sparse'
                combined.append(doc)
        
        logger.debug(f"Combined: {len(combined)} results (dense target: {dense_target}, sparse target: {sparse_target})")
        return combined

    def _assign_hybrid_scores(self, documents: List[Document]) -> List[Document]:
        """
        Assign hybrid scores to documents based on their retrieval type and position.
        
        Uses a combination of:
        - Position-based score (higher for earlier results)
        - Retrieval type bonus (dense gets slight preference)
        """
        for i, doc in enumerate(documents):
            # Position-based score (0-1)
            position_score = 1.0 - (i / max(len(documents), 1))
            
            # Type bonus
            retrieval_type = doc.metadata.get('retrieval_type', 'unknown')
            if retrieval_type == 'dense':
                type_bonus = 0.05  # Small preference for dense
            elif retrieval_type == 'sparse':
                type_bonus = 0.0
            else:
                type_bonus = 0.0
            
            # Combined hybrid score
            hybrid_score = min(1.0, position_score + type_bonus)
            doc.metadata['hybrid_score'] = hybrid_score
            doc.metadata['retrieval_position'] = i
        
        return documents

    def retrieve(
        self, 
        query: str, 
        k: int = 5, 
        file_ids: Optional[List[str]] = None,
        use_reranker: Optional[bool] = None,
        use_mmr: bool = True,
        mmr_fetch_k: Optional[int] = None,
        mmr_lambda_mult: Optional[float] = None,
        use_hybrid: bool = False,
        sparse_ratio: Optional[float] = None
    ) -> List[Document]:
        """
        Main retrieval pipeline with configurable options.
        
        Args:
            query: User query
            k: Number of results to return
            file_ids: Optional list of file IDs to filter by
            use_reranker: Whether to use reranker (default: from env)
            use_mmr: Whether to use MMR (default: True)
            mmr_fetch_k: Number of candidates for MMR
            mmr_lambda_mult: MMR lambda parameter
            use_hybrid: Whether to use hybrid retrieval with sparse ratio (default: False)
            sparse_ratio: Ratio of sparse results (only used if use_hybrid=True)
        """
        # If hybrid mode is enabled, use the hybrid retrieval method
        if use_hybrid:
            logger.info("🔄 Using hybrid retrieval with sparse ratio")
            return self.retrieve_hybrid_with_sparse_ratio(
                query=query,
                k=k,
                file_ids=file_ids,
                use_reranker=use_reranker,
                sparse_ratio=sparse_ratio,
                use_mmr=use_mmr,
                mmr_fetch_k=mmr_fetch_k,
                mmr_lambda_mult=mmr_lambda_mult
            )
        
        # Original retrieval pipeline
        if use_reranker is None:
            use_reranker = self.enable_reranker and self.reranker_model.is_available()
        
        candidate_k = k * 3 if use_reranker else k
        if file_ids:
            candidate_k = candidate_k * 2
        
        # Get results with configurable MMR and file filtering
        if use_mmr:
            dense_results = self._dense_search_with_mmr(
                query, candidate_k,
                fetch_k=mmr_fetch_k or self.mmr_fetch_k,
                lambda_mult=mmr_lambda_mult or self.mmr_lambda_mult,
                file_ids=file_ids
            )
        else:
            dense_results = self._dense_search_similarity(
                query, candidate_k,
                file_ids=file_ids
            )
        
        sparse_results = self._sparse_search(query, candidate_k, file_ids)
        
        # Final filtering (safety check)
        if file_ids:
            file_id_set = set(str(fid) for fid in file_ids)
            dense_results = [d for d in dense_results if str(d.metadata.get('document_id')) in file_id_set]
            sparse_results = [d for d in sparse_results if str(d.metadata.get('document_id')) in file_id_set]
            logger.info(f"📁 Filtered to {len(dense_results)} Dense, {len(sparse_results)} Sparse results")
        
        combined = self._combine_results(dense_results, sparse_results)
        logger.info(f"📊 Combined {len(combined)} unique results")
        
        if not combined:
            return []
        
        if use_reranker and self.reranker_model.is_available():
            combined = self._rerank_optimized(query, combined, top_k=k)
        else:
            combined = combined[:k]
        
        logger.info(f"✅ Returning {len(combined)} results")
        return combined