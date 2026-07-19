import logging
from typing import Dict, Any, List, AsyncGenerator
from injector import inject
from langchain_core.documents import Document

from src.services.coreference_resolver import CoreferenceResolver
from src.services.query_rewriting_service import QueryRewritingService
from src.services.retrieval_service import RetrievalService
from src.services.hyde_service import HyDEService
from src.services.generation_service import GenerationService

logger = logging.getLogger(__name__)


class OrchestratorNodes:
    """Nodes for the LangGraph orchestration pipeline."""
    @inject
    def __init__(
        self,
        coreference_resolver: CoreferenceResolver,
        query_rewriting_service: QueryRewritingService,
        retrieval_service: RetrievalService,
        hyde_service: HyDEService,
        generation_service: GenerationService
    ):
        self.coreference_resolver = coreference_resolver
        self.query_rewriting_service = query_rewriting_service
        self.retrieval_service = retrieval_service
        self.hyde_service = hyde_service
        self.generation_service = generation_service
    
    # ============================================================
    # NODE 1: Coreference Resolution
    # ============================================================
    async def coreference_resolution(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        history = state.get("history", [])
        
        resolved_query = await self.coreference_resolver.resolve(query, history)
        coreference_used = resolved_query != query
        
        return {
            "resolved_query": resolved_query,
            "coreference_used": coreference_used
        }
    
    # ============================================================
    # NODE 2: Query Analysis
    # ============================================================
    async def query_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("resolved_query", state.get("query", ""))
        
        analysis = await self.query_rewriting_service.analyze_query(query)
        
        logger.info(f"📊 Query Analysis: {analysis}")
        
        return {
            "query_state": analysis["query_state"],
            "query_length": analysis["query_length"],
            "needs_rewriting": analysis["needs_rewriting"]
        }
    
    # ============================================================
    # NODE 3: Query Rewriting (Expand or Decompose)
    # ============================================================
    async def query_rewriting(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("resolved_query", state.get("query", ""))
        
        result = await self.query_rewriting_service.rewrite_query(query)
        
        logger.info(f"✏️ Query Rewriting: {result}")
        
        return {
            "rewritten_queries": result["rewritten_queries"],
            "query_state": result["query_state"],
            "query_length": result["query_length"],
            "needs_rewriting": result["needs_rewriting"]
        }
    
    # ============================================================
    # NODE 4: HyDE Generation
    # ============================================================
    async def hyde_generation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate HyDE document from the query.
        Only triggered when use_hyde is True and quality is poor.
        """
        logger.info("=" * 60)
        logger.info("🔮 [Node 4] HyDE Generation")
        logger.info("=" * 60)
        
        use_hyde = state.get("use_hyde", False)
        
        if not use_hyde:
            logger.info("   ⏭️ HyDE disabled (use_hyde=False)")
            return {
                "hyde_query": state.get("resolved_query", state.get("query", "")),
                "hyde_used": False
            }
        
        query = state.get("resolved_query", state.get("query", ""))
        
        result = await self.hyde_service.generate_hyde(query)
        
        logger.info(f"   HyDE generated: {result['hyde_document'][:150]}...")
        
        return {
            "hyde_query": result["hyde_document"],
            "hyde_used": result["hyde_used"],
            "hyde_original": result["original_query"]
        }
    
    # ============================================================
    # NODE 5: Integrated Dense Retrieval with MMR
    # ============================================================
    async def dense_retrieval(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrated dense retrieval with MMR based on parameter from the start.
        Uses MMR if use_mmr=True, otherwise uses similarity search.
        """
        logger.info("=" * 60)
        logger.info("🔍 [Node 5] Integrated Dense Retrieval")
        logger.info("=" * 60)
        
        # Check if HyDE was used
        hyde_used = state.get("hyde_used", False)
        hyde_query = state.get("hyde_query", "")
        
        # Get queries to search with
        if hyde_used and hyde_query:
            queries = [hyde_query]
            logger.info("Using HyDE query for retrieval")
        else:
            queries = state.get("rewritten_queries", [])
            if not queries:
                queries = [state.get("resolved_query", state.get("query", ""))]
        
        file_ids = state.get("file_ids", None)
        retrieval_k = state.get("retrieval_k", 100)
        
        # Get MMR settings from state - USE FROM THE START
        use_mmr = state.get("use_mmr", False)
        mmr_fetch_k = state.get("mmr_fetch_k", 200)
        mmr_lambda_mult = state.get("mmr_lambda_mult", 0.5)
        
        logger.info(f"   Queries: {len(queries)}")
        logger.info(f"   Retrieval k: {retrieval_k}")
        logger.info(f"   File IDs: {file_ids if file_ids else 'All'}")
        logger.info(f"   MMR enabled (from start): {use_mmr}")
        if use_mmr:
            logger.info(f"   MMR fetch_k: {mmr_fetch_k}, lambda: {mmr_lambda_mult}")
        
        all_results = []
        score_map: Dict[str, float] = {}
        
        for query in queries:
            logger.info(f"   Query: {query[:80]}...")
            
            # Use MMR based on parameter from the start
            results = self.retrieval_service.retrieve_dense(
                query=query,
                k=retrieval_k,
                file_ids=file_ids,
                use_mmr=use_mmr,
                mmr_fetch_k=mmr_fetch_k,
                mmr_lambda_mult=mmr_lambda_mult
            )
            
            logger.info(f"      Retrieved {len(results)} results")
            
            for doc in results:
                key = doc.page_content[:200]
                score = doc.metadata.get('dense_score', 0.0)
                if key not in score_map or score > score_map[key]:
                    score_map[key] = score
                all_results.append(doc)
        
        seen = {}
        unique_results = []
        for doc in all_results:
            key = doc.page_content[:200]
            if key not in seen:
                seen[key] = doc
                doc.metadata['dense_score'] = score_map.get(key, 0.0)
                doc.metadata['hyde_used'] = hyde_used
                doc.metadata['mmr_used'] = use_mmr
                unique_results.append(doc)
        
        unique_results.sort(
            key=lambda x: x.metadata.get('dense_score', 0.0),
            reverse=True
        )
        
        scores = [doc.metadata.get('dense_score', 0.0) for doc in unique_results]
        
        logger.info(f"   Total unique: {len(unique_results)}")
        if scores:
            logger.info(f"   Score range: {min(scores):.4f} - {max(scores):.4f}")
            logger.info(f"   Avg score: {sum(scores)/len(scores):.4f}")
        
        return {
            "dense_retrieved_chunks": unique_results,
            "dense_scores": scores,
            "dense_score_map": score_map,
            "hyde_used": hyde_used,
            "retrieval_source": "hyde" if hyde_used else "dense",
            "mmr_used": use_mmr,
            "mmr_lambda_mult": mmr_lambda_mult
        }
    
    # ============================================================
    # NODE 6: Quality Evaluation + Sparse Attachment
    # ============================================================
    async def quality_evaluation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quality evaluation: Check if documents meet threshold.
        If quality passes, attach sparse results to the dense results.
        """
        logger.info("=" * 60)
        logger.info("📊 [Node 6] Quality Evaluation + Sparse Attachment")
        logger.info("=" * 60)
        
        chunks = state.get("dense_retrieved_chunks", [])
        
        threshold = state.get("similarity_threshold", 0.5)
        min_docs_required = state.get("min_docs_required", 3)
        sparse_ratio = state.get("sparse_ratio", 0.2)
        total_k = state.get("retrieval_total_k", 20)
        query = state.get("resolved_query", state.get("query", ""))
        file_ids = state.get("file_ids", None)
        
        logger.info(f"   Total dense chunks: {len(chunks)}")
        logger.info(f"   Threshold: {threshold}")
        logger.info(f"   Min docs required: {min_docs_required}")
        logger.info(f"   Sparse ratio: {sparse_ratio}")
        logger.info(f"   Total K for final: {total_k}")
        
        if not chunks:
            logger.warning("   No chunks to evaluate")
            return {
                "quality_passed": False,
                "quality_reason": "No chunks retrieved",
                "total_chunks": 0,
                "docs_above_threshold": 0,
                "docs_below_threshold": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "threshold_used": threshold,
                "min_docs_required": min_docs_required,
                "chunks_for_reranker": chunks,
                "should_use_hybrid": False,
                "sparse_results_attached": False
            }
        
        # Evaluate quality (check only, no filtering)
        quality_result = self.retrieval_service.evaluate_retrieval_quality(
            documents=chunks,
            threshold=threshold,
            min_docs_required=min_docs_required,
            score_field="dense_score"
        )
        
        logger.info(f"   Quality Result: {quality_result['reason']}")
        
        # Initialize variables
        final_chunks = chunks
        sparse_results = []
        sparse_attached = False
        
        # If quality passed, attach sparse results
        if quality_result["passed"]:
            logger.info("   🎯 Quality PASSED → Attaching sparse results")
            
            # Calculate sparse count
            sparse_count = max(1, int(total_k * sparse_ratio))
            logger.info(f"   Sparse count: {sparse_count}")
            
            # Perform sparse retrieval
            sparse_results = self.retrieval_service.retrieve_sparse(
                query=query,
                k=sparse_count,
                file_ids=file_ids
            )
            
            logger.info(f"   Sparse retrieved: {len(sparse_results)} documents")
            
            # Attach sparse results to dense results
            if sparse_results:
                # Combine dense + sparse
                combined = self._combine_dense_sparse(chunks, sparse_results)
                final_chunks = combined
                sparse_attached = True
                logger.info(f"   ✅ Combined: {len(final_chunks)} total chunks (Dense: {len(chunks)}, Sparse: {len(sparse_results)})")
            else:
                logger.info("   ⚠️ No sparse results to attach, using dense only")
        else:
            logger.info("   ⚠️ Quality FAILED → Using dense only (no sparse attachment)")
        
        return {
            "quality_passed": quality_result["passed"],
            "quality_reason": quality_result["reason"],
            "total_chunks": quality_result["total_docs"],
            "docs_above_threshold": quality_result["docs_above_threshold"],
            "docs_below_threshold": quality_result["docs_below_threshold"],
            "avg_score": quality_result["avg_score"],
            "max_score": quality_result["max_score"],
            "min_score": quality_result["min_score"],
            "threshold_used": quality_result["threshold"],
            "min_docs_required": quality_result["min_docs_required"],
            "chunks_for_reranker": final_chunks,
            "scores_above": quality_result["scores_above"],
            "scores_below": quality_result["scores_below"],
            "passed_documents": quality_result["passed_documents"],
            "hyde_used": state.get("hyde_used", False),
            "should_use_hybrid": quality_result["passed"],
            "sparse_results_attached": sparse_attached,
            "sparse_count_attached": len(sparse_results) if sparse_results else 0
        }
    
    def _combine_dense_sparse(self, dense_results: List[Document], sparse_results: List[Document]) -> List[Document]:
        """
        Combine dense and sparse results with deduplication.
        Dense results take priority (they come first).
        """
        seen = set()
        combined = []
        
        # Add dense results first (priority)
        for doc in dense_results:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                doc.metadata['retrieval_type'] = 'dense'
                combined.append(doc)
        
        # Add sparse results (fill remaining)
        for doc in sparse_results:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                doc.metadata['retrieval_type'] = 'sparse'
                combined.append(doc)
        
        logger.debug(f"Combined: {len(combined)} results (Dense: {len(dense_results)}, Sparse: {len(sparse_results)})")
        return combined
    
    # ============================================================
    # NODE 7: Rerank (All documents - Dense + Sparse if attached)
    # ============================================================
    async def rerank(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rerank ALL retrieved documents using cross-encoder.
        Works with dense only or dense+sparse combined results.
        """
        logger.info("=" * 60)
        logger.info("💰 [Node 7] Reranking")
        logger.info("=" * 60)
        
        query = state.get("query", "")
        top_k = state.get("top_k", 10)
        
        # Get chunks from quality evaluation
        chunks = state.get("chunks_for_reranker", [])
        
        if not chunks:
            chunks = state.get("dense_retrieved_chunks", [])
            logger.info("   Using dense_retrieved_chunks as fallback")
        
        if not chunks:
            logger.warning("   No chunks to rerank")
            return {
                "reranked_chunks": [],
                "rerank_scores": [],
                "rerank_used": False
            }
        
        # Check if sparse was attached
        sparse_attached = state.get("sparse_results_attached", False)
        sparse_count = state.get("sparse_count_attached", 0)
        dense_count = len(chunks) - sparse_count
        
        if sparse_attached:
            logger.info(f"   Reranking DENSE+SPARSE results: {len(chunks)} chunks (Dense: {dense_count}, Sparse: {sparse_count})")
        else:
            logger.info(f"   Reranking DENSE results: {len(chunks)} chunks")
        
        logger.info(f"   → Top {top_k}")
        
        # Log MMR and HyDE usage
        mmr_used = state.get("mmr_used", False)
        hyde_used = state.get("hyde_used", False)
        
        if mmr_used:
            logger.info("   ℹ️ MMR was used for retrieval")
        if hyde_used:
            logger.info("   ℹ️ HyDE was used for retrieval")
        
        try:
            # Rerank ALL chunks
            docs, scores = self.retrieval_service.rerank_with_scores(
                query=query,
                chunks=chunks,
                top_k=top_k
            )
            
            logger.info(f"   ✅ Reranked {len(docs)} chunks")
            
            # Log top results
            for i, (doc, score) in enumerate(zip(docs[:3], scores[:3]), 1):
                preview = doc.page_content[:80].replace('\n', ' ')
                filename = doc.metadata.get('filename', 'unknown')
                retrieval_type = doc.metadata.get('retrieval_type', 'dense')
                mmr_tag = " [MMR]" if doc.metadata.get('mmr_used', False) else ""
                hyde_tag = " [HyDE]" if doc.metadata.get('hyde_used', False) else ""
                logger.info(f"      {i}. [{retrieval_type.upper()}]{mmr_tag}{hyde_tag} Score={score:.4f} - {preview}... (from: {filename})")
            
            return {
                "reranked_chunks": docs,
                "rerank_scores": scores,
                "rerank_used": True,
                "rerank_count": len(docs),
                "hyde_used": hyde_used,
                "mmr_used": mmr_used,
                "sparse_attached": sparse_attached,
                "reranked_retrieval_types": [
                    doc.metadata.get('retrieval_type', 'dense') for doc in docs
                ]
            }
            
        except Exception as e:
            logger.error(f"   ❌ Reranking failed: {e}")
            chunks.sort(
                key=lambda x: x.metadata.get('dense_score', 0.0), 
                reverse=True
            )
            return {
                "reranked_chunks": chunks[:top_k],
                "rerank_scores": [
                    doc.metadata.get('dense_score', 0.0) 
                    for doc in chunks[:top_k]
                ],
                "rerank_used": False,
                "rerank_error": str(e),
                "hyde_used": hyde_used,
                "mmr_used": mmr_used,
                "sparse_attached": sparse_attached
            }
    
    # ============================================================
    # NODE 8: Generation (Answer Generation with Citations)
    # ============================================================
    async def generation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final answer with source citations.
        """
        logger.info("=" * 60)
        logger.info("🤖 [Node 8] Generation with Citations")
        logger.info("=" * 60)
        
        query = state.get("query", "")
        conversation_history = state.get("history", [])
        
        # Get reranked chunks
        documents = state.get("reranked_chunks", [])
        if not documents:
            documents = state.get("chunks_for_reranker", [])
            logger.info("   Using chunks_for_reranker as fallback")
        
        if not documents:
            documents = state.get("dense_retrieved_chunks", [])
            logger.info("   Using dense_retrieved_chunks as fallback")
        
        if not documents:
            logger.warning("   No documents to generate answer from")
            return {
                "generated_answer": "",
                "generation_used": False,
                "sources": [],
                "sources_used": [],
                "citations": []
            }
        
        # Log retrieval method used
        sparse_attached = state.get("sparse_attached", False)
        mmr_used = state.get("mmr_used", False)
        
        if sparse_attached:
            sparse_count = state.get("sparse_count_attached", 0)
            dense_count = len(documents) - sparse_count
            mmr_tag = " [with MMR]" if mmr_used else ""
            logger.info(f"   Generating from DENSE+SPARSE results{mmr_tag}: {len(documents)} chunks (Dense: {dense_count}, Sparse: {sparse_count})")
        else:
            mmr_tag = " [with MMR]" if mmr_used else ""
            logger.info(f"   Generating from DENSE results{mmr_tag}: {len(documents)} chunks")
        
        # Generate answer with citations
        result = await self.generation_service.generate(
            query=query,
            documents=documents,
            conversation_history=conversation_history,
            include_sources=True
        )
        
        logger.info(f"   Answer generated with {len(result.get('citations', []))} citations")
        
        # Add retrieval method info
        result["retrieval_method"] = "hybrid" if sparse_attached else "dense"
        result["mmr_used"] = mmr_used
        result["sparse_attached"] = sparse_attached
        
        return {
            "generated_answer": result["answer"],
            "generation_used": result["generation_used"],
            "sources": result["sources"],
            "sources_used": result["sources_used"],
            "citations": result["citations"],
            "retrieval_method": result["retrieval_method"],
            "mmr_used": mmr_used,
            "sparse_attached": sparse_attached
        }
    
    # ============================================================
    # NODE 8 STREAM: Generation Streaming (for streaming responses)
    # ============================================================
    async def generation_stream(
        self, 
        state: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate final answer with streaming.
        
        Yields DICT events:
        - {"type": "chunk", "chunk": str, "chunk_index": int, "is_last": bool}
        - {"type": "sources", "sources": List[Dict], "sources_text": str}
        - {"type": "complete", "full_answer": str, "metadata": Dict}
        - {"type": "error", "error": str}
        """
        logger.info("=" * 60)
        logger.info("🤖 [Node 8 Stream] Generation with Citations (Streaming)")
        logger.info("=" * 60)
        
        query = state.get("query", "")
        conversation_history = state.get("history", [])
        
        # Get reranked chunks
        documents = state.get("reranked_chunks", [])
        if not documents:
            documents = state.get("chunks_for_reranker", [])
            logger.info("   Using chunks_for_reranker as fallback")
        
        if not documents:
            documents = state.get("dense_retrieved_chunks", [])
            logger.info("   Using dense_retrieved_chunks as fallback")
        
        if not documents:
            logger.warning("   No documents to generate answer from")
            yield {
                "type": "error",
                "error": "No documents available for generation"
            }
            return
        
        # Log retrieval method used
        sparse_attached = state.get("sparse_attached", False)
        mmr_used = state.get("mmr_used", False)
        
        if sparse_attached:
            sparse_count = state.get("sparse_count_attached", 0)
            dense_count = len(documents) - sparse_count
            mmr_tag = " [with MMR]" if mmr_used else ""
            logger.info(f"   Generating from DENSE+SPARSE results{mmr_tag}: {len(documents)} chunks (Dense: {dense_count}, Sparse: {sparse_count})")
        else:
            mmr_tag = " [with MMR]" if mmr_used else ""
            logger.info(f"   Generating from DENSE results{mmr_tag}: {len(documents)} chunks")
        
        # Stream the generation
        full_answer = ""
        sources = []
        citations = []
        sources_used = []
        
        async for stream_event in self.generation_service.generate_stream(
            query=query,
            documents=documents,
            conversation_history=conversation_history,
            max_sources=5
        ):
            event_type = stream_event.get("type")
            
            if event_type == "chunk":
                chunk_text = stream_event.get("chunk", "")
                chunk_index = stream_event.get("chunk_index", 0)
                full_answer += chunk_text
                
                yield {
                    "type": "chunk",
                    "chunk": chunk_text,
                    "chunk_index": chunk_index,
                    "is_last": stream_event.get("is_last", False),
                    "metadata": {
                        "total_so_far": len(full_answer)
                    }
                }
                
            elif event_type == "sources":
                sources = stream_event.get("sources", [])
                citations = stream_event.get("citations", [])
                sources_used = stream_event.get("sources_used", [])
                
                logger.info(f"   📚 Sources: {len(sources)} sources, {len(citations)} citations")
                
                yield {
                    "type": "sources",
                    "sources": sources,
                    "sources_text": stream_event.get("sources_text", ""),
                    "citations": citations,
                    "sources_used": sources_used,
                    "metadata": stream_event.get("metadata", {})
                }
                
            elif event_type == "complete":
                metadata = stream_event.get("metadata", {})
                logger.info("   ✅ Generation streaming completed")
                logger.info(f"      Answer length: {len(full_answer)} chars")
                logger.info(f"      Sources: {metadata.get('source_count', 0)}")
                logger.info(f"      Citations: {metadata.get('citation_count', 0)}")
                
                yield {
                    "type": "complete",
                    "full_answer": full_answer,
                    "full_answer_with_sources": stream_event.get("full_answer_with_sources", full_answer),
                    "metadata": metadata,
                    "sources": sources,
                    "sources_used": sources_used,
                    "citations": citations,
                    "generation_used": True
                }
                
            elif event_type == "error":
                logger.error(f"   ❌ Generation error: {stream_event.get('error')}")
                yield {
                    "type": "error",
                    "error": stream_event.get("error", "Unknown generation error")
                }
                return