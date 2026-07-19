import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
from injector import inject
from langgraph.graph import StateGraph, END
from .orchestrator_nodes import OrchestratorNodes
from .orchestrator_state import OrchestratorState

logger = logging.getLogger(__name__)


class OrchestratorGraph:
    """
    Orchestrates the conversational RAG pipeline using LangGraph.
    
    Pipeline Flow:
    1. Coreference Resolution - Replace pronouns with entities
    2. Query Analysis - Analyze query length
    3. Query Rewriting - Expand short or decompose long queries
    4. HyDE Generation - Generate hypothetical document (if use_hyde=True and quality fails)
    5. Integrated Dense Retrieval - FAISS with MMR based on parameter from start
    6. Quality Evaluation + Sparse Attachment - Check threshold, attach sparse if passed
    7. Reranking - Rerank ALL documents with cross-encoder
    8. Generation - Generate final answer from retrieved documents
    """
    
    @inject
    def __init__(self, nodes: OrchestratorNodes):
        self.nodes = nodes
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes
        workflow.add_node("coreference_resolution", self.nodes.coreference_resolution) #type: ignore
        workflow.add_node("query_analysis", self.nodes.query_analysis)                 #type: ignore
        workflow.add_node("query_rewriting", self.nodes.query_rewriting)               #type: ignore
        workflow.add_node("hyde_generation", self.nodes.hyde_generation)               #type: ignore
        workflow.add_node("dense_retrieval", self.nodes.dense_retrieval)               #type: ignore
        workflow.add_node("quality_evaluation", self.nodes.quality_evaluation)         #type: ignore
        workflow.add_node("rerank", self.nodes.rerank)                                 #type: ignore
        workflow.add_node("generation", self.nodes.generation)                         #type: ignore
        
        # Set entry point
        workflow.set_entry_point("coreference_resolution")
        
        # Add edges
        workflow.add_edge("coreference_resolution", "query_analysis")
        
        workflow.add_conditional_edges(
            "query_analysis",
            self._should_rewrite,
            {
                True: "query_rewriting",
                False: "dense_retrieval"
            }
        )
        
        workflow.add_edge("query_rewriting", "dense_retrieval")
        workflow.add_edge("dense_retrieval", "quality_evaluation")
        
        # Conditional edge from quality_evaluation
        workflow.add_conditional_edges(
            "quality_evaluation",
            self._route_after_quality,
            {
                "rerank": "rerank",
                "hyde": "hyde_generation",
                "generation": "generation"
            }
        )
        
        # HyDE path
        workflow.add_edge("hyde_generation", "dense_retrieval")
        workflow.add_edge("dense_retrieval", "quality_evaluation")
        
        # Rerank → Generation
        workflow.add_edge("rerank", "generation")
        workflow.add_edge("generation", END)
        
        return workflow.compile()  #type: ignore
    
    def _should_rewrite(self, state: OrchestratorState) -> bool:
        query_state = state.get("query_state", "well_formed")
        if query_state in ["short", "long"]:
            return True
        return state.get("needs_rewriting", False)
    
    def _route_after_quality(self, state: OrchestratorState) -> str:
        """
        Route after quality evaluation.
        
        Returns:
            - "rerank": Quality passed → Go to rerank
            - "hyde": Quality failed and HyDE enabled and not used yet
            - "generation": Quality failed and HyDE disabled or already used
        """
        quality_passed = state.get("quality_passed", False)
        use_hyde = state.get("use_hyde", False)
        hyde_used = state.get("hyde_used", False)
        
        if quality_passed:
            logger.info("   ✅ Quality PASSED → Proceeding to rerank")
            return "rerank"
        
        logger.info(f"   ❌ Quality FAILED (use_hyde={use_hyde}, hyde_already_used={hyde_used})")
        
        if use_hyde and not hyde_used:
            logger.info("   🔮 Triggering HyDE fallback")
            return "hyde"
        
        logger.info("   ⏭️ Proceeding to generation with best available")
        return "generation"
    
    async def run(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        file_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the RAG pipeline (non-streaming version).
        
        Args:
            query: User query
            conversation_history: List of conversation turns
            file_ids: Optional list of file IDs to filter by
            **kwargs: Additional parameters including:
                - similarity_threshold: Score threshold (default: 0.5)
                - min_docs_required: Minimum docs above threshold (default: 3)
                - top_k: Number of final results (default: 10)
                - use_hyde: Whether to use HyDE fallback (default: False)
                - sparse_ratio: Ratio of sparse results (default: 0.2)
                - retrieval_total_k: Total k for results (default: 20)
                - use_reranker: Whether to use reranker (default: True)
                - use_mmr: Whether to use MMR (default: False)
                - mmr_fetch_k: MMR fetch k (default: 200)
                - mmr_lambda_mult: MMR lambda (default: 0.5)
        """
        initial_state = self._prepare_initial_state(query, conversation_history, file_ids, **kwargs)
        
        # Log configuration
        self._log_config(initial_state)
        
        try:
            final_state = await self.graph.ainvoke(initial_state) #type: ignore
            return self._prepare_response(final_state)
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return self._handle_error(initial_state, e)
    
    async def run_stream(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        file_ids: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the RAG pipeline with TRUE CHARACTER-BY-CHARACTER STREAMING.
        
        This method:
        1. Runs the full pipeline (coreference → retrieval → rerank)
        2. Then uses generation_stream() to stream the answer character-by-character
        
        Yields DICT events:
        - {"type": "status", "status": str, "message": str, "metadata": Dict}
        - {"type": "chunk", "chunk": str, "chunk_index": int, "is_last": bool, "metadata": Dict}
        - {"type": "sources", "sources": List[Dict], "sources_text": str, "metadata": Dict}
        - {"type": "complete", "full_answer": str, "metadata": Dict}
        - {"type": "error", "error": str}
        """
        initial_state = self._prepare_initial_state(query, conversation_history, file_ids, **kwargs)
        
        # Log configuration
        self._log_config(initial_state)
        
        try:
            # Yield status update
            yield {
                "type": "status",
                "status": "processing",
                "message": "Starting RAG pipeline...",
                "metadata": {
                    "query": query,
                    "history_turns": len(conversation_history or []),
                    "use_hyde": initial_state.get("use_hyde", False),
                    "use_mmr": initial_state.get("use_mmr", False)
                }
            }
            
            # STEP 1: Run the pipeline up to generation (non-streaming)
            # This includes coreference, retrieval, quality check, rerank
            final_state = await self.graph.ainvoke(initial_state)  #type: ignore
            
            # Check if we have documents to generate from
            documents = final_state.get("reranked_chunks", [])
            if not documents:
                documents = final_state.get("chunks_for_reranker", [])
            if not documents:
                documents = final_state.get("dense_retrieved_chunks", [])
            
            if not documents:
                yield {
                    "type": "error",
                    "error": "No documents available for generation"
                }
                return
            
            # STEP 2: Use the streaming generation node for TRUE character-by-character streaming
            streaming_state = {
                "query": query,
                "history": conversation_history or [],
                "reranked_chunks": documents,
                "chunks_for_reranker": final_state.get("chunks_for_reranker", []),
                "dense_retrieved_chunks": final_state.get("dense_retrieved_chunks", []),
                "sparse_attached": final_state.get("sparse_results_attached", False),
                "sparse_count_attached": final_state.get("sparse_count_attached", 0),
                "mmr_used": final_state.get("mmr_used", False),
                "hyde_used": final_state.get("hyde_used", False)
            }
            
            # Stream the generation character-by-character
            full_answer = ""
            sources = []
            citations = []
            sources_used = []
            
            async for stream_event in self.nodes.generation_stream(streaming_state):
                event_type = stream_event.get("type")
                
                if event_type == "chunk":
                    chunk_text = stream_event.get("chunk", "")
                    full_answer += chunk_text
                    yield stream_event
                    
                elif event_type == "sources":
                    sources = stream_event.get("sources", [])
                    citations = stream_event.get("citations", [])
                    sources_used = stream_event.get("sources_used", [])
                    yield stream_event
                    
                elif event_type == "complete":
                    # Add retrieval method info
                    metadata = stream_event.get("metadata", {})
                    metadata["retrieval_method"] = final_state.get("retrieval_method", "dense")
                    metadata["mmr_used"] = final_state.get("mmr_used", False)
                    metadata["hyde_used"] = final_state.get("hyde_used", False)
                    metadata["quality_passed"] = final_state.get("quality_passed", False)
                    
                    stream_event["metadata"] = metadata
                    yield stream_event
                    
                elif event_type == "error":
                    yield stream_event
                    return
            
        except Exception as e:
            logger.error(f"❌ Pipeline streaming failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield {
                "type": "error",
                "error": str(e)
            }
    
    def _prepare_initial_state(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict]], 
        file_ids: Optional[List[str]], 
        **kwargs
    ) -> OrchestratorState:
        """Prepare the initial state for the pipeline."""
        return {  #type: ignore
            "query": query,
            "history": conversation_history or [],
            "file_ids": file_ids,
            "pipeline_steps": [],
            
            "similarity_threshold": kwargs.get("similarity_threshold", 0.5),
            "min_docs_required": kwargs.get("min_docs_required", 3),
            "top_k": kwargs.get("top_k", 10),
            
            "use_hyde": kwargs.get("use_hyde", False),
            
            "sparse_ratio": kwargs.get("sparse_ratio", 0.2),
            "retrieval_total_k": kwargs.get("retrieval_total_k", 20),
            "use_reranker": kwargs.get("use_reranker", True),
            
            "use_mmr": kwargs.get("use_mmr", False),
            "mmr_fetch_k": kwargs.get("mmr_fetch_k", 200),
            "mmr_lambda_mult": kwargs.get("mmr_lambda_mult", 0.5),
            
            **kwargs  #type: ignore
        }
    
    def _log_config(self, state: OrchestratorState) -> None:
        """Log pipeline configuration."""
        logger.info("=" * 80)
        logger.info("🚀 STARTING CONVERSATIONAL RAG PIPELINE")
        logger.info(f"   Query: {state.get('query')}")
        logger.info(f"   History turns: {len(state.get('history', []))}") #type: ignore
        logger.info(f"   Threshold: {state.get('similarity_threshold')}")
        logger.info(f"   Top K: {state.get('top_k')}")
        logger.info(f"   HyDE: {state.get('use_hyde')}")
        logger.info(f"   Sparse Ratio: {state.get('sparse_ratio')}")
        logger.info(f"   Total K: {state.get('retrieval_total_k')}")
        logger.info(f"   MMR: {'Enabled' if state.get('use_mmr') else 'Disabled'}")
        if state.get('use_mmr'):
            logger.info(f"   MMR Fetch K: {state.get('mmr_fetch_k')}")
            logger.info(f"   MMR Lambda: {state.get('mmr_lambda_mult')}")
        logger.info("=" * 80)
    
    def _prepare_response(self, state: OrchestratorState) -> Dict[str, Any]:
        """Prepare final response."""
        threshold_used = state.get("threshold_used")
        if threshold_used is None:
            threshold_used = state.get("similarity_threshold", 0.5)
        
        return {
            "original_query": state.get("query"),
            "resolved_query": state.get("resolved_query"),
            "coreference_used": state.get("coreference_used", False),
            "query_state": state.get("query_state"),
            "query_length": state.get("query_length"),
            "needs_rewriting": state.get("needs_rewriting", False),
            "rewritten_queries": state.get("rewritten_queries"),
            
            "use_mmr": state.get("use_mmr", False),
            "mmr_lambda_mult": state.get("mmr_lambda_mult", 0.5),
            "mmr_fetch_k": state.get("mmr_fetch_k", 200),
            "mmr_used": state.get("mmr_used", False),
            
            "hyde_used": state.get("hyde_used", False),
            "hyde_query": state.get("hyde_query", ""),
            
            "dense_retrieved_chunks": state.get("dense_retrieved_chunks", []),
            "dense_scores": state.get("dense_scores", []),
            
            "quality_passed": state.get("quality_passed", False),
            "quality_reason": state.get("quality_reason", ""),
            "docs_above_threshold": state.get("docs_above_threshold", 0),
            "docs_below_threshold": state.get("docs_below_threshold", 0),
            "threshold_used": threshold_used,
            
            "sparse_attached": state.get("sparse_results_attached", False),
            "sparse_count": state.get("sparse_count_attached", 0),
            
            "reranked_chunks": state.get("reranked_chunks", []),
            "rerank_scores": state.get("rerank_scores", []),
            "rerank_used": state.get("rerank_used", False),
            
            "generated_answer": state.get("generated_answer", ""),
            "generation_used": state.get("generation_used", False),
            "sources": state.get("sources", []),
            "citations": state.get("citations", []),
            "retrieval_method": state.get("retrieval_method", "dense"),
            
            "pipeline_steps": state.get("pipeline_steps", [])
        }
    
    def _handle_error(self, state: OrchestratorState, error: Exception) -> Dict[str, Any]:
        """Handle pipeline errors without fallback messages."""
        return {
            "original_query": state.get("query"),
            "resolved_query": state.get("resolved_query", state.get("query")),
            "error": str(error),
            "generated_answer": "",
            "generation_used": False,
            "sources": [],
            "citations": [],
            "pipeline_steps": state.get("pipeline_steps", [])
        }