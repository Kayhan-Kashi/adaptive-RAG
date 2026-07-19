from typing import List, Optional, Dict, Any, TypedDict, Literal
from langchain_core.documents import Document


class OrchestratorState(TypedDict, total=False):
    # Input fields
    query: str
    conversation_id: str
    dialogue_id: str
    file_ids: Optional[List[str]]
    history: Optional[List[Dict[str, str]]]
    
    # Configuration fields
    use_hyde: bool
    is_adaptive: bool
    use_sparse: bool
    top_k: int
    similarity_threshold: float
    char_threshold: int
    quality_ratio: float
    retrieval_k: int
    mmr_k: int
    min_docs_required: int
    
    # Integrated retrieval configuration
    sparse_ratio: float  # Ratio of sparse results (e.g., 0.2 = 20%)
    retrieval_total_k: int  # Total k for combined results
    use_reranker: bool  # Whether to use reranker
    
    # ============================================================
    # MMR (Maximum Marginal Relevance) Configuration
    # ============================================================
    use_mmr: bool  # Whether to use MMR for dense retrieval (default: False)
    mmr_fetch_k: int  # Number of candidates to fetch for MMR (default: 200)
    mmr_lambda_mult: float  # MMR lambda parameter - controls diversity (0-1, default: 0.5)
    
    # Node 1: Coreference Resolution
    resolved_query: str
    coreference_used: bool
    
    # Node 2: Query Analysis
    query_state: Literal["short", "well_formed", "long"]
    query_length: int
    needs_rewriting: bool
    
    # Node 3: Query Rewriting
    rewritten_queries: List[str]
    original_query_for_reference: str
    
    # Node 4: HyDE Generation
    hyde_query: str
    hyde_used: bool
    hyde_original: str
    hyde_document: str
    
    # Node 5: Integrated Dense Retrieval (with MMR)
    dense_retrieved_chunks: List[Document]
    dense_scores: List[float]
    dense_score_map: Dict[str, float]
    retrieval_source: Literal["dense", "hyde", "ensemble"]
    mmr_used: bool  # Whether MMR was used in dense retrieval
    mmr_used_dense: bool  # Alias for mmr_used
    
    # Node 6: Quality Evaluation + Sparse Attachment
    quality_passed: bool
    quality_reason: str
    total_chunks: int
    docs_above_threshold: int
    docs_below_threshold: int
    avg_score: float
    max_score: float
    min_score: float
    threshold_used: float
    min_docs_required_used: int
    scores_above: List[float]
    scores_below: List[float]
    passed_documents: List[Document]
    
    # Sparse attachment tracking
    sparse_results_attached: bool  # Whether sparse results were attached to dense
    sparse_count_attached: int  # Number of sparse results attached
    
    # Node 6 → Node 7: Combined chunks (dense + sparse if attached)
    chunks_for_reranker: List[Document]
    
    # Node 7: Reranking (All documents → Cross-encoder → Top K)
    reranked_chunks: List[Document]
    rerank_scores: List[float]
    rerank_used: bool
    rerank_count: int
    rerank_error: Optional[str]
    reranked_retrieval_types: List[str]  # Types in reranked results
    
    # Sparse tracking in rerank
    sparse_attached: bool  # Whether sparse was attached (passed through from quality)
    sparse_count_final: int  # Number of sparse results in reranked chunks
    
    # Node 8: Generation (Answer Generation)
    generated_answer: str
    generation_used: bool
    sources: List[Document]
    sources_used: List[str]  # List of source identifiers
    context_used: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    retrieval_method: Literal["dense", "hybrid", "hyde", "ensemble"]
    mmr_used_generation: bool  # Whether MMR was used (for generation metadata)
    sparse_attached_generation: bool  # Whether sparse was attached (for generation metadata)
    
    # Node 9: Sparse Retrieval (standalone)
    sparse_retrieved_chunks: List[Document]
    sparse_scores: List[float]
    sparse_retrieval_used: bool
    
    # Node 10: Ensemble Combination (future)
    ensemble_retrieved_chunks: List[Document]
    ensemble_scores: List[float]
    retrieval_method_ensemble: Literal["dense", "ensemble", "hyde", "hybrid"]
    
    # Final Output
    final_results: List[Document]
    final_scores: List[float]
    
    # Metadata
    pipeline_steps: List[str]
    pipeline_start_time: float
    pipeline_end_time: float
    pipeline_version: str  # For tracking pipeline versions