from typing import List, Optional, Dict, Any

from .base_event import BaseEvent


class PromptAnswerRequestedEvent(BaseEvent):
    conversation_id: str
    dialogue_id: str
    prompt: str
    user_id: str
    file_ids: Optional[List[str]] = None
    history: Optional[List[Dict[str, Any]]] = None
    
    # Retrieval settings
    retrieval_k: int = 20
    similarity_threshold: float = 0.5
    min_docs_required: int = 3
    top_k: int = 5
    use_hyde: bool = True
    sparse_ratio: float = 0.2
    retrieval_total_k: int = 20
    use_reranker: bool = True
    
    # MMR settings
    use_mmr: bool = True
    mmr_fetch_k: int = 200
    mmr_lambda_mult: float = 0.8
    
    # ============================================================
    # NEW: Evaluation settings (controlled by UI)
    # ============================================================
    enable_evaluation: bool = True  # Whether to run LLM-as-Judge evaluation
    evaluation_threshold: float = 0.7  # Threshold for evaluation pass
    
    @classmethod
    def event_type(cls) -> str:
        return "prompt_answer_requested"
    
    @classmethod
    def topic(cls) -> str:
        return "prompt-answer-requests"