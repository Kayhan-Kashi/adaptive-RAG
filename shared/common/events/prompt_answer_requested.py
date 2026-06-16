# common/events/prompt_answer_requested.py
from typing import List, Optional, Dict, Any

from common.events.base_event import BaseEvent


class PromptAnswerRequestedEvent(BaseEvent):
    conversation_id: str
    dialogue_id: str
    prompt: str
    user_id: str
    file_ids: Optional[List[str]] = None
    history: Optional[List[Dict[str, Any]]] = None  # ← ADDED: Conversation history
    
    @classmethod
    def event_type(cls) -> str:
        return "prompt_answer_requested"
    
    @classmethod
    def topic(cls) -> str:
        return "prompt-answer-requests"