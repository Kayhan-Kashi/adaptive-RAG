from .base_event import BaseEvent


class PromptAnswerChunkStreamed(BaseEvent):
    conversation_id: str
    dialogue_id: str
    prompt: str
    chunk: str
    chunk_index: int
    is_last: bool = False
    
    @classmethod
    def event_type(cls) -> str:
        return "prompt_answer_chunk_streamed"
    
    @classmethod
    def topic(cls) -> str:
        return "prompt-answer-chunk-streamed"