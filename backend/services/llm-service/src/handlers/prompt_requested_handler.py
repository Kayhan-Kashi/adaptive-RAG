import time
import logging
from typing import Optional, Any, AsyncGenerator
from injector import inject
from common.events import PromptAnswerRequestedEvent #type: ignore
from common.events.prompt_answer_chunk_streamed import PromptAnswerChunkStreamed #type: ignore
from common.events.prompt_answer_completed import PromptAnswerCompletedEvent #type: ignore
from src.services.llm_service_stream import LLMService

logger = logging.getLogger(__name__)


class PromptAnswerRequestedHandler:
    """Handler for PromptAnswerRequestedEvent using LLM Service with streaming"""
    
    @inject
    def __init__(self, llm_service: LLMService):
        """Initialize handler with injected LLM service"""
        self.llm_service = llm_service
        logger.info("✅ PromptAnswerRequestedHandler initialized")
    
    async def handle(self, event: PromptAnswerRequestedEvent, db: Optional[Any] = None) -> AsyncGenerator:
        """Handle consumed prompt request with streaming chunks"""
        try:
            logger.info(f"📥 [LLM] Processing prompt request with streaming")
            logger.info(f"   Event ID: {event.event_id[:8]}...")
            logger.info(f"   Prompt: {event.prompt[:150]}...")
            logger.info(f"   Conversation ID: {event.conversation_id}")
            logger.info(f"   Dialogue ID: {event.dialogue_id}")
            
            file_ids = getattr(event, 'file_ids', None)
            history = getattr(event, 'history', None)
            
            if file_ids:
                logger.info(f"   📁 File IDs: {file_ids}")
            if history:
                logger.info(f"   📜 History: {len(history)} messages")
            
            start_time = time.time()
            full_answer = ""
            total_chunks = 0
            
            # STREAM CHUNKS
            async for chunk_index, chunk_text, is_last in self.llm_service.generate_stream(
                prompt=event.prompt,
                conversation_id=event.conversation_id,
                dialogue_id=event.dialogue_id,
                file_ids=file_ids,
                history=history
            ):
                full_answer += chunk_text
                total_chunks = chunk_index + 1
                
                # Create and yield chunk event
                chunk_event = PromptAnswerChunkStreamed(
                    conversation_id=event.conversation_id,
                    dialogue_id=event.dialogue_id,
                    prompt=event.prompt,
                    chunk=chunk_text,
                    chunk_index=chunk_index,
                    is_last=is_last
                )
                
                logger.debug(f"   📤 Chunk {chunk_index}: {len(chunk_text)} chars")
                yield chunk_event
            
            elapsed = time.time() - start_time
            
            logger.info(f"✅ [LLM] Answer streaming completed in {elapsed:.2f}s")
            logger.info(f"   Total chunks: {total_chunks}")
            logger.info(f"   Answer length: {len(full_answer)} characters")
            
            # Return final completion event
            print("==========PromptAnswerCompletedEvent==============", flush=True)
            print(PromptAnswerCompletedEvent(
                conversation_id=event.conversation_id,
                dialogue_id=event.dialogue_id,
                prompt=event.prompt,
                full_answer=full_answer
            ).model_dump(), flush=True)
            print("========================", flush=True)
            
            yield PromptAnswerCompletedEvent(
                conversation_id=event.conversation_id,
                dialogue_id=event.dialogue_id,
                prompt=event.prompt,
                full_answer=full_answer
            )
            
        except Exception as e:
            logger.error(f"❌ [LLM] Error: {e}")
            raise