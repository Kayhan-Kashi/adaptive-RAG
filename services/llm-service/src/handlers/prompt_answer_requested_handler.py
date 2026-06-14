# llm-service/src/handlers/prompt_answer_requested_handler.py
import time
import logging
from typing import Optional, Any
from injector import inject
from common.events import PromptAnswerRequestedEvent, PromptAnswerCompletedEvent #type: ignore
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class PromptAnswerRequestedHandler:
    """Handler for PromptAnswerRequestedEvent using LLM Service"""
    
    @inject
    def __init__(self, llm_service: LLMService):
        """Initialize handler with injected LLM service"""
        self.llm_service = llm_service
        logger.info("✅ PromptAnswerRequestedHandler initialized")
    
    async def handle(self, event: PromptAnswerRequestedEvent, db: Optional[Any] = None):
        """Handle consumed prompt request"""
        try:
            logger.info(f"📥 [LLM] Processing prompt request")
            logger.info(f"   Event ID: {event.event_id[:8]}...")
            logger.info(f"   Prompt: {event.prompt[:150]}...")
            
            # Generate answer using LLM service
            start_time = time.time()
            answer = await self.llm_service.generate(event.prompt)
            elapsed = time.time() - start_time
            
            logger.info(f"✅ [LLM] Answer generated in {elapsed:.2f}s")
            logger.info(f"   Answer length: {len(answer)} characters")
            
            # Return completion event
            return PromptAnswerCompletedEvent(
                conversation_id=event.conversation_id,
                dialogue_id=event.dialogue_id,
                prompt=event.prompt,
                full_answer=answer
            )
            
        except Exception as e:
            logger.error(f"❌ [LLM] Error: {e}")
            raise