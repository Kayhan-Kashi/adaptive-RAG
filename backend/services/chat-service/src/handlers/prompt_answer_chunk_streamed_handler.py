# handlers/prompt_answer_chunk_streamed_handler.py
import logging
from typing import Optional, Any
from injector import inject
from common.events.prompt_answer_chunk_streamed import PromptAnswerChunkStreamed #type: ignore
from api.websocket.manager import connection_manager

logger = logging.getLogger(__name__)


class PromptAnswerChunkStreamedHandler:
    """Handler for PromptAnswerChunkStreamedEvent - forwards chunks to WebSocket"""
    
    @inject
    def __init__(self):
        """Initialize handler"""
        logger.info("✅ PromptAnswerChunkStreamedHandler initialized")
    
    async def handle(self, event: PromptAnswerChunkStreamed, db: Optional[Any] = None):
        """
        Handle streamed chunk: forward to WebSocket
        
        Args:
            event: The chunk event containing the text chunk
            db: Database session (injected by message bus) - not used
        """
        try:
            logger.debug(f"📤 [Chat] Forwarding chunk {event.chunk_index}")
            logger.debug(f"   Chunk length: {len(event.chunk)} characters")
            logger.debug(f"   Is last: {event.is_last}")
            
            # Send chunk to WebSocket using conversation routing
            success = await connection_manager.send_chunk(
                conversation_id=event.conversation_id,
                chunk=event.chunk,
                chunk_index=event.chunk_index,
                is_last=event.is_last
            )
            
            if success:
                if event.is_last:
                    logger.info(f"✅ [Chat] Final chunk sent for dialogue {event.dialogue_id[:8]}...")
                else:
                    logger.debug(f"✅ [Chat] Chunk {event.chunk_index} sent to WebSocket")
            else:
                logger.warning(f"⚠️ [Chat] WebSocket not available for conversation {event.conversation_id[:8]}...")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ [Chat] Error forwarding chunk: {e}")
            logger.exception("Stack trace:")
            raise