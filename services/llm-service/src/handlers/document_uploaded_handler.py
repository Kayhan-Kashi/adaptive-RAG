# llm-service/src/handlers/document_uploaded_handler.py
import time
import logging
from typing import Optional, Any
from injector import inject
from common.events import DocumentUploadedEvent #type: ignore
from common.events.document_embedding_done import DocumentEmbeddingDoneEvent #type: ignore
from src.services.rag_service import RagService

logger = logging.getLogger(__name__)


class DocumentUploadedHandler:
    """Handler for DocumentUploadedEvent using RagService"""
    
    @inject
    def __init__(self, rag_service: RagService):
        """Initialize handler with injected RagService"""
        self.rag_service = rag_service
        logger.info("✅ DocumentUploadedHandler initialized")
    
    async def handle(self, event: DocumentUploadedEvent, db: Optional[Any] = None):
        """Handle consumed document upload event and return completion event"""
        try:
            logger.info(f"📥 [LLM] Processing document upload event")
            logger.info(f"   Document ID: {event.document_id[:8]}...")
            logger.info(f"   Filename: {event.filename}")
            logger.info(f"   File type: {event.filetype}")
            
            # Get the file path
            file_path = f"/app/uploads/{event.document_id}{event.filetype}"
            
            # Process document (prepare, chunk, and embed)
            start_time = time.time()
            chunks = self.rag_service.prepare_document(
                file_path=file_path,
                document_id=event.document_id,
                filename=event.filename,
                filetype=event.filetype,
                chunk_size=1000,
                chunk_overlap=200
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ [LLM] Document processed in {elapsed:.2f}s")
            logger.info(f"   Created {len(chunks)} chunks")
            
            # Return completion event (worker will publish it)
            return DocumentEmbeddingDoneEvent(
                document_id=event.document_id,
                filetype=event.filetype,
                filename=event.filename
            )
            
        except FileNotFoundError as e:
            logger.error(f"❌ [LLM] File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ [LLM] Error processing document: {e}")
            raise