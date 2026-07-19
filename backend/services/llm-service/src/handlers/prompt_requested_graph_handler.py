import time
import logging
from typing import Optional, Any, AsyncGenerator
from injector import inject
from common.events import PromptAnswerRequestedEvent #type: ignore
from common.events.prompt_answer_chunk_streamed import PromptAnswerChunkStreamed #type: ignore
from common.events.prompt_answer_completed import PromptAnswerCompletedEvent #type: ignore
from src.graph.orchestrator_graph import OrchestratorGraph

logger = logging.getLogger(__name__)


class PromptAnswerRequestedHandler:
    """Handler for PromptAnswerRequestedEvent using OrchestratorGraph with streaming"""
    
    @inject
    def __init__(self, orchestrator_graph: OrchestratorGraph):
        """Initialize handler with injected OrchestratorGraph"""
        self.orchestrator_graph = orchestrator_graph
        logger.info("✅ PromptAnswerRequestedHandler initialized")
    
    async def handle(self, event: PromptAnswerRequestedEvent, db: Optional[Any] = None) -> AsyncGenerator:
        """Handle consumed prompt request with streaming chunks using the graph pipeline"""
        try:
            logger.info("=" * 80)
            logger.info("📥 [RAG] Processing prompt request with streaming")
            logger.info("=" * 80)
            logger.info(f"   Event ID: {event.event_id[:8]}...")
            logger.info(f"   Prompt: {event.prompt[:150]}...")
            logger.info(f"   Conversation ID: {event.conversation_id}")
            logger.info(f"   Dialogue ID: {event.dialogue_id}")
            
            # Extract optional fields
            file_ids = getattr(event, 'file_ids', None)
            history = getattr(event, 'history', None)
            
            # Extract configuration from event or use defaults
            retrieval_k = getattr(event, 'retrieval_k', 20)
            similarity_threshold = getattr(event, 'similarity_threshold', 0.5)
            min_docs_required = getattr(event, 'min_docs_required', 3)
            top_k = getattr(event, 'top_k', 5)
            use_hyde = getattr(event, 'use_hyde', True)
            sparse_ratio = getattr(event, 'sparse_ratio', 0.2)
            retrieval_total_k = getattr(event, 'retrieval_total_k', 20)
            use_reranker = getattr(event, 'use_reranker', True)
            use_mmr = getattr(event, 'use_mmr', True)
            mmr_fetch_k = getattr(event, 'mmr_fetch_k', 200)
            mmr_lambda_mult = getattr(event, 'mmr_lambda_mult', 0.8)
            
            if file_ids:
                logger.info(f"   📁 File IDs: {file_ids}")
            if history:
                logger.info(f"   📜 History: {len(history)} messages")
            
            logger.info(f"   ⚙️ Config: threshold={similarity_threshold}, top_k={top_k}, hyde={use_hyde}, mmr={use_mmr}")
            logger.info("=" * 80)
            
            start_time = time.time()
            full_answer = ""
            total_chunks = 0
            sources_list = []
            citations_list = []
            chunk_times = []
            
            # STREAM CHUNKS from the graph
            logger.info("🔄 Starting stream from orchestrator graph...")
            
            async for stream_event in self.orchestrator_graph.run_stream(
                query=event.prompt,
                conversation_history=history,
                file_ids=file_ids,
                retrieval_k=retrieval_k,
                similarity_threshold=similarity_threshold,
                min_docs_required=min_docs_required,
                top_k=top_k,
                use_hyde=use_hyde,
                sparse_ratio=sparse_ratio,
                retrieval_total_k=retrieval_total_k,
                use_reranker=use_reranker,
                use_mmr=use_mmr,
                mmr_fetch_k=mmr_fetch_k,
                mmr_lambda_mult=mmr_lambda_mult
            ):
                event_type = stream_event.get("type")
                
                if event_type == "status":
                    # Log status updates
                    status = stream_event.get("status", "processing")
                    message = stream_event.get("message", "")
                    metadata = stream_event.get("metadata", {})
                    logger.info(f"   📊 STATUS: {status} - {message}")
                    if metadata:
                        logger.debug(f"      Metadata: {metadata}")
                    
                elif event_type == "chunk":
                    # Yield chunk event
                    chunk_text = stream_event.get("chunk", "")
                    chunk_index = stream_event.get("chunk_index", 0)
                    is_last = stream_event.get("is_last", False)
                    metadata = stream_event.get("metadata", {})
                    
                    chunk_start = time.time()
                    full_answer += chunk_text
                    total_chunks = chunk_index + 1
                    
                    # Log the chunk
                    chunk_preview = chunk_text.replace('\n', ' ')[:50] + "..." if len(chunk_text) > 50 else chunk_text.replace('\n', ' ')
                    logger.info(f"   📤 CHUNK {chunk_index}: {len(chunk_text)} chars")
                    logger.info(f"      Content: \"{chunk_preview}\"")
                    if metadata:
                        logger.debug(f"      Metadata: {metadata}")
                    
                    # Create and yield the concrete chunk event
                    chunk_event = PromptAnswerChunkStreamed(
                        conversation_id=event.conversation_id,
                        dialogue_id=event.dialogue_id,
                        prompt=event.prompt,
                        chunk=chunk_text,
                        chunk_index=chunk_index,
                        is_last=is_last
                    )
                    
                    chunk_times.append(time.time() - chunk_start)
                    yield chunk_event
                    
                elif event_type == "sources":
                    # Store sources info
                    sources_list = stream_event.get("sources", [])
                    citations_list = stream_event.get("citations", [])
                    sources_text = stream_event.get("sources_text", "")
                    metadata = stream_event.get("metadata", {})
                    
                    logger.info(f"   📚 SOURCES: {len(sources_list)} sources, {len(citations_list)} citations")
                    for i, source in enumerate(sources_list[:3]):  # Show first 3 sources
                        logger.info(f"      Source {i+1}: {source.get('filename')} (Page {source.get('page')})")
                    if len(sources_list) > 3:
                        logger.info(f"      ... and {len(sources_list) - 3} more sources")
                    
                    # Optionally yield sources as a special chunk
                    if sources_text:
                        # Send sources as a final chunk if not already sent
                        sources_chunk_event = PromptAnswerChunkStreamed(
                            conversation_id=event.conversation_id,
                            dialogue_id=event.dialogue_id,
                            prompt=event.prompt,
                            chunk="\n\n" + sources_text,
                            chunk_index=total_chunks,
                            is_last=False
                        )
                        logger.info(f"   📤 SOURCES CHUNK: {len(sources_text)} chars")
                        yield sources_chunk_event
                        total_chunks += 1
                    
                elif event_type == "complete":
                    # Log completion metadata
                    metadata = stream_event.get("metadata", {})
                    logger.info("=" * 80)
                    logger.info("   ✅ PIPELINE COMPLETE")
                    logger.info("=" * 80)
                    logger.info(f"      Answer length: {metadata.get('answer_length', 0)} chars")
                    logger.info(f"      Total chunks: {total_chunks}")
                    logger.info(f"      Sources: {metadata.get('source_count', 0)}")
                    logger.info(f"      Citations: {metadata.get('citation_count', 0)}")
                    logger.info(f"      Retrieval method: {metadata.get('retrieval_method')}")
                    logger.info(f"      MMR used: {metadata.get('mmr_used', False)}")
                    logger.info(f"      HyDE used: {metadata.get('hyde_used', False)}")
                    logger.info(f"      Quality passed: {metadata.get('quality_passed', False)}")
                    
                    # Show full answer preview
                    answer_preview = full_answer[:200].replace('\n', ' ') + "..." if len(full_answer) > 200 else full_answer.replace('\n', ' ')
                    logger.info(f"      Answer preview: {answer_preview}")
                    
                elif event_type == "error":
                    logger.error(f"   ❌ PIPELINE ERROR: {stream_event.get('error')}")
                    raise Exception(stream_event.get('error', 'Unknown error'))
            
            elapsed = time.time() - start_time
            
            # Log final statistics
            logger.info("=" * 80)
            logger.info(f"✅ [RAG] Pipeline completed in {elapsed:.2f}s")
            logger.info("=" * 80)
            logger.info(f"   Total chunks: {total_chunks}")
            logger.info(f"   Answer length: {len(full_answer)} characters")
            logger.info(f"   Sources: {len(sources_list)}")
            logger.info(f"   Citations: {len(citations_list)}")
            
            if chunk_times:
                avg_chunk_time = sum(chunk_times) / len(chunk_times)
                logger.info(f"   Avg chunk time: {avg_chunk_time:.4f}s")
                logger.info(f"   Total streaming time: {sum(chunk_times):.4f}s")
            
            # Yield final completion event
            completion_event = PromptAnswerCompletedEvent(
                conversation_id=event.conversation_id,
                dialogue_id=event.dialogue_id,
                prompt=event.prompt,
                full_answer=full_answer
            )
            
            logger.info(f"📤 Yielding completion event for conversation {event.conversation_id}")
            yield completion_event
            
        except Exception as e:
            logger.error(f"❌ [RAG] Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise