# llm-service/src/services/llm_service.py
import os
import logging
from injector import inject
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
from src.services.rag_service import RagService

logger = logging.getLogger(__name__)


class LLMService:
    """Service for RAG-enhanced LLM operations"""
    
    @inject
    def __init__(self, rag_service: RagService):
        self.rag_service = rag_service
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
        )
        
        # RAG Prompt with history support
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the user's question based ONLY on the provided context and previous conversation_history.
        If the answer is not in the context or in the conversation history, say that you don't know.
        
        <conversation_history>
        {history}
        </conversation_history>
        
        
        <context>
        {context}
        </context>

        Question: {input}
        
        Answer: 
        """)
        
        logger.info("✅ LLMService initialized")

    async def generate(
        self, 
        prompt: str, 
        file_ids: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate answer using RAG with optional conversation history.
        
        Args:
            prompt: User question
            file_ids: Optional list of file IDs to filter by
            history: Optional conversation history as list of {role, content} dicts
        
        Returns:
            Generated answer
        """
        try:
            logger.info(f"📝 User Question: {prompt}")
            if file_ids:
                logger.info(f"   📁 Filtering to file_ids: {file_ids}")
            if history:
                logger.info(f"   📜 History: {len(history)} messages")
            
            # 1. RETRIEVE: Get relevant chunks using RagService
            retrieved_chunks = self.rag_service.retrieve(
                query=prompt,
                k=10,
                file_ids=file_ids
            )
            
            if not retrieved_chunks:
                if file_ids:
                    return f"I don't have any relevant information in the selected documents to answer: '{prompt}'."
                return "I don't have any relevant information to answer this question."
            
            # Log retrieved chunks
            logger.info(f"📚 Retrieved {len(retrieved_chunks)} chunks:")
            for i, chunk in enumerate(retrieved_chunks[:5], 1):
                chunk_preview = chunk.page_content[:150].replace('\n', ' ')
                score = chunk.metadata.get('reranker_score', 'N/A')
                filename = chunk.metadata.get('filename', 'unknown')
                logger.info(f"   {i}. Score={score} - {chunk_preview}... (from: {filename})")
            
            # 2. BUILD CONTEXT
            context_text = "\n\n---\n\n".join([
                f"[Source: {chunk.metadata.get('filename', 'unknown')}]\n{chunk.page_content}"
                for chunk in retrieved_chunks
            ])
            
            # 3. BUILD HISTORY STRING
            history_text = ""
            if history:
                history_lines = []
                for msg in history:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'user':
                        history_lines.append(f"User: {content}")
                    else:
                        history_lines.append(f"Assistant: {content}")
                history_text = "Previous conversation:\n" + "\n".join(history_lines) + "\n"
                logger.info(f"📜 Added {len(history)} history messages to prompt")
            
            # 4. FORMAT PROMPT with history
            formatted_messages = self.prompt.format_messages(
                context=context_text,
                input=prompt,
                history=history_text
            )
            
            # 5. LOG PROMPT (for debugging)
            logger.info("=" * 100)
            logger.info("🤖🤖🤖 FINAL PROMPT SENT TO OLLAMA 🤖🤖🤖")
            logger.info("=" * 100)
            logger.info(f"📊 STATISTICS:")
            logger.info(f"   - Chunks in context: {len(retrieved_chunks)}")
            logger.info(f"   - Context length: {len(context_text)} characters")
            logger.info(f"   - History messages: {len(history) if history else 0}")
            logger.info(f"   - Query length: {len(prompt)} characters")
            if file_ids:
                logger.info(f"   - Filtering to: {file_ids}")
            logger.info("-" * 100)
            logger.info("📝 FULL PROMPT CONTENT:")
            logger.info("-" * 100)
            logger.info(formatted_messages[0].content[:2000])
            if len(formatted_messages[0].content) > 2000:
                logger.info(f"... (truncated, total {len(formatted_messages[0].content)} characters)")
            logger.info("=" * 100)
            
            # 6. GENERATE
            response = await self.llm.ainvoke(formatted_messages)
            answer = response.content.strip()
            
            logger.info(f"💬 Response: {answer[:300]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error generating response: {str(e)}"
    
    async def generate_with_sources(
        self, 
        prompt: str, 
        file_ids: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate answer and return with source documents.
        
        Args:
            prompt: User question
            file_ids: Optional list of file IDs to filter by
            history: Optional conversation history
        
        Returns:
            Dictionary with answer and sources
        """
        try:
            logger.info(f"📝 User Question: {prompt}")
            if file_ids:
                logger.info(f"   📁 Filtering to file_ids: {file_ids}")
            if history:
                logger.info(f"   📜 History: {len(history)} messages")
            
            retrieved_chunks = self.rag_service.retrieve(
                query=prompt,
                k=10,
                file_ids=file_ids
            )
            
            if not retrieved_chunks:
                return {
                    "answer": "I don't have any relevant information to answer this question.",
                    "sources": [],
                    "total_chunks_retrieved": 0
                }
            
            context_text = "\n\n---\n\n".join([
                f"[Source: {chunk.metadata.get('filename', 'unknown')}]\n{chunk.page_content}"
                for chunk in retrieved_chunks
            ])
            
            # Build history string
            history_text = ""
            if history:
                history_lines = []
                for msg in history:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'user':
                        history_lines.append(f"User: {content}")
                    else:
                        history_lines.append(f"Assistant: {content}")
                history_text = "Previous conversation:\n" + "\n".join(history_lines) + "\n"
            
            # HIGHLY VISIBLE PROMPT LOGGING WITH SOURCES
            logger.info("=" * 100)
            logger.info("🤖🤖🤖 FINAL PROMPT SENT TO OLLAMA (WITH SOURCES) 🤖🤖🤖")
            logger.info("=" * 100)
            logger.info(f"📊 STATISTICS:")
            logger.info(f"   - Total chunks: {len(retrieved_chunks)}")
            logger.info(f"   - Context length: {len(context_text)} chars")
            logger.info(f"   - History messages: {len(history) if history else 0}")
            if file_ids:
                logger.info(f"   - Filtering to: {file_ids}")
            logger.info("-" * 100)
            logger.info("📝 FULL PROMPT CONTENT:")
            logger.info("-" * 100)
            logger.info(context_text[:2000])
            if len(context_text) > 2000:
                logger.info(f"... (truncated, total {len(context_text)} characters)")
            logger.info("=" * 100)
            
            formatted_messages = self.prompt.format_messages(
                context=context_text,
                input=prompt,
                history=history_text
            )
            
            response = await self.llm.ainvoke(formatted_messages)
            answer = response.content.strip()
            
            sources = [
                {
                    "rank": i,
                    "content_preview": chunk.page_content[:200],
                    "full_content": chunk.page_content,
                    "document_id": chunk.metadata.get('document_id'),
                    "filename": chunk.metadata.get('filename'),
                    "reranker_score": chunk.metadata.get('reranker_score')
                }
                for i, chunk in enumerate(retrieved_chunks, 1)
            ]
            
            return {
                "answer": answer,
                "sources": sources,
                "total_chunks_retrieved": len(retrieved_chunks)
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return self.rag_service.get_stats()