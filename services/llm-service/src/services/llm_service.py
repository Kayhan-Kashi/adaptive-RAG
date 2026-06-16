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
        
        # RAG Prompt
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the user's question based ONLY on the provided context.
        If the answer is not in the context, say that you don't know.
        
        <context>
        {context}
        </context>

        Question: {input}
        
        Answer: 
        """)
        
        logger.info("✅ LLMService initialized")

    async def generate(self, prompt: str, file_ids: Optional[List[str]] = None) -> str:
        """
        Generate answer using RAG.
        
        Args:
            prompt: User question
            file_ids: Optional list of file IDs to filter by
        
        Returns:
            Generated answer
        """
        try:
            logger.info(f"📝 User Question: {prompt}")
            if file_ids:
                logger.info(f"   📁 Filtering to file_ids: {file_ids}")
            
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
            
            # 3. FORMAT PROMPT
            formatted_messages = self.prompt.format_messages(
                context=context_text,
                input=prompt
            )
            
            # 4. LOG PROMPT (for debugging)
            logger.info("=" * 100)
            logger.info("🤖🤖🤖 FINAL PROMPT SENT TO OLLAMA 🤖🤖🤖")
            logger.info("=" * 100)
            logger.info(f"📊 STATISTICS:")
            logger.info(f"   - Chunks in context: {len(retrieved_chunks)}")
            logger.info(f"   - Context length: {len(context_text)} characters")
            logger.info(f"   - Query length: {len(prompt)} characters")
            if file_ids:
                logger.info(f"   - Filtering to: {file_ids}")
            logger.info("-" * 100)
            logger.info("📝 FULL PROMPT CONTENT:")
            logger.info("-" * 100)
            logger.info(formatted_messages[0].content[:2000])  # Limit to avoid log spam
            if len(formatted_messages[0].content) > 2000:
                logger.info(f"... (truncated, total {len(formatted_messages[0].content)} characters)")
            logger.info("=" * 100)
            
            # 5. GENERATE
            response = await self.llm.ainvoke(formatted_messages)
            answer = response.content.strip()
            
            logger.info(f"💬 Response: {answer[:300]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error generating response: {str(e)}"
    
    async def generate_with_sources(self, prompt: str, file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate answer and return with source documents.
        
        Args:
            prompt: User question
            file_ids: Optional list of file IDs to filter by
        
        Returns:
            Dictionary with answer and sources
        """
        try:
            logger.info(f"📝 User Question: {prompt}")
            if file_ids:
                logger.info(f"   📁 Filtering to file_ids: {file_ids}")
            
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
            
            # HIGHLY VISIBLE PROMPT LOGGING WITH SOURCES
            logger.info("=" * 100)
            logger.info("🤖🤖🤖 FINAL PROMPT SENT TO OLLAMA (WITH SOURCES) 🤖🤖🤖")
            logger.info("=" * 100)
            logger.info(f"📊 STATISTICS:")
            logger.info(f"   - Total chunks: {len(retrieved_chunks)}")
            logger.info(f"   - Context length: {len(context_text)} chars")
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
                input=prompt
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