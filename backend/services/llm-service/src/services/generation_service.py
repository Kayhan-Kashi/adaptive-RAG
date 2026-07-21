import logging
import os
import asyncio
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
from injector import inject
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from .llm_service import LLMService

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Handles answer generation using retrieved documents.
    This is the third step in RAG: Retrieve → Rerank → Generate.
    Uses the LLM to generate answers based on retrieved context.
    """
    
    # Configurable limits
    MIN_ANSWER_LENGTH = int(os.getenv("MIN_ANSWER_LENGTH", "15"))
    MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))
    MAX_SNIPPET_CHARS = int(os.getenv("MAX_SNIPPET_CHARS", "500"))
    
    # Streaming delay configuration (in seconds)
    STREAM_CHAR_DELAY = float(os.getenv("STREAM_CHAR_DELAY", "0.02"))
    STREAM_CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", "3"))
    STREAM_SOURCE_DELAY = float(os.getenv("STREAM_SOURCE_DELAY", "0.3"))
    
    @inject
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def generate(
        self,
        query: str,
        documents: List[Document],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_context_length: int = 2000,
        include_sources: bool = True,
        max_sources: int = 5
    ) -> Dict[str, Any]:
        """
        Generate an answer based on retrieved documents.
        """
        logger.info("=" * 60)
        logger.info("🤖 [Generation] Generating answer with citations...")
        logger.info("=" * 60)
        
        logger.info(f"📝 COMPLETE QUERY TO LLM: {query}")
        
        if not query:
            return {
                "original_query": query,
                "answer": "",
                "sources": [],
                "sources_used": [],
                "citations": [],
                "generation_used": False
            }
        
        if not documents:
            return {
                "original_query": query,
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "sources_used": [],
                "citations": [],
                "generation_used": False
            }
        
        try:
            documents_to_use = documents[:max_sources]
            logger.info(f"   Using {len(documents_to_use)} documents for generation")
            
            context_text, context_snippets, source_map = self._build_context_with_sources(
                documents_to_use,
                max_context_length
            )
            
            logger.info(f"   Context length: {len(context_text)} chars")
            logger.info(f"   Context preview: {context_text[:200]}...")
            
            history_text = self._format_history(conversation_history)
            
            prompt_template = self._get_citation_prompt()
            formatted_prompt = prompt_template.format(
                context=context_text,
                query=query,
                history=history_text
            )
            
            logger.info("=" * 60)
            logger.info("📝 COMPLETE PROMPT SENT TO LLM:")
            logger.info("=" * 60)
            logger.info(f"{formatted_prompt}")
            logger.info("=" * 60)
            
            chain = prompt_template | self.llm_service.llm | StrOutputParser()
            
            answer = await chain.ainvoke({
                "context": context_text,
                "query": query,
                "history": history_text
            })
            
            answer = answer.strip()
            
            logger.info(f"   Raw answer length: {len(answer)} chars")
            
            logger.info("=" * 80)
            logger.info("📝 COMPLETE ANSWER FROM LLM:")
            logger.info("=" * 80)
            logger.info(f"\n{answer}\n")
            logger.info("=" * 80)
            
            if not answer or len(answer) < self.MIN_ANSWER_LENGTH:
                logger.warning(f"   ⚠️ LLM returned empty or too short response: {len(answer)} chars")
                return {
                    "original_query": query,
                    "answer": "",
                    "sources": [],
                    "sources_used": [],
                    "citations": [],
                    "generation_used": False
                }
            
            logger.info(f"   ✅ Answer generated")
            
            return {
                "original_query": query,
                "answer": answer,
                "sources": documents_to_use[:5],
                "sources_used": [],
                "citations": [],
                "generation_used": True
            }
            
        except Exception as e:
            logger.error(f"   ❌ Generation failed: {e}")
            return {
                "original_query": query,
                "answer": "",
                "sources": [],
                "sources_used": [],
                "citations": [],
                "generation_used": False
            }
    
    async def generate_stream(
        self,
        query: str,
        documents: List[Document],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_context_length: int = 2000,
        max_sources: int = 5,
        char_delay: Optional[float] = None,
        chunk_size: Optional[int] = None,
        source_delay: Optional[float] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate an answer with streaming.
        The LLM generates the complete answer including Sources section naturally.
        """
        char_delay = char_delay if char_delay is not None else self.STREAM_CHAR_DELAY
        chunk_size = chunk_size if chunk_size is not None else self.STREAM_CHUNK_SIZE
        source_delay = source_delay if source_delay is not None else self.STREAM_SOURCE_DELAY
        
        logger.info("=" * 60)
        logger.info("🤖 [Generation] Streaming answer...")
        logger.info("=" * 60)
        logger.info(f"   Character delay: {char_delay}s")
        logger.info(f"   Chunk size: {chunk_size} chars per chunk")
        
        logger.info(f"📝 COMPLETE QUERY TO LLM: {query}")
        
        if not query:
            yield {"type": "error", "error": "Empty query provided"}
            return
        
        if not documents:
            yield {"type": "error", "error": "No documents provided"}
            return
        
        try:
            documents_to_use = documents[:max_sources]
            logger.info(f"   Using {len(documents_to_use)} documents for generation")
            
            context_text, context_snippets, source_map = self._build_context_with_sources(
                documents_to_use,
                max_context_length
            )
            
            logger.info(f"   Context length: {len(context_text)} chars")
            logger.info(f"   Context preview: {context_text[:200]}...")
            
            history_text = self._format_history(conversation_history)
            
            prompt_template = self._get_citation_prompt()
            formatted_prompt = prompt_template.format(
                context=context_text,
                query=query,
                history=history_text
            )
            
            logger.info("=" * 60)
            logger.info("📝 COMPLETE PROMPT SENT TO LLM:")
            logger.info("=" * 60)
            logger.info(f"{formatted_prompt}")
            logger.info("=" * 60)
            
            chain = prompt_template | self.llm_service.llm | StrOutputParser()
            
            logger.info("📝 GENERATING ANSWER FROM LLM...")
            
            full_answer = await chain.ainvoke({
                "context": context_text,
                "query": query,
                "history": history_text
            })
            
            full_answer = full_answer.strip()
            
            logger.info(f"   Full answer length: {len(full_answer)} chars")
            
            logger.info("=" * 80)
            logger.info("📝 COMPLETE ANSWER FROM LLM:")
            logger.info("=" * 80)
            logger.info(f"\n{full_answer}\n")
            logger.info("=" * 80)
            
            if not full_answer or len(full_answer) < self.MIN_ANSWER_LENGTH:
                logger.warning(f"   ⚠️ LLM returned empty or too short response: {len(full_answer)} chars")
                yield {
                    "type": "error",
                    "error": f"LLM response too short: {len(full_answer)} chars"
                }
                return
            
            # Stream the answer as-is
            logger.info("=" * 60)
            logger.info("📝 STREAMING ANSWER:")
            logger.info("-" * 60)
            
            chunk_index = 0
            total_len = len(full_answer)
            
            for i in range(0, total_len, chunk_size):
                chunk = full_answer[i:i + chunk_size]
                
                yield {
                    "type": "chunk",
                    "chunk": chunk,
                    "chunk_index": chunk_index,
                    "is_last": (i + chunk_size >= total_len),
                    "metadata": {
                        "chunk_size": len(chunk),
                        "total_so_far": min(i + chunk_size, total_len),
                        "total_answer_length": total_len,
                        "progress": f"{min(i + chunk_size, total_len)}/{total_len}"
                    }
                }
                chunk_index += 1
                
                if char_delay > 0:
                    await asyncio.sleep(char_delay)
            
            logger.info("-" * 60)
            logger.info(f"   Total chunks: {chunk_index}")
            logger.info(f"   Total characters: {len(full_answer)}")
            
            
            print("=========++++++++++++++++++++++++++=====================", flush=True)
            print(f"{full_answer}", flush=True)
            print("=========++++++++++++++++++++++++++=====================", flush=True)

            
            # Yield completion with the LLM's answer
            yield {
                "type": "complete",
                "full_answer": full_answer,
                "full_answer_with_sources": full_answer,
                "metadata": {
                    "answer_length": len(full_answer),
                    "total_chunks": chunk_index
                }
            }
            
            logger.info(f"   ✅ Streaming completed")
            
        except Exception as e:
            logger.error(f"   ❌ Generation streaming failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield {
                "type": "error",
                "error": str(e)
            }
    
    def _build_context_with_sources(
        self,
        documents: List[Document],
        max_length: int
    ) -> tuple:
        """
        Build context with smart truncation to fit within limits.
        """
        context_parts = []
        context_snippets = []
        source_map = {}
        total_length = 0
        
        for i, doc in enumerate(documents):
            content = doc.page_content
            filename = doc.metadata.get('filename', f'Source_{i+1}.pdf')
            page = doc.metadata.get('page_number', 'N/A')
            doc_id = doc.metadata.get('document_id', f'doc_{i+1}')
            
            if len(content) > self.MAX_SNIPPET_CHARS:
                first_part = content[:300]
                last_part = content[-200:] if len(content) > 500 else ""
                content = first_part
                if last_part:
                    content += f"\n... (truncated) ...\n{last_part}"
            
            citation_num = i + 1
            source_key = f"[{citation_num}]"
            
            snippet = f"{source_key} {filename} (Page {page}):\n{content}"
            
            if total_length + len(snippet) > max_length:
                remaining = max_length - total_length
                if remaining > 100:
                    snippet = snippet[:remaining] + "..."
                    context_parts.append(snippet)
                    total_length += len(snippet)
                break
            else:
                context_parts.append(snippet)
                total_length += len(snippet)
            
            source_map[source_key] = {
                "citation_number": citation_num,
                "filename": filename,
                "page": page,
                "document_id": doc_id,
                "content": content
            }
            
            context_snippets.append({
                "content": content,
                "source": filename,
                "page": page,
                "citation": citation_num
            })
        
        context_text = "\n\n---\n\n".join(context_parts)
        logger.debug(f"Total context length: {len(context_text)} chars (limit: {max_length})")
        return context_text, context_snippets, source_map
    
    def _format_history(self, history: Optional[List[Dict[str, str]]]) -> str:
        """Format conversation history for prompt."""
        if not history:
            return "No previous conversation."
        
        recent = history[-4:] if len(history) > 4 else history
        formatted = []
        
        for i in range(0, len(recent), 2):
            if i + 1 < len(recent):
                user_msg = recent[i]
                assistant_msg = recent[i + 1]
                
                if user_msg.get('role') == 'user':
                    content = user_msg.get('content', '')
                    if len(content) > 300:
                        content = content[:300] + "..."
                    formatted.append(f"User: {content}")
                if assistant_msg.get('role') == 'assistant':
                    content = assistant_msg.get('content', '')
                    if len(content) > 300:
                        content = content[:300] + "..."
                    formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted)
  
    def _get_citation_prompt(self) -> ChatPromptTemplate:
        """Get prompt for answer generation with sources."""
        return ChatPromptTemplate.from_template("""
    You are a helpful assistant that answers questions based ONLY on the provided context.

    ### Instructions:
    1. Answer the user's question using ONLY the information from the context below.
    2. Do NOT use any external knowledge or your own knowledge.
    3. If the context does not contain the answer, say "I don't have enough information to answer this question."
    4. Provide a **comprehensive and detailed answer** that fully explains the topic.
    5. Use multiple sentences and paragraphs to elaborate on the information.
    6. After your answer, ALWAYS include a "Sources:" section.
    7. In the Sources section, list each source you used with its filename and page number.

    ### Important Guidelines for Quality Answers:
    - Be thorough - explain concepts in detail, not just one sentence.
    - Use the full context provided - include all relevant information.
    - Structure your answer with clear explanations and logical flow.
    - If there are multiple aspects to the topic, address all of them.
    - Aim for at least 3-5 sentences for substantive questions.
    - Use natural language and avoid bullet points in the answer text.

    ### Example of a GOOD answer:
    Here is how your answer should look (notice the detail):

    Retrieval Augmented Generation (RAG) is a technique that enhances large language models 
    by incorporating external knowledge retrieval into the generation process. The system first 
    takes the user's query and uses it to search a knowledge base or vector database for 
    relevant documents or chunks. These retrieved documents are then used as additional 
    context for the LLM when generating the response. This approach significantly reduces 
    hallucinations because the LLM has access to factual, up-to-date information from the 
    knowledge base rather than relying solely on its training data. RAG is particularly useful 
    for tasks that require specific, current, or domain-specific knowledge, such as customer 
    support, medical information retrieval, or legal document analysis. The retrieval component 
    can use various techniques including dense vector search (FAISS), keyword-based search 
    (BM25), or hybrid approaches that combine both methods for optimal results.

    **Sources:**
    - Intoduction_to_RAG.pdf (Page 3)
    - RAG_architecture.pdf (Page 5)

    ### Context:
    {context}

    ### User Question:
    {query}

    ### Answer (provide a DETAILED and COMPREHENSIVE response):
    """)