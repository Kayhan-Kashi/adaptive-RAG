import logging
from typing import Dict, Any
from injector import inject
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_service import LLMService

logger = logging.getLogger(__name__)


class HyDEService:
    """
    Handles HyDE (Hypothetical Document Embeddings) generation.
    Generates a hypothetical document that would answer the query,
    which is then used for retrieval.
    """
    @inject
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def generate_hyde(self, query: str) -> Dict[str, Any]:
        """
        Generate a hypothetical document from the query.
        
        Args:
            query: User query
        
        Returns:
            Dict with:
            - original_query: str
            - hyde_document: str - The generated hypothetical document
            - hyde_used: bool - Whether HyDE was applied
        """
        logger.info(f"🔮 Generating HyDE for: {query[:100]}...")
        
        if not query or len(query.strip()) < 2:
            logger.warning("   Empty query, skipping HyDE")
            return {
                "original_query": query,
                "hyde_document": query,
                "hyde_used": False
            }
        
        try:
            chain = self._get_hyde_prompt() | self.llm_service.llm | StrOutputParser()
            hyde_document = await chain.ainvoke({"query": query})
            hyde_document = hyde_document.strip()
            
            # Validate output
            if not hyde_document or len(hyde_document) < 10:
                logger.warning(f"   HyDE output too short, using original query")
                return {
                    "original_query": query,
                    "hyde_document": query,
                    "hyde_used": False
                }
            
            logger.info(f"   ✅ HyDE generated: {hyde_document[:150]}...")
            
            return {
                "original_query": query,
                "hyde_document": hyde_document,
                "hyde_used": True
            }
            
        except Exception as e:
            logger.error(f"   ❌ HyDE generation failed: {e}")
            return {
                "original_query": query,
                "hyde_document": query,
                "hyde_used": False
            }
    
    async def generate_hyde_with_context(
        self, 
        query: str, 
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Generate a hypothetical document with additional context.
        
        Args:
            query: User query
            context: Additional context to guide generation
        
        Returns:
            Dict with:
            - original_query: str
            - hyde_document: str
            - hyde_used: bool
            - context_used: bool
        """
        logger.info(f"🔮 Generating HyDE with context for: {query[:100]}...")
        
        if not query or len(query.strip()) < 2:
            return {
                "original_query": query,
                "hyde_document": query,
                "hyde_used": False,
                "context_used": False
            }
        
        try:
            prompt = self._get_hyde_with_context_prompt()
            chain = prompt | self.llm_service.llm | StrOutputParser()
            
            hyde_document = await chain.ainvoke({
                "query": query,
                "context": context if context else "No additional context provided."
            })
            hyde_document = hyde_document.strip()
            
            if not hyde_document or len(hyde_document) < 10:
                return {
                    "original_query": query,
                    "hyde_document": query,
                    "hyde_used": False,
                    "context_used": False
                }
            
            return {
                "original_query": query,
                "hyde_document": hyde_document,
                "hyde_used": True,
                "context_used": bool(context)
            }
            
        except Exception as e:
            logger.error(f"   ❌ HyDE with context failed: {e}")
            return {
                "original_query": query,
                "hyde_document": query,
                "hyde_used": False,
                "context_used": False
            }
    
    async def generate_hypothetical_answer(self, query: str) -> Dict[str, Any]:
        """
        Generate a hypothetical answer (alias for generate_hyde).
        
        Args:
            query: User query
        
        Returns:
            Dict with:
            - original_query: str
            - hypothetical_answer: str
            - hyde_used: bool
        """
        result = await self.generate_hyde(query)
        # Rename key for clarity
        result["hypothetical_answer"] = result.pop("hyde_document")
        return result
    
    def _get_hyde_prompt(self) -> ChatPromptTemplate:
        """Get prompt for HyDE generation."""
        return ChatPromptTemplate.from_template("""
You are a helpful assistant. Given a user question, write a hypothetical document that would contain the answer.
This hypothetical document will be used for semantic search to find similar real documents.

Write a clear, detailed, and informative passage that answers the question. The passage should be 3-5 sentences long.

User Question: {query}

Hypothetical Document:
""")
    
    def _get_hyde_with_context_prompt(self) -> ChatPromptTemplate:
        """Get prompt for HyDE generation with additional context."""
        return ChatPromptTemplate.from_template("""
You are a helpful assistant. Given a user question and additional context, write a hypothetical document that would contain the answer.
This hypothetical document will be used for semantic search to find similar real documents.

Write a clear, detailed, and informative passage that answers the question. The passage should be 3-5 sentences long.

Additional Context: {context}

User Question: {query}

Hypothetical Document:
""")