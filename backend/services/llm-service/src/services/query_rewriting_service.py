import logging
from typing import List, Dict, Any
from injector import inject
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_service import LLMService
from .query_state import QueryState

logger = logging.getLogger(__name__)


class QueryRewritingService:
    """Handles query expansion and decomposition based on query length."""
    
    @inject
    def __init__(self, llm_service: LLMService):
        self.llm_service = LLMService()
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and determine its state based on length.
        
        Returns:
            Dict with:
            - query_state: str ("short", "well_formed", "long")
            - query_length: int
            - needs_rewriting: bool
        """
        query_state = QueryState.analyze_query_state(query)
        query_length = len(query.strip())
        
        needs_rewriting = query_state in [QueryState.SHORT, QueryState.LONG]
        
        return {
            "query_state": query_state.value,
            "query_length": query_length,
            "needs_rewriting": needs_rewriting
        }
    
    async def rewrite_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and rewrite it based on length.
        
        Returns:
            Dict with:
            - original_query: str
            - rewritten_queries: List[str]
            - query_state: str ("short", "well_formed", "long")
            - query_length: int
            - needs_rewriting: bool
        """
        analysis = await self.analyze_query(query)
        
        query_state = analysis["query_state"]
        query_length = analysis["query_length"]
        needs_rewriting = analysis["needs_rewriting"]
        
        if not needs_rewriting:
            return {
                "original_query": query,
                "rewritten_queries": [query],
                "query_state": query_state,
                "query_length": query_length,
                "needs_rewriting": False
            }
        
        query_rewrite_info = await self._modify_query(query, query_state)
        
        if query_state == QueryState.LONG.value:
            rewritten_queries = [
                q.strip("-•1234567890. ").strip() 
                for q in query_rewrite_info["rewritten_query"].split("\n")
                if q.strip()
            ]
        else:
            rewritten_queries = [query_rewrite_info["rewritten_query"]]
        
        return {
            "original_query": query,
            "rewritten_queries": rewritten_queries,
            "query_state": query_state,
            "query_length": query_length,
            "needs_rewriting": True
        }
    
    async def _modify_query(self, query: str, query_state: str) -> Dict[str, Any]:
        """Modify query based on its state."""
        
        if query_state == QueryState.SHORT.value:
            template = self._get_expand_prompt()
        elif query_state == QueryState.LONG.value:
            template = self._get_decompose_prompt()
        else:
            return {
                "original_query": query,
                "rewritten_query": query,
                "query_state": query_state
            }
        
        chain = template | self.llm_service.llm | StrOutputParser()
        
        try:
            rewritten_query = await chain.ainvoke({"query": query})
            rewritten_query = rewritten_query.strip()
            
            if not rewritten_query or len(rewritten_query) < 3:
                rewritten_query = query
            
            return {
                "original_query": query,
                "rewritten_query": rewritten_query,
                "query_state": query_state
            }
        except Exception as e:
            logger.error(f"Query modification failed: {e}")
            return {
                "original_query": query,
                "rewritten_query": query,
                "query_state": query_state
            }
    
    def _get_expand_prompt(self) -> ChatPromptTemplate:
        """Get prompt for expanding short queries."""
        return ChatPromptTemplate.from_template("""
You are a helpful assistant. Expand the following query to improve document retrieval by adding relevant synonyms, technical terms, and helpful context.

Original query: "{query}"

Expanded query (only the expanded query, no explanation):
""")
    
    def _get_decompose_prompt(self) -> ChatPromptTemplate:
        """Get prompt for decomposing long queries."""
        return ChatPromptTemplate.from_template("""
You are a helpful assistant. Decompose the following query into smaller, related components for better document retrieval.

Original query: "{query}"

Decomposed query (one question per line):
""")