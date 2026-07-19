import logging
import re
from typing import Dict, List, Optional
from injector import inject
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_service import LLMService

logger = logging.getLogger(__name__)


class CoreferenceResolver:
    """Handles coreference resolution for multilingual queries."""
    
    @inject
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def resolve(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Resolve pronouns and references in the query using conversation history.
        """
        if not query:
            return query
        
        # If no history, just return the query
        if not history:
            logger.debug("No history provided, returning original query")
            return query
        
        # Check if query contains pronouns
        pronouns = ["it", "this", "that", "these", "those", "they", "them", "he", "she", "his", "her"]
        has_pronoun = any(p in query.lower().split() for p in pronouns)
        
        if not has_pronoun:
            logger.debug("Query has no pronouns, returning original")
            return query
        
        history_text = self._format_history(history)
        
        # Improved prompt with better instructions and examples
        prompt = ChatPromptTemplate.from_template("""
You are an expert at resolving pronouns and references in questions using conversation history.

### CONVERSATION HISTORY:
{history}

### CURRENT QUESTION:
{query}

### TASK:
Replace ALL pronouns in the CURRENT QUESTION with the correct entities from the CONVERSATION HISTORY.

### IMPORTANT RULES:
1. Identify the main entity/subject from the conversation history (e.g., RAG, GPT, Transformers)
2. Replace ALL pronouns (it, they, them, this, that, these, those) with the correct entity name
3. Keep the question structure exactly the same - only change the pronouns
4. Output ONLY the rewritten question, nothing else
5. Do NOT use markdown, code blocks, HTML tags, or any formatting
6. Do NOT add extra text, explanations, or punctuation

### EXAMPLES:
History: User: "What is RAG?" Assistant: "RAG is Retrieval Augmented Generation..."
Question: "What is it used for?"
Output: What is RAG used for?

History: User: "Tell me about transformers." Assistant: "Transformers are neural networks..."
Question: "How do they work?"
Output: How do transformers work?

History: User: "What is GPT?" Assistant: "GPT is Generative Pre-trained Transformer..."
Question: "What are the benefits?"
Output: What are the benefits of GPT?

### REWRITTEN QUESTION (ONLY THE QUESTION, NO FORMATTING):
""")
        
        chain = prompt | self.llm_service.llm | StrOutputParser()
        
        try:
            resolved = await chain.ainvoke({
                "history": history_text,
                "query": query
            })
            
            # Clean the response
            resolved = self._clean_response(resolved, query)
            
            # Validate the result
            if resolved and resolved != query and len(resolved) > 3:
                logger.info(f"Resolved: '{query}' → '{resolved}'")
                return resolved
            
            logger.debug(f"Resolution failed or unchanged, returning original: '{query}'")
            return query
            
        except Exception as e:
            logger.error(f"Coreference resolution failed: {e}")
            return query
    
    def _clean_response(self, response: str, original_query: str) -> str:
        """
        Clean the LLM response to extract just the resolved query.
        Removes markdown, code blocks, HTML tags, and extra formatting.
        """
        if not response:
            return original_query
        
        # Remove markdown code blocks (```python, ```, etc.)
        response = re.sub(r'```[\w]*\n?', '', response)
        response = re.sub(r'```\n?', '', response)
        
        # Remove HTML tags (<sub>, </sub>, <h1>, </h1>, etc.)
        response = re.sub(r'<[^>]+>', '', response)
        
        # Remove any remaining backticks
        response = response.replace('`', '')
        
        # Remove extra newlines
        response = re.sub(r'\n+', ' ', response)
        
        # Remove extra spaces
        response = re.sub(r'\s+', ' ', response)
        
        # Strip quotes and whitespace
        response = response.strip('"\' ')
        
        # If response is empty or just whitespace, return original
        if not response or len(response) < 2:
            return original_query
        
        # If response is too long (more than 200 chars), extract first sentence
        if len(response) > 200:
            # Try to get the first sentence
            first_sentence = response.split('.')[0].strip()
            if len(first_sentence) > 10:
                return first_sentence
            # If no good sentence, return original
            return original_query
        
        return response
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return "No previous conversation."
        
        # Take last 4 turns (8 messages max)
        recent = history[-4:] if len(history) > 4 else history
        formatted = []
        
        for i in range(0, len(recent), 2):
            if i + 1 < len(recent):
                user_msg = recent[i]
                assistant_msg = recent[i + 1]
                
                if user_msg.get('role') == 'user':
                    content = user_msg.get('content', '')
                    formatted.append(f"User: {content[:500]}")
                if assistant_msg.get('role') == 'assistant':
                    content = assistant_msg.get('content', '')
                    formatted.append(f"Assistant: {content[:500]}")
        
        return "\n".join(formatted)