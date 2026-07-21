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
        pronouns = ["it", "this", "that", "these", "those", "they", "them", "he", "she", "his", "her", "its"]
        has_pronoun = any(p in query.lower().split() for p in pronouns)
        
        if not has_pronoun:
            logger.debug("Query has no pronouns, returning original")
            return query
        
        # Format history - ONLY USER QUESTIONS
        history_text = self._format_history(history)
        
        # ✅ Log the prompt being sent
        logger.info("=" * 60)
        logger.info("📝 COREFERENCE RESOLUTION PROMPT:")
        logger.info("=" * 60)
        logger.info(f"User Questions History: {history_text}")
        logger.info(f"Current Query: {query}")
        logger.info("-" * 60)
        
        # ✅ IMPROVED SYSTEM PROMPT - STRICT OUTPUT FORMAT
        prompt = ChatPromptTemplate.from_template("""
You are an expert at resolving pronouns in follow-up questions.

### PREVIOUS USER QUESTIONS:
{history}

### CURRENT QUESTION:
{query}

### TASK:
Replace ALL pronouns in the CURRENT QUESTION with the correct entities from the PREVIOUS USER QUESTIONS.

### CRITICAL RULES:
1. Look at the user's PREVIOUS QUESTIONS to find the MAIN TOPIC
2. Replace ALL pronouns (it, this, that, they, them, etc.) with the FULL ENTITY NAME
3. PRESERVE the original question structure - only change the pronouns
4. OUTPUT ONLY the rewritten question, NOTHING ELSE

### OUTPUT FORMAT - STRICT:
- You MUST output ONLY the rewritten question
- Do NOT add "Answer:", "Output:", "Result:", or any other prefixes
- Do NOT add explanations, quotes, or extra text
- Do NOT add a period at the end unless the original had one
- The output must be a SINGLE sentence

### Examples:
Previous Questions:
- User: What is hallucination?
Current Question: How does it cause problems?
OUTPUT: How does hallucination cause problems?

Previous Questions:
- User: What is RAG?
Current Question: What are its benefits?
OUTPUT: What are RAG's benefits?

Previous Questions:
- User: What is hallucination?
Current Question: By what tool can we solve it?
OUTPUT: By what tool can we solve hallucination?

Previous Questions:
- User: What is RAG?
Current Question: How it causes cost?
OUTPUT: How does RAG cause cost?

### REWRITTEN QUESTION (OUTPUT ONLY THIS, NOTHING ELSE):
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
        
        # Remove common prefixes
        prefixes = [
            "Answer:",
            "Output:",
            "Result:",
            "Resolved:",
            "Rewritten:",
            "Question:",
            "OUTPUT:",
            "ANSWER:",
        ]
        for prefix in prefixes:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        # Remove quotes
        response = response.strip('"\' ')
        
        # If response is empty or just whitespace, return original
        if not response or len(response) < 2:
            return original_query
        
        # If response is too long (more than 200 chars), extract first sentence
        if len(response) > 200:
            # Try to get the first sentence
            first_sentence = response.split('.')[0].strip()
            if len(first_sentence) > 10:
                # Check if it still has extra text
                if "Answer:" in first_sentence or "Output:" in first_sentence:
                    return original_query
                return first_sentence
            # If no good sentence, return original
            return original_query
        
        # Final check: if response still contains "Answer:" or other prefixes
        if "Answer:" in response or "Output:" in response or "Result:" in response:
            # Try to extract just the question
            for prefix in prefixes:
                if prefix in response:
                    parts = response.split(prefix, 1)
                    if len(parts) > 1:
                        response = parts[1].strip()
                    break
        
        return response
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """
        Format conversation history - ONLY extract user questions.
        Assistant responses are ignored because they can introduce confusion.
        """
        if not history:
            return "No previous conversation."
        
        user_questions = []
        
        for msg in history:
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            
            # ✅ ONLY collect user messages
            if role == 'user':
                user_questions.append(f"User: {content[:500]}")
            # Try other common formats
            elif 'prompt' in msg:
                user_questions.append(f"User: {msg.get('prompt', '')[:500]}")
            elif 'user' in msg and not 'assistant' in msg:
                user_questions.append(f"User: {msg.get('user', '')[:500]}")
        
        # ✅ Take last 5 user questions maximum
        if len(user_questions) > 5:
            user_questions = user_questions[-5:]
        
        result = "\n".join(user_questions)
        
        if not result:
            return "No previous conversation."
        
        logger.debug(f"User questions history: {result}")
        return result