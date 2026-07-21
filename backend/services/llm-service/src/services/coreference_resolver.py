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
        
        # ✅ IMPROVED SYSTEM PROMPT WITH LENGTH CONSTRAINT
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
4. Keep the question CONCISE - between 40 to 80 characters
5. Do NOT add extra words, explanations, or rephrase the question
6. OUTPUT ONLY the rewritten question, NOTHING ELSE

### OUTPUT FORMAT - STRICT:
- You MUST output ONLY the rewritten question
- The question should be 40-80 characters long (concise)
- Do NOT add "Answer:", "Output:", "Result:", or any other prefixes
- Do NOT add explanations, quotes, or extra text
- Do NOT expand the question into multiple sentences or paragraphs
- Keep it as a SINGLE short sentence

### Examples:
Previous Questions:
- User: What is hallucination?
Current Question: How does it cause problems?
OUTPUT: How does hallucination cause problems?
✅ Length: 39 characters (concise)

Previous Questions:
- User: What is RAG?
Current Question: What are its benefits?
OUTPUT: What are RAG's benefits?
✅ Length: 25 characters (concise)

Previous Questions:
- User: What is hallucination?
Current Question: By what tool can we solve it?
OUTPUT: By what tool can we solve hallucination?
✅ Length: 44 characters (concise)

Previous Questions:
- User: What is RAG?
Current Question: How it causes cost?
OUTPUT: How does RAG cause cost?
✅ Length: 27 characters (concise)

❌ BAD EXAMPLES (TOO LONG):
Previous Questions:
- User: What is RAG?
Current Question: Can you explain it to me?
BAD OUTPUT: Can you explain Retrieval Augmented Generation to me in detail please? 
❌ Too long and adds extra words

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
            
            # ✅ Enforce length constraint
            resolved = self._enforce_length_constraint(resolved, query)
            
            # Validate the result
            if resolved and resolved != query and len(resolved) > 3:
                logger.info(f"Resolved: '{query}' → '{resolved}'")
                return resolved
            
            logger.debug(f"Resolution failed or unchanged, returning original: '{query}'")
            return query
            
        except Exception as e:
            logger.error(f"Coreference resolution failed: {e}")
            return query
    
    def _enforce_length_constraint(self, resolved: str, original_query: str) -> str:
        """
        Enforce length constraint on resolved query.
        If too long (>80 chars), extract the core question.
        If too short (<20 chars), keep as is but log warning.
        """
        if not resolved:
            return original_query
        
        # If already within reasonable length (20-80 chars)
        if 20 <= len(resolved) <= 80:
            return resolved
        
        # If too short (less than 20 chars), it might be incomplete
        if len(resolved) < 20:
            logger.debug(f"Resolved query too short ({len(resolved)} chars): '{resolved}'")
            return resolved if len(resolved) > 5 else original_query
        
        # If too long (more than 80 chars), try to extract first sentence
        if len(resolved) > 80:
            logger.debug(f"Resolved query too long ({len(resolved)} chars), extracting first sentence")
            
            # Split by periods, question marks, or exclamation points
            sentences = re.split(r'[.!?]\s+', resolved)
            if sentences:
                first_sentence = sentences[0].strip()
                # If first sentence is still too long, truncate
                if len(first_sentence) > 80:
                    # Try to truncate at a comma or space
                    truncate_at = first_sentence[:80].rfind(' ')
                    if truncate_at > 40:
                        first_sentence = first_sentence[:truncate_at] + "?"
                    else:
                        first_sentence = first_sentence[:80] + "?"
                return first_sentence
        
        return resolved
    
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