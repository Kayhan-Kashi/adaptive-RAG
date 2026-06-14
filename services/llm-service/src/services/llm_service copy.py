# llm-service/src/services/llm_service.py
import os
import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM operations using Ollama"""
    
    def __init__(self):
        # Initialize Ollama LLM
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer concisely and accurately."),
            ("human", "{question}")
        ])
        
        # Create chain with pipe operator
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        logger.info(f"✅ LLM Service initialized")
        logger.info(f"   Model: {os.getenv('OLLAMA_MODEL', 'gemma3:12b')}")
        logger.info(f"   Ollama URL: {os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')}")
    
    async def generate(self, prompt: str) -> str:
        """Generate answer from prompt"""
        try:
            response = await self.chain.ainvoke({"question": prompt})
            return response.strip()
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error generating response: {str(e)}"