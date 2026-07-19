import os
from injector import inject
from langchain_ollama import ChatOllama


class LLMService:

    @inject
    def __init__(self):
        self._llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
        )
    
    @property
    def llm(self):
        """Get the underlying LLM instance for chaining."""
        return self._llm