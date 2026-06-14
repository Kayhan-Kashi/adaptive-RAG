import os
import logging
from injector import inject
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from src.services.rag_service import RagService

logger = logging.getLogger(__name__)

class LLMService:
    """Service for RAG-enhanced LLM operations"""
    
    @inject
    def __init__(self, rag_service: RagService):
        self.rag_service = rag_service
        
        # 1. Initialize LLM
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
        )
        
        # 2. Define RAG Prompt
        # This prompt forces the model to use the {context} retrieved from FAISS
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the user's question based ONLY on the provided context.
        If the answer is not in the context, say that you don't know.
        
        <context>
        {context}
        </context>

        Question: {input}
        """)
        
        # 3. Create Chains
        # "Stuff" documents puts all retrieved context into the prompt
        combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # 4. Create Retrieval Chain
        # This chain connects the retriever (FAISS) to the docs chain
        # NOTE: We ensure vector store is ready
        if self.rag_service.vector_store:
            retriever = self.rag_service.vector_store.as_retriever(
                search_kwargs={"k": 10}  # Retrieve top 10 chunks
            )
            self.chain = create_retrieval_chain(retriever, combine_docs_chain)
        else:
            logger.error("❌ RagService vector_store is missing! RAG chain will fail.")
            self.chain = None
        
        logger.info(f"✅ LLM Service initialized with RAG support")

    async def generate(self, prompt: str) -> str:
        """Generate answer using RAG"""
        if not self.chain:
            return "Error: RAG system is not initialized (no vector store found)."

        try:
            # Log the user's question
            logger.info(f"📝 User Question: {prompt}")
            
            # First, retrieve chunks manually for logging
            retrieved_chunks = self.rag_service.retrieve_similar_chunks(prompt, k=10)
            
            # Log retrieved chunks
            logger.info(f"📚 Retrieved {len(retrieved_chunks)} chunks from vector store:")
            for i, chunk in enumerate(retrieved_chunks, 1):
                chunk_preview = chunk.page_content[:150].replace('\n', ' ')
                logger.info(f"   Chunk {i}: {chunk_preview}...")
                logger.info(f"      Metadata: document_id={chunk.metadata.get('document_id')}, "
                           f"filename={chunk.metadata.get('filename')}")
            
            # Format the context from retrieved chunks
            context_text = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
            
            # Format the final prompt that will be sent to LLM
            formatted_prompt = self.prompt.format_messages(
                context=context_text,
                input=prompt
            )
            
            # Log the final prompt sent to LLM
            logger.info("=" * 80)
            logger.info("🤖 FINAL PROMPT SENT TO LLM:")
            logger.info("=" * 80)
            for message in formatted_prompt:
                logger.info(f"{message.type.upper()}: {message.content}")
            logger.info("=" * 80)
            
            # The retrieval chain expects an 'input' key
            # It returns a dictionary: {'answer': '...', 'context': [...]}
            response = await self.chain.ainvoke({"input": prompt})
            
            # Log the LLM's response
            logger.info(f"💬 LLM Response: {response['answer'][:300]}...")
            
            return response["answer"].strip()
            
        except Exception as e:
            logger.error(f"LLM RAG generation error: {e}")
            return f"Error generating response: {str(e)}"