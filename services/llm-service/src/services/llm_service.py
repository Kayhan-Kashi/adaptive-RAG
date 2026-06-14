import os
import logging
from injector import inject
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
from src.services.rag_service import RagService

logger = logging.getLogger(__name__)

class LLMService:
    """Service for RAG-enhanced LLM operations using Ensemble Retriever only"""
    
    @inject
    def __init__(self, rag_service: RagService):
        self.rag_service = rag_service
        
        # 1. Initialize LLM
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
        )
        
        # 2. Define RAG Prompt with better instructions
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the user's question based ONLY on the provided context.
        If the answer is not in the context, say that you don't know.
        
        <context>
        {context}
        </context>

        Question: {input}
        
        Answer: 
        """)
        
        # 3. Create document combination chain
        combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # 4. Create retrieval chain with Ensemble Retriever ONLY
        # Get ensemble retriever that combines FAISS and BM25
        self.ensemble_retriever = self.rag_service.get_ensemble_retriever(
            faiss_weight=float(os.getenv("FAISS_WEIGHT", "0.5")),
            bm25_weight=float(os.getenv("BM25_WEIGHT", "0.5")),
            faiss_k=int(os.getenv("FAISS_RETRIEVAL_K", "10")),
            bm25_k=int(os.getenv("BM25_RETRIEVAL_K", "10"))
        )
        
        # Check if ensemble retriever is available
        if self.ensemble_retriever is None:
            error_msg = (
                "❌ Ensemble Retriever is not available! "
                "Please ensure both FAISS and BM25 retrievers are initialized. "
                "Check if documents have been loaded and indexes exist."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Create the chain with ensemble retriever only
        self.chain = create_retrieval_chain(self.ensemble_retriever, combine_docs_chain)
        self.retriever_type = "ensemble (FAISS + BM25)"
        
        logger.info("✅ LLM Service initialized with Ensemble Retriever ONLY")
        logger.info(f"   FAISS weight: {os.getenv('FAISS_WEIGHT', '0.5')}")
        logger.info(f"   BM25 weight: {os.getenv('BM25_WEIGHT', '0.5')}")
        logger.info(f"   FAISS k: {os.getenv('FAISS_RETRIEVAL_K', '10')}")
        logger.info(f"   BM25 k: {os.getenv('BM25_RETRIEVAL_K', '10')}")
        
        # Log status of both retrievers
        stats = self.rag_service.get_stats()
        if stats['faiss']['total_vectors'] > 0:
            logger.info(f"   FAISS: {stats['faiss']['total_vectors']} vectors loaded")
        else:
            logger.warning("   ⚠️ FAISS vector store is empty")
        
        if stats['bm25']['total_chunks'] > 0:
            logger.info(f"   BM25: {stats['bm25']['total_chunks']} chunks loaded")
        else:
            logger.warning("   ⚠️ BM25 retriever is empty")

    async def generate(self, prompt: str) -> str:
        """Generate answer using RAG with Ensemble Retriever"""
        if not self.chain:
            error_msg = "Ensemble retriever chain is not initialized"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        try:
            # Log the user's question
            logger.info(f"📝 User Question: {prompt}")
            
            # Retrieve chunks using ensemble retriever for detailed logging
            retrieved_chunks = await self._retrieve_with_details(prompt)
            
            if not retrieved_chunks:
                logger.warning("No chunks retrieved from ensemble retriever")
                return "I don't have any relevant information to answer this question. Please add some documents first."
            
            # Log retrieved chunks
            logger.info(f"📚 Retrieved {len(retrieved_chunks)} chunks from {self.retriever_type}:")
            for i, chunk in enumerate(retrieved_chunks, 1):
                chunk_preview = chunk.page_content[:150].replace('\n', ' ')
                logger.info(f"   Chunk {i}: {chunk_preview}...")
                logger.info(f"      Metadata: document_id={chunk.metadata.get('document_id')}, "
                           f"filename={chunk.metadata.get('filename')}")
            
            # Run the chain
            response = await self.chain.ainvoke({"input": prompt})
            
            # Log the LLM's response
            logger.info(f"💬 LLM Response: {response['answer'][:300]}...")
            
            return response["answer"].strip()
            
        except Exception as e:
            logger.error(f"LLM RAG generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    async def generate_with_sources(self, prompt: str) -> Dict[str, Any]:
        """
        Generate answer and return with source documents and retrieval method info
        
        Returns:
            Dictionary with 'answer', 'sources', and 'retrieval_method'
        """
        if not self.chain:
            return {
                "answer": "Error: Ensemble retriever chain is not initialized.",
                "sources": [],
                "retrieval_method": "none"
            }

        try:
            logger.info(f"📝 User Question: {prompt}")
            
            # Retrieve chunks using ensemble retriever
            retrieved_chunks = await self._retrieve_with_details(prompt)
            
            if not retrieved_chunks:
                return {
                    "answer": "I don't have any relevant information to answer this question. Please add some documents first.",
                    "sources": [],
                    "retrieval_method": self.retriever_type,
                    "total_chunks_retrieved": 0
                }
            
            # Run the chain
            response = await self.chain.ainvoke({"input": prompt})
            
            # Prepare source information
            sources = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                sources.append({
                    "rank": i,
                    "content_preview": chunk.page_content[:200],
                    "full_content": chunk.page_content,
                    "document_id": chunk.metadata.get('document_id'),
                    "filename": chunk.metadata.get('filename'),
                    "filetype": chunk.metadata.get('filetype'),
                    "source": chunk.metadata.get('source', 'unknown')
                })
            
            logger.info(f"💬 Generated answer with {len(sources)} sources")
            
            return {
                "answer": response["answer"].strip(),
                "sources": sources,
                "retrieval_method": self.retriever_type,
                "total_chunks_retrieved": len(retrieved_chunks),
                "config": {
                    "faiss_weight": float(os.getenv("FAISS_WEIGHT", "0.5")),
                    "bm25_weight": float(os.getenv("BM25_WEIGHT", "0.5"))
                }
            }
            
        except Exception as e:
            logger.error(f"LLM RAG generation error: {e}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "retrieval_method": self.retriever_type,
                "error": str(e)
            }
    
    async def generate_with_ensemble_only(self, prompt: str, k: int = 5) -> Dict[str, Any]:
        """
        Generate answer using ONLY ensemble retriever with detailed output
        
        Args:
            prompt: User question
            k: Number of chunks to retrieve
        
        Returns:
            Dictionary with answer, retrieved chunks, and weights used
        """
        if not self.chain:
            return {
                "answer": "Ensemble retriever is not available",
                "retrieved_chunks": [],
                "weights_used": None
            }
        
        try:
            # Get the ensemble retriever directly
            ensemble_retriever = self.rag_service.get_ensemble_retriever(
                faiss_weight=float(os.getenv("FAISS_WEIGHT", "0.5")),
                bm25_weight=float(os.getenv("BM25_WEIGHT", "0.5")),
                faiss_k=k * 2,
                bm25_k=k * 2
            )
            
            # Retrieve chunks
            retrieved_chunks = await ensemble_retriever.ainvoke(prompt)
            retrieved_chunks = retrieved_chunks[:k]
            
            # Generate answer
            response = await self.chain.ainvoke({"input": prompt})
            
            return {
                "answer": response["answer"].strip(),
                "retrieved_chunks": [
                    {
                        "content": chunk.page_content,
                        "metadata": chunk.metadata
                    }
                    for chunk in retrieved_chunks
                ],
                "weights_used": {
                    "faiss": float(os.getenv("FAISS_WEIGHT", "0.5")),
                    "bm25": float(os.getenv("BM25_WEIGHT", "0.5"))
                },
                "num_chunks_retrieved": len(retrieved_chunks)
            }
            
        except Exception as e:
            logger.error(f"Ensemble only generation error: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "retrieved_chunks": [],
                "weights_used": None
            }
    
    async def _retrieve_with_details(self, query: str, k: int = 10) -> List[Document]:
        """
        Retrieve chunks using ensemble retriever
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of documents from ensemble retrieval
        """
        # Use the search_with_ensemble method from rag_service
        if hasattr(self.rag_service, 'search_with_ensemble'):
            results = self.rag_service.search_with_ensemble(
                query=query,
                final_k=k,
                faiss_weight=float(os.getenv("FAISS_WEIGHT", "0.5")),
                bm25_weight=float(os.getenv("BM25_WEIGHT", "0.5")),
                faiss_k=int(os.getenv("FAISS_RETRIEVAL_K", "10")),
                bm25_k=int(os.getenv("BM25_RETRIEVAL_K", "10"))
            )
            return results
        else:
            # If search_with_ensemble doesn't exist, use ensemble retriever directly
            if self.ensemble_retriever:
                results = await self.ensemble_retriever.ainvoke(query)
                return results[:k]
            return []
    
    def check_ensemble_status(self) -> Dict[str, Any]:
        """
        Check if ensemble retriever is properly configured
        
        Returns:
            Dictionary with status information
        """
        stats = self.rag_service.get_stats()
        
        is_faiss_ready = stats['faiss']['total_vectors'] > 0
        is_bm25_ready = stats['bm25']['total_chunks'] > 0
        is_ensemble_ready = is_faiss_ready and is_bm25_ready
        
        return {
            "ensemble_ready": is_ensemble_ready,
            "faiss_ready": is_faiss_ready,
            "bm25_ready": is_bm25_ready,
            "faiss_vectors": stats['faiss']['total_vectors'],
            "bm25_chunks": stats['bm25']['total_chunks'],
            "retriever_type": self.retriever_type if hasattr(self, 'retriever_type') else "unknown",
            "config": {
                "faiss_weight": os.getenv("FAISS_WEIGHT", "0.5"),
                "bm25_weight": os.getenv("BM25_WEIGHT", "0.5"),
                "faiss_k": os.getenv("FAISS_RETRIEVAL_K", "10"),
                "bm25_k": os.getenv("BM25_RETRIEVAL_K", "10")
            }
        }
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the current retrieval setup"""
        stats = self.rag_service.get_stats()
        stats["retriever_type"] = self.retriever_type
        stats["ensemble_active"] = self.ensemble_retriever is not None
        stats["config"] = {
            "faiss_weight": os.getenv("FAISS_WEIGHT", "0.5"),
            "bm25_weight": os.getenv("BM25_WEIGHT", "0.5"),
            "faiss_k": os.getenv("FAISS_RETRIEVAL_K", "10"),
            "bm25_k": os.getenv("BM25_RETRIEVAL_K", "10"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            "temperature": os.getenv("OLLAMA_TEMPERATURE", "0.3")
        }
        return stats