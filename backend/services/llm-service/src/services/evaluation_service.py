import logging
from typing import List, Dict, Any
from injector import inject
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from .llm_service import LLMService

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for evaluating RAG pipeline outputs."""
    
    @inject
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def evaluate_faithfulness(
        self, 
        answer: str, 
        context: List[Document]
    ) -> Dict[str, Any]:
        """
        Evaluate if the answer is faithful to the provided context.
        Checks for hallucinations.
        """
        if not answer or not context:
            return {"score": 0.0, "is_faithful": False, "reason": "No answer or context"}
        
        context_text = "\n".join([doc.page_content[:500] for doc in context[:3]])
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator for RAG systems. Determine if the answer is faithful to the provided context.

### CONTEXT:
{context}

### ANSWER:
{answer}

### TASK:
Determine if the answer is FAITHFUL to the context (doesn't contain hallucinations or unsupported claims).

### OUTPUT FORMAT:
Return a JSON with:
- "score": a float between 0 and 1 (1 = completely faithful)
- "is_faithful": true/false
- "reason": brief explanation
- "hallucinated_parts": list of parts that are not supported by context (if any)

### EVALUATION:
""")
        
        # Log the prompt being sent
        prompt_messages = prompt.format_messages(context=context_text, answer=answer)
        logger.info(f"=== FAITHFULNESS EVALUATION PROMPT ===\n{prompt_messages[0].content}\n=== END PROMPT ===")
        
        chain = prompt | self.llm_service.llm | StrOutputParser()
        
        try:
            result = await chain.ainvoke({
                "context": context_text,
                "answer": answer
            })
            
            logger.info(f"=== FAITHFULNESS EVALUATION RESPONSE ===\n{result}\n=== END RESPONSE ===")
            
            # Parse result (simplified - use JSON parser in production)
            import json
            try:
                data = json.loads(result)
                return {
                    "score": data.get("score", 0.0),
                    "is_faithful": data.get("is_faithful", False),
                    "reason": data.get("reason", ""),
                    "hallucinated_parts": data.get("hallucinated_parts", [])
                }
            except:
                # Fallback parsing
                if "faithful" in result.lower() and "true" in result.lower():
                    return {"score": 0.9, "is_faithful": True, "reason": "Deemed faithful by LLM"}
                else:
                    return {"score": 0.3, "is_faithful": False, "reason": "Potential hallucinations detected"}
                    
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return {"score": 0.0, "is_faithful": False, "reason": f"Evaluation error: {e}"}
    
    async def evaluate_answer_relevance(
        self,
        query: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate if the answer is relevant to the query.
        """
        if not query or not answer:
            return {"score": 0.0, "is_relevant": False, "reason": "No query or answer"}
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator for RAG systems. Determine if the answer is relevant to the query.

### QUERY:
{query}

### ANSWER:
{answer}

### TASK:
Determine if the answer RELEVANT to the query (directly addresses the question).

### OUTPUT FORMAT:
Return a JSON with:
- "score": a float between 0 and 1 (1 = completely relevant)
- "is_relevant": true/false
- "reason": brief explanation

### EVALUATION:
""")
        
        # Log the prompt being sent
        prompt_messages = prompt.format_messages(query=query, answer=answer)
        logger.info(f"=== ANSWER RELEVANCE EVALUATION PROMPT ===\n{prompt_messages[0].content}\n=== END PROMPT ===")
        
        chain = prompt | self.llm_service.llm | StrOutputParser()
        
        try:
            result = await chain.ainvoke({
                "query": query,
                "answer": answer
            })
            
            logger.info(f"=== ANSWER RELEVANCE EVALUATION RESPONSE ===\n{result}\n=== END RESPONSE ===")
            
            import json
            try:
                data = json.loads(result)
                return {
                    "score": data.get("score", 0.0),
                    "is_relevant": data.get("is_relevant", False),
                    "reason": data.get("reason", "")
                }
            except:
                if "relevant" in result.lower() and "true" in result.lower():
                    return {"score": 0.9, "is_relevant": True, "reason": "Deemed relevant by LLM"}
                else:
                    return {"score": 0.3, "is_relevant": False, "reason": "Deemed not relevant by LLM"}
                    
        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            return {"score": 0.0, "is_relevant": False, "reason": f"Evaluation error: {e}"}
    
    async def evaluate_retrieval_quality(
        self,
        query: str,
        retrieved_chunks: List[Document]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of retrieved chunks.
        """
        if not query or not retrieved_chunks:
            return {"score": 0.0, "is_good": False, "reason": "No query or chunks"}
        
        chunks_text = "\n".join([doc.page_content[:300] for doc in retrieved_chunks[:3]])
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator for RAG retrieval. Determine if the retrieved chunks are relevant to the query.

### QUERY:
{query}

### TOP RETRIEVED CHUNKS:
{chunks}

### TASK:
Determine if the RETRIEVED CHUNKS are relevant to the query.

### OUTPUT FORMAT:
Return a JSON with:
- "score": a float between 0 and 1 (1 = all relevant)
- "is_good": true/false
- "reason": brief explanation
- "relevant_count": number of relevant chunks out of {total_chunks}

### EVALUATION:
""")
        
        # Log the prompt being sent
        prompt_messages = prompt.format_messages(
            query=query, 
            chunks=chunks_text, 
            total_chunks=len(retrieved_chunks)
        )
        logger.info(f"=== RETRIEVAL QUALITY EVALUATION PROMPT ===\n{prompt_messages[0].content}\n=== END PROMPT ===")
        
        chain = prompt | self.llm_service.llm | StrOutputParser()
        
        try:
            result = await chain.ainvoke({
                "query": query,
                "chunks": chunks_text,
                "total_chunks": len(retrieved_chunks)
            })
            
            logger.info(f"=== RETRIEVAL QUALITY EVALUATION RESPONSE ===\n{result}\n=== END RESPONSE ===")
            
            import json
            try:
                data = json.loads(result)
                return {
                    "score": data.get("score", 0.0),
                    "is_good": data.get("is_good", False),
                    "reason": data.get("reason", ""),
                    "relevant_count": data.get("relevant_count", 0)
                }
            except:
                if "good" in result.lower() and "true" in result.lower():
                    return {"score": 0.8, "is_good": True, "reason": "Retrieved chunks seem relevant"}
                else:
                    return {"score": 0.3, "is_good": False, "reason": "Retrieved chunks seem irrelevant"}
                    
        except Exception as e:
            logger.error(f"Retrieval quality evaluation failed: {e}")
            return {"score": 0.0, "is_good": False, "reason": f"Evaluation error: {e}"}