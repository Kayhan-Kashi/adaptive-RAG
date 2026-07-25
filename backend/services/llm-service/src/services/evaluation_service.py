import logging
import json
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
        
        # ✅ IMPROVED PROMPT: Stronger instructions for JSON output
        prompt = ChatPromptTemplate.from_template("""
You are a strict evaluator for RAG systems. Your task is to determine if the answer is faithful to the provided context.

### CONTEXT:
{context}

### ANSWER:
{answer}

### TASK:
Carefully check if the ANSWER contains any information that is NOT supported by the CONTEXT.
- A faithful answer only uses information present in the context
- An unfaithful answer contains hallucinations, made-up facts, or unsupported claims
- Be strict and critical in your evaluation

### INSTRUCTIONS:
1. Compare each claim in the answer against the context
2. Identify any statements that cannot be verified from the context
3. Provide a score from 0.0 to 1.0 where:
   - 1.0 = All claims are fully supported by context
   - 0.7-0.9 = Mostly supported with minor unsupported details
   - 0.4-0.6 = Partially supported with significant unsupported claims
   - 0.0-0.3 = Mostly unsupported or completely hallucinated

### OUTPUT FORMAT:
You MUST respond with ONLY a valid JSON object. No other text.

Example:
{{"score": 0.95, "is_faithful": true, "reason": "All claims are supported by the context", "hallucinated_parts": []}}

### YOUR JSON RESPONSE:
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
            
            # ✅ IMPROVED PARSING: Extract JSON from response
            try:
                # Try to find JSON in the response
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = result[json_start:json_end]
                    data = json.loads(json_str)
                    
                    # Ensure we have valid scores
                    score = float(data.get("score", 0.0))
                    score = max(0.0, min(1.0, score))  # Clamp to 0-1
                    
                    return {
                        "score": score,
                        "is_faithful": data.get("is_faithful", score > 0.5),
                        "reason": data.get("reason", ""),
                        "hallucinated_parts": data.get("hallucinated_parts", [])
                    }
                else:
                    # No JSON found
                    logger.warning("No JSON found in response, using fallback")
                    return self._fallback_faithfulness_parse(result)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed: {e}, response: {result[:200]}")
                return self._fallback_faithfulness_parse(result)
                    
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return {"score": 0.0, "is_faithful": False, "reason": f"Evaluation error: {e}"}
    
    def _fallback_faithfulness_parse(self, result: str) -> Dict[str, Any]:
        """
        Improved fallback parsing for faithfulness evaluation.
        """
        result_lower = result.lower()
        
        # Check for explicit indicators
        if "faithful" in result_lower:
            if "not faithful" in result_lower or "unfaithful" in result_lower:
                return {"score": 0.2, "is_faithful": False, "reason": "Deemed unfaithful by LLM", "hallucinated_parts": []}
            elif "partially faithful" in result_lower:
                return {"score": 0.5, "is_faithful": False, "reason": "Partially faithful, some issues", "hallucinated_parts": []}
            elif "completely faithful" in result_lower or "fully faithful" in result_lower:
                return {"score": 0.95, "is_faithful": True, "reason": "Completely faithful", "hallucinated_parts": []}
            else:
                # Check for hallucination indicators
                if "hallucination" in result_lower or "not supported" in result_lower:
                    return {"score": 0.3, "is_faithful": False, "reason": "Hallucinations or unsupported claims detected", "hallucinated_parts": []}
                else:
                    return {"score": 0.7, "is_faithful": True, "reason": "Generally faithful", "hallucinated_parts": []}
        else:
            return {"score": 0.5, "is_faithful": False, "reason": "Unclear evaluation", "hallucinated_parts": []}
    
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
        
        # ✅ IMPROVED PROMPT: Clearer relevance evaluation
        prompt = ChatPromptTemplate.from_template("""
You are a strict evaluator for RAG systems. Determine if the answer is relevant to the query.

### QUERY:
{query}

### ANSWER:
{answer}

### TASK:
Evaluate if the ANSWER directly addresses and responds to the QUERY.
- A relevant answer directly answers the question asked
- An irrelevant answer goes off-topic or fails to address the question
- Be strict: partial relevance should be scored accordingly

### SCORING:
- 0.9-1.0: Directly and completely answers the query
- 0.7-0.89: Mostly answers with minor gaps
- 0.4-0.69: Partially answers or tangential
- 0.0-0.39: Doesn't answer the query

### OUTPUT FORMAT:
You MUST respond with ONLY a valid JSON object.
Example: {{"score": 0.95, "is_relevant": true, "reason": "Directly answers the question"}}

### YOUR JSON RESPONSE:
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
            
            try:
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = result[json_start:json_end]
                    data = json.loads(json_str)
                    score = float(data.get("score", 0.0))
                    score = max(0.0, min(1.0, score))
                    return {
                        "score": score,
                        "is_relevant": data.get("is_relevant", score > 0.5),
                        "reason": data.get("reason", "")
                    }
            except:
                pass
            
            # Fallback
            result_lower = result.lower()
            if "relevant" in result_lower:
                if "not relevant" in result_lower or "irrelevant" in result_lower:
                    return {"score": 0.2, "is_relevant": False, "reason": "Deemed irrelevant by LLM"}
                elif "partially relevant" in result_lower:
                    return {"score": 0.5, "is_relevant": False, "reason": "Partially relevant"}
                else:
                    return {"score": 0.8, "is_relevant": True, "reason": "Deemed relevant by LLM"}
            else:
                return {"score": 0.5, "is_relevant": False, "reason": "Unclear evaluation"}
                    
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
        
        # ✅ IMPROVED PROMPT: Better retrieval quality evaluation
        prompt = ChatPromptTemplate.from_template("""
You are a strict evaluator for RAG retrieval. Determine if the retrieved chunks are relevant to the query.

### QUERY:
{query}

### TOP RETRIEVED CHUNKS:
{chunks}

### TASK:
Evaluate if the retrieved chunks contain information relevant to answering the query.
- Good retrieval means most chunks are on-topic and useful
- Poor retrieval means chunks are off-topic or irrelevant

### SCORING:
- 0.9-1.0: All chunks are highly relevant
- 0.7-0.89: Most chunks are relevant
- 0.4-0.69: Some relevant, some not
- 0.0-0.39: Mostly irrelevant chunks

### OUTPUT FORMAT:
You MUST respond with ONLY a valid JSON object.
Example: {{"score": 0.85, "is_good": true, "reason": "Most chunks are relevant", "relevant_count": 4}}

### YOUR JSON RESPONSE:
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
            
            try:
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = result[json_start:json_end]
                    data = json.loads(json_str)
                    score = float(data.get("score", 0.0))
                    score = max(0.0, min(1.0, score))
                    return {
                        "score": score,
                        "is_good": data.get("is_good", score > 0.5),
                        "reason": data.get("reason", ""),
                        "relevant_count": data.get("relevant_count", 0)
                    }
            except:
                pass
            
            # Fallback
            result_lower = result.lower()
            if "good" in result_lower:
                if "not good" in result_lower or "poor" in result_lower:
                    return {"score": 0.3, "is_good": False, "reason": "Retrieval quality poor"}
                else:
                    return {"score": 0.8, "is_good": True, "reason": "Retrieval quality good"}
            else:
                return {"score": 0.5, "is_good": False, "reason": "Unclear evaluation"}
                    
        except Exception as e:
            logger.error(f"Retrieval quality evaluation failed: {e}")
            return {"score": 0.0, "is_good": False, "reason": f"Evaluation error: {e}"}