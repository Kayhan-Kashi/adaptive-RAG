from langchain_core.embeddings import Embeddings
import numpy as np
import logging

logger = logging.getLogger(__name__)

class JinaLangChainWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents"""
        try:
            # Ensure texts is a list
            if not isinstance(texts, list):
                texts = [texts]
            
            # Some Jina models need the task parameter
            if hasattr(self.model, 'encode'):
                # Try with task parameter for Jina v3
                try:
                    embeddings = self.model.encode(
                        texts, 
                        task="text-matching"  # or "retrieval.passage"
                    )
                except TypeError:
                    # Fallback without task parameter
                    embeddings = self.model.encode(texts)
                
                # Convert to list of lists
                if hasattr(embeddings, 'tolist'):
                    return embeddings.tolist()
                elif isinstance(embeddings, np.ndarray):
                    return embeddings.tolist()
                else:
                    return [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
            else:
                # If model doesn't have encode method, try direct call
                raise AttributeError("Model has no 'encode' method")
                
        except Exception as e:
            logger.error(f"Error in embed_documents: {e}")
            logger.error(f"Texts type: {type(texts)}, length: {len(texts) if hasattr(texts, '__len__') else 'N/A'}")
            raise

    def embed_query(self, text: str) -> list[float]:
        """Embed a single string for search queries"""
        try:
            # Ensure text is string
            if not isinstance(text, str):
                text = str(text)
            
            # Some Jina models need the task parameter
            if hasattr(self.model, 'encode'):
                # Try with task parameter for Jina v3
                try:
                    embeddings = self.model.encode(
                        [text],  # Pass as list
                        task="text-matching"  # or "retrieval.query"
                    )
                except TypeError:
                    # Fallback without task parameter
                    embeddings = self.model.encode([text])
                
                # Extract first embedding and convert to list
                if hasattr(embeddings, 'tolist'):
                    return embeddings[0].tolist()
                elif isinstance(embeddings, np.ndarray):
                    return embeddings[0].tolist()
                elif isinstance(embeddings, list):
                    if hasattr(embeddings[0], 'tolist'):
                        return embeddings[0].tolist()
                    return embeddings[0]
                else:
                    return embeddings[0]
            else:
                # If model doesn't have encode method, try direct call
                raise AttributeError("Model has no 'encode' method")
                
        except Exception as e:
            logger.error(f"Error in embed_query: {e}")
            logger.error(f"Text type: {type(text)}")
            raise