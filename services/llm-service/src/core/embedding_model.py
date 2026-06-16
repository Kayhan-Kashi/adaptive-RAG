import os
import logging
from injector import inject
from transformers import AutoModel

logger = logging.getLogger(__name__)

class EmbeddingModel:
    @inject
    def __init__(self):
        # 1. Config setup
        self.model_path = os.getenv("MODEL_PATH", "/app/models/snapshot/jina-embeddings-v3")
        
        # We check the environment to set the offline flag
        # If TRANSFORMERS_OFFLINE=1, local_files_only=True
        self.is_offline = os.getenv("TRANSFORMERS_OFFLINE", "1") == "1"
        
        # 2. Validation
        self._validate_model()
        
        # 3. Load model
        self._model = self._load_model()

    def _validate_model(self):
        """Ensures the directory and config exist before attempting load."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model path does not exist: {self.model_path}")
            
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ 'config.json' not found in {self.model_path}. Is the directory empty?")

    def _load_model(self):
        """Loads the model from disk."""
        logger.info(f"📦 Loading Jina v3 from {self.model_path} (Offline mode: {self.is_offline})...")
        try:
            return AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=self.is_offline
            )
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise e

    @property
    def model(self):
        return self._model

    def embed_texts(self, texts: list[str], task: str = "text-matching"):
        return self._model.encode(texts, task=task)
