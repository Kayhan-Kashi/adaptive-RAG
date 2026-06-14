# BGE-local.py
import os
from FlagEmbedding import FlagReranker

# Define model and cache path
model_id = 'BAAI/bge-reranker-v2-m3'
cache_dir = '../../../models/BAAI'

# Check if model already exists in cache
model_path = os.path.join(cache_dir, 'models--BAAI--bge-reranker-v2-m3')
if os.path.exists(model_path):
    print(f"✅ Model found in cache. Loading existing model...")
else:
    print(f"📥 Model not found. Downloading {model_id} to {cache_dir}...")

# Use the model name directly - it will download automatically if not cached
reranker = FlagReranker(
    model_id,
    use_fp16=True,
    cache_dir=cache_dir
)

print("✅ Model loaded successfully!")

# Test it
query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Python is a programming language.",
    "Deep learning uses neural networks."
]

scores = reranker.compute_score([(query, doc) for doc in documents])

for doc, score in zip(documents, scores):
    print(f"Score: {score:.4f} - {doc}")  # Removed the extra 's' at the end