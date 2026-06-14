# jina-demo-simple.py
import os
import torch
from transformers import AutoModel

# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Load model
model = AutoModel.from_pretrained(
    "../../models/jina-embeddings-v3",
    trust_remote_code=True,
    local_files_only=True
)

texts = [
    "Follow the white rabbit.",
    "Sigue al conejo blanco.",
]

# Encode using the model's encode method
embeddings = model.encode(texts, task="text-matching")

# Print each sentence with its embedding
print("\n" + "=" * 60)
for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    print(f"\nSentence {i}: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 values): {embedding[:10].tolist()}")
    print("-" * 40)

# Calculate similarity
similarity = embeddings[0] @ embeddings[1].T

print(f"\n📊 Similarity between sentences:")
print(f"   '{texts[0]}'")
print(f"   '{texts[1]}'")
print(f"   Score: {similarity.item():.6f}")
print("=" * 60)