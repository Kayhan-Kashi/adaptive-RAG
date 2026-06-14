import os
import torch
from transformers import AutoModel
from huggingface_hub import snapshot_download

# --- SECTION 1: SNAPSHOT LOGIC ---
# MODEL_DIR = "../../../models/snapshot/jina-embeddings-v3"
MODEL_DIR = "/app/models/snapshot/jina-embeddings-v3"
REPO_ID = "jinaai/jina-embeddings-v3"

is_offline = os.environ.get("TRANSFORMERS_OFFLINE", "1") == "1"

# Ensure the folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if model files are present; if not, download them.
if not os.listdir(MODEL_DIR):
    print(f"❌ Model not found in {MODEL_DIR}. Starting download...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=[
            "*.msgpack",      # Flax
            "flax_model*",    # Flax
            "tf_model*",      # TensorFlow
            "*.onnx",         # ONNX
            "onnx/*"          # ONNX
        ]
    )
    print("✅ Download finished.")
else:
    print(f"✅ Snapshot confirmed at {MODEL_DIR}. Using existing files.")


# --- SECTION 2: LOAD AND USE THE SNAPSHOT ---
print("Loading model into memory...")
model = AutoModel.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=is_offline
)
print("Model loaded successfully!")

texts = [
    "Follow the white rabbit.",
    "Sigue al conejo blanco.",
]

# Encode using the model's encode method
print("Encoding texts...")
embeddings = model.encode(texts, task="text-matching")

# Print each sentence with its embedding
print("\n" + "=" * 60)
for i, embedding in enumerate(embeddings):
    print(f"\nSentence {i}: {texts[i]}")
    print(f"Embedding shape: {embedding.shape}")
    # Convert numpy array to list for printing first 10 values
    print(f"Embedding (first 10 values): {embedding[:10].tolist()}")
    print("-" * 40)

# Calculate similarity
# Ensure embeddings are tensors
emb_tensor = torch.tensor(embeddings)
similarity = emb_tensor[0] @ emb_tensor[1].T

print(f"\n📊 Similarity between sentences:")
print(f"   '{texts[0]}'")
print(f"   '{texts[1]}'")
print(f"   Score: {similarity.item():.6f}")
print("=" * 60)
