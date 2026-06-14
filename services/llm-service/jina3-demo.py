from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jinaai/jina-embeddings-v3",
    local_dir="./jina-embeddings-v3",
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"]
)