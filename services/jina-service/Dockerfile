# services/jina-service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

ARG PIP_INDEX_URL=https://mirror-pypi.runflare.com/simple

RUN pip install -vvv torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -vvv transformers==4.44.2 --index-url ${PIP_INDEX_URL}
RUN pip install -vvv sentence-transformers==3.2.0 --index-url ${PIP_INDEX_URL}
RUN pip install -vvv langchain-huggingface==1.2.2 --index-url ${PIP_INDEX_URL}
RUN pip install -vvv einops==0.8.2 --index-url ${PIP_INDEX_URL}
RUN pip install -vvv faiss-cpu==1.14.2 --index-url ${PIP_INDEX_URL}
RUN pip install huggingface-hub --index-url ${PIP_INDEX_URL}

ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0
ENV HF_DATASETS_OFFLINE=0
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1


# Copy LLM service - note: this copies contents of src/ to /app/src/
COPY services/jina-service/ ./
# COPY services/llm-service/models/jina-embeddings-v3 /app/models/jina-embeddings-v3

ENV PYTHONPATH=/app

CMD ["python", "src/jina-demo.py"]