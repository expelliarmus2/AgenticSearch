#!/bin/bash
# Launch the local faiss retriever (CPU, no pyserini/Java) on port 3000.
cd "$(dirname "$0")/../.." || exit 1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python s3/search/retrieval_server.py \
    --index_path data/local/e5_Flat.index \
    --corpus_path data/local/corpus.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model $(pwd)/models/e5-small-v2 \
    --embedding_api_url none \
    --port 3000
