#!/bin/bash
# Launch BM25 retrieval server over the BrowseComp-Plus corpus (100,195 docs).
# Usage:
#   bash scripts/deploy_retriever/retrieval_launch_browsecomp_plus.sh [port]
PORT=${1:-3000}

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=""

index_file=data/browsecomp-plus/indexes/bm25
corpus_file=data/browsecomp-plus/corpus.jsonl
retriever_name=bm25

python s3/search/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 12 \
    --retriever_name $retriever_name \
    --port $PORT
