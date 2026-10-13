# Launch the retrieval server with the local bge-m3 embedding API (vLLM, port 8091).
# Machine: aarch64 + Ascend NPU -> CPU faiss + remote embeddings, no local GPU encoder.
#
# Prerequisites:
#   1. conda env:  /data/conda_envs/agentic_s3
#   2. embedding service running: curl http://127.0.0.1:8091/v1/models  (bge-m3)
#   3. corpus + index under $save_path (see below how to build the index)
#
# Build a bge-m3 dense index from a corpus (e.g. wiki-18.jsonl) first:
#   /data/conda_envs/agentic_s3/bin/python s3/search/index_builder.py \
#       --retrieval_method bge-m3 \
#       --corpus_path $save_path/wiki-18.jsonl \
#       --save_dir $save_path \
#       --max_length 256 --batch_size 64 --faiss_type Flat --save_embedding \
#       --embedding_api_url http://127.0.0.1:8091/v1/embeddings \
#       --embedding_model_name bge-m3
#   -> creates $save_path/bge-m3_Flat.index
#
# Or build a BM25 index (no embeddings needed):
#   /data/conda_envs/agentic_s3/bin/python s3/search/index_builder.py \
#       --retrieval_method bm25 --corpus_path $save_path/wiki-18.jsonl --save_dir $save_path

export CUDA_VISIBLE_DEVICES=""

save_path=${SAVE_PATH:-/data/yf/search_data}

index_file=$save_path/bge-m3_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=bge-m3
embedding_api_url=http://127.0.0.1:8091/v1/embeddings
embedding_model_name=bge-m3

/data/conda_envs/agentic_s3/bin/python s3/search/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 12 \
    --retriever_name $retriever_name \
    --embedding_api_url $embedding_api_url \
    --embedding_model_name $embedding_model_name \
    --port 3000
