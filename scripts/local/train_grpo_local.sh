#!/bin/bash
# Local single-GPU (GTX 1070 Ti, 8GB, Pascal) run of s3 agentic-search RL.
# Changes vs scripts/train/train_s3.sh:
#   - Qwen2.5-0.5B-Instruct instead of 7B
#   - HF rollout instead of vLLM (vLLM cannot run on Pascal sm_61)
#   - GRPO instead of GAE+critic (halves VRAM: no critic model)
#   - fp32 everywhere (Pascal has no native bf16)
#   - tiny local dataset + tiny faiss retriever
set -x

DATA_DIR=data/local
BASE_MODEL=${BASE_MODEL:-models/Qwen2.5-0.5B-Instruct}
EXPERIMENT_NAME=local_grpo_$(date +%m%d_%H%M%S)
mkdir -p train_logs verl_checkpoints

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
# 15GB RAM box: shrink ray object store (default would be ~3.6GB)
export RAY_object_store_memory=1073741824
# 15GB box: allow ray up to 98% before killing workers
export RAY_memory_usage_threshold=0.98
PYTHON=${PYTHON:-$HOME/miniconda3/envs/s3_rl/bin/python}
# no proxy: this box uses a dead 127.0.0.1:7877 proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

export PATH=$(dirname $PYTHON):$PATH

$PYTHON -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.max_start_length=512 \
    data.max_obs_length=512 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_penalty=none \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    +actor_rollout_ref.actor.optim.optimizer=sgd \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1000 \
    trainer.project_name=local_s3 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    +data.random_seed=42 \
    max_turns=3 \
    +generator_llm="local-rule" \
    +output_context_dir="data/output_sequences_local" \
    +trainer.val_before_train=false \
    retriever.url="http://127.0.0.1:3000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee train_logs/$EXPERIMENT_NAME.log
