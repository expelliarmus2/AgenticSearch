# 本地单卡跑通 s3 (AgenticSearch) RL —— 改动说明

目标：在 **GTX 1070 Ti (8GB, Pascal sm_61)** 上把 s3 的 verl 训练循环跑通
（GRPO，无 vLLM / 无 flash-attn / 无外部 generator-LLM API / 无 pyserini）。

## 环境

```bash
conda create -y -n s3_rl python=3.10
# 本机需先 unset 失效代理：unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
pip install torch==2.4.0 numpy==1.26.4 transformers==4.47.1 tensordict==0.5.0 \
    ray==2.9.3 datasets==2.21.0 hydra-core==1.3.2 omegaconf accelerate \
    codetiming dill pandas==2.2.2 pyarrow tqdm sentencepiece protobuf \
    fastapi uvicorn pydantic faiss-cpu pybind11 matplotlib IPython pyyaml \
    requests einops wandb
# 模型走 HF 镜像
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
    --local-dir models/Qwen2.5-0.5B-Instruct
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download intfloat/e5-small-v2 \
    --local-dir models/e5-small-v2
```

## 运行步骤

```bash
# 1. 构造小数据（train/val parquet + 检索语料）
python scripts/local/make_local_data.py

# 2. 用本地 e5 建 faiss 索引（需 s3_rl 环境 + 模型已下载）
python scripts/local/build_local_index.py

# 3. 启动检索服务（CPU，端口 3000）
bash scripts/local/run_retriever.sh &

# 4. 训练（GRPO, 2 steps）
bash scripts/local/train_grpo_local.sh
```

## 代码改动清单

| 文件 | 改动 | 原因 |
|---|---|---|
| `verl/utils/reward_score/rag_local.py` | **新增**：纯规则奖励 | 原 `rag_2.py` 依赖外部 generator-LLM API + pyserini(Java) |
| `verl/trainer/main_ppo.py` | import 换成 `rag_local`；`_select_rm_score_fn` 返回 `compute_score_local` | 同上 |
| `verl/workers/rollout/hf_rollout.py` | autocast `bfloat16→float32` | Pascal 无 bf16 |
| `verl/workers/actor/dp_actor.py` | autocast `bfloat16→float32`；`torch.compile` 去掉（Triton 需 sm_70+） | Pascal 兼容 |
| `verl/workers/fsdp_workers.py` | 3 处 `attn_implementation='flash_attention_2'→'sdpa'`；param_dtype 默认 `bf16→float32`；模型加载 dtype 默认 `float32` | 无 flash-attn；Pascal 无 bf16 |

## 配置要点（`scripts/local/train_grpo_local.sh`）

- `actor_rollout_ref.rollout.name=hf`：HF rollout，替代 vLLM（vLLM 需 CC>=7.0）
- `algorithm.adv_estimator=grpo`：GRPO 不需要 critic 模型，显存减半
- `actor_rollout_ref.actor.fsdp_config.*_offload=true`：参数/梯度/优化器卸载到 CPU
- `model_dtype=float32`：Pascal 无 bf16
- `data.max_prompt_length=1024 / max_response_length=256 / max_turns=3`：缩小轨迹
- `actor_rollout_ref.rollout.n=2`：GRPO group size

## 说明

- 奖励为纯规则：答案 EM/F1 匹配得 generation_score，搜索/回答格式给 format_bonus
- 检索语料为每条问题一条"答案文档"（含 golden answer）+ 干扰文档，保证检索有正信号
- 目标只是**跑通训练循环**（rollout → reward → advantage → policy update），
  不是复现 s3 的 SOTA 效果
