#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/model}"
PPO_CHECKPOINT_PATH="${PPO_CHECKPOINT_PATH:-/path/to/best.pth}"
DATASET="${DATASET:-math500}"

python -u eval_interweave_controller.py \
  --dataset "$DATASET" \
  --model_name "$MODEL_PATH" \
  --num_gpus "${NUM_GPUS:-8}" \
  --max_running_requests "${MAX_RUNNING_REQUESTS:-128}" \
  --mem_fraction_static "${MEM_FRAC:-0.5}" \
  --ppo_agent_checkpoint_path "$PPO_CHECKPOINT_PATH" \
  --force_mode "${FORCE_MODE:-ppo}" \
  --enable_soft_thinking \
  --max_topk "${MAX_TOPK:-20}" \
  --max_generated_tokens "${MAX_GENERATED_TOKENS:-32768}" \
  --temperature "${TEMPERATURE:-0.6}" \
  --top_p "${TOP_P:-0.95}" \
  --top_k "${TOP_K:-30}" \
  --output_dir "${OUTPUT_DIR:-results}"
