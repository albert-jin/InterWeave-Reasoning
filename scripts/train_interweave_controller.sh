#!/bin/bash
set -euo pipefail

# Configure these paths before running.
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
MODEL_ID_SCOPE="${MODEL_ID_SCOPE:-}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
MEM_FRAC="${MEM_FRAC:-0.8}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"

TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-datasets/train_gsm8k.json}"
SAVE_DIR="${SAVE_DIR:-ppo_checkpoints/interweave_gsm8k_$(date +%Y%m%d_%H%M%S)}"

python -u train_interweave_controller.py \
  --model_name "$MODEL_PATH" \
  --model_id_scope "$MODEL_ID_SCOPE" \
  --num_gpus "$NUM_GPUS" \
  --max_running_requests "$MAX_RUNNING_REQUESTS" \
  --mem_fraction_static "$MEM_FRAC" \
  --log_level info \
  --disable_overlap_schedule \
  --enable_soft_thinking \
  --max_topk 20 \
  --train_dataset train_gsm8k \
  --dataset_path "$TRAIN_DATASET_PATH" \
  --num_steps "${NUM_STEPS:-1000}" \
  --wrong_question_prob "${WRONG_QUESTION_PROB:-0.7}" \
  --batch_size "${TRAIN_BATCH_SIZE:-16}" \
  --save_dir "$SAVE_DIR" \
  --save_interval "${SAVE_INTERVAL:-50}" \
  --api_key "$OPENAI_API_KEY" \
  --max_generated_tokens "${MAX_GENERATED_TOKENS:-1024}" \
  --temperature "${TEMPERATURE:-0.6}" \
  --top_p "${TOP_P:-0.95}" \
  --think_end_str "</think>"
