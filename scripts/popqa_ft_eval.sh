#!/usr/bin/env bash
# batch eval on multiple forget_splits for PopQA (with logging to file)

set -euo pipefail

# If you need a random free port (for distributed eval), uncomment these two lines:
export MASTER_PORT=$(python - <<<'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Master port: $MASTER_PORT"

# Path to your finetuned model checkpoint
PRETRAINED_MODEL_PATH='/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_5ep_ft_popqa'  #'meta-llama/Llama-3.1-8B'
# Base output directory for all eval runs
OUTPUT_BASE="saves/eval/popqa/llama3.1-8b_ft_5ep" #llama3.2-1b_ft_5ep" orig-llama3.1-8b

# List of splits to evaluate (will be passed as forget_split)
SPLITS=(
  "rare_forget10"
  "popular_forget10"
  "duplicate_rare_forget10"
  "duplicate_popular_forget10"
  # "rare_retain90"
  # "popular_retain90"
  "retain_intersection90"
)

# Make sure output base exists
mkdir -p "${OUTPUT_BASE}"

for split in "${SPLITS[@]}"; do
  echo "=== Evaluating split: $split ==="

  # Prepare output directory for this split
  SPLIT_DIR="${OUTPUT_BASE}/${split}"
  mkdir -p "${SPLIT_DIR}"

  # Define task name
  TASK_NAME="popqa_llama3.1-8b_${split}"

  # Run evaluation, both to console and to log file
  CUDA_VISIBLE_DEVICES=1 python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/popqa/default \
    forget_split="${split}" \
    task_name="${TASK_NAME}" \
    paths.output_dir="${SPLIT_DIR}" \
    model.model_args.pretrained_model_name_or_path="${PRETRAINED_MODEL_PATH}" \
    2>&1 | tee "${SPLIT_DIR}/${TASK_NAME}_eval_results.txt"

  echo
done
