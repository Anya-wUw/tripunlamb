#!/usr/bin/env bash

set -euo pipefail

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

export MASTER_PORT=$(python - <<<'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Master port: $MASTER_PORT"

PRETRAINED_MODEL_PATH='meta-llama/Llama-3.1-8B' #'/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/Phi-3.5-mini-instruct_full_5ep_ft_popqa' #'meta-llama/Llama-3.2-1B'  #'meta-llama/Llama-3.1-8B'
OUTPUT_BASE="saves/eval/popqa/orig-llama3.1-8b" #Phi-3.5-mini-instruct_full_5ep_ft_popqa" #llama3.2-1b_ft_5ep" orig-llama3.1-8b

SPLITS=(
  "rare_forget15"
  "popular_forget15"
  # "rare_forget10"
  # "popular_forget10"
  # "duplicate_answers_rare_forget10"
  # "duplicate_answers_popular_forget10"
  # "duplicate_subjects_rare_forget10"
  # "duplicate_subjects_popular_forget10"
  # "retain_intersection80"
)

mkdir -p "${OUTPUT_BASE}"

for split in "${SPLITS[@]}"; do
  echo "=== Evaluating split: $split ==="

  SPLIT_DIR="${OUTPUT_BASE}/${split}"
  mkdir -p "${SPLIT_DIR}"

  TASK_NAME="popqa_llama_3-1B_popqa_${split}"

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
