#!/usr/bin/env bash
# model finetuning with conditional overrides for epochs and LoRA hyperparams

set -euo pipefail

MODELS=(
  # "Llama-3.2-1B-Instruct"
  # "Llama-3.2-3B-Instruct"
  # "Llama-3.1-8B-Instruct"
  # "gemma-7b-it"
  # "Phi-3.5-mini-instruct"
)
TASK_NAMES=(
  # "llama3.2-1b_full_5ep_ft_popqa"
  # "llama3.2-3b_full_5ep_ft_popqa"
  # "llama3.1-8b_full_1ep_ft_popqa"
  # "gemma-7b-it_full_5ep_ft_popqa"
  # "Phi-3.5-mini-instruct_full_5ep_ft_popqa"
)

GPU_ID=1

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  TASK="${TASK_NAMES[$i]}"

  EPOCHS=5
  LORA_R=32
  LORA_ALPHA=64

  echo "=== Finetuning $MODEL ==="
  echo "   task_name: $TASK"
  echo "   epochs: $EPOCHS, lora_r: $LORA_R, lora_alpha: $LORA_ALPHA"
  CUDA_VISIBLE_DEVICES=$GPU_ID \
    python src/train.py \
      --config-name=train.yaml \
      experiment=finetune/popqa/default \
      model="$MODEL" \
      task_name="$TASK" \
      trainer.args.num_train_epochs="$EPOCHS" \
      peft.lora.r="$LORA_R" \
      peft.lora.alpha="$LORA_ALPHA"

  echo "=== Done $MODEL ==="
  echo
done
