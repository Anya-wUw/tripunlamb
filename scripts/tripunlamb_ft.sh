#!/usr/bin/env bash
# model finetuning with conditional overrides for epochs and LoRA hyperparams

set -euo pipefail

MODELS=(
  # "Llama-3.2-1B-Instruct"
  # "Llama-3.2-3B-Instruct"
  "Llama-3.1-8B-Instruct"
)
TASK_NAMES=(
  "llama3.1-8b_full_3ep_ft_tripunlamb"
)

GPU_ID=0

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  TASK="${TASK_NAMES[$i]}"

  EPOCHS=3
  LORA_R=32
  LORA_ALPHA=64

  echo "=== Finetuning $MODEL ==="
  echo "   task_name: $TASK"
  echo "   epochs: $EPOCHS, lora_r: $LORA_R, lora_alpha: $LORA_ALPHA"
  CUDA_VISIBLE_DEVICES=$GPU_ID \
    python src/train.py \
      --config-name=train.yaml \
      experiment=finetune/tripunlamb/default \
      model="$MODEL" \
      task_name="$TASK" \
      trainer.args.num_train_epochs="$EPOCHS" \
      peft.lora.r="$LORA_R" \
      peft.lora.alpha="$LORA_ALPHA"

  echo "=== Done $MODEL ==="
  echo
done
