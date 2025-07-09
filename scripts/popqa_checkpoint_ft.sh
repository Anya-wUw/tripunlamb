#!/usr/bin/env bash
# Resume LoRA finetuning from existing checkpoints up to TOTAL_EPOCHS.
# Usage: bash popqa_checkpoint_ft.sh

set -euo pipefail

GPU_ID=0

# Базовая папка с вашими finetune-чекпоинтами
CKPT_BASE="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune"

# Модели
MODELS=(
  "Llama-3.1-8B-Instruct"
  "Llama-3.2-3B-Instruct"
  "Llama-3.2-1B-Instruct"
)

# Названия папок с предыдущими 3-эпохными прогонками
PREV_TASKS=(
  "llama3.1-8b_full_3ep_ft_popqa"
  "llama3.2-3b_full_3ep_ft_popqa"
  "llama3.2-1b_full_3ep_ft_popqa"
)

# Новые названия задач (будем докатывать до 5 эпох)
NEW_TASKS=(
  "llama3.1-8b_full_5ep_ft_popqa"
  "llama3.2-3b_full_5ep_ft_popqa"
  "llama3.2-1b_full_5ep_ft_popqa"
)

# Сколько всего эпох в итоге хотим получить
TOTAL_EPOCHS=5

# Параметры LoRA
LORA_R=32
LORA_ALPHA=16


# ----------------------------
#   Цикл по всем моделям
# ----------------------------
for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  PREV="${PREV_TASKS[$i]}"
  NEW="${NEW_TASKS[$i]}"
  BASE_DIR="${CKPT_BASE}/${PREV}"

  # Проверяем, что папка есть
  if [ ! -d "$BASE_DIR" ]; then
    echo "Error: папка чекпоинтов не найдена: $BASE_DIR" >&2
    exit 1
  fi
у 
  # Находим самый свежий checkpoint-*
  CKPT_DIR=$(ls -d "$BASE_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n1)
  if [ -z "$CKPT_DIR" ]; then
    echo "Error: в $BASE_DIR нет подпапок checkpoint-*" >&2
    exit 1
  fi

  echo "=== Resuming ${MODEL} ==="
  echo "    From checkpoint: ${CKPT_DIR}"
  echo "    To new task:     ${NEW}"
  echo "    Total epochs:    ${TOTAL_EPOCHS}"
  echo "    LoRA r:          ${LORA_R}"
  echo "    LoRA alpha:      ${LORA_ALPHA}"
  echo

  CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python src/train.py \
      --config-name=train.yaml \
      experiment=finetune/popqa/default \
      model="${MODEL}" \
      task_name="${NEW}" \
      trainer.args.num_train_epochs="${TOTAL_EPOCHS}" \
      peft.lora.r="${LORA_R}" \
      peft.lora.alpha="${LORA_ALPHA}" \
      +trainer.resume_from_checkpoint="${CKPT_DIR}"

  echo "=== Done ${NEW} ==="
  echo
done
