#!/usr/bin/env bash
# model finetuning with conditional overrides for epochs and LoRA hyperparams

set -euo pipefail

# Список моделей и соответствующих им task_name (все в lowercase)
MODELS=(
  # "Llama-3.1-8B-Instruct"
  "Llama-3.2-3B-Instruct"
  "Llama-3.2-1B-Instruct"
)
TASK_NAMES=(
  # "llama3.1-8b_full_3ep_ft_popqa"
  "llama3.2-3b_full_5ep_ft_popqa"
  "llama3.2-1b_full_5ep_ft_popqa"
)

# GPU, который мы используем
GPU_ID=0

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  TASK="${TASK_NAMES[$i]}"

  # По умолчанию: 3 эпохи, LoRA r=32, alpha=16
  EPOCHS=3
  LORA_R=32
  LORA_ALPHA=16

  # Для 2-й (index 1) и 3-й (index 2) моделей ставим 5 эпох и alpha=64
  if [[ "$i" -eq 0 || "$i" -eq 1 ]]; then
    EPOCHS=5
    LORA_ALPHA=64
  fi

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

# # model finetuning

# # export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# # echo "Master Port: $MASTER_PORT"

# # Список моделей и соответствующих им task_name (все в lowercase)
# MODELS=(
#   "Llama-3.1-8B-Instruct"
#   "Llama-3.2-3B-Instruct"
#   "Llama-3.2-1B-Instruct"
# )
# TASK_NAMES=(
#   "llama3.1-8b_full_3ep_ft_popqa"
#   "llama3.2-3b_full_3ep_ft_popqa"
#   "llama3.2-1b_full_3ep_ft_popqa"
# )

# # GPU, который мы используем
# GPU_ID=0

# for i in "${!MODELS[@]}"; do
#   MODEL="${MODELS[$i]}"
#   TASK="${TASK_NAMES[$i]}"

#   echo "=== Finetuning $MODEL (task: $TASK) ==="
#   CUDA_VISIBLE_DEVICES=$GPU_ID \
#   python src/train.py \
#     --config-name=train.yaml \
#     experiment=finetune/popqa/default \
#     model="$MODEL" \
#     task_name="$TASK"

#   echo "=== Done $MODEL ==="
#   echo
# done



# #!/bin/bash

# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# echo "Master Port: $MASTER_PORT"


# models=(
#     "Llama-3.2-1B-Instruct"
#     "Llama-3.2-3B-Instruct"
#     "Llama-3.1-8B-Instruct"
# )
# per_device_train_batch_size=4 # Effective batch size 32 on two GPUs with gradent_accumulation_steps=8

# splits=(
#     "rare_forget10 rare_retain90 rare_retain90"
#     "popular_forget10 popular_retain90 popular_retain90"
# )

# #TODO REWRITE TO POPQA

# ########################################################################################################################
# ########################################### RETAIN Finetuned TOFU ######################################################
# ########################################################################################################################

# for split in "${splits[@]}"; do
#     forget_split=$(echo $split | cut -d' ' -f1)
#     holdout_split=$(echo $split | cut -d' ' -f2)
#     retain_split=$(echo $split | cut -d' ' -f3)
    
#     for model in "${models[@]}"; do
#         CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#         src/train.py experiment=finetune/tofu/default.yaml \
#         task_name=tofu_${model}_${retain_split} \
#         model=${model} \
#         data/datasets@data.train=TOFU_QA_retain \
#         data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
#         trainer.args.per_device_train_batch_size=4 \
#         trainer.args.ddp_find_unused_parameters=true \
#         trainer.args.gradient_checkpointing=true

    
#         CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
#         forget_split=${forget_split} \
#         holdout_split=${holdout_split} \
#         task_name=tofu_${model}_${retain_split} \
#         model=${model} \
#         model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_${retain_split}
#     done
# done


# # ########################################################################################################################
# # ########################################### FULL Finetuned TOFU models #################################################
# # ########################################################################################################################


# for model in "${models[@]}"; do
#     CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#     src/train.py experiment=finetune/tofu/default.yaml \
#     task_name=tofu_${model}_full \
#     model=${model} \
#     data/datasets@data.train=TOFU_QA_full \
#     data.train.TOFU_QA_full.args.hf_args.name=full \
#     trainer.args.per_device_train_batch_size=4 \
#     trainer.args.ddp_find_unused_parameters=true \
#     trainer.args.gradient_checkpointing=true

#     # Evaluate the full models on each forget split
#     for split in "${splits[@]}"; do
#         forget_split=$(echo $split | cut -d' ' -f1)
#         holdout_split=$(echo $split | cut -d' ' -f2)
#         retain_split=$(echo $split | cut -d' ' -f3)

#         CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
#         forget_split=${forget_split} \
#         holdout_split=${holdout_split} \
#         task_name=tofu_${model}_full_${forget_split} \
#         model=${model} \
#         model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_full \
#         retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
#         paths.output_dir=saves/eval/tofu_${model}_full/evals_${forget_split}
#     done
# done
