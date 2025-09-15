#!/bin/bash
set -euo pipefail

# свободный порт для DDP
export MASTER_PORT=$(python - <<'PY'
import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)

models=("Llama-3.1-8B-Instruct")
trainers_experiments=(
  "GradAscent  unlearn/tripunlamb/default.yaml"
  "GradDiff    unlearn/tripunlamb/default.yaml"
  "NPO         unlearn/tripunlamb/default.yaml"
  "RMU         unlearn/tripunlamb/default.yaml"
)

# сколько забываем (p%)
split_="10"
learning_rates=("1e-5" "5e-5" "2e-4")

for prefix in "rare_forget_" "popular_forget_"; do
  forget_split="${prefix}${split_}"
  retain_split="retain_$((100 - 2 * split_))"

  for model in "${models[@]}"; do
    model_path="meta-llama/Llama-3.1-8B-Instruct"

    for te in "${trainers_experiments[@]}"; do
      read -r trainer experiment <<< "$te"

      for lr in "${learning_rates[@]}"; do
        task_name="tripunlamb_${model}_${forget_split}_${trainer}_lr_${lr}"
        echo "=== UNLEARN  ${task_name} ==="

        log_dir="tripunlamb_lr/tripunlamb_${model}_${retain_split}"
        mkdir -p "$log_dir"

        # Собираем Hydra overrides в массив — без завершающего слеша
        overrides=(
          "--config-name=unlearn.yaml"
          "experiment=${experiment}"
          "trainer=${trainer}"
          "task_name=${task_name}"
          "model=${model}"
          "forget_split=${forget_split}"
          "retain_split=${retain_split}"
          "model.model_args.pretrained_model_name_or_path=${model_path}"
          "retain_logs_path=${log_dir}/tripunlamb_EVAL.json"
          "trainer.args.learning_rate=${lr}"
          "trainer.args.per_device_train_batch_size=4"
          "trainer.args.gradient_accumulation_steps=4"
          "trainer.args.ddp_find_unused_parameters=true"
          "trainer.args.gradient_checkpointing=true"
        )

        CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --mixed_precision=bf16 \
          src/train.py "${overrides[@]}" || {
            echo "!!! TRAIN FAILED for ${task_name}, skipping."
            continue
          }
      done
    done
  done
done
