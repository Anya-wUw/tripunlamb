# #!/bin/bash
# set -euo pipefail

# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# models=( "gemma-7b-it" ) #"Llama-3.1-8B-Instruct"
# trainers_experiments=(
#     "GradAscent unlearn/popqa/default.yaml"
#     "GradDiff  unlearn/popqa/default.yaml"
#     "NPO       unlearn/popqa/default.yaml"
#     "RMU       unlearn/popqa/default.yaml"
# )

# split_="10"
# splits=(
#     "rare_forget${split_} retain_intersection80 retain_intersection80" # CALC
#     "popular_forget${split_} retain_intersection80 retain_intersection80"
#     # "rare_forget5 retain_intersection90 retain_intersection90"
#     # "popular_forget5 retain_intersection90 retain_intersection90"
# )

# for split in "${splits[@]}"; do
#   read -r forget_split holdout_split retain_split <<< "$split"
#   for model in "${models[@]}"; do
#     for te in "${trainers_experiments[@]}"; do
#       read -r trainer experiment <<< "$te"
#       task_name=popqa_${model}_${forget_split}_${trainer}
#       model_path=/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/gemma-7b-it_full_5ep_ft_popqa

#       echo "=== UNLEARN  $task_name ==="


#       if ! CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --mixed_precision=bf16 \
#           src/train.py --config-name=unlearn.yaml \
#             experiment=${experiment} trainer=${trainer} task_name=${task_name} \
#             model=${model} forget_split=${forget_split} retain_split=${retain_split} \
#             model.model_args.pretrained_model_name_or_path=${model_path} \
#             retain_logs_path=saves/eval/forget_${split_}/popqa_${model}_${retain_split}/PopQA_EVAL.json \
#             trainer.args.per_device_train_batch_size=4 \
#             trainer.args.gradient_accumulation_steps=4 \
#             trainer.args.ddp_find_unused_parameters=true \
#             trainer.args.gradient_checkpointing=true
#       then
#         echo "!!! TRAIN FAILED for $task_name, skipping eval."
#         continue
#       fi
#     done
#   done
# done


#!/bin/bash
set -euo pipefail

# Подбираем случайный свободный порт для распределённого обучения
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# Модели и эксперименты
models=( "Llama-3.1-8B-Instruct" )  # при желании можно добавить ещё; "gemma-7b-it" "Phi-3.5-mini-instruct" 
trainers_experiments=(
    "GradAscent unlearn/popqa/default.yaml"
    "GradDiff    unlearn/popqa/default.yaml"
    "NPO         unlearn/popqa/default.yaml"
    "RMU         unlearn/popqa/default.yaml"
)

# Здесь вручную задаёте величину forget, например 5, 10, 15 и т.д.
split_="15"

for prefix in "rare_forget" "popular_forget"; do
  # Формируем имя сплита, который нужно забыть
  forget_split="${prefix}${split_}"
  # Вычисляем retain_intersection = 100 − 2*split_
  inter=$((100 - 2 * split_))
  retain_split="retain_intersection${inter}"

  for model in "${models[@]}"; do
    for te in "${trainers_experiments[@]}"; do
      read -r trainer experiment <<< "$te"
      task_name=popqa_${model}_${forget_split}_${trainer}
      model_path=/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_5ep_ft_popqa #/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.2-1b_full_5ep_ft_popqa #/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/Phi-3.5-mini-instruct_full_5ep_ft_popqa

      echo "=== UNLEARN  $task_name ==="

      if ! CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --mixed_precision=bf16 \
          src/train.py --config-name=unlearn.yaml \
            experiment=${experiment} \
            trainer=${trainer} \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \

            model.model_args.pretrained_model_name_or_path=${model_path} \
            retain_logs_path=saves/eval/forget_${split_}/popqa_${model}_${retain_split}/PopQA_EVAL.json \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4 \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true
      then
        echo "!!! TRAIN FAILED for $task_name, skipping eval."
        continue
      fi
    done
  done
done
