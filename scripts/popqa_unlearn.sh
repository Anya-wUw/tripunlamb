# #!/bin/bash


# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# echo "Master Port: $MASTER_PORT"

# models=(
#     "Llama-3.2-1B-Instruct"
#     # "Llama-3.2-3B-Instruct"
#     # "Llama-3.1-8B-Instruct"
# )
# trainers_experiments=(
#     "GradAscent unlearn/popqa/default.yaml"
#     "GradDiff unlearn/popqa/default.yaml"
#     "NPO unlearn/popqa/default.yaml"
#     # "DPO unlearn/popqa/idk.yaml"
#     "RMU unlearn/popqa/default.yaml"
# )
# splits=(
#     "rare_forget10 rare_retain90 rare_retain90"
#     "popular_forget10 popular_retain90 popular_retain90"
# )


# per_device_train_batch_size=4 # on two gpus would make effective batch size 32
# gradient_accumulation_steps=4


# ########################################################################################################################
# ########################################### Unlearn TOFU models ########################################################
# ########################################################################################################################


# for split in "${splits[@]}"; do
#     forget_split=$(echo $split | cut -d' ' -f1)
#     # retain_split=$(echo $split | cut -d' ' -f2)
#     holdout_split=$(echo $split | cut -d' ' -f2)
#     retain_split=$(echo $split | cut -d' ' -f3)

#     for model in "${models[@]}"; do
#         for trainer_experiment in "${trainers_experiments[@]}"; do
#             trainer=$(echo $trainer_experiment | cut -d' ' -f1)
#             experiment=$(echo $trainer_experiment | cut -d' ' -f2)
            
#             task_name=popqa_${model}_${forget_split}_${trainer} 
#             model_path=/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.2-1B_finetune_singleAnsw #open-unlearning/popqa_${model}_full
#             echo ${task_name}: Unlearning ${model_path} using ${trainer}

#             # Unlearn
#             # CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#             CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --mixed_precision=bf16 \
#             src/train.py --config-name=unlearn.yaml \
#             experiment=${experiment} \
#             trainer=${trainer} \
#             task_name=${task_name} \
#             model=${model} \
#             forget_split=${forget_split} \
#             retain_split=${retain_split} \
#             model.model_args.pretrained_model_name_or_path=${model_path} \
#             retain_logs_path=saves/eval/popqa_${model}_${retain_split}/PopQA_EVAL.json \
#             trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
#             trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
#             trainer.args.ddp_find_unused_parameters=true \
#             trainer.args.gradient_checkpointing=true

#             # Eval
#             CUDA_VISIBLE_DEVICES=0 python src/eval.py \
#             experiment=eval/popqa/default.yaml \
#             task_name=${task_name} \
#             forget_split=${forget_split} \
#             retain_split=${retain_split} \
#             model=${model} \
#             model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
#             paths.output_dir=saves/unlearn/${task_name}/evals \
#             retain_logs_path=saves/eval/popqa_${model}_${retain_split}/PopQA_EVAL.json
#         done
#     done
# done

#!/bin/bash
set -euo pipefail

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

models=( "Llama-3.1-8B-Instruct" ) #"Llama-3.1-8B-Instruct"
trainers_experiments=(
    "GradAscent unlearn/popqa/default.yaml"
    "GradDiff  unlearn/popqa/default.yaml"
    "NPO       unlearn/popqa/default.yaml"
    "RMU       unlearn/popqa/default.yaml"
)
splits=(
    "rare_forget10 retain_intersection90 retain_intersection90"
    "popular_forget10 retain_intersection90 retain_intersection90"
)

for split in "${splits[@]}"; do
  read -r forget_split holdout_split retain_split <<< "$split"
  for model in "${models[@]}"; do
    for te in "${trainers_experiments[@]}"; do
      read -r trainer experiment <<< "$te"
      task_name=popqa_${model}_${forget_split}_${trainer}
      model_path=/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_2ep_ft_popqa

      echo "=== TRAIN  $task_name ==="

      # if [[ "$task_name" == "popqa_Llama-3.1-8B-Instruct_rare_forget10_GradAscent" || \
      # "$task_name" == "popqa_Llama-3.1-8B-Instruct_rare_forget10_GradDiff" ]]; then
      #   echo ">>> Skipping already computed $task_name"
      #   continue
      # fi

      if ! CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --mixed_precision=bf16 \
          src/train.py --config-name=unlearn.yaml \
            experiment=${experiment} trainer=${trainer} task_name=${task_name} \
            model=${model} forget_split=${forget_split} retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            retain_logs_path=saves/eval/popqa_${model}_${retain_split}/PopQA_EVAL.json \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=4 \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true
      then
        echo "!!! TRAIN FAILED for $task_name, skipping eval."
        continue
      fi

      echo "=== EVAL   $task_name ==="
      CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/popqa/default.yaml task_name=${task_name} \
        forget_split=${forget_split}\
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
        paths.output_dir=saves/unlearn/${task_name}/evals \
        retain_logs_path=saves/eval/popqa_${model}_${retain_split}/PopQA_EVAL.json \
      || echo "!!! EVAL FAILED for $task_name"
    done
  done
done
