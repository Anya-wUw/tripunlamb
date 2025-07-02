#!/bin/bash
set -euo pipefail

export MASTER_PORT=$(python - <<'PYCODE'
import socket
s = socket.socket(); s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PYCODE
)

model="Llama-3.2-3B-Instruct"
orig_model_path="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/Llama-3.2-3B-Instruct_finetune_SA
"

# DS Splits
declare -A forget_splits=( ["rare"]="rare_forget10" ["popular"]="popular_forget10" )
declare -A retain_splits=( ["rare"]="rare_retain90" ["popular"]="popular_retain90" )

# Unlearning algorithms
trainers=("RMU" ) # "GradAscent" "GradDiff" "NPO" 

for split_type in "popular"; do #"rare" "popular";
  forget_split=${forget_splits[$split_type]}
  retain_split=${retain_splits[$split_type]}

  out_root="saves/eval/eval_final_experements/${retain_split}"
  mkdir -p "${out_root}"

  # Eval Orig FT-model
  # task_name="popqa_${model}_${forget_split}_Original"
  # echo "=== EVAL RETAIN (ORIGINAL) $task_name on $retain_split ==="
  # CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 src/eval.py \
  #   --config-name=eval.yaml \
  #   experiment=eval/popqa/default \
  #   forget_split=${retain_split} \
  #   task_name=${task_name} \
  #   paths.output_dir=${out_root}/${task_name} \
  #   model.model_args.pretrained_model_name_or_path=${orig_model_path}

  # Eval for each Unlearn model
  for trainer in "${trainers[@]}"; do
    task_name="popqa_${model}_${forget_split}_${trainer}"
    model_path="saves/unlearn/${task_name}"
    echo "=== EVAL RETAIN ($trainer) $task_name on $retain_split ==="
    CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 src/eval.py \
      --config-name=eval.yaml \
      experiment=eval/popqa/default \
      forget_split=${retain_split} \
      task_name=${task_name} \
      paths.output_dir=${out_root}/${task_name} \
      model.model_args.pretrained_model_name_or_path=${model_path}
  done
done
