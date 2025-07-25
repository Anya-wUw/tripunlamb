#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
cd "${PROJECT_ROOT}"

export MASTER_PORT=$(python - <<'PYCODE'
import socket
s = socket.socket(); s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PYCODE
)

model="Llama-3.1-8B-Instruct"
orig_model_path="${PROJECT_ROOT}/saves/finetune/llama3.1-8b_full_5ep_ft_popqa"

orig_splits=("rare_forget5" "popular_forget5" ) 
trainers=("GradAscent" "GradDiff" "NPO" "RMU" ) 

eval_splits=(
  # "rare_forget10"
  # "popular_forget10"
  # "duplicate_answers_rare_forget10"
  # "duplicate_answers_popular_forget10"
  # "duplicate_entities_rare_forget10"
  # "duplicate_entities_popular_forget10"
  # "retain_intersection80"
  "rare_forget5"
  "popular_forget5"
  "duplicate_answers_rare_forget5"
  "duplicate_answers_popular_forget5"
  "duplicate_entities_rare_forget5"
  "duplicate_entities_popular_forget5"
  "retain_intersection90"
)

metrics_dir="${PROJECT_ROOT}/saves/eval/forget_5/metrics_unlear_${model}"
mkdir -p "${metrics_dir}"
metrics_file="${metrics_dir}/res.txt"

{
  echo "==== Metrics for ${model} ===="
  echo ""
} > "${metrics_file}"

echo
echo ">>> EVAL ORIGINAL FT MODEL"
echo "==== ${model}_Original ====" >> "${metrics_file}"
for split in "${eval_splits[@]}"; do
  echo "--- on split: ${split} ---"
  echo "*** ${split} ***" >> "${metrics_file}"

  out_dir="${PROJECT_ROOT}/saves/eval/forget_5/eval_final_experiments/${model}/Original/on_${split}"
  mkdir -p "${out_dir}"

  set +e
  CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/popqa/default \
    forget_split="${split}" \
    task_name="popqa_${model}_Original" \
    paths.output_dir="${out_dir}" \
    model.model_args.pretrained_model_name_or_path="${orig_model_path}" \
    2>&1 | tee "${out_dir}/eval_results.txt"
  exit_code=$?
  set -e

  if [ $exit_code -ne 0 ]; then
    msg="🚨 Original eval failed on split ${split} (code ${exit_code})"
    echo "${msg}"
    echo "${msg}" >> "${metrics_file}"
  else
    if grep -q "^Result for metric" "${out_dir}/eval_results.txt"; then
      grep "^Result for metric" "${out_dir}/eval_results.txt" >> "${metrics_file}"
    else
      msg="⚠️  No metrics found for Original on split ${split}"
      echo "${msg}"
      echo "${msg}" >> "${metrics_file}"
    fi
  fi

  echo "" >> "${metrics_file}"
done



for orig in "${orig_splits[@]}"; do
  for trainer in "${trainers[@]}"; do
    task_name="popqa_${model}_${orig}_${trainer}"
    model_path="${PROJECT_ROOT}/saves/unlearn/forget_5/${task_name}"

    if [ ! -d "${model_path}" ]; then
      echo "⚠️  Пропускаю ${task_name}: папка не найдена"
      continue
    fi

    echo
    echo ">>> EVAL ${task_name}"
    echo "==== ${task_name} ====" >> "${metrics_file}"

    for split in "${eval_splits[@]}"; do
      echo "--- on split: ${split} ---"
      echo "*** ${split} ***" >> "${metrics_file}"

      out_dir="${PROJECT_ROOT}/saves/eval/forget_5/eval_final_experiments/${model}/${orig}_${trainer}/on_${split}"
      mkdir -p "${out_dir}"

      set +e
      CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 src/eval.py \
        --config-name=eval.yaml \
        experiment=eval/popqa/default \
        forget_split="${split}" \
        task_name="${task_name}" \
        paths.output_dir="${out_dir}" \
        model.model_args.pretrained_model_name_or_path="${model_path}" \
        2>&1 | tee "${out_dir}/eval_results.txt"
      exit_code=$?
      set -e

      if [ $exit_code -ne 0 ]; then
        msg="🚨 Eval failed for ${task_name} on split ${split} (code ${exit_code})"
        echo "${msg}"
        echo "${msg}" >> "${metrics_file}"
      else
        if grep -q "^Result for metric" "${out_dir}/eval_results.txt"; then
          grep "^Result for metric" "${out_dir}/eval_results.txt" >> "${metrics_file}"
        else
          msg="⚠️  No metrics found for ${task_name} on split ${split}"
          echo "${msg}"
          echo "${msg}" >> "${metrics_file}"
        fi
      fi

      echo "" >> "${metrics_file}"
    done
  done
done