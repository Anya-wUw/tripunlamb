#!/usr/bin/env bash
set -euo pipefail
set -x 

PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
cd "${PROJECT_ROOT}"

models=(
  "Llama-3.1-8B-Instruct"
  "Llama-3.2-1B-Instruct"
  "Llama-3.2-3B-Instruct"
)

orig_splits=("rare_forget10" "popular_forget10")
trainers=( "GradAscent" "GradDiff" "NPO" "RMU" )

tasks=("hellaswag" "mmlu") #  "gsm8k"  "triviaqa" 

run_lm_eval() {
  local task="$1"        
  local name="$2"        
  local orig_up="$3"     
  local result_dir="$4"  
  local orig_low="${orig_up//Llama/llama}"
  local path=""
  local renamed=false

  if [ -d "$orig_up" ]; then
    mv "$orig_up" "$orig_low"
    path="$orig_low"
    renamed=true
  elif [ -d "$orig_low" ]; then
    path="$orig_low"
  else
    echo "⚠️  Пропускаю ${name}: не найдена папка $orig_up или $orig_low" >&2
    return
  fi

  mkdir -p "$result_dir"
  local full_log="${result_dir}/${name}.log"
  local table_file="${result_dir}/${name}.txt"

  echo "=== $(date +'%Y-%m-%d %H:%M:%S') START ${task} ${name} ===" >> "$full_log"

  lm_eval \
    --model hf \
    --model_args "pretrained=${path}" \
    --apply_chat_template \
    --tasks "${task}" \
    --device cuda:1 \
    --batch_size auto:4 \
    2>&1 | tee -a "$full_log" | {
      sed -n '/^|/p' > "$table_file"
    }

  echo "=== $(date +'%Y-%m-%d %H:%M:%S') END   ${task} ${name} ===" >> "$full_log"
  echo ">>> Готово: ${task}/${name} → лог ${full_log}, таблица ${table_file}" >&2

  if [ "$renamed" = true ]; then
    mv "$orig_low" "$orig_up"
  fi
}

for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    results_dir="${PROJECT_ROOT}/saves/lm_eval/${model}/${task}"

    base_name="popqa_${model}"
    base_path="${PROJECT_ROOT}/saves/unlearn/${base_name}"
    run_lm_eval "$task" "$base_name" "$base_path" "$results_dir"

    for orig in "${orig_splits[@]}"; do
      for trainer in "${trainers[@]}"; do
        name="unlearn_${orig}_${trainer}"
        path_up="${PROJECT_ROOT}/saves/unlearn/popqa_${model}_${orig}_${trainer}"
        run_lm_eval "$task" "$name" "$path_up" "$results_dir"
      done
    done
  done
done
