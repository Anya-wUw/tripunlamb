#!/usr/bin/env bash
set -euo pipefail
set -x                           # включаем трассировку команд

########################################
# 1. Настройки проекта
########################################
PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
cd "${PROJECT_ROOT}"

########################################
# 2. Параметры моделей и вывод
########################################
model="Llama-3.2-3B-Instruct"
results_dir="${PROJECT_ROOT}/saves/lm_eval/${model}"
mkdir -p "${results_dir}"

orig_splits=("rare_forget10" "popular_forget10" )     # можно добавить "rare_forget10"
trainers=( "GradAscent" "GradDiff" "NPO" "RMU")  # можно добавить "RMU"

########################################
# 3. Функция с дополнительным логированием
########################################
run_lm_eval() {
  local name="$1"
  local orig_up="$2"
  local orig_low="${2//Llama/llama}"
  local path=""
  local renamed=false

  echo ">>> Проверяю папки для ${name}" >&2
  if [ -d "$orig_up" ]; then
    echo "   Найдена папка с Llama: $orig_up" >&2
    mv "$orig_up" "$orig_low"
    path="$orig_low"
    renamed=true

  elif [ -d "$orig_low" ]; then
    echo "   Найдена папка с llama: $orig_low" >&2
    path="$orig_low"

  else
    echo "⚠️  Пропускаю ${name}: ни $orig_up ни $orig_low не найдены" >&2
    return
  fi

  echo
  echo ">>> Запускаю LM-EVAL for ${name} (path=${path})" >&2

  local full_log="${results_dir}/${name//\//_}.log"
  local table_file="${results_dir}/${name//\//_}.txt"

  # Запись времени начала
  echo "=== $(date +'%Y-%m-%d %H:%M:%S') START ${name} ===" >> "$full_log"

  # сам запуск
  lm_eval \
    --model hf \
    --model_args "pretrained=${path}" \
    --apply_chat_template \
    --tasks mmlu \
    --device cuda:1 \
    --batch_size auto:4 \
    2>&1 | tee -a "$full_log" | {
      # Параллельно вытягиваем таблицу
      sed -n '/^|/p' > "$table_file"
    }

  # Запись времени окончания
  echo "=== $(date +'%Y-%m-%d %H:%M:%S') END   ${name} ===" >> "$full_log"

  echo ">>> Готово для ${name}: полный лог — ${full_log}, таблица — ${table_file}" >&2

  if [ "$renamed" = true ]; then
    mv "$orig_low" "$orig_up"
  fi
}

########################################
# 4. Запускаем eval для всех комбинаций
########################################
for orig in "${orig_splits[@]}"; do
  for trainer in "${trainers[@]}"; do
    name="unlearn_${orig}_${trainer}"
    path_up="${PROJECT_ROOT}/saves/unlearn/popqa_${model}_${orig}_${trainer}"
    run_lm_eval "$name" "$path_up"
  done
done



# #!/usr/bin/env bash
# OUT_DIR="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/gridearch_res"
# CSV="${OUT_DIR}/mmlu_results.csv"

# echo "run_id,mmlu" > "$CSV"

# # Ищем все папки checkpoint-* в результатах гридсерча
# find "$OUT_DIR" -type d -name "checkpoint-*" | sort | while read ckpt; do
#   # extract run_id from path: .../<run_id>/checkpoint-XXX
#   run_dir=$(dirname "$ckpt")
#   run_id=$(basename "$run_dir")

#   echo "Evaluating MMLU for $run_id..."
#   # lm_eval сохранит результат по умолчанию в lm_eval_results.json в cwd
#   TMP_JSON="${ckpt}/mmlu_tmp.json"
#   lm_eval --model hf \
#     --model_args pretrained=local_model_path="$ckpt" \
#     --apply_chat_template \
#     --tasks mmlu \
#     --device cuda:0 \
#     --batch_size auto:4 \
#     --output_path "$TMP_JSON"

#   # Парсим из JSON
#   mmlu=$(jq -r '.results.mmlu' "$TMP_JSON")
#   echo "${run_id},${mmlu}" >> "$CSV"
# done

# echo "Done. Results saved to $CSV"
