# #!/usr/bin/env bash
# set -euo pipefail

# # ===== НАСТРОЙКИ =====
# PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
# cd "$PROJECT_ROOT"

# BASE_MODEL_PATH="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb"
# MODEL_NAME="Llama-3.1-8B-Instruct"
# UNLEARN_SAVE_DIR="${PROJECT_ROOT}/saves/unlearn/tripunlamb_Llama8B"

# RUN_TAG="${RUN_TAG_OVERRIDE:-tripunlamb_pop1_fastret_$(date +%Y%m%d)}"
# BASE_EVAL_DIR="${PROJECT_ROOT}/saves/eval/${RUN_TAG}/tripunlamb_FT/${MODEL_NAME}"

# EVAL_CFG="eval/tripunlamb/default"
# RETAIN_SPLIT="fast_retain"
# GPU="${GPU:-0}"
# NUM_PROCESSES_EVAL=1

# MERGED_CACHE_DIR="${PROJECT_ROOT}/saves/unlearn/_merged_cache/${RUN_TAG}"
# mkdir -p "$MERGED_CACHE_DIR"

# # ===== УТИЛИТЫ =====
# is_full_model_dir () { [[ -f "$1/config.json" ]]; }
# is_peft_dir       () { [[ -f "$1/adapter_config.json" ]]; }

# # Возвращает самый свежий checkpoint, но только если он валиден
# latest_valid_checkpoint () {
#   local base="$1"
#   local ckpt
#   ckpt="$(ls -d "$base"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
#   if [[ -n "${ckpt}" ]] && { is_full_model_dir "$ckpt" || is_peft_dir "$ckpt"; }; then
#     echo "$ckpt"
#   fi
# }

# # Находит корень модели, где реально лежит config.json или adapter_config.json
# # 1) base; 2) latest checkpoint (если валиден); 3) рекурсивный поиск до 2 уровней
# find_model_root () {
#   local base="${1%/}"  # срезать хвостовой /
#   # 1) на уровне base
#   if is_full_model_dir "$base" || is_peft_dir "$base"; then
#     echo "$base"; return 0
#   fi
#   # 2) валидный чекпоинт
#   local ckpt; ckpt="$(latest_valid_checkpoint "$base" || true)"
#   if [[ -n "${ckpt:-}" ]]; then
#     echo "$ckpt"; return 0
#   fi
#   # 3) поиск на глубину 2 (учтём случаи, когда файлы лежат внутри подкаталогов)
#   local found
#   found="$(find "$base" -maxdepth 2 -type f \( -name 'config.json' -o -name 'adapter_config.json' \) 2>/dev/null | head -n1 || true)"
#   if [[ -n "${found:-}" ]]; then
#     echo "$(dirname "$found")"; return 0
#   fi
#   return 1
# }

# # Если PEFT — сливаем; если full — возвращаем путь как есть
# merge_if_needed_and_get_path () {
#   local mp="$1"
#   if is_full_model_dir "$mp"; then
#     echo "$mp"; return 0
#   fi
#   if is_peft_dir "$mp"; then
#     local rel out
#     rel="$(echo "$mp" | sed 's|/|_|g')"
#     out="${MERGED_CACHE_DIR}/merged_${rel}"
#     if [[ -f "$out/config.json" ]]; then
#       echo "$out"; return 0
#     fi
#     echo "🔧 Merging PEFT adapter with base model → $out"
#     python3 -u "${PROJECT_ROOT}/tools/merge_peft.py" --base "$BASE_MODEL_PATH" --adapter "$mp" --out "$out"
#     echo "$out"; return 0
#   fi
#   return 1
# }

# eval_one () {
#   local label="$1" model_path="$2"
#   local out_dir="${BASE_EVAL_DIR}/${label}/on_${RETAIN_SPLIT}"
#   mkdir -p "$out_dir"

#   # не евалим второй раз
#   if [[ -f "${out_dir}/.success" ]]; then
#     echo "✅ Уже успешно: ${label}"; return 0
#   fi
#   if [[ -s "${out_dir}/eval_results.txt" ]]; then
#     echo "📄 Уже есть eval_results.txt (не пустой) → пропуск: ${label}"; return 0
#   fi

#   set +e
#   CUDA_VISIBLE_DEVICES="$GPU" accelerate launch --num_processes="${NUM_PROCESSES_EVAL}" src/eval.py \
#     --config-name=eval.yaml \
#     experiment="$EVAL_CFG" \
#     forget_split="$RETAIN_SPLIT" \
#     task_name="tripunlamb_${MODEL_NAME}_${label}" \
#     paths.output_dir="$out_dir" \
#     model.model_args.pretrained_model_name_or_path="$model_path" \
#     2>&1 | tee "${out_dir}/eval_results.txt"
#   local code=$?
#   set -e

#   if [[ $code -eq 0 ]] && grep -q "^Result for metric" "${out_dir}/eval_results.txt"; then
#     echo "OK" > "${out_dir}/.success"
#     rm -f "${out_dir}/.failed" 2>/dev/null || true
#   else
#     echo "FAIL (code=${code})" > "${out_dir}/.failed"
#   fi
# }

# # Явные «немодельные» каталоги можно пропускать сразу (маски дополняй при желании)
# should_skip_dirname () {
#   local name="$1"
#   [[ "$name" =~ ^all_logs_ ]] && return 0
#   return 1
# }

# # ===== ОСНОВНОЙ ПРОГОН =====
# echo "🔁 Retrying failed/missing evals under: ${UNLEARN_SAVE_DIR}"
# shopt -s nullglob
# for dir in "${UNLEARN_SAVE_DIR}"/*/; do
#   dir="${dir%/}"
#   label="$(basename "$dir")"

#   if should_skip_dirname "$label"; then
#     echo "⚠️  Пропуск: ${label} — служебный каталог"; continue
#   fi

#   # Уже есть успех/результаты?
#   out_dir="${BASE_EVAL_DIR}/${label}/on_${RETAIN_SPLIT}"
#   if [[ -f "${out_dir}/.success" ]]; then
#     echo "✅ Уже успешно: ${label}"; continue
#   fi
#   if [[ -s "${out_dir}/eval_results.txt" ]]; then
#     echo "✅ Уже есть результаты (eval_results.txt) — пропуск: ${label}"; continue
#   fi

#   # Поищем корень модели
#   if root="$(find_model_root "$dir")"; then
#     if eval_path="$(merge_if_needed_and_get_path "$root")"; then
#       echo "▶️  RETRY EVAL: ${label}"
#       echo "   • model root: $root"
#       echo "   • eval path : $eval_path"
#       eval_one "$label" "$eval_path"
#     else
#       echo "⚠️  Пропуск: ${label} — найден корень '$root', но нет ни config.json, ни adapter_config.json на нём (для merge)."
#     fi
#   else
#     echo "⚠️  Пропуск: ${label} — не найден ни config.json, ни adapter_config.json (даже глубина 2). Содержимое:"
#     ls -la "$dir" || true
#   fi
# done
# shopt -u nullglob

# echo "✨ Retry pass complete."

