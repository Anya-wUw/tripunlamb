#!/usr/bin/env bash
set -euo pipefail

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# Свободный порт (короткий однострочник без heredoc — не сломает кавычки)
export MASTER_PORT="$(python - <<<'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
echo "Master port: ${MASTER_PORT}"

# === НАСТРОЙКИ ===
# Список моделей (HF id или локальный путь) — БЕЗ запятых!
MODELS=(
  "meta-llama/Llama-3.1-8B"
  "/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb"
)

OUTPUT_BASE="saves/eval/ft_models_tripunlamb/orig-llama3.1-8b"

SPLITS=(
  "fast_retain"
)

# GPU по умолчанию = 1; можно переопределить: GPU=0 ./triplamb_ft_eval.sh
: "${GPU:=1}"

mkdir -p "${OUTPUT_BASE}"

# Нормализация имени модели для путей/тегов
safe_tag() {
  local s="$1"
  s="${s//\//__}"   # '/' -> '__'
  s="${s// /_}"     # ' ' -> '_'
  echo "$s"
}

for MODEL in "${MODELS[@]}"; do
  MODEL_TAG="$(safe_tag "$MODEL")"
  MODEL_DIR="${OUTPUT_BASE}/${MODEL_TAG}"
  mkdir -p "${MODEL_DIR}"

  echo "=== Evaluating model: ${MODEL} (tag: ${MODEL_TAG}) ==="

  for split in "${SPLITS[@]}"; do
    echo "--- Split: ${split} ---"

    SPLIT_DIR="${MODEL_DIR}/${split}"
    mkdir -p "${SPLIT_DIR}"

    TASK_NAME="tripunlamb_${MODEL_TAG}_${split}"
    EVAL_FILE="${SPLIT_DIR}/${TASK_NAME}_eval_results.txt"

    # Пропуск, если файл уже есть, не пустой и содержит метрики
    if [[ -s "${EVAL_FILE}" ]] && grep -q "^Result for metric" "${EVAL_FILE}"; then
      echo "↳ Уже есть валидные результаты: ${EVAL_FILE} — пропуск."
      continue
    fi

    set +e
    CUDA_VISIBLE_DEVICES="${GPU}" python -u src/eval.py \
      --config-name=eval.yaml \
      experiment=eval/tripunlamb/default \
      forget_split="${split}" \
      task_name="${TASK_NAME}" \
      paths.output_dir="${SPLIT_DIR}" \
      model.model_args.pretrained_model_name_or_path="${MODEL}" \
      2>&1 | tee "${EVAL_FILE}"
    code=$?
    set -e

    if [[ $code -ne 0 ]]; then
      echo "❌ Eval failed (code=${code}) for ${MODEL} / ${split}"
    else
      echo "✅ Done: ${EVAL_FILE}"
    fi

    echo
  done
done
