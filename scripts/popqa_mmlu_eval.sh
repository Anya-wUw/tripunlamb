#!/usr/bin/env bash
OUT_DIR="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/gridearch_res"
CSV="${OUT_DIR}/mmlu_results.csv"

echo "run_id,mmlu" > "$CSV"

# Ищем все папки checkpoint-* в результатах гридсерча
find "$OUT_DIR" -type d -name "checkpoint-*" | sort | while read ckpt; do
  # extract run_id from path: .../<run_id>/checkpoint-XXX
  run_dir=$(dirname "$ckpt")
  run_id=$(basename "$run_dir")

  echo "Evaluating MMLU for $run_id..."
  # lm_eval сохранит результат по умолчанию в lm_eval_results.json в cwd
  TMP_JSON="${ckpt}/mmlu_tmp.json"
  lm_eval --model hf \
    --model_args pretrained=local_model_path="$ckpt" \
    --apply_chat_template \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path "$TMP_JSON"

  # Парсим из JSON
  mmlu=$(jq -r '.results.mmlu' "$TMP_JSON")
  echo "${run_id},${mmlu}" >> "$CSV"
done

echo "Done. Results saved to $CSV"
