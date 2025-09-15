#!/usr/bin/env bash
set -euo pipefail

# ============ БАЗА ============
PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
cd "$PROJECT_ROOT"

MODEL_NAME="Llama-3.1-8B-Instruct"
DATE_TAG="$(date +%Y%m%d_%H%M)"
RUN_TAG="${RUN_TAG_OVERRIDE:-tripunlamb_${DATE_TAG}}"

# Базы: orig — HF id, ft — локальная full
declare -A BASES=(
  ["orig"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
  ["ft"]="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb"
)

# RMU временно выключен
TRAINERS=( "GradAscent" "GradDiff" "NPO" )
# TRAINERS=( "GradAscent" "GradDiff" "NPO" "RMU" )

LEARNING_RATES=( "1e-6" "5e-6" "1e-5" "2e-5" "4e-5" "5e-5") # "1e-5" 

# UNLEARN идёт по FORGET-сетам; RETAIN используется как "ретейн" при train и как target при eval
FORGET_SPLITS=( "know_intersection" )
RETAIN_SPLIT="fast_retain"

UNLEARN_EXPERIMENT="unlearn/tripunlamb/default.yaml"
EVAL_EXPERIMENT="eval/tripunlamb/default"

GPU="${GPU:-0}"
NUM_PROCESSES_TRAIN=1
NUM_PROCESSES_EVAL=1
ACCELERATE_TRAIN_FLAGS=( --num_machines=1 --num_processes="${NUM_PROCESSES_TRAIN}" --mixed_precision=bf16 --dynamo_backend=no )
ACCELERATE_EVAL_FLAGS=(  --num_machines=1 --num_processes="${NUM_PROCESSES_EVAL}"  --mixed_precision=no   --dynamo_backend=no )

# Лог: в него попадут только echo из этого скрипта
LOG_DIR="${PROJECT_ROOT}/saves/logs/${RUN_TAG}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/run_unlearn_and_eval_two_bases.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "▶ PROJECT_ROOT: $PROJECT_ROOT"
echo "▶ RUN_TAG     : $RUN_TAG"
echo "▶ LOG_FILE    : $LOG_FILE"

# ============ УТИЛИТЫ ============
_has_weight_files() { local d="$1"; [[ -f "$d/model.safetensors" ]] || [[ -f "$d/model.safetensors.index.json" ]] || [[ -f "$d/pytorch_model.bin" ]] || [[ -f "$d/pytorch_model.bin.index.json" ]]; }
is_peft_dir () { local d="$1"; [[ -f "$d/adapter_config.json" ]] || [[ -f "$d/adapter_model.safetensors" ]]; }
is_full_model_dir () { local d="$1"; [[ -f "$d/config.json" ]] && _has_weight_files "$d"; }
latest_valid_checkpoint () { local base="$1"; local ckpt; ckpt="$(ls -d "$base"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"; if [[ -n "${ckpt:-}" ]] && { is_full_model_dir "$ckpt" || is_peft_dir "$ckpt"; }; then echo "$ckpt"; fi; }
find_model_root () {
  local base="${1%/}"
  if is_peft_dir "$base"; then echo "$base"; return 0; fi
  if is_full_model_dir "$base"; then echo "$base"; return 0; fi
  local ckpt; ckpt="$(latest_valid_checkpoint "$base" || true)"
  if [[ -n "${ckpt:-}" ]]; then echo "$ckpt"; return 0; fi
  local acfg; acfg="$(find "$base" -maxdepth 3 -type f -name 'adapter_config.json' 2>/dev/null | head -n1 || true)"
  if [[ -n "${acfg:-}" ]]; then echo "$(dirname "$acfg")"; return 0; fi
  local wf; wf="$(find "$base" -maxdepth 3 -type f \( -name 'model.safetensors' -o -name 'model.safetensors.index.json' -o -name 'pytorch_model.bin' -o -name 'pytorch_model.bin.index.json' \) 2>/dev/null | head -n1 || true)"
  if [[ -n "${wf:-}" ]]; then echo "$(dirname "$wf")"; return 0; fi
  return 1
}
merge_if_needed_and_get_path () {
  local maybe_adapter="$1" base_path="$2" cache_dir="$3"
  if is_peft_dir "$maybe_adapter"; then
    local rel out; rel="$(echo "$maybe_adapter" | sed 's|/|_|g')"
    out="${cache_dir}/merged_${rel}"
    if [[ -f "$out/config.json" ]] && _has_weight_files "$out"; then echo "$out"; return 0; fi
    echo "🔧 Merging PEFT with base → $out"
    python3 -u "${PROJECT_ROOT}/tools/merge_peft.py" --base "$base_path" --adapter "$maybe_adapter" --out "$out" > "${out}.merge.log" 2>&1
    echo "$out"; return 0
  fi
  if is_full_model_dir "$maybe_adapter"; then echo "$maybe_adapter"; return 0; fi
  return 1
}
abbr_trainer () { case "$1" in GradAscent) echo "GA";; GradDiff) echo "GD";; NPO) echo "NPO";; # RMU) echo "RMU";;
  *) echo "$1";; esac; }
print_overrides () { local title="$1"; shift; echo "——— ${title}: FINAL HYDRA OVERRIDES ———"; for arg in "$@"; do echo "  $arg"; done; echo "———————————————————————————————————————"; }

# ============ EVAL helper ============
eval_one () {
  local label="$1" model_path="$2" base_label="$3" out_dir="$4" retain_split="$5"
  mkdir -p "$out_dir"
  [[ "$out_dir" == *"_${base_label}" ]] || { echo "❌ out_dir должен заканчиваться на _${base_label}: $out_dir"; return 1; }

  if [[ -f "$out_dir/.success" ]]; then
    echo "✅ EVAL already OK: [$base_label] $label → $(basename "$out_dir")"
    return 0
  fi

  echo "🧪 EVAL: [$base_label] $label → $(basename "$out_dir")"

  # ВАЖНО: для eval forget_split получает RETAIN-набор
  EVAL_OVERRIDES=(
    "experiment=${EVAL_EXPERIMENT}"
    "forget_split=${retain_split}"
    "task_name=tripunlamb_${MODEL_NAME}_${label}_${base_label}"
    "paths.output_dir=${out_dir}"
    "model.model_args.pretrained_model_name_or_path=${model_path}"
    ${HUGGINGFACE_HUB_TOKEN:+model.model_args.hf_token=${HUGGINGFACE_HUB_TOKEN}}
    "hydra.run.dir=."
    "hydra.output_subdir=null"
  )
  print_overrides "EVAL ${label} (${base_label})" "${EVAL_OVERRIDES[@]}"

  set +e
  CUDA_VISIBLE_DEVICES="$GPU" accelerate launch "${ACCELERATE_EVAL_FLAGS[@]}" src/eval.py \
    --config-name=eval.yaml \
    "${EVAL_OVERRIDES[@]}" \
    > "${out_dir}/eval_results.txt" 2>&1
  local code=$?
  set -e

  if [[ $code -eq 0 ]] && grep -q "Result for metric" "${out_dir}/eval_results.txt"; then
    echo "OK" > "${out_dir}/.success"
    rm -f "${out_dir}/.failed" 2>/dev/null || true
    echo "✅ EVAL SUCCESS: [$base_label] $label"
  else
    echo "FAIL (code=${code})" > "${out_dir}/.failed"
    echo "❌ EVAL FAIL: [$base_label] $label (code=${code})"
  fi
}

# ============ ШАГ 1: UNLEARN ============
for base_label in orig ft; do
  BASE_MODEL_PATH="${BASES[$base_label]}"
  echo
  echo "==================== UNLEARN BASE: ${base_label} ===================="
  echo "BASE_MODEL_PATH: ${BASE_MODEL_PATH}"

  for forget in "${FORGET_SPLITS[@]}"; do
    for tr in "${TRAINERS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        UNL_OUT="${PROJECT_ROOT}/saves/unlearn/${base_label}_${MODEL_NAME}_${tr}_${forget}_lr_${lr}"
        mkdir -p "$UNL_OUT"
        if [[ -f "${UNL_OUT}/.done" ]]; then
          echo "✅ UNLEARN already done: $(basename "$UNL_OUT")"
          continue
        fi
        echo "🎓 UNLEARN: ${base_label} | ${tr} | ${forget} | lr=${lr}"

        TRAIN_OVERRIDES=(
          "experiment=${UNLEARN_EXPERIMENT}"
          "trainer=${tr}"
          "task_name=tripunlamb_${MODEL_NAME}_${tr}_${forget}_lr_${lr}_${base_label}"
          "model=${MODEL_NAME}"
          "forget_split=${forget}"
          "retain_split=${RETAIN_SPLIT}"
          "model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH}"
          "trainer.args.learning_rate=${lr}"
          "trainer.args.per_device_train_batch_size=4"
          "trainer.args.gradient_accumulation_steps=4"
          "trainer.args.ddp_find_unused_parameters=true"
          "trainer.args.gradient_checkpointing=true"
          "paths.output_dir=${UNL_OUT}"
        )
        print_overrides "UNLEARN ${tr} ${forget} lr=${lr} (${base_label})" "${TRAIN_OVERRIDES[@]}"

        set +e
        CUDA_VISIBLE_DEVICES="$GPU" accelerate launch "${ACCELERATE_TRAIN_FLAGS[@]}" src/train.py \
          --config-name=unlearn.yaml \
          "${TRAIN_OVERRIDES[@]}" \
          > "${UNL_OUT}/train.log" 2>&1
        code=$?
        set -e
        if [[ $code -eq 0 ]]; then
          echo "OK" > "${UNL_OUT}/.done"
          echo "✅ UNLEARN SUCCESS: $(basename "$UNL_OUT")"
        else
          echo "❌ UNLEARN FAIL: $(basename "$UNL_OUT") (code=${code})"
        fi
      done
    done
  done
done
echo; echo "🌳 UNLEARN loops finished."
# ============ ШАГ 2: EVAL (только RETAIN) ============
LEGACY_ROOT="${PROJECT_ROOT}/saves/unlearn/tripunlamb_Llama8B"

# Экранируем MODEL_NAME для bash-regex
MODEL_RE="$(printf '%s' "$MODEL_NAME" | sed -E 's/[][(){}.^$|*+?\\]/\\&/g')"

for base_label in orig ft; do
  BASE_MODEL_PATH="${BASES[$base_label]}"
  echo
  echo "==================== EVAL BASE: ${base_label} ===================="
  echo "BASE_MODEL_PATH: ${BASE_MODEL_PATH}"

  # 0) База без анлерна (на RETAIN)
  OUT_BASE="${PROJECT_ROOT}/saves/eval/${base_label}_${MODEL_NAME}_BASE_ONLY_${RETAIN_SPLIT}_set_${base_label}"
  eval_one "BASE_ONLY" "$BASE_MODEL_PATH" "$base_label" "$OUT_BASE" "$RETAIN_SPLIT"

  # 1) Новые артефакты текущего запуска (если есть)
  for forget in "${FORGET_SPLITS[@]}"; do
    for tr in "${TRAINERS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        NEW_DIR="${PROJECT_ROOT}/saves/unlearn/${base_label}_${MODEL_NAME}_${tr}_${forget}_lr_${lr}"
        [[ -d "$NEW_DIR" ]] || continue

        ROOT="$(find_model_root "$NEW_DIR" || true)" || true
        [[ -n "${ROOT:-}" ]] || { echo "⚠️  no model root in $NEW_DIR"; continue; }

        MERGE_CACHE="${PROJECT_ROOT}/saves/_merged_cache/${RUN_TAG}/${base_label}"
        mkdir -p "$MERGE_CACHE"
        EVAL_MODEL_PATH="$(merge_if_needed_and_get_path "$ROOT" "$BASE_MODEL_PATH" "$MERGE_CACHE" || true)" || true
        [[ -n "${EVAL_MODEL_PATH:-}" ]] || { echo "⚠️  cannot prepare eval model from $ROOT"; continue; }

        abbr="$(abbr_trainer "$tr")"
        OUT_DIR="${PROJECT_ROOT}/saves/eval/${base_label}_${MODEL_NAME}_${abbr}_${forget}_lr_${lr}_${RETAIN_SPLIT}_set_${base_label}"
        eval_one "${tr}_${forget}_lr_${lr}" "$EVAL_MODEL_PATH" "$base_label" "$OUT_DIR" "$RETAIN_SPLIT"
      done
    done
  done

  # 2) LEGACY-артефакты формата:
  # tripunlamb_<MODEL_NAME>_<TRAINER>_<FORGET>_lr_<LR>_(ft|orig)
  if [[ -d "$LEGACY_ROOT" ]]; then
    echo "🔎 LEGACY scan: $LEGACY_ROOT"
    shopt -s nullglob
    for d in "$LEGACY_ROOT"/tripunlamb_${MODEL_NAME}_*_lr_*_{ft,orig}; do
      [[ -d "$d" ]] || continue
      label="$(basename "$d")"

      tr=""; forget=""; lr=""; base_suffix=""
      if [[ "$label" =~ ^tripunlamb_${MODEL_RE}_(GradAscent|GradDiff|NPO|RMU)_(.+)_lr_([0-9eE.+-]+)_(ft|orig)$ ]]; then
        tr="${BASH_REMATCH[1]}"
        forget="${BASH_REMATCH[2]}"
        lr="${BASH_REMATCH[3]}"
        base_suffix="${BASH_REMATCH[4]}"
      else
        echo "⚠️  skip (name parse failed): $label"
        continue
      fi

      # фильтр по текущей базе цикла
      [[ "$base_suffix" == "$base_label" ]] || continue

      ROOT="$(find_model_root "$d" || true)" || true
      [[ -n "${ROOT:-}" ]] || { echo "⚠️  no model root in $d"; continue; }

      MERGE_CACHE="${PROJECT_ROOT}/saves/_merged_cache/${RUN_TAG}/${base_label}"
      mkdir -p "$MERGE_CACHE"
      EVAL_MODEL_PATH="$(merge_if_needed_and_get_path "$ROOT" "$BASE_MODEL_PATH" "$MERGE_CACHE" || true)" || true
      [[ -n "${EVAL_MODEL_PATH:-}" ]] || { echo "⚠️  cannot prepare eval model from $ROOT"; continue; }

      abbr="$(abbr_trainer "$tr")"
      OUT_DIR="${PROJECT_ROOT}/saves/eval/${base_label}_${MODEL_NAME}_${abbr}_${forget}_lr_${lr}_${RETAIN_SPLIT}_set_${base_label}"
      echo "   • parsed: tr=${tr} | split=${forget} | lr=${lr} | base=${base_suffix}"
      eval_one "${tr}_${forget}_lr_${lr}" "$EVAL_MODEL_PATH" "$base_label" "$OUT_DIR" "$RETAIN_SPLIT"
    done
    shopt -u nullglob
  fi

  echo "✅ EVAL finished for base: ${base_label}"
done


echo; echo "🌼 All done."
echo "📜 Full log: $LOG_FILE"
