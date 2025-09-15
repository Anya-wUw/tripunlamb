
#!/usr/bin/env bashset -euo pipefail

# ========================= BASICS =========================
PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
cd "$PROJECT_ROOT"

MODEL_NAME="Llama-3.1-8B-Instruct"
DATE_TAG="$(date +%Y%m%d_%H%M)"
RUN_TAG="${RUN_TAG_OVERRIDE:-tripunlamb_eval_${DATE_TAG}}"

# HF base and local FT base for "BASE_ONLY" evals
declare -A BASES=(
  ["orig"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
  ["ft"]="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb"
)

# What we evaluate on (your retain target)
RETAIN_SPLIT="${RETAIN_SPLIT:-fast_retain}"
EVAL_EXPERIMENT="eval/tripunlamb/default"

LEGACY_ROOT="${PROJECT_ROOT}/saves/unlearn/tripunlamb_Llama8B"

GPU="${GPU:-0}"
NUM_PROCESSES_EVAL="${NUM_PROCESSES_EVAL:-1}"
ACCELERATE_EVAL_FLAGS=( --num_machines=1 --num_processes="${NUM_PROCESSES_EVAL}" --mixed_precision=no --dynamo_backend=no )

# Log: only this script's echo output is written
LOG_DIR="${PROJECT_ROOT}/saves/logs/${RUN_TAG}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/run_eval_only.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "▶ PROJECT_ROOT : $PROJECT_ROOT"
echo "▶ RUN_TAG      : $RUN_TAG"
echo "▶ LOG_FILE     : $LOG_FILE"
echo "▶ LEGACY_ROOT  : $LEGACY_ROOT"
echo "▶ RETAIN_SPLIT : $RETAIN_SPLIT"

# ========================= HELPERS =========================
_has_weight_files() {
  local d="$1"
  [[ -f "$d/model.safetensors" ]] \
  || [[ -f "$d/model.safetensors.index.json" ]] \
  || [[ -f "$d/pytorch_model.bin" ]] \
  || [[ -f "$d/pytorch_model.bin.index.json" ]]
}
is_peft_dir () { local d="$1"; [[ -f "$d/adapter_config.json" ]] || [[ -f "$d/adapter_model.safetensors" ]]; }
is_full_model_dir () { local d="$1"; [[ -f "$d/config.json" ]] && _has_weight_files "$d"; }

latest_valid_checkpoint () {
  local base="$1"; local ckpt
  ckpt="$(ls -d "$base"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
  if [[ -n "${ckpt:-}" ]] && { is_full_model_dir "$ckpt" || is_peft_dir "$ckpt"; }; then
    echo "$ckpt"
  fi
}

find_model_root () {
  local base="${1%/}"
  # Direct PEFT or full
  if is_peft_dir "$base"; then echo "$base"; return 0; fi
  if is_full_model_dir "$base"; then echo "$base"; return 0; fi
  # Latest checkpoint
  local ckpt; ckpt="$(latest_valid_checkpoint "$base" || true)"
  if [[ -n "${ckpt:-}" ]]; then echo "$ckpt"; return 0; fi
  # Search shallow for adapter/full
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
    if [[ -f "$out/config.json" ]] && _has_weight_files "$out"; then
      echo "$out"; return 0
    fi
    echo "🔧 Merging PEFT with base → $out"
    python3 -u "${PROJECT_ROOT}/tools/merge_peft.py" \
      --base "$base_path" --adapter "$maybe_adapter" --out "$out" \
      > "${out}.merge.log" 2>&1
    echo "$out"; return 0
  fi
  if is_full_model_dir "$maybe_adapter"; then
    echo "$maybe_adapter"; return 0
  fi
  return 1
}

abbr_trainer () {
  case "$1" in
    GradAscent) echo "GA";;
    GradDiff)   echo "GD";;
    NPO)        echo "NPO";;
    RMU)        echo "RMU";;
    *)          echo "$1";;
  esac
}

print_overrides () {
  local title="$1"; shift
  echo "——— ${title}: FINAL HYDRA OVERRIDES ———"
  for arg in "$@"; do echo "  $arg"; done
  echo "———————————————————————————————————————"
}

eval_one () {
  local label="$1" model_path="$2" base_label="$3" out_dir="$4" retain_split="$5"
  mkdir -p "$out_dir"
  if [[ -f "$out_dir/.success" ]]; then
    echo "✅ EVAL already OK: [$base_label] $label → $(basename "$out_dir")"
    return 0
  fi
  echo "🧪 EVAL: [$base_label] $label → $(basename "$out_dir")"

  local EVAL_OVERRIDES=(
    "experiment=${EVAL_EXPERIMENT}"
    "forget_split=${retain_split}"               # use retain set as eval target
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

# ========================= EVAL: BASE ONLY =========================
for base_label in orig ft; do
  BASE_MODEL_PATH="${BASES[$base_label]}"
  echo
  echo "==================== EVAL BASE ONLY: ${base_label} ===================="
  echo "BASE_MODEL_PATH: ${BASE_MODEL_PATH}"
  OUT_BASE="${PROJECT_ROOT}/saves/eval/${base_label}_${MODEL_NAME}_BASE_ONLY_${RETAIN_SPLIT}_set_${base_label}"
  eval_one "BASE_ONLY" "$BASE_MODEL_PATH" "$base_label" "$OUT_BASE" "$RETAIN_SPLIT"
done

echo
echo "🔎 Scan LEGACY artifacts under: $LEGACY_ROOT"

# Escape MODEL_NAME for literal use in regex
MODEL_RE="$(printf '%s' "$MODEL_NAME" | sed -E 's/[][(){}.^$|*+?\\]/\\&/g')"

shopt -s nullglob
for d in "$LEGACY_ROOT"/tripunlamb_${MODEL_NAME}_*_lr_*_{ft,orig}; do
  [[ -d "$d" ]] || continue
  label="$(basename "$d")"

  # Regex:
  # tripunlamb_<MODEL_NAME>_<TRAINER>_<FORGET>_lr_<LR>_<BASE>
  # where <FORGET> can contain underscores; <LR> like 1e-6, 2e-5, 4e-5, 5e-6, etc.
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

  BASE_MODEL_PATH="${BASES[$base_suffix]}"

  ROOT="$(find_model_root "$d" || true)" || true
  if [[ -z "${ROOT:-}" ]]; then
    echo "⚠️  no model root in $d"
    continue
  fi

  MERGE_CACHE="${PROJECT_ROOT}/saves/_merged_cache/${RUN_TAG}/${base_suffix}"
  mkdir -p "$MERGE_CACHE"
  EVAL_MODEL_PATH="$(merge_if_needed_and_get_path "$ROOT" "$BASE_MODEL_PATH" "$MERGE_CACHE" || true)" || true
  if [[ -z "${EVAL_MODEL_PATH:-}" ]]; then
    echo "⚠️  cannot prepare eval model from $ROOT"
    continue
  fi

  abbr="$(abbr_trainer "$tr")"
  OUT_DIR="${PROJECT_ROOT}/saves/eval/${base_suffix}_${MODEL_NAME}_${abbr}_${forget}_lr_${lr}_${RETAIN_SPLIT}_set_${base_suffix}"
  echo "🔎 parsed → tr=${tr} | split=${forget} | lr=${lr} | base=${base_suffix}"
  eval_one "${tr}_${forget}_lr_${lr}" "$EVAL_MODEL_PATH" "$base_suffix" "$OUT_DIR" "$RETAIN_SPLIT"
done
shopt -u nullglob
