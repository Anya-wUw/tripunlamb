#!/usr/bin/env bash
set -euo pipefail

########################################
# Базовые настройки
########################################

PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
cd "${PROJECT_ROOT}"

# Случайный свободный порт
export MASTER_PORT=$(python - <<'PYCODE'
import socket
s = socket.socket(); s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PYCODE
)

########################################
# Параметры эксперимента
########################################

split_="10"  # можно менять
model="Llama-3.1-8B-Instruct"
orig_model_path="${PROJECT_ROOT}/saves/finetune/llama3.1-8b_full_5ep_ft_popqa"

# Алгоритм и конфиг — только GradDiff
trainer="GradDiff"
experiment="unlearn/popqa/default.yaml"

# Grid search по alpha
alphas=(0.1 0.2 0.3 0.4 0.6 0.8)

# Префиксы разучивания
prefixes=( "rare_forget" "popular_forget" )

# retain_intersection
inter=$(( 100 - 2 * split_ ))
retain_split="retain_intersection${inter}"

########################################
# Общая директория для всех алгоритмов
########################################

base_eval_root="${PROJECT_ROOT}/saves/eval/forget_${split_}"
mkdir -p "${base_eval_root}"

########################################
# Внешний файл агрегированных метрик
########################################

global_metrics_dir="${base_eval_root}/metrics_unlearn_${model}"
mkdir -p "${global_metrics_dir}"
global_metrics_file="${global_metrics_dir}/res.txt"

{
  echo "==== Grid search alphas for ${model} with GradDiff ===="
  echo "Split to forget: ${split_}"
  echo "Retain intersection: ${retain_split}"
  echo ""
} > "${global_metrics_file}"

########################################
# Цикл по rare/popular и alpha
########################################

for prefix in "${prefixes[@]}"; do
  forget_split="${prefix}${split_}"
  for alpha in "${alphas[@]}"; do
    task_name="popqa_${model}_${forget_split}_GradDiff_alpha${alpha}"
    unlearn_dir="${base_eval_root}/by_algo/GradDiff/alpha${alpha}/unlearn/${forget_split}"
    eval_base_dir="${base_eval_root}/by_algo/GradDiff/alpha${alpha}/eval/${forget_split}"
    local_metrics_dir="${base_eval_root}/by_algo/GradDiff/alpha${alpha}/metrics"
    mkdir -p "${unlearn_dir}" "${eval_base_dir}" "${local_metrics_dir}"

    model_path="${PROJECT_ROOT}/saves/unlearn/${task_name}"

    # Заголовки локальных метрик
    local_metrics_file="${local_metrics_dir}/metrics_${forget_split}_alpha${alpha}.txt"
    {
      echo "=== Metrics for ${task_name} ==="
      echo "Forget split: ${forget_split}"
      echo "Alpha: ${alpha}"
      echo "Retain split: ${retain_split}"
      echo ""
    } > "${local_metrics_file}"

    echo
    echo ">>> UNLEARN: ${task_name} (alpha=${alpha})"
    echo "==== ${task_name} ====" >> "${global_metrics_file}"
    echo "*** unlearn on ${forget_split} with alpha=${alpha} ***" >> "${global_metrics_file}"

    # Запуск unlearn если не существует
    if [ ! -d "${model_path}" ]; then
      set +e
      CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --mixed_precision=bf16 \
        src/train.py --config-name=unlearn.yaml \
          experiment="${experiment}" \
          trainer="${trainer}" \
          task_name="${task_name}" \
          model="${model}" \
          forget_split="${forget_split}" \
          retain_split="${retain_split}" \
          model.model_args.pretrained_model_name_or_path="${orig_model_path}" \
          trainer.args.per_device_train_batch_size=4 \
          trainer.args.gradient_accumulation_steps=4 \
          trainer.args.ddp_find_unused_parameters=true \
          trainer.args.gradient_checkpointing=true \
          trainer.method_args.alpha=${alpha} \
          retain_logs_path="saves/eval/forget_${split_}/popqa_${model}_${retain_split}/PopQA_EVAL.json" \
          2>&1 | tee "${unlearn_dir}/train_log.txt"
      exit_code=$?
      set -e

      if [ ${exit_code} -ne 0 ]; then
        echo "!!! UNLEARN FAILED for ${task_name} (alpha=${alpha}), skipping eval." | tee -a "${global_metrics_file}" -a "${local_metrics_file}"
        echo "" >> "${global_metrics_file}"
        continue
      fi
    else
      echo "🟡 Skip unlearn: model dir exists for ${task_name}"
    fi

    ########################################
    # Eval после unlearn
    ########################################

    echo ">>> EVAL ${task_name} (alpha=${alpha})"
    echo "==== Eval for ${task_name} ====" >> "${global_metrics_file}"
    echo "*** eval on relevant splits ***" >> "${global_metrics_file}"
    echo "*** eval on relevant splits ***" >> "${local_metrics_file}"

    # Список split-ов: собственный forget и retain
    eval_splits=("${forget_split}" "${retain_split}")

    for split in "${eval_splits[@]}"; do
      # Фильтрация: rare -> rare или retain, popular -> popular или retain
      if [[ "${forget_split}" == rare* ]]; then
        [[ "${split}" == *rare* || "${split}" == *retain* ]] || continue
      elif [[ "${forget_split}" == popular* ]]; then
        [[ "${split}" == *popular* || "${split}" == *retain* ]] || continue
      fi

      echo "--- Eval on split: ${split} ---"
      echo "*** on ${split} ***" >> "${global_metrics_file}"
      echo "*** on ${split} ***" >> "${local_metrics_file}"

      out_dir="${eval_base_dir}/on_${split}"
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

      if [ ${exit_code} -ne 0 ]; then
        echo "🚨 Eval failed for ${task_name} on ${split} (code ${exit_code})" | tee -a "${global_metrics_file}" -a "${local_metrics_file}"
      else
        if grep -q "^Result for metric" "${out_dir}/eval_results.txt"; then
          grep "^Result for metric" "${out_dir}/eval_results.txt" >> "${global_metrics_file}"
          grep "^Result for metric" "${out_dir}/eval_results.txt" >> "${local_metrics_file}"
        else
          echo "⚠️  No metrics found for ${task_name} on ${split}" | tee -a "${global_metrics_file}" -a "${local_metrics_file}"
        fi
      fi

      echo "" >> "${global_metrics_file}"
      echo "" >> "${local_metrics_file}"
    done
  done
done
