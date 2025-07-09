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

########################################
# 2. Параметры модели и сплитов
########################################
model="Llama-3.1-8B-Instruct"
orig_model_path="${PROJECT_ROOT}/saves/finetune/llama3.1-8b_full_2ep_ft_popqa"

# сплиты, на которых проводилось unlearning
orig_splits=("rare_forget10" "popular_forget10" ) #
# алгоритмы unlearning
trainers=("GradAscent" "GradDiff" "NPO" "RMU" ) #
# сплиты для валидации
eval_splits=(
  "duplicate_rare_forget10"
  "duplicate_popular_forget10"
  "rare_forget10"
  "popular_forget10"
  "retain_intersection90"
)

########################################
# 3. Файл для сбора метрик
########################################
metrics_dir="${PROJECT_ROOT}/saves/eval/metrics_unlear_${model}"
mkdir -p "${metrics_dir}"
metrics_file="${metrics_dir}/res.txt"

# --- сбросим старый файл ---
{
  echo "==== Metrics for ${model} ===="
  echo ""
} > "${metrics_file}"

########################################
# 4. Eval оригинальной ft-модели
########################################
echo
echo ">>> EVAL ORIGINAL FT MODEL"
echo "==== ${model}_Original ====" >> "${metrics_file}"
for split in "${eval_splits[@]}"; do
  echo "--- on split: ${split} ---"
  echo "*** ${split} ***" >> "${metrics_file}"

  out_dir="${PROJECT_ROOT}/saves/eval/eval_final_experiments/${model}/Original/on_${split}"
  mkdir -p "${out_dir}"

  set +e
  CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 src/eval.py \
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

########################################
# 5. Eval fine-tuned (unlearned) моделей
########################################
for orig in "${orig_splits[@]}"; do
  for trainer in "${trainers[@]}"; do
    task_name="popqa_${model}_${orig}_${trainer}"
    model_path="${PROJECT_ROOT}/saves/unlearn/${task_name}"

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

      out_dir="${PROJECT_ROOT}/saves/eval/eval_final_experiments/${model}/${orig}_${trainer}/on_${split}"
      mkdir -p "${out_dir}"

      set +e
      CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 src/eval.py \
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

# #!/usr/bin/env bash
# set -euo pipefail

# ########################################
# # 1. Настройки проекта и порт для accelerate
# ########################################
# PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
# cd "${PROJECT_ROOT}"

# export MASTER_PORT=$(python - <<'PYCODE'
# import socket
# s = socket.socket(); s.bind(('', 0))
# print(s.getsockname()[1])
# s.close()
# PYCODE
# )

# ########################################
# # 2. Параметры модели и сплитов
# ########################################
# model="Llama-3.2-1B-Instruct"
# orig_model_path="${PROJECT_ROOT}/saves/finetune/llama3.2-1b_full_5ep_ft_popqa"

# orig_splits=( "rare_forget10" "popular_forget10" )
# trainers=( "GradAscent" "GradDiff" "NPO" "RMU" )
# eval_splits=(
#   "duplicate_rare_forget10"
#   "duplicate_popular_forget10"
#   "retain_intersection90"
#   # "rare_forget10"
#   # "popular_forget10"
# )

# ########################################
# # 3. Файл для сбора метрик
# ########################################
# metrics_dir="${PROJECT_ROOT}/saves/eval/metrics_unlear_${model}"
# mkdir -p "${metrics_dir}"
# metrics_file="${metrics_dir}/res.txt"

# # Сброс старого
# {
#   echo "==== Metrics for ${model} ===="
#   echo ""
# } > "${metrics_file}"

# ########################################
# # 4. Eval fine-tuned (unlearned) моделей
# ########################################
# for orig in "${orig_splits[@]}"; do
#   for trainer in "${trainers[@]}"; do
#     task_name="popqa_${model}_${orig}_${trainer}"
#     model_path="${PROJECT_ROOT}/saves/unlearn/${task_name}"

#     if [ ! -d "${model_path}" ]; then
#       echo "⚠️  Пропускаю ${task_name}: папка не найдена"
#       continue
#     fi

#     echo
#     echo ">>> EVAL ${task_name}"
#     echo "==== ${task_name} ====" >> "${metrics_file}"

#     for split in "${eval_splits[@]}"; do
#       echo "--- on split: ${split} ---"
#       echo "*** ${split} ***" >> "${metrics_file}"

#       out_dir="${PROJECT_ROOT}/saves/eval/eval_final_experiments/${model}/${orig}_${trainer}/on_${split}"
#       mkdir -p "${out_dir}"

#       # запускаем ускоренно, ловим статус
#       set +e
#       CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 src/eval.py \
#         --config-name=eval.yaml \
#         experiment=eval/popqa/default \
#         forget_split="${split}" \
#         task_name="${task_name}" \
#         paths.output_dir="${out_dir}" \
#         model.model_args.pretrained_model_name_or_path="${model_path}" \
#         2>&1 | tee "${out_dir}/eval_results.txt"
#       exit_code=$?
#       set -e

#       if [ $exit_code -ne 0 ]; then
#         err_msg="🚨 Eval failed for ${task_name} on split ${split}, exit code ${exit_code}"
#         echo "${err_msg}"
#         echo "${err_msg}" >> "${metrics_file}"
#       else
#         # парсим метрики — не падаем, даже если grep ничего не найдёт
#         if grep -q "^Result for metric" "${out_dir}/eval_results.txt"; then
#           grep "^Result for metric" "${out_dir}/eval_results.txt" >> "${metrics_file}"
#         else
#           no_msg="⚠️  No metrics found for ${task_name} on split ${split}"
#           echo "${no_msg}"
#           echo "${no_msg}" >> "${metrics_file}"
#         fi
#       fi

#       echo "" >> "${metrics_file}"
#     done
#   done
# done









# #!/usr/bin/env bash
# set -euo pipefail

# #
# # Настройки проекта и порт для accelerate
# #
# PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
# cd "${PROJECT_ROOT}"

# export MASTER_PORT=$(python - <<'PYCODE'
# import socket
# s = socket.socket(); s.bind(('', 0))
# print(s.getsockname()[1])
# s.close()
# PYCODE
# )

# #
# # Параметры моделей и сплитов
# #
# model="Llama-3.2-1B-Instruct"
# orig_model_path="${PROJECT_ROOT}/saves/finetune/llama3.2-1b_full_5ep_ft_popqa"

# # unlearning sets (только эти две)
# orig_splits=( "rare_forget10" "popular_forget10" )

# # все сплиты, на которых мерить
# eval_splits=(
#   "rare_forget10"
#   "popular_forget10"
#   "duplicate_rare_forget10"
#   "duplicate_popular_forget10"
#   "retain_intersection90"
# )

# # Unlearning алгоритмы
# trainers=( "GradAscent" "GradDiff" "NPO" "RMU" )

# #
# # Файл для сбора всех метрик оригинальных моделей
# #
# metrics_dir="${PROJECT_ROOT}/saves/eval/metrics_unlear_${model}"
# mkdir -p "${metrics_dir}"
# metrics_file="${metrics_dir}/res.txt"

# # Заголовок в файле
# # echo "==== ${model} ORIGINAL MODELS ====" > "${metrics_file}"

# # #
# # 1) Eval оригинальных моделей (rare_forget10, popular_forget10) на всех сплитах
# #
# for orig in "${orig_splits[@]}"; do
#   echo "*** MODEL: popqa_${model}_${orig}_Original ***" | tee -a "${metrics_file}"
#   for split in "${eval_splits[@]}"; do
#     echo "*** eval on split ${split} ***" | tee -a "${metrics_file}"

#     out_dir="${PROJECT_ROOT}/saves/eval/eval_final_experiments/${model}/${orig}/Original_on_${split}"
#     mkdir -p "${out_dir}"

#     CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 src/eval.py \
#       --config-name=eval.yaml \
#       experiment=eval/popqa/default \
#       forget_split="${split}" \
#       task_name="popqa_${model}_${orig}_Original" \
#       paths.output_dir="${out_dir}" \
#       model.model_args.pretrained_model_name_or_path="${orig_model_path}" \
#       2>&1 | tee "${out_dir}/eval_results.txt" | tee -a "${metrics_file}"

#     echo "" | tee -a "${metrics_file}"
#   done
# done

# #
# # 2) Eval unlearn-моделей (как было раньше)
# #
# for split in "${eval_splits[@]}"; do
#   out_root="${PROJECT_ROOT}/saves/eval/eval_final_experiments/${model}/${split}"
#   mkdir -p "${out_root}"

#   for trainer in "${trainers[@]}"; do
#     task_name="popqa_${model}_${split}_${trainer}"
#     model_path="${PROJECT_ROOT}/saves/unlearn/${task_name}"
#     eval_dir="${out_root}/${task_name}"
#     mkdir -p "${eval_dir}"

#     echo "=== EVAL ${trainer}: ${task_name} on split ${split} ==="
#     CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 src/eval.py \
#       --config-name=eval.yaml \
#       experiment=eval/popqa/default \
#       forget_split="${split}" \
#       task_name="${task_name}" \
#       paths.output_dir="${eval_dir}" \
#       model.model_args.pretrained_model_name_or_path="${model_path}" \
#       2>&1 | tee "${eval_dir}/eval_results.txt"
#     echo
#   done
# done









# #!/usr/bin/env bash
# set -euo pipefail

# # случайный свободный порт для accelerate (если нужно)
# export MASTER_PORT=$(python - <<'PYCODE'
# import socket
# s = socket.socket(); s.bind(('', 0))
# print(s.getsockname()[1])
# s.close()
# PYCODE
# )

# model="Llama-3.2-1B-Instruct" #"Llama-3.1-8B-Instruct"
# orig_model_path="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.2-1b_full_5ep_ft_popqa"

# # Unlearning algorithms (имена папок совпадают с task_name)
# trainers=( "GradAscent" "GradDiff" "NPO" "RMU" )

# # Все сплиты, по которым хотим делать eval
# eval_splits=(
#   "rare_forget10"
#   "popular_forget10"
#   "duplicate_rare_forget10"
#   "duplicate_popular_forget10"
#   "retain_intersection90"
# )

# for split in "${eval_splits[@]}"; do
#   out_root="saves/eval/eval_final_experiments/${model}/${split}"
#   mkdir -p "${out_root}"

#   ### 1) Eval оригинальной finetuned модели ###
#   task_name="popqa_${model}_${split}_Original"
#   eval_dir="${out_root}/${task_name}"
#   mkdir -p "${eval_dir}"

#   echo "=== EVAL ORIGINAL: $task_name on split $split ==="
#   CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 src/eval.py \
#     --config-name=eval.yaml \
#     experiment=eval/popqa/default \
#     forget_split=${split} \
#     task_name=${task_name} \
#     paths.output_dir=${eval_dir} \
#     model.model_args.pretrained_model_name_or_path=${orig_model_path} \
#     2>&1 | tee "${eval_dir}/${model}_eval_results.txt"
#   echo

#   ### 2) Eval каждой unlearn-модели ###
#   for trainer in "${trainers[@]}"; do
#     task_name="popqa_${model}_${split}_${trainer}"
#     model_path="saves/unlearn/${task_name}"
#     eval_dir="${out_root}/${task_name}"
#     mkdir -p "${eval_dir}"

#     echo "=== EVAL ${trainer}: $task_name on split $split ==="
#     CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 src/eval.py \
#       --config-name=eval.yaml \
#       experiment=eval/popqa/default \
#       forget_split=${split} \
#       task_name=${task_name} \
#       paths.output_dir=${eval_dir} \
#       model.model_args.pretrained_model_name_or_path=${model_path} \
#       2>&1 | tee "${eval_dir}/${model}_${trainer}_eval_results.txt"
#     echo
#   done
# done