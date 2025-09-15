#!/usr/bin/env bash
set -euo pipefail

########################################
# Настройки проекта и окружения
########################################

PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
cd "${PROJECT_ROOT}"

# Подбираем случайный свободный порт для распределённого запуска
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

# Размер «забывания» (можно менять на 5, 10, 15 и т.п.)
split_="15"

# Базовая модель и путь к её FT-весам
model="Llama-3.1-8B-Instruct" #"Phi-3.5-mini-instruct"
orig_model_path="${PROJECT_ROOT}/saves/finetune/${model}_full_5ep_ft_popqa"

# Алгоритмы «разучивания»
trainers=( "GradAscent" "GradDiff" "NPO" "RMU" )

# Сплиты, на которых мы «забываем»
orig_splits=( "rare_forget${split_}" "popular_forget${split_}" )

# Все сплиты для финальной оценки:
eval_splits=()
for base in rare_forget popular_forget \
            duplicate_answers_rare_forget duplicate_answers_popular_forget \
            duplicate_subjects_rare_forget duplicate_subjects_popular_forget; do
  eval_splits+=( "${base}${split_}" )
done
# 2) retain_intersection = 100 − 2*split_
inter=$(( 100 - 2 * split_ ))
eval_splits+=( "retain_intersection${inter}" )

########################################
# Подготовка директории для метрик
########################################

metrics_dir="${PROJECT_ROOT}/saves/eval/forget_${split_}/metrics_unlearn_${model}"
mkdir -p "${metrics_dir}"
metrics_file="${metrics_dir}/res.txt"

# Заголовок в файл метрик
{
  echo "==== Metrics for ${model} ===="
  echo ""
} > "${metrics_file}"

########################################
# Шаг 1: Оценка оригинальной FT-модели
########################################

echo
echo ">>> EVAL ORIGINAL FT MODEL"
echo "==== ${model}_Original ====" >> "${metrics_file}"

for split in "${eval_splits[@]}"; do
  echo "--- on split: ${split} ---"
  echo "*** ${split} ***" >> "${metrics_file}"

  out_dir="${PROJECT_ROOT}/saves/eval/forget_${split_}/eval_final_experiments/${model}/Original/on_${split}"
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

  if [ ${exit_code} -ne 0 ]; then
    msg="🚨 Original eval failed on split ${split} (code ${exit_code})"
    echo "${msg}" | tee -a "${metrics_file}"
  else
    if grep -q "^Result for metric" "${out_dir}/eval_results.txt"; then
      grep "^Result for metric" "${out_dir}/eval_results.txt" >> "${metrics_file}"
    else
      msg="⚠️  No metrics found for Original on split ${split}"
      echo "${msg}" | tee -a "${metrics_file}"
    fi
  fi

  echo "" >> "${metrics_file}"
done

########################################
# Шаг 2: Оценка после «разучивания» (UNLEARN)
########################################

for orig in "${orig_splits[@]}"; do
  for trainer in "${trainers[@]}"; do
    task_name="popqa_${model}_${orig}_${trainer}"
    model_path="${PROJECT_ROOT}/saves/unlearn/${task_name}"

    if [ ! -d "${model_path}" ]; then
      echo "⚠️  Skipping ${task_name}: directory not found at ${model_path}"
      continue
    fi

    echo
    echo ">>> EVAL ${task_name}"
    echo "==== ${task_name} ====" >> "${metrics_file}"

    for split in "${eval_splits[@]}"; do
      # Фильтрация сплитов: для rare — только rare или retain, для popular — только popular или retain
      if [[ "${orig}" == rare* ]]; then
        [[ "${split}" == *rare* || "${split}" == *retain* ]] || continue
      elif [[ "${orig}" == popular* ]]; then
        [[ "${split}" == *popular* || "${split}" == *retain* ]] || continue
      fi

      echo "--- on split: ${split} ---"
      echo "*** ${split} ***" >> "${metrics_file}"

      out_dir="${PROJECT_ROOT}/saves/eval/forget_${split_}/eval_final_experiments/${model}/${orig}_${trainer}/on_${split}"
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
        msg="🚨 Eval failed for ${task_name} on split ${split} (code ${exit_code})"
        echo "${msg}" | tee -a "${metrics_file}"
      else
        if grep -q "^Result for metric" "${out_dir}/eval_results.txt"; then
          grep "^Result for metric" "${out_dir}/eval_results.txt" >> "${metrics_file}"
        else
          msg="⚠️  No metrics found for ${task_name} on split ${split}"
          echo "${msg}" | tee -a "${metrics_file}"
        fi
      fi

      echo "" >> "${metrics_file}"
    done
  done
done







# #!/usr/bin/env bash
# set -euo pipefail

# ########################################
# # Настройки проекта и окружения
# ########################################

# PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
# cd "${PROJECT_ROOT}"

# # Подбираем случайный свободный порт для распределённого запуска
# export MASTER_PORT=$(python - <<'PYCODE'
# import socket
# s = socket.socket(); s.bind(('', 0))
# print(s.getsockname()[1])
# s.close()
# PYCODE
# )

# ########################################
# # Параметры эксперимента
# ########################################

# # Размер «забывания» (можно менять на 5, 10, 15 и т.п.)
# split_="10"

# # Базовая модель и путь к её FT‑весам
# model="Phi-3.5-mini-instruct" #"gemma-7b-it"
# orig_model_path="${PROJECT_ROOT}/saves/finetune/${model}_full_5ep_ft_popqa"

# # Алгоритмы «разучивания»
# trainers=( "GradAscent" "GradDiff" "NPO" "RMU" )

# # Сплиты, на которых мы «забываем»
# orig_splits=( "rare_forget${split_}" "popular_forget${split_}" )

# # Все сплиты для финальной оценки:
# # 1) forget‑сплиты (rare, popular, duplicate_answers, duplicate_subjects, duplicate_entities)
# eval_splits=()
# for base in rare_forget popular_forget \
#             duplicate_answers_rare_forget duplicate_answers_popular_forget \
#             duplicate_subjects_rare_forget duplicate_subjects_popular_forget; do #\
#             # duplicate_entities_rare_forget duplicate_entities_popular_forget; do
#   eval_splits+=( "${base}${split_}" )
# done
# # 2) retain_intersection = 100 − 2*split_
# inter=$(( 100 - 2 * split_ ))
# eval_splits+=( "retain_intersection${inter}" )

# ########################################
# # Подготовка директории для метрик
# ########################################

# metrics_dir="${PROJECT_ROOT}/saves/eval/forget_${split_}/metrics_unlearn_${model}"
# mkdir -p "${metrics_dir}"
# metrics_file="${metrics_dir}/res.txt"

# # Заголовок в файл метрик
# {
#   echo "==== Metrics for ${model} ===="
#   echo ""
# } > "${metrics_file}"

# # #######################################
# # Шаг 1: Оценка оригинальной FT‑модели
# # #######################################

# echo
# echo ">>> EVAL ORIGINAL FT MODEL"
# echo "==== ${model}_Original ====" >> "${metrics_file}"

# for split in "${eval_splits[@]}"; do
#   echo "--- on split: ${split} ---"
#   echo "*** ${split} ***" >> "${metrics_file}"

#   out_dir="${PROJECT_ROOT}/saves/eval/forget_${split_}/eval_final_experiments/${model}/Original/on_${split}"
#   mkdir -p "${out_dir}"

#   set +e
#   CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 src/eval.py \
#     --config-name=eval.yaml \
#     experiment=eval/popqa/default \
#     forget_split="${split}" \
#     task_name="popqa_${model}_Original" \
#     paths.output_dir="${out_dir}" \
#     model.model_args.pretrained_model_name_or_path="${orig_model_path}" \
#     2>&1 | tee "${out_dir}/eval_results.txt"
#   exit_code=$?
#   set -e

#   if [ ${exit_code} -ne 0 ]; then
#     msg="🚨 Original eval failed on split ${split} (code ${exit_code})"
#     echo "${msg}" | tee -a "${metrics_file}"
#   else
#     if grep -q "^Result for metric" "${out_dir}/eval_results.txt"; then
#       grep "^Result for metric" "${out_dir}/eval_results.txt" >> "${metrics_file}"
#     else
#       msg="⚠️  No metrics found for Original on split ${split}"
#       echo "${msg}" | tee -a "${metrics_file}"
#     fi
#   fi

#   echo "" >> "${metrics_file}"
# done

# ########################################
# # Шаг 2: Оценка после «разучивания» (UNLEARN)
# ########################################

# for orig in "${orig_splits[@]}"; do
#   for trainer in "${trainers[@]}"; do
#     task_name="popqa_${model}_${orig}_${trainer}"
#     # путь к папке с моделью после unlearn
#     model_path="${PROJECT_ROOT}/saves/unlearn/${task_name}"

#     if [ ! -d "${model_path}" ]; then
#       echo "⚠️  Skipping ${task_name}: directory not found at ${model_path}"
#       continue
#     fi

#     echo
#     echo ">>> EVAL ${task_name}"
#     echo "==== ${task_name} ====" >> "${metrics_file}"

#     for split in "${eval_splits[@]}"; do
#       echo "--- on split: ${split} ---"
#       echo "*** ${split} ***" >> "${metrics_file}"

#       out_dir="${PROJECT_ROOT}/saves/eval/forget_${split_}/eval_final_experiments/${model}/${orig}_${trainer}/on_${split}"
#       mkdir -p "${out_dir}"

#       set +e
#       CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 src/eval.py \
#         --config-name=eval.yaml \
#         experiment=eval/popqa/default \
#         forget_split="${split}" \
#         task_name="${task_name}" \
#         paths.output_dir="${out_dir}" \
#         model.model_args.pretrained_model_name_or_path="${model_path}" \
#         2>&1 | tee "${out_dir}/eval_results.txt"
#       exit_code=$?
#       set -e

#       if [ ${exit_code} -ne 0 ]; then
#         msg="🚨 Eval failed for ${task_name} on split ${split} (code ${exit_code})"
#         echo "${msg}" | tee -a "${metrics_file}"
#       else
#         if grep -q "^Result for metric" "${out_dir}/eval_results.txt"; then
#           grep "^Result for metric" "${out_dir}/eval_results.txt" >> "${metrics_file}"
#         else
#           msg="⚠️  No metrics found for ${task_name} on split ${split}"
#           echo "${msg}" | tee -a "${metrics_file}"
#         fi
#       fi

#       echo "" >> "${metrics_file}"
#     done
#   done
# done



# #!/usr/bin/env bash
# set -euo pipefail

# PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
# cd "${PROJECT_ROOT}"

# export MASTER_PORT=$(python - <<'PYCODE'
# import socket
# s = socket.socket(); s.bind(('', 0))
# print(s.getsockname()[1])
# s.close()
# PYCODE
# )

# split_="10"
# model="gemma-7b-it" #"Llama-3.1-8B-Instruct"
# orig_model_path="${PROJECT_ROOT}/saves/finetune/gemma-7b-it_full_5ep_ft_popqa"

# orig_splits=("rare_forget10" "popular_forget10" ) #"rare_forget5" "popular_forget5" 
# trainers=("GradAscent" "GradDiff" "NPO" "RMU" ) 

# eval_splits=(
#   "rare_forget${split_}"
#   "popular_forget${split_}"
#   "duplicate_answers_rare_forget${split_}"
#   "duplicate_answers_popular_forget${split_}"
#   "duplicate_entities_rare_forget${split_}"
#   "duplicate_entities_popular_forget${split_}"
#   "retain_intersection80" #CALC!
#   # "rare_forget5"
#   # "popular_forget5"
#   # "duplicate_answers_rare_forget5"
#   # "duplicate_answers_popular_forget5"
#   # "duplicate_entities_rare_forget5"
#   # "duplicate_entities_popular_forget5"
#   # "retain_intersection90"
# )

# metrics_dir="${PROJECT_ROOT}/saves/eval/forget_${split_}/metrics_unlear_${model}"
# mkdir -p "${metrics_dir}"
# metrics_file="${metrics_dir}/res.txt"

# {
#   echo "==== Metrics for ${model} ===="
#   echo ""
# } > "${metrics_file}"

# echo
# echo ">>> EVAL ORIGINAL FT MODEL"
# echo "==== ${model}_Original ====" >> "${metrics_file}"
# for split in "${eval_splits[@]}"; do
#   echo "--- on split: ${split} ---"
#   echo "*** ${split} ***" >> "${metrics_file}"

#   out_dir="${PROJECT_ROOT}/saves/eval/forget_${split_}/eval_final_experiments/${model}/Original/on_${split}"
#   mkdir -p "${out_dir}"

#   set +e
#   CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 src/eval.py \
#     --config-name=eval.yaml \
#     experiment=eval/popqa/default \
#     forget_split="${split}" \
#     task_name="popqa_${model}_Original" \
#     paths.output_dir="${out_dir}" \
#     model.model_args.pretrained_model_name_or_path="${orig_model_path}" \
#     2>&1 | tee "${out_dir}/eval_results.txt"
#   exit_code=$?
#   set -e

#   if [ $exit_code -ne 0 ]; then
#     msg="🚨 Original eval failed on split ${split} (code ${exit_code})"
#     echo "${msg}"
#     echo "${msg}" >> "${metrics_file}"
#   else
#     if grep -q "^Result for metric" "${out_dir}/eval_results.txt"; then
#       grep "^Result for metric" "${out_dir}/eval_results.txt" >> "${metrics_file}"
#     else
#       msg="⚠️  No metrics found for Original on split ${split}"
#       echo "${msg}"
#       echo "${msg}" >> "${metrics_file}"
#     fi
#   fi

#   echo "" >> "${metrics_file}"
# done



# for orig in "${orig_splits[@]}"; do
#   for trainer in "${trainers[@]}"; do
#     task_name="popqa_${model}_${orig}_${trainer}"
#     model_path="${PROJECT_ROOT}/saves/unlearn/forget_${split_}/${task_name}"

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

#       out_dir="${PROJECT_ROOT}/saves/eval/forget_${split_}/eval_final_experiments/${model}/${orig}_${trainer}/on_${split}"
#       mkdir -p "${out_dir}"

#       set +e
#       CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 src/eval.py \
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
#         msg="🚨 Eval failed for ${task_name} on split ${split} (code ${exit_code})"
#         echo "${msg}"
#         echo "${msg}" >> "${metrics_file}"
#       else
#         if grep -q "^Result for metric" "${out_dir}/eval_results.txt"; then
#           grep "^Result for metric" "${out_dir}/eval_results.txt" >> "${metrics_file}"
#         else
#           msg="⚠️  No metrics found for ${task_name} on split ${split}"
#           echo "${msg}"
#           echo "${msg}" >> "${metrics_file}"
#         fi
#       fi

#       echo "" >> "${metrics_file}"
#     done
#   done
# done