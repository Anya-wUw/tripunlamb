#!/usr/bin/env python3
import json
import itertools

# Жёстко заданный словарь файлов и их коротких меток
FILES = {
    "GradDiff": "/mnt/extremessd10tb/borisiuk/open-unlearning/saves/eval/"
                "eval_final_experiments/Llama-3.2-1B-Instruct/"
                "rare_forget10_GradDiff/on_retain_intersection90/PopQA_EVAL.json",
    
    "GradAscent": "/mnt/extremessd10tb/borisiuk/open-unlearning/saves/eval/"
                  "eval_final_experiments/Llama-3.2-1B-Instruct/"
                  "rare_forget10_GradAscent/on_retain_intersection90/PopQA_EVAL.json",
    
    "orig_FT": "/mnt/extremessd10tb/borisiuk/open-unlearning/saves/eval/"
               "eval_final_experiments/Llama-3.2-1B-Instruct/"
               "retain_intersection90/popqa_Llama-3.2-1B-Instruct_retain_intersection90_Original/"
               "PopQA_EVAL.json",
    
    "NPO": "/mnt/extremessd10tb/borisiuk/open-unlearning/saves/eval/"
           "eval_final_experiments/Llama-3.2-1B-Instruct/"
           "rare_forget10_NPO/on_retain_intersection90/PopQA_EVAL.json",
    
    "RMU": "/mnt/extremessd10tb/borisiuk/open-unlearning/saves/eval/"
           "eval_final_experiments/Llama-3.2-1B-Instruct/"
           "rare_forget10_RMU/on_retain_intersection90/PopQA_EVAL.json"
}

def load_zero_indices(path):
    """
    Загружает JSON из path и возвращает множество индексов (int),
    где 'rouge1_recall' == 0.0
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vbi = data.get("forget_Q_A_ROUGE", {}).get("value_by_index", {})
    return {
        int(idx)
        for idx, entry in vbi.items()
        if entry.get("rouge1_recall") == 0.0
    }

def main():
    # Шаг 1: подсчёт нулевых rouge1_recall в каждом файле
    zero_indices = {}
    print("1) Количество индексов с rouge1_recall == 0.0:")
    for label, path in FILES.items():
        zeros = load_zero_indices(path)
        zero_indices[label] = zeros
        print(f"- {label}: {len(zeros)}")

    # Шаг 2: пересечения между всеми парами меток
    print("\n2) Пересечения индексов с rouge1_recall == 0.0 между файлами:")
    for label1, label2 in itertools.combinations(FILES.keys(), 2):
        inter = zero_indices[label1] & zero_indices[label2]
        print(f"- {label1} ↔ {label2}: {len(inter)}")

if __name__ == "__main__":
    main()



# #!/usr/bin/env python3
# import os
# import json
# import argparse
# from collections import defaultdict

# def find_popqa_files(root_dir):
#     """Найти все PopQA_EVAL.json в дереве каталогов root_dir."""
#     popqa_paths = []
#     for dirpath, _, files in os.walk(root_dir):
#         if 'PopQA_EVAL.json' in files:
#             popqa_paths.append(os.path.join(dirpath, 'PopQA_EVAL.json'))
#     return popqa_paths

# def load_zero_indices(json_path):
#     """Вернуть множество индексов (int), где rouge1_recall == 0.0."""
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     vbi = data.get('forget_Q_A_ROUGE', {}).get('value_by_index', {})
#     zeros = {
#         int(idx)
#         for idx, entry in vbi.items()
#         if entry.get('rouge1_recall') == 0.0
#     }
#     return zeros

# def main(root_dir):
#     files = find_popqa_files(root_dir)
#     if not files:
#         print(f"В каталоге {root_dir} не найдено ни одного PopQA_EVAL.json")
#         return

#     # Шаг 1: для каждого файла — посчитать количество нулевых rouge1_recall
#     zero_by_file = {}
#     print("Количество индексов с rouge1_recall == 0.0:")
#     for path in files:
#         zeros = load_zero_indices(path)
#         zero_by_file[path] = zeros
#         print(f"- {path}: {len(zeros)}")

#     # Шаг 2: пересечения между каждой парой
#     print("\nПересечения индексов с rouge1_recall == 0.0 между файлами:")
#     n = len(files)
#     for i in range(n):
#         for j in range(i + 1, n):
#             f1, f2 = files[i], files[j]
#             inter = zero_by_file[f1] & zero_by_file[f2]
#             name1 = os.path.relpath(os.path.dirname(f1), root_dir)
#             name2 = os.path.relpath(os.path.dirname(f2), root_dir)
#             print(f"- [{name1}] ↔ [{name2}]: {len(inter)} общих индексов")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Подсчет нулевых rouge1_recall и их пересечений в PopQA_EVAL.json"
#     )
#     parser.add_argument(
#         "root_dir",
#         help="Корневая папка, в которой искать PopQA_EVAL.json"
#     )
#     args = parser.parse_args()
#     main(args.root_dir)

