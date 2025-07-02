# #!/usr/bin/env python3
# import os
# import subprocess
# import json
# import glob
# import pandas as pd

# # Путь к корню, куда train.py сохраняет модели:
# BASE_SAVE = "saves/finetune"
# # Куда складывать финальный CSV
# OUT_CSV = "/mnt/extremessd10tb/borisiuk/open-unlearning/saves/gridearch_res/grid_search_results.csv"
# os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# def run_one(lr: float, r: int, alpha: int, num_epochs: int):
#     run_id = f"popqa_llama-3.1-8B-Instruct_full_finetune_{num_epochs}ep_{r}r_{alpha}a_{lr:.0e}lr"
#     save_dir = os.path.join(BASE_SAVE, run_id)

#     # 1) Тренировка
#     cmd = [
#         "python", "src/train.py",
#         "--config-name=train.yaml",
#         "experiment=finetune/popqa/default",
#         "model=Llama-3.1-8B-Instruct",
#         f"trainer.args.learning_rate={lr}",
#         f"trainer.args.num_train_epochs={num_epochs}",
#         f"peft.lora.r={r}",
#         f"peft.lora.alpha={alpha}",
#         f"task_name={run_id}"
#     ]
#     print(">>> TRAIN:", " ".join(cmd))
#     subprocess.check_call(cmd)

#     # 2) Находим последний checkpoint-*
#     ckpts = glob.glob(os.path.join(save_dir, "checkpoint-*"))
#     if not ckpts:
#         raise FileNotFoundError(f"No checkpoints in {save_dir}")
#     # сортировка по номеру чекпоинта
#     ckpts = sorted(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))
#     last_ckpt = ckpts[-1]

#     # 3) Путь к PopQA_SUMMARY.json
#     summary_path = os.path.join(last_ckpt, "evals", "PopQA_SUMMARY.json")
#     if not os.path.isfile(summary_path):
#         raise FileNotFoundError(f"Cannot find PopQA summary at {summary_path}")

#     # 4) Читаем метрику
#     with open(summary_path, "r") as f:
#         summary = json.load(f)
#     rouge = summary["forget_Q_A_ROUGE"]

#     return {"run_id": run_id, "learning_rate": lr, "rouge": rouge}

# def main():
#     learning_rates = [3e-5, 5e-5, 1e-4, 2e-4, 3e-4] #1e-5, 2e-5, 
#     r = 8
#     alpha = 8
#     num_epochs = 2

#     rows = []
#     for lr in learning_rates:
#         row = run_one(lr, r, alpha, num_epochs)
#         rows.append(row)
#         # Сохраняем промежуточные результаты
#         pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
#         print("Saved interim results to", OUT_CSV)

#     print("=== Final results ===")
#     print(pd.DataFrame(rows))

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import os
import subprocess
import json
import glob
import re
import pandas as pd

BASE_SAVE = "saves/finetune"
OUT_CSV = "/mnt/extremessd10tb/borisiuk/open-unlearning/saves/gridearch_res/grid_search_results.csv"
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

def run_one(lr: float, r: int, alpha: int, num_epochs: int):
    run_id = f"popqa_llama-3.1-8B-Instruct_full_finetune_{num_epochs}ep_{r}r_{alpha}a_{lr:.0e}lr"
    save_dir = os.path.join(BASE_SAVE, run_id)

    cmd = [
        "python", "src/train.py",
        "--config-name=train.yaml",
        "experiment=finetune/popqa/default",
        "model=Llama-3.1-8B-Instruct",
        f"trainer.args.learning_rate={lr}",
        f"trainer.args.num_train_epochs={num_epochs}",
        f"peft.lora.r={r}",
        f"peft.lora.alpha={alpha}",
        f"task_name={run_id}"
    ]
    print(">>> TRAIN:", " ".join(cmd))
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)

    m = re.search(r"'train_loss':\s*([0-9]+(?:\.[0-9]+)?)", out)
    train_loss = float(m.group(1)) if m else None
    print(f"  → train_loss = {train_loss}")

    ckpts = glob.glob(os.path.join(save_dir, "checkpoint-*"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {save_dir}")
    ckpts = sorted(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))
    last_ckpt = ckpts[-1]

    summary_path = os.path.join(last_ckpt, "evals", "PopQA_SUMMARY.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Cannot find PopQA summary at {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)
    rouge = summary["forget_Q_A_ROUGE"]
    print(f"  → rouge = {rouge}")

    return {
        "run_id": run_id,
        "learning_rate": lr,
        "train_loss": train_loss,
        "rouge": rouge,
    }

def main():
    learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
    r = 8
    alpha = 8
    num_epochs = 2

    rows = []
    for lr in learning_rates:
        row = run_one(lr, r, alpha, num_epochs)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print("Saved interim results to", OUT_CSV)

    print("=== Final results ===")
    print(pd.DataFrame(rows))

if __name__ == "__main__":
    main()
