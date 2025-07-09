import os
import subprocess
import json
import glob
import re
import pandas as pd

# Hydra + OmegaConf
from hydra import initialize, compose
from omegaconf import OmegaConf

# --- 1) Загрузим базовый конфиг, чтобы достать r, alpha и num_epochs ---
with initialize(config_path="../configs", job_name="grid_search"):
    cfg = compose(
        config_name="train.yaml",
        overrides=[
            "experiment=finetune/popqa/default",
            "model=Llama-3.1-8B-Instruct"
        ]
    )
# Теперь можно брать их напрямую из cfg:
r = cfg.peft.lora.r
alpha = cfg.peft.lora.alpha
num_epochs = cfg.trainer.args.num_train_epochs

BASE_SAVE = "saves/finetune"
OUT_CSV = "/mnt/extremessd10tb/borisiuk/open-unlearning/saves/gridearch_res/grid_search_results.csv"
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

def run_one(lr: float):
    run_id = f"popqa_llama-3.1-8B-Instruct_full_finetune_{num_epochs}ep_{r}r_{alpha}a_{lr:.0e}lr"
    save_dir = os.path.join(BASE_SAVE, run_id)

    cmd = [
        "python", "src/train.py",
        "--config-name=train.yaml",
        "experiment=finetune/popqa/default",
        "model=Llama-3.1-8B-Instruct",
        # переопределяем lr как есть в конфиге
        f"trainer.args.learning_rate={lr}",
        # добавляем новое поле run_name
        f"+trainer.args.run_name={run_id}",
        # Hydra уже загрузит r, alpha и num_train_epochs из default-конфига
        f"task_name={run_id}"
    ]

    print(">>> TRAIN:", " ".join(cmd))

    # Стримим вывод сразу в консоль
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    # После завершения можно снова вызвать check_output, чтобы достать train_loss
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    m = re.search(r"'train_loss':\s*([0-9]+(?:\.[0-9]+)?)", out)
    train_loss = float(m.group(1)) if m else None
    print(f"  → train_loss = {train_loss}")

    # Сбор eval-метрик
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
    # только LR в сетке, остальные параметры — из конфига
    learning_rates = [3e-5, 1e-5, 2e-4, 1e-4, 1e-3]

    rows = []
    for lr in learning_rates:
        row = run_one(lr)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print("Saved interim results to", OUT_CSV)

    print("=== Final results ===")
    print(pd.DataFrame(rows))

if __name__ == "__main__":
    main()
