#!/usr/bin/env python3
# tools/merge_peft.py
import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Путь к базовой полной модели (config.json внутри)")
    ap.add_argument("--adapter", required=True, help="Путь к PEFT/LoRA адаптеру (adapter_config.json внутри)")
    ap.add_argument("--out", required=True, help="Куда сохранить слитую полную модель")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[merge_peft] Loading base: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype="auto", device_map=None)

    print(f"[merge_peft] Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("[merge_peft] Merging and unloading...")
    model = model.merge_and_unload()

    print(f"[merge_peft] Saving merged to: {out}")
    tok.save_pretrained(out.as_posix())
    model.save_pretrained(out.as_posix())

    print("[merge_peft] Done.")

if __name__ == "__main__":
    main()
