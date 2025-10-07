# 🧩 Welcome to DUAL

**DUAL** (**Dual Unlearning Evaluation across Learning stages**) is a benchmark and dataset introduced in the paper  
**_Anatomy of Forgetting: The Dual Impact of Fact Salience and Model Fine-Tuning_**.  

It provides a systematic framework to study **how fact salience (popularity)** and **model training stage (Pretrained vs. SFT)** jointly affect the success of **Machine Unlearning** in Large Language Models (LLMs).  

The benchmark consists of **28.6k Wikidata-derived question–answer pairs**, annotated with fact popularity and organized into multiple forget/retain splits (1 %, 5 %, 10 %) for reproducible experiments.

---

## ⚡ Quickstart

```bash
# Environment setup
conda create -n dual python=3.11
conda activate dual
pip install .[lm_eval]
pip install --no-build-isolation flash-attn==2.6.3
```

### 📦 Dataset setup

```bash
# Download and prepare DUAL data
python setup_data.py --eval
# This populates saves/eval with logs and evaluation data
# For available dataset variants, run:
# python setup_data.py --help
```

---

## 🧠 DUAL Benchmark Structure

- **Rare vs. Popular facts:** grouped by real-world prominence (Wikipedia sitelinks and LLM salience scores)  
- **Pretrained vs. SFT regimes:** same architecture evaluated under different training paradigms  
- **Forgetting & Retention metrics:** ROUGE-L, accuracy, and composite Δ-scores  
- **Algorithms:** GradAscent, GradDiff, NPO, and other unlearning baselines  

---

## 🔧 Scripts

### 🪶 Fine-tuning

```bash
bash dual/scripts/dual_finetune.sh
```

### 🧩 Unlearning

```bash
bash dual/scripts/dual_unlearn.sh
```

### 📊 Evaluation (Forgetting & Retention)

```bash
bash dual/scripts/dual_eval_metrics.sh
```

### 🧪 General Benchmarks (e.g., MMLU, HellaSwag)

```bash
bash dual/scripts/dual_llm_eval.sh
```

---

## 📊 Example Evaluation

Run DUAL evaluation for the **city_forget_rare_10** split:

```bash
model=gemma-7b-it
python src/eval.py --config-name=eval.yaml experiment=eval/dual/default   model=${model}   model.model_args.pretrained_model_name_or_path=saves/finetune/dual_${model}_full   retain_logs_path=saves/eval/dual_${model}_retain/city_fast_retain_500.json   task_name=RARE_FORGET_10
```

---

## 🧑‍🔬 Baseline Experiments

The following scripts reproduce baseline results for DUAL benchmark experiments:

```bash
bash scripts/dual_unlearn.sh
bash scripts/dual_eval.sh
```

Expected metrics and plots are available in [`docs/repro.md`](docs/repro.md).

---

## 🤝 Acknowledgements

This benchmark builds upon the [Open-Unlearning](https://github.com/locuslab/open-unlearning) framework  
and extends it with popularity-aware and stage-aware evaluation protocols.

---

## 📄 License

This project is licensed under the MIT License.  
See the [`LICENSE`](LICENSE) file for details.
