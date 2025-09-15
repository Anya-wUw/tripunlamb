# import torch
from torch.utils.data import Dataset
import random
from data.utils import (
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
)


class CompletionDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prefix_key="prompt",
        text_key="text",
        max_length=2048,
        predict_with_generate=False,
        insert_space=False,
    ):
        super(CompletionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        # if either key does not exist in dataset, it is taken as ""
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, text_content, index=-1):
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            prefix,
            text_content,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        if index != -1:
            item_dct["index"] = index
        return item_dct

    def __getitem__(self, idx):
        pref = self.data[idx].get(self.prefix_key, "")
        text_content = self.data[idx].get(self.text_key, "")
        index = self.data[idx]["index"]
        item = self._process_sample(pref, text_content, index)
        return item


class PretrainingDataset(Dataset):
    def __init__(
        self, hf_args, template_args, tokenizer, text_key="text", max_length=2048
    ):
        super(PretrainingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunks = self._chunk_raw_text(load_hf_dataset(**hf_args)[text_key])

    def _chunk_raw_text(self, raw_text):
        raw_text = "\n\n".join(raw_text)
        full_token_sequence = self.tokenizer(raw_text, add_special_tokens=False)[
            "input_ids"
        ]
        num_chunks = len(full_token_sequence) // self.max_length + 1
        chunks = []
        for i in range(num_chunks):
            chunks.append(
                self.tokenizer.decode(
                    full_token_sequence[i * self.max_length : (i + 1) * self.max_length]
                )
            )
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return preprocess_pretraining_instance(
            self.tokenizer, "", self.chunks[idx], self.max_length
        )






# TripUnlamb
# import torch
import re
# ---- helpers: маппинг имени модели -> колонки метрик в df/dataset ----
def _normalize_model_name(name: str) -> str:
    """
    'meta-llama/Llama-3.2-1B-Instruct' -> 'llama-3.2-1b-instruct'
    'Llama-3.2-1B-Instract'            -> 'llama-3.2-1b-instruct'
    'google/gemma-7b-it'               -> 'gemma-7b-it'
    """
    s = name.split("/")[-1]  # без компании
    s = s.strip().lower()
    s = s.replace("instract", "instruct")  # частая опечатка
    s = re.sub(r"\s+", "-", s)
    return s
def _metric_columns_for_model(model_name: str):
    """
    Возвращает (recall_col, sim_col) под твои имена колонок.
    Бросает NotImplementedError, если не знаем такую модель.
    """
    key = _normalize_model_name(model_name)

    # Основные варианты
    mapping = {
        # Llama-3.*
        "llama-3.2-1b-instruct": ("gen_recall_Llama_1b_Instract", "bert_sim_Llama_1b_Instract"),
        "llama-3.2-3b-instruct": ("gen_recall_Llama_3b_Instract", "bert_sim_Llama_3b_Instract"),
        "llama-3.1-8b-instruct": ("gen_recall_Llama_8b_Instract", "bert_sim_Llama_8b_Instract"),
        # Gemma
        "gemma-7b-it": ("gen_recall_Gemma_7b_IT", "bert_sim_Gemma_7b_IT"),

        # Zephyr
        "zephyr-7b-beta": ("gen_recall_Zephyr_7b_Beta", "bert_sim_Zephyr_7b_Beta"),

        # Phi
        "phi-3.5-mini-instruct": ("gen_recall_Phi3_5_mini_Instruct", "bert_sim_Phi3_5_mini_Instruct"),
        "phi3.5-mini-instruct":  ("gen_recall_Phi3_5_mini_Instruct", "bert_sim_Phi3_5_mini_Instruct"),
    }

    if key in mapping:
        return mapping[key]

    raise NotImplementedError(
        f"No gold_set filter mapping for model_name='{model_name}' "
        f"(normalized '{key}'). Add it to _metric_columns_for_model()."
    )
class TripUnlambQADataset(Dataset):
    """
     идентично MultiAnswerQADataset, но:
      - dataset колонки: 'question' и 'answer'
      - gold_set=True -> фильтруем по выбранной модели: recall<0.5 & bert_sim<0.5
    """
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key: str = "question",
        answers_key: str = "answer",
        model_name: str = "Llama-3.2-1B-Instruct",
        max_length: int = 512,
        predict_with_generate: bool = False,
        insert_space: bool = False,
        gold_set: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 1) грузим датасет (HF datasets)
        raw_data = load_hf_dataset(**hf_args)

        # 2) при gold_set — фильтр по колонкам для выбранной модели
        if gold_set:
            rec_col, sim_col = _metric_columns_for_model(model_name)

            # safeguard: если колонок нет — пусть будет явная ошибка
            for col in (rec_col, sim_col):
                if col not in raw_data.column_names:
                    raise KeyError(
                        f"Column '{col}' not found in dataset. "
                        f"Available: {raw_data.column_names}"
                    )

            def _is_gold(example):
                # берем только те, где обе метрики строго < 0.5
                # NaN/None считаем невалидными (не пропускаем)
                try:
                    rec = float(example[rec_col]) if example[rec_col] is not None else 1.0
                    sim = float(example[sim_col]) if example[sim_col] is not None else 1.0
                except Exception:
                    return False
                return (rec < 0.5) and (sim < 0.5)

            raw_data = raw_data.filter(_is_gold)

        # 3) добавляем индекс строки
        self.data = add_dataset_index(raw_data)

        self.question_key = question_key
        self.answer_key = answers_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        tokenized = preprocess_pretraining_instance(
            self.tokenizer,
            question,
            answer,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item = {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "attention_mask": tokenized["attention_mask"],
        }
        if index != -1:
            item["index"] = index
        return item

    def __getitem__(self, idx):
        ex = self.data[idx]
        question = ex.get(self.question_key, "") or ""

        # answer может быть строкой или списком — поддержим оба случая
        ans_val = ex.get(self.answer_key, "")
        if isinstance(ans_val, list):
            answer = random.choice(ans_val) if ans_val else ""
        else:
            answer = ans_val or ""

        index = ex["index"]
        return self._process_sample(question, answer, index)






# PopQA Dataset
class MultiAnswerQADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answers_key="possible_answers",
        model_name="Llama-3.2-1B-Instruct",
        max_length=2048,
        predict_with_generate=False,
        insert_space=False,
        gold_set=False,
    ):
        super(MultiAnswerQADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        raw_data = load_hf_dataset(**hf_args)

        # Filter
        if gold_set:
            if model_name == "Llama-3.2-1B-Instruct":
                # by knowlage Llama-3.2-1B
                raw_data = raw_data.filter(
                    lambda x: not x["know_llama3_1b"] and not x["emb_llama3_1b"]
                )
            else:
                raise NotImplementedError(
                    f"No filter_columns for gold_set for {model_name} model."
                )

        self.data = add_dataset_index(raw_data)

        self.question_key = question_key
        self.answers_key = answers_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            question,
            answer,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        if index != -1:
            item_dct["index"] = index
        return item_dct

    def __getitem__(self, idx):
        question = self.data[idx].get(self.question_key, "")
        answers = self.data[idx].get(self.answers_key, [])
        if not answers:
            answers = [""]  # empty

        selected_answer = random.choice(answers)
        index = self.data[idx]["index"]

        return self._process_sample(question, selected_answer, index)
