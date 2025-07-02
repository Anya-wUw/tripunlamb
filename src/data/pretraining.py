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
