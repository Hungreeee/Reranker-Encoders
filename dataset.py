import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextPairDataset(Dataset):
    def __init__(
        self, 
        base_dataset: Dataset, 
        tokenizer: AutoTokenizer, 
        max_length: int = 512
    ):
        self.base_dataset = base_dataset
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.queries = [a for b in [[query] * 10 for query in base_dataset["query"]] for a in b]
        self.passages = [passage for passages in base_dataset["passages"] for passage in passages["passage_text"]]
        self.labels = [label for labels in base_dataset["passages"] for label in labels["is_selected"]]


    def __len__(self):
        return len(self.base_dataset)


    def __getitem__(self, index: int):
        query = self.queries[index]
        passage = self.passages[index]
        label = self.labels[index]

        query_inputs = self.tokenizer.encode_plus(
            query, 
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        context_inputs = self.tokenizer.encode_plus(
            passage, 
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return \
            query_inputs["input_ids"].squeeze(0), \
            query_inputs["attention_mask"].squeeze(0), \
            context_inputs["input_ids"].squeeze(0), \
            context_inputs["attention_mask"].squeeze(0), \
            torch.tensor(label, dtype=torch.float)



