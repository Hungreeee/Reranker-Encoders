import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextPairDataset(Dataset):
  def __init__(self, base_dataset, tokenizer, max_length=512):
    self.base_dataset = base_dataset
    self.max_length = max_length
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    self.queries = [a for b in [[query] * 10 for query in base_dataset["query"]] for a in b]
    self.passages = [passage for passages in base_dataset["passages"] for passage in passages["passage_text"]]
    self.labels = [label for labels in base_dataset["passages"] for label in labels["is_selected"]]

  def __len__(self):
    return len(self.base_dataset)
  
  def __getitem__(self, index):
    query = self.queries[index]
    passage = self.passages[index]
    label = self.labels[index]

    inputs = self.tokenizer.encode_plus(
      query, 
      passage,
      add_special_tokens=True,
      truncation=True,
      max_length=self.max_length,
      padding="max_length",
      return_tensors="pt"
    )

    return \
      inputs["input_ids"].squeeze(0), \
      inputs["attention_mask"].squeeze(0), \
      torch.tensor(label, dtype=torch.long)



