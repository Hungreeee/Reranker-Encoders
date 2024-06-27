import torch.nn as nn
from transformers import AutoModel


class CrossEncoder(nn.Module):
  """
  Cross-encoder
  """
  def __init__(self, transformer_base_name: str):
    super().__init__()
    self.transformer_base = AutoModel.from_pretrained(transformer_base_name)
    self.fc = nn.Linear(self.transformer_base.config.hidden_size, 1)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, input_ids, attention_mask):
    outputs = self.transformer_base(input_ids=input_ids, attention_mask=attention_mask)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    logits = self.fc(cls_embeddings)
    return self.sigmoid(logits)