# %%
import argparse, os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from encoders import BiEncoder, CrossEncoder, PolyEncoder
from transformers import AutoModel, AutoTokenizer

from utils import *

# %%
transformer = AutoModel.from_pretrained("prajjwal1/bert-tiny")
model = BiEncoder(transformer)

# %%

train_loss = [1, 2, 3]
val_loss = [2, 9, 1]

output_dir = "./model_checkpoints/"

log_writer = open(os.path.join(output_dir, 'logs.txt'), 'a', encoding='utf-8')
log_writer.write(''.join([f'Epoch {i + 1}/{12}: Train loss = {loss[0]} | Validation loss = {loss[1]}\n' for i, loss in enumerate(zip(train_loss, val_loss))]) + '\n')

model_save_path = output_dir + "/" + ("_".join(["Aa", "Aaa"]) + ".pth")
torch.save(model, model_save_path)
print(f"Model saved to {model_save_path}.")
