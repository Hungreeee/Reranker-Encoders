from tqdm import tqdm
import torch
from torcheval.metrics.functional import reciprocal_rank

def train(model, train_loader, val_loader, criterion, optimizer, num_epoch, device="cpu", eval=True):
  model.to(device)
  train_loss = []
  
  for epoch in tqdm(range(num_epoch)):
    model.train()
    total_loss = 0

    for input_ids, attention_mask, labels in train_loader:
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)
      labels = labels.to(device)

      outputs = model(input_ids, attention_mask)
      loss = criterion(outputs.squeeze(-1), labels.float())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_loss.append(avg_train_loss)

    if eval:
      avg_val_loss = evaluate(model, val_loader, criterion, device)
      print(f"Epoch {epoch + 1}/{num_epoch}: train_loss = {avg_train_loss} | val_loss = {avg_val_loss}")

    else:
      print(f"Epoch {epoch + 1}/{num_epoch}: train_loss = {avg_train_loss}")
       
  return train_loss


def evaluate(model, test_loader, criterion, device="cpu"):
  with torch.no_grad():
    model.eval()
    total_loss = 0

    for input_ids, attention_mask, labels in test_loader:
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)
      labels = labels.to(device)
         
      outputs = model(input_ids, attention_mask)
      loss = criterion(outputs.squeeze(-1), labels.float())

      total_loss += loss.item()
      
  return total_loss / len(test_loader)
