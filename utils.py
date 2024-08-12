import torch
from tqdm import tqdm
import configparser
from datasets import Dataset, load_dataset
from dataset import TextPairDataset
from transformers import AutoTokenizer

from encoders import BaseEncoder


def count_parameters(model_parameters):
  	return sum(p.numel() for p in model_parameters if p.requires_grad)


def train(
	model: BaseEncoder, 
	trainloader, 
	valloader, 
	criterion, 
	optimizer, 
	num_epoch: int, 
	device: str = "cpu", 
	eval: bool = True
):
	model.to(device)
	train_loss = []
	val_loss = []
	
	for epoch in tqdm(range(num_epoch)):
		model.train()
		total_loss = 0

		for query_input_ids, query_attention_masks, \
			context_input_ids, context_attention_masks, \
				labels in trainloader:
			
			query_input_ids = query_input_ids.to(device)
			query_attention_masks = query_attention_masks.to(device)
			context_input_ids = context_input_ids.to(device)
			context_attention_masks = context_attention_masks.to(device)
			labels = labels.to(device)

			outputs = model(query_input_ids, query_attention_masks, context_input_ids, context_attention_masks, labels)
			loss = criterion(outputs, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item()

			avg_train_loss = total_loss / len(trainloader)
			train_loss.append(avg_train_loss)

		if eval:
			avg_val_loss = evaluate(model, valloader, criterion, device)
			val_loss.append(avg_val_loss)
			print(f"Epoch {epoch + 1}/{num_epoch}: train_loss = {avg_train_loss} | val_loss = {avg_val_loss}")
		else:
			print(f"Epoch {epoch + 1}/{num_epoch}: train_loss = {avg_train_loss}")
		
	return train_loss, val_loss 


def evaluate(
	model: BaseEncoder, 
	testloader, 
	criterion, 
	device: str = "cpu"
):
	with torch.no_grad():
		model.eval()
		total_loss = 0

		for query_input_ids, query_attention_mask, \
			context_input_ids, context_attention_mask, \
				labels in testloader:
			
			query_input_ids = query_input_ids.to(device)
			query_attention_mask = query_attention_mask.to(device)
			context_input_ids = context_input_ids.to(device)
			context_attention_mask = context_attention_mask.to(device)
			labels = labels.to(device)
				
			outputs = model(query_input_ids, query_attention_mask, context_input_ids, context_attention_mask, labels)
			loss = criterion(outputs, labels)

			total_loss += loss.item()
		
	return total_loss / len(testloader)


def load_huggingface_dataset(
	dataset_object: str, 
	dataset_config: dict, 
	tokenizer: AutoTokenizer,
	max_seq_length: int = 512
):
	raw_trainset, raw_valset = load_dataset(dataset_object, split=("train", "validation"), **dataset_config)
	text_pair_trainset = TextPairDataset(raw_trainset, tokenizer, max_seq_length)
	text_pair_valset = TextPairDataset(raw_valset, tokenizer, max_seq_length)

	return text_pair_trainset, text_pair_valset


def read_config_file(path: str):
	dataset_configs = configparser.ConfigParser()
	dataset_configs.read(path)

	dataset_configs_dict = {}
	for section in dataset_configs.sections():
		dataset_configs_dict[section] = {}
		for option in dataset_configs.options(section):
			dataset_configs_dict[section][option] = dataset_configs.get(section, option)

	return dataset_configs_dict
