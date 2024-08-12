import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class BaseEncoder(nn.Module):
	def __init__(
		self,
		transformer: Any,
		pooling = "cls",
		device = "cpu"
	):
		super(BaseEncoder, self).__init__()
		self.transformer = transformer
		self.pooling = pooling
		self.device = device

	
	def cosine_similarity(self, query_embeddings, context_embeddings, batch_size):
		normalized_query_embeddings = query_embeddings / torch.norm(query_embeddings, dim=1, keepdim=True)
		normalized_cotext_embeddings = context_embeddings / torch.norm(context_embeddings, dim=1, keepdim=True)

		mask = torch.eye(batch_size).to(self.device)
		dot_product_mat = torch.matmul(normalized_query_embeddings, normalized_cotext_embeddings.T)
		dot_product = (dot_product_mat * mask).sum(dim=1)

		return dot_product

	
	def pooled_embedding(self, logits, input_masks):
		if self.pooling == "cls":
			return logits.last_hidden_state[:, 0, :]
		
		elif self.pooling == "mean":
			sum_hidden_states = torch.sum(logits.last_hidden_state, dim=1)
			num_tokens_per_seq = torch.sum(input_masks, dim=1)
			num_tokens_per_seq = torch.clamp(num_tokens_per_seq, min=1e-9)
			mean_pooled_embeddings = sum_hidden_states / num_tokens_per_seq

			return mean_pooled_embeddings
		
	
	def forward(self):
		raise NotImplementedError


class BiEncoder(BaseEncoder):
	"""
	Bi-encoder architecture
	"""
	def __init__(
		self, 
		transformer, 
		pooling = "cls", 
		device = "cpu",
		**krawgs
	):
		super(BiEncoder, self).__init__(transformer, pooling, device)
		
		
	def forward(
		self, 
		query_input_ids, 
		query_input_masks, 
		context_input_ids, 
		context_input_masks, 
		labels=None
	):
		batch_size = context_input_ids.shape[0]
		query_logits = self.transformer(input_ids=query_input_ids, attention_mask=query_input_masks)
		query_embeddings = self.pooled_embedding(query_logits, query_input_ids)

		context_logits = self.transformer(input_ids=context_input_ids, attention_mask=context_input_masks)
		context_embeddings = self.pooled_embedding(context_logits, query_input_ids)

		scores = self.cosine_similarity(query_embeddings, context_embeddings, batch_size)
		return scores
	

class CrossEncoder(BaseEncoder):
	def __init__(
		self, 
		transformer, 
		pooling = "cls", 
		device = "cpu",
		**krawgs
	):
		super(CrossEncoder, self).__init__(transformer, pooling, device)
		self.linear = nn.Linear(self.transformer.config.hidden_size, 1)

	
	def forward(
		self, 
		query_input_ids, 
		query_input_masks, 
		context_input_ids, 
		context_input_masks, 
		labels=None
	):
		context_input_ids = context_input_ids[:, 1:]
		context_input_masks = context_input_masks[:, 1:]

		pair_input_ids = torch.cat((query_input_ids, context_input_ids), dim=1).to(self.device)
		pair_input_masks = torch.cat((query_input_masks, context_input_masks), dim=1).to(self.device)

		logits = self.transformer(input_ids=pair_input_ids, attention_mask=pair_input_masks)
		pair_embeddings = self.pooled_embedding(logits, pair_input_masks)

		scores = self.linear(pair_embeddings)
		return scores


class PolyEncoder(BaseEncoder):
	def __init__(
		self, 
		transformer, 
		num_global_features,
		pooling = "cls", 
		device = "cpu",
		**krawgs
	):
		super(PolyEncoder, self).__init__(transformer, pooling, device)

		self.num_global_features = num_global_features
		self.embedding = nn.Embedding(self.num_global_features, self.transformer.config.hidden_size)


	@staticmethod
	def dot_product_attention(query, key, value, mask=None):
		scaled_qk = torch.matmul(query, key.transpose(2, 1)) 

		if mask is not None:
			scaled_qk = scaled_qk.masked_fill(mask == 0, float('-inf'))
	
		attention_score = torch.matmul(F.softmax(scaled_qk), value)
		return attention_score
		

	def forward(
		self,
		query_input_ids, 
		query_input_masks, 
		context_input_ids, 
		context_input_masks, 
		labels=None
	):
		batch_size = query_input_ids.shape[0]

		query_logits = self.transformer(query_input_ids, query_input_masks)
		query_embeddings = self.pooled_embedding(query_logits, query_input_masks).unsqueeze(1)

		context_logits = self.transformer(context_input_ids, context_input_masks).last_hidden_state

		context_codes_ids = torch.arange(self.num_global_features).to(self.device) 
		context_codes_ids = context_codes_ids.unsqueeze(0).expand(batch_size, -1)
		context_codes_embeddings = self.embedding(context_codes_ids)

		global_feat_attended_context = self.dot_product_attention(context_codes_embeddings, context_logits, context_logits)
		context_embeddings = self.dot_product_attention(query_embeddings, global_feat_attended_context, global_feat_attended_context)
		
		scores = self.cosine_similarity(query_embeddings, context_embeddings, batch_size)
		return scores




		



		




