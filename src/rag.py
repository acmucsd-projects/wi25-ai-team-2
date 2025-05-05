import numpy as np
from flashrank import Ranker, RerankRequest
import json
import os
import torch
import heapq
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util #, SentenceTransformer

PROJECT_PATH = os.path.join(os.path.dirname(__file__), '../')
DATA = os.path.join(PROJECT_PATH, "data")

document_file = os.path.join(DATA, "documents2025-2.json")
embedding_file = os.path.join(DATA, "embeddings.json")

PROMPT = "What is the powerhouse of the cell?"
TOP_K = 10
TOP_P = 3


def top_k(prompt, k):
	with open(embedding_file, 'r', encoding='utf-8') as f:
		model = AutoModel.from_pretrained("./embedding_model")
		tokenizer = AutoTokenizer.from_pretrained("./embedding_model")

		def get_embedding(inputs):
			with torch.no_grad():
				outputs = model(**inputs)

			# Mean pooling
			token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
			attention_mask = inputs['attention_mask']

			input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
			sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
			sum_mask = input_mask_expanded.sum(dim=1)
			mean_pooled = sum_embeddings / sum_mask

			return mean_pooled[0].tolist()  # Convert tensor to list
		
		inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
		prompt_embedding = get_embedding(inputs)

		top_k_heap = []
		i = -1
		for line in f:
			i += 1
			try:
				doc = json.loads(line)

				text_embedding = doc.get("embedding")
				similarity = util.pytorch_cos_sim(prompt_embedding, text_embedding).item()

				if len(top_k_heap) < k:
					heapq.heappush(top_k_heap, (similarity, doc))
				else:
					heapq.heappushpop(top_k_heap, (similarity, doc))

			except json.JSONDecodeError:
				print(str(i) + ": JSON Decode Error")
				continue
		
		return top_k_heap
	return

def rerank(prompt, k): 
	ranker = Ranker(max_length=128)

	docs = top_k(prompt, k)
	if docs is None:
		print("[Error] No documents found")
		return
	
	passages = [t[1] for t in docs]

	rerankrequest = RerankRequest(query=prompt, passages=passages)
	results = ranker.rerank(rerankrequest)

	# https://github.com/PrithivirajDamodaran/FlashRank
	return results

def run():
	print("Started")

	model_name = "deepseek-ai/deepseek-llm-7b-base"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(
		model_name, 
		torch_dtype=torch.bfloat16, 
		device_map="auto", 
		offload_folder="./offload"
	)
	model.generation_config = GenerationConfig.from_pretrained(model_name)
	model.generation_config.pad_token_id = model.generation_config.eos_token_id

	docs = rerank(PROMPT, TOP_K)

	query = "Relevant documents: \n\n"
	for i in range(TOP_P):
		query += str(-(i+1)) + ": " + docs[i].get("text")[:3000] + "... \n\n"
	query += "User question: " + PROMPT + "\n\n"
	print(query)

	inputs = tokenizer(query, return_tensors="pt")
	outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

	# QUERY THE LLM
	result = tokenizer.decode(outputs[0], skip_special_tokens=True)
	print(result)
	
run()
