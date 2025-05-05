import os
import json
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_PATH = os.path.join(os.path.dirname(__file__), '../')
DATA = os.path.join(PROJECT_PATH, "data")

document_file = os.path.join(DATA, "documents2025-2.json")
output_file = os.path.join(DATA, "embeddings.json")

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Example of a sentence embedding model

def save_model(model_name):
	model = AutoModel.from_pretrained(model_name)
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	# Save both locally
	model.save_pretrained("./embedding_model")
	tokenizer.save_pretrained("./embedding_model")

def put_embeddings():
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

	with open(document_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding="utf-8") as outfile:
		line_num = 0
		for line in infile:
			data = json.loads(line)

			inputs = tokenizer(data.get("text")[:400], return_tensors="pt", truncation=True, padding=True)
			embedding = get_embedding(inputs)

			data["embedding"] = embedding

			outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

			line_num += 1
			if line_num % 1000 == 0:
				print(f"Processed {line_num} lines...")

#save_model(MODEL_NAME)
put_embeddings()
