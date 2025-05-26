import torch
from transformers import (
	AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
	AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_compute_dtype=torch.float16,
	bnb_4bit_use_double_quant=True,
	bnb_4bit_quant_type="nf4"
)

def load_encoder(name, quantized=False):
	tokenizer = AutoTokenizer.from_pretrained(name)
	if quantized:
		model = AutoModel.from_pretrained(name, device_map="auto", quantization_config=bnb_config)
	else:
		model = AutoModel.from_pretrained(name).to(device)
	return tokenizer, model

def load_reranker(name, quantized=False):
	tokenizer = AutoTokenizer.from_pretrained(name)
	if quantized:
		model = AutoModelForSequenceClassification.from_pretrained(name, device_map="auto", quantization_config=bnb_config)
	else:
		model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
	return tokenizer, model

def load_summarizer(name, quantized=False):
	tokenizer = AutoTokenizer.from_pretrained(name)
	if quantized:
		model = AutoModelForSeq2SeqLM.from_pretrained(name, device_map="auto", quantization_config=bnb_config)
	else:
		model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
	return tokenizer, model

def load_generator(model_name, quantized=True):
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.pad_token = tokenizer.eos_token
	if quantized:
		model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
	else:
		model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
	return tokenizer, model

def encode_query(query, tokenizer, model, max_query_length):
	inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=max_query_length).to(device)
	with torch.no_grad():
		embeddings = model.base_model(**inputs).last_hidden_state.mean(dim=1)
	return embeddings

def rerank(query, candidates, tokenizer, model):
	inputs = [tokenizer(query, doc, return_tensors="pt", padding=True, truncation=True).to(device) for doc in candidates]
	scores = [model(**input).logits[0].item() for input in inputs]
	doc_scores = list(zip(candidates, scores))
	doc_scores.sort(key=lambda x: x[1], reverse=True)
	
	return doc_scores

def summarize(text, tokenizer, model, max_input_len=512, max_output_len=150):
	inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_len).to(device)
	print("summarizing")
	summary_ids = model.generate(
		inputs["input_ids"],
		max_length=max_output_len,
		num_beams=4,
		early_stopping=True
	)
	print("SUMMARIZED")
	return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_answer(query, wiki_context, user_context, tokenizer, model, max_new_tokens, temperature, top_p):
	input_text = f"Question: {query}\n"
	input_text += f"Most relevant information: {user_context}\n"
	input_text += f"Additional reference (Wikipedia): {wiki_context}\n"
	input_text += "Answer:"

	print(input_text)

	inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
	model.config.pad_token_id = model.config.eos_token_id
	outputs = model.generate(
		input_ids=inputs["input_ids"],
		attention_mask=inputs["attention_mask"],
		max_new_tokens=max_new_tokens,
		do_sample=True,
		temperature=temperature,
		top_p=top_p,
	)

	return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[1].strip()