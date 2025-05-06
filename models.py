import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForCausalLM, AutoModelForSeq2SeqLM
)

# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_encoder(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device)
    return tokenizer, model

def load_reranker(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
    return tokenizer, model

def load_summarizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
    return tokenizer, model

def load_generator(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model

def encode_query(query, tokenizer, model, max_query_length):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=max_query_length).to(device)
    with torch.no_grad():
        embeddings = model.base_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def rerank(query, candidates, tokenizer, model):
    inputs = [tokenizer(query, doc, return_tensors="pt", padding=True, truncation=True).to(device) for doc in candidates]
    scores = [model(**input).logits.softmax(dim=-1).max().item() for input in inputs]
    ranked = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]
    return ranked

def summarize(text, tokenizer, model, max_input_len=1024, max_output_len=150):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_len).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_output_len,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_answer(query, context, tokenizer, model, max_new_tokens, temperature, top_p):
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
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
