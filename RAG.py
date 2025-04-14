import os
import torch
import numpy as np
from transformers import (
    DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRContextEncoder,
    AutoTokenizer, AutoModelForCausalLM
)
from datasets import load_dataset
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# === Config ===
config = {
    "num_samples": 1000,
    "question_encoder_model": "facebook/dpr-question_encoder-single-nq-base",
    "context_encoder_model": "facebook/dpr-ctx_encoder-single-nq-base",
    "gpt_neo_model": "EleutherAI/gpt-neo-1.3B",
    "k": 5,
    "max_new_tokens": 60,
    "encoded_passages_path": './encoded_passages.npy',
    "faiss_index_path": './faiss_index.index',
    "generated_answer_path": './generated_answer.txt',
}

# === Step 1: Clean slate
if os.path.exists(config["generated_answer_path"]):
    os.remove(config["generated_answer_path"])

# === Step 2: Load dataset (AG News dataset always used here)
print("Loading dataset...")
try:
    # Load the "ag_news" dataset directly
    dataset = load_dataset("ag_news", split="train[:1000]")
except Exception as e:
    print(f"Error loading 'ag_news' dataset: {e}")
    dataset = []  # Empty dataset if loading fails, but shouldn't reach here

# Check structure of dataset and adjust
passages = []
for sample in dataset:
    if 'text' in sample:
        passages.append(sample['text'])
    elif 'content' in sample:
        passages.append(sample['content'])
    else:
        print("Unexpected structure in dataset sample:", sample)
        passages.append(str(sample))  # Fallback to whole sample as string

# === Step 3: Load models and tokenizers
print("Loading models...")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(config["question_encoder_model"])
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(config["context_encoder_model"])
question_encoder = DPRQuestionEncoder.from_pretrained(config["question_encoder_model"])
context_encoder = DPRContextEncoder.from_pretrained(config["context_encoder_model"])
gpt_tokenizer = AutoTokenizer.from_pretrained(config["gpt_neo_model"])
gpt_model = AutoModelForCausalLM.from_pretrained(config["gpt_neo_model"])
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token  # Patch padding

# === Step 4: Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_encoder.to(device)
context_encoder.to(device)
gpt_model.to(device)

# === Step 5: Encode passages
def encode_passages(passages):
    embeddings = []
    for passage in tqdm(passages, desc="Encoding Passages", unit="passage"):
        inputs = context_tokenizer(passage, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = context_encoder(**inputs).pooler_output
        embeddings.append(output.cpu().numpy().flatten())
    return np.array(embeddings)

if os.path.exists(config["encoded_passages_path"]):
    passage_embeddings = np.load(config["encoded_passages_path"])
    print("Loaded encoded passages.")
else:
    passage_embeddings = encode_passages(passages)
    np.save(config["encoded_passages_path"], passage_embeddings)

# === Step 6: FAISS index
if os.path.exists(config["faiss_index_path"]):
    index = faiss.read_index(config["faiss_index_path"])
    print("Loaded FAISS index.")
else:
    index = faiss.IndexFlatL2(passage_embeddings.shape[1])
    index.add(passage_embeddings)
    faiss.write_index(index, config["faiss_index_path"])

# === Step 7: Query
query = "What is Taj Mahal?"
print("Query:", query)
q_inputs = question_tokenizer(query, return_tensors='pt')
q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
with torch.no_grad():
    q_embedding = question_encoder(**q_inputs).pooler_output.cpu().numpy().flatten()

# Retrieve top-k and inspect full passages
_, top_indices = index.search(np.array([q_embedding]), config["k"])

# Debugging: Check the shape and values of `top_indices`
print(f"Top indices shape: {top_indices.shape}")
print(f"Top indices: {top_indices}")

# Ensure top_indices does not exceed the number of passages
valid_top_indices = []

# Validate and filter out invalid indices
for idx in top_indices[0]:
    if 0 <= idx < len(passages):
        valid_top_indices.append(idx)
    else:
        print(f"Invalid index found: {idx}")

# Retrieve the valid top-k passages
retrieved_passages = [passages[i] for i in valid_top_indices]

print("\nTop-k Retrieved Passages (Full):")
for idx, p in enumerate(retrieved_passages):
    print(f"[{idx}] {p}")

# === Step 9: Rerank using Sentence Transformers
print("Reranking using Sentence Transformers...")
reranker = SentenceTransformer('all-MiniLM-L6-v2')
q_embedding = reranker.encode(query)  # Replace DPR output
retrieved_embeddings = reranker.encode(retrieved_passages, show_progress_bar=True)
similarities = cosine_similarity([q_embedding], retrieved_embeddings)[0]
reranked = [retrieved_passages[i] for i in similarities.argsort()[::-1]]

# === Step 10: Create context
context = " ".join(reranked[:3])

print("\nFull Context Used:\n")
for i, p in enumerate(reranked[:3]):
    print(f"[{i}] {p}")

# === Step 11: Generate answer with progress bar (Optimized)

prompt = f"Answer the question: {query} using the context below. Please provide reasoning before answering.\nContext: {context}"

inputs = gpt_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
attention_mask = inputs['attention_mask']

# Reduce the number of tokens generated for faster results
config["max_new_tokens"] = 30  # Reduced from 60 to 30

# Add progress bar here for generating the final answer
print("\nGenerating final answer with progress...")

# Initialize progress bar for generating tokens
total_steps = config["max_new_tokens"]  # Set to max_new_tokens to update after each token
progress_bar = tqdm(total=total_steps, desc="Answer Generation", unit="token")

generated_text = inputs['input_ids']
attention_mask = inputs['attention_mask']

with torch.no_grad():
    for step in range(config["max_new_tokens"]):
        # Generate one token at a time
        output = gpt_model.generate(
            input_ids=generated_text,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=gpt_tokenizer.eos_token_id
        )
        
        # Add generated token to the current sequence
        generated_text = torch.cat((generated_text, output[:, -1:]), dim=-1)
        
        # Update the attention mask for the new token generated
        attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.size(0), 1), device=device)), dim=1)
        
        # Update progress bar after each token
        progress_bar.update(1)

# Decode the final generated text
generated_text = gpt_tokenizer.decode(generated_text[0], skip_special_tokens=True)

# Finalize the progress bar once done
progress_bar.close()

# === Step 12: Save and print
with open(config["generated_answer_path"], "w") as f:
    f.write(generated_text)
print("\nGenerated Answer:\n", generated_text)
