from datasets import load_dataset
from tqdm import tqdm
import torch
import os

from utils import (
    clean_text, load, save, build_index, save_index, load_index,
    clean_and_overwrite_answer_file, reset_memory
)
from models import (
    load_encoder, load_reranker, load_generator,
    encode_query, rerank, generate_answer, device
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Settings
top_k = 10
docs_to_embed = 5000
batch_size = 8
max_query_length = 256
max_new_tokens = 100
temperature = 0.7
top_p = 0.9
answer_path = "answer.txt"

encoder_model_name = "BAAI/bge-base-en-v1.5"
reranker_model_name = "BAAI/bge-reranker-large"
generator_model_name = "deepcogito/cogito-v1-preview-llama-3B"
summarizer_model_name = "facebook/bart-large-cnn"

queries = [
    "How does photosynthesis work in plants?",
    "What were the main causes of World War II?",
    "What is the theory of general relativity?",
    "How do vaccines help prevent diseases?",
    "What are the functions of the human nervous system?"
]

# Load models (CPU or GPU if available)
gen_tok, gen_model = load_generator(generator_model_name)
enc_tok, enc_model = load_encoder(encoder_model_name)
rr_tok, rr_model = load_reranker(reranker_model_name)
sum_tok = AutoTokenizer.from_pretrained(summarizer_model_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name).to(device)

# Load or compute embeddings
embs, docs = load()
if embs is None or docs is None:
    wiki = load_dataset("wikipedia", "20220301.en", split=f"train[:{docs_to_embed}]", trust_remote_code=True)
    docs = [clean_text(ex["text"]) for ex in wiki]
    all_embs = []
    for i in tqdm(range(0, len(docs), batch_size)):
        batch = docs[i:i+batch_size]
        inputs = enc_tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = enc_model(**inputs)
            embs_batch = outputs.last_hidden_state.mean(dim=1).cpu()
            all_embs.append(embs_batch)
        reset_memory()
    embs = torch.cat(all_embs, dim=0)
    save(embs, docs)

# Build or load FAISS
if os.path.exists("faiss_index.index"):
    index = load_index()
else:
    index = build_index(embs)
    save_index(index)

# Main RAG loop
with open(answer_path, "w", encoding="utf-8") as f:
    for query in queries:
        query_emb = encode_query(query, enc_tok, enc_model, max_query_length)
        _, top_idx = index.search(query_emb.cpu().numpy(), top_k)
        candidates = [docs[i] for i in top_idx[0]]
        reranked = rerank(query, candidates, rr_tok, rr_model)
        context = " ".join(reranked[:top_k])[:2048]
        sum_inputs = sum_tok(context, return_tensors="pt", max_length=1024, truncation=True).to(device)
        with torch.no_grad():
            summary_ids = sum_model.generate(
                sum_inputs["input_ids"],
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
        summary = sum_tok.decode(summary_ids[0], skip_special_tokens=True)

        # Generate answer based on summary
        answer = generate_answer(query, summary, gen_tok, gen_model, max_new_tokens, temperature, top_p)
        f.write(f"Query: {query}\n\nAnswer: {answer}\n\n")

clean_and_overwrite_answer_file()
print(f"Finished. Cleaned answers saved to: {answer_path}")
