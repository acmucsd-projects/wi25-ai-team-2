from datasets import load_dataset
from tqdm import tqdm
import torch
import os

from ocr_pipeline import process_uploaded_files
from utils import (
    clean_text, load, save, build_index, save_index, load_index,
    load_ocr_docs, clean_and_overwrite_answer_file, reset_memory
)
from models import (
    load_encoder, load_reranker, load_generator,
    encode_query, rerank, generate_answer, device
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
embedding_path = os.path.join(BASE_DIR, "documents_and_index/embeddings.npy")
document_path = os.path.join(BASE_DIR, "documents_and_index/documents.txt")
faiss_index_path = os.path.join(BASE_DIR, "documents_and_index/faiss_index.index")
ocr_docs_path = os.path.join(BASE_DIR, "documents_and_index/ocr_docs.txt")
answer_path = os.path.join(BASE_DIR, "answer.txt")
input_folder = os.path.join(BASE_DIR, "uploaded_files")
output_pages = os.path.join(BASE_DIR, "output_pages")

# Settings
top_k = 10
docs_to_embed = 5000
batch_size = 8
max_query_length = 512
max_new_tokens = 200
temperature = 0.7
top_p = 0.9

encoder_model_name = "BAAI/bge-base-en-v1.5"
reranker_model_name = "BAAI/bge-reranker-large"
generator_model_name = "deepcogito/cogito-v1-preview-llama-3B"
summarizer_model_name = "facebook/bart-large-cnn"

queries = [
    "How to find the surface integral of a vector field and explain the orientation of a surface?"
]

# Load models (CPU or GPU if available)
gen_tok, gen_model = load_generator(generator_model_name)
enc_tok, enc_model = load_encoder(encoder_model_name)
rr_tok, rr_model = load_reranker(reranker_model_name)
sum_tok = AutoTokenizer.from_pretrained(summarizer_model_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name).to(device)

# Load or compute embeddings
embs, docs = load(embedding_path, document_path)
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
    save(embs, docs, embedding_path, document_path)

# Build or load FAISS
if os.path.exists(faiss_index_path):
    index = load_index(faiss_index_path)
else:
    index = build_index(embs)
    save_index(index, faiss_index_path)

# Load OCR docs (if any)
ocr_docs = load_ocr_docs(ocr_docs_path)

# If no OCR docs are loaded, process uploaded files to generate OCR content
if not ocr_docs:
    print("No OCR documents found. Processing uploaded files...")
    # Process files (assuming OCR docs are saved as chunks in ocr_docs.txt)
    chunks = process_uploaded_files(input_folder, output_txt=ocr_docs_path, output_pages=output_pages, chunk_size=300)
    ocr_docs = chunks  # Load the OCR processed docs into ocr_docs

ocr_context = " ".join(ocr_docs) if len(ocr_docs) > 0 else ""

# Main RAG loop
with open(answer_path, "w", encoding="utf-8") as f:
    for query in queries:
        # Get top-k docs based on wiki context
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

        # Generate answer based on both contexts (Wiki and OCR)
        answer = generate_answer(query, summary, ocr_context, gen_tok, gen_model, max_new_tokens, temperature, top_p)
        f.write(f"Query: {query}\n\nAnswer: {answer}\n\n")

clean_and_overwrite_answer_file(answer_path)
print(f"Finished. Cleaned answers saved to: {answer_path}")
