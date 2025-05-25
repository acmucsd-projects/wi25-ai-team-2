from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from contextlib import asynccontextmanager

if torch.cuda.is_available():
	print("CUDA is available!")
	print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
	print("CUDA is not available.")

from ocr_pipeline import process_uploaded_files
from utils import (
	clean_text, load, save, build_index, save_index, load_index,
	load_user_docs, clean_and_overwrite_answer_file, reset_memory
)
from models import (
	load_encoder, load_reranker, load_generator,load_summarizer,
	encode_query, rerank, generate_answer, device
)

BASE_DIR = "."

documents_and_index = os.path.join(BASE_DIR, "documents_and_index")
embedding_path = os.path.join(documents_and_index, "embeddings.npy")
document_path = os.path.join(documents_and_index, "documents.txt")
faiss_index_path = os.path.join(documents_and_index, "faiss_index.index")
user_docs_path = os.path.join(documents_and_index, "user_docs.txt")
answer_path = os.path.join(BASE_DIR, "answer.txt")
input_folder = os.path.join(BASE_DIR, "uploaded_files")
output_pages = os.path.join(BASE_DIR, "output_pages")

top_k = 5
docs_to_embed = 1000
batch_size = 8
max_query_length = 256
max_new_tokens = 100
temperature = 0.7
top_p = 0.9

encoder_model_name = "BAAI/bge-base-en-v1.5"
reranker_model_name = "BAAI/bge-reranker-large"
generator_model_name = "deepcogito/cogito-v1-preview-llama-3B"
summarizer_model_name = "facebook/bart-large-cnn"

# Globals initialized on startup
gen_tok = gen_model = None
enc_tok = enc_model = None
rr_tok = rr_model = None
sum_tok = sum_model = None
summarizer_tokenizer, summarizer_model = load_summarizer(summarizer_model_name)
embs = None
docs = None
index = None
user_docs = []

# Keep track of processed files to avoid re-processing
processed_files = set()

public_url = None

@asynccontextmanager
async def lifespan(app: FastAPI):
	global gen_tok, gen_model
	global enc_tok, enc_model
	global rr_tok, rr_model
	global sum_tok, sum_model
	global embs, docs, index
	global user_docs
	global processed_files
	
	os.makedirs(input_folder, exist_ok=True)
	os.makedirs(output_pages, exist_ok=True)
	os.makedirs(documents_and_index, exist_ok=True)

	print("Loading models...")
	gen_tok, gen_model = load_generator(generator_model_name)
	enc_tok, enc_model = load_encoder(encoder_model_name)
	rr_tok, rr_model = load_reranker(reranker_model_name)
	sum_tok = AutoTokenizer.from_pretrained(summarizer_model_name)
	sum_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name).to(device)

	print("Loading embeddings and documents...")
	embs, docs = load(embedding_path, document_path)

	if embs is None or docs is None:
		print("Embeddings or docs missing, building from Wikipedia dataset...")
		wiki = load_dataset("wikipedia", "20220301.en", split=f"train[:{docs_to_embed}]", trust_remote_code=True)
		docs = [clean_text(ex["text"]) for ex in wiki]
		all_embs = []
		for i in tqdm(range(0, len(docs), batch_size), desc="Embedding batches"):
			batch = docs[i:i+batch_size]
			inputs = enc_tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
			with torch.no_grad():
				outputs = enc_model(**inputs)
				embs_batch = outputs.last_hidden_state.mean(dim=1).cpu()
				all_embs.append(embs_batch)
			reset_memory()
		embs = torch.cat(all_embs, dim=0)
		save(embs, docs, embedding_path, document_path)
	else:
		print(f"Loaded {len(docs)} documents and embeddings.")

	if os.path.exists(faiss_index_path):
		print("Loading FAISS index...")
		index = load_index(faiss_index_path)
	else:
		print("Building FAISS index...")
		index = build_index(embs)
		save_index(index, faiss_index_path)

	user_docs = load_user_docs(user_docs_path)
	print(f"User docs loaded")

	# Initialize processed_files set with existing files in input_folder
	processed_files = set(os.listdir(input_folder))

	yield  # Application runs here

# Create FastAPI app with lifespan handler
app = FastAPI(lifespan=lifespan)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

class QueryRequest(BaseModel):
	query: str

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
	global processed_files
	global user_docs

	if file.filename in processed_files:
		return {"message": f"File '{file.filename}' already processed."}

	save_path = Path(input_folder) / file.filename
	with save_path.open("wb") as buffer:
		shutil.copyfileobj(file.file, buffer)
	processed_files.add(file.filename)

	# Process only the newly uploaded file (pass list of file paths)
	user_docs = process_uploaded_files(
		file_paths=[str(save_path)],
		output_txt=user_docs_path,
		output_pages=output_pages,
		summarizer_tokenizer=summarizer_tokenizer,
		summarizer_model=summarizer_model
	)

	return {"message": f"Processed file '{file.filename}'"}

@app.post("/query/")
def answer_query(req: QueryRequest):
	query = req.query

	global user_docs
	if not user_docs:
		user_docs = load_user_docs(user_docs_path)

	# --- Step 1: Rerank user docs and filter by score ---
	user_reranked = rerank(query, user_docs, rr_tok, rr_model)  # returns list of (doc, score)

	relevance_threshold = -2.5
	high_relevance_user_docs = [(doc, score) for doc, score in user_reranked if score > relevance_threshold]

	# Sort again by score descending
	high_relevance_user_docs.sort(key=lambda x: x[1], reverse=True)
	high_relevance_docs = [doc for doc, _ in high_relevance_user_docs]

	# Cap at top_k if too many high-score user docs
	final_docs = high_relevance_docs[:top_k]
	needed_wiki_docs = max(0, top_k - len(final_docs))

	# --- Step 2: Pad with relevant Wikipedia docs if needed ---
	if needed_wiki_docs > 0:
		query_emb = encode_query(query, enc_tok, enc_model, max_query_length)
		_, top_idx = index.search(query_emb.cpu().numpy(), top_k)  # retrieve more for reranking quality
		wiki_candidates = [docs[i] for i in top_idx[0]]

		wiki_reranked = rerank(query, wiki_candidates, rr_tok, rr_model)  # (doc, score)
		wiki_reranked.sort(key=lambda x: x[1], reverse=True)
		top_wiki_docs = [doc for doc, _ in wiki_reranked[:needed_wiki_docs]]

		final_docs.extend(top_wiki_docs)
  
	wiki_context = " ".join(final_docs)

	# --- Step 3: Summarize each user doc individually ---
	user_summary = ""
	if user_docs:
		summarized_user_docs = []
		for doc in user_docs:
			if doc.strip():
				sum_inputs = sum_tok(doc[:2048], return_tensors="pt", max_length=1024, truncation=True).to(device)
				with torch.no_grad():
					summary_ids = sum_model.generate(
						sum_inputs["input_ids"],
						max_length=256,
						num_beams=4,
						early_stopping=True
					)
				summary = sum_tok.decode(summary_ids[0], skip_special_tokens=True)
				summarized_user_docs.append(summary)
		user_summary = " ".join(summarized_user_docs)
		
	answer = generate_answer(query, wiki_context, user_summary, gen_tok, gen_model, max_new_tokens, temperature, top_p)

	with open(answer_path, "w", encoding="utf-8") as f:
		f.write(f"Query: {query}\n\nAnswer: {answer}\n\n")
	clean_and_overwrite_answer_file(answer_path)

	return JSONResponse(content={"answer": answer})

@app.get("/status/")
def get_status():
	return {"status": "ok"}
