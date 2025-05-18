from fastapi import FastAPI, UploadFile, File, Request
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

from ocr_pipeline import process_uploaded_files
from utils import (
    clean_text, load, save, build_index, save_index, load_index,
    load_ocr_docs, clean_and_overwrite_answer_file, reset_memory
)
from models import (
    load_encoder, load_reranker, load_generator,
    encode_query, rerank, generate_answer, device
)

BASE_DIR = "./backend"

documents_and_index = os.path.join(BASE_DIR, "documents_and_index")
embedding_path = os.path.join(documents_and_index, "embeddings.npy")
document_path = os.path.join(documents_and_index, "documents.txt")
faiss_index_path = os.path.join(documents_and_index, "faiss_index.index")
ocr_docs_path = os.path.join(documents_and_index, "ocr_docs.txt")
answer_path = os.path.join(BASE_DIR, "answer.txt")
input_folder = os.path.join(BASE_DIR, "uploaded_files")
output_pages = os.path.join(BASE_DIR, "output_pages")

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_pages, exist_ok=True)
os.makedirs(documents_and_index, exist_ok=True)

top_k = 10
docs_to_embed = 5000
batch_size = 8
max_query_length = 256
max_new_tokens = 200
temperature = 0.7
top_p = 0.9

encoder_model_name = "BAAI/bge-base-en-v1.5"
reranker_model_name = "BAAI/bge-reranker-large"
generator_model_name = "deepcogito/cogito-v1-preview-llama-3B"
summarizer_model_name = "facebook/bart-large-cnn"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals initialized on startup
gen_tok = gen_model = None
enc_tok = enc_model = None
rr_tok = rr_model = None
sum_tok = sum_model = None
embs = None
docs = None
index = None
ocr_docs = []

# Keep track of processed files to avoid re-processing
processed_files = set()

public_url = None

@app.on_event("startup")
async def startup_event():
    global gen_tok, gen_model
    global enc_tok, enc_model
    global rr_tok, rr_model
    global sum_tok, sum_model
    global embs, docs, index
    global ocr_docs
    global processed_files

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

    ocr_docs = load_ocr_docs(ocr_docs_path)
    print(f"OCR docs loaded: {len(ocr_docs)}")

    # Initialize processed_files set with existing files in input_folder
    processed_files = set(os.listdir(input_folder))


class QueryRequest(BaseModel):
    query: str

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global processed_files
    global ocr_docs

    if file.filename in processed_files:
        return {"message": f"File '{file.filename}' already processed."}

    save_path = Path(input_folder) / file.filename
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    processed_files.add(file.filename)

    # Process only the newly uploaded file (pass list of file paths)
    chunks = process_uploaded_files(
        file_paths=[str(save_path)],
        output_txt=ocr_docs_path,
        output_pages=output_pages,
        chunk_size=300
    )

    ocr_docs = chunks

    return {"message": f"Processed file '{file.filename}'.", "chunks": len(chunks)}

@app.post("/query/")
def answer_query(req: QueryRequest):
    query = req.query

    global ocr_docs
    if not ocr_docs:
        ocr_docs = load_ocr_docs(ocr_docs_path)
    ocr_context = " ".join(ocr_docs) if ocr_docs else ""

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

    answer = generate_answer(query, summary, ocr_context, gen_tok, gen_model, max_new_tokens, temperature, top_p)

    with open(answer_path, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\nAnswer: {answer}\n\n")
    clean_and_overwrite_answer_file(answer_path)

    return JSONResponse(content={"answer": answer})

@app.get("/status/")
def get_status():
    return {"status": "ok"}

@app.get("/ngrok_url/")
def get_ngrok_url():
    global public_url
    if public_url is None:
        return {"url": "Not set"}
    return {"url": public_url}

@app.post("/set_ngrok_url/")
async def set_ngrok_url(req: Request):
    global public_url
    data = await req.json()
    url = data.get("url")
    if url:
        public_url = url
        return {"message": "ngrok URL updated", "url": public_url}
    else:
        return {"error": "Missing 'url' in request"}, 400
