from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import os
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from contextlib import asynccontextmanager
import asyncio
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")

from ocr_pipeline import process_uploaded_files, chunk_text
from utils import (
    clean_text, build_index, save_index, load_index,
    load_user_docs, clean_and_overwrite_answer_file, reset_memory
)
from models import (
    load_encoder, load_reranker, load_generator, load_summarizer,
    encode_query, summarize, rerank, generate_answer, device
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
grok_url_file = os.path.join(BASE_DIR, "grok_url.txt")

top_k = 5
docs_to_embed = 1000
batch_size = 8
max_query_length = 512
max_new_tokens = 500
relevance_threshold = -3
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
embs = None  # torch.Tensor
docs = None  # list of str
index = None
user_docs = []

processed_files = set()
public_url = None

async def launch_grok_and_capture_url():
    """
    Launches `grok http 8000` as a subprocess,
    reads its stdout lines asynchronously,
    extracts the public URL, writes it to grok_url.txt,
    and returns the URL.
    """
    global public_url

    # Make sure grok command is available on your system path
    proc = await asyncio.create_subprocess_exec(
        "grok", "http", "8000",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        text=True
    )

    url_pattern = re.compile(r"https://[a-zA-Z0-9.-]+\.grok\.dev")

    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        print(f"[grok] {line.strip()}")
        match = url_pattern.search(line)
        if match:
            public_url = match.group(0)
            print(f"Detected Grok public URL: {public_url}")
            with open(grok_url_file, "w", encoding="utf-8") as f:
                f.write(public_url)
            # Optionally: break after first URL detected
            break

    # Note: proc keeps running (grok tunnels traffic)
    return public_url

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

    # Load embeddings and documents if they exist
    if os.path.exists(embedding_path) and os.path.exists(document_path):
        print("Loading embeddings and documents from disk...")
        embs_np = np.load(embedding_path)
        embs = torch.tensor(embs_np)
        with open(document_path, "r", encoding="utf-8") as f:
            docs = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(docs)} documents and embeddings.")
    else:
        print("Embeddings or docs missing, building from Wikipedia dataset...")
        wiki = load_dataset("wikipedia", "20220301.en", split=f"train[:{docs_to_embed}]", trust_remote_code=True)

        docs = []
        all_embs = []
        max_tokens = 512
        stride = 256

        print("Embedding Wikipedia documents...")
        for ex in tqdm(wiki, desc="Processing Wikipedia docs"):
            raw_text = clean_text(ex["text"])

            # Word-level chunking (max 100 words)
            word_chunks = chunk_text(raw_text, max_words=100)

            for chunk_text_piece in word_chunks:
                tokens = enc_tok.encode(chunk_text_piece, add_special_tokens=False)
                for i in range(0, len(tokens), stride):
                    chunk_tokens = tokens[i:i + max_tokens]
                    if len(chunk_tokens) < 10:
                        continue

                    chunk_tokens_tensor = torch.tensor([chunk_tokens]).to(device)
                    attention_mask = torch.ones_like(chunk_tokens_tensor).to(device)

                    with torch.no_grad():
                        outputs = enc_model(input_ids=chunk_tokens_tensor, attention_mask=attention_mask)
                        emb = outputs.last_hidden_state.mean(dim=1).cpu()

                    all_embs.append(emb)
                    chunk_decoded_text = enc_tok.decode(chunk_tokens, skip_special_tokens=True)
                    docs.append(chunk_decoded_text)

                    reset_memory()

        embs = torch.cat(all_embs, dim=0)
        # Save embeddings as numpy
        np.save(embedding_path, embs.numpy())
        with open(document_path, "w", encoding="utf-8") as f:
            f.write("\n".join(docs))

    # Load or build FAISS index
    if os.path.exists(faiss_index_path):
        print("Loading FAISS index...")
        index = load_index(faiss_index_path)
    else:
        print("Building FAISS index...")
        index = build_index(embs)
        save_index(index, faiss_index_path)

    # Load user docs from disk
    user_docs = load_user_docs(user_docs_path)
    print(f"Loaded {len(user_docs)} user docs")

    # Track processed files to avoid re-processing
    processed_files = set(os.listdir(input_folder))

    yield  # app runs here

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

    # Process only the newly uploaded file
    user_docs = process_uploaded_files(
        file_paths=[str(save_path)],
        output_txt=user_docs_path,
        output_pages=output_pages,
        summarizer_tokenizer=summarizer_tokenizer,
        summarizer_model=summarizer_model,
    )

    return {"message": f"Processed file '{file.filename}'"}

@app.post("/query/")
def answer_query(req: QueryRequest):
    query = req.query

    global user_docs
    if not user_docs:
        user_docs = load_user_docs(user_docs_path)

    # Rerank user docs and filter by relevance threshold
    user_reranked = rerank(query, user_docs, rr_tok, rr_model)
    high_relevance_user_docs = [(doc, score) for doc, score in user_reranked if score > relevance_threshold]
    high_relevance_user_docs.sort(key=lambda x: x[1], reverse=True)
    high_relevance_docs = [doc for doc, _ in high_relevance_user_docs]

    # Take up to top_k relevant user docs for user_summary
    user_summary_docs = high_relevance_docs[:top_k]

    needed_wiki_docs = max(0, top_k - len(user_summary_docs))
    summarized_wiki_docs = []

    if needed_wiki_docs > 0:
        # Get query embedding
        query_emb = encode_query(query, enc_tok, enc_model, max_query_length)
        _, top_idx = index.search(query_emb.cpu().numpy(), top_k)
        wiki_candidates = [docs[i] for i in top_idx[0]]

        # Rerank wiki candidates
        wiki_reranked = rerank(query, wiki_candidates, rr_tok, rr_model)
        # Filter by relevance threshold
        wiki_reranked = [(doc, score) for doc, score in wiki_reranked if score > relevance_threshold]
        # Sort descending by score
        wiki_reranked.sort(key=lambda x: x[1], reverse=True)
        # Pick top needed wiki docs after threshold filtering
        top_wiki_docs = [doc for doc, _ in wiki_reranked[:needed_wiki_docs]]

        # Summarize wiki docs
        summarized_wiki_docs = [
            summarize(doc[:2048], sum_tok, sum_model, max_input_len=1024, max_output_len=150)
            for doc in top_wiki_docs if doc.strip()
        ]

    wiki_context = "\n\n".join(summarized_wiki_docs)
    user_summary = "\n\n".join(user_summary_docs)

    if not user_summary_docs and not summarized_wiki_docs:
        return JSONResponse(content={"answer": "Sorry, I couldn't find any relevant information to answer that."})

    answer = generate_answer(query, wiki_context, user_summary, gen_tok, gen_model, max_new_tokens, temperature, top_p)

    with open(answer_path, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\nAnswer: {answer}\n\n")

    clean_and_overwrite_answer_file(answer_path)

    return JSONResponse(content={"answer": answer})

@app.get("/status/")
def get_status():
    return {"status": "ok"}
