import os
import torch
import numpy as np
import faiss
import re
import gc

embedding_path = "embeddings.npy"
document_path = "documents.txt"
faiss_index_path = "faiss_index.index"

def reset_memory():
    gc.collect()

def clean_text(t):
    return re.sub(r'\s+', ' ', t.strip())

def save(embs, docs):
    np.save(embedding_path, embs.cpu().numpy())
    with open(document_path, "w", encoding="utf-8") as f:
        f.writelines(f"{doc}\n" for doc in docs)

def load():
    if not os.path.exists(embedding_path):
        return None, None
    embs = torch.tensor(np.load(embedding_path))
    docs = open(document_path).read().splitlines()
    return embs, docs

def save_index(index):
    faiss.write_index(index, faiss_index_path)

def load_index():
    return faiss.read_index(faiss_index_path)

def build_index(embs):
    embs = embs.cpu().numpy().astype("float32")
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index

def clean_and_overwrite_answer_file(file_path="answer.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    qa_blocks = re.findall(r"Query: (.*?)\n+Answer: (.*?)(?=\n+Query:|\Z)", content, re.DOTALL)
    cleaned_output = ""
    for query, answer in qa_blocks:
        cleaned_output += f"Question: {query.strip()}\nAnswer: {answer.strip()}\n\n"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_output.strip())
    print(f"Answers cleaned and saved to: {file_path}")
