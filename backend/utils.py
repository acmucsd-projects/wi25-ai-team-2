import os
import torch
import numpy as np
import faiss
import re
import gc

def reset_memory():
    gc.collect()

def clean_text(t):
    """Clean and normalize the input text."""
    return re.sub(r'\s+', ' ', t.strip())

def save(embs, docs, embedding_path, document_path):
    """Save embeddings and documents to disk."""
    np.save(embedding_path, embs.cpu().numpy())
    with open(document_path, "w", encoding="utf-8") as f:
        f.writelines(f"{doc}\n" for doc in docs)

def load(embedding_path, document_path):
    """Load embeddings and documents from disk."""
    if not os.path.exists(embedding_path) or not os.path.exists(document_path):
        return None, None
    embs = torch.tensor(np.load(embedding_path))
    with open(document_path, "r", encoding="utf-8") as f:
        docs = f.read().splitlines()
    return embs, docs

def save_index(index, faiss_index_path):
    """Save FAISS index to disk."""
    faiss.write_index(index, faiss_index_path)

def load_index(faiss_index_path):
    """Load FAISS index from disk."""
    return faiss.read_index(faiss_index_path)

def build_index(embs):
    """Build a FAISS index from embeddings."""
    embs = embs.cpu().numpy().astype("float32")
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index

def clean_and_overwrite_answer_file(file_path):
    """Clean and format the answers in the provided file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    qa_blocks = re.findall(r"Query: (.*?)\n+Answer: (.*?)(?=\n+Query:|\Z)", content, re.DOTALL)
    cleaned_output = ""
    for query, answer in qa_blocks:
        cleaned_output += f"Question: {query.strip()}\nAnswer: {answer.strip()}\n\n"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_output.strip())
    print(f"Answers cleaned and saved to: {file_path}")

def load_user_docs(user_file):
    """Load and clean documents from a file."""
    if os.path.exists(user_file):
        with open(user_file, "r", encoding="utf-8") as file:
            user_docs = file.readlines()
        return [clean_text(doc) for doc in user_docs]
    return []
