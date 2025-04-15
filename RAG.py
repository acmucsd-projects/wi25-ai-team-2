import os
import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)
import faiss
import re
import warnings
import numpy as np
import multiprocessing
from sklearn.decomposition import PCA
from tqdm import tqdm

# Setup
multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", message="resource_tracker: There appear to")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EMBEDDING_DIM = 300
TOP_K = 20
DOCS_TO_EMBED = 10000
CONFIDENCE_THRESHOLD = 0.7
QUERIES = [
    "What is the largest planet in our solar system?",
    "How does a computer virus spread?",
    "What is the theory of quantum mechanics?",
    "Who invented the telephone?",
    "What are the main causes of climate change?",
    "Explain the process of cellular respiration.",
    "What is the difference between speed and velocity?",
    "Who was the first person to walk on the moon?",
    "What are the benefits of exercise?",
    "How do black holes form?",
    "What is the significance of the Magna Carta?",
    "What is the fastest animal in the world?",
    "What are the properties of water?",
    "Explain the process of natural selection.",
    "Who painted the Mona Lisa?",
    "What is the difference between an asteroid and a comet?",
    "How does the internet work?",
    "What is the significance of the discovery of penicillin?",
    "What is the role of mitochondria in cells?",
    "How does a nuclear reactor work?"
]

# Configuration
CONFIG = {
    "encoder_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "reranker_model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "generator_model_name": "google/flan-t5-large",
    "faiss_index_name": "faiss_index.index",
    "embeddings_name": "encoded_passages.npy",
    "documents_name": "clean_documents.txt",
    "retriever_top_k": 5,
    "embedding_dim": 384,
    "num_beams": 1,
    "top_k": 0,
    "top_p": 1.0,
    "temperature": 0.7,
    "use_dense_retrieval": True,
    "use_sparse_retrieval": False,
    "log_retrieval_steps": False,
    "log_generation_steps": False,
    "answer_post_processing": True,
}


# Paths
EMBEDDINGS_PATH = CONFIG["embeddings_name"]
DOCUMENTS_PATH = CONFIG["documents_name"]
FAISS_INDEX_PATH = CONFIG["faiss_index_name"]

logging.disable(logging.CRITICAL)
logger = logging.getLogger()

def load_data():
    try:
        dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]", trust_remote_code=True)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def load_models():
    try:
        encoder_tokenizer = AutoTokenizer.from_pretrained(CONFIG["encoder_model_name"])
        encoder_model = AutoModel.from_pretrained(CONFIG["encoder_model_name"]).to(device)

        reranker_tokenizer = AutoTokenizer.from_pretrained(CONFIG["reranker_model_name"])
        reranker = AutoModelForSequenceClassification.from_pretrained(CONFIG["reranker_model_name"]).to(device)

        generator_tokenizer = AutoTokenizer.from_pretrained(CONFIG["generator_model_name"])
        generator = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["generator_model_name"]).to(device)

        return encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def sanitize_context(text):
    text = re.sub(r'-lrb-', '(', text)
    text = re.sub(r'-rrb-', ')', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'\"()\-]', '', text)
    return text.strip()

def save_embeddings_and_documents(embeddings, documents):
    np.save(EMBEDDINGS_PATH, embeddings.cpu().numpy())
    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        f.writelines([f"{doc}\n" for doc in documents])

def load_embeddings_and_documents():
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(DOCUMENTS_PATH):
        return None, None
    embeddings = torch.tensor(np.load(EMBEDDINGS_PATH))
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f]
    return embeddings, documents

def save_faiss_index(index):
    faiss.write_index(index, FAISS_INDEX_PATH)

def load_faiss_index():
    return faiss.read_index(FAISS_INDEX_PATH) if os.path.exists(FAISS_INDEX_PATH) else None

def reduce_embeddings(embeddings, new_dim=300):
    pca = PCA(n_components=min(new_dim, embeddings.shape[0], embeddings.shape[1]))
    reduced = pca.fit_transform(embeddings.cpu().numpy())
    return torch.tensor(reduced)

def generate_embeddings(docs, encoder_tokenizer, encoder_model, num_docs):
    embeddings, clean_docs = [], []
    for doc in docs[:num_docs]:
        if not doc or not isinstance(doc, str): continue
        try:
            cleaned = clean_text(doc)
            inputs = encoder_tokenizer(f"passage: {cleaned}", return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            with torch.no_grad():
                emb = encoder_model(**inputs).last_hidden_state[:, 0, :]
            embeddings.append(emb.squeeze(0).cpu())
            clean_docs.append(cleaned)
        except Exception as e:
            print(f"Embedding error: {e}")
    embeddings = torch.stack(embeddings) if embeddings else None
    return embeddings, clean_docs

def create_faiss_index(embeddings):
    if embeddings.ndimension() != 2:
        raise ValueError(f"Embeddings should have shape (num_docs, embedding_dim), but got {embeddings.shape}")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    try:
        index.add(embeddings.cpu().numpy())
    except Exception as e:
        print(f"Error adding embeddings to FAISS index: {e}")
        return None
    return index

def rerank(query, docs, reranker_tokenizer, reranker, top_n=TOP_K):
    inputs = reranker_tokenizer([[query, doc] for doc in docs], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        scores = reranker(**inputs).logits.squeeze(-1)
    top_indices = torch.topk(scores, k=min(top_n, len(scores))).indices.tolist()
    reranked_docs = [docs[i] for i in top_indices]
    reranked_scores = [scores[i].item() for i in top_indices]
    return [(doc, score) for doc, score in zip(reranked_docs, reranked_scores) if score >= CONFIDENCE_THRESHOLD]

def encode_query(query, encoder_tokenizer, encoder_model, target_dim=EMBEDDING_DIM):
    query_inputs = encoder_tokenizer(f"query: {query}", return_tensors="pt", truncation=True, padding=True).to(device)
    query_embedding = encoder_model(**query_inputs).last_hidden_state[:, 0, :]
    if query_embedding.shape[1] != target_dim:
        query_embedding = query_embedding[:, :target_dim]
    return query_embedding.detach().cpu().numpy().astype("float32")

def run_rag_pipeline(query, index, docs, encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator):
    try:
        q_embedding = encode_query(query, encoder_tokenizer, encoder_model)
        k = min(TOP_K * 3, len(docs))
        distances, top_idxs = index.search(q_embedding, k=k)
        retrieved_docs = [docs[i] for i in top_idxs[0]]
        reranked_docs_with_scores = rerank(query, retrieved_docs, reranker_tokenizer, reranker)
        reranked_docs = [doc for doc, _ in reranked_docs_with_scores]
        sanitized_docs = [sanitize_context(doc) for doc in reranked_docs]
        context = " ".join(sanitized_docs)[:2048]
        input_text = f"Based on the context below, answer the question: {query}\n\nContext: {context}"
        gen_inputs = generator_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        output_ids = generator.generate(
            **gen_inputs,
            num_beams=10,
            max_length=100,
            no_repeat_ngram_size=2,
            min_length=10,
            top_p=0.95,
            temperature=0.7,
            do_sample=True
        )
        answer = generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"Error in processing the query: {e}"

def main():
    print("Loading data...")
    dataset = load_data()
    if dataset is None:
        print("Dataset loading failed.")
        return

    print("Loading models...")
    models = load_models()
    if models is None or len(models) != 6:
        print("Model loading failed.")
        return

    encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator = models
    embeddings, docs = load_embeddings_and_documents()
    index = load_faiss_index()

    if embeddings is None or docs is None or index is None:
        print("Building new embeddings and index...")
        raw_docs = [item["text"] for item in dataset]
        embeddings, docs = generate_embeddings(raw_docs, encoder_tokenizer, encoder_model, DOCS_TO_EMBED)
        if embeddings is None:
            print("Embedding generation failed.")
            return
        embeddings = reduce_embeddings(embeddings, new_dim=EMBEDDING_DIM)
        index = create_faiss_index(embeddings)
        save_embeddings_and_documents(embeddings, docs)
        save_faiss_index(index)
    else:
        print("Using cached embeddings and index...")

    print("Processing queries...")
    for query in QUERIES:
        answer = run_rag_pipeline(
            query,
            index,
            docs,
            encoder_tokenizer,
            encoder_model,
            reranker_tokenizer,
            reranker,
            generator_tokenizer,
            generator
        )
        print(f"Query: {query}\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
