import os
import logging
import torch
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
from rank_bm25 import BM25Okapi  # Add BM25 for hybrid retrieval

# Setup
multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", message="resource_tracker: There appear to")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EMBEDDING_DIM = 300  # Reduced to 300D to match PCA reduced dimension
TOP_K = 20
DOCS_TO_EMBED = 10000  # Reduced docs to embed for speed

# Paths
EMBEDDINGS_PATH = "encoded_passages.npy"
DOCUMENTS_PATH = "clean_documents.txt"
FAISS_INDEX_PATH = "faiss_index.index"

logging.disable(logging.CRITICAL)
logger = logging.getLogger()


def load_data():
    try:
        dataset = load_dataset("ag_news", split="train")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def load_models():
    try:
        encoder_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")  # Smaller model for encoder
        encoder_model = AutoModel.from_pretrained("intfloat/e5-base").to(device)

        reranker_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2").to(device)

        generator_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")  # Smaller model for generator
        generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

        return encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator
    except Exception as e:
        print(f"Error loading models: {e}")
        return None


def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())


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
    
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(embeddings.shape[1]), embeddings.shape[1], 100, 8, 8)
    index.train(embeddings.cpu().numpy())
    index.add(embeddings.cpu().numpy())
    return index


def rerank(query, docs, reranker_tokenizer, reranker, top_n=TOP_K):
    inputs = reranker_tokenizer([[query, doc] for doc in docs], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        scores = reranker(**inputs).logits.squeeze(-1)
    top_indices = torch.topk(scores, k=min(top_n, len(scores))).indices.tolist()
    return [docs[i] for i in top_indices]


def encode_query(query, encoder_tokenizer, encoder_model, target_dim=EMBEDDING_DIM):
    query_inputs = encoder_tokenizer(f"query: {query}", return_tensors="pt", truncation=True, padding=True).to(device)
    query_embedding = encoder_model(**query_inputs).last_hidden_state[:, 0, :]

    if query_embedding.shape[1] != target_dim:
        query_embedding = query_embedding[:, :target_dim]

    query_embedding = query_embedding.detach().cpu().numpy().astype("float32")
    return query_embedding


def hybrid_retrieve(query, docs, index, bm25_model, encoder_tokenizer, encoder_model, k=TOP_K):
    # Get dense retrieval from FAISS
    q_embedding = encode_query(query, encoder_tokenizer, encoder_model)
    distances, top_idxs = index.search(q_embedding, k=k)
    retrieved_docs = [docs[i] for i in top_idxs[0]]
    
    # Get sparse retrieval from BM25
    bm25_scores = bm25_model.get_scores(query.split())  # BM25 score for the query
    bm25_top_docs = [docs[i] for i in np.argsort(bm25_scores)[-k:]]  # Top k BM25 docs
    
    # Combine the results from both methods
    hybrid_docs = list(set(retrieved_docs + bm25_top_docs))  # Merge without duplicates
    
    return hybrid_docs


def run_rag_pipeline(query, index, docs, encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator, filter_keyword=None):
    try:
        q_embedding = encode_query(query, encoder_tokenizer, encoder_model)

        # Increase the number of docs retrieved for better chances of finding relevant ones
        k = min(TOP_K * 3, len(docs))  # Increased retrieval range

        try:
            distances, top_idxs = index.search(q_embedding, k=k)
        except Exception as e:
            return "Error retrieving documents."

        try:
            retrieved_docs = [docs[i] for i in top_idxs[0]]
        except Exception as e:
            return "Error accessing documents."

        try:
            reranked_docs = rerank(query, retrieved_docs, reranker_tokenizer, reranker)
        except Exception as e:
            return "Error reranking documents."

        try:
            if filter_keyword:
                filtered_docs = [doc for doc in reranked_docs if filter_keyword.lower() in doc.lower()]
            else:
                filtered_docs = reranked_docs
        except Exception as e:
            return "Error filtering documents."

        # Add specific context to the prompt for generation
        try:
            context = " ".join(filtered_docs if filtered_docs else reranked_docs)[:2048]
            input_text = f"Given the following context: {context}, answer the question: {query}"
            gen_inputs = generator_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            output_ids = generator.generate(**gen_inputs, num_beams=5, max_length=100, no_repeat_ngram_size=2, min_length=10)
            answer = generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            return "Error generating answer."

    except Exception as e:
        return "Error in processing the query."


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

    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        filter_keyword = input("Enter a keyword to filter documents by (or press Enter to skip): ").strip()
        if not filter_keyword:
            filter_keyword = None
        
        answer = run_rag_pipeline(
            query,
            index,
            docs,
            encoder_tokenizer,
            encoder_model,
            reranker_tokenizer,
            reranker,
            generator_tokenizer,
            generator,
            filter_keyword=filter_keyword
        )
        
        if answer:
            print(f"Query: {query}\nAnswer: {answer}")
        else:
            print(f"Query: {query}\nAnswer: No valid answer found.")


if __name__ == "__main__":
    main()
