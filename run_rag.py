import os
import torch
import logging
import re
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import faiss
from tqdm import tqdm
import google.generativeai as genai

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
CONFIG = {
    "encoder_model_name": "intfloat/e5-base-v2",
    "reranker_model_name": "BAAI/bge-reranker-base",
    "generator_model_name": "google/flan-t5-large",
    "faiss_index_name": "faiss_index.index",
    "embeddings_name": "encoded_passages.npy",
    "documents_name": "clean_documents.txt",
    "answer_file": "answer.txt",
    "top_k": 50,
    "embedding_dim": 768,
    "docs_to_embed": 5000,
    "queries": [
        "What causes earthquakes and how are they measured?",
        "Who were the main figures in the Russian Revolution?",
        "How does CRISPR gene editing work in simple terms?"
    ]
}

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Helper functions
def load_data():
    """Load the dataset from Wikipedia."""
    return load_dataset("wikipedia", "20220301.en", split="train[:1%]", trust_remote_code=True)

def load_models():
    """Load the encoder, reranker, and generator models."""
    encoder_tokenizer = AutoTokenizer.from_pretrained(CONFIG["encoder_model_name"])
    encoder_model = AutoModel.from_pretrained(CONFIG["encoder_model_name"]).to(device)

    reranker_tokenizer = AutoTokenizer.from_pretrained(CONFIG["reranker_model_name"])
    reranker = AutoModelForSequenceClassification.from_pretrained(CONFIG["reranker_model_name"]).to(device)

    generator_tokenizer = AutoTokenizer.from_pretrained(CONFIG["generator_model_name"])
    generator = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["generator_model_name"]).to(device)

    return encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator

def clean_text(text):
    """Clean up whitespace and sanitize text."""
    return re.sub(r'\s+', ' ', text.strip())

def save_embeddings_and_documents(embeddings, documents):
    """Save embeddings and documents to disk."""
    np.save(CONFIG["embeddings_name"], embeddings.cpu().numpy())
    with open(CONFIG["documents_name"], "w", encoding="utf-8") as f:
        f.writelines([f"{doc}\n" for doc in documents])

def load_embeddings_and_documents():
    """Load embeddings and documents from disk."""
    if not os.path.exists(CONFIG["embeddings_name"]) or not os.path.exists(CONFIG["documents_name"]):
        return None, None
    embeddings = torch.tensor(np.load(CONFIG["embeddings_name"]))
    with open(CONFIG["documents_name"], "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f]
    return embeddings, documents

def create_faiss_index(embeddings):
    """Create and return a FAISS index from embeddings."""
    faiss.normalize_L2(embeddings.cpu().numpy())
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.cpu().numpy())
    return index

def generate_embeddings(docs, encoder_tokenizer, encoder_model, batch_size=32):
    """Generate embeddings for a batch of documents."""
    embeddings, clean_docs = [], []
    for i in tqdm(range(0, len(docs), batch_size)):
        batch_docs = [clean_text(doc) for doc in docs[i:i+batch_size]]
        inputs = encoder_tokenizer(batch_docs, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = encoder_model(**inputs).last_hidden_state[:, 0, :]
            embeddings.extend(outputs.cpu())
            clean_docs.extend(batch_docs)
    return torch.stack(embeddings), clean_docs

def rerank(query, docs, reranker_tokenizer, reranker):
    """Rerank the retrieved documents using the reranker model."""
    inputs = reranker_tokenizer([[query, doc] for doc in docs], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        scores = reranker(**inputs).logits.squeeze(-1)
    top_indices = torch.topk(scores, k=min(CONFIG["top_k"], len(scores))).indices.tolist()
    return [(docs[i], scores[i].item()) for i in top_indices]

def encode_query(query, encoder_tokenizer, encoder_model):
    """Encode a query into an embedding."""
    query_inputs = encoder_tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(device)
    return encoder_model(**query_inputs).last_hidden_state[:, 0, :].detach().cpu().numpy()

def generate_answer(query, context, generator_tokenizer, generator):
    """Generate an answer to the query based on the context."""
    input_text = f"Answer the following questions clearly and accurately in 1-2 sentences.\nQuestion: {query}\nAnswer:"
    gen_inputs = generator_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    output_ids = generator.generate(**gen_inputs, num_beams=1, max_new_tokens=50, no_repeat_ngram_size=2, top_p=0.95, temperature=0.7, do_sample=True)
    return generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def run_rag_pipeline(query, index, docs, encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator):
    """Run the RAG pipeline to generate an answer to a query."""
    q_embedding = encode_query(query, encoder_tokenizer, encoder_model)
    faiss.normalize_L2(q_embedding)
    distances, top_idxs = index.search(q_embedding, CONFIG["top_k"])
    retrieved_docs = [docs[i] for i in top_idxs[0]]
    reranked_docs_with_scores = rerank(query, retrieved_docs, reranker_tokenizer, reranker)
    context = " ".join([doc for doc, _ in reranked_docs_with_scores])[:2048]
    return generate_answer(query, context, generator_tokenizer, generator)

def main():
    """Main function to load data, models, and process queries."""
    dataset = load_data()
    if dataset is None:
        logger.error("Failed to load dataset.")
        return

    encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator = load_models()

    embeddings, clean_docs = load_embeddings_and_documents()
    faiss_index = faiss.read_index(CONFIG["faiss_index_name"]) if os.path.exists(CONFIG["faiss_index_name"]) else None

    if embeddings is None or clean_docs is None or faiss_index is None:
        docs = [clean_text(doc) for doc in dataset["text"][:CONFIG["docs_to_embed"]]]
        embeddings, clean_docs = generate_embeddings(docs, encoder_tokenizer, encoder_model)
        faiss_index = create_faiss_index(embeddings)
        save_embeddings_and_documents(embeddings, clean_docs)
        faiss.write_index(faiss_index, CONFIG["faiss_index_name"])

    with open(CONFIG["answer_file"], "w", encoding="utf-8") as f:
        for query in CONFIG["queries"]:
            answer = run_rag_pipeline(
                query, faiss_index, clean_docs,
                encoder_tokenizer, encoder_model,
                reranker_tokenizer, reranker,
                generator_tokenizer, generator
            )

            result = f"Query: {query}\nAnswer: {answer}\n\n"

            print(result, end="")
            f.write(result)

if __name__ == "__main__":
    main()
