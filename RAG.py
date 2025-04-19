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
from tqdm import tqdm
import traceback
import google.generativeai as genai

# Setup
multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", message="resource_tracker: There appear to")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter
EMBEDDING_DIM = 768
TOP_K = 50
DOCS_TO_EMBED = 5000
CONFIDENCE_THRESHOLD = 0.7
QUERIES = [
    "What causes earthquakes and how are they measured?",
    "Who were the main figures in the Russian Revolution?",
    "How does CRISPR gene editing work in simple terms?",
    "What is the difference between machine learning and deep learning?",
    "What are the key differences between the Arctic and Antarctic regions?",
    "Why did the Roman Empire fall?",
    "How does the human immune system fight off viruses?",
    "What are the symptoms and treatments of Parkinson’s disease?",
    "What are black holes and how are they formed?",
    "Explain the theory of general relativity in layman’s terms."
]


# Configuration
CONFIG = {
    "encoder_model_name": "intfloat/e5-base-v2",
    "reranker_model_name": "BAAI/bge-reranker-base",
    "generator_model_name": "google/flan-t5-large",
    "faiss_index_name": "faiss_index.index",
    "embeddings_name": "encoded_passages.npy",
    "documents_name": "clean_documents.txt",
    "answer_file": "answer.txt",
    "retriever_top_k": 5,
    "embedding_dim": 768,  # Set to 768 for E5 encoder model

    # Generation settings
    "num_beams": 1,               # Beam search disabled for now (using sampling)
    "top_k": 50,                  # Limit to top 50 logits before sampling
    "top_p": 0.95,                # Nucleus sampling for natural output
    "temperature": 0.7,           # Controls randomness
    "do_sample": True,            # Enable sampling-based generation

    # Retrieval toggles
    "use_dense_retrieval": True,
    "use_sparse_retrieval": False,

    # Logging and post-processing
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
        print(f"Embeddings or documents file not found at {EMBEDDINGS_PATH} or {DOCUMENTS_PATH}")
        return None, None
    try:
        embeddings = torch.tensor(np.load(EMBEDDINGS_PATH))
        with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
            documents = [line.strip() for line in f]
        print(f"Successfully loaded embeddings with shape {embeddings.shape} and {len(documents)} documents.")
        return embeddings, documents
    except Exception as e:
        print(f"Error loading embeddings and documents: {e}")
        return None, None

def save_faiss_index(index):
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            return faiss.read_index(FAISS_INDEX_PATH)
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None
    else:
        return None

def generate_embeddings(docs, encoder_tokenizer, encoder_model, num_docs, batch_size=32):
    embeddings, clean_docs = [], []
    for i in tqdm(range(0, num_docs, batch_size)):
        batch_docs = [clean_text(docs[j]) for j in range(i, min(i + batch_size, num_docs)) if isinstance(docs[j], str)]
        try:
            inputs = encoder_tokenizer([f"passage: {doc}" for doc in batch_docs], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = encoder_model(**inputs).last_hidden_state[:, 0, :]
                embeddings.extend(outputs.cpu())
                clean_docs.extend(batch_docs)
        except Exception as e:
            print(f"Embedding batch error: {e}")
    return torch.stack(embeddings), clean_docs

def create_faiss_index(embeddings):
    if embeddings.ndimension() != 2:
        raise ValueError(f"Embeddings should have shape (num_docs, embedding_dim), but got {embeddings.shape}")
    
    # Normalize embeddings for cosine similarity (L2 normalization)
    faiss.normalize_L2(embeddings.cpu().numpy())  # In-place normalization
    
    # Use IndexFlatIP for cosine similarity (inner product)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    
    try:
        index.add(embeddings.cpu().numpy())  # Add normalized embeddings to index
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
        # Debug: Print the query being processed
        logger.info(f"Processing query: {query}")
        
        # Generate query embedding
        q_embedding = encode_query(query, encoder_tokenizer, encoder_model)
        logger.info(f"Query embedding shape: {q_embedding.shape}")  # Log the shape of the query embedding
        
        # Ensure that the query embedding matches the FAISS index dimension
        if q_embedding.shape[1] != index.d:
            logger.error(f"Embedding dimension mismatch: query has {q_embedding.shape[1]} but FAISS index has {index.d}")
            return f"Error: Embedding dimension mismatch."

        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(q_embedding)

        # Search the index with the query embedding
        k = min(TOP_K * 3, len(docs))
        distances, top_idxs = index.search(q_embedding, k=k)
        logger.info(f"Top indices: {top_idxs}")  # Log the top indices retrieved
        
        retrieved_docs = [docs[i] for i in top_idxs[0]]
        reranked_docs_with_scores = rerank(query, retrieved_docs, reranker_tokenizer, reranker)
        reranked_docs = [doc for doc, _ in reranked_docs_with_scores]
        sanitized_docs = [sanitize_context(doc) for doc in reranked_docs]
        context = " ".join(sanitized_docs)[:2048]

        # Prompt engineering
        input_text = f"""Answer the following questions clearly and accurately in 1-2 sentences.
        Question: {query}
        Answer:"""

        gen_inputs = generator_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        output_ids = generator.generate(
            **gen_inputs,
            num_beams=CONFIG["num_beams"],
            max_new_tokens=50,
            no_repeat_ngram_size=2,
            min_length=10,
            top_p=CONFIG["top_p"],
            temperature=CONFIG["temperature"],
            do_sample=CONFIG["top_k"] > 0 or CONFIG["top_p"] < 1.0
        )
        answer = generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        # Log the full error message and stack trace
        logger.error(f"Error processing query: {query}")
        logger.error(f"Error details: {str(e)}")
        logger.error("Stack Trace:")
        logger.error(traceback.format_exc())
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

    # Check if pre-existing embeddings, documents, and FAISS index are available
    print("Checking for existing embeddings and FAISS index...")
    embeddings, clean_docs = load_embeddings_and_documents()
    faiss_index = load_faiss_index()

    if embeddings is not None and clean_docs is not None and faiss_index is not None:
        print("Loaded existing embeddings and FAISS index.")
    else:
        # If embeddings and FAISS index are not available, generate them
        print("Generating embeddings...")
        docs = [clean_text(doc) for doc in dataset["text"][:DOCS_TO_EMBED]]  # Adjust number of docs based on your setup
        embeddings, clean_docs = generate_embeddings(docs, encoder_tokenizer, encoder_model, DOCS_TO_EMBED)

        print("Creating FAISS index...")
        faiss_index = create_faiss_index(embeddings)  # Use original embeddings directly
        if faiss_index is None:
            print("Failed to create FAISS index.")
            return

        # Save embeddings, documents, and FAISS index
        print("Saving embeddings, documents, and FAISS index...")
        save_embeddings_and_documents(embeddings, clean_docs)
        save_faiss_index(faiss_index)

    # Processing queries and writing to the file specified in the config
    print("Processing queries...")
    with open(CONFIG["answer_file"], "w", encoding="utf-8") as f:  # Use the config file path here
        for query in QUERIES:
            print(f"Query: {query}")
            answer = run_rag_pipeline(query, faiss_index, clean_docs, encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator)
            print(f"Answer: {answer}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Answer: {answer}\n\n")

if __name__ == "__main__":
    main()
