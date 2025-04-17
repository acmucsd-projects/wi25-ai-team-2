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

# Setup
multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", message="resource_tracker: There appear to")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Ensure usage of GPU

# Hyperparameters
EMBEDDING_DIM = 768
TOP_K = 50
DOCS_TO_EMBED = 5000
CONFIDENCE_THRESHOLD = 0.7
QUERIES = [
    "Who is the president of the United States?",
    "What year was the Eiffel Tower completed?",
    "Which anime features a character named Goku?",
    "What is the capital of Japan?",
    "Who created the video game 'Super Mario Bros.'?",
    "Which country is home to the Great Wall of China?",
    "What is the most-watched TV show of all time?",
    "Where is the Statue of Liberty located?",
    "Who won the 2018 FIFA World Cup?",
    "Which city is known as the 'City of Lights'?"
]

# Configuration
CONFIG = {
    "encoder_model_name": "intfloat/e5-base-v2",
    "reranker_model_name": "BAAI/bge-reranker-base",
    "generator_model_name": "google/flan-t5-large",
    "answer_file_path": "/content/RAG_Reranker_LLM/RAG_Reranker_LLM/answer.txt",
    "faiss_index_name": "/content/RAG_Reranker_LLM/RAG_Reranker_LLM/faiss_index.index",
    "embeddings_name": "/content/RAG_Reranker_LLM/RAG_Reranker_LLM/encoded_passages.npy",
    "documents_name": "/content/RAG_Reranker_LLM/RAG_Reranker_LLM/clean_documents.txt",
    "retriever_top_k": 10,
    "embedding_dim": 768,

    # Generation settings
    "num_beams": 3,
    "top_k": 50,
    "top_p": 0.9,
    "temperature": 0.7,
    "do_sample": True,

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
ANSWER_FILE_PATH = CONFIG["answer_file_path"]  # Absolute path to answer file

# Ensure the directory for the answer file exists
os.makedirs(os.path.dirname(ANSWER_FILE_PATH), exist_ok=True)

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
        encoder_model = AutoModel.from_pretrained(CONFIG["encoder_model_name"]).to(device)  # Move to GPU

        reranker_tokenizer = AutoTokenizer.from_pretrained(CONFIG["reranker_model_name"])
        reranker = AutoModelForSequenceClassification.from_pretrained(CONFIG["reranker_model_name"]).to(device)

        generator_tokenizer = AutoTokenizer.from_pretrained(CONFIG["generator_model_name"])
        generator = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["generator_model_name"]).to(device)

        return encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def sanitize_context(text):
    text = re.sub(r'-lrb-', '(', text)
    text = re.sub(r'-rrb-', ')', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'\"()\-]', '', text)
    return text.strip()

def generate_embeddings(docs, encoder_tokenizer, encoder_model, num_docs, batch_size=32):
    embeddings, clean_docs = [], []
    for i in tqdm(range(0, num_docs, batch_size)):
        batch_docs = [sanitize_context(docs[j]) for j in range(i, min(i + batch_size, num_docs)) if isinstance(docs[j], str)]
        try:
            inputs = encoder_tokenizer([f"passage: {doc}" for doc in batch_docs], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)  # Move to GPU
            with torch.no_grad():
                outputs = encoder_model(**inputs).last_hidden_state[:, 0, :]
                embeddings.extend(outputs.cpu())  # Move results back to CPU
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
    inputs = reranker_tokenizer([[query, doc] for doc in docs], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)  # Move to GPU
    with torch.no_grad():
        scores = reranker(**inputs).logits.squeeze(-1)
    top_indices = torch.topk(scores, k=min(top_n, len(scores))).indices.tolist()
    reranked_docs = [docs[i] for i in top_indices]
    reranked_scores = [scores[i].item() for i in top_indices]
    return [(doc, score) for doc, score in zip(reranked_docs, reranked_scores) if score >= CONFIDENCE_THRESHOLD]

def encode_query(query, encoder_tokenizer, encoder_model, target_dim=EMBEDDING_DIM):
    query_inputs = encoder_tokenizer(f"query: {query}", return_tensors="pt", truncation=True, padding=True).to(device)  # Move to GPU
    query_embedding = encoder_model(**query_inputs).last_hidden_state[:, 0, :]
    if query_embedding.shape[1] != target_dim:
        query_embedding = query_embedding[:, :target_dim]
    return query_embedding.detach().cpu().numpy().astype("float32")  # Move results back to CPU

def run_rag_pipeline(query, index, docs, encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator):
    try:
        # Debug: Print the query being processed
        logger.info(f"Processing query: {query}")

        # Generate query embedding
        q_embedding = encode_query(query, encoder_tokenizer, encoder_model)
        logger.info(f"Query embedding shape: {q_embedding.shape}")

        # Ensure that the query embedding matches the FAISS index dimension
        if q_embedding.shape[1] != index.d:
            logger.error(f"Embedding dimension mismatch: query has {q_embedding.shape[1]} but FAISS index has {index.d}")
            return f"Error: Embedding dimension mismatch."

        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(q_embedding)

        # Search the index with the query embedding
        k = min(TOP_K * 3, len(docs))
        distances, top_idxs = index.search(q_embedding, k=k)
        logger.info(f"Top indices: {top_idxs}")

        retrieved_docs = [docs[i] for i in top_idxs[0]]
        reranked_docs_with_scores = rerank(query, retrieved_docs, reranker_tokenizer, reranker)
        reranked_docs = [doc for doc, _ in reranked_docs_with_scores]
        sanitized_docs = [sanitize_context(doc) for doc in reranked_docs]
        context = " ".join(sanitized_docs)[:2048]

        # Prompt engineering
        input_text = f"""Answer the following questions clearly and accurately in 1-2 sentences.
        Question: {query}
        Answer:"""

        gen_inputs = generator_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)  # Move to GPU
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

def load_embeddings_and_documents():
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(DOCUMENTS_PATH) or not os.path.exists(FAISS_INDEX_PATH):
        print("One or more required files are missing. Regenerating...")
        # Regenerate documents and embeddings
        regenerate_resources()
    
    try:
        # Load the pre-saved embeddings and documents
        embeddings = np.load(EMBEDDINGS_PATH)  # Load the pre-saved embeddings
        with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
            clean_docs = f.readlines()  # Load the documents
        return embeddings, clean_docs
    except Exception as e:
        print(f"Error loading embeddings or documents: {e}")
        return None, None

def regenerate_resources():
    # Load the dataset
    dataset = load_data()
    if dataset is None:
        print("Dataset loading failed.")
        return

    # Load the models
    models = load_models()
    if models is None or len(models) != 6:
        print("Model loading failed.")
        return

    encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator = models

    # Generate embeddings for documents
    print("Generating embeddings for documents...")
    embeddings, clean_docs = generate_embeddings(dataset["text"], encoder_tokenizer, encoder_model, num_docs=DOCS_TO_EMBED)

    # Save the clean documents and embeddings
    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        f.writelines(clean_docs)

    np.save(EMBEDDINGS_PATH, embeddings.numpy())  # Save embeddings
    print("Embeddings and documents saved successfully.")

    # Create the FAISS index
    print("Creating FAISS index...")
    index = create_faiss_index(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)  # Save FAISS index
    print("FAISS index created and saved successfully.")

def load_faiss_index():
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)  # Load the FAISS index
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

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
    print("Loading embeddings, documents, and FAISS index...")
    embeddings, clean_docs = load_embeddings_and_documents()
    index = load_faiss_index()

    if embeddings is None or clean_docs is None or index is None:
        print("Failed to load necessary resources.")
        return
        
    answers = []
    for query in QUERIES:
        print(f"Answering query: {query}")
        answer = run_rag_pipeline(query, index, clean_docs, encoder_tokenizer, encoder_model, reranker_tokenizer, reranker, generator_tokenizer, generator)
        print(f"Answer: {answer}\n")
        answers.append((query, answer))

    # Now write all answers to the file
    with open(ANSWER_FILE_PATH, 'w', encoding="utf-8") as f:  # Use 'w' to overwrite the file
        for query, answer in answers:
            f.write(f"Query: {query}\nAnswer: {answer}\n\n")  # Write each query-answer pair


if __name__ == "__main__":
    main()
