# This code runs only on a GPU in Google Colab.

import os, re, torch, faiss, numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForCausalLM, BitsAndBytesConfig
)
import warnings; warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
top_k, embedding_dim, docs_to_embed, batch_size = 30, 768, 5000, 16  # Hyperparameters
faiss_index_path = "/content/faiss_index.index"  # Path for FAISS index
embedding_path = "/content/embeddings.npy"  # Path for saved embeddings
document_path = "/content/documents.txt"  # Path for saved documents
answer_path = "/content/answer.txt"  # Path to save generated answers
queries = [
    "What is the process of photosynthesis?",
    "How do vaccines work in the human body?",
    "What are the primary causes of climate change?",
    "Explain the theory of relativity in simple terms.",
    "What are the benefits of regular exercise on mental health?"
]   # List of queries for the QA system

# Model names for different parts of the pipeline
encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Sentence encoder model
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"  # Cross-encoder reranker model
generator_model_name = "deepcogito/cogito-v1-preview-llama-3B"  # Causal language model for generating answers

# Utility functions
def clean_text(t): 
    return re.sub(r'\s+', ' ', t.strip())  # Clean extra spaces in text

def split_chunks(t, max_len=256, stride=64):  
    # Split text into chunks to avoid exceeding model's max token limit
    words = t.split()  
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len - stride)]

def save(embs, docs):
    np.save(embedding_path, embs.cpu().numpy())  # Save embeddings to disk
    with open(document_path, "w", encoding="utf-8") as f: 
        f.writelines(f"{doc}\n" for doc in docs)  # Save documents to disk

def load():
    if not os.path.exists(embedding_path): 
        return None, None  # Return None if no saved embeddings
    embs = torch.tensor(np.load(embedding_path))  # Load saved embeddings
    docs = open(document_path).read().splitlines()  # Load saved documents
    return embs, docs

def build_index(embs):
    embs = embs.cpu().numpy().astype("float32")  # Convert embeddings to float32 for FAISS
    faiss.normalize_L2(embs)  # Normalize embeddings for similarity search
    index = faiss.IndexFlatIP(embs.shape[1])  # Create FAISS index for inner product similarity
    index.add(embs)  # Add embeddings to the index
    return index

# Load models
def load_encoder():
    t = AutoTokenizer.from_pretrained(encoder_model_name)  # Load tokenizer for encoder
    m = AutoModel.from_pretrained(encoder_model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")  # Load encoder model with 8-bit quantization
    return t, m

def load_reranker():
    t = AutoTokenizer.from_pretrained(reranker_model_name)  # Load tokenizer for reranker
    m = AutoModelForSequenceClassification.from_pretrained(reranker_model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")  # Load reranker model
    return t, m

def load_generator():
    t = AutoTokenizer.from_pretrained(generator_model_name)  # Load tokenizer for generator
    t.pad_token = t.eos_token  # Set padding token to EOS token
    m = AutoModelForCausalLM.from_pretrained(generator_model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")  # Load generator model
    return t, m

# Main embedding function
def embed_documents(docs, tokenizer, model):
    chunks = [chunk for d in docs for chunk in split_chunks(clean_text(d))]  # Split documents into smaller chunks
    embeddings, flat_docs = [], []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):  # Process in batches
        batch = chunks[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)  # Tokenize batch
        with torch.no_grad():  # Disable gradient computation for inference
            embs = model(**tokens).last_hidden_state[:, 0, :].cpu()  # Get embeddings for each chunk
        embeddings.extend(embs)  # Collect embeddings
        flat_docs.extend(batch)  # Collect corresponding documents
    return torch.stack(embeddings), flat_docs

def encode_query(q, tokenizer, model):
    toks = tokenizer(q, return_tensors="pt", truncation=True, padding=True).to(device)  # Tokenize query
    with torch.no_grad():  # Disable gradient computation for inference
        vec = model(**toks).last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)  # Get query embedding
    faiss.normalize_L2(vec)  # Normalize query embedding
    return vec

def rerank(query, docs, tokenizer, model):
    scores = []
    for i in range(0, len(docs), batch_size):  # Process in batches
        pairs = [[query, d] for d in docs[i:i+batch_size]]  # Create query-document pairs
        toks = tokenizer(pairs, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)  # Tokenize pairs
        with torch.no_grad():  # Disable gradient computation for inference
            logits = model(**toks).logits.squeeze(-1).cpu()  # Get model scores for pairs
        scores.extend(logits.tolist())  # Collect scores
    idxs = torch.topk(torch.tensor(scores), min(top_k, len(scores))).indices.tolist()  # Get top-k documents
    return [docs[i] for i in idxs]  # Return the top-k documents

def generate_answer(query, context, tokenizer, model):
    prompt = f"Answer this question clearly and directly, using the context below only if needed.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2, return_dict_in_generate=True)
    answer = tokenizer.decode(output_ids.sequences[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[-1].strip()  # Extract the generated answer
    answer = re.sub(r'^(According to|Based on)( the)? (provided )?(context|text)[:,]?\s*', '', answer, flags=re.IGNORECASE)  # Clean unnecessary prefixes
    answer = answer.replace("\n", " ").strip()  # Replace newlines with spaces and remove extra spaces
    answer = answer[:500].rsplit('.', 1)[0] + '.'  # Ensure answer doesn't exceed 500 characters
    return answer

def main():
    # Load dataset
    dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{docs_to_embed}]", cache_dir="/content/hf_cache", trust_remote_code=True)  # Load from Hugging Face
    raw_docs = [clean_text(doc) for doc in dataset["text"]]  # Clean the raw text documents

    # Load or build index
    embeddings, flat_docs = load()  # Load existing embeddings if available
    if embeddings is None:  # If no embeddings exist, generate new ones
        enc_tok, enc_model = load_encoder()
        embeddings, flat_docs = embed_documents(raw_docs, enc_tok, enc_model)  # Generate document embeddings
        index = build_index(embeddings)  # Build FAISS index for fast retrieval
        save(embeddings, flat_docs)  # Save the embeddings and documents
        faiss.write_index(index, faiss_index_path)  # Save FAISS index
    else:
        index = faiss.read_index(faiss_index_path)  # Load existing FAISS index

    # Load models for encoding, reranking, and generation
    enc_tok, enc_model = load_encoder()  # Encoder model
    rr_tok, rr_model = load_reranker()  # Reranker model
    gen_tok, gen_model = load_generator()  # Generator model

    # Process each query
    with open(answer_path, "w") as f:
        for q in queries:  # For each query
            q_emb = encode_query(q, enc_tok, enc_model)  # Encode query
            _, idxs = index.search(q_emb, top_k)  # Retrieve top-k documents
            candidates = [flat_docs[i] for i in idxs[0]]  # Get corresponding documents
            reranked = rerank(q, candidates, rr_tok, rr_model)  # Rerank the documents
            context = " ".join(reranked)[:2048]  # Create context for answer (limited to 2048 tokens)
            ans = generate_answer(q, context, gen_tok, gen_model)  # Generate the answer
            f.write(f"Query: {q}\n\nAnswer: {ans}\n\n")  # Save query-answer pair to file

if __name__ == "__main__":
    main()  # Run the main function
