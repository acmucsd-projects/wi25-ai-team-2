import os, re, torch, faiss, numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForCausalLM, BitsAndBytesConfig
)
import warnings; warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Ensure the code runs on GPU if available
top_k, embedding_dim, docs_to_embed, batch_size = 30, 768, 5000, 16  # Define hyperparameters
faiss_index_path = "/content/faiss_index.index"
embedding_path = "/content/embeddings.npy"
document_path = "/content/documents.txt"
answer_path = "/content/answer.txt"
queries = [
    "What are black holes and how do they form?",
    "How does photosynthesis work in plants?",
    "What causes climate change and global warming?",
    "Explain the theory of evolution by natural selection.",
    "How does blockchain technology ensure data security?"
]

# Model names
encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
generator_model_name = "deepcogito/cogito-v1-preview-llama-3B"

# Utilities
def clean_text(t): return re.sub(r'\s+', ' ', t.strip())  # Clean extra spaces in text
def split_chunks(t, max_len=256, stride=64):  # Split text into chunks for embedding
    words = t.split(); return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len - stride)]

def save(embs, docs):
    np.save(embedding_path, embs.cpu().numpy())  # Save embeddings to file
    with open(document_path, "w", encoding="utf-8") as f: f.writelines(f"{doc}\n" for doc in docs)  # Save documents

def load():
    if not os.path.exists(embedding_path): return None, None  # If no saved embeddings exist, return None
    embs = torch.tensor(np.load(embedding_path))  # Load saved embeddings
    docs = open(document_path).read().splitlines()  # Load saved documents
    return embs, docs

def build_index(embs):
    embs = embs.cpu().numpy().astype("float32")  # Convert embeddings to float32
    faiss.normalize_L2(embs)  # Normalize embeddings for similarity search
    index = faiss.IndexFlatIP(embs.shape[1])  # Create FAISS index
    index.add(embs)  # Add embeddings to index
    return index

# Load models once
def load_encoder():
    t = AutoTokenizer.from_pretrained(encoder_model_name)  # Load tokenizer for encoder
    m = AutoModel.from_pretrained(encoder_model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")  # Load encoder model
    return t, m

def load_reranker():
    t = AutoTokenizer.from_pretrained(reranker_model_name)  # Load tokenizer for reranker
    m = AutoModelForSequenceClassification.from_pretrained(reranker_model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")  # Load reranker model
    return t, m

def load_generator():
    t = AutoTokenizer.from_pretrained(generator_model_name)  # Load tokenizer for generator
    t.pad_token = t.eos_token  # Set padding token
    m = AutoModelForCausalLM.from_pretrained(generator_model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")  # Load generator model
    return t, m

# Main steps
def embed_documents(docs, tokenizer, model):
    chunks = [chunk for d in docs for chunk in split_chunks(clean_text(d))]  # Split documents into smaller chunks
    embeddings, flat_docs = [], []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):  # Process in batches
        batch = chunks[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)  # Tokenize and move to GPU
        with torch.no_grad():  # No need to compute gradients for inference
            embs = model(**tokens).last_hidden_state[:, 0, :].cpu()  # Get embeddings
        embeddings.extend(embs); flat_docs.extend(batch)  # Collect embeddings and corresponding documents
    return torch.stack(embeddings), flat_docs

def encode_query(q, tokenizer, model):
    toks = tokenizer(q, return_tensors="pt", truncation=True, padding=True).to(device)  # Tokenize query and move to GPU
    with torch.no_grad():  # No need to compute gradients for inference
        vec = model(**toks).last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)  # Get query embedding
    faiss.normalize_L2(vec)  # Normalize query embedding for similarity search
    return vec

def rerank(query, docs, tokenizer, model):
    scores = []
    for i in range(0, len(docs), batch_size):  # Process in batches
        pairs = [[query, d] for d in docs[i:i+batch_size]]  # Create query-document pairs
        toks = tokenizer(pairs, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)  # Tokenize pairs and move to GPU
        with torch.no_grad():  # No need to compute gradients for inference
            logits = model(**toks).logits.squeeze(-1).cpu()  # Get model scores for pairs
        scores.extend(logits.tolist())  # Collect scores
    idxs = torch.topk(torch.tensor(scores), min(top_k, len(scores))).indices.tolist()  # Get top-k scores
    return [docs[i] for i in idxs]  # Return the top-k reranked documents

def generate_answer(query, context, tokenizer, model):
    prompt = f"Answer concisely and clearly based only on the context:\nContext: {context}\nQuestion: {query}\nAnswer:"  # Create prompt for LLM
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)  # Tokenize prompt and move to GPU
    output_ids = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7, top_k=50, top_p=0.9, return_dict_in_generate=True)  # Generate answer
    answer = tokenizer.decode(output_ids.sequences[0], skip_special_tokens=True)  # Decode generated tokens
    return answer.split("Answer:")[-1].strip()  # Extract and clean the answer

def main():
    # Load dataset
    dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{docs_to_embed}]", cache_dir="/content/hf_cache", trust_remote_code=True)  # Load dataset from Hugging Face
    raw_docs = [clean_text(doc) for doc in dataset["text"]]  # Clean raw documents

    # Load or build index
    embeddings, flat_docs = load()  # Load saved embeddings if available
    if embeddings is None:  # If no embeddings exist, create new ones
        enc_tok, enc_model = load_encoder()
        embeddings, flat_docs = embed_documents(raw_docs, enc_tok, enc_model)  # Generate embeddings for documents
        index = build_index(embeddings)  # Build FAISS index for fast retrieval
        save(embeddings, flat_docs)  # Save embeddings and documents
        faiss.write_index(index, faiss_index_path)  # Save FAISS index
    else:
        index = faiss.read_index(faiss_index_path)  # Load existing FAISS index

    # Load models
    enc_tok, enc_model = load_encoder()  # Load encoder model
    rr_tok, rr_model = load_reranker()  # Load reranker model
    gen_tok, gen_model = load_generator()  # Load generator model

    # Run pipeline
    with open(answer_path, "w") as f:
        for q in queries:  # Process each query
            q_emb = encode_query(q, enc_tok, enc_model)  # Encode query
            _, idxs = index.search(q_emb, top_k)  # Search for top-k relevant documents
            candidates = [flat_docs[i] for i in idxs[0]]  # Get top-k candidate documents
            reranked = rerank(q, candidates, rr_tok, rr_model)  # Rerank the documents
            context = " ".join(reranked)[:2048]  # Create context for the answer (limit to 2048 tokens)
            ans = generate_answer(q, context, gen_tok, gen_model)  # Generate answer based on context
            f.write(f"Query: {q}\nAnswer: {ans}\n\n")  # Write query and answer to file

if __name__ == "__main__":
    main()  # Run the main function
