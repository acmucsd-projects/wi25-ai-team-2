import os
import torch
import numpy as np
import faiss
import re
import gc
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, AutoModel, DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from tqdm import tqdm
import warnings

# Disable unnecessary warnings
warnings.filterwarnings("ignore")

# Disable WANDB logging (for disabling Weights & Biases tracking)
os.environ["WANDB_MODE"] = "disabled"

# Hyperparameters (top section)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device for computation

# General Parameters
top_k = 30  # Number of top documents to retrieve from FAISS
embedding_dim = 768  # Embedding dimension size
docs_to_embed = 5000  # Number of documents to embed from Wikipedia
batch_size = 8  # Batch size for processing
faiss_index_path = "/content/faiss_index.index"  # Path for FAISS index
embedding_path = "/content/embeddings.npy"  # Path for embeddings file
document_path = "/content/documents.txt"  # Path for storing documents
answer_path = "/content/answer.txt"  # Path for storing generated answers
os.makedirs("/content", exist_ok=True)  # Create project directory if not exists

# Query Parameters
queries = [  # List of queries for which we will retrieve answers
    "What is the process of photosynthesis?"
]

# Model Names
encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Encoder model for text embedding
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"  # Reranker model for ranking retrieved documents
generator_model_name = "deepcogito/cogito-v1-preview-llama-3B"  # Generator model for answer generation

# Tokenizer and Model Constants
max_length = 512  # Maximum length for tokenization
max_new_tokens = 100  # Maximum number of tokens to generate in the output
temperature = 0.7  # Temperature for sampling to control randomness
top_p = 0.9  # Top-p (nucleus sampling) to limit candidate pool for next token
max_query_length = 256  # Maximum length for query tokenization
max_answer_length = 64  # Maximum length for answer tokenization

# Functions for resetting memory, cleaning text, saving/loading data
def reset_memory():
    """Clears memory to avoid out-of-memory errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def clean_text(t):
    """Cleans the input text by removing extra spaces."""
    return re.sub(r'\s+', ' ', t.strip())

def save(embs, docs):
    """Saves embeddings and documents to specified paths."""
    np.save(embedding_path, embs.cpu().numpy())
    with open(document_path, "w", encoding="utf-8") as f:
        f.writelines(f"{doc}\n" for doc in docs)

def load():
    """Loads saved embeddings and documents if they exist."""
    if not os.path.exists(embedding_path):
        return None, None
    embs = torch.tensor(np.load(embedding_path))
    docs = open(document_path).read().splitlines()
    return embs, docs

def save_index(index):
    """Saves FAISS index to the specified path."""
    faiss.write_index(index, faiss_index_path)

def load_index():
    """Loads the FAISS index from the saved path."""
    return faiss.read_index(faiss_index_path)

def build_index(embs):
    """Builds a FAISS index from the embeddings."""
    embs = embs.cpu().numpy().astype("float32")
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])  # Using inner product as the distance metric
    index.add(embs)  # Adds embeddings to the index
    return index

def preprocess_squad_data(dataset, tokenizer):
    """Preprocesses the SQuAD dataset for fine-tuning."""
    def preprocess(example):
        """Preprocessing function for each example in the dataset."""
        input_text = f"Question: {example['question']} Context: {example['context']} Answer:"
        target_text = example['answers']['text'][0] if example['answers']['text'] else ""
        model_inputs = tokenizer(input_text, truncation=True, padding='max_length', max_length=max_length)
        labels = tokenizer(target_text, truncation=True, padding='max_length', max_length=max_answer_length)['input_ids']
        labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
        model_inputs['labels'] = labels + [-100] * (max_length - len(labels))
        return model_inputs
    return dataset.map(preprocess, remove_columns=dataset.column_names)

# Functions for loading models
def fine_tune_generator(model, tokenizer, train_dataset, eval_dataset, output_dir="/content/fine_tuned_generator"):
    """Fine-tunes the generator model using the SQuAD dataset."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=0.001,
        weight_decay=0.01,
        logging_dir='./logs',
        save_strategy="epoch",
        report_to=None,
        remove_unused_columns=False,
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()  # Start training
    model.save_pretrained(output_dir)  # Save the fine-tuned model
    tokenizer.save_pretrained(output_dir)  # Save the tokenizer

def load_encoder():
    """Loads the encoder model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
    model = AutoModel.from_pretrained(encoder_model_name).to(device)
    return tokenizer, model

def load_reranker():
    """Loads the reranker model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name).to(device)
    return tokenizer, model

def load_quantized_generator(model_name=generator_model_name, for_training=False):
    """Loads the quantized generator model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        device_map="auto"
    )

    if for_training:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    return tokenizer, model

def encode_query(query, tokenizer, model):
    """Encodes the query into embeddings."""
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=max_query_length).to(device)
    with torch.no_grad():
        embeddings = model.base_model(**inputs).last_hidden_state.mean(dim=1)
    reset_memory()
    return embeddings

def rerank(query, candidates, tokenizer, model):
    """Ranks candidate documents based on relevance to the query."""
    inputs = [tokenizer(query, doc, return_tensors="pt", padding=True, truncation=True).to(device) for doc in candidates]
    scores = [model(**input).logits.softmax(dim=-1).max().item() for input in inputs]
    ranked_candidates = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]
    reset_memory()
    return ranked_candidates

def generate_answer(query, context, tokenizer, model):
    """Generates an answer based on the query and context."""
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    model.config.pad_token_id = model.config.eos_token_id
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        early_stopping=True
    )
    reset_memory()
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[1].strip()

def clean_and_overwrite_answer_file():
    """Cleans the generated answer file."""
    file_path = "/content/answer.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    qa_blocks = re.findall(r"Query: (.*?)\n+Answer: (.*?)(?=\n+Query:|\Z)", content, re.DOTALL)
    cleaned_output = ""
    for query, answer in qa_blocks:
        cleaned_output += f"Answer: {answer.strip()}\n\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_output.strip())

# Main function that coordinates everything
def main():
    """Main function to orchestrate the RAG pipeline."""
    # Step 1: Fine-tune generator
    squad = load_dataset("squad")  # Load SQuAD dataset
    gen_tok, gen_model = load_quantized_generator(for_training=True)  # Load the generator model
    train_dataset = preprocess_squad_data(squad["train"], gen_tok)  # Preprocess training data
    eval_dataset = preprocess_squad_data(squad["validation"], gen_tok)  # Preprocess validation data
    fine_tune_generator(gen_model, gen_tok, train_dataset, eval_dataset)  # Fine-tune the model

    reset_memory()

    # Step 2: Load models
    gen_tok, gen_model = load_quantized_generator()  # Load the fine-tuned generator
    enc_tok, enc_model = load_encoder()  # Load the encoder
    rr_tok, rr_model = load_reranker()  # Load the reranker

    reset_memory()

    # Step 3: Load or embed Wikipedia
    embs, docs = load()  # Load existing embeddings and documents
    if embs is None or docs is None:  # If no embeddings found, embed new documents
        print("Embedding Wikipedia text...")
        wiki = load_dataset("wikipedia", "20220301.en", split=f"train[:{docs_to_embed}]", trust_remote_code=True)
        docs = [clean_text(example["text"]) for example in wiki]
        all_embeddings = []
        for i in tqdm(range(0, len(docs), batch_size)):  # Embed documents in batches
            batch = docs[i:i+batch_size]
            inputs = enc_tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = enc_model(**inputs)
                batch_embs = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(batch_embs.cpu())
            reset_memory()
        embs = torch.cat(all_embeddings, dim=0)  # Concatenate all embeddings
        save(embs, docs)  # Save embeddings and documents

    reset_memory()

    # Step 4: Build or load FAISS index
    if os.path.exists(faiss_index_path):  # Check if FAISS index already exists
        index = load_index()  # Load existing FAISS index
    else:
        index = build_index(embs)  # Build FAISS index from embeddings
        save_index(index)  # Save the FAISS index

    # Step 5: RAG inference
    with open(answer_path, "w") as f:
        for q in queries:  # For each query
            q_emb = encode_query(q, enc_tok, enc_model)  # Get the embedding of the query
            _, idxs = index.search(q_emb.cpu().numpy(), top_k)  # Retrieve top-k documents from FAISS
            candidates = [docs[i] for i in idxs[0]]
            reranked = rerank(q, candidates, rr_tok, rr_model)  # Rerank documents
            context = " ".join(reranked)[:2048]  # Get the top context
            ans = generate_answer(q, context, gen_tok, gen_model)  # Generate an answer based on the query and context
            f.write(f"Query: {q}\n\nAnswer: {ans}\n\n")  # Write the answer to the output file

    # Step 6: Clean the output
    clean_and_overwrite_answer_file()  # Clean and overwrite the answer file
    print("Answer file cleaned and overwritten.")

if __name__ == "__main__":
    main()  # Run the main function
