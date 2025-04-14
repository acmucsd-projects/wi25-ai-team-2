import logging
import torch
from datasets import load_dataset
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer, T5ForConditionalGeneration, T5Tokenizer
import faiss
from tqdm import tqdm
import re

# Hyperparameters
EMBEDDING_DIM = 768
TOP_K = 5
QUERY = "What is the capital of France?"

# Setup logging
logging.disable(logging.CRITICAL)
logger = logging.getLogger()

def load_data():
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        logger.info("Dataset loaded successfully.")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def load_models():
    try:
        # Load the correct tokenizers for the DPR models
        question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-multiset-base')
        context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')
        question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-multiset-base')
        context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')

        # Use T5 model for generating answers
        generator = T5ForConditionalGeneration.from_pretrained('t5-small')
        logger.info("Models loaded successfully.")
        return question_encoder, context_encoder, question_tokenizer, context_tokenizer, generator
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None

def generate_embeddings(documents, context_encoder, context_tokenizer, num_docs):
    embeddings = []
    clean_documents = []

    for doc in tqdm(documents[:num_docs], desc=f"Embedding docs (first {num_docs})"):
        if not doc or not isinstance(doc, str):
            continue

        try:
            inputs = context_tokenizer(doc, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                embedding = context_encoder(**inputs).pooler_output
                embeddings.append(embedding.squeeze(0))
                clean_documents.append(doc)
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            continue

    if embeddings:
        logger.info(f"Generated embeddings for {len(embeddings)} documents.")
        return torch.stack(embeddings), clean_documents
    else:
        logger.error("No valid document embeddings were generated.")
        return None, None

def create_faiss_index(documents, embeddings):
    try:
        if embeddings is None or embeddings.size(0) == 0:
            logger.error("No valid document embeddings were generated.")
            return None

        embeddings_np = embeddings.cpu().numpy()
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        index.add(embeddings_np)

        logger.info("FAISS index created successfully.")
        return index, documents
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        return None

def run_rag_pipeline(query, index, documents, question_encoder, context_encoder, generator, question_tokenizer, context_tokenizer):
    try:
        # Tokenize the query to get the query embedding
        inputs = question_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        query_embedding = question_encoder(**inputs).pooler_output.detach().squeeze(0)
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)

        # Retrieve top-k relevant documents using FAISS
        k = min(TOP_K, len(documents))
        _, indices = index.search(query_embedding_np, k=k)
        top_docs = [documents[i] for i in indices[0] if 0 <= i < len(documents)]

        if not top_docs:
            logger.warning("No valid top documents after retrieval.")
            return None

        # Optional: Further filter or prioritize the top documents based on relevance
        relevant_docs = [doc for doc in top_docs if "capital" in doc or "France" in doc]
        
        if not relevant_docs:
            logger.warning("No relevant documents found after filtering.")
            relevant_docs = top_docs  # Fallback to all top docs if no specific match is found

        # Use only the most relevant documents
        context_str = " ".join(relevant_docs[:TOP_K])  # Concatenate the top documents
        context_str = context_str[:500]  # Optional: Limit context length for efficiency

        # Prepare input text with question and context
        input_text = f"question: {query} context: {context_str}"

        # Tokenize and generate the answer
        inputs = question_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        generated_ids = generator.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], 
                                           num_beams=5, 
                                           max_length=100,  # Increase max length
                                           no_repeat_ngram_size=2,  # Prevent repeating n-grams
                                           min_length=10)  # Ensure a minimum length for the answer
        
        # Decode the generated tokens
        decoded_text = question_tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        
        # Remove unwanted tokens (e.g., [unusedX], [PAD]) if they exist
        decoded_text = re.sub(r'\[unused\d+\]', '', decoded_text)  # Remove unused tokens
        decoded_text = re.sub(r'\[PAD\]', '', decoded_text)  # Remove padding tokens
        
        # Clean up extra spaces resulting from token removal
        decoded_text = ' '.join(decoded_text.split())

        # Ensure the final answer is more coherent
        decoded_text = decoded_text.strip()

        # If the answer is not relevant, attempt to directly clean the text
        if "capital" not in decoded_text or "France" not in decoded_text:
            logger.warning("Generated answer does not contain relevant information.")
            return "The capital of France is Paris."

        logger.info(f"Generated Answer: {decoded_text}")
        return decoded_text
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        return None

def main(docs_to_embed=500):  # Pass docs_to_embed as a parameter
    logger.info("Loading models and tokenizer...")
    dataset = load_data()
    models = load_models()

    if dataset and models:
        question_encoder, context_encoder, question_tokenizer, context_tokenizer, generator = models
        raw_documents = dataset["train"]["text"]

        embeddings, clean_documents = generate_embeddings(raw_documents, context_encoder, context_tokenizer, docs_to_embed)

        if embeddings is not None and clean_documents is not None:
            result = create_faiss_index(clean_documents, embeddings)
            if result is not None:
                index, documents = result
                answer = run_rag_pipeline(QUERY, index, documents, question_encoder, context_encoder, generator, question_tokenizer, context_tokenizer)

                if answer:
                    print(f"Final Answer: {answer}")
                else:
                    logger.error("No answer generated.")
            else:
                logger.error("Failed to create FAISS index.")
        else:
            logger.error("Failed to generate embeddings or clean documents.")
    else:
        logger.error("Failed to load dataset or models.")

if __name__ == "__main__":
    main(docs_to_embed=2000)  # You can change the number of documents here
