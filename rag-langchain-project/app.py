from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
import sys


# Optional: Clean error output
def err_remove(er):
    lin = "------------"
    er = str(er)
    if lin in er:
        start_index = er.find(lin) + len(lin)
        end_index = er.rfind(lin)
        return er[start_index:end_index].strip()
    else:
        return er  # fallback to whole err

# Step 1: Load dataset
dataset_name = "databricks/databricks-dolly-15k"
page_content_column = "context"
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
data = loader.load()

# Step 2: Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = splitter.split_documents(data)

# Step 3: Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store in FAISS vector store
db = FAISS.from_documents(docs, embeddings)

# Step 5: Load QA model from Hugging Face
model_name = "Intel/dynamic_tinybert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

qa_pipeline = pipeline(
    "question-answering", 
    model=model_name, 
    tokenizer=tokenizer,
    return_tensors="pt"
)

llm = HuggingFacePipeline(pipeline=qa_pipeline, model_kwargs={"temperature": 0.7, "max_length": 512})

# Step 6: Create retriever and RetrievalQA chain
retriever = db.as_retriever(search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)

# Step 7: Ask your question
question = "Who is Thomas Jefferson?"

try:
    result = qa.invoke({"query": question})
    print("\nAnswer:")
    print(result["result"])
except Exception as e:
    answer = err_remove(e)
    print("\nAn error occurred:")
    print(answer)
# result = qa.run({"query": question})

# print("\nAnswer:")
# print(result)